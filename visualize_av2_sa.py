#!/usr/bin/env python3
"""Select and render strong, complex, multimodal AV2 predictions."""

import argparse
import csv
import html
import json
import math
import shlex
from pathlib import Path

import numpy as np


METRIC_FIELDS = (
    "min_ade",
    "min_fde",
    "nearby_agents",
    "intersection_lanes",
    "maneuver_m",
    "endpoint_diversity_m",
    "effective_modes",
)
SCORE_FIELDS = ("complexity_score", "quality_score", "multimodality_score", "score")


def parse_ints(value):
    return [int(item) for item in value.split(",") if item.strip()]


def parse_ids(value, maximum):
    ids = set()
    for part in value.split(","):
        bounds = part.strip().split("-", 1)
        if not bounds[0]:
            continue
        start = int(bounds[0])
        end = int(bounds[-1])
        if start > end:
            raise ValueError(f"Invalid descending range: {part}")
        ids.update(range(start, end + 1))
    invalid = sorted(item for item in ids if item < 1 or item > maximum)
    if invalid:
        raise ValueError(f"Candidate IDs out of range 1..{maximum}: {invalid}")
    return sorted(ids)


def percentile_score(values, higher_is_better=True):
    """Tie-aware empirical percentile in (0, 1)."""
    values = np.asarray(values)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    lower = np.cumsum(counts) - counts
    score = (lower[inverse] + 0.5 * counts[inverse]) / len(values)
    return score if higher_is_better else 1.0 - score


def add_selection_scores(records, min_percentile, max_min_ade, max_min_fde, count):
    def values(name):
        return np.asarray([row[name] for row in records], dtype=np.float64)

    complexity = (
        0.45 * percentile_score(values("nearby_agents"))
        + 0.25 * percentile_score(values("intersection_lanes"))
        + 0.30 * percentile_score(values("maneuver_m"))
    )
    quality = (
        0.60 * percentile_score(values("min_ade"), False)
        + 0.40 * percentile_score(values("min_fde"), False)
    )
    multimodality = (
        0.70 * percentile_score(values("endpoint_diversity_m"))
        + 0.30 * percentile_score(values("effective_modes"))
    )
    score = np.cbrt(complexity * quality * multimodality)
    eligible = (
        (complexity >= min_percentile)
        & (quality >= min_percentile)
        & (multimodality >= min_percentile)
        & (values("min_ade") <= max_min_ade)
        & (values("min_fde") <= max_min_fde)
    )

    for index, row in enumerate(records):
        row.update(
            complexity_score=float(complexity[index]),
            quality_score=float(quality[index]),
            multimodality_score=float(multimodality[index]),
            score=float(score[index]),
            candidate_id="",
        )

    ranked = np.flatnonzero(eligible)
    ranked = ranked[np.argsort(-score[ranked], kind="stable")][:count]
    selected = []
    for candidate_id, index in enumerate(ranked, 1):
        records[index]["candidate_id"] = candidate_id
        selected.append(records[index])
    return selected


def batch_metrics(data, output, nearby_radius):
    import torch

    prediction = output["y_hat"][..., :2]
    target = data["target"][:, 0, :, :2]
    steps = min(prediction.shape[-2], target.shape[-2])
    prediction, target = prediction[:, :, :steps], target[:, :steps]
    valid = data["target_mask"][:, 0, :steps]
    if not valid.any(dim=-1).all():
        raise ValueError("A focal agent has no valid future target")

    error = torch.linalg.vector_norm(prediction - target[:, None], dim=-1)
    ade = (error * valid[:, None]).sum(-1) / valid.sum(-1)[:, None]
    timeline = torch.arange(steps, device=valid.device)
    last = torch.where(valid, timeline, -1).max(-1).values
    final_prediction = prediction.gather(
        2, last[:, None, None, None].expand(-1, prediction.shape[1], 1, 2)
    ).squeeze(2)
    final_target = target.gather(
        1, last[:, None, None].expand(-1, 1, 2)
    ).squeeze(1)
    fde = torch.linalg.vector_norm(final_prediction - final_target[:, None], dim=-1)
    best_mode = (ade + fde).argmin(-1)

    probability = torch.softmax(output["pi"].double(), dim=-1)
    endpoint_distance = torch.cdist(final_prediction.double(), final_prediction.double())
    pair_mask = torch.triu(
        torch.ones_like(endpoint_distance, dtype=torch.bool), diagonal=1
    )
    pair_mass = probability[:, :, None] * probability[:, None, :]
    endpoint_diversity = (endpoint_distance * pair_mass * pair_mask).sum((1, 2))
    endpoint_diversity /= (pair_mass * pair_mask).sum((1, 2)).clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1)

    focal_center = data["x_centers"][:, :1, :2]
    agent_distance = torch.linalg.vector_norm(
        data["x_centers"][..., :2] - focal_center, dim=-1
    )
    agent_mask = data["x_key_valid_mask"].clone()
    agent_mask[:, 0] = False
    nearby_agents = (agent_mask & (agent_distance <= nearby_radius)).sum(-1)

    lane_distance = torch.linalg.vector_norm(
        data["lane_centers"][..., :2] - focal_center, dim=-1
    )
    intersection_lanes = (
        data["lane_key_valid_mask"]
        & data["is_intersections"].bool()
        & (lane_distance <= nearby_radius)
    ).sum(-1)

    displacement = torch.linalg.vector_norm(final_target, dim=-1)
    denominator = displacement.clamp_min(1e-6)
    cross = torch.abs(
        target[..., 0] * final_target[:, None, 1]
        - target[..., 1] * final_target[:, None, 0]
    )
    maneuver = torch.where(valid, cross / denominator[:, None], 0).max(-1).values
    maneuver = torch.where(displacement >= 5, maneuver, 0)

    tensors = {
        "min_ade": ade.min(-1).values,
        "min_fde": fde.min(-1).values,
        "best_mode": best_mode,
        "nearby_agents": nearby_agents,
        "intersection_lanes": intersection_lanes,
        "maneuver_m": maneuver,
        "endpoint_diversity_m": endpoint_diversity,
        "effective_modes": entropy.exp(),
    }
    return {name: value.detach().cpu().numpy() for name, value in tensors.items()}


def move_to_device(sequence, device):
    import torch

    for data in sequence:
        for key, value in data.items():
            if torch.is_tensor(value):
                data[key] = value.to(device, non_blocking=True)


def predict_sequence(model, sequence):
    memory = None
    output = None
    for data in sequence:
        data["memory_dict"] = memory
        output = model(data)
        memory = output["memory_dict"]
    return output


def make_loader(dataset, indices, args):
    from torch.utils.data import DataLoader, Subset

    from src.datamodules.av2_dataset import collate_fn

    return DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collate_fn,
    )


def load_model_and_dataset(args):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from src.datamodules.av2_dataset import Av2Dataset

    checkpoint = Path(args.checkpoint)
    config = (
        Path(args.config)
        if args.config
        else checkpoint.parent.parent / ".hydra" / "config.yaml"
    )
    for path, label in ((checkpoint, "checkpoint"), (config, "config")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu only if the model supports it")

    cfg = OmegaConf.load(config)
    model = instantiate(cfg.model.pl_module)
    model.load_chkpt(checkpoint)
    model = model.eval().to(args.device)
    dataset = Av2Dataset(
        data_root=Path(args.data_root),
        split=args.split,
        num_historical_steps=args.num_historical_steps,
        split_points=parse_ints(args.split_points),
        radius=args.scene_radius,
        num_future_steps=args.future_steps,
    )
    if not len(dataset):
        raise FileNotFoundError(f"No processed .pt files under {args.data_root}/{args.split}")
    return model, dataset


def scan_dataset(model, dataset, args):
    import torch
    from tqdm import tqdm

    size = min(len(dataset), args.max_scenarios or len(dataset))
    indices = list(range(size))
    records, cursor = [], 0
    with torch.inference_mode():
        for sequence in tqdm(make_loader(dataset, indices, args), desc="Scoring"):
            move_to_device(sequence, args.device)
            output = predict_sequence(model, sequence)
            final_data = sequence[-1]
            metrics = batch_metrics(final_data, output, args.nearby_radius)
            for offset, scene_id in enumerate(final_data["scenario_id"]):
                row = {
                    "dataset_index": indices[cursor + offset],
                    "scenario_id": scene_id,
                    "timestep": int(round(float(final_data["timestamp"][offset]) * 10)),
                }
                row.update(
                    {
                        name: int(values[offset])
                        if name in ("best_mode", "nearby_agents", "intersection_lanes")
                        else float(values[offset])
                        for name, values in metrics.items()
                    }
                )
                records.append(row)
            cursor += len(final_data["scenario_id"])
    return records


def collect_selected_predictions(model, dataset, selected, args):
    import torch

    predictions, cursor = [], 0
    indices = [row["dataset_index"] for row in selected]
    with torch.inference_mode():
        for sequence in make_loader(dataset, indices, args):
            move_to_device(sequence, args.device)
            output = predict_sequence(model, sequence)
            data = sequence[-1]
            steps = data["target"].shape[-2]
            local = output["y_hat"][:, :, :steps, :2]
            theta = data["theta"]
            rotation = torch.stack(
                (torch.cos(theta), torch.sin(theta), -torch.sin(theta), torch.cos(theta)),
                dim=1,
            ).reshape(-1, 2, 2)
            global_prediction = torch.matmul(local.double(), rotation[:, None].double())
            global_prediction += data["origin"][:, None, None].double()

            for offset, scene_id in enumerate(data["scenario_id"]):
                expected = selected[cursor + offset]["scenario_id"]
                if scene_id != expected:
                    raise RuntimeError(f"Selection order changed: expected {expected}, got {scene_id}")
                predictions.append(global_prediction[offset].float().cpu().numpy())
            cursor += len(data["scenario_id"])
    return np.stack(predictions)


def write_csv(path, records):
    fields = (
        "candidate_id",
        "dataset_index",
        "scenario_id",
        "timestep",
        *METRIC_FIELDS,
        *SCORE_FIELDS,
        "best_mode",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def save_candidates(path, selected, predictions):
    arrays = {
        "candidate_id": np.asarray([row["candidate_id"] for row in selected]),
        "scenario_id": np.asarray([row["scenario_id"] for row in selected]),
        "timestep": np.asarray([row["timestep"] for row in selected]),
        "prediction": predictions,
    }
    for field in (*METRIC_FIELDS, *SCORE_FIELDS, "best_mode"):
        arrays[field] = np.asarray([row[field] for row in selected])
    np.savez_compressed(path, **arrays)


def load_candidates(output_dir):
    path = Path(output_dir) / "candidates.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Candidate cache not found: {path}; run scan first")
    with np.load(path, allow_pickle=False) as pack:
        return {name: pack[name] for name in pack.files}


def raw_split_dir(raw_data_root, split):
    root = Path(raw_data_root)
    path = root / split
    if path.is_dir():
        return path
    if root.name == split and root.is_dir():
        return root
    raise FileNotFoundError(f"Raw AV2 split not found: {path}")


def render_scene(pack, index, raw_dir, path, dpi):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from av2.datasets.motion_forecasting import scenario_serialization
    from av2.map.map_api import ArgoverseStaticMap

    from src.utils.vis import visualize_scenario

    scene_id = str(pack["scenario_id"][index])
    scene_dir = raw_dir / scene_id
    scenario = scenario_serialization.load_argoverse_scenario_parquet(
        scene_dir / f"scenario_{scene_id}.parquet"
    )
    static_map = ArgoverseStaticMap.from_json(
        scene_dir / f"log_map_archive_{scene_id}.json"
    )
    candidate_id = int(pack["candidate_id"][index])
    title = (
        f"#{candidate_id:03d}  score={pack['score'][index]:.2f}  "
        f"ADE/FDE={pack['min_ade'][index]:.2f}/{pack['min_fde'][index]:.2f}m"
    )
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    plt.sca(ax)
    visualize_scenario(
        scenario,
        static_map,
        prediction=pack["prediction"][index],
        timestep=int(pack["timestep"][index]),
        title=title,
        create_fig=False,
    )
    ax.text(
        0.02,
        0.98,
        f"#{candidate_id:03d}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=13,
        fontweight="bold",
        bbox={"facecolor": "black", "alpha": 0.75, "pad": 4, "edgecolor": "none"},
        zorder=3000,
    )
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def make_contact_sheets(thumbnails, output_dir, per_sheet, columns, dpi):
    import matplotlib.pyplot as plt

    sheet_paths = []
    for page, start in enumerate(range(0, len(thumbnails), per_sheet), 1):
        chunk = thumbnails[start : start + per_sheet]
        rows = math.ceil(len(chunk) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
        axes = np.asarray(axes, dtype=object).reshape(-1)
        for ax, image_path in zip(axes, chunk):
            ax.imshow(plt.imread(image_path))
            ax.axis("off")
        for ax in axes[len(chunk) :]:
            ax.axis("off")
        fig.tight_layout(pad=0.15)
        sheet_path = output_dir / f"contact_sheet_{page:02d}.jpg"
        fig.savefig(sheet_path, dpi=dpi, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        sheet_paths.append(sheet_path)
    return sheet_paths


def write_gallery(path, pack, thumbnails, sheets, render_command):
    cards = []
    for index, thumbnail in enumerate(thumbnails):
        candidate_id = int(pack["candidate_id"][index])
        cards.append(
            f"""
            <label class="card">
              <input type="checkbox" value="{candidate_id}" onchange="update()">
              <img src="{html.escape(thumbnail.relative_to(path.parent).as_posix())}" loading="lazy">
              <span>#{candidate_id:03d} · C {pack['complexity_score'][index]:.2f}
                · Q {pack['quality_score'][index]:.2f}
                · M {pack['multimodality_score'][index]:.2f}</span>
            </label>"""
        )
    links = " · ".join(
        f'<a href="{html.escape(sheet.name)}">候选联系表 {index}</a>'
        for index, sheet in enumerate(sheets, 1)
    )
    command_json = json.dumps(render_command)
    page = f"""<!doctype html>
<meta charset="utf-8">
<title>AV2 candidate gallery</title>
<style>
body{{margin:0;background:#111;color:#eee;font:14px system-ui}}
header{{position:sticky;top:0;z-index:2;padding:12px 18px;background:#1d1d1df2}}
a{{color:#8cc8ff}} button{{margin-left:8px}} code{{user-select:all}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px;padding:12px}}
.card{{background:#222;padding:8px;border-radius:8px;cursor:pointer}}
.card:has(input:checked){{outline:3px solid #48a9ff}} img{{width:100%;display:block}}
.card span{{display:block;padding:7px 3px 2px}} input{{position:absolute;transform:scale(1.5);margin:14px}}
</style>
<header>
  <div>{links}</div>
  <div>已选编号：<b id="ids">无</b><button onclick="copyCommand()">复制精绘命令</button></div>
  <code id="command"></code>
</header>
<main class="grid">{''.join(cards)}</main>
<script>
const prefix={command_json};
function update(){{
  const ids=[...document.querySelectorAll('input:checked')].map(x=>x.value).join(',');
  document.getElementById('ids').textContent=ids||'无';
  document.getElementById('command').textContent=ids ? prefix+ids : '';
}}
async function copyCommand(){{
  const command=document.getElementById('command').textContent;
  if(command) await navigator.clipboard.writeText(command);
}}
update();
</script>"""
    path.write_text(page, encoding="utf-8")


def render_candidate_gallery(pack, args):
    output_dir = Path(args.output_dir)
    raw_dir = raw_split_dir(args.raw_data_root, args.split)
    thumbnail_dir = output_dir / "thumbnails"
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    thumbnails = []
    for index, candidate_id in enumerate(pack["candidate_id"]):
        path = thumbnail_dir / f"{int(candidate_id):03d}.jpg"
        render_scene(pack, index, raw_dir, path, args.thumbnail_dpi)
        thumbnails.append(path)
    sheets = make_contact_sheets(
        thumbnails, output_dir, args.per_sheet, args.columns, args.thumbnail_dpi
    )
    render_command = (
        "python visualize_av2_sa.py render "
        f"--output-dir {shlex.quote(str(output_dir.resolve()))} "
        f"--raw-data-root {shlex.quote(str(Path(args.raw_data_root).resolve()))} "
        f"--split {args.split} --ids "
    )
    write_gallery(output_dir / "gallery.html", pack, thumbnails, sheets, render_command)
    return sheets


def run_scan(args):
    import torch

    if not 0 <= args.min_component_percentile < 1:
        raise ValueError("--min-component-percentile must be in [0, 1)")
    if min(
        args.num_candidates,
        args.per_sheet,
        args.columns,
        args.thumbnail_dpi,
        args.batch_size,
        args.future_steps,
        args.scene_radius,
        args.nearby_radius,
        args.max_min_ade,
        args.max_min_fde,
    ) <= 0:
        raise ValueError("counts, radii, quality thresholds and DPI must be positive")
    if args.num_workers < 0 or args.max_scenarios < 0:
        raise ValueError("--num-workers and --max-scenarios cannot be negative")
    raw_split_dir(args.raw_data_root, args.split)
    model, dataset = load_model_and_dataset(args)
    records = scan_dataset(model, dataset, args)
    selected = add_selection_scores(
        records,
        args.min_component_percentile,
        args.max_min_ade,
        args.max_min_fde,
        args.num_candidates,
    )
    if not selected:
        raise RuntimeError(
            "No candidate passed the filters; lower --min-component-percentile "
            "or relax --max-min-ade/--max-min-fde"
        )
    if len(selected) < args.num_candidates:
        print(f"WARNING: only {len(selected)} scenes passed all filters")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "all_scores.csv", records)
    write_csv(output_dir / "manifest.csv", selected)
    predictions = collect_selected_predictions(model, dataset, selected, args)
    save_candidates(output_dir / "candidates.npz", selected, predictions)
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    pack = load_candidates(output_dir)
    sheets = render_candidate_gallery(pack, args)
    print(f"Selected {len(selected)} / {len(records)} scenes")
    print(output_dir / "gallery.html")
    for sheet in sheets:
        print(sheet)


def run_render(args):
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    pack = load_candidates(args.output_dir)
    try:
        ids = parse_ids(args.ids, len(pack["candidate_id"]))
    except ValueError as error:
        raise ValueError(f"Invalid --ids: {error}") from error
    if not ids:
        raise ValueError("--ids must contain at least one candidate number")
    raw_dir = raw_split_dir(args.raw_data_root, args.split)
    selected_dir = Path(args.output_dir) / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    for candidate_id in ids:
        index = int(np.flatnonzero(pack["candidate_id"] == candidate_id)[0])
        scene_id = str(pack["scenario_id"][index])
        path = selected_dir / f"{candidate_id:03d}_{scene_id}.{args.format}"
        render_scene(pack, index, raw_dir, path, args.dpi)
        print(path)


def self_check():
    import torch

    assert parse_ids("1,3-5,5", 5) == [1, 3, 4, 5]
    score = percentile_score([0, 0, 2])
    np.testing.assert_allclose(score, [1 / 3, 1 / 3, 5 / 6])
    target = torch.tensor([[[[1.0, 0], [2, 0], [3, 0]]]])
    metrics = batch_metrics(
        {
            "target": target,
            "target_mask": torch.ones(1, 1, 3, dtype=torch.bool),
            "x_centers": torch.zeros(1, 2, 2),
            "x_key_valid_mask": torch.ones(1, 2, dtype=torch.bool),
            "lane_centers": torch.zeros(1, 2, 2),
            "lane_key_valid_mask": torch.ones(1, 2, dtype=torch.bool),
            "is_intersections": torch.tensor([[1, 0]]),
        },
        {
            "y_hat": target[:, 0, None].repeat(1, 2, 1, 1),
            "pi": torch.zeros(1, 2),
        },
        30,
    )
    assert metrics["min_fde"].shape == (1,) and metrics["min_fde"][0] == 0
    records = [
        {
            "min_ade": 2 - value,
            "min_fde": 4 - value,
            "nearby_agents": value,
            "intersection_lanes": value,
            "maneuver_m": value,
            "endpoint_diversity_m": value,
            "effective_modes": value + 1,
        }
        for value in range(3)
    ]
    selected = add_selection_scores(records, 0, 10, 10, 1)
    assert selected[0] is records[-1] and selected[0]["candidate_id"] == 1
    print("self-check passed")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="score the split and create candidate sheets")
    scan.add_argument("--data-root", required=True, help="processed SHARP data root")
    scan.add_argument("--raw-data-root", required=True, help="AV2 motion-forecasting root")
    scan.add_argument(
        "--checkpoint",
        default="exps/av2_single_agent/checkpoints/av2_sa.ckpt",
    )
    scan.add_argument("--config", help="defaults to <experiment>/.hydra/config.yaml")
    scan.add_argument("--output-dir", default="output/av2_candidates")
    scan.add_argument("--split", choices=("train", "val"), default="val")
    scan.add_argument("--device", default="cuda")
    scan.add_argument("--batch-size", type=int, default=32)
    scan.add_argument("--num-workers", type=int, default=4)
    scan.add_argument("--max-scenarios", type=int, default=0, help="0 scans all")
    scan.add_argument("--num-historical-steps", type=int, default=10)
    scan.add_argument("--future-steps", type=int, default=60)
    scan.add_argument("--split-points", default="10,20,30,40,50")
    scan.add_argument("--scene-radius", type=float, default=150)
    scan.add_argument("--nearby-radius", type=float, default=30)
    scan.add_argument("--num-candidates", type=int, default=48)
    scan.add_argument("--min-component-percentile", type=float, default=0.55)
    scan.add_argument("--max-min-ade", type=float, default=2.0)
    scan.add_argument("--max-min-fde", type=float, default=4.0)
    scan.add_argument("--per-sheet", type=int, default=12)
    scan.add_argument("--columns", type=int, default=4)
    scan.add_argument("--thumbnail-dpi", type=int, default=110)
    scan.set_defaults(func=run_scan)

    render = subparsers.add_parser("render", help="render cached candidates by number")
    render.add_argument("--output-dir", default="output/av2_candidates")
    render.add_argument("--raw-data-root", required=True)
    render.add_argument("--split", choices=("train", "val"), default="val")
    render.add_argument("--ids", required=True, help="for example: 1,4,7-9")
    render.add_argument("--format", choices=("png", "pdf"), default="png")
    render.add_argument("--dpi", type=int, default=300)
    render.set_defaults(func=run_render)

    check = subparsers.add_parser("self-check", help="run the lightweight scoring check")
    check.set_defaults(func=lambda _args: self_check())
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
