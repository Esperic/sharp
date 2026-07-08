#!/usr/bin/env python
import argparse
import json
import pickle
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_torch(path):
    if Path(path).suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            return pickle.load(f)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _progress(iterable, total=None, desc="", unit="it", disable=False):
    if disable:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc, unit=unit)
    except ImportError:
        def _fallback():
            for idx, item in enumerate(iterable, start=1):
                if idx == 1 or idx % 1000 == 0 or (total is not None and idx == total):
                    total_str = f"/{total}" if total is not None else ""
                    print(f"{desc}: {idx}{total_str} {unit}")
                yield item

        return _fallback()


def _target_from_processed_dict(sample, future_steps):
    if not isinstance(sample, dict) or "target" not in sample or sample["target"] is None:
        return None
    target = sample["target"]
    if not torch.is_tensor(target):
        target = torch.as_tensor(target)
    if target.ndim == 3:
        traj = target[0]
    elif target.ndim == 2:
        traj = target
    else:
        return None
    if traj.shape[-2:] != (future_steps, 2):
        return None
    target_mask = sample.get("target_mask")
    if target_mask is not None:
        if not torch.is_tensor(target_mask):
            target_mask = torch.as_tensor(target_mask)
        mask = target_mask[0] if target_mask.ndim == 2 else target_mask
        if mask.shape[0] >= future_steps and not bool(mask[:future_steps].all()):
            return None
    return traj.float()


def _rotation_matrix(theta):
    return torch.stack(
        [
            torch.stack([torch.cos(theta), -torch.sin(theta)]),
            torch.stack([torch.sin(theta), torch.cos(theta)]),
        ]
    )


def _targets_from_av2_scene_dict(sample, future_steps, split_points=(10, 20, 30, 40, 50), num_historical_steps=10):
    if not isinstance(sample, dict):
        return []
    required = {"focal_idx", "x_positions", "x_angles", "x_valid_mask", "x_attr"}
    if not required.issubset(sample.keys()):
        return []

    idx = int(sample["focal_idx"])
    x_positions = sample["x_positions"]
    x_angles = sample["x_angles"]
    x_valid_mask = sample["x_valid_mask"]
    x_attr = sample["x_attr"]
    if not torch.is_tensor(x_positions):
        x_positions = torch.as_tensor(x_positions)
    if not torch.is_tensor(x_angles):
        x_angles = torch.as_tensor(x_angles)
    if not torch.is_tensor(x_valid_mask):
        x_valid_mask = torch.as_tensor(x_valid_mask)
    if not torch.is_tensor(x_attr):
        x_attr = torch.as_tensor(x_attr)

    if idx >= x_positions.shape[0] or x_positions.shape[-1] != 2:
        return []
    if x_attr[idx, -1].item() == 3:
        return []

    trajs = []
    total_steps = x_positions.shape[1]
    for step in split_points:
        st = step - num_historical_steps
        ed = step + future_steps
        if st < 0 or ed > total_steps:
            continue
        valid = x_valid_mask[idx, st:ed].bool()
        if valid.shape[0] != num_historical_steps + future_steps:
            continue
        target_mask = valid[num_historical_steps - 1] & valid[num_historical_steps:]
        if not bool(target_mask.all()):
            continue

        origin = x_positions[idx, step - 1]
        theta = x_angles[idx, step - 1]
        rot_mat = _rotation_matrix(theta).to(dtype=x_positions.dtype, device=x_positions.device)
        local = torch.matmul(x_positions[idx, st:ed] - origin, rot_mat)
        pos_ctr = local[num_historical_steps - 1].clone()
        target = local[num_historical_steps:] - pos_ctr.unsqueeze(0)
        if target.shape == (future_steps, 2) and torch.isfinite(target).all():
            trajs.append(target.float())
    return trajs


def _dataset_for(dataset, root, future_steps):
    split = root.name
    data_root = root.parent if split in {"train", "val", "test"} else root
    split = split if split in {"train", "val", "test"} else "train"
    if dataset == "av2":
        from src.datamodules.av2_dataset import Av2Dataset

        return Av2Dataset(
            data_root=data_root,
            split=split,
            num_historical_steps=10,
            split_points=[10, 20, 30, 40, 50],
            num_future_steps=future_steps,
            randomize=False,
        )
    if dataset == "av1":
        from src.datamodules.av1_dataset import Av1Dataset

        return Av1Dataset(
            data_root=data_root,
            split=split,
            num_historical_steps=20,
            split_points=[5, 10, 15, 20],
            num_future_steps=future_steps,
        )
    if dataset in {"nus", "nuscenes"}:
        from src.datamodules.nus_dataset import NusDataset

        return NusDataset(
            data_root=data_root,
            split=split,
            num_historical_steps=5,
            split_points=[4, 5],
            num_future_steps=future_steps,
        )
    raise ValueError(f"Unsupported dataset={dataset}")


def iter_future_trajs(
    processed_root,
    dataset,
    future_steps,
    max_samples=None,
    dry_run=False,
    dry_run_samples=16,
    progress=True,
):
    root = Path(processed_root)
    files = sorted(root.glob("*.pt")) + sorted(root.glob("*.pkl")) + sorted(root.glob("*.pickle"))
    if dry_run:
        print(f"processed_root={root}")
        print(f"found_files={len(files)}")
        print(f"dry_run_samples={dry_run_samples}")
        if files:
            sample = _load_torch(files[0])
            if isinstance(sample, dict):
                print(f"first_file={files[0].name}")
                print(f"sample_keys={sorted(sample.keys())}")

    count = 0
    file_limit = dry_run_samples if dry_run else None
    if dataset == "av2":
        if dry_run:
            print("extractor=av2_fast_focal_scene_dict")
        selected_files = files[:file_limit]
        if progress:
            try:
                from tqdm import tqdm

                progress_bar = tqdm(total=len(selected_files), desc="extract_av2_futures", unit="file")
            except ImportError:
                progress_bar = None
        else:
            progress_bar = None
        try:
            for file_idx, path in enumerate(selected_files, start=1):
                obj = _load_torch(path)
                if progress_bar is not None:
                    progress_bar.update(1)
                elif progress and (file_idx == 1 or file_idx % 1000 == 0 or file_idx == len(selected_files)):
                    print(f"extract_av2_futures: {file_idx}/{len(selected_files)} file")
                for traj in _targets_from_av2_scene_dict(obj, future_steps):
                    yield traj.numpy()
                    count += 1
                    if max_samples is not None and count >= max_samples:
                        return
        finally:
            if progress_bar is not None:
                progress_bar.close()
        return

    use_dataset = True
    try:
        ds = _dataset_for(dataset, root, future_steps)
    except Exception as exc:
        print(f"Dataset adapter unavailable ({exc}); falling back to direct target extraction.")
        ds = None
        use_dataset = False

    if use_dataset:
        indices = range(len(ds))
        for idx in _progress(indices, total=len(ds), desc=f"extract_{dataset}_futures", unit="sample", disable=not progress):
            item = ds[idx]
            seq = item if isinstance(item, list) else [item]
            for sample in seq:
                traj = _target_from_processed_dict(sample, future_steps)
                if traj is None or not torch.isfinite(traj).all():
                    continue
                yield traj.numpy()
                count += 1
                if max_samples is not None and count >= max_samples:
                    return
    else:
        for path in _progress(files, total=len(files), desc="extract_targets", unit="file", disable=not progress):
            obj = _load_torch(path)
            candidates = obj if isinstance(obj, list) else [obj]
            for sample in candidates:
                traj = _target_from_processed_dict(sample, future_steps)
                if traj is None or not torch.isfinite(traj).all():
                    continue
                yield traj.numpy()
                count += 1
                if max_samples is not None and count >= max_samples:
                    return


def _fit_kmeans(flat, k, seed):
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as exc:
        raise ImportError("scikit-learn is required to build GMP params") from exc
    try:
        print(f"kmeans_start samples={flat.shape[0]} dims={flat.shape[1]} k={k}")
        start = perf_counter()
        model = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init="auto")
        labels = model.fit_predict(flat)
    except TypeError:
        print(f"kmeans_start samples={flat.shape[0]} dims={flat.shape[1]} k={k}")
        start = perf_counter()
        model = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=10)
        labels = model.fit_predict(flat)
    print(f"kmeans_done seconds={perf_counter() - start:.2f}")
    return labels


def build_gmp(trajs, k, std_floor, seed, normalize):
    if trajs.ndim != 3 or trajs.shape[-1] != 2:
        raise ValueError(f"Expected trajectories [N,T,2], got {trajs.shape}")
    norm_meta = {"type": normalize}
    if normalize == "mean_range":
        points = trajs.reshape(-1, 2)
        mean = points.mean(axis=0).astype(np.float32)
        scale = np.maximum(np.ptp(points, axis=0), 1e-6).astype(np.float32)
        trajs_for_cluster = ((trajs - mean.reshape(1, 1, 2)) / scale.reshape(1, 1, 2)).astype(np.float32)
        flat_for_cluster = trajs_for_cluster.reshape(trajs.shape[0], -1)
        norm_meta["mean"] = mean.tolist()
        norm_meta["scale"] = scale.tolist()
    elif normalize == "none":
        trajs_for_cluster = trajs
        flat_for_cluster = trajs.reshape(trajs.shape[0], -1)
    else:
        raise ValueError("normalize must be one of: none, mean_range")

    labels = _fit_kmeans(flat_for_cluster, k, seed)
    cluster_trajs = np.zeros((k, trajs.shape[1], 2), dtype=np.float32)
    cluster_trajs_raw = np.zeros((k, trajs.shape[1], 2), dtype=np.float32)
    center_points = np.zeros((k, 2), dtype=np.float32)
    center_points_raw = np.zeros((k, 2), dtype=np.float32)
    center_std = np.zeros((k, 2), dtype=np.float32)
    center_std_raw = np.zeros((k, 2), dtype=np.float32)
    mixture_weights = np.zeros(k, dtype=np.float32)
    counts = np.zeros(k, dtype=np.int64)

    for comp in range(k):
        member = trajs_for_cluster[labels == comp]
        member_raw = trajs[labels == comp]
        if member.shape[0] == 0:
            fallback = np.random.default_rng(seed + comp).integers(0, trajs.shape[0])
            member = trajs_for_cluster[[fallback]]
            member_raw = trajs[[fallback]]
        counts[comp] = member.shape[0]
        cluster_trajs[comp] = member.mean(axis=0)
        cluster_trajs_raw[comp] = member_raw.mean(axis=0)
        points = member.reshape(-1, 2)
        points_raw = member_raw.reshape(-1, 2)
        center_points[comp] = points.mean(axis=0)
        center_std[comp] = np.maximum(points.std(axis=0), std_floor)
        center_points_raw[comp] = points_raw.mean(axis=0)
        center_std_raw[comp] = np.maximum(points_raw.std(axis=0), std_floor)
        mixture_weights[comp] = float(member.shape[0]) / float(trajs.shape[0])

    endpoints = cluster_trajs_raw[:, -1] if normalize == "mean_range" else cluster_trajs[:, -1]
    angles = np.arctan2(endpoints[:, 1], endpoints[:, 0])
    distances = np.linalg.norm(endpoints, axis=-1)
    order = np.lexsort((-counts, distances, angles))
    return {
        "center_points": center_points[order].astype(np.float32),
        "center_std": center_std[order].astype(np.float32),
        "center_points_raw": center_points_raw[order].astype(np.float32),
        "center_std_raw": center_std_raw[order].astype(np.float32),
        "mixture_weights": mixture_weights[order].astype(np.float32),
        "cluster_trajs": cluster_trajs[order].astype(np.float32),
        "cluster_trajs_raw": cluster_trajs_raw[order].astype(np.float32),
        "counts": counts[order],
        "endpoint_order": endpoints[order],
        "normalization": norm_meta,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", required=True)
    parser.add_argument("--dataset", choices=["av2", "av1", "nus", "nuscenes"], required=True)
    parser.add_argument("--future_steps", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--std_floor", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dry_run_samples", type=int, default=16)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--normalize", choices=["none", "mean_range"], default="none")
    args = parser.parse_args()

    trajs = list(iter_future_trajs(
        args.processed_root,
        args.dataset,
        args.future_steps,
        max_samples=args.max_samples,
        dry_run=args.dry_run,
        dry_run_samples=args.dry_run_samples,
        progress=not args.no_progress,
    ))
    if not trajs:
        raise RuntimeError("No valid full-length training future trajectories were extracted.")
    trajs = np.stack(trajs).astype(np.float32)
    print(f"extracted_trajs={trajs.shape}")
    print(f"traj_range=[{trajs.min():.4f}, {trajs.max():.4f}]")
    if args.dry_run:
        print("dry_run=true; not saving GMP params")
        return

    params = build_gmp(trajs, args.k, args.std_floor, args.seed, args.normalize)
    metadata = {
        "dataset": args.dataset,
        "future_steps": args.future_steps,
        "k": args.k,
        "num_samples": int(trajs.shape[0]),
        "std_floor": args.std_floor,
        "seed": args.seed,
        "processed_root": str(Path(args.processed_root)),
        "normalization": params["normalization"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        center_points=params["center_points"],
        center_std=params["center_std"],
        center_points_raw=params["center_points_raw"],
        center_std_raw=params["center_std_raw"],
        mixture_weights=params["mixture_weights"],
        cluster_trajs=params["cluster_trajs"],
        cluster_trajs_raw=params["cluster_trajs_raw"],
        metadata=json.dumps(metadata),
    )
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps({**metadata, "counts": params["counts"].tolist()}, indent=2))
    print(f"K={args.k}, future_steps={args.future_steps}")
    print(f"center_points range=[{params['center_points'].min():.4f}, {params['center_points'].max():.4f}]")
    print(f"center_std range=[{params['center_std'].min():.4f}, {params['center_std'].max():.4f}]")
    print(f"mixture_weights={params['mixture_weights'].tolist()}")
    print(f"endpoint_order={params['endpoint_order'].tolist()}")
    print(f"saved={output}")
    print(f"metadata={sidecar}")


if __name__ == "__main__":
    main()
