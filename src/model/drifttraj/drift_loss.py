from typing import Dict, Iterable, Tuple

import torch
import torch.distributed as torch_dist
import torch.nn.functional as F


def flatten_trajectories(y: torch.Tensor, space: str = "traj", normalize_space: bool = True) -> torch.Tensor:
    if y.shape[-1] != 2:
        raise ValueError(f"trajectory tensor must end in 2, got shape {tuple(y.shape)}")
    if space == "endpoint":
        flat = y[..., -1, :]
    elif space == "traj":
        flat = y.reshape(*y.shape[:-2], -1)
    elif space == "mixed":
        flat = torch.cat([y.reshape(*y.shape[:-2], -1), y[..., -1, :]], dim=-1)
    else:
        raise ValueError(f"Unsupported drift_space={space}")
    if normalize_space:
        flat = flat / (flat.shape[-1] ** 0.5)
    return flat


def select_winner(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    metric: str = "ade_fde",
    fde_weight: float = 1.0,
) -> torch.Tensor:
    if y_hat.ndim != 4 or y.ndim != 3:
        raise ValueError(f"Expected y_hat [B,Q,T,2], y [B,T,2], got {tuple(y_hat.shape)}, {tuple(y.shape)}")
    dist = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1)
    ade = dist.mean(dim=-1)
    fde = torch.norm(y_hat[:, :, -1, :2] - y[:, -1].unsqueeze(1), dim=-1)
    if metric == "l2_sum":
        score = dist.sum(dim=-1)
    elif metric == "ade":
        score = ade
    elif metric == "fde":
        score = fde
    elif metric == "ade_fde":
        score = ade + float(fde_weight) * fde
    else:
        raise ValueError(f"Unsupported winner_metric={metric}")
    return torch.argmin(score, dim=-1)


def split_winner_and_others(y_hat: torch.Tensor, best_mode: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if y_hat.ndim != 4:
        raise ValueError(f"Expected y_hat [B,Q,T,2], got {tuple(y_hat.shape)}")
    bsz, num_modes = y_hat.shape[:2]
    batch_idx = torch.arange(bsz, device=y_hat.device)
    winner = y_hat[batch_idx, best_mode]
    others_mask = torch.ones(bsz, num_modes, dtype=torch.bool, device=y_hat.device)
    others_mask[batch_idx, best_mode] = False
    others = y_hat[others_mask].view(bsz, num_modes - 1, *y_hat.shape[2:])
    return winner, others


def endpoint_diversity_loss(
    y_hat: torch.Tensor,
    sigma: float = 2.0,
    valid_mask: torch.Tensor = None,
) -> torch.Tensor:
    endpoints = y_hat[..., -1, :2]
    dist2 = torch.cdist(endpoints, endpoints).square()
    num_modes = endpoints.shape[1]
    offdiag = ~torch.eye(num_modes, dtype=torch.bool, device=endpoints.device).unsqueeze(0)
    penalty = torch.exp(-dist2 / max(float(sigma) ** 2, 1e-8))
    if valid_mask is not None:
        offdiag = offdiag & valid_mask[:, None, None]
    values = penalty[offdiag.expand_as(penalty)]
    if values.numel() == 0:
        return y_hat.new_zeros(())
    return values.mean()


def _safe_force(vec: torch.Tensor, aff: torch.Tensor, normalize_force: bool) -> torch.Tensor:
    force = aff.unsqueeze(-1) * vec
    if normalize_force:
        denom = vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        force = force / denom
    return force


def _remove_away_component(force: torch.Tensor, target_direction: torch.Tensor, eps: float) -> torch.Tensor:
    alignment = (force * target_direction).sum(dim=-1, keepdim=True)
    away_component = alignment.clamp_max(0.0) * target_direction
    away_component = away_component / target_direction.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    return force - away_component


def _batched_cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    xy_dot = torch.einsum("bns,bms->bnm", x, y)
    x_norm = x.square().sum(dim=-1, keepdim=True)
    y_norm = y.square().sum(dim=-1, keepdim=True).transpose(1, 2)
    sq_dist = (x_norm + y_norm - 2.0 * xy_dot).clamp_min(0.0)
    return (sq_dist + eps).sqrt()


def sparse_trajectory_features(
    y: torch.Tensor,
    num_waypoints: int = 10,
    endpoint_weight: float = 2.0,
) -> torch.Tensor:
    """Compact AV2-friendly path representation without per-sample normalization."""
    if y.shape[-1] != 2 or y.shape[-2] < 1:
        raise ValueError(f"Expected trajectories [...,T,2], got {tuple(y.shape)}")
    count = min(max(int(num_waypoints), 1), y.shape[-2])
    indices = (torch.arange(1, count + 1, device=y.device) * y.shape[-2] // count - 1).long()
    waypoints = y.index_select(-2, indices)
    return torch.cat(
        [waypoints.flatten(start_dim=-2), y[..., -1, :] * float(endpoint_weight)],
        dim=-1,
    )


def _all_gather_detached(value: torch.Tensor) -> Tuple[torch.Tensor, int]:
    value = value.detach().contiguous()
    if not torch_dist.is_available() or not torch_dist.is_initialized():
        return value, 0
    gathered = [torch.empty_like(value) for _ in range(torch_dist.get_world_size())]
    torch_dist.all_gather(gathered, value)
    return torch.cat(gathered, dim=0), torch_dist.get_rank() * value.shape[0]


def joint_drifting_loss(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    scene_feature: torch.Tensor,
    r_list: Iterable[float] = (0.05, 0.2, 0.5),
    num_waypoints: int = 10,
    endpoint_weight: float = 2.0,
    context_alpha: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Scene-conditioned Drift over every mode and the full DDP batch pool."""
    if predictions.ndim != 4 or ground_truth.ndim != 3 or scene_feature.ndim != 2:
        raise ValueError(
            "Expected predictions [B,K,T,2], ground_truth [B,T,2], and scene_feature [B,D]"
        )
    if predictions.shape[0] != ground_truth.shape[0] or predictions.shape[0] != scene_feature.shape[0]:
        raise ValueError("predictions, ground_truth, and scene_feature must share the batch dimension")
    if predictions.shape[2:] != ground_truth.shape[1:]:
        raise ValueError("predictions and ground_truth must share trajectory length and coordinates")
    if predictions.shape[1] < 2:
        raise ValueError("joint_drifting_loss requires at least two generated modes")
    if not all(torch.isfinite(value).all() for value in (predictions, ground_truth, scene_feature)):
        raise FloatingPointError("joint_drifting_loss received non-finite inputs")

    radii = tuple(float(radius) for radius in r_list)
    if not radii or min(radii) <= 0:
        raise ValueError(f"r_list must contain positive radii, got {radii}")

    local_traj = sparse_trajectory_features(
        predictions, num_waypoints=num_waypoints, endpoint_weight=endpoint_weight
    ).float()
    gt_traj = sparse_trajectory_features(
        ground_truth, num_waypoints=num_waypoints, endpoint_weight=endpoint_weight
    ).float()
    local_scene = F.normalize(scene_feature.detach().float(), dim=-1)

    with torch.no_grad():
        global_scene, scene_offset = _all_gather_detached(local_scene)
        global_gt, _ = _all_gather_detached(gt_traj)
        global_pred, _ = _all_gather_detached(local_traj)

        global_batch, num_modes, traj_dim = global_pred.shape
        num_queries = global_batch * num_modes
        global_pred = global_pred.reshape(num_queries, traj_dim)

        # Pool normalization preserves relative path length while balancing the
        # trajectory block against the unit-norm context block.
        traj_rms = torch.cat([global_pred, global_gt], dim=0).square().mean().clamp_min(eps).sqrt()
        traj_block_scale = traj_rms * (traj_dim ** 0.5)
        global_pred = global_pred / traj_block_scale
        global_gt = global_gt / traj_block_scale

        query_scene = global_scene[:, None, :].expand(-1, num_modes, -1).reshape(num_queries, -1)
        query_joint = torch.cat([float(context_alpha) * query_scene, global_pred], dim=-1)
        positive_joint = torch.cat([float(context_alpha) * global_scene, global_gt], dim=-1)
        targets_joint = torch.cat([query_joint, positive_joint], dim=0)

        query_joint = query_joint.unsqueeze(0)
        targets_joint = targets_joint.unsqueeze(0)
        pair_dist = _batched_cdist(query_joint, targets_joint, eps=eps)
        scale = pair_dist.mean().clamp_min(1e-3)
        scale_inputs = (scale / (query_joint.shape[-1] ** 0.5)).clamp_min(1e-3)
        dist_normed = pair_dist / scale
        dist_normed[:, :, :num_queries] += torch.eye(
            num_queries, device=predictions.device, dtype=dist_normed.dtype
        )[None] * 100.0

        old_traj = (global_pred / scale_inputs).unsqueeze(0)
        target_traj = torch.cat([global_pred, global_gt], dim=0).unsqueeze(0) / scale_inputs
        total_force = torch.zeros_like(old_traj)
        positive_distributions = []
        pos_masses = []
        neg_masses = []
        raw_force_rms = []
        stats = {"scale": scale.detach(), "traj_scale": traj_block_scale.detach()}
        query_idx = torch.arange(num_queries, device=predictions.device)

        for radius in radii:
            logits = -dist_normed / radius
            affinity = torch.softmax(logits, dim=-1)
            affinity_t = torch.softmax(logits, dim=-2)
            affinity = torch.sqrt((affinity * affinity_t).clamp_min(0.0))
            aff_neg = affinity[:, :, :num_queries]
            aff_pos = affinity[:, :, num_queries:]
            aff_neg[:, query_idx, query_idx] = 0.0

            sum_pos = aff_pos.sum(dim=-1, keepdim=True)
            sum_neg = aff_neg.sum(dim=-1, keepdim=True)
            coeff = torch.cat([-aff_neg * sum_pos, aff_pos * sum_neg], dim=-1)
            force = torch.matmul(coeff, target_traj)
            force = force - coeff.sum(dim=-1, keepdim=True) * old_traj
            force_rms = force.square().mean().clamp_min(1e-8).sqrt()
            total_force = total_force + force / force_rms

            positive_distributions.append(aff_pos / sum_pos.clamp_min(eps))
            pos_masses.append(sum_pos.mean())
            neg_masses.append(sum_neg.mean())
            raw_force_rms.append(force_rms)
            stats[f"loss_R_{radius:g}"] = force.square().mean().detach()

        positive_distribution = torch.stack(positive_distributions).mean(dim=0).squeeze(0)
        own_scene_idx = torch.div(query_idx, num_modes, rounding_mode="floor")
        own_gt_affinity = positive_distribution[query_idx, own_scene_idx]
        effective_positive_count = positive_distribution.square().sum(dim=-1).clamp_min(eps).reciprocal()
        self_negative_affinity = affinity[0, query_idx, query_idx]
        goal = (old_traj + total_force).detach().squeeze(0)

        local_query_offset = scene_offset * num_modes
        local_query_count = predictions.shape[0] * num_modes
        local_goal = goal[local_query_offset:local_query_offset + local_query_count]
        target_shift_rms = total_force.square().mean().sqrt()

        endpoints = predictions.detach()[..., -1, :].float()
        endpoint_dist = torch.cdist(endpoints, endpoints)
        offdiag = ~torch.eye(num_modes, device=predictions.device, dtype=torch.bool)[None]

        stats.update({
            "force_norm": total_force.norm(dim=-1).mean().detach(),
            "raw_force_norm": torch.stack(raw_force_rms).mean().detach(),
            "target_shift_rms": target_shift_rms.detach(),
            "pos_aff": torch.stack(pos_masses).mean().detach(),
            "neg_aff": torch.stack(neg_masses).mean().detach(),
            "effective_positive_count": effective_positive_count.mean().detach(),
            "own_gt_affinity": own_gt_affinity.mean().detach(),
            "cross_scene_gt_affinity": (1.0 - own_gt_affinity).mean().detach(),
            "self_negative_affinity": self_negative_affinity.mean().detach(),
            "pairwise_endpoint_distance": endpoint_dist[offdiag.expand_as(endpoint_dist)].mean().detach(),
        })

    local_traj_scaled = local_traj.reshape(-1, local_traj.shape[-1]) / traj_block_scale / scale_inputs
    loss = F.mse_loss(local_traj_scaled, local_goal)
    if not torch.isfinite(loss):
        raise FloatingPointError("joint_drifting_loss produced NaN or Inf")
    return loss, stats


def _official_drifting_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor = None,
    r_list: Iterable[float] = (0.02, 0.1, 0.5),
    space: str = "traj",
    normalize_space: bool = False,
    soft_tau: float = 0.05,
    error_gate: float = 1.0,
    protect_gt_direction: bool = True,
    force_clip: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    gen_flat = flatten_trajectories(gen, space=space, normalize_space=normalize_space).unsqueeze(1)
    pos_flat = flatten_trajectories(fixed_pos.detach(), space=space, normalize_space=normalize_space).unsqueeze(1)
    if fixed_neg is None or fixed_neg.numel() == 0:
        neg_flat = gen_flat[:, :0].detach()
    else:
        neg_flat = flatten_trajectories(fixed_neg.detach(), space=space, normalize_space=normalize_space)

    old_gen = gen_flat.detach()
    targets = torch.cat([old_gen, neg_flat, pos_flat], dim=1)
    weight_gen = torch.ones(*old_gen.shape[:2], device=gen.device, dtype=gen.dtype)
    weight_neg = torch.ones(*neg_flat.shape[:2], device=gen.device, dtype=gen.dtype)
    weight_pos = torch.ones(*pos_flat.shape[:2], device=gen.device, dtype=gen.dtype)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    radii = tuple(float(r) for r in r_list)
    if not radii or min(radii) <= 0:
        raise ValueError(f"r_list must contain positive radii, got {radii}")
    if soft_tau <= 0:
        raise ValueError(f"soft_tau must be positive, got {soft_tau}")
    if error_gate <= 0:
        raise ValueError(f"error_gate must be positive, got {error_gate}")

    with torch.no_grad():
        dist = _batched_cdist(old_gen, targets, eps=eps)
        weighted_dist = dist * targets_w[:, None, :]
        scale = weighted_dist.mean(dim=(1, 2), keepdim=True)
        scale = scale / targets_w.mean(dim=1, keepdim=True).unsqueeze(-1).clamp_min(eps)
        scale = scale.clamp_min(1e-3)
        scale_inputs = (scale / (gen_flat.shape[-1] ** 0.5)).clamp_min(1e-3)

        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs
        dist_normed = dist / scale.clamp_min(1e-3)

        self_mask = torch.eye(old_gen.shape[1], device=gen.device, dtype=gen.dtype)[None] * 100.0
        dist_normed[:, :, :old_gen.shape[1]] = dist_normed[:, :, :old_gen.shape[1]] + self_mask

        total_force = torch.zeros_like(old_gen_scaled)
        split_idx = old_gen.shape[1] + neg_flat.shape[1]
        raw_force_norms = []
        pos_aff_values = []
        neg_aff_values = []
        stats = {"scale": scale.mean().detach()}
        for radius in radii:
            logits = -dist_normed / radius
            affinity = torch.softmax(logits, dim=-1)
            affinity_t = torch.softmax(logits, dim=-2)
            affinity = torch.sqrt((affinity * affinity_t).clamp_min(0.0))
            affinity = affinity * targets_w[:, None, :]

            aff_neg = affinity[:, :, :split_idx]
            aff_pos = affinity[:, :, split_idx:]

            sum_pos = aff_pos.sum(dim=-1, keepdim=True)
            coeff_neg = -aff_neg * sum_pos
            sum_neg = aff_neg.sum(dim=-1, keepdim=True)
            coeff_pos = aff_pos * sum_neg

            coeff = torch.cat([coeff_neg, coeff_pos], dim=-1)
            force = torch.matmul(coeff, targets_scaled)
            coeff_sum = coeff.sum(dim=-1, keepdim=True)
            force = force - coeff_sum * old_gen_scaled
            force_rms = force.square().mean(dim=(-1, -2), keepdim=True).sqrt()
            total_force = total_force + force / (force_rms + float(soft_tau))
            raw_force_norms.append(force_rms.mean())
            pos_aff_values.append(aff_pos.mean())
            neg_aff_values.append(aff_neg.mean())
            stats[f"loss_R_{radius:g}"] = force_rms.mean().detach()

        to_gt = pos_flat / scale_inputs - old_gen_scaled
        if protect_gt_direction:
            total_force = _remove_away_component(total_force, to_gt, eps)
            if space == "traj":
                protected_endpoint = _remove_away_component(total_force[..., -2:], to_gt[..., -2:], eps)
                total_force = torch.cat([total_force[..., :-2], protected_endpoint], dim=-1)
            elif space == "mixed":
                # The mixed representation contains the endpoint twice: once in
                # the flattened trajectory and once as the appended endpoint.
                protected_endpoint = _remove_away_component(total_force[..., -4:], to_gt[..., -4:], eps)
                total_force = torch.cat([total_force[..., :-4], protected_endpoint], dim=-1)

        winner_ade = torch.norm(gen.detach() - fixed_pos.detach(), dim=-1).mean(dim=-1)
        accuracy_gate = (winner_ade / float(error_gate)).clamp(min=0.0, max=1.0)[:, None, None]
        total_force = total_force * accuracy_gate

        if force_clip is not None and force_clip > 0:
            total_force_rms = total_force.square().mean(dim=(-1, -2), keepdim=True).sqrt()
            clip_scale = (float(force_clip) / total_force_rms.clamp_min(eps)).clamp_max(1.0)
            total_force = total_force * clip_scale

        goal_scaled = (old_gen_scaled + total_force).detach()

    gen_scaled = gen_flat / scale_inputs
    loss = F.mse_loss(gen_scaled, goal_scaled)
    if not torch.isfinite(loss):
        raise FloatingPointError("official drifting_loss produced NaN or Inf")
    stats["force_norm"] = total_force.norm(dim=-1).mean().detach()
    stats["raw_force_norm"] = torch.stack(raw_force_norms).mean().detach()
    stats["pos_aff"] = torch.stack(pos_aff_values).mean().detach()
    stats["neg_aff"] = torch.stack(neg_aff_values).mean().detach()
    return loss, stats


def drifting_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor = None,
    r_list: Iterable[float] = (0.1,),
    use_mdf: bool = False,
    include_old_gen_as_neg: bool = True,
    normalize_force: bool = True,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
    force_clip: float = 1.0,
    detach_target: bool = True,
    space: str = "traj",
    normalize_space: bool = True,
    loss_type: str = "legacy",
    soft_tau: float = 0.05,
    error_gate: float = 1.0,
    protect_gt_direction: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if gen.shape != fixed_pos.shape:
        raise ValueError(f"gen and fixed_pos must match, got {tuple(gen.shape)} and {tuple(fixed_pos.shape)}")
    if not torch.isfinite(gen).all() or not torch.isfinite(fixed_pos).all():
        raise FloatingPointError("drifting_loss received non-finite gen or fixed_pos")

    if loss_type in {"official", "method"}:
        return _official_drifting_loss(
            gen=gen,
            fixed_pos=fixed_pos,
            fixed_neg=fixed_neg,
            r_list=r_list,
            space=space,
            normalize_space=False,
            soft_tau=soft_tau,
            error_gate=error_gate,
            protect_gt_direction=protect_gt_direction,
            force_clip=force_clip,
        )
    if loss_type != "legacy":
        raise ValueError(f"Unsupported drift_loss_type={loss_type}")

    radii = tuple(float(r) for r in r_list)
    if not use_mdf:
        radii = radii[:1]
    if not radii or min(radii) <= 0:
        raise ValueError(f"r_list must contain positive radii, got {radii}")

    old_gen_flat = flatten_trajectories(gen.detach(), space=space, normalize_space=normalize_space)
    gen_flat = flatten_trajectories(gen, space=space, normalize_space=normalize_space)
    pos = fixed_pos.detach() if detach_target else fixed_pos
    pos_flat = flatten_trajectories(pos, space=space, normalize_space=normalize_space)

    neg_flat = None
    if fixed_neg is not None and fixed_neg.numel() > 0:
        neg = fixed_neg.detach() if detach_target else fixed_neg
        neg_flat = flatten_trajectories(neg, space=space, normalize_space=normalize_space)
    if include_old_gen_as_neg:
        old_as_neg = old_gen_flat.unsqueeze(1)
        neg_flat = old_as_neg if neg_flat is None else torch.cat([neg_flat, old_as_neg], dim=1)

    total_force = torch.zeros_like(old_gen_flat)
    pos_aff_values = []
    neg_aff_values = []
    for radius in radii:
        pos_vec = pos_flat - old_gen_flat
        pos_dist2 = pos_vec.square().mean(dim=-1)
        pos_aff = torch.exp(-pos_dist2 / (2.0 * radius * radius))
        total_force = total_force + float(pos_weight) * _safe_force(pos_vec, pos_aff, normalize_force)
        pos_aff_values.append(pos_aff.mean())

        if neg_flat is not None and neg_flat.shape[1] > 0:
            neg_vec = old_gen_flat.unsqueeze(1) - neg_flat
            neg_dist2 = neg_vec.square().mean(dim=-1)
            neg_aff = torch.exp(-neg_dist2 / (2.0 * radius * radius))
            neg_force = _safe_force(neg_vec, neg_aff, normalize_force).mean(dim=1)
            total_force = total_force + float(neg_weight) * neg_force
            neg_aff_values.append(neg_aff.mean())

    total_force = total_force / float(len(radii))
    force_norm = total_force.norm(dim=-1, keepdim=True)
    if force_clip is not None and force_clip > 0:
        scale = (float(force_clip) / force_norm.clamp_min(1e-6)).clamp_max(1.0)
        total_force = total_force * scale
        force_norm = total_force.norm(dim=-1)
    else:
        force_norm = force_norm.squeeze(-1)

    goal = old_gen_flat + total_force
    loss = F.mse_loss(gen_flat, goal.detach())
    if not torch.isfinite(loss):
        raise FloatingPointError("drifting_loss produced NaN or Inf")

    stats = {
        "force_norm": force_norm.mean().detach(),
        "pos_aff": torch.stack(pos_aff_values).mean().detach(),
        "neg_aff": (
            torch.stack(neg_aff_values).mean().detach()
            if neg_aff_values
            else gen.new_zeros(())
        ),
    }
    return loss, stats
