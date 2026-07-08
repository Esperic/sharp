from typing import Dict, Iterable, Tuple

import torch
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


def _batched_cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    xy_dot = torch.einsum("bns,bms->bnm", x, y)
    x_norm = x.square().sum(dim=-1, keepdim=True)
    y_norm = y.square().sum(dim=-1, keepdim=True).transpose(1, 2)
    sq_dist = (x_norm + y_norm - 2.0 * xy_dot).clamp_min(0.0)
    return (sq_dist + eps).sqrt()


def _official_drifting_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor = None,
    r_list: Iterable[float] = (0.02, 0.1, 0.5),
    space: str = "traj",
    normalize_space: bool = False,
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

    with torch.no_grad():
        dist = _batched_cdist(old_gen, targets, eps=eps)
        weighted_dist = dist * targets_w[:, None, :]
        scale = (weighted_dist.mean() / targets_w.mean().clamp_min(eps)).clamp_min(1e-3)
        scale_inputs = (scale / (gen_flat.shape[-1] ** 0.5)).clamp_min(1e-3)

        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs
        dist_normed = dist / scale.clamp_min(1e-3)

        self_mask = torch.eye(old_gen.shape[1], device=gen.device, dtype=gen.dtype)[None] * 100.0
        dist_normed[:, :, :old_gen.shape[1]] = dist_normed[:, :, :old_gen.shape[1]] + self_mask

        total_force = torch.zeros_like(old_gen_scaled)
        split_idx = old_gen.shape[1] + neg_flat.shape[1]
        stats = {"scale": scale.detach()}
        for radius in radii:
            logits = -dist_normed / radius
            affinity = torch.softmax(logits, dim=-1)
            affinity_t = torch.softmax(logits, dim=-2)
            affinity = torch.sqrt((affinity * affinity_t).clamp_min(1e-8))
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
            force_norm = force.square().mean().clamp_min(eps).sqrt()
            total_force = total_force + force / force_norm
            stats[f"loss_R_{radius:g}"] = force_norm.detach()

        goal_scaled = (old_gen_scaled + total_force).detach()

    gen_scaled = gen_flat / scale_inputs
    loss = F.mse_loss(gen_scaled, goal_scaled)
    if not torch.isfinite(loss):
        raise FloatingPointError("official drifting_loss produced NaN or Inf")
    stats["force_norm"] = total_force.norm(dim=-1).mean().detach()
    stats["pos_aff"] = gen.new_zeros(())
    stats["neg_aff"] = gen.new_zeros(())
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
