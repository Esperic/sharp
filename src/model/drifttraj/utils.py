import math

import torch


def scheduled_weight(
    base_weight: float,
    epoch: int,
    warmup_epochs: int,
    decay_start_epoch: int = None,
    decay_end_epoch: int = None,
) -> float:
    if base_weight <= 0:
        return 0.0

    epoch = float(epoch)
    if warmup_epochs > 0:
        warmup_progress = min(max(epoch / float(warmup_epochs), 0.0), 1.0)
        weight = float(base_weight) * warmup_progress
    else:
        weight = float(base_weight)

    if decay_start_epoch is None and decay_end_epoch is None:
        return weight
    if decay_start_epoch is None or decay_end_epoch is None:
        raise ValueError("decay_start_epoch and decay_end_epoch must be set together")
    if decay_end_epoch <= decay_start_epoch:
        raise ValueError("decay_end_epoch must be greater than decay_start_epoch")
    if decay_start_epoch < max(int(warmup_epochs), 0):
        raise ValueError("decay_start_epoch must not precede the end of warmup")
    if epoch <= float(decay_start_epoch):
        return weight
    if epoch >= float(decay_end_epoch):
        return 0.0

    decay_progress = (epoch - float(decay_start_epoch)) / float(decay_end_epoch - decay_start_epoch)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return float(base_weight) * cosine_decay


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor is not None and not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
