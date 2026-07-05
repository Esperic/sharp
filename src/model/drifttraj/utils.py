import torch


def scheduled_weight(base_weight: float, epoch: int, warmup_epochs: int) -> float:
    if base_weight <= 0:
        return 0.0
    if warmup_epochs <= 0:
        return float(base_weight)
    progress = min(max(float(epoch) / float(warmup_epochs), 0.0), 1.0)
    return float(base_weight) * progress


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor is not None and not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
