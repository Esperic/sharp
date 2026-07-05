import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _as_tensor(array: np.ndarray, name: str, ndim: int) -> torch.Tensor:
    tensor = torch.as_tensor(array, dtype=torch.float32)
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dims, got shape {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor


class GaussianMixturePrior(nn.Module):
    def __init__(
        self,
        path: str,
        k: int,
        embed_dim: int,
        std_floor: float = 0.05,
        sampling: str = "query_aligned",
        train_noise_scale: float = 1.0,
        eval_noise_scale: float = 0.0,
        condition_query: bool = True,
        condition_pi: bool = True,
        condition_memory: bool = False,
        use_prior_logit_bias: bool = True,
    ) -> None:
        super().__init__()
        if path is None:
            raise ValueError("gmp_path must be set when use_gmp=True")
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"GMP parameter file does not exist: {path}")
        if sampling not in {"query_aligned", "random", "prior_topk"}:
            raise ValueError(f"Unsupported gmp_sampling={sampling}")

        payload = np.load(path_obj, allow_pickle=False)
        center_points = _as_tensor(payload["center_points"], "center_points", 2)
        center_std = _as_tensor(payload["center_std"], "center_std", 2).clamp_min(float(std_floor))
        mixture_weights = _as_tensor(payload["mixture_weights"], "mixture_weights", 1)
        if center_points.shape[-1] != 2 or center_std.shape[-1] != 2:
            raise ValueError("center_points and center_std must have shape [K, 2]")
        if center_points.shape[0] != center_std.shape[0] or center_points.shape[0] != mixture_weights.shape[0]:
            raise ValueError(
                "GMP center_points, center_std, and mixture_weights must share the same K; "
                f"got {center_points.shape[0]}, {center_std.shape[0]}, {mixture_weights.shape[0]}"
            )

        mixture_weights = mixture_weights.clamp_min(0)
        weight_sum = mixture_weights.sum()
        if weight_sum <= 0:
            raise ValueError("mixture_weights must contain at least one positive value")
        mixture_weights = mixture_weights / weight_sum

        self.k = int(k)
        self.gmp_k = int(center_points.shape[0])
        self.embed_dim = int(embed_dim)
        self.sampling = sampling
        self.train_noise_scale = float(train_noise_scale)
        self.eval_noise_scale = float(eval_noise_scale)
        self.condition_query = bool(condition_query)
        self.condition_pi = bool(condition_pi)
        self.condition_memory = bool(condition_memory)
        self.use_prior_logit_bias = bool(use_prior_logit_bias)
        self.metadata = {}
        if "metadata" in payload.files:
            try:
                self.metadata = json.loads(str(payload["metadata"].item()))
            except (json.JSONDecodeError, ValueError, AttributeError):
                self.metadata = {}

        self.register_buffer("center_points", center_points, persistent=True)
        self.register_buffer("center_std", center_std, persistent=True)
        self.register_buffer("mixture_weights", mixture_weights, persistent=True)

        self.gmp_query_proj = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.gmp_pi_proj = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        self.gmp_memory_film = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2 * embed_dim),
        )
        self._init_conditioners()

    def _init_conditioners(self) -> None:
        for module in (self.gmp_query_proj, self.gmp_pi_proj, self.gmp_memory_film):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
            last = module[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _component_indices(self, batch_size: int, num_queries: int, device: torch.device) -> torch.Tensor:
        if self.sampling == "random" and self.training:
            return torch.multinomial(
                self.mixture_weights.to(device),
                num_samples=batch_size * num_queries,
                replacement=True,
            ).view(batch_size, num_queries)

        if self.sampling == "prior_topk":
            order = torch.argsort(self.mixture_weights.to(device), descending=True)
            base = order[torch.arange(num_queries, device=device) % order.numel()]
        else:
            base = torch.arange(num_queries, device=device) % self.gmp_k
        return base.view(1, num_queries).repeat(batch_size, 1)

    def sample_xy(
        self,
        batch_size: int,
        num_queries: int,
        device: torch.device,
        training: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        training = self.training if training is None else bool(training)
        comp_idx = self._component_indices(batch_size, num_queries, device)
        means = self.center_points.to(device)[comp_idx]
        std = self.center_std.to(device)[comp_idx]
        noise_scale = self.train_noise_scale if training else self.eval_noise_scale
        if noise_scale > 0:
            xy = means + torch.randn_like(means) * std * noise_scale
        else:
            xy = means
        if xy.shape != (batch_size, num_queries, 2):
            raise RuntimeError(f"GMP sample shape mismatch: got {tuple(xy.shape)}")
        if not torch.isfinite(xy).all():
            raise FloatingPointError("GMP sample contains NaN or Inf")
        return xy, comp_idx

    def xy_to_query_delta(self, xy: torch.Tensor) -> torch.Tensor:
        if not self.condition_query:
            return xy.new_zeros(*xy.shape[:-1], self.embed_dim)
        if xy.shape[-1] != 2:
            raise ValueError(f"GMP xy must end in 2, got {tuple(xy.shape)}")
        return self.gmp_query_proj(xy)

    def xy_to_pi_bias(self, xy: torch.Tensor, comp_idx: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if not self.condition_pi:
            return None
        bias = self.gmp_pi_proj(xy).squeeze(-1)
        if self.use_prior_logit_bias and comp_idx is not None:
            log_w = torch.log(self.mixture_weights.to(xy.device).clamp_min(1e-8))
            bias = bias + log_w[comp_idx]
        return bias

    def memory_film(self, x_encoder: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        if not self.condition_memory:
            return x_encoder
        pooled_xy = xy.mean(dim=1)
        gamma_beta = self.gmp_memory_film(pooled_xy).unsqueeze(1)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return x_encoder * (1.0 + 0.01 * torch.tanh(gamma)) + 0.01 * beta

    def component_entropy(self, comp_idx: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(comp_idx.reshape(-1), minlength=self.gmp_k).float()
        probs = counts / counts.sum().clamp_min(1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum()
        return entropy
