import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


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
        latent_dim: int = 16,
        full_traj_query: bool = False,
        future_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        if path is None:
            raise ValueError("gmp_path must be set when use_gmp=True")
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"GMP parameter file does not exist: {path}")
        if sampling not in {"query_aligned", "random", "prior_topk"}:
            raise ValueError(f"Unsupported gmp_sampling={sampling}")
        if full_traj_query and future_steps is None:
            raise ValueError("future_steps is required when gmp_full_traj_query=True")

        payload = np.load(path_obj, allow_pickle=False)
        center_points = torch.as_tensor(payload["center_points"], dtype=torch.float32)
        center_std = torch.as_tensor(payload["center_std"], dtype=torch.float32).clamp_min(float(std_floor))
        mixture_weights = torch.as_tensor(payload["mixture_weights"], dtype=torch.float32)
        if center_points.ndim == 2:
            center_points = center_points.unsqueeze(0)
            center_std = center_std.unsqueeze(0)
            mixture_weights = mixture_weights.unsqueeze(0)
        if center_points.ndim != 3 or center_std.ndim != 3 or mixture_weights.ndim != 2:
            raise ValueError(
                "GMP arrays must be [K,2]/[K] or type-specific [C,K,2]/[C,K]; "
                f"got {tuple(center_points.shape)}, {tuple(center_std.shape)}, "
                f"{tuple(mixture_weights.shape)}"
            )
        if not all(torch.isfinite(x).all() for x in (center_points, center_std, mixture_weights)):
            raise ValueError("GMP arrays contain non-finite values")
        if center_points.shape[-1] != 2 or center_std.shape[-1] != 2:
            raise ValueError("center_points and center_std must end in 2")
        if center_points.shape[:2] != center_std.shape[:2] or center_points.shape[:2] != mixture_weights.shape:
            raise ValueError(
                "GMP center_points, center_std, and mixture_weights must share [C,K]; "
                f"got {tuple(center_points.shape)}, {tuple(center_std.shape)}, "
                f"{tuple(mixture_weights.shape)}"
            )

        mixture_weights = mixture_weights.clamp_min(0)
        weight_sum = mixture_weights.sum(dim=-1, keepdim=True)
        if (weight_sum <= 0).any():
            raise ValueError("Each target type must have at least one positive mixture weight")
        mixture_weights = mixture_weights / weight_sum

        self.k = int(k)
        self.num_types = int(center_points.shape[0])
        self.gmp_k = int(center_points.shape[1])
        if full_traj_query and self.gmp_k != self.k:
            raise ValueError(
                "Full trajectory queries require one anchor per decoder mode; "
                f"got GMP K={self.gmp_k}, model K={self.k}"
            )
        self.embed_dim = int(embed_dim)
        self.latent_dim = int(latent_dim)
        self.sampling = sampling
        self.train_noise_scale = float(train_noise_scale)
        self.eval_noise_scale = float(eval_noise_scale)
        self.condition_query = bool(condition_query)
        self.condition_pi = bool(condition_pi)
        self.condition_memory = bool(condition_memory)
        self.use_prior_logit_bias = bool(use_prior_logit_bias)
        self.full_traj_query = bool(full_traj_query)
        self.metadata = {}
        if "metadata" in payload.files:
            try:
                self.metadata = json.loads(str(payload["metadata"].item()))
            except (json.JSONDecodeError, ValueError, AttributeError):
                self.metadata = {}

        self.register_buffer("center_points", center_points, persistent=True)
        self.register_buffer("center_std", center_std, persistent=True)
        self.register_buffer("mixture_weights", mixture_weights, persistent=True)

        if self.full_traj_query:
            if self.sampling == "random":
                raise ValueError("gmp_sampling=random is incompatible with stable trajectory queries")
            if "cluster_trajs" not in payload.files:
                raise ValueError("Full trajectory query mode requires cluster_trajs in the GMP file")
            cluster_trajs = torch.as_tensor(payload["cluster_trajs"], dtype=torch.float32)
            if cluster_trajs.ndim == 3:
                cluster_trajs = cluster_trajs.unsqueeze(0)
            expected_prefix = (self.num_types, self.gmp_k)
            if (
                cluster_trajs.ndim != 4
                or cluster_trajs.shape[:2] != expected_prefix
                or cluster_trajs.shape[-1] != 2
            ):
                raise ValueError(
                    "cluster_trajs must share [C,K,T,2] with the GMP; "
                    f"got {tuple(cluster_trajs.shape)}"
                )
            if cluster_trajs.shape[2] != int(future_steps):
                raise ValueError(
                    f"GMP trajectories have T={cluster_trajs.shape[2]}, expected future_steps={future_steps}"
                )
            if not torch.isfinite(cluster_trajs).all():
                raise ValueError("GMP trajectories contain non-finite values")
            self.register_buffer("cluster_trajs", cluster_trajs, persistent=True)

        self.gmp_xy_to_latent = nn.Linear(2, self.latent_dim)
        self.gmp_query_proj = nn.Linear(self.latent_dim, embed_dim)
        self.gmp_traj_query_proj = (
            nn.Sequential(
                nn.Linear(int(future_steps) * 2, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            if self.full_traj_query
            else None
        )
        self.gmp_pi_proj = nn.Sequential(
            nn.Linear(self.latent_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        self.gmp_memory_norm = nn.LayerNorm(embed_dim)
        self.gmp_memory_film = nn.Linear(self.latent_dim, 2 * embed_dim)
        self._init_conditioners()
        if self.gmp_traj_query_proj is not None:
            nn.init.zeros_(self.gmp_traj_query_proj[-1].weight)
            nn.init.zeros_(self.gmp_traj_query_proj[-1].bias)
        self._freeze_disabled_conditioners()

    def _init_conditioners(self) -> None:
        modules = [self.gmp_xy_to_latent, self.gmp_query_proj, self.gmp_pi_proj, self.gmp_memory_film]
        if self.gmp_traj_query_proj is not None:
            modules.append(self.gmp_traj_query_proj)
        for module in modules:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        nn.init.ones_(self.gmp_memory_norm.weight)
        nn.init.zeros_(self.gmp_memory_norm.bias)

    @staticmethod
    def _set_trainable(module: nn.Module, trainable: bool) -> None:
        for param in module.parameters():
            param.requires_grad = trainable

    def _freeze_disabled_conditioners(self) -> None:
        uses_latent = (
            (self.condition_query and not self.full_traj_query)
            or self.condition_pi
            or self.condition_memory
        )
        self._set_trainable(self.gmp_xy_to_latent, uses_latent)
        self._set_trainable(self.gmp_query_proj, self.condition_query and not self.full_traj_query)
        if self.gmp_traj_query_proj is not None:
            self._set_trainable(self.gmp_traj_query_proj, self.condition_query)
        self._set_trainable(self.gmp_pi_proj, self.condition_pi)
        self._set_trainable(self.gmp_memory_film, self.condition_memory)
        self._set_trainable(self.gmp_memory_norm, self.condition_memory)

    def _type_indices(
        self,
        batch_size: int,
        device: torch.device,
        target_types: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.num_types == 1:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if target_types is None:
            raise ValueError("target_types is required for a type-specific GMP")
        target_types = torch.as_tensor(target_types, dtype=torch.long, device=device).reshape(-1)
        if target_types.numel() != batch_size:
            raise ValueError(f"target_types must have {batch_size} entries, got {target_types.numel()}")
        if (target_types < 0).any() or (target_types >= self.num_types).any():
            raise ValueError(
                f"target_types must be in [0, {self.num_types - 1}], got "
                f"{target_types.unique().tolist()}"
            )
        return target_types

    def _component_indices(
        self,
        batch_size: int,
        num_queries: int,
        device: torch.device,
        target_types: torch.Tensor,
        training: bool,
    ) -> torch.Tensor:
        weights = self.mixture_weights.to(device)[target_types]
        if self.sampling == "random" and training:
            return torch.multinomial(weights, num_samples=num_queries, replacement=True)

        if self.sampling == "prior_topk":
            order = torch.argsort(weights, dim=-1, descending=True)
            return order[:, torch.arange(num_queries, device=device) % self.gmp_k]
        else:
            base = torch.arange(num_queries, device=device) % self.gmp_k
        return base.view(1, num_queries).repeat(batch_size, 1)

    def sample_xy(
        self,
        batch_size: int,
        num_queries: int,
        device: torch.device,
        training: Optional[bool] = None,
        target_types: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        training = self.training if training is None else bool(training)
        type_idx = self._type_indices(batch_size, device, target_types)
        comp_idx = self._component_indices(batch_size, num_queries, device, type_idx, training)
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
        means = self.center_points.to(device)[type_idx][batch_idx, comp_idx]
        std = self.center_std.to(device)[type_idx][batch_idx, comp_idx]
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
        return self.gmp_query_proj(self.xy_to_latent(xy))

    def trajectory_queries(
        self,
        comp_idx: torch.Tensor,
        target_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.full_traj_query:
            raise RuntimeError("Trajectory queries are available only when gmp_full_traj_query=True")
        type_idx = self._type_indices(comp_idx.shape[0], comp_idx.device, target_types)
        batch_idx = torch.arange(comp_idx.shape[0], device=comp_idx.device).unsqueeze(1)
        return self.cluster_trajs.to(comp_idx.device)[type_idx][batch_idx, comp_idx]

    def trajectory_to_query_delta(self, trajectories: torch.Tensor) -> torch.Tensor:
        if not self.condition_query:
            return trajectories.new_zeros(*trajectories.shape[:2], self.embed_dim)
        if self.gmp_traj_query_proj is None:
            raise RuntimeError("Trajectory query projection is not initialized")
        return self.gmp_traj_query_proj(trajectories.flatten(start_dim=2))

    def xy_to_pi_bias(
        self,
        xy: torch.Tensor,
        comp_idx: Optional[torch.Tensor] = None,
        target_types: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not self.condition_pi:
            return None
        bias = self.gmp_pi_proj(self.xy_to_latent(xy)).squeeze(-1)
        if self.use_prior_logit_bias and comp_idx is not None:
            type_idx = self._type_indices(xy.shape[0], xy.device, target_types)
            weights = self.mixture_weights.to(xy.device)[type_idx]
            bias = bias + torch.log(weights.gather(1, comp_idx).clamp_min(1e-8))
        return bias

    def xy_to_latent(self, xy: torch.Tensor) -> torch.Tensor:
        if xy.shape[-1] != 2:
            raise ValueError(f"GMP xy must end in 2, got {tuple(xy.shape)}")
        return self.gmp_xy_to_latent(xy)

    def memory_film(self, x_encoder: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        if not self.condition_memory:
            return x_encoder
        pooled_z = self.xy_to_latent(xy).mean(dim=1)
        gamma_beta = self.gmp_memory_film(pooled_z).unsqueeze(1)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        x_encoder = self.gmp_memory_norm(x_encoder)
        return x_encoder * (1.0 + gamma) + beta

    def component_entropy(self, comp_idx: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(comp_idx.reshape(-1), minlength=self.gmp_k).float()
        probs = counts / counts.sum().clamp_min(1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum()
        return entropy
