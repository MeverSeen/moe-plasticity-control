"""Two-layer fixed-routing residual MoE used by the v8 retention experiments.

Routing is a deterministic function of the *raw image*: a 4x4 average-pooled
RGB image is projected with frozen random matrices.  Consequently, training the
stem or an expert cannot change the selected experts for an image.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _Expert(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, width)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _MoELayer(nn.Module):
    def __init__(self, experts: int, width: int, expert_hidden: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [_Expert(width, expert_hidden) for _ in range(experts)]
        )

    def forward(self, x: Tensor, indices: Tensor, gates: Tensor) -> Tensor:
        """Apply only selected experts, then add their gated result residually."""
        update = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            selected = indices.eq(expert_id)
            rows, slots = selected.nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            values = expert(x.index_select(0, rows))
            update.index_add_(0, rows, values * gates[rows, slots].unsqueeze(1))
        return x + update


class MultiLayerMoE(nn.Module):
    """A task-ID-visible 5-way classifier with two fixed-routing MoE layers."""

    num_layers = 2
    feature_dim = 3 * 4 * 4

    def __init__(
        self,
        tasks: int,
        experts: int = 8,
        width: int = 32,
        expert_hidden: int = 64,
        top_k: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if tasks < 1 or experts < 1 or not 1 <= top_k <= experts:
            raise ValueError("tasks and experts must be positive, with 1 <= top_k <= experts")
        self.tasks, self.num_experts, self.width, self.top_k = tasks, experts, width, top_k
        # Isolate every randomized initialization from the caller's RNG stream.
        # This also makes `seed` cover the stem, experts, heads, and routers.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.stem = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(16 * 2 * 2, width),
            )
            self.layers = nn.ModuleList(
                [_MoELayer(experts, width, expert_hidden) for _ in range(self.num_layers)]
            )
            self.heads = nn.ModuleList([nn.Linear(width, 5) for _ in range(tasks)])
            # Buffers travel with state_dict but are never optimized.
            matrices = torch.randn(self.num_layers, self.feature_dim, experts)
            matrices.div_(self.feature_dim ** 0.5)
        self.register_buffer("router_matrices", matrices)

    def _raw_features(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1:] != (3, 32, 32):
            raise ValueError("x must have shape [N, 3, 32, 32]")
        return F.adaptive_avg_pool2d(x, (4, 4)).flatten(1)

    def _routes(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self._raw_features(x)
        scores = torch.einsum("nf,lfe->lne", features, self.router_matrices)
        top_scores, indices = scores.topk(self.top_k, dim=-1)
        return indices, F.softmax(top_scores, dim=-1)

    def forward(self, x: Tensor, task_ids: Tensor) -> Tensor:
        if task_ids.ndim != 1 or task_ids.shape[0] != x.shape[0]:
            raise ValueError("task_ids must have shape [N]")
        if task_ids.dtype != torch.long:
            raise TypeError("task_ids must be torch.long")
        if task_ids.numel() and (task_ids.min() < 0 or task_ids.max() >= self.tasks):
            raise ValueError("task_ids contains an unknown task")
        indices, gates = self._routes(x)
        hidden = self.stem(x)
        for layer_id, layer in enumerate(self.layers):
            hidden = layer(hidden, indices[layer_id], gates[layer_id])
        all_logits = torch.stack([head(hidden) for head in self.heads], dim=1)
        return all_logits[torch.arange(x.shape[0], device=x.device), task_ids]

    @torch.no_grad()
    def route_stats(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return per-layer usage and gate mass, each normalized by batch size.

        ``usage.sum(-1) == top_k`` and ``gate_mass.sum(-1) == 1`` for a
        non-empty batch.  These are fractions per input, not raw counts.
        """
        if x.shape[0] == 0:
            raise ValueError("route_stats requires a non-empty batch")
        indices, gates = self._routes(x)
        usage = x.new_zeros((self.num_layers, self.num_experts))
        mass = x.new_zeros((self.num_layers, self.num_experts))
        ones = torch.ones_like(indices, dtype=x.dtype)
        for layer_id in range(self.num_layers):
            usage[layer_id].scatter_add_(0, indices[layer_id].reshape(-1), ones[layer_id].reshape(-1))
            mass[layer_id].scatter_add_(0, indices[layer_id].reshape(-1), gates[layer_id].reshape(-1))
        return usage / x.shape[0], mass / x.shape[0]

    def expert_parameters(self) -> List[List[List[nn.Parameter]]]:
        """Parameters grouped as ``layers -> experts -> parameters``."""
        return [[list(expert.parameters()) for expert in layer.experts] for layer in self.layers]


class ScaledAdam(torch.optim.Adam):
    """Adam whose final expert parameter displacement is scaled per expert.

    ``step(multipliers)`` first performs a normal Adam update, including all
    moment updates, then rescales each expert's actual displacement.  A zero
    multiplier therefore leaves expert weights unchanged while retaining Adam
    moments.  Experts not selected by routing have no gradient, so Adam creates
    no new state and this class does not move them.
    """

    def __init__(self, model: MultiLayerMoE, lr: float, **kwargs: object) -> None:
        super().__init__(model.parameters(), lr=lr, **kwargs)
        self._expert_parameters = model.expert_parameters()

    def __getstate__(self):
        """Keep expert grouping when RetentionEnv is deep-copied for branches."""
        state = super().__getstate__()
        state["_expert_parameters"] = self._expert_parameters
        return state

    @torch.no_grad()
    def step(self, multipliers: Tensor, closure: Optional[Callable[[], float]] = None):  # type: ignore[override]
        expected = (len(self._expert_parameters), len(self._expert_parameters[0]))
        if tuple(multipliers.shape) != expected:
            raise ValueError("multipliers must have shape [2, experts]")
        if not multipliers.is_floating_point():
            raise TypeError("multipliers must be floating point")
        if not torch.isfinite(multipliers).all():
            raise ValueError("multipliers must be finite")
        if (multipliers < 0).any():
            raise ValueError("multipliers must be non-negative")
        before = [
            [[parameter.detach().clone() for parameter in expert] for expert in layer]
            for layer in self._expert_parameters
        ]
        loss = super().step(closure)
        for layer_id, layer in enumerate(self._expert_parameters):
            for expert_id, expert in enumerate(layer):
                scale = multipliers[layer_id, expert_id]
                for parameter, old in zip(expert, before[layer_id][expert_id]):
                    parameter.copy_(old + (parameter - old) * scale.to(parameter))
        return loss


def optimizer_step(model: MultiLayerMoE, optimizer: ScaledAdam, multipliers: Tensor):
    """Explicit helper for experiment loops using a scaled optimizer step."""
    if not isinstance(optimizer, ScaledAdam):
        raise TypeError("optimizer_step requires ScaledAdam")
    return optimizer.step(multipliers)
