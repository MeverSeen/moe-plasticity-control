"""Frozen minimal candidate action set. Actions scale v8 actual Adam displacement."""
from __future__ import annotations

import hashlib

import torch

ACTIONS = ("global_0", "global_0.5", "global_1", "static_top2",
           "route_new_top2", "route_old_top2")
STATE_BLIND = ("global_0", "global_0.5", "global_1", "static_top2")


def action_for(observation: torch.Tensor, name: str) -> torch.Tensor:
    """Return [layers, experts] multipliers from v8 observation features.

    v8 obs[...,1] is old routing gate mass; obs[...,3] is new routing gate mass.
    The two routed actions always select exactly two experts per layer.
    """
    if observation.ndim != 3 or observation.shape[0] != 2 or observation.shape[-1] < 4:
        raise ValueError("v8 observation must have [2, experts, features] shape")
    n = observation.shape[1]
    if n < 2:
        raise ValueError("need at least two experts per layer")
    if name not in ACTIONS:
        raise ValueError(f"unregistered action {name}")
    if name.startswith("global_"):
        return torch.full((2, n), float(name.removeprefix("global_")), dtype=observation.dtype)
    result = torch.zeros((2, n), dtype=observation.dtype)
    if name == "static_top2":
        result[:, :2] = 1.0
    else:
        new_minus_old = observation[..., 3] - observation[..., 1]
        scores = new_minus_old if name == "route_new_top2" else -new_minus_old
        winners = scores.topk(2, dim=-1).indices
        result.scatter_(dim=1, index=winners, value=1.)
    return result


def stream_seed(root: int, reference: str, stage: int, stream: int) -> int:
    """Deterministic per-state stream seed; does not depend on the candidate."""
    payload = f"v12:{root}:{reference}:{stage}:{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def batch_stream_fingerprint(root: int, reference: str, stage: int, stream: int,
                             new_size: int, old_size: int, steps: int, batch: int) -> str:
    """Hash the paired new/replay index sequence expected during one v8 interval."""
    generator = torch.Generator().manual_seed(stream_seed(root, reference, stage, stream))
    digest = hashlib.sha256()
    for _ in range(steps):
        digest.update(torch.randint(new_size, (batch,), generator=generator).numpy().tobytes())
        if old_size:
            digest.update(torch.randint(old_size, (batch,), generator=generator).numpy().tobytes())
    return digest.hexdigest()
