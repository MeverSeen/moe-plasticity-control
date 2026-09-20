"""Distances of effective LoRA updates, invariant to factor rescaling."""
import torch


@torch.no_grad()
def effective_distance(weights, reference, scale=1.0):
    """Return per-expert squared Frobenius distance without materializing BA.

    Factor ordering is [gate.A, gate.B, up.A, up.B, down.A, down.B].
    Float64 contractions limit cancellation when the two products are close.
    """
    if len(weights) != len(reference) or len(weights) % 6:
        raise ValueError("Expected three A/B factor pairs per expert")
    totals = torch.zeros(len(weights) // 6, dtype=torch.float64)
    factors = torch.zeros_like(totals)
    for projection in range(3):
        a = torch.stack([weights[i + 2 * projection].detach().cpu().double()
                         for i in range(0, len(weights), 6)])
        b = torch.stack([weights[i + 2 * projection + 1].detach().cpu().double()
                         for i in range(0, len(weights), 6)])
        a0 = torch.stack([reference[i + 2 * projection].detach().cpu().double()
                          for i in range(0, len(weights), 6)])
        b0 = torch.stack([reference[i + 2 * projection + 1].detach().cpu().double()
                          for i in range(0, len(weights), 6)])
        aa, bb = a @ a.transpose(-1, -2), b.transpose(-1, -2) @ b
        aa0, bb0 = a0 @ a0.transpose(-1, -2), b0.transpose(-1, -2) @ b0
        cross = (b.transpose(-1, -2) @ b0) * (a @ a0.transpose(-1, -2))
        distance = ((aa * bb).sum((-1, -2)) + (aa0 * bb0).sum((-1, -2))
                    - 2 * cross.sum((-1, -2)))
        totals += distance.clamp_min(0) * scale ** 2
        factors += (a - a0).square().sum((-1, -2)) + (b - b0).square().sum((-1, -2))
    return {"effective_sq_per_expert": totals.tolist(),
            "effective_sq": float(totals.sum()),
            "factor_sq_per_expert": factors.tolist(),
            "factor_sq": float(factors.sum())}
