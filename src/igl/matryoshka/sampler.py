"""Truncation-level samplers for Matryoshka training.

Each sampler maps ``d_max → k ∈ {1, …, d_max}`` per training step.
"""

import torch

from igl.exceptions import IGLConfigError


class UniformSampler:
    """Uniform sampler: ``k ~ Uniform{1, …, d_max}``.

    The simplest Matryoshka sampling strategy, and the default. It gives equal
    weight to all truncation levels and lets the encoder discover the true
    intrinsic dimension without prior bias.
    """

    def __call__(self, d_max: int, /) -> int:
        if d_max < 1:
            raise IGLConfigError(f"d_max must be >= 1, got {d_max}")
        return int(torch.randint(1, d_max + 1, ()).item())


class PowerLawSampler:
    """Power-law sampler: ``P(k) ∝ k^{-α}``.

    Useful when prior knowledge suggests the effective dimension is small —
    biases sampling toward lower truncation levels so the encoder is more
    aggressively forced to compress.

    Args:
        alpha: Exponent (must be positive). Larger ``alpha`` puts more mass on
            small ``k``. Default ``1.0``.
    """

    alpha: float

    def __init__(self, *, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise IGLConfigError(f"alpha must be > 0, got {alpha}")
        self.alpha = alpha

    def __call__(self, d_max: int, /) -> int:
        if d_max < 1:
            raise IGLConfigError(f"d_max must be >= 1, got {d_max}")
        weights = torch.arange(1, d_max + 1, dtype=torch.float) ** (-self.alpha)
        weights = weights / weights.sum()
        return int(torch.multinomial(weights, 1).item()) + 1


__all__ = ["PowerLawSampler", "UniformSampler"]
