# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
"""Hermite-function basis on ``R`` — eigenstates of the harmonic oscillator.

Eigenfunctions: ``φₙ(z) = Hₙ(z) · exp(-z²/2) / √(2ⁿ n! √π)`` where ``Hₙ``
is the physicists' Hermite polynomial.
Eigenvalues:    ``λₙ = 2n + 1`` (no null space).

Suitable for unbounded latents (Gaussian-weighted domain). Inputs are
expected in roughly ``z ∈ [-4, 4]`` for ``n_modes ≤ 16``.
"""

import math

import torch
from torch import nn

from igl.exceptions import IGLConfigError


def _hermite_recurrence(z: torch.Tensor, n_modes: int) -> torch.Tensor:
    """Evaluate ``H_0(z) … H_{n_modes-1}(z)`` via physicists' recurrence."""
    if n_modes == 1:
        return torch.ones((*z.shape, 1), dtype=z.dtype, device=z.device)
    polys = [torch.ones_like(z), 2.0 * z]
    for n in range(1, n_modes - 1):
        next_p = 2.0 * z * polys[-1] - 2.0 * n * polys[-2]
        polys.append(next_p)
    return torch.stack(polys, dim=-1)


class HermiteBasis(nn.Module):
    """1-D Hermite-function spectral basis (Gaussian-weighted domain).

    Args:
        n_modes: Number of modes ``K``.

    Raises:
        IGLConfigError: For ``n_modes < 1``.
    """

    n_modes: int
    null_indices: tuple[int, ...]
    domain: tuple[float, float]

    def __init__(self, n_modes: int = 16) -> None:
        super().__init__()
        if n_modes < 1:
            raise IGLConfigError(f"n_modes must be >= 1, got {n_modes}")
        self.n_modes = n_modes
        self.null_indices = ()
        self.domain = (-float("inf"), float("inf"))
        indices = torch.arange(0, n_modes, dtype=torch.float32)
        self.register_buffer("eigenvalues", 2.0 * indices + 1.0)
        # Precompute normalisation constants 1 / sqrt(2^n · n! · √π).
        log_factorial = torch.lgamma(indices + 1.0)
        log_norm = -0.5 * (indices * math.log(2.0) + log_factorial + 0.5 * math.log(math.pi))
        self.register_buffer("_norm", torch.exp(log_norm))

    def evaluate(self, z: torch.Tensor, /) -> torch.Tensor:
        polys = _hermite_recurrence(z, self.n_modes)
        weight = torch.exp(-0.5 * z**2).unsqueeze(-1)
        norm_buf: torch.Tensor = self._norm  # pyright: ignore[reportAssignmentType]
        norm = norm_buf.view(*([1] * z.dim()), -1)
        return norm * weight * polys

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.evaluate(z)


__all__ = ["HermiteBasis"]
