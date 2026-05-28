"""Laguerre-function basis on ``[0, ∞)`` — exponentially-weighted domain.

Eigenfunctions: ``φₙ(z) = Lₙ(z) · exp(-z/2)`` where ``Lₙ`` is the
(simple, ``α = 0``) Laguerre polynomial.
Eigenvalues:    ``λₙ = n + 1/2`` for the associated operator
``-z u'' - (1 - z) u' + (1/2) u``.

Suitable for non-negative semi-infinite latents.
"""

import torch
from torch import nn

from igl.exceptions import IGLConfigError


def _laguerre_recurrence(z: torch.Tensor, n_modes: int) -> torch.Tensor:
    """Evaluate ``L_0(z) … L_{n_modes-1}(z)`` via the standard recurrence."""
    if n_modes == 1:
        return torch.ones((*z.shape, 1), dtype=z.dtype, device=z.device)
    polys = [torch.ones_like(z), 1.0 - z]
    for n in range(1, n_modes - 1):
        next_p = ((2.0 * n + 1.0 - z) * polys[-1] - n * polys[-2]) / (n + 1.0)
        polys.append(next_p)
    return torch.stack(polys, dim=-1)


class LaguerreBasis(nn.Module):
    """1-D Laguerre-function spectral basis (exponentially-weighted domain).

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
        self.domain = (0.0, float("inf"))
        indices = torch.arange(0, n_modes, dtype=torch.float32)
        self.register_buffer("eigenvalues", indices + 0.5)

    def evaluate(self, z: torch.Tensor, /) -> torch.Tensor:
        polys = _laguerre_recurrence(z, self.n_modes)
        weight = torch.exp(-0.5 * z).unsqueeze(-1)
        return weight * polys

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.evaluate(z)


__all__ = ["LaguerreBasis"]
