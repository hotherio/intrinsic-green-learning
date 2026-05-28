"""Legendre polynomial basis on ``[-1, 1]`` (input mapped from ``[0, 1]``).

Eigenfunctions: ``Pₙ(x)`` Legendre polynomials.
Eigenvalues for the Legendre operator: ``λₙ = n(n + 1)`` (``λ₀ = ε``).
"""

import torch
from torch import nn

from igl.exceptions import IGLConfigError


def _legendre_recurrence(x: torch.Tensor, n_modes: int) -> torch.Tensor:
    """Evaluate ``P_0(x) … P_{n_modes-1}(x)`` via Bonnet's recurrence."""
    if n_modes == 1:
        return torch.ones((*x.shape, 1), dtype=x.dtype, device=x.device)
    polys = [torch.ones_like(x), x.clone()]
    for n in range(1, n_modes - 1):
        next_p = ((2.0 * n + 1.0) * x * polys[-1] - n * polys[-2]) / (n + 1.0)
        polys.append(next_p)
    return torch.stack(polys, dim=-1)


class LegendreBasis(nn.Module):
    """1-D Legendre spectral basis.

    Inputs mapped from ``[0, 1]`` to ``[-1, 1]`` via ``x = 2 z − 1``.

    Args:
        n_modes: Number of modes ``K`` (default ``16``); index ``0`` is
            the constant ``P₀``.
        epsilon: Floor applied to ``λ₀``.

    Raises:
        IGLConfigError: For ``n_modes < 1`` or ``epsilon <= 0``.
    """

    n_modes: int
    null_indices: tuple[int, ...]
    domain: tuple[float, float]

    def __init__(self, n_modes: int = 16, *, epsilon: float = 1e-4) -> None:
        super().__init__()
        if n_modes < 1:
            raise IGLConfigError(f"n_modes must be >= 1, got {n_modes}")
        if epsilon <= 0:
            raise IGLConfigError(f"epsilon must be > 0, got {epsilon}")
        self.n_modes = n_modes
        self.null_indices = (0,)
        self.domain = (0.0, 1.0)
        indices = torch.arange(0, n_modes, dtype=torch.float32)
        eigenvalues = indices * (indices + 1.0)
        eigenvalues[0] = epsilon
        self.register_buffer("eigenvalues", eigenvalues)

    def evaluate(self, z: torch.Tensor, /) -> torch.Tensor:
        x = 2.0 * z - 1.0
        return _legendre_recurrence(x, self.n_modes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.evaluate(z)


__all__ = ["LegendreBasis"]
