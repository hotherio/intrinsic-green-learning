"""Fourier sine basis — Laplacian on ``[0, 1]`` with Dirichlet BCs.

Eigenfunctions: ``φₙ(z) = √2 · sin(n π z)`` for ``n = 1, 2, …``.
Eigenvalues:   ``λₙ = (n π)²``.

No null space (the smallest eigenvalue is ``π² > 0``).
"""

import math

import torch
from torch import nn

from igl.exceptions import IGLConfigError


class FourierSineBasis(nn.Module):
    """1-D Fourier sine spectral basis (Dirichlet BCs).

    Args:
        n_modes: Number of modes ``K`` (default ``16``).

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
        self.domain = (0.0, 1.0)
        indices = torch.arange(1, n_modes + 1, dtype=torch.float32)
        self.register_buffer("_indices", indices)
        self.register_buffer("eigenvalues", (indices * math.pi) ** 2)

    def evaluate(self, z: torch.Tensor, /) -> torch.Tensor:
        # ``z``: ``[...]``; returns ``[..., n_modes]``.
        z_exp = z.unsqueeze(-1)
        indices: torch.Tensor = self._indices  # pyright: ignore[reportAssignmentType]
        broadcast = indices.view(*([1] * z.dim()), -1)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        return math.sqrt(2.0) * torch.sin(math.pi * broadcast * z_exp)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.evaluate(z)


__all__ = ["FourierSineBasis"]
