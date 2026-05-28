"""Fourier cosine basis — Laplacian on ``[0, 1]`` with Neumann BCs.

Eigenfunctions:
    ``φ₀ = 1`` (the constant — the null mode),
    ``φₙ(z) = √2 · cos(n π z)`` for ``n ≥ 1``.
Eigenvalues:   ``λₙ = (n π)²`` with ``λ₀ = 0`` (regularised to ``ε``).

This basis is the spectral form of the Neumann Laplacian, the
operator whose null space contains the constants. Use it whenever the
target function has a non-zero average that the kernel must reach.
"""

import math

import torch
from torch import nn

from igl.exceptions import IGLConfigError


class FourierCosineBasis(nn.Module):
    """1-D Fourier cosine spectral basis (Neumann BCs).

    Args:
        n_modes: Number of modes ``K`` (default ``16``); index ``0`` is
            the constant.
        epsilon: Floor applied to the zero eigenvalue so the kernel's
            ``1/λ`` weighting stays finite. Default ``1e-4``.

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
        eigenvalues = (indices * math.pi) ** 2
        eigenvalues[0] = epsilon  # regularise λ₀
        self.register_buffer("_indices", indices)
        self.register_buffer("eigenvalues", eigenvalues)

    def evaluate(self, z: torch.Tensor, /) -> torch.Tensor:
        z_exp = z.unsqueeze(-1)
        indices_buf: torch.Tensor = self._indices  # pyright: ignore[reportAssignmentType]
        indices = indices_buf.view(*([1] * z.dim()), -1)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        zeros = torch.zeros_like(indices, dtype=torch.bool) | (indices == 0)
        cos_part = math.sqrt(2.0) * torch.cos(math.pi * indices * z_exp)
        ones_part = torch.ones_like(cos_part)
        return torch.where(zeros, ones_part, cos_part)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.evaluate(z)


__all__ = ["FourierCosineBasis"]
