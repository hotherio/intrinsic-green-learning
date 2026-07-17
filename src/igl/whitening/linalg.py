"""Symmetric-PSD square roots for metric whitening."""

from typing import cast

import torch

from igl.exceptions import IGLConfigError

__all__ = ["psd_sqrt_inv"]

_MATRIX_NDIM = 2


def psd_sqrt_inv(g: torch.Tensor, *, clamp: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the square root and inverse square root of a symmetric PSD matrix.

    A single eigendecomposition of the symmetrised input is used for both
    outputs, with eigenvalues clamped to ``clamp * lambda_max`` so the
    inverse root is the exact inverse of the root on the retained spectrum.
    Sampled metrics (e.g. :func:`igl.whitening.fisher_pullback`) leave the
    near-null spectrum ill-conditioned; the clamp keeps both factors finite.

    Args:
        g: Symmetric PSD matrix ``[C, C]``.
        clamp: Eigenvalue floor relative to the largest eigenvalue.

    Returns:
        A tuple ``(a, a_inv)`` with ``a = g^{1/2}`` and ``a_inv = g^{-1/2}``,
        both symmetric ``[C, C]``.

    Raises:
        IGLConfigError: If ``g`` is not a square 2-D tensor or ``clamp`` is
            not in ``(0, 1)``.
    """
    if g.dim() != _MATRIX_NDIM or g.shape[0] != g.shape[1]:
        raise IGLConfigError(f"g must be a square matrix, got shape {tuple(g.shape)}")
    if not 0.0 < clamp < 1.0:
        raise IGLConfigError(f"clamp must be in (0, 1), got {clamp}")
    sym = 0.5 * (g + g.T)
    # torch.linalg.eigh has partial stubs; recover the known types.
    eigvals, eigvecs = cast(tuple[torch.Tensor, torch.Tensor], torch.linalg.eigh(sym))  # pyright: ignore[reportUnknownMemberType]
    floor = clamp * eigvals.max().clamp_min(torch.finfo(sym.dtype).tiny)
    lam = eigvals.clamp_min(floor)
    a = eigvecs @ torch.diag(lam.sqrt()) @ eigvecs.T
    a_inv = eigvecs @ torch.diag(lam.rsqrt()) @ eigvecs.T
    return a, a_inv
