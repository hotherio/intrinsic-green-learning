"""Batched linear algebra on symmetric positive-definite (SPD) matrices.

All functions take and return PyTorch tensors with a batch leading dimension
``[B, d, d]``. Eigendecomposition is used everywhere — fast enough for the
``d ≤ 64`` matrices that arise in EEG/clinical pipelines, numerically stable,
and differentiable via ``torch.linalg.eigh``.

The :func:`unpack_sym_vec` function inverts :class:`igl.spd.LogEigVectorizer`'s
packing scheme: upper-triangle (with diagonal) flattened, off-diagonal entries
scaled by √2. Use it when you need to recover a symmetric matrix from the
vectorized log-Eig representation.
"""

import math
from typing import cast

import torch

from igl.exceptions import IGLConfigError


def _eigh(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``torch.linalg.eigh`` with proper static typing.

    The torch stubs return Unknown for the named-tuple decomposition; we cast
    once here so downstream code stays clean.
    """
    eigvals, eigvecs = cast(
        tuple[torch.Tensor, torch.Tensor],
        torch.linalg.eigh(m),  # pyright: ignore[reportUnknownMemberType]
    )
    return eigvals, eigvecs


def _triu_idx(d: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return upper-triangle indices ``(rows, cols)`` for a ``[d, d]`` matrix."""
    rows, cols = torch.triu_indices(d, d, offset=0, device=device)
    return rows, cols


def unpack_sym_vec(vec: torch.Tensor, d: int) -> torch.Tensor:
    """Invert :class:`igl.spd.LogEigVectorizer`'s upper-triangle packing.

    Args:
        vec: ``[B, D]`` where ``D = d * (d + 1) / 2`` — the upper triangle of a
            symmetric matrix in row-major order, with off-diagonal entries
            scaled by √2 (so the Frobenius norm in vector form matches the
            matrix Frobenius norm).
        d: Side length of the resulting symmetric matrices.

    Returns:
        ``[B, d, d]`` symmetric matrices.

    Raises:
        IGLConfigError: If ``vec.shape[-1] != d * (d + 1) / 2``.
    """
    expected = d * (d + 1) // 2
    if vec.shape[-1] != expected:
        raise IGLConfigError(
            f"vec.shape[-1] ({vec.shape[-1]}) does not match d*(d+1)/2 ({expected}) for d={d}",
        )
    batch = vec.shape[0]
    rows, cols = _triu_idx(d, vec.device)
    on_diag = (rows == cols).to(vec.dtype)
    inv_scale = on_diag + (1.0 - on_diag) / math.sqrt(2.0)
    vec_unscaled = vec * inv_scale

    sym = torch.zeros(batch, d, d, dtype=vec.dtype, device=vec.device)
    sym[:, rows, cols] = vec_unscaled
    off_diag_mask = rows != cols
    sym[:, cols[off_diag_mask], rows[off_diag_mask]] = vec_unscaled[:, off_diag_mask]
    return sym


def matrix_exp_sym(s: torch.Tensor) -> torch.Tensor:
    """Batched matrix exponential of symmetric matrices via eigendecomposition.

    For a real symmetric ``S = U Λ U^T``, ``exp(S) = U diag(exp Λ) U^T``.

    Args:
        s: ``[B, d, d]`` symmetric matrices.

    Returns:
        ``[B, d, d]`` symmetric positive-definite matrices.
    """
    eigvals, eigvecs = _eigh(s)
    return eigvecs @ torch.diag_embed(torch.exp(eigvals)) @ eigvecs.transpose(-1, -2)


def matrix_log_sym(c: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Batched matrix logarithm of SPD matrices via eigendecomposition.

    Args:
        c: ``[B, d, d]`` SPD matrices.
        eps: Eigenvalue clamp for numerical safety (default ``1e-8``).

    Returns:
        ``[B, d, d]`` symmetric matrices (log of input).
    """
    eigvals, eigvecs = _eigh(c)
    log_eigvals = torch.log(eigvals.clamp(min=eps))
    return eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-1, -2)


def matrix_pow_sym(c: torch.Tensor, p: float, *, eps: float = 1e-8) -> torch.Tensor:
    """Batched real-power of SPD matrices via eigendecomposition.

    For SPD ``C`` and real ``p``, ``C^p = U diag(Λ^p) U^T``.

    Args:
        c: ``[B, d, d]`` SPD matrices.
        p: Real exponent (e.g. ``-0.5`` for matrix-inverse-square-root).
        eps: Eigenvalue clamp.

    Returns:
        ``[B, d, d]`` SPD matrices.
    """
    eigvals, eigvecs = _eigh(c)
    pow_eigvals = eigvals.clamp(min=eps) ** p
    return eigvecs @ torch.diag_embed(pow_eigvals) @ eigvecs.transpose(-1, -2)


__all__ = [
    "matrix_exp_sym",
    "matrix_log_sym",
    "matrix_pow_sym",
    "unpack_sym_vec",
]
