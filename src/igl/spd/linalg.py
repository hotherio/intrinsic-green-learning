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
from typing import Literal, cast

import torch

from igl.exceptions import IGLConfigError

MatrixMethod = Literal["eigh", "iterative"]
"""How the SPD matrix functions are computed.

``"eigh"`` (default) diagonalises; exact to fp32 rounding but the CUDA
eigensolver synchronises with the host. ``"iterative"`` uses only matrix
products, solves and inverses (``torch.linalg.matrix_exp``, inverse scaling and
squaring with Denman–Beavers square roots for the logarithm, Denman–Beavers for
the inverse square root): no host synchronisation, and on EEG-like spectra
spanning six orders of magnitude the fp32 error (2e-4..2e-3 measured) is in the
same band as fp32 ``eigh``'s own (5e-4..8e-3): three Denman–Beavers roots
(fewer roots keep the ``2^k`` error amplification small in fp32) and a
twelve-term ``atanh`` series cover condition numbers up to about ``1e7``. It
costs ~80 batched small inverses, so it pays only where a synchronisation
stalls a pipeline.
"""
_DB_ITERS = 12
_ISS_ROOTS = 3
_ATANH_TERMS = 12


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


def _denman_beavers(a: torch.Tensor, iters: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Coupled Denman–Beavers iteration: returns ``(A^{1/2}, A^{-1/2})`` without an eigensolve."""
    y = a
    z = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device).expand_as(a).clone()
    for _ in range(iters):
        y_inv = cast(torch.Tensor, torch.linalg.inv_ex(y)[0])  # pyright: ignore[reportUnknownMemberType]
        z_inv = cast(torch.Tensor, torch.linalg.inv_ex(z)[0])  # pyright: ignore[reportUnknownMemberType]
        y, z = 0.5 * (y + z_inv), 0.5 * (z + y_inv)
    return y, z


def _trace_scale(c: torch.Tensor) -> torch.Tensor:
    """Per-matrix ``trace / d`` as ``[B, 1, 1]``: the scaling that centres the spectrum around 1."""
    return c.diagonal(dim1=-2, dim2=-1).mean(dim=-1)[:, None, None]


def matrix_exp_sym(s: torch.Tensor, *, method: MatrixMethod = "eigh") -> torch.Tensor:
    """Batched matrix exponential of symmetric matrices.

    For a real symmetric ``S = U Λ U^T``, ``exp(S) = U diag(exp Λ) U^T``; the
    iterative method is ``torch.linalg.matrix_exp`` (scaling and squaring with
    Padé approximants, matrix products only).

    Args:
        s: ``[B, d, d]`` symmetric matrices.
        method: See :data:`MatrixMethod`.

    Returns:
        ``[B, d, d]`` symmetric positive-definite matrices.
    """
    if method == "iterative":
        out = cast(torch.Tensor, torch.linalg.matrix_exp(s))  # pyright: ignore[reportUnknownMemberType]
        return 0.5 * (out + out.transpose(-1, -2))
    eigvals, eigvecs = _eigh(s)
    return eigvecs @ torch.diag_embed(torch.exp(eigvals)) @ eigvecs.transpose(-1, -2)


def matrix_log_sym(c: torch.Tensor, *, eps: float = 1e-8, method: MatrixMethod = "eigh") -> torch.Tensor:
    """Batched matrix logarithm of SPD matrices.

    The iterative method scales by the mean eigenvalue, takes ``2^k``-th roots
    by Denman–Beavers until the spectrum sits near 1, evaluates
    ``log(I + X) = 2 atanh(X (2I + X)^{-1})`` by its series, and undoes the
    scaling (inverse scaling and squaring).

    Args:
        c: ``[B, d, d]`` SPD matrices.
        eps: Eigenvalue clamp for numerical safety (``eigh`` method only).
        method: See :data:`MatrixMethod`.

    Returns:
        ``[B, d, d]`` symmetric matrices (log of input).
    """
    if method == "iterative":
        eye = torch.eye(c.shape[-1], dtype=c.dtype, device=c.device).expand_as(c)
        scale = _trace_scale(c)
        a = c / scale
        for _ in range(_ISS_ROOTS):
            a, _ = _denman_beavers(a, _DB_ITERS)
        x = a - eye
        t = cast(torch.Tensor, torch.linalg.solve_ex(2 * eye + x, x)[0])  # pyright: ignore[reportUnknownMemberType]
        t2 = t @ t
        acc = t.clone()
        power = t
        for m in range(1, _ATANH_TERMS):
            power = power @ t2
            acc = acc + power / (2 * m + 1)
        out = (2**_ISS_ROOTS) * 2 * acc + torch.log(scale) * eye
        return 0.5 * (out + out.transpose(-1, -2))
    eigvals, eigvecs = _eigh(c)
    log_eigvals = torch.log(eigvals.clamp(min=eps))
    return eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-1, -2)


def matrix_pow_sym(c: torch.Tensor, p: float, *, eps: float = 1e-8, method: MatrixMethod = "eigh") -> torch.Tensor:
    """Batched real-power of SPD matrices.

    For SPD ``C`` and real ``p``, ``C^p = U diag(Λ^p) U^T``. The iterative
    method supports ``p = -0.5`` and ``p = 0.5`` (Denman–Beavers on the
    trace-scaled matrix) and raises for other exponents.

    Args:
        c: ``[B, d, d]`` SPD matrices.
        p: Real exponent (e.g. ``-0.5`` for matrix-inverse-square-root).
        eps: Eigenvalue clamp (``eigh`` method only).
        method: See :data:`MatrixMethod`.

    Returns:
        ``[B, d, d]`` SPD matrices.

    Raises:
        IGLConfigError: For the iterative method with an exponent other than ``±0.5``.
    """
    if method == "iterative":
        if p not in (0.5, -0.5):
            raise IGLConfigError(f"the iterative method supports p = ±0.5, got {p}")
        scale = _trace_scale(c)
        root, inv_root = _denman_beavers(c / scale, 2 * _DB_ITERS)
        out = root * torch.sqrt(scale) if p > 0 else inv_root / torch.sqrt(scale)
        return 0.5 * (out + out.transpose(-1, -2))
    eigvals, eigvecs = _eigh(c)
    pow_eigvals = eigvals.clamp(min=eps) ** p
    return eigvecs @ torch.diag_embed(pow_eigvals) @ eigvecs.transpose(-1, -2)


__all__ = [
    "MatrixMethod",
    "matrix_exp_sym",
    "matrix_log_sym",
    "matrix_pow_sym",
    "unpack_sym_vec",
]
