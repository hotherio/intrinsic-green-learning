# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportAttributeAccessIssue=false
"""Smallest eigenpairs of a sparse symmetric positive-semidefinite Laplacian.

Shared by the data-driven bases. ``scipy.sparse.linalg.eigsh(which="SM")``
without shift-invert is the textbook call, but on realistic latents (a
noisy circle, a torus projection) ARPACK fails to converge in its 5000
iterations; shift-invert around a small negative ``sigma`` (``L ⪰ 0`` so
``L − σI`` is positive definite) converges in milliseconds. Small graphs
skip ARPACK altogether and use a dense ``eigh``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse  # noqa: ICN001
import scipy.sparse.linalg
from scipy.sparse.linalg import ArpackNoConvergence

_DENSE_MAX_NODES = 2000
_SHIFT = -1e-3

__all__ = ["smallest_eigenpairs"]


def smallest_eigenpairs(lap: scipy.sparse.spmatrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``k`` smallest eigenpairs of ``lap``, eigenvalues ascending.

    Args:
        lap: Sparse symmetric PSD matrix ``[n, n]``.
        k: Number of eigenpairs, ``1 <= k < n``.

    Returns:
        ``(eigenvalues [k], eigenvectors [n, k])`` sorted by ascending eigenvalue.
    """
    n = lap.shape[0]
    if n <= _DENSE_MAX_NODES:
        vals, vecs = np.linalg.eigh(np.asarray(lap.todense()))
        return vals[:k], vecs[:, :k]
    try:
        vals, vecs = scipy.sparse.linalg.eigsh(lap, k=k, sigma=_SHIFT, which="LM")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    except (ArpackNoConvergence, RuntimeError):  # pragma: no cover  # ARPACK fallback, rarely reached
        vals, vecs = scipy.sparse.linalg.eigsh(lap, k=k, which="SM")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    order = np.argsort(vals)  # pyright: ignore[reportUnknownArgumentType]
    return np.asarray(vals)[order], np.asarray(vecs)[:, order]
