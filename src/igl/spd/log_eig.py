"""Log-Eig vectorizer: SPD → log-Euclidean tangent-space vector.

The log-Eig embedding maps an SPD matrix ``C`` to ``log(C) = U log(Λ) U^T``
(matrix log via eigendecomposition), then flattens the upper triangle into a
vector with off-diagonal entries scaled by √2. The factor of √2 ensures the
L2-norm in vector space equals the Frobenius norm in matrix space.

This is the SPDNet "LogEig" boundary layer: it sends the SPD manifold into a
Euclidean tangent space at the identity so that downstream Euclidean methods
(IGL, linear models, etc.) can operate without manifold-aware loss functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray


class LogEigVectorizer(BaseEstimator, TransformerMixin):
    """Vectorize SPD matrices via the log-Euclidean embedding.

    ``fit(X)`` records the matrix size from ``X.shape[1]`` and caches the
    upper-triangle indices + √2 scaling. ``transform(X)`` returns one row per
    input SPD matrix.

    Args:
        eps: Eigenvalue clamp for the matrix-log step (default ``1e-8``).
    """

    eps: float

    def __init__(self, *, eps: float = 1e-8) -> None:
        self.eps = eps

    def fit(self, x: NDArray[np.floating], y: object = None) -> LogEigVectorizer:  # noqa: ARG002
        """Cache the matrix size and packing indices from ``X``."""
        n = x.shape[1]
        self.n_features_in_: int = n
        self.triu_idx_: tuple[NDArray[np.int_], NDArray[np.int_]] = np.triu_indices(n)
        diag_mask = self.triu_idx_[0] == self.triu_idx_[1]
        self.scale_: NDArray[np.floating] = np.where(diag_mask, 1.0, np.sqrt(2.0))
        return self

    def transform(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply ``S → vec(log(S))`` with √2 off-diagonal scaling.

        Args:
            x: ``[B, n, n]`` SPD matrices.

        Returns:
            ``[B, n * (n + 1) / 2]`` log-Eig vectors.
        """
        eigvals, eigvecs = np.linalg.eigh(x)
        eigvals = np.log(np.clip(eigvals, self.eps, None))
        log_x = (eigvecs * eigvals[:, np.newaxis, :]) @ eigvecs.swapaxes(1, 2)
        rows, cols = self.triu_idx_
        flat = log_x[:, rows, cols]
        return np.asarray(flat * self.scale_, dtype=np.float64)


__all__ = ["LogEigVectorizer"]
