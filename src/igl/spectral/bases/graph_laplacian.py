# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportAttributeAccessIssue=false
"""Graph Laplacian basis — eigendecomposition of a user-supplied adjacency matrix.

For natively graph-valued data (functional-connectivity matrices,
brain-region atlases, social networks, mesh signals), the user
supplies an adjacency matrix ``W``. This module:

1. Forms the graph Laplacian according to ``normalization``:

   - ``"symmetric"``: ``L = I − D^{-1/2} W D^{-1/2}``.
   - ``"rw"``: ``L = I − D^{-1} W`` (random-walk normalization).
   - ``"unnormalized"``: ``L = D − W``.

2. Solves ``L φ = λ φ`` for the ``n_modes`` smallest eigenvalues via
   :func:`scipy.sparse.linalg.eigsh`.

3. Caches the eigenvectors **per graph node**. :meth:`evaluate` takes
   a node-index tensor (the only basis that doesn't take a real-valued
   latent input — graph data has no other natural input).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import scipy.sparse  # noqa: ICN001
import scipy.sparse.linalg
import torch
from torch import nn

from igl.exceptions import IGLConfigError
from igl.types import GraphLaplacianNorm, GraphLaplacianNormLike

_DEGREE_FLOOR: float = 1e-12
"""Lower bound on node degree before inverting; protects 1/d and 1/sqrt(d) at isolated nodes."""


def _build_laplacian(
    adjacency: scipy.sparse.spmatrix,
    norm: GraphLaplacianNorm,
) -> scipy.sparse.spmatrix:
    n = adjacency.shape[0]
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    if norm is GraphLaplacianNorm.UNNORMALIZED:
        return scipy.sparse.diags(degrees) - adjacency
    if norm is GraphLaplacianNorm.RANDOM_WALK:
        d_inv = scipy.sparse.diags(1.0 / np.clip(degrees, _DEGREE_FLOOR, None))
        return scipy.sparse.eye(n) - d_inv @ adjacency
    # Symmetric (default).
    d_inv_sqrt = scipy.sparse.diags(1.0 / np.sqrt(np.clip(degrees, _DEGREE_FLOOR, None)))
    return scipy.sparse.eye(n) - d_inv_sqrt @ adjacency @ d_inv_sqrt


class GraphLaplacianBasis(nn.Module):
    """Graph-Laplacian spectral basis.

    Args:
        adjacency: ``[n, n]`` adjacency matrix. Accepts a dense PyTorch
            tensor, a dense numpy array, or any scipy sparse matrix.
        n_modes: Number of eigenmodes ``K``.
        normalization: One of :class:`GraphLaplacianNorm`. Default
            ``SYMMETRIC``.
        epsilon: Floor applied to ``λ₀`` (which is always 0 for a
            connected graph in the symmetric / unnormalized cases).

    Raises:
        IGLConfigError: For invalid shapes or hyperparameters.
    """

    n_modes: int
    null_indices: tuple[int, ...]
    domain: tuple[float, float]
    n_nodes: int

    def __init__(
        self,
        adjacency: torch.Tensor | np.ndarray | scipy.sparse.spmatrix,
        *,
        n_modes: int = 16,
        normalization: GraphLaplacianNormLike = GraphLaplacianNorm.SYMMETRIC,
        epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        if n_modes < 1:
            raise IGLConfigError(f"n_modes must be >= 1, got {n_modes}")
        if epsilon <= 0:
            raise IGLConfigError(f"epsilon must be > 0, got {epsilon}")
        norm_enum = GraphLaplacianNorm(normalization)

        adj_sparse = _to_sparse(adjacency)
        if adj_sparse.shape[0] != adj_sparse.shape[1]:
            raise IGLConfigError(
                f"adjacency must be square; got {adj_sparse.shape}",
            )
        n = adj_sparse.shape[0]
        if n_modes > n - 1:
            raise IGLConfigError(
                f"n_modes ({n_modes}) must be < n_nodes ({n}) for sparse eigsh",
            )

        lap = _build_laplacian(adj_sparse, norm_enum)
        eigvals, eigvecs = cast(
            tuple[np.ndarray, np.ndarray],
            scipy.sparse.linalg.eigsh(lap, k=n_modes, which="SM"),  # pyright: ignore[reportUnknownMemberType]
        )
        order = np.argsort(eigvals)
        eigvals = np.clip(eigvals[order], epsilon, None)
        eigvecs = eigvecs[:, order]

        self.n_modes = n_modes
        self.null_indices = (0,)
        self.domain = (0.0, float(n - 1))
        self.n_nodes = n
        self.register_buffer("eigenvalues", torch.as_tensor(eigvals, dtype=torch.float32))
        self.register_buffer("_eigenvectors", torch.as_tensor(eigvecs, dtype=torch.float32))

    def evaluate(self, z: torch.Tensor, /) -> torch.Tensor:
        """Return eigenfunction values at the given node indices.

        Args:
            z: ``[N]`` long-tensor of node indices, or ``[N, 1]`` with
                integer dtype.

        Returns:
            ``[N, n_modes]`` eigenfunction values.
        """
        idx = z.long().view(-1)
        eigvecs = cast(torch.Tensor, self._eigenvectors)
        return eigvecs[idx]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.evaluate(z)


def _to_sparse(
    adjacency: torch.Tensor | np.ndarray | scipy.sparse.spmatrix,
) -> scipy.sparse.spmatrix:
    if isinstance(adjacency, torch.Tensor):
        return scipy.sparse.csr_matrix(adjacency.detach().cpu().numpy())
    if isinstance(adjacency, np.ndarray):
        return scipy.sparse.csr_matrix(adjacency)
    return adjacency.tocsr()


__all__ = ["GraphLaplacianBasis"]
