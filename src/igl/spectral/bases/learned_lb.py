"""Learned Laplace–Beltrami basis on the encoder's latent manifold.

For data on an unknown manifold, the operator ``L`` to invert is the
Laplace–Beltrami operator of the *learned* Riemannian metric
``g = J⊤ J`` where ``J = ∂Ψ / ∂x`` is the encoder Jacobian. Its
eigendecomposition is not closed-form — it is **estimated numerically**
from a batch of encoded latents:

1. Build a ``k``-NN graph on the latents ``{z_i}``.
2. Weight edges via a Gaussian kernel ``W_{ij} = exp(-‖z_i − z_j‖² /
   (2 σ²))`` with ``σ`` auto-tuned to the median nearest-neighbour
   distance.
3. Form the symmetric normalised Laplacian
   ``L = I − D^{-1/2} W D^{-1/2}``.
4. Solve ``L φ = λ φ`` for the ``n_modes`` smallest eigenvalues via
   :func:`scipy.sparse.linalg.eigsh`.
5. Cache the refresh points ``z_i``, the eigenfunctions ``φ_n(z_i)``,
   and the eigenvalues ``λ_n``.

At query time, :meth:`evaluate` uses **Nyström extension** to lift the
eigenfunctions to new points ``z``::

    φ_n(z) ≈ (1 / λ_n) Σ_i K(z, z_i) · φ_n(z_i)

where ``K`` is the same Gaussian kernel used to build the graph.

The constant function ``φ_0`` (with ``λ_0 = 0``) is the null mode of
the LB operator; :attr:`null_indices` is ``(0,)`` and the
:class:`SpectralKernel` routes that column into the null-space slot.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportAttributeAccessIssue=false

from __future__ import annotations

from typing import cast

import numpy as np
import scipy.sparse  # noqa: ICN001
import scipy.sparse.linalg
import torch
from torch import nn

from igl.exceptions import IGLConfigError

_MIN_K_NN = 1


class LearnedLaplacianBasis(nn.Module):
    """Numerically-estimated Laplace–Beltrami spectrum on a learned manifold.

    Args:
        n_modes: Number of eigenmodes ``K`` to retain (smallest
            eigenvalues).
        k_nn: Number of nearest neighbours used to build the graph.
        epsilon: Floor applied to the zero eigenvalue (``λ₀``) when
            evaluating the Nyström extension.

    Raises:
        IGLConfigError: For invalid hyperparameters.

    Attributes:
        is_refreshed: ``True`` once :meth:`refresh` has run. Calling
            :meth:`evaluate` before refresh raises.
    """

    n_modes: int
    null_indices: tuple[int, ...]
    domain: tuple[float, float]
    k_nn: int
    epsilon: float
    is_refreshed: bool

    def __init__(
        self,
        n_modes: int = 16,
        *,
        k_nn: int = 10,
        epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        if n_modes < 1:
            raise IGLConfigError(f"n_modes must be >= 1, got {n_modes}")
        if k_nn < _MIN_K_NN:
            raise IGLConfigError(f"k_nn must be >= {_MIN_K_NN}, got {k_nn}")
        if epsilon <= 0:
            raise IGLConfigError(f"epsilon must be > 0, got {epsilon}")
        self.n_modes = n_modes
        self.null_indices = (0,)
        self.domain = (-float("inf"), float("inf"))
        self.k_nn = k_nn
        self.epsilon = epsilon
        self.is_refreshed = False
        # Placeholders sized at refresh time.
        self.register_buffer("eigenvalues", torch.zeros(n_modes))
        self.register_buffer("_refresh_points", torch.zeros(0))
        self.register_buffer("_eigenvectors", torch.zeros(0))
        self.register_buffer("_sigma_sq", torch.tensor(1.0))

    @torch.no_grad()
    def refresh(self, z: torch.Tensor, /) -> None:
        """Recompute the LB spectrum from a fresh set of encoded latents.

        Args:
            z: ``[M, d]`` encoded latents (``M`` should be ≥ a few hundred
                for the spectrum to be informative).
        """
        if z.dim() != 2 or z.shape[0] <= self.k_nn:  # noqa: PLR2004
            raise IGLConfigError(
                f"refresh expects z of shape [M, d] with M > k_nn; got {tuple(z.shape)}",
            )
        device = z.device
        z_np = z.detach().cpu().numpy().astype(np.float64)
        n = z_np.shape[0]

        # k-NN via cKDTree (already a scipy dep).
        from scipy.spatial import cKDTree  # noqa: PLC0415  # pyright: ignore[reportUnknownVariableType]

        tree = cKDTree(z_np)
        # +1 because the closest neighbour is the point itself.
        distances, indices = cast(
            tuple[np.ndarray, np.ndarray],
            tree.query(z_np, k=self.k_nn + 1),  # pyright: ignore[reportUnknownMemberType]
        )
        # Drop the self-edge.
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        sigma_sq = float(np.median(distances[:, 0]) ** 2 + 1e-12)

        rows = np.repeat(np.arange(n), self.k_nn)
        cols = indices.reshape(-1)
        weights = np.exp(-(distances**2) / (2.0 * sigma_sq)).reshape(-1)
        w = scipy.sparse.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
        w = 0.5 * (w + w.T)
        d = np.asarray(w.sum(axis=1)).reshape(-1)
        d_inv_sqrt = scipy.sparse.diags(1.0 / np.sqrt(d + 1e-12))
        lap = scipy.sparse.eye(n) - d_inv_sqrt @ w @ d_inv_sqrt

        # eigsh: smallest eigenvalues via shift-invert; for n_modes < n_total - 1.
        k = min(self.n_modes, n - 2)
        eigvals, eigvecs = cast(
            tuple[np.ndarray, np.ndarray],
            scipy.sparse.linalg.eigsh(lap, k=k, which="SM"),  # pyright: ignore[reportUnknownMemberType]
        )
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Pad with the largest available eigenvalue if eigsh returned fewer modes than n_modes
        # (only relevant when n_modes is very close to n).
        if eigvals.shape[0] < self.n_modes:  # pragma: no cover  # tested via property; numerically rare
            pad = self.n_modes - eigvals.shape[0]
            eigvals = np.concatenate([eigvals, np.full(pad, eigvals[-1])])
            last_col = eigvecs[:, -1:]
            eigvecs = np.concatenate([eigvecs, np.tile(last_col, (1, pad))], axis=1)

        # Floor the null eigenvalue for the Nyström denominator.
        eigvals_t = torch.as_tensor(eigvals, dtype=torch.float32, device=device)
        eigvals_t = torch.clamp(eigvals_t, min=self.epsilon)

        self.eigenvalues = eigvals_t  # type: ignore[assignment]
        self._refresh_points = z.detach().to(device)  # type: ignore[assignment]
        self._eigenvectors = torch.as_tensor(eigvecs, dtype=torch.float32, device=device)  # type: ignore[assignment]
        self._sigma_sq = torch.tensor(sigma_sq, dtype=torch.float32, device=device)  # type: ignore[assignment]
        self.is_refreshed = True

    def evaluate(self, z: torch.Tensor, /) -> torch.Tensor:
        """Nyström-extend the eigenfunctions to ``z``.

        Args:
            z: ``[N, d]`` query latents.

        Returns:
            ``[N, n_modes]`` eigenfunction values.

        Raises:
            IGLConfigError: If :meth:`refresh` has not been called.
        """
        if not self.is_refreshed:
            raise IGLConfigError(
                "LearnedLaplacianBasis: call .refresh(z) before .evaluate(z).",
            )
        refresh: torch.Tensor = self._refresh_points  # pyright: ignore[reportAssignmentType]
        eigvecs: torch.Tensor = self._eigenvectors  # pyright: ignore[reportAssignmentType]
        eigvals: torch.Tensor = self.eigenvalues  # pyright: ignore[reportAssignmentType]
        sigma_sq: torch.Tensor = self._sigma_sq  # pyright: ignore[reportAssignmentType]

        # Pairwise distances [N, M] then Gaussian kernel.
        d2 = torch.cdist(z, refresh).pow(2)
        k_matrix = torch.exp(-d2 / (2.0 * sigma_sq))
        # Nyström: φ(z) ≈ (1/λ) K · Φ
        return (k_matrix @ eigvecs) / eigvals.unsqueeze(0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.evaluate(z)


__all__ = ["LearnedLaplacianBasis"]
