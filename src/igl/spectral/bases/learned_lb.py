"""Learned Laplace–Beltrami basis on the encoder's latent manifold.

For data on an unknown manifold, the operator ``L`` to invert is the
Laplace–Beltrami operator of the *learned* Riemannian metric
``g = J⊤ J`` where ``J = ∂Ψ / ∂x`` is the encoder Jacobian. Its
eigendecomposition is not closed-form — it is **estimated numerically**
from a batch of encoded latents:

1. Build a ``k``-NN graph on the latents ``{z_i}``.
2. Weight edges via a Gaussian kernel ``W_{ij} = exp(-‖z_i − z_j‖² /
   (2 σ²))`` with ``σ`` auto-tuned to the median nearest-neighbour
   distance, then symmetrise.
3. Form the symmetric normalised Laplacian
   ``L = I − D^{-1/2} W D^{-1/2}``.
4. Solve ``L φ = λ φ`` for the ``n_modes`` smallest eigenvalues
   (dense ``eigh`` for small graphs, shift-invert ``eigsh`` otherwise).
5. Cache the refresh points ``z_i``, their graph degrees ``d_i``, the
   eigenvectors ``φ_n(z_i)`` and the eigenvalues ``λ_n``.

At query time, :meth:`evaluate` uses the **Nyström extension** of the
eigenvectors of ``S = D^{-1/2} W D^{-1/2}`` (eigenvalue ``μ_n = 1 − λ_n``)::

    φ_n(z) ≈ (1 / μ_n) Σ_i  w(z, z_i) / sqrt(d(z) d_i) · φ_n(z_i),
    d(z) = Σ_i w(z, z_i),

with ``w`` the graph's own symmetrised kNN affinity: half weight for a
one-sided edge (``z_i`` among the ``k_nn`` nearest of ``z`` or ``z`` inside
``z_i``'s ``k_nn`` radius), full weight for a mutual one. At a refresh point
this reproduces the stored eigenvector row exactly. Dividing by the
Laplacian eigenvalue ``λ_n`` instead (a previous version) scaled every mode
wrongly, by up to ``1/ε`` on the null mode.

This basis is **joint**: it is a function of the whole latent vector, so
:class:`igl.spectral.SpectralKernel` evaluates it once on ``[N, d]`` instead
of taking a per-dimension product. The constant function ``φ_0`` (with
``λ_0 = 0``) is the null mode of the LB operator; :attr:`null_indices` is
``(0,)`` and the kernel routes it to the null-space columns.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportAttributeAccessIssue=false

from __future__ import annotations

from typing import cast

import numpy as np
import scipy.sparse  # noqa: ICN001
import torch
from scipy.spatial import cKDTree
from torch import nn

from igl.exceptions import IGLConfigError
from igl.spectral.bases._eigsolve import smallest_eigenpairs

_MIN_K_NN = 1
_MEDIAN_DIST_FLOOR: float = 1e-12
"""Lower bound on the squared median nearest-neighbour distance — keeps the kNN
Gaussian similarity finite when all points happen to coincide."""
_DEGREE_FLOOR: float = 1e-12
"""Lower bound on graph-node degree before inverting; mirrors the same constant
in :mod:`igl.spectral.bases.graph_laplacian`."""
_MATRIX_NDIM = 2
_SELF_DIST_SQ: float = 1e-12
"""Squared distance under which a query is taken to be the refresh point itself (no self-edges)."""
_RADIUS_SLACK: float = 1e-6
"""Relative slack on a refresh point's k-NN radius so float rounding cannot drop its k-th neighbour."""


class LearnedLaplacianBasis(nn.Module):
    """Numerically-estimated Laplace–Beltrami spectrum on a learned manifold.

    Args:
        n_modes: Number of eigenmodes ``K`` to retain (smallest
            eigenvalues).
        k_nn: Number of nearest neighbours used to build the graph and to
            extend the eigenfunctions.
        epsilon: Floor applied to the zero eigenvalue (``λ₀``) in
            :attr:`eigenvalues`, and to ``μ_n = 1 − λ_n`` in the Nyström
            denominator.

    Raises:
        IGLConfigError: For invalid hyperparameters.

    Attributes:
        is_joint: Always ``True`` — the basis takes the whole latent vector.
        is_refreshed: ``True`` once :meth:`refresh` has run. Calling
            :meth:`evaluate` before refresh raises.
    """

    n_modes: int
    null_indices: tuple[int, ...]
    domain: tuple[float, float]
    is_joint: bool = True
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
        self.register_buffer("_affinity_eigs", torch.ones(n_modes))
        self.register_buffer("_refresh_points", torch.zeros(0))
        self.register_buffer("_eigenvectors", torch.zeros(0))
        self.register_buffer("_degrees", torch.zeros(0))
        self.register_buffer("_knn_radius_sq", torch.zeros(0))
        self.register_buffer("_sigma_sq", torch.tensor(1.0))

    @torch.no_grad()
    def refresh(self, z: torch.Tensor, /) -> None:
        """Recompute the LB spectrum from a fresh set of encoded latents.

        Args:
            z: ``[M, d]`` encoded latents (``M`` should be ≥ a few hundred
                for the spectrum to be informative).
        """
        if z.dim() != _MATRIX_NDIM or z.shape[0] <= self.k_nn:
            raise IGLConfigError(
                f"refresh expects z of shape [M, d] with M > k_nn; got {tuple(z.shape)}",
            )
        device = z.device
        if device.type == "cuda":  # pragma: no cover  # CUDA only
            self._refresh_on_device(z)
            return
        z_np = z.detach().cpu().numpy().astype(np.float64)
        n = z_np.shape[0]

        tree = cKDTree(z_np)
        # +1 because the closest neighbour is the point itself.
        distances, indices = cast(
            tuple[np.ndarray, np.ndarray],
            tree.query(z_np, k=self.k_nn + 1),  # pyright: ignore[reportUnknownMemberType]
        )
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        sigma_sq = float(np.median(distances[:, 0]) ** 2 + _MEDIAN_DIST_FLOOR)

        rows = np.repeat(np.arange(n), self.k_nn)
        cols = indices.reshape(-1)
        weights = np.exp(-(distances**2) / (2.0 * sigma_sq)).reshape(-1)
        w = scipy.sparse.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
        w = 0.5 * (w + w.T)
        d = np.asarray(w.sum(axis=1)).reshape(-1)
        d_inv_sqrt = scipy.sparse.diags(1.0 / np.sqrt(d + _DEGREE_FLOOR))
        lap = scipy.sparse.eye(n) - d_inv_sqrt @ w @ d_inv_sqrt

        k = min(self.n_modes, n - 2)
        eigvals, eigvecs = smallest_eigenpairs(lap, k)

        # Pad with the largest available eigenvalue if the solver returned fewer modes than n_modes
        # (only relevant when n_modes is very close to n).
        if eigvals.shape[0] < self.n_modes:  # pragma: no cover  # tested via property; numerically rare
            pad = self.n_modes - eigvals.shape[0]
            eigvals = np.concatenate([eigvals, np.full(pad, eigvals[-1])])
            last_col = eigvecs[:, -1:]
            eigvecs = np.concatenate([eigvecs, np.tile(last_col, (1, pad))], axis=1)

        lam_raw = torch.as_tensor(eigvals, dtype=torch.float32, device=device)
        self.eigenvalues = torch.clamp(lam_raw, min=self.epsilon)  # type: ignore[assignment]
        # Nyström divides by the affinity eigenvalue μ = 1 − λ; the null mode has μ₀ = 1.
        self._affinity_eigs = torch.clamp(1.0 - lam_raw, min=self.epsilon)  # type: ignore[assignment]
        self._refresh_points = z.detach().to(device)  # type: ignore[assignment]
        self._eigenvectors = torch.as_tensor(eigvecs, dtype=torch.float32, device=device)  # type: ignore[assignment]
        self._degrees = torch.as_tensor(d, dtype=torch.float32, device=device)  # type: ignore[assignment]
        # Squared distance to each point's k-th neighbour: a query lies in point j's
        # neighbourhood iff its squared distance to z_j is at most this radius.
        self._knn_radius_sq = torch.as_tensor(distances[:, -1] ** 2, dtype=torch.float32, device=device)  # type: ignore[assignment]
        self._sigma_sq = torch.tensor(sigma_sq, dtype=torch.float32, device=device)  # type: ignore[assignment]
        self.is_refreshed = True

    @torch.no_grad()
    def _refresh_on_device(self, z: torch.Tensor) -> None:
        """The refresh entirely in torch on the latents' device (CUDA).

        Same graph as the scipy path (kNN Gaussian affinities, symmetrised,
        symmetric normalised Laplacian), dense at these sizes, and a dense
        ``torch.linalg.eigh``: one host synchronisation per refresh (the
        eigensolver checks its status) instead of a round trip of the whole
        latent batch through scipy.
        """
        z = z.detach()
        n = z.shape[0]
        d2 = torch.cdist(z, z, compute_mode="donot_use_mm_for_euclid_dist").pow(2)
        d2.fill_diagonal_(float("inf"))
        d2_k, idx = d2.topk(self.k_nn, dim=-1, largest=False)  # [n, k]
        sigma_sq = d2_k[:, 0].median() + _MEDIAN_DIST_FLOOR
        weights = torch.exp(-d2_k / (2.0 * sigma_sq))
        w = torch.zeros(n, n, device=z.device, dtype=z.dtype).scatter_(1, idx, weights)
        w = 0.5 * (w + w.T)
        degrees = w.sum(dim=1)
        d_inv_sqrt = 1.0 / torch.sqrt(degrees + _DEGREE_FLOOR)
        lap = torch.eye(n, device=z.device, dtype=z.dtype) - d_inv_sqrt[:, None] * w * d_inv_sqrt[None, :]
        eigvals, eigvecs = torch.linalg.eigh(lap)
        k = min(self.n_modes, n - 2)
        lam_raw, vecs = eigvals[:k], eigvecs[:, :k]
        if k < self.n_modes:
            pad = self.n_modes - k
            lam_raw = torch.cat([lam_raw, lam_raw[-1:].expand(pad)])
            vecs = torch.cat([vecs, vecs[:, -1:].expand(-1, pad)], dim=1)
        self.eigenvalues = torch.clamp(lam_raw.float(), min=self.epsilon)  # type: ignore[assignment]
        self._affinity_eigs = torch.clamp(1.0 - lam_raw.float(), min=self.epsilon)  # type: ignore[assignment]
        self._refresh_points = z  # type: ignore[assignment]
        self._eigenvectors = vecs.float()  # type: ignore[assignment]
        self._degrees = degrees.float()  # type: ignore[assignment]
        self._knn_radius_sq = d2_k[:, -1].float()  # type: ignore[assignment]
        self._sigma_sq = sigma_sq.float()  # type: ignore[assignment]
        self.is_refreshed = True

    def evaluate(self, z: torch.Tensor, /) -> torch.Tensor:
        """Nyström-extend the eigenfunctions to ``z``.

        Args:
            z: ``[N, d]`` query latents (a ``[N]`` tensor is read as ``d = 1``).

        Returns:
            ``[N, n_modes]`` eigenfunction values.

        Raises:
            IGLConfigError: If :meth:`refresh` has not been called, or ``z``
                does not have the refresh latents' width.
        """
        if not self.is_refreshed:
            raise IGLConfigError(
                "LearnedLaplacianBasis: call .refresh(z) before .evaluate(z).",
            )
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        refresh: torch.Tensor = self._refresh_points  # pyright: ignore[reportAssignmentType]
        if z.shape[-1] != refresh.shape[-1]:
            raise IGLConfigError(
                f"z has {z.shape[-1]} coordinates; the basis was refreshed on {refresh.shape[-1]}-dimensional latents",
            )
        eigvecs: torch.Tensor = self._eigenvectors  # pyright: ignore[reportAssignmentType]
        degrees: torch.Tensor = self._degrees  # pyright: ignore[reportAssignmentType]
        radius_sq: torch.Tensor = self._knn_radius_sq  # pyright: ignore[reportAssignmentType]
        mu: torch.Tensor = self._affinity_eigs  # pyright: ignore[reportAssignmentType]
        sigma_sq: torch.Tensor = self._sigma_sq  # pyright: ignore[reportAssignmentType]

        # The graph's symmetrised kNN affinity, rebuilt for the query: a point that
        # coincides with a refresh point is that point (the graph has no self-edges).
        # Exact (non-matmul) distances so a point that equals a refresh point has
        # distance exactly zero and is recognised as that point.
        d2 = torch.cdist(z, refresh, compute_mode="donot_use_mm_for_euclid_dist").pow(2)  # [N, M]
        d2_no_self = d2.masked_fill(d2 <= _SELF_DIST_SQ, float("inf"))
        k = min(self.k_nn, refresh.shape[0] - 1)
        nearest = d2_no_self.topk(k, dim=-1, largest=False).indices  # [N, k]
        forward = torch.zeros_like(d2).scatter_(-1, nearest, 1.0)  # z_j among the k nearest of z
        reverse = (d2_no_self <= radius_sq.unsqueeze(0) * (1.0 + _RADIUS_SLACK)).to(z.dtype)  # z inside z_j's k-NN radius
        affinity = 0.5 * torch.exp(-d2 / (2.0 * sigma_sq)) * (forward + reverse)  # [N, M]
        deg_z = affinity.sum(dim=-1, keepdim=True).clamp_min(_DEGREE_FLOOR)
        coef = affinity / torch.sqrt(deg_z * degrees.clamp_min(_DEGREE_FLOOR).unsqueeze(0))
        return (coef @ eigvecs) / mu.unsqueeze(0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.evaluate(z)


__all__ = ["LearnedLaplacianBasis"]
