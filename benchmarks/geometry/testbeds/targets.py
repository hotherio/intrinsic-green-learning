"""Targets ``u`` on the test beds: solutions of ``-Delta_g u = f`` with a simple source, the premise of IGL.

Three families, in order of preference:

1. **grid Poisson** (``d <= 4``): the bed's metric evaluated on a grid of its parameter patch,
   the curved operator in conservative form, a rank-one Gaussian heater, zero Dirichlet data on
   the patch's edge; exact up to finite differences (hills, graphs, sphere caps, ellipsoids,
   products in ``d = 4``, SPD(2) in log-Eig coordinates);
2. **eigenfunctions** (any ``d``): on sphere caps the restrictions of harmonic polynomials, on
   flat tori Fourier modes; ``f = mu u`` exactly;
3. **graph Laplacian** (generic beds, ``d > 4``): a kNN graph with Gaussian weights and the
   ``alpha = 1`` density normalisation of diffusion maps, ``u = (L + eps I)^-1 f``; validated on
   ``S^2`` against family 2 and on SPD(2) against family 1 before use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from sklearn.neighbors import NearestNeighbors
from torch.func import jacrev, vmap

from benchmarks.geometry.testbeds import graphs
from benchmarks.geometry.testbeds.manifolds import Bed

Array = NDArray[np.float64]


def metric_on_grid(bed: Bed, coords: list[Array]) -> tuple[list[list[Array]], Array]:
    """``g^{ij}`` and ``sqrt|g|`` of ``bed`` at every point of a parameter grid (batched autograd of the embedding)."""
    d = bed.d
    shape = coords[0].shape
    t = torch.from_numpy(np.stack([c.ravel() for c in coords], axis=1)).to(torch.float64)
    j = vmap(jacrev(bed.embedding))(t)  # [N, D, d]
    if bed.ambient_metric is None:
        g = j.transpose(1, 2) @ j
    else:
        x = vmap(bed.embedding)(t)
        h = torch.stack([bed.ambient_metric(xi) for xi in x])
        g = j.transpose(1, 2) @ h @ j
    ginv = torch.linalg.inv(g).numpy()
    sqrtg = torch.sqrt(torch.linalg.det(g)).numpy().reshape(shape)
    fields = [[np.ascontiguousarray(ginv[:, i, j].reshape(shape)) for j in range(d)] for i in range(d)]
    return fields, sqrtg


def parameter_grid(bed: Bed, n: int) -> tuple[list[Array], float, list[Array]]:
    """A uniform grid of ``n`` points per axis over the bed's parameter box; also the 1-D axes."""
    axes = [np.linspace(float(lo), float(hi), n) for lo, hi in zip(bed.low.tolist(), bed.high.tolist(), strict=True)]
    coords = [np.asarray(c, dtype=float) for c in np.meshgrid(*axes, indexing="ij")]
    h = float(axes[0][1] - axes[0][0])
    for a in axes[1:]:
        if abs(float(a[1] - a[0]) - h) > 1e-12:
            raise ValueError("grid Poisson targets need a parameter box with equal side lengths")
    return coords, h, axes


def heater_on_grid(coords: list[Array], centre: list[float] | None, width: float) -> Array:
    """Rank-one Gaussian heater in parameter space."""
    d = len(coords)
    c0 = centre if centre is not None else [0.0] * d
    out = np.ones_like(coords[0])
    for c, m in zip(coords, c0, strict=True):
        out = out * np.exp(-((c - m) ** 2) / (2 * width**2))
    return out


@dataclass(frozen=True, slots=True)
class GridTargets:
    """A grid Poisson problem on a bed with an interpolator for the field at sampled parameter points."""

    bed: Bed
    problem: graphs.GraphProblem
    axes: list[Array]
    scale: float

    def u_at(self, t: torch.Tensor) -> torch.Tensor:
        """The field ``u`` at parameter points ``t`` ``[n, d]`` by linear interpolation on the grid."""
        interp = RegularGridInterpolator(tuple(self.axes), self.problem.u, bounds_error=False, fill_value=0.0)
        return torch.from_numpy(np.asarray(interp(t.numpy()), dtype=float))

    def f_at(self, t: torch.Tensor) -> torch.Tensor:
        """The source ``f`` at parameter points ``t``."""
        interp = RegularGridInterpolator(tuple(self.axes), self.problem.f, bounds_error=False, fill_value=0.0)
        return torch.from_numpy(np.asarray(interp(t.numpy()), dtype=float))


GRID_POINTS = {2: 49, 3: 33, 4: 21}


def grid_poisson(
    bed: Bed, *, n: int | None = None, centre: list[float] | None = None, width: float | None = None
) -> GridTargets:
    """Solve ``-Delta_g u = f`` on the bed's parameter box with the bed's analytic metric (``d <= 4``)."""
    if bed.d > 4:
        raise ValueError("grid Poisson targets are for d <= 4; use eigen or graph targets above")
    n = n or GRID_POINTS[bed.d]
    coords, h, axes = parameter_grid(bed, n)
    ginv, sqrtg = metric_on_grid(bed, coords)
    box = float(bed.high[0] - bed.low[0])
    w = width if width is not None else 0.12 * box * {2: 1.0, 3: 1.3, 4: 1.6}[bed.d]
    c0 = centre if centre is not None else [0.15 * box, 0.05 * box] + [0.0] * (bed.d - 2)
    f = heater_on_grid(coords, c0, w)
    prob = graphs.from_metric(bed.name, coords, h, ginv, sqrtg, f)
    return GridTargets(bed, prob, axes, graphs.best_scale(prob))


def eigen_targets(bed: Bed, t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(u, f)`` with ``-Delta_g u = f`` from eigenfunctions, on sphere caps and flat tori; ``None`` elsewhere."""
    if bed.name.startswith("S") and "x" not in bed.name and "axes" not in bed.name and bed.ambient == bed.d + 1:
        p = bed.d
        lin = x[:, 0] - 0.5 * x[:, 1]  # degree 1: eigenvalue p
        quad = x[:, 0] * x[:, 2] + 0.5 * (x[:, 1] ** 2 - x[:, 3 % bed.ambient] ** 2)  # degree 2: eigenvalue 2(p+1)
        u = lin + quad
        f = p * lin + 2 * (p + 1) * quad
        return u, f
    if bed.name.startswith("T"):
        k1 = torch.tensor([1.0] + [0.0] * (bed.d - 1), dtype=t.dtype)
        k2 = torch.tensor([1.0, 1.0] + [0.0] * (bed.d - 2), dtype=t.dtype)
        u = torch.cos(t @ k1) + 0.5 * torch.sin(t @ k2)
        f = torch.cos(t @ k1) * 1.0 + 0.5 * torch.sin(t @ k2) * 2.0
        return u, f
    return None


def graph_laplacian_targets(
    x_all: torch.Tensor,
    f_all: torch.Tensor,
    *,
    k: int = 40,
    alpha: float = 1.0,
    eps: float = 1e-3,
    dist: Array | None = None,
    cut_quantile: float = 0.99,
) -> torch.Tensor:
    """``u = (L + eps L_scale)^-1 f`` on a fixed-radius graph over ``x_all`` with the ``alpha``-normalised Gaussian kernel.

    The graph is an epsilon-ball graph: the radius ``R`` is the median distance to the ``k``-th neighbour and
    the Gaussian bandwidth ``sigma`` is set so that the ball holds the ``cut_quantile`` mass of a Gaussian in
    ``d`` dimensions (``R = sigma * sqrt(chi2_d^{-1}(cut_quantile))``), so the kernel is a fixed-bandwidth
    kernel truncated far in its tail. A kNN truncation would cut the kernel at a density-dependent radius
    (inside the Gaussian in ``d >= 8``), which breaks the ``alpha = 1`` density correction of diffusion maps.
    ``dist`` may supply a precomputed pairwise distance matrix (affine-invariant distances on SPD). ``eps``
    is in units of ``I - P``; the regulariser is ``eps * 2 / sigma^2``.
    """
    from scipy.stats import chi2

    x = x_all.detach().cpu().numpy().astype(float)
    n, d = x.shape
    if dist is None:
        nn = NearestNeighbors(n_neighbors=k + 1).fit(x)
        dists_k, _ = nn.kneighbors(x)
        radius = float(np.median(dists_k[:, k]))
        graph = nn.radius_neighbors_graph(x, radius=radius, mode="distance")
        graph = sp.csr_matrix(graph)
    else:
        dist = np.asarray(dist, dtype=float)
        radius = float(np.median(np.sort(dist, axis=1)[:, k]))
        graph = sp.csr_matrix(np.where(dist <= radius, dist, 0.0))
        d = int(x.shape[1])
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    sigma = radius / math.sqrt(float(chi2.ppf(cut_quantile, d)))
    kern = graph.copy()
    kern.data = np.exp(-(kern.data**2) / (2 * sigma**2))
    kern = kern + sp.identity(n, format="csr")  # the self-weight exp(0) = 1 dropped by the sparse graph
    kern = sp.csr_matrix(0.5 * (kern + kern.T))
    deg = np.asarray(kern.sum(1)).ravel()
    dinv = sp.diags(deg**-alpha)
    kern_a = dinv @ kern @ dinv  # density correction (alpha = 1 recovers the Laplace-Beltrami operator)
    deg_a = np.asarray(kern_a.sum(1)).ravel()
    # random-walk generator L = (2/sigma^2)(I - D^-1 K) -> -Delta; solved through its symmetric twin
    # L_sym = D^{-1/2} (D - K) D^{-1/2} with w = D^{1/2} u, by conjugate gradients
    d_isqrt = sp.diags(deg_a**-0.5)
    l_sym = sp.csr_matrix(d_isqrt @ (sp.diags(deg_a) - kern_a) @ d_isqrt * (2.0 / sigma**2))
    f = f_all.detach().cpu().numpy().astype(float)
    f = f - f.mean()
    rhs = np.sqrt(deg_a) * f
    reg = eps * (2.0 / sigma**2)
    w, info = spla.cg(l_sym + reg * sp.identity(n), rhs, rtol=1e-8, maxiter=5000)
    if info != 0:
        raise RuntimeError(f"graph Laplacian solve did not converge (info={info})")
    u = np.asarray(w, dtype=float) / np.sqrt(deg_a)
    return torch.from_numpy(u - u.mean())


def targets_for(bed: Bed, t: torch.Tensor, x: torch.Tensor, *, family: str = "auto") -> dict[str, Any]:
    """``u`` (and ``f`` when known) at the sampled points of ``bed`` by the preferred family."""
    fam = family
    if fam == "auto":
        fam = "grid" if bed.d <= 4 else ("eigen" if eigen_targets(bed, t[:1], x[:1]) is not None else "graph")
    if fam == "grid":
        gt = grid_poisson(bed)
        return {"family": "grid", "u": gt.u_at(t), "f": gt.f_at(t), "grid": gt}
    if fam == "eigen":
        pair = eigen_targets(bed, t, x)
        if pair is None:
            raise ValueError(f"no eigenfunction targets for {bed.name}")
        return {"family": "eigen", "u": pair[0], "f": pair[1]}
    box = float(bed.high[0] - bed.low[0])
    c0 = torch.tensor([0.15 * box, 0.05 * box] + [0.0] * (bed.d - 2), dtype=t.dtype)
    f = torch.exp(-((t - c0) ** 2).sum(1) / (2 * (0.2 * box) ** 2))
    dist = None
    if bed.extra.get("metric") == "airm":
        from benchmarks.geometry.testbeds.spd_beds import airm_pairwise_distance

        dist = airm_pairwise_distance(t, int(bed.extra["n"])).numpy()  # the graph must live in the AIRM geometry
    # validated on closed spheres against harmonics (shape error of u, degree 1 / 2): S^2 with k = 100 and the
    # 99 % cut 4 / 6 % at 3000 points; S^8 with k = 800 and the 90 % cut 6 / 9 % at 6000 points. On the beds'
    # bounded patches no boundary condition is imposed: u is a smooth graph-Laplacian field of f, not the
    # Dirichlet solution, and every arm sees the same u.
    k, cut = (100, 0.99) if bed.d <= 4 else (800, 0.9)
    return {"family": "graph", "u": graph_laplacian_targets(x, f, k=k, dist=dist, cut_quantile=cut), "f": f}


__all__ = [
    "GridTargets",
    "eigen_targets",
    "graph_laplacian_targets",
    "grid_poisson",
    "heater_on_grid",
    "metric_on_grid",
    "parameter_grid",
    "targets_for",
]
