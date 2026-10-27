"""Chart-only cost measures on a frozen chart ``z = Psi(x)``: what a target costs there, independent of the kernel's training.

Two measures of "products", computed after rescaling each axis of the chart by a constant to ``[0, 1]``
(the kernel's per-axis widths make a constant per-axis rescaling free, and nothing more: a nonlinear
relabelling of an axis is not free, so the axes are not rank-normalised):

- ``chart_rank``: ``u`` resampled on a regular grid of the chart, separable rank at a tolerance (SVD in
  ``d = 2``, tensor-train ranks above);
- ``bump_count``: the number of Gaussian atoms on a fixed grid of centres and widths that a sparse fit
  needs to reach the tolerance (orthogonal matching pursuit), the closest proxy of IGL's anchor count
  that does not depend on how the anchors were trained.

``greedy_prune`` is the deployed counterpart: starting from a trained design matrix, remove the anchor
whose deletion costs least and refit the readout, giving the error-against-anchors curve on one chart.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.interpolate import griddata
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.neighbors import KNeighborsRegressor

from benchmarks.geometry.testbeds.graphs import tt_ranks
from benchmarks.geometry.testbeds.hills import separable_rank

Array = NDArray[np.float64]
__all__ = ["affine_normalise", "bump_count", "chart_rank", "greedy_prune", "quantile_normalise"]


def affine_normalise(z: torch.Tensor) -> torch.Tensor:
    """Map each axis of ``z`` to ``[0, 1]`` by a constant rescaling (what the kernel's per-axis widths absorb)."""
    z = z.double()
    lo = z.min(dim=0).values
    hi = z.max(dim=0).values
    return (z - lo) / (hi - lo).clamp_min(1e-12)


def quantile_normalise(z: torch.Tensor) -> torch.Tensor:
    """Map each axis of ``z`` to ``[0, 1]`` by its empirical ranks (a monotone relabelling, more than the kernel absorbs)."""
    ranks = z.argsort(dim=0).argsort(dim=0).double()
    return ranks / max(z.shape[0] - 1, 1)


def chart_rank(z: torch.Tensor, u: torch.Tensor, *, tol: float = 0.01, points: int | None = None) -> dict[str, Any]:
    """Separable rank of ``u`` on a regular grid of the normalised chart (``d = 2``: SVD; ``d >= 3``: tensor-train ranks)."""
    zq = affine_normalise(z).numpy()
    d = zq.shape[1]
    n = points or {1: 65, 2: 33, 3: 17, 4: 11}.get(d, 7)
    axes = [np.linspace(0, 1, n)] * d
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, d)
    uv = u.detach().cpu().numpy().astype(float)
    if d == 2:
        vals = griddata(zq, uv, grid, method="linear")
        hole = np.isnan(vals)
        if hole.any():
            vals[hole] = griddata(zq, uv, grid[hole], method="nearest")
    else:
        knn = KNeighborsRegressor(n_neighbors=min(8, len(uv)), weights="distance").fit(zq, uv)
        vals = knn.predict(grid)
    field = np.asarray(vals, dtype=float).reshape((n,) * d)
    if d == 2:
        return {"rank": separable_rank(field, tol), "tt_ranks": [separable_rank(field, tol)], "grid": n}
    ranks = tt_ranks(field, tol)
    return {"rank": int(max(ranks)), "tt_ranks": ranks, "grid": n}


def bump_count(
    z: torch.Tensor,
    u: torch.Tensor,
    *,
    tol: float = 0.01,
    centres_per_axis: int | None = None,
    width_factors: tuple[float, ...] = (0.5, 1.0, 2.0),
    max_atoms: int = 256,
) -> dict[str, Any]:
    """Smallest number of fixed-grid Gaussian atoms whose sparse fit reaches relative error ``tol`` (matching pursuit).

    The centres form a grid of about 1024 / len(width_factors) points per width; the widths are the grid
    spacing times ``width_factors``, so the dictionary covers the chart in any dimension.
    """
    zq = affine_normalise(z).numpy()
    d = zq.shape[1]
    m = centres_per_axis or max(2, int(round((1024 / len(width_factors)) ** (1 / d))))
    axes = [np.linspace(0, 1, m)] * d
    centres = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, d)
    spacing = 1.0 / max(m - 1, 1)
    cols = []
    for w in (spacing * f for f in width_factors):
        d2 = ((zq[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        cols.append(np.exp(-d2 / (2 * w * w)))
    phi = np.concatenate(cols, axis=1)
    phi = phi / np.linalg.norm(phi, axis=0, keepdims=True).clip(1e-12)
    y = u.detach().cpu().numpy().astype(float)
    y = y - y.mean()
    norm = np.linalg.norm(y)
    curve: list[float] = []
    count = None
    for r in range(1, min(max_atoms, phi.shape[1]) + 1):
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=r, fit_intercept=True).fit(phi, y)
        err = float(np.linalg.norm(y - omp.predict(phi)) / norm)
        curve.append(err)
        if err <= tol:
            count = r
            break
    counts = {f"{t:g}": next((i + 1 for i, e in enumerate(curve) if e <= t), None) for t in (0.05, 0.02, 0.01)}
    return {"count": count, "counts": counts, "curve": curve, "n_candidates": int(phi.shape[1]), "censored": count is None}


def greedy_prune(
    phi_blocks: list[torch.Tensor], y: torch.Tensor, *, l2: float = 1e-3, min_blocks: int = 1
) -> list[dict[str, float]]:
    """Error against the number of anchors when anchors are removed greedily from a trained design, the readout refit by ridge.

    ``phi_blocks`` holds one ``[n, K]`` block of design columns per anchor (its scales); ``y`` ``[n]``.
    """
    y = y.double()
    norm = float(y.norm())
    alive = list(range(len(phi_blocks)))

    def fit_error(idx: list[int]) -> float:
        phi = torch.cat([phi_blocks[i].double() for i in idx], dim=1)
        gram = phi.T @ phi + l2 * torch.eye(phi.shape[1], dtype=phi.dtype)
        w = torch.linalg.solve(gram, phi.T @ y)
        return float((y - phi @ w).norm() / norm)

    curve = [{"anchors": len(alive), "error": fit_error(alive)}]
    while len(alive) > min_blocks:
        best = min(alive, key=lambda i: fit_error([j for j in alive if j != i]))
        alive.remove(best)
        curve.append({"anchors": len(alive), "error": fit_error(alive)})
    return curve
