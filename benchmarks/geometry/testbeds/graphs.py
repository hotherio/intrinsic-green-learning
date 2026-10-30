"""Graph hypersurfaces ``x -> (x, h(x))`` over ``[-1, 1]^d`` for ``d = 2, 3, 4``: the hills in any dimension.

The metric is analytic, ``g_ij = delta_ij + h_i h_j``, ``g^{ij} = delta_ij - h_i h_j / D`` with
``D = 1 + |grad h|^2`` and ``sqrt|g| = sqrt D``; the drift is ``b^j = D^{-1/2} d_i (sqrt D g^{ij})``.
A radial bump is a hypersurface of revolution (conformally flat in every dimension); a generic
bump is not in ``d >= 4``. Everything is matrix-free: the flat Dirichlet solve is a discrete sine
transform, the curved operator a conservative finite-difference stencil, the true field a
conjugate-gradient solve preconditioned by the flat solve, and the Born series and its spectral
radius follow from the same two operators.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse.linalg as spla
from numpy.typing import NDArray
from scipy.fft import dstn, idstn

Array = NDArray[np.float64]
Bump = Callable[[list[Array]], tuple[Array, list[Array]]]

POINTS = {2: 49, 3: 33, 4: 21}
HEATER_CENTRE = (0.3, 0.1)
HEATER_WIDTH = {2: 0.18, 3: 0.25, 4: 0.3}
NBORN = 40


def radial_bump(coords: list[Array]) -> tuple[Array, list[Array]]:
    """``0.6 exp(-|x|^2 / 2 sigma^2)``, a hypersurface of revolution: height and its gradient."""
    r2 = sum(c**2 for c in coords)
    z = 0.6 * np.exp(-r2 / (2 * 0.35**2))
    return z, [-z * c / 0.35**2 for c in coords]


def generic_bump(coords: list[Array]) -> tuple[Array, list[Array]]:
    """An off-centre, rotated, anisotropic bump plus a smaller second one: no symmetry in any dimension."""
    d = len(coords)
    c, s = np.cos(np.radians(35)), np.sin(np.radians(35))
    x0, x1 = coords[0] - 0.35, coords[1] - 0.2
    u, w = c * x0 + s * x1, -s * x0 + c * x1
    rest = sum(coords[k] ** 2 for k in range(2, d)) if d > 2 else 0.0
    z1 = 0.6 * np.exp(-(u**2 / (2 * 0.32**2) + w**2 / (2 * 0.16**2) + rest / (2 * 0.4**2)))
    du, dw = -z1 * u / 0.32**2, -z1 * w / 0.16**2
    grad = [c * du - s * dw, s * du + c * dw] + [-z1 * coords[k] / 0.4**2 for k in range(2, d)]
    r2 = sum((coords[k] + 0.4 * (-1) ** k) ** 2 for k in range(d))
    z2 = 0.25 * np.exp(-r2 / (2 * 0.25**2))
    grad = [g - z2 * (coords[k] + 0.4 * (-1) ** k) / 0.25**2 for k, g in enumerate(grad)]
    return z1 + z2, grad


BUMPS: dict[str, Bump] = {"radial": radial_bump, "generic": generic_bump}


def _axis_first(a: Array, axis: int) -> Array:
    return np.moveaxis(a, axis, 0)


def _axis_back(a: Array, axis: int) -> Array:
    return np.moveaxis(a, 0, axis)


@dataclass(frozen=True, slots=True)
class GraphProblem:
    """A graph hypersurface at height factor ``s``: metric fields, source, true field and compensated source."""

    name: str
    d: int
    n: int
    s: float
    h: float
    coords: list[Array]
    ginv: list[list[Array]]
    sqrtg: Array
    drift: list[Array]
    f: Array
    u: Array
    fhat: Array

    @property
    def inner(self) -> tuple[slice, ...]:
        """Slices of the interior points."""
        return tuple(slice(1, -1) for _ in range(self.d))

    @property
    def max_dev(self) -> float:
        """``max |g^{-1} - I|`` over the grid."""
        dev = 0.0
        for i in range(self.d):
            for j in range(self.d):
                dev = max(dev, float(np.abs(self.ginv[i][j] - (1.0 if i == j else 0.0)).max()))
        return dev


def make_coords(d: int, n: int) -> tuple[list[Array], float]:
    """Mesh of ``n`` points per axis on ``[-1, 1]^d`` and the spacing."""
    x = np.linspace(-1, 1, n)
    grids = np.meshgrid(*([x] * d), indexing="ij")
    return [np.asarray(g, dtype=float) for g in grids], float(x[1] - x[0])


def metric_fields(coords: list[Array], h: float, bump: Bump, s: float) -> tuple[list[list[Array]], Array, list[Array]]:
    """Inverse metric ``g^{ij}``, ``sqrt|g|`` and the drift of ``z = s h(x)``."""
    d = len(coords)
    _, grad = bump(coords)
    grad = [s * g for g in grad]
    det = 1 + sum(g**2 for g in grad)
    sqrtg = np.sqrt(det)
    ginv = [[(1.0 if i == j else 0.0) - grad[i] * grad[j] / det for j in range(d)] for i in range(d)]
    drift = [sum(np.gradient(sqrtg * ginv[i][j], h, axis=i) for i in range(d)) / sqrtg for j in range(d)]
    return ginv, sqrtg, drift


def heater(coords: list[Array], width: float) -> Array:
    """Rank-one Gaussian source centred at ``(0.3, 0.1, 0, ...)``."""
    d = len(coords)
    centre = [HEATER_CENTRE[0], HEATER_CENTRE[1]] + [0.0] * (d - 2)
    out = np.ones_like(coords[0])
    for c, c0 in zip(coords, centre, strict=True):
        out = out * np.exp(-((c - c0) ** 2) / (2 * width**2))
    return out


def flat_solve(rhs: Array, h: float) -> Array:
    """``(-Delta)^{-1} rhs`` on the interior points with zero Dirichlet data, by a type-I sine transform."""
    d = rhs.ndim
    n_in = rhs.shape[0]
    k = np.arange(1, n_in + 1)
    lam1 = (2 - 2 * np.cos(np.pi * k / (n_in + 1))) / h**2
    lam = np.zeros_like(rhs)
    for axis in range(d):
        shape = [1] * d
        shape[axis] = n_in
        lam = lam + lam1.reshape(shape)
    coeff = dstn(rhs, type=1, norm="ortho")
    return np.asarray(idstn(coeff / lam, type=1, norm="ortho"), dtype=float)


def apply_flat(u_full: Array, h: float) -> Array:
    """``-Delta u`` on the interior points (``u`` given on the full grid with a zero edge)."""
    d = u_full.ndim
    out = np.zeros_like(u_full)
    for axis in range(d):
        a = _axis_first(u_full, axis)
        o = _axis_first(out, axis)
        o[1:-1] += (2 * a[1:-1] - a[2:] - a[:-2]) / h**2
    inner = tuple(slice(1, -1) for _ in range(d))
    return out[inner]


def curved_apply(u_full: Array, ginv: list[list[Array]], sqrtg: Array, h: float) -> Array:
    """``-Delta_g u`` on the interior points for metric fields on a grid (half-point fluxes, central mixed terms)."""
    d = u_full.ndim
    div = np.zeros_like(u_full)
    for i in range(d):
        p = sqrtg * ginv[i][i]
        pa, ua, da = _axis_first(p, i), _axis_first(u_full, i), _axis_first(div, i)
        half = 0.5 * (pa[1:] + pa[:-1])
        flux = half * (ua[1:] - ua[:-1]) / h
        da[1:-1] += (flux[1:] - flux[:-1]) / h
    mix = np.zeros_like(u_full)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            m = sqrtg * ginv[i][j]
            uj = np.gradient(u_full, h, axis=j)
            mix += np.gradient(m * uj, h, axis=i)
    out = -(div + mix) / sqrtg
    return out[tuple(slice(1, -1) for _ in range(d))]


def apply_curved(u_full: Array, prob: GraphProblem) -> Array:
    """``-Delta_g u`` on the interior points of ``prob``'s grid."""
    return curved_apply(u_full, prob.ginv, prob.sqrtg, prob.h)


def _embed(v: Array, d: int, n: int) -> Array:
    u = np.zeros((n,) * d)
    u[tuple(slice(1, -1) for _ in range(d))] = v
    return u


def solve_poisson(ginv: list[list[Array]], sqrtg: Array, f: Array, h: float, *, tol: float = 1e-10) -> Array:
    """``u`` with ``-Delta_g u = f`` and ``u = 0`` on the edge, for metric fields on a grid (preconditioned GMRES)."""
    d, n = f.ndim, f.shape[0]
    inner = tuple(slice(1, -1) for _ in range(d))
    f_in = f[inner]
    shape, size = f_in.shape, f_in.size

    def curved_mv(v: Array) -> Array:
        return curved_apply(_embed(v.reshape(shape), d, n), ginv, sqrtg, h).ravel()

    def flat_inv_mv(v: Array) -> Array:
        return flat_solve(v.reshape(shape), h).ravel()

    op = spla.LinearOperator((size, size), matvec=curved_mv, dtype=float)
    pre = spla.LinearOperator((size, size), matvec=flat_inv_mv, dtype=float)
    u_in, info = spla.gmres(op, f_in.ravel(), M=pre, rtol=tol, restart=50, maxiter=400)
    if info != 0:
        raise RuntimeError(f"curved solve did not converge (info={info})")
    return _embed(np.asarray(u_in).reshape(shape), d, n)


def from_metric(
    name: str, coords: list[Array], h: float, ginv: list[list[Array]], sqrtg: Array, f: Array, *, s: float = 1.0
) -> GraphProblem:
    """A grid problem from arbitrary metric fields (conformally flat metrics, SPD(2) in log-Eig coordinates, ...)."""
    d, n = f.ndim, f.shape[0]
    drift = [sum(np.gradient(sqrtg * ginv[i][j], h, axis=i) for i in range(d)) / sqrtg for j in range(d)]
    u_full = solve_poisson(ginv, sqrtg, f, h)
    fhat = _embed(apply_flat(u_full, h), d, n)
    return GraphProblem(name, d, n, s, h, coords, ginv, sqrtg, drift, f, u_full, fhat)


def make_problem(name: str, d: int, s: float, *, n: int | None = None, cg_tol: float = 1e-10) -> GraphProblem:
    """Assemble the fields and solve the curved problem for bump ``name`` in dimension ``d`` at height factor ``s``."""
    n = n or POINTS[d]
    coords, h = make_coords(d, n)
    ginv, sqrtg, drift = metric_fields(coords, h, BUMPS[name], s)
    f = heater(coords, HEATER_WIDTH[d])
    prob = GraphProblem(name, d, n, s, h, coords, ginv, sqrtg, drift, f, np.zeros_like(f), np.zeros_like(f))
    inner = prob.inner
    f_in = f[inner]
    shape = f_in.shape
    size = f_in.size

    def curved_mv(v: Array) -> Array:
        return apply_curved(_embed(v.reshape(shape), d, n), prob).ravel()

    def flat_inv_mv(v: Array) -> Array:
        return flat_solve(v.reshape(shape), h).ravel()

    op = spla.LinearOperator((size, size), matvec=curved_mv, dtype=float)
    pre = spla.LinearOperator((size, size), matvec=flat_inv_mv, dtype=float)
    u_in, info = spla.gmres(op, f_in.ravel(), M=pre, rtol=cg_tol, restart=50, maxiter=400)
    if info != 0:
        raise RuntimeError(f"curved solve did not converge (info={info})")
    u_full = _embed(np.asarray(u_in).reshape(shape), d, n)
    fhat = _embed(apply_flat(u_full, h), d, n)
    return GraphProblem(name, d, n, s, h, coords, ginv, sqrtg, drift, f, u_full, fhat)


def best_scale(prob: GraphProblem) -> float:
    """Global scale ``c`` of the flat kernel, ``mean tr(g^{-1}) / d``: the chart is compared with ``c I``, not ``I``."""
    return float(sum(prob.ginv[i][i].mean() for i in range(prob.d)) / prob.d)


def born_errors(prob: GraphProblem, n_born: int = NBORN, *, scale: float = 1.0) -> list[float]:
    """Relative error of the Born partial sums ``u_{n+1} = G_c f + G_c K_c u_n``, flat kernel ``G_c = (-c Delta)^{-1}``."""
    inner = prob.inner
    f_in, u_true = prob.f[inner], prob.u[inner]
    gf = flat_solve(f_in, prob.h) / scale
    u_n = gf.copy()
    norm = np.linalg.norm(u_true)
    errors: list[float] = []
    for k in range(n_born + 1):
        errors.append(float(np.linalg.norm(u_n - u_true) / norm))
        if k == n_born:
            break
        u_full = _embed(u_n, prob.d, prob.n)
        ku = scale * apply_flat(u_full, prob.h) - apply_curved(u_full, prob)
        u_n = gf + flat_solve(ku, prob.h) / scale
    return errors


def depth(errors: list[float], eps: float) -> int | None:
    """First Born index whose relative error is below ``eps``."""
    return next((k for k, e in enumerate(errors) if e < eps), None)


def spectral_radius(prob: GraphProblem, iters: int = 300, seed: int = 0, *, scale: float = 1.0) -> float:
    """``rho(G_c K_c)`` by power iteration, with the flat kernel at global scale ``c``.

    The operator is not symmetric, so the norm ratio of one step oscillates; the estimate is the geometric
    mean of the ratios over the second half of the iterations, which converges to the spectral radius.
    """
    rng = np.random.default_rng(seed)
    inner_shape = prob.f[prob.inner].shape
    v = rng.standard_normal(inner_shape)
    v /= np.linalg.norm(v)
    logs: list[float] = []
    for _ in range(iters):
        u_full = _embed(v, prob.d, prob.n)
        w = flat_solve(scale * apply_flat(u_full, prob.h) - apply_curved(u_full, prob), prob.h) / scale
        nw = max(float(np.linalg.norm(w)), 1e-300)
        logs.append(math.log(nw))
        v = w / nw
    return math.exp(float(np.mean(logs[len(logs) // 2 :])))


def max_dev_scaled(prob: GraphProblem, scale: float) -> float:
    """``max |g^{-1}/c - I|`` in operator norm over the grid."""
    d = prob.d
    stack = np.stack(
        [np.stack([prob.ginv[i][j] / scale - (1.0 if i == j else 0.0) for j in range(d)], -1) for i in range(d)], -2
    )
    return float(np.linalg.norm(stack.reshape(-1, d, d), ord=2, axis=(1, 2)).max())


def tt_ranks(f: Array, tol: float = 0.01) -> list[int]:
    """Tensor-train ranks at relative tolerance ``tol`` (the separable rank of a matrix, in any dimension)."""
    d = f.ndim
    if d == 1:
        return [1]
    total = np.linalg.norm(f)
    budget = (tol * total) ** 2 / max(d - 1, 1)
    ranks: list[int] = []
    core = f.reshape(f.shape[0], -1)
    r = 1
    for k in range(d - 1):
        core = core.reshape(r * f.shape[k], -1)
        u, sv, vt = np.linalg.svd(core, full_matrices=False)
        tail = np.cumsum((sv**2)[::-1])[::-1]
        keep = int(np.sum(tail > budget)) or 1
        ranks.append(keep)
        core = sv[:keep, None] * vt[:keep]
        r = keep
    return ranks


def analyse(name: str, d: int, s: float, *, n: int | None = None, n_born: int = NBORN, scaled: bool = False) -> dict[str, Any]:
    """Contraction, Born depth, naive error, ranks and the shares of the two correction terms.

    With ``scaled`` the flat kernel is taken at the best global scale of the chart; the default
    ``1.0`` reproduces the deck, whose hills have ``mean tr(g^{-1})/d`` close to one anyway.
    """
    prob = make_problem(name, d, s, n=n)
    scale = best_scale(prob) if scaled else 1.0
    errors = born_errors(prob, n_born, scale=scale)
    rho = spectral_radius(prob, scale=scale)
    inner = prob.inner
    gf = flat_solve(prob.f[inner], prob.h)
    u_true = prob.u[inner]
    grads = [np.gradient(prob.u, prob.h, axis=i) for i in range(d)]
    stretch = np.zeros_like(prob.u)
    for i in range(d):
        for j in range(d):
            delta = 1.0 if i == j else 0.0
            stretch += (prob.ginv[i][j] - delta) * np.gradient(grads[j], prob.h, axis=i)
    drift = sum(prob.drift[j] * grads[j] for j in range(d))
    fmax = float(np.abs(prob.fhat).max())
    return {
        "bed": f"graph-{name}-d{d}",
        "d": d,
        "n": prob.n,
        "s": float(s),
        "rho": rho,
        "scale": scale,
        "max_dev": prob.max_dev if scale == 1.0 else max_dev_scaled(prob, scale),
        "errors": errors,
        "depth": {"1e-2": depth(errors, 1e-2), "1e-3": depth(errors, 1e-3), "1e-6": depth(errors, 1e-6)},
        "naive_error": float(np.linalg.norm(gf - u_true) / np.linalg.norm(u_true)),
        "tt_rank_f": tt_ranks(prob.f),
        "tt_rank_fhat": tt_ranks(prob.fhat),
        "tt_rank_u": tt_ranks(prob.u),
        "share_stretch": float(np.abs(stretch).max() / fmax) if fmax > 0 else 0.0,
        "share_drift": float(np.abs(drift).max() / fmax) if fmax > 0 else 0.0,
    }


__all__ = [
    "BUMPS",
    "POINTS",
    "GraphProblem",
    "analyse",
    "apply_curved",
    "apply_flat",
    "best_scale",
    "born_errors",
    "curved_apply",
    "depth",
    "flat_solve",
    "from_metric",
    "generic_bump",
    "heater",
    "make_coords",
    "make_problem",
    "max_dev_scaled",
    "metric_fields",
    "radial_bump",
    "solve_poisson",
    "spectral_radius",
    "tt_ranks",
]
