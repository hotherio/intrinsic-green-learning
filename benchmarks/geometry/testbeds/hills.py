"""The two-dimensional hill test bed: a graph surface over the unit square with an analytic metric.

Setting: the square ``[-1, 1]^2`` with ``u = 0`` on its edge, a graph surface ``z = s h(x, y)``
whose height factor ``s`` runs from 0 (flat) to 1, and one Gaussian heater ``f`` (a product of
two one-dimensional Gaussians, so of separable rank one).

    curved problem   -Delta_g u = f          (Laplace-Beltrami of the surface)
    flat chart       -Delta u = fhat,  fhat = f + (g^ij - delta^ij) d_i d_j u + b^j d_j u
    Born series      u_{n+1} = G f + G K u_n,   G = (-Delta)^-1,  K = Delta_g - Delta

Finite differences on an ``m x m`` grid, conservative form for the curved operator. Ported
from the research deck's build script (``make_born_data.py``) so that the deck's numbers are
reproducible from the repository: at full height the round hill gives ``rho = 0.507``, a Born
depth of 8 to ``1e-3`` and a naive error of 33 %; the lopsided hill ``0.739``, 18 and 49 %.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray
from scipy.interpolate import griddata

Array = NDArray[np.float64]
Hill = Callable[[Array, Array], tuple[Array, Array, Array]]

M_DEFAULT = 49
TOL = 0.01
HEATER = (0.3, 0.1, 0.18)
NBORN = 40
A_WARP = 0.5
MAP = 33


def round_hill(x: Array, y: Array) -> tuple[Array, Array, Array]:
    """Rotationally symmetric Gaussian bump: height and its two partial derivatives."""
    z = 0.6 * np.exp(-(x**2 + y**2) / (2 * 0.35**2))
    return z, -z * x / 0.35**2, -z * y / 0.35**2


def lopsided_hill(x: Array, y: Array) -> tuple[Array, Array, Array]:
    """Anisotropic bump, off-centre and rotated by 35 degrees: no symmetry, no separable chart."""
    c, s = np.cos(np.radians(35)), np.sin(np.radians(35))
    u, w = c * (x - 0.35) + s * (y - 0.2), -s * (x - 0.35) + c * (y - 0.2)
    z = 0.6 * np.exp(-(u**2 / (2 * 0.32**2) + w**2 / (2 * 0.16**2)))
    du, dw = -z * u / 0.32**2, -z * w / 0.16**2
    return z, c * du - s * dw, s * du + c * dw


HILLS: dict[str, Hill] = {"round": round_hill, "lop": lopsided_hill}


@dataclass(frozen=True, slots=True)
class Grid:
    """A uniform ``m x m`` grid on the square with its spacing and interior slices."""

    m: int
    x: Array
    h: float
    X: Array
    Y: Array

    @property
    def inner(self) -> tuple[slice, slice]:
        """Slices selecting the interior points (the unknowns of the Dirichlet problem)."""
        return (slice(1, -1), slice(1, -1))


def make_grid(m: int = M_DEFAULT) -> Grid:
    """Uniform grid of ``m`` points per axis on ``[-1, 1]``."""
    x = np.linspace(-1, 1, m)
    big_x, big_y = np.meshgrid(x, x, indexing="ij")
    return Grid(m=m, x=x, h=float(x[1] - x[0]), X=big_x, Y=big_y)


def heater(grid: Grid, centre: tuple[float, float] = HEATER[:2], width: float = HEATER[2]) -> Array:
    """The rank-one Gaussian source."""
    cx, cy = centre
    return np.exp(-((grid.X - cx) ** 2) / (2 * width**2)) * np.exp(-((grid.Y - cy) ** 2) / (2 * width**2))


@dataclass(frozen=True, slots=True)
class Metric:
    """Inverse metric entries, ``sqrt|g|`` and the drift ``b^j`` of the graph surface on the grid."""

    gxx: Array
    gxy: Array
    gyy: Array
    sq: Array
    bx: Array
    by: Array

    @property
    def max_dev(self) -> float:
        """``max |g^{-1} - I|`` over the grid, the deck's proxy for the Born contraction."""
        return float(np.max(np.abs(np.stack([self.gxx - 1, self.gxy, self.gyy - 1]))))


def metric(grid: Grid, hill: Hill, s: float) -> Metric:
    """Metric data of ``z = s h(x, y)``: ``g^{-1} = I - grad h grad h^T / D`` with ``D = 1 + |grad h|^2``."""
    _, hx, hy = hill(grid.X, grid.Y)
    hx, hy = s * hx, s * hy
    det = 1 + hx**2 + hy**2
    gxx, gxy, gyy = (1 + hy**2) / det, -hx * hy / det, (1 + hx**2) / det
    sq = np.sqrt(det)
    h = grid.h
    bx = (np.gradient(sq * gxx, h, axis=0) + np.gradient(sq * gxy, h, axis=1)) / sq
    by = (np.gradient(sq * gxy, h, axis=0) + np.gradient(sq * gyy, h, axis=1)) / sq
    return Metric(gxx, gxy, gyy, sq, bx, by)


def flat_operator(grid: Grid) -> sp.csc_matrix:
    """``-Delta`` with zero Dirichlet data on the interior points, as a sparse matrix."""
    n = grid.m - 2
    e = np.ones(n)
    t = sp.diags([-e, 2 * e, -e], [-1, 0, 1], shape=(n, n))
    eye = sp.identity(n)
    return sp.csc_matrix((sp.kron(eye, t) + sp.kron(t, eye)) / grid.h**2)


def apply_curved(grid: Grid, u: Array, met: Metric) -> Array:
    """``-Delta_g u`` on the full grid (``u`` zero on the edge), conservative form, central mixed terms."""
    h = grid.h
    p, q, m = met.sq * met.gxx, met.sq * met.gyy, met.sq * met.gxy
    out = np.zeros_like(u)
    px = 0.5 * (p[1:, :] + p[:-1, :])
    qy = 0.5 * (q[:, 1:] + q[:, :-1])
    fx = px * (u[1:, :] - u[:-1, :]) / h
    fy = qy * (u[:, 1:] - u[:, :-1]) / h
    div = np.zeros_like(u)
    div[1:-1, :] += (fx[1:, :] - fx[:-1, :]) / h
    div[:, 1:-1] += (fy[:, 1:] - fy[:, :-1]) / h
    uy = np.zeros_like(u)
    uy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * h)
    ux = np.zeros_like(u)
    ux[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * h)
    mix = np.zeros_like(u)
    mix[1:-1, :] += (m[2:, :] * uy[2:, :] - m[:-2, :] * uy[:-2, :]) / (2 * h)
    mix[:, 1:-1] += (m[:, 2:] * ux[:, 2:] - m[:, :-2] * ux[:, :-2]) / (2 * h)
    inner = grid.inner
    out[inner] = -(div[inner] + mix[inner]) / met.sq[inner]
    return out


def curved_operator(grid: Grid, met: Metric) -> sp.csc_matrix:
    """The curved operator on the interior points as a sparse matrix, assembled column by column."""
    n = (grid.m - 2) ** 2
    cols: list[Array] = []
    u = np.zeros((grid.m, grid.m))
    for k in range(n):
        i, j = divmod(k, grid.m - 2)
        u[...] = 0.0
        u[i + 1, j + 1] = 1.0
        cols.append(apply_curved(grid, u, met)[grid.inner].ravel())
    return sp.csc_matrix(np.array(cols).T)


def full(grid: Grid, v: Array) -> Array:
    """Interior vector to the full grid with a zero edge."""
    u = np.zeros((grid.m, grid.m))
    u[grid.inner] = v.reshape(grid.m - 2, grid.m - 2)
    return u


def separable_rank(f: Array, tol: float = TOL) -> int:
    """Smallest ``r`` whose relative Frobenius SVD tail is at most ``tol`` (the deck's "products")."""
    e = np.linalg.svd(f, compute_uv=False) ** 2
    if e.sum() == 0:
        return 0
    tail = np.append(np.cumsum(e[::-1])[::-1][1:], 0)
    return int(np.argmax(np.sqrt(tail / e.sum()) <= tol) + 1)


def additive_fit(f: Array, mask: NDArray[np.bool_] | None = None, iters: int = 30) -> tuple[float, float]:
    """Best fit ``f ~ a(x) + b(y)`` over ``mask`` by alternating means: ``(R^2, RMS residual)``.

    The RMS is the part of ``f`` no one-dimensional relabelling of the axes removes; unlike
    ``R^2`` it does not grow when a reparametrisation adds a large additive part.
    """
    w = np.ones_like(f) if mask is None else mask.astype(float)
    a = np.zeros(f.shape[0])
    b = np.zeros(f.shape[1])
    for _ in range(iters):
        a = ((f - b[None, :]) * w).sum(1) / np.maximum(w.sum(1), 1)
        b = ((f - a[:, None]) * w).sum(0) / np.maximum(w.sum(0), 1)
    fit = a[:, None] + b[None, :]
    mean = (f * w).sum() / w.sum()
    ss = (((f - mean) ** 2) * w).sum()
    res = (((f - fit) ** 2) * w).sum()
    r2 = float(1 - res / ss) if ss > 0 else 1.0
    return r2, float(np.sqrt(res / w.sum()))


def second_derivatives(grid: Grid, u: Array) -> tuple[Array, Array, Array, Array, Array]:
    """``u_x, u_y, u_xx, u_xy, u_yy`` by central differences (zero second derivatives on the edge)."""
    h = grid.h
    uxx = np.zeros_like(u)
    uyy = np.zeros_like(u)
    uxy = np.zeros_like(u)
    uxx[1:-1, :] = (u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / h**2
    uyy[:, 1:-1] = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / h**2
    uxy[1:-1, 1:-1] = (u[2:, 2:] - u[2:, :-2] - u[:-2, 2:] + u[:-2, :-2]) / (4 * h**2)
    return np.gradient(u, h, axis=0), np.gradient(u, h, axis=1), uxx, uxy, uyy


def neumann_integrate(grid: Grid, vx: Array, vy: Array) -> Array:
    """Least-squares potential ``q`` with ``grad q ~ (vx, vy)`` on the full grid, centre pinned to zero."""
    m, h = grid.m, grid.h
    n = m * m
    k = np.arange(n).reshape(m, m)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs: list[float] = []
    r = 0
    for i in range(m - 1):
        for j in range(m):
            rows += [r, r]
            cols += [int(k[i + 1, j]), int(k[i, j])]
            vals += [1 / h, -1 / h]
            rhs.append(0.5 * (vx[i + 1, j] + vx[i, j]))
            r += 1
    for i in range(m):
        for j in range(m - 1):
            rows += [r, r]
            cols += [int(k[i, j + 1]), int(k[i, j])]
            vals += [1 / h, -1 / h]
            rhs.append(0.5 * (vy[i, j + 1] + vy[i, j]))
            r += 1
    rows.append(r)
    cols.append(int(k[m // 2, m // 2]))
    vals.append(1.0)
    rhs.append(0.0)
    gm = sp.csr_matrix((vals, (rows, cols)), shape=(r + 1, n))
    q = spla.spsolve((gm.T @ gm).tocsc(), gm.T @ np.array(rhs))
    return np.asarray(q).reshape(m, m)


def chart_residuals(m11: Array, m12: Array, m22: Array) -> dict[str, float]:
    """Three residuals of a chart from its ``2 x 2`` inverse metric, each the square root of a penalty value.

    ``orth`` is IGL's orthogonality penalty (zero iff the coordinate lines are perpendicular),
    ``conf`` the conformality penalty (zero iff the chart is conformal), ``iso`` the distance from
    the identity (zero iff the chart is an isometry). A 2-cell border is dropped: one-sided
    gradients there.
    """
    tr = 0.5 * (m11 + m22)
    inner = (slice(2, -2), slice(2, -2))
    norm = (m11**2 + 2 * m12**2 + m22**2)[inner].mean()
    diag = (m11**2 + m22**2)[inner].mean()
    orth = np.sqrt((2 * m12**2)[inner].mean() / diag)
    conf = np.sqrt(((m11 - tr) ** 2 + 2 * m12**2 + (m22 - tr) ** 2)[inner].mean() / norm)
    iso = np.sqrt(((m11 - 1) ** 2 + 2 * m12**2 + (m22 - 1) ** 2)[inner].mean() / norm)
    return {"orth": float(orth), "conf": float(conf), "iso": float(iso)}


@dataclass(frozen=True, slots=True)
class IsothermalChart:
    """The conformal chart ``(p, q)`` of the surface and its conformal factor on the ``x-y`` grid."""

    p: Array
    q: Array
    lam: Array
    residuals: dict[str, float]
    cols_q: tuple[Array, Array]


def isothermal_chart(grid: Grid, hill: Hill, s: float, met: Metric, curved: sp.csc_matrix) -> IsothermalChart:
    """Harmonic coordinate ``p`` (``Delta_g p = 0``, ``p = x`` on the edge) and its conjugate ``q`` (``dq = *dp``)."""
    m, h = grid.m, grid.h
    edge = (np.arange(m)[:, None] % (m - 1) == 0) | (np.arange(m)[None, :] % (m - 1) == 0)
    xb = np.where(edge, grid.X, 0.0)
    rhs = -apply_curved(grid, xb, met)[grid.inner].ravel()
    p = xb.copy()
    p[grid.inner] = np.asarray(spla.spsolve(curved, rhs)).reshape(m - 2, m - 2)
    px, py = np.gradient(p, h, axis=0), np.gradient(p, h, axis=1)
    qx = -met.sq * (met.gxy * px + met.gyy * py)
    qy = met.sq * (met.gxx * px + met.gxy * py)
    q = neumann_integrate(grid, qx, qy)
    _, hx, hy = hill(grid.X, grid.Y)
    hx, hy = s * hx, s * hy
    g11, g12, g22 = 1 + hx**2, hx * hy, 1 + hy**2
    jqx, jqy = np.gradient(q, h, axis=0), np.gradient(q, h, axis=1)
    det = px * jqy - py * jqx
    a11, a21 = jqy / det, -jqx / det
    a12, a22 = -py / det, px / det

    def quad(u1: Array, u2: Array, v1: Array, v2: Array) -> Array:
        return g11 * u1 * v1 + g12 * (u1 * v2 + u2 * v1) + g22 * u2 * v2

    m11, m12, m22 = quad(a11, a21, a11, a21), quad(a11, a21, a12, a22), quad(a12, a22, a12, a22)
    lam = 0.5 * (m11 + m22)
    det_m = m11 * m22 - m12**2
    res = chart_residuals(m22 / det_m, -m12 / det_m, m11 / det_m)
    return IsothermalChart(p=p, q=q, lam=lam, residuals=res, cols_q=(a12, a22))


def resample(
    grid: Grid, p: Array, q: Array, fields: list[Array], n: int = MAP
) -> tuple[list[Array], tuple[float, float, float, float]]:
    """Fields on the ``x-y`` grid (2-cell border dropped) resampled on a regular ``(p, q)`` grid spanning the image."""
    core = (slice(2, -2), slice(2, -2))
    pts = np.column_stack([p[core].ravel(), q[core].ravel()])
    pg = np.linspace(p[core].min(), p[core].max(), n)
    qg = np.linspace(q[core].min(), q[core].max(), n)
    big_p, big_q = np.meshgrid(pg, qg, indexing="ij")
    out: list[Array] = []
    for f in fields:
        r = np.asarray(griddata(pts, f[core].ravel(), (big_p, big_q), method="linear"), dtype=float)
        hole = np.isnan(r)
        if hole.any():
            r[hole] = griddata(pts, f[core].ravel(), (big_p[hole], big_q[hole]), method="nearest")
        out.append(r)
    del grid
    return out, (float(pg[0]), float(pg[-1]), float(qg[0]), float(qg[-1]))


def orthogonal_chart(
    grid: Grid, chart: IsothermalChart, f: Array, u: Array, box: tuple[float, float, float, float], warp: float = A_WARP
) -> tuple[Array, int, dict[str, float]]:
    """An orthogonal chart that is not conformal: ``(p, Q(q))`` with ``Q`` a monotone relabelling of ``q``.

    Its metric is ``diag(lambda, lambda / Q'^2)``: the orthogonality penalty is exactly zero on it, the
    conformality penalty is not, and the source the flat kernel needs there is
    ``lambda f + (1 - Q'^-2) u_qq + (Q'' / Q'^3) u_q``. Returns the source on a regular ``(p, Q)``
    grid, its separable rank, and the chart's residuals.
    """
    h = grid.h
    xq, yq = chart.cols_q
    ux, uy = np.gradient(u, h, axis=0), np.gradient(u, h, axis=1)
    uq = ux * xq + uy * yq
    uqq = np.gradient(uq, h, axis=0) * xq + np.gradient(uq, h, axis=1) * yq
    (lamf_pq, uq_pq, uqq_pq, lam_pq), _ = resample(grid, chart.p, chart.q, [chart.lam * f, uq, uqq, chart.lam])
    n = lamf_pq.shape[1]
    qs = np.linspace(box[2], box[3], n)
    q0, half = 0.5 * (box[2] + box[3]), 0.5 * (box[3] - box[2])
    big_q = qs + warp * (half / np.pi) * np.sin(np.pi * (qs - q0) / half)
    q1 = 1 + warp * np.cos(np.pi * (qs - q0) / half)
    q2 = -warp * (np.pi / half) * np.sin(np.pi * (qs - q0) / half)
    c1, c2 = (1 - q1**-2)[None, :], (q2 / q1**3)[None, :]
    fhat_pq = lamf_pq + c1 * uqq_pq + c2 * uq_pq
    qg = np.linspace(big_q[0], big_q[-1], n)
    q_of = np.interp(qg, big_q, qs)
    fhat = np.array([np.interp(q_of, qs, row) for row in fhat_pq])
    lam_safe = np.where(lam_pq > 0, lam_pq, 1.0)
    m11 = 1 / lam_safe
    m22 = (q1[None, :] ** 2) / lam_safe
    return fhat, separable_rank(fhat), chart_residuals(m11, np.zeros_like(m11), m22)


@dataclass(frozen=True, slots=True)
class HillProblem:
    """One hill at one height: operators, the true field and the compensated source on the grid."""

    name: str
    s: float
    grid: Grid
    met: Metric
    flat: sp.csc_matrix
    flat_lu: Any
    curved: sp.csc_matrix
    f: Array
    u: Array
    fhat: Array

    @property
    def height(self) -> float:
        """Peak height of the surface, ``0.6 s``."""
        return 0.6 * self.s


def make_problem(name: str, s: float, *, m: int = M_DEFAULT, flat_lu: Any | None = None) -> HillProblem:
    """Assemble the operators and solve the curved problem for ``hill`` at height factor ``s``."""
    hill = HILLS[name]
    grid = make_grid(m)
    met = metric(grid, hill, s)
    flat = flat_operator(grid)
    lu = flat_lu if flat_lu is not None else spla.splu(flat)
    curved = curved_operator(grid, met)
    f = heater(grid)
    fv = f[grid.inner].ravel()
    u = full(grid, np.asarray(spla.spsolve(curved, fv)))
    fhat = full(grid, flat @ u[grid.inner].ravel())
    return HillProblem(name, s, grid, met, flat, lu, curved, f, u, fhat)


def born_errors(prob: HillProblem, n_born: int = NBORN) -> list[float]:
    """Relative error of the Born partial sums ``u_{n+1} = G f + G K u_n`` against the true field."""
    fv = prob.f[prob.grid.inner].ravel()
    u_true = prob.u[prob.grid.inner].ravel()
    gf = prob.flat_lu.solve(fv)
    u_n = gf.copy()
    norm = np.linalg.norm(u_true)
    errors: list[float] = []
    for n in range(n_born + 1):
        errors.append(float(np.linalg.norm(u_n - u_true) / norm))
        if n == n_born:
            break
        ku = prob.flat @ u_n - prob.curved @ u_n
        u_n = gf + prob.flat_lu.solve(ku)
    return errors


def depth(errors: list[float], eps: float) -> int | None:
    """First Born index whose relative error is below ``eps``, or ``None`` if never reached."""
    return next((n for n, e in enumerate(errors) if e < eps), None)


def spectral_radius(prob: HillProblem, iters: int = 80, seed: int = 0) -> float:
    """``rho(G K)`` by power iteration, the contraction of the Born series."""
    rng = np.random.default_rng(seed)
    n = (prob.grid.m - 2) ** 2
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    rho = 0.0
    for _ in range(iters):
        w = prob.flat_lu.solve(prob.flat @ v - prob.curved @ v)
        rho, v = float(np.linalg.norm(w)), w / max(np.linalg.norm(w), 1e-30)
    return rho


def analyse(name: str, s: float, *, m: int = M_DEFAULT, n_born: int = NBORN, charts: bool = True) -> dict[str, Any]:
    """Every number the deck reports for one hill at one height, without the widget maps."""
    prob = make_problem(name, s, m=m)
    grid, met = prob.grid, prob.met
    errors = born_errors(prob, n_born)
    rho = spectral_radius(prob)
    ux, uy, uxx, uxy, uyy = second_derivatives(grid, prob.u)
    stretch = (met.gxx - 1) * uxx + 2 * met.gxy * uxy + (met.gyy - 1) * uyy
    drift = met.bx * ux + met.by * uy
    coeff_ranks = [separable_rank(a) for a in (met.gxx - 1, met.gxy, met.gyy - 1, met.bx, met.by)]
    r_g = int(sum(coeff_ranks))
    rank_f, rank_fhat, rank_u = separable_rank(prob.f), separable_rank(prob.fhat), separable_rank(prob.u)
    fmax = float(np.abs(prob.fhat).max())
    gf = prob.flat_lu.solve(prob.f[grid.inner].ravel())
    u_true = prob.u[grid.inner].ravel()
    out: dict[str, Any] = {
        "bed": f"hill-{name}",
        "s": float(s),
        "height": prob.height,
        "rho": rho,
        "max_dev": met.max_dev,
        "errors": errors,
        "depth": {"1e-2": depth(errors, 1e-2), "1e-3": depth(errors, 1e-3), "1e-6": depth(errors, 1e-6)},
        "naive_error": float(np.linalg.norm(gf - u_true) / np.linalg.norm(u_true)),
        "rank_f": rank_f,
        "rank_fhat": rank_fhat,
        "rank_u": rank_u,
        "coeff_ranks": coeff_ranks,
        "r_g": r_g,
        "bound": rank_f + r_g * rank_u,
        "share_stretch": float(np.abs(stretch).max() / fmax) if fmax > 0 else 0.0,
        "share_drift": float(np.abs(drift).max() / fmax) if fmax > 0 else 0.0,
        "residuals_xy": chart_residuals(met.gxx, met.gxy, met.gyy),
    }
    if charts and s > 0:
        chart = isothermal_chart(grid, HILLS[name], s, met, prob.curved)
        (lam_pq, f_pq, lamf_pq, u_pq), box = resample(grid, chart.p, chart.q, [chart.lam, prob.f, chart.lam * prob.f, prob.u])
        fhat_orth, rank_orth, res_orth = orthogonal_chart(grid, chart, prob.f, prob.u, box)
        inside = lam_pq > 0
        r2_loglam, res_loglam = additive_fit(np.log(np.where(inside, lam_pq, 1.0)), inside)
        out["conformal"] = {
            "residuals_pq": chart.residuals,
            "rank_lambda": separable_rank(lam_pq),
            "rank_f_pq": separable_rank(f_pq),
            "rank_lambda_f": separable_rank(lamf_pq),
            "rank_u_pq": separable_rank(u_pq),
            "lambda_range": [float(chart.lam[2:-2, 2:-2].min()), float(chart.lam[2:-2, 2:-2].max())],
            "r2_loglambda_pq": r2_loglam,
            "res_loglambda_pq": res_loglam,
            "orth_chart": {"rank_fhat": rank_orth, "residuals": res_orth},
        }
    return out


__all__ = [
    "HILLS",
    "Grid",
    "HillProblem",
    "IsothermalChart",
    "Metric",
    "additive_fit",
    "analyse",
    "apply_curved",
    "born_errors",
    "chart_residuals",
    "curved_operator",
    "depth",
    "flat_operator",
    "full",
    "heater",
    "isothermal_chart",
    "lopsided_hill",
    "make_grid",
    "make_problem",
    "metric",
    "neumann_integrate",
    "orthogonal_chart",
    "resample",
    "round_hill",
    "second_derivatives",
    "separable_rank",
    "spectral_radius",
]
