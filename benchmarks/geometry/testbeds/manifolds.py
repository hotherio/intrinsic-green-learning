"""Embedded manifolds with analytic geometry: the rows of the manifold matrix other than hills, graphs and SPD.

Every bed is an embedding ``t -> x`` of a parameter patch into ``R^D`` written as a torch function,
so that the tangent basis, the induced metric, the curvature and the Weyl and Cotton tensors all
follow by nested automatic differentiation in double precision. Where the true conformal chart
is known it is exposed (stereographic coordinates on spheres, ``r = e^t`` on cylinders, block
charts on products, the angles on a flat torus); where it is not (ellipsoids) ``conformal_chart``
is ``None`` and the oracle floor comes from the chart optimisation block.

Sampling is uniform in the parameters of the patch, not in the area of the manifold; the patches
stay away from coordinate singularities.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.func import jacrev

Embedding = Callable[[torch.Tensor], torch.Tensor]


def stereographic(xi: torch.Tensor) -> torch.Tensor:
    """Inverse stereographic map ``R^p -> S^p subset R^{p+1}``, a conformal chart with ``lambda = 4 / (1 + |xi|^2)^2``."""
    r2 = (xi**2).sum()
    return torch.cat([2 * xi, (r2 - 1).reshape(1)]) / (r2 + 1)


@dataclass(frozen=True, slots=True)
class Bed:
    """An embedded manifold: name, intrinsic and ambient dimensions, the embedding and its chart conventions."""

    name: str
    d: int
    ambient: int
    embedding: Embedding
    low: torch.Tensor
    high: torch.Tensor
    conformal_chart: Callable[[torch.Tensor], torch.Tensor] | None = None
    blocks: tuple[int, ...] = ()
    weyl_zero: bool | None = None
    notes: str = ""
    ambient_metric: Callable[[torch.Tensor], torch.Tensor] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def sample(self, n: int, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """``n`` parameter points uniform in the patch and their images in ``R^D`` (double precision)."""
        gen = torch.Generator().manual_seed(seed)
        t = self.low + (self.high - self.low) * torch.rand(n, self.d, generator=gen, dtype=torch.float64)
        x = torch.stack([self.embedding(ti) for ti in t])
        return t, x

    def tangent_basis(self, t: torch.Tensor) -> torch.Tensor:
        """Jacobian of the embedding at ``t``: columns span the tangent space, ``[D, d]``."""
        return jacrev(self.embedding)(t)

    def _ambient(self, t: torch.Tensor) -> torch.Tensor:
        if self.ambient_metric is None:
            return torch.eye(self.ambient, dtype=torch.float64)
        return self.ambient_metric(self.embedding(t))

    def metric(self, t: torch.Tensor) -> torch.Tensor:
        """Induced metric ``J^T H J`` in the parameters, ``[d, d]`` (``H`` the ambient metric, the identity by default)."""
        j = self.tangent_basis(t)
        return j.T @ self._ambient(t) @ j

    def tangent_projector(self, t: torch.Tensor) -> torch.Tensor:
        """``H``-orthogonal projector of ``R^D`` onto the tangent space at ``t``, ``[D, D]``."""
        j = self.tangent_basis(t)
        h = self._ambient(t)
        return j @ torch.linalg.solve(j.T @ h @ j, j.T @ h)

    def metric_inv(self, t: torch.Tensor) -> torch.Tensor:
        """``M(x) = J (J^T H J)^{-1} J^T``: what the penalty's ``metric_inv`` receives (``A = J_enc M J_enc^T``).

        Equal to the tangent projector for the Euclidean ambient metric and to ``H^{-1}`` when the
        bed is the whole input space under the metric ``H`` (SPD under the affine-invariant metric).
        """
        j = self.tangent_basis(t)
        return j @ torch.linalg.solve(j.T @ self._ambient(t) @ j, j.T)

    def metric_inv_batch(self, t: torch.Tensor) -> torch.Tensor:
        """``metric_inv`` for a batch of parameter points, ``[n, D, D]`` (a loop; the beds are small)."""
        return torch.stack([self.metric_inv(ti) for ti in t])


# --- curvature by nested autograd ----------------------------------------------------------------


def _christoffel(bed: Bed, t: torch.Tensor) -> torch.Tensor:
    g = bed.metric(t)
    dg = jacrev(bed.metric)(t)  # [i, j, k] = d_k g_ij
    ginv = torch.linalg.inv(g)
    # Gamma^k_ij = 1/2 g^{kl} (d_i g_jl + d_j g_il - d_l g_ij)
    term = dg.permute(2, 0, 1) + dg.permute(0, 2, 1) - dg  # [i, j, l]: d_i g_jl + d_j g_il - d_l g_ij
    return 0.5 * torch.einsum("kl,ijl->kij", ginv, term)


def curvature(bed: Bed, t: torch.Tensor) -> dict[str, torch.Tensor]:
    """Riemann (all indices down), Ricci, scalar curvature, Weyl and Schouten tensors at ``t``."""
    d = bed.d
    g = bed.metric(t)
    ginv = torch.linalg.inv(g)
    gam = _christoffel(bed, t)
    dgam = jacrev(lambda tt: _christoffel(bed, tt))(t)  # [k, i, j, m] = d_m Gamma^k_ij
    # R^r_{s m n} = d_m G^r_{n s} - d_n G^r_{m s} + G^r_{m l} G^l_{n s} - G^r_{n l} G^l_{m s}
    r_up = (
        dgam.permute(0, 2, 3, 1)
        - dgam.permute(0, 2, 1, 3)
        + torch.einsum("rml,lns->rsmn", gam, gam)
        - torch.einsum("rnl,lms->rsmn", gam, gam)
    )
    riem = torch.einsum("ra,asmn->rsmn", g, r_up)
    ric = torch.einsum("rsrn->sn", r_up)
    scal = torch.einsum("sn,sn->", ginv, ric)
    out = {"g": g, "ginv": ginv, "riem": riem, "ric": ric, "scal": scal}
    if d >= 3:
        schouten = (ric - scal * g / (2 * (d - 1))) / (d - 2)
        kn = (
            torch.einsum("ik,jl->ijkl", g, schouten)
            - torch.einsum("il,jk->ijkl", g, schouten)
            + torch.einsum("jl,ik->ijkl", g, schouten)
            - torch.einsum("jk,il->ijkl", g, schouten)
        )
        out["schouten"] = schouten
        out["weyl"] = riem - kn
    return out


def _norm2_4(t4: torch.Tensor, ginv: torch.Tensor) -> float:
    """``T_ijkl T^ijkl``: the indices are raised one at a time, so memory stays ``O(d^4)`` (a single six-operand einsum
    let torch build an eight-index intermediate: 34 GB at ``d = 16``, which took the machine down on 2026-10-26)."""
    up = torch.einsum("ia,ajkl->ijkl", ginv, t4)
    up = torch.einsum("jb,ibkl->ijkl", ginv, up)
    up = torch.einsum("kc,ijcl->ijkl", ginv, up)
    up = torch.einsum("ld,ijkd->ijkl", ginv, up)
    return float((t4 * up).sum())


def weyl_share(bed: Bed, t: torch.Tensor) -> dict[str, float]:
    """``|W|^2 / |Riem|^2`` at ``t``, chart-free and scale-free; zero means a conformal chart exists locally (``d >= 4``)."""
    c = curvature(bed, t)
    riem2 = _norm2_4(c["riem"], c["ginv"])
    if bed.d < 4:
        return {"weyl2": 0.0, "riem2": riem2, "share": 0.0}
    w2 = _norm2_4(c["weyl"], c["ginv"])
    return {"weyl2": w2, "riem2": riem2, "share": w2 / riem2 if riem2 > 1e-14 else 0.0}


def cotton_share(bed: Bed, t: torch.Tensor) -> dict[str, float]:
    """Dimensionless size of the Cotton tensor in ``d = 3``: ``|C|^2 / |Riem|^3``; zero means conformally flat."""
    if bed.d != 3:
        raise ValueError("the Cotton tensor is the obstruction in d = 3 only")
    c = curvature(bed, t)
    gam = _christoffel(bed, t)

    def schouten_at(tt: torch.Tensor) -> torch.Tensor:
        return curvature(bed, tt)["schouten"]

    dp = jacrev(schouten_at)(t)  # [i, j, k] = d_k P_ij
    p = c["schouten"]
    nabla = dp - torch.einsum("lki,lj->ijk", gam, p) - torch.einsum("lkj,il->ijk", gam, p)  # nabla_k P_ij
    cotton = nabla - nabla.permute(0, 2, 1)  # C_ijk = nabla_k P_ij - nabla_j P_ik
    ginv = c["ginv"]
    up3 = torch.einsum("ia,ajk->ijk", ginv, cotton)
    up3 = torch.einsum("jb,ibk->ijk", ginv, up3)
    up3 = torch.einsum("kc,ijc->ijk", ginv, up3)
    c2 = float((cotton * up3).sum())
    riem2 = _norm2_4(c["riem"], ginv)
    return {"cotton2": c2, "riem2": riem2, "share": c2 / riem2**1.5 if riem2 > 1e-14 else 0.0}


def obstruction(bed: Bed, *, n_points: int = 8, seed: int = 1) -> dict[str, Any]:
    """Mean and max of the obstruction share over random points of the patch: Weyl in ``d >= 4``, Cotton in ``d = 3``."""
    t, _ = bed.sample(n_points, seed=seed)
    if bed.d >= 4:
        parts = [weyl_share(bed, ti) for ti in t]
        num, den = sum(p["weyl2"] for p in parts), sum(p["riem2"] for p in parts)
        shares = [p["share"] for p in parts]
        kind = "weyl"
    elif bed.d == 3:
        parts = [cotton_share(bed, ti) for ti in t]
        num, den = sum(p["cotton2"] for p in parts), sum(p["riem2"] ** 1.5 for p in parts)
        shares = [p["share"] for p in parts]
        kind = "cotton"
    else:
        num, den, shares, kind = 0.0, 1.0, [0.0], "none"
    # the ratio of sums is robust where the curvature vanishes at a point (a per-point ratio can blow up there)
    return {
        "kind": kind,
        "mean": num / den if den > 1e-14 else 0.0,
        "max": max(shares),
        "mean_of_ratios": sum(shares) / len(shares),
    }


# --- the beds -------------------------------------------------------------------------------------


def _box(d: int, r: float) -> tuple[torch.Tensor, torch.Tensor]:
    return -r * torch.ones(d, dtype=torch.float64), r * torch.ones(d, dtype=torch.float64)


def sphere_cap(d: int, *, radius: float = 0.8, axes: tuple[float, ...] | None = None) -> Bed:
    """A cap of ``S^d`` in stereographic coordinates (the conformal chart); box corners at ``|xi| = radius``."""
    scale = None if axes is None else torch.tensor(axes, dtype=torch.float64)

    def emb(xi: torch.Tensor) -> torch.Tensor:
        p = stereographic(xi)
        return p if scale is None else p * scale

    low, high = _box(d, radius / math.sqrt(d))  # the box's corners stay inside |xi| = radius (< 1: one hemisphere)
    tag = "" if axes is None else "-axes" + "x".join(f"{a:g}" for a in axes)
    return Bed(
        name=f"S{d}cap{tag}",
        d=d,
        ambient=d + 1,
        embedding=emb,
        low=low,
        high=high,
        conformal_chart=None if axes is not None and len(set(axes)) > 1 else (lambda xi: xi),
        weyl_zero=True if axes is None else None,
        notes="stereographic chart is conformal, lambda = 4/(1+|xi|^2)^2"
        if axes is None
        else "ellipsoid: no conformal chart in general",
    )


def ellipsoid_family(steps: int = 5) -> list[Bed]:
    """``S^4`` with axes interpolating from ``(1,1,1,1,1)`` to ``(1,1,1,2,3)``: the Weyl share from 0 to about 0.1."""
    beds: list[Bed] = []
    for k in range(steps):
        a = k / (steps - 1)
        beds.append(sphere_cap(4, axes=(1.0, 1.0, 1.0, 1.0 + a, 1.0 + 2 * a)))
    return beds


def cylinder(d: int, *, radius: float = 0.8, length: float = 0.8) -> Bed:
    """``S^{d-1} x R`` in ``R^{d+1}``: parameters ``(xi, t)``, ``xi`` stereographic on the sphere, ``t`` the height."""

    def emb(p: torch.Tensor) -> torch.Tensor:
        return torch.cat([stereographic(p[:-1]), p[-1:]])

    def conformal(p: torch.Tensor) -> torch.Tensor:
        # r = e^t times the unit sphere point: Cartesian coordinates of R^d \ {0}, where g = e^{-2t} delta
        return torch.exp(p[..., -1:]) * stereographic_batch(p[..., :-1])

    r = radius / math.sqrt(d - 1)
    low = torch.tensor([-r] * (d - 1) + [-r], dtype=torch.float64)
    high = torch.tensor([r] * (d - 1) + [r], dtype=torch.float64)
    del length
    return Bed(
        name=f"S{d - 1}xR",
        d=d,
        ambient=d + 1,
        embedding=emb,
        low=low,
        high=high,
        conformal_chart=conformal,
        weyl_zero=True,
        notes="conformally flat; the conformal factor in the Cartesian chart is radial, not additive",
    )


def stereographic_batch(xi: torch.Tensor) -> torch.Tensor:
    """Batched inverse stereographic map, ``[..., p] -> [..., p + 1]``."""
    r2 = (xi**2).sum(-1, keepdim=True)
    return torch.cat([2 * xi, r2 - 1], dim=-1) / (r2 + 1)


def product_sphere_flat(p: int, q: int, *, radius: float = 0.8, length: float = 0.8) -> Bed:
    """``S^p x R^q`` in ``R^{p+1+q}``; block chart ``(xi, y)``; Weyl share ``1/3`` for ``p = q = 2``, never zero."""

    def emb(t: torch.Tensor) -> torch.Tensor:
        return torch.cat([stereographic(t[:p]), t[p:]])

    r = radius / math.sqrt(p)
    low = torch.tensor([-r] * p + [-r] * q, dtype=torch.float64)
    high = torch.tensor([r] * p + [r] * q, dtype=torch.float64)
    del length
    return Bed(
        name=f"S{p}xR{q}",
        d=p + q,
        ambient=p + 1 + q,
        embedding=emb,
        low=low,
        high=high,
        conformal_chart=None,
        blocks=(p, q),
        weyl_zero=False,
        notes="product of a sphere and a flat factor: block-conformal, not conformally flat",
    )


def product_spheres(p: int, q: int, *, radius: float = 0.8) -> Bed:
    """``S^p x S^q`` in ``R^{p+q+2}``; block chart; Weyl share ``2/3`` for ``p = q = 2``."""

    def emb(t: torch.Tensor) -> torch.Tensor:
        return torch.cat([stereographic(t[:p]), stereographic(t[p:])])

    low, high = _box(p + q, radius / math.sqrt(max(p, q)))
    return Bed(
        name=f"S{p}xS{q}",
        d=p + q,
        ambient=p + q + 2,
        embedding=emb,
        low=low,
        high=high,
        conformal_chart=None,
        blocks=(p, q),
        weyl_zero=False,
        notes="product of two spheres: block-conformal only",
    )


def flat_torus(d: int, *, half_width: float = 2.5) -> Bed:
    """A patch of the flat torus ``T^d subset R^{2d}`` (angles away from the seam): the angles are an isometric chart."""

    def emb(theta: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1).reshape(-1)

    low, high = _box(d, half_width)
    return Bed(
        name=f"T{d}",
        d=d,
        ambient=2 * d,
        embedding=emb,
        low=low,
        high=high,
        conformal_chart=lambda th: th,
        weyl_zero=True,
        notes="flat: the chart penalties have a zero floor at the isometric chart",
    )


def hill_surface(which: str = "lop") -> Bed:
    """One of the two hills as an embedded surface ``(x, y, s h(x, y)) subset R^3`` at full height."""
    import numpy as np

    from benchmarks.geometry.testbeds import hills

    hill = hills.HILLS[which]

    def emb(t: torch.Tensor) -> torch.Tensor:
        x, y = t[0], t[1]
        if which == "round":
            z = 0.6 * torch.exp(-(x**2 + y**2) / (2 * 0.35**2))
        else:
            c, s = math.cos(math.radians(35)), math.sin(math.radians(35))
            u, w = c * (x - 0.35) + s * (y - 0.2), -s * (x - 0.35) + c * (y - 0.2)
            z = 0.6 * torch.exp(-(u**2 / (2 * 0.32**2) + w**2 / (2 * 0.16**2)))
        return torch.stack([x, y, z])

    low, high = _box(2, 1.0)
    cache: dict[str, Any] = {}

    def conformal(t: torch.Tensor) -> torch.Tensor:
        # the isothermal chart (p, q) of testbeds.hills, interpolated from its grid (cubic splines, so the
        # Jacobian used by the oracle floor is smooth); computed once per bed
        if "interp" not in cache:
            from scipy.interpolate import RectBivariateSpline

            # a 145-point grid: on the 49-point grid of the deck the interpolated chart is conformal only to
            # 6e-2 at the lopsided peak, which would be read as a floor of the chart instead of its resolution
            prob = hills.make_problem(which, 1.0, m=145)
            chart = hills.isothermal_chart(prob.grid, hill, 1.0, prob.met, prob.curved)
            x = prob.grid.x
            cache["interp"] = (RectBivariateSpline(x, x, chart.p), RectBivariateSpline(x, x, chart.q))
        sp_p, sp_q = cache["interp"]
        tt = t.detach().cpu().numpy().reshape(-1, 2)
        out = np.stack([sp_p.ev(tt[:, 0], tt[:, 1]), sp_q.ev(tt[:, 0], tt[:, 1])], axis=-1)
        return torch.from_numpy(out).to(t.dtype).reshape(t.shape)

    return Bed(
        name=f"hill-{which}",
        d=2,
        ambient=3,
        embedding=emb,
        low=low,
        high=high,
        conformal_chart=conformal,
        weyl_zero=True,
        notes="d = 2: the isothermal chart of testbeds.hills, interpolated (not differentiable by autograd)",
    )


def product_of_hills() -> Bed:
    """The round hill times the lopsided hill, a 4-manifold in ``R^6`` with a block isothermal chart."""
    a, b = hill_surface("round"), hill_surface("lop")

    def emb(t: torch.Tensor) -> torch.Tensor:
        return torch.cat([a.embedding(t[:2]), b.embedding(t[2:])])

    low, high = _box(4, 1.0)
    return Bed(
        name="hillxhill",
        d=4,
        ambient=6,
        embedding=emb,
        low=low,
        high=high,
        conformal_chart=None,
        blocks=(2, 2),
        weyl_zero=False,
        notes="product of two curved surfaces: not conformally flat; block isothermal chart exists",
    )


@dataclass(frozen=True, slots=True)
class HyperbolicHalfSpace:
    """``H^d`` in the half-space model, chart only: ``g = delta / xi_d^2``, ``log lambda = -2 log xi_d`` is additive."""

    d: int = 3
    low: float = 0.5
    high: float = 2.0

    def sample(self, n: int, *, seed: int = 0) -> torch.Tensor:
        """``n`` chart points with ``xi_d`` in ``[low, high]`` and the other coordinates in ``[-1, 1]``."""
        gen = torch.Generator().manual_seed(seed)
        xi = 2 * torch.rand(n, self.d, generator=gen, dtype=torch.float64) - 1
        xi[:, -1] = self.low + (self.high - self.low) * torch.rand(n, generator=gen, dtype=torch.float64)
        return xi

    def lam(self, xi: torch.Tensor) -> torch.Tensor:
        """The conformal factor ``1 / xi_d^2``."""
        return 1.0 / xi[..., -1] ** 2

    def potential(self, xi: torch.Tensor) -> torch.Tensor:
        """``V = s (Delta log lambda + s |grad log lambda|^2)`` with ``s = (d - 2)/4``: ``(2s + 4s^2) / xi_d^2``."""
        s = (self.d - 2) / 4
        return (2 * s + 4 * s * s) / xi[..., -1] ** 2


def catalogue() -> dict[str, Bed]:
    """The analytic beds of the manifold matrix, by name."""
    beds = [
        sphere_cap(2),
        sphere_cap(4),
        sphere_cap(8),
        sphere_cap(16),
        *ellipsoid_family(),
        cylinder(4),
        cylinder(8),
        product_sphere_flat(2, 2),
        product_sphere_flat(4, 4),
        product_spheres(2, 2),
        product_spheres(4, 4),
        flat_torus(2),
        flat_torus(4),
        flat_torus(8),
        hill_surface("round"),
        hill_surface("lop"),
        product_of_hills(),
    ]
    return {b.name: b for b in beds}


__all__ = [
    "Bed",
    "HyperbolicHalfSpace",
    "catalogue",
    "cotton_share",
    "curvature",
    "cylinder",
    "ellipsoid_family",
    "flat_torus",
    "hill_surface",
    "obstruction",
    "product_of_hills",
    "product_sphere_flat",
    "product_spheres",
    "sphere_cap",
    "stereographic",
    "stereographic_batch",
    "weyl_share",
]


def graph_hypersurface(d: int, bump: str = "radial", *, s: float = 1.0) -> Bed:
    """The graph hypersurface ``x -> (x, s h(x))`` of :mod:`benchmarks.geometry.testbeds.graphs` as an embedded bed (torch)."""

    def radial(x: torch.Tensor) -> torch.Tensor:
        return 0.6 * torch.exp(-(x**2).sum() / (2 * 0.35**2))

    def generic(x: torch.Tensor) -> torch.Tensor:
        c, sn = math.cos(math.radians(35)), math.sin(math.radians(35))
        x0, x1 = x[0] - 0.35, x[1] - 0.2
        u, w = c * x0 + sn * x1, -sn * x0 + c * x1
        rest = (x[2:] ** 2).sum() if d > 2 else torch.zeros((), dtype=x.dtype)
        z1 = 0.6 * torch.exp(-(u**2 / (2 * 0.32**2) + w**2 / (2 * 0.16**2) + rest / (2 * 0.4**2)))
        signs = torch.tensor([0.4 * (-1) ** k for k in range(d)], dtype=x.dtype)
        z2 = 0.25 * torch.exp(-((x + signs) ** 2).sum() / (2 * 0.25**2))
        return z1 + z2

    height = radial if bump == "radial" else generic

    def emb(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, (s * height(x)).reshape(1)])

    low, high = _box(d, 1.0)
    return Bed(
        name=f"graph-{bump}-d{d}",
        d=d,
        ambient=d + 1,
        embedding=emb,
        low=low,
        high=high,
        conformal_chart=None,
        weyl_zero=bump == "radial",
        notes="hypersurface of revolution: conformally flat" if bump == "radial" else "generic hypersurface",
    )


__all__.append("graph_hypersurface")
