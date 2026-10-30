"""SPD(n) as a test bed: the known curved manifold, in the coordinates IGL's data path uses.

The inputs are the log-Eig vectors ``v`` of ``P = exp(S(v))`` (``d = n(n+1)/2`` numbers, the
diagonal of ``S`` and its upper triangle scaled by ``sqrt 2`` so that ``|v| = |S|_F``). Two metrics
on the same inputs:

- **affine-invariant (AIRM)**: ``<A, B>_P = tr(P^-1 A P^-1 B)``, pulled back to ``v`` through the
  differential of ``v -> P``; SPD(n) is then a symmetric space of non-positive curvature, SPD(2) is
  isometric to ``R x H^2`` and conformally flat (Cotton tensor zero), SPD(n >= 3) is not (Weyl tensor
  non-zero). In the deck's terms the ambient metric is a modelling choice made explicit.
- **log-Euclidean**: the Euclidean metric on ``v``, flat; the do-no-harm twin.

Both are :class:`benchmarks.geometry.testbeds.manifolds.Bed` instances whose embedding is the
identity on ``v`` and whose ``ambient_metric`` carries the difference, so the curvature machinery,
the oracle-floor optimisation and the geometry report treat them like every other bed.
"""

from __future__ import annotations

import math

import torch
from torch.func import jacrev

from benchmarks.geometry.testbeds.manifolds import Bed

SQRT2 = math.sqrt(2.0)


def sym_from_vec(v: torch.Tensor, n: int) -> torch.Tensor:
    """Symmetric ``n x n`` matrix from its log-Eig vector (diagonal first, then the upper triangle over ``sqrt 2``)."""
    s = torch.zeros(n, n, dtype=v.dtype)
    idx = torch.arange(n)
    s = s + torch.diag(v[:n])
    iu = torch.triu_indices(n, n, offset=1)
    off = v[n:] / SQRT2
    s = s.index_put((iu[0], iu[1]), off).index_put((iu[1], iu[0]), off)
    del idx
    return s


def vec_from_sym(s: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`sym_from_vec`."""
    n = s.shape[-1]
    iu = torch.triu_indices(n, n, offset=1)
    return torch.cat([torch.diagonal(s, dim1=-2, dim2=-1), SQRT2 * s[..., iu[0], iu[1]]], dim=-1)


def spd_from_vec(v: torch.Tensor, n: int) -> torch.Tensor:
    """``P = exp(S(v))``."""
    return torch.linalg.matrix_exp(sym_from_vec(v, n))


def airm_metric_in_log_coords(n: int):
    """The AIRM tensor at ``v`` in log-Eig coordinates: ``J_phi^T (P^-1 (x) P^-1) J_phi`` with ``phi(v) = vec P``."""

    def metric(v: torch.Tensor) -> torch.Tensor:
        p = spd_from_vec(v, n)
        pinv = torch.linalg.inv(p)
        jphi = jacrev(lambda vv: spd_from_vec(vv, n).reshape(-1))(v)  # [n*n, d]
        h = torch.kron(pinv, pinv)
        return jphi.T @ h @ jphi

    return metric


def spd_bed(n: int, *, metric: str = "airm", box: float = 1.0) -> Bed:
    """SPD(n) in log-Eig coordinates under the affine-invariant (``"airm"``) or log-Euclidean (``"logeuclid"``) metric."""
    if metric not in ("airm", "logeuclid"):
        raise ValueError(f"metric must be 'airm' or 'logeuclid', got {metric!r}")
    d = n * (n + 1) // 2
    low = -box * torch.ones(d, dtype=torch.float64)
    high = box * torch.ones(d, dtype=torch.float64)
    return Bed(
        name=f"SPD{n}-{metric}",
        d=d,
        ambient=d,
        embedding=lambda v: v,
        low=low,
        high=high,
        conformal_chart=(lambda v: v) if metric == "logeuclid" else None,
        weyl_zero=metric == "logeuclid" or n == 2,
        notes=(
            "flat in log-Eig coordinates"
            if metric == "logeuclid"
            else "R x H^2, conformally flat (Cotton 0)"
            if n == 2
            else "symmetric space of rank >= 2: Weyl non-zero"
        ),
        ambient_metric=None if metric == "logeuclid" else airm_metric_in_log_coords(n),
        extra={"n": n, "metric": metric},
    )


def airm_distance2(v: torch.Tensor, v0: torch.Tensor, n: int) -> torch.Tensor:
    """Squared affine-invariant distance between ``P(v)`` (batched) and ``P(v0)``: a smooth intrinsic target."""
    p0 = spd_from_vec(v0, n)
    w, q = torch.linalg.eigh(p0)
    p0_isqrt = q @ torch.diag(w.rsqrt()) @ q.T
    out = []
    for vi in v:
        m = p0_isqrt @ spd_from_vec(vi, n) @ p0_isqrt
        ev = torch.linalg.eigvalsh(0.5 * (m + m.T))
        out.append((torch.log(ev) ** 2).sum())
    return torch.stack(out)


__all__ = [
    "airm_distance2",
    "airm_metric_in_log_coords",
    "spd_bed",
    "spd_from_vec",
    "sym_from_vec",
    "vec_from_sym",
    "airm_pairwise_distance",
]


def airm_pairwise_distance(v: torch.Tensor, n: int, *, chunk: int = 256) -> torch.Tensor:
    """Pairwise affine-invariant distances ``[N, N]`` between the matrices ``P(v_i)``, in float32, row chunks."""
    p_all = torch.stack([spd_from_vec(vi, n) for vi in v])  # [N, n, n]
    w, q = torch.linalg.eigh(p_all)
    p_isqrt = q @ torch.diag_embed(w.rsqrt()) @ q.transpose(-1, -2)  # [N, n, n]
    out = torch.empty(v.shape[0], v.shape[0], dtype=torch.float32)
    for i0 in range(0, v.shape[0], chunk):
        a = p_isqrt[i0 : i0 + chunk]  # [c, n, n]
        m = a[:, None] @ p_all[None] @ a[:, None]  # [c, N, n, n]
        ev = torch.linalg.eigvalsh(0.5 * (m + m.transpose(-1, -2))).clamp_min(1e-300)
        out[i0 : i0 + chunk] = (ev.log() ** 2).sum(-1).sqrt().to(torch.float32)
    return out
