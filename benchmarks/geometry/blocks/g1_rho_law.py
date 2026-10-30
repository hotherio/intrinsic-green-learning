"""G1: the Born contraction across dimensions, and how ``rho`` grows with ``d`` (no training).

Part A, hills and graph hypersurfaces in ``d = 2, 3, 4``: ``rho(G K)`` by power iteration against
``max |g^-1 - I|``, the Born depth to ``1e-3`` against ``n(eps) = log(1/eps) / log(1/rho)``, and
the error of the flat solve that ignores the hill.

Part B, the ``sqrt d`` law: random charts of the flat torus in ``d = 2, 4, 8, 16`` with a fixed
per-entry distortion ``eps`` of three kinds (diagonal, incoherent, coherent rank-one shear);
``max |A - I|`` in operator norm against ``d``. The deck's count says ``sqrt d``; the prediction
to falsify is that only the incoherent kind grows that way.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from benchmarks.geometry._harness import write_block_result
from benchmarks.geometry.blocks._cli import figure_dir, gate, parser
from benchmarks.geometry.testbeds import graphs, hills

BLOCK = "g1_rho_law"


def predicted_depth(rho: float, eps: float, naive: float = 1.0) -> float:
    """``n(eps) = log(naive/eps) / log(1/rho)`` for ``rho < 1``: after ``n`` Born steps the error is ``rho^n`` x naive."""
    if not 0 < rho < 1:
        return float("inf")
    return math.log(naive / eps) / math.log(1 / rho) if naive > eps else 0.0


def part_a(*, smoke: bool, label: str | None) -> list[dict[str, Any]]:
    """Hills and graphs: ``rho`` against ``max_dev``, measured against predicted depth, naive error."""
    rows: list[dict[str, Any]] = []
    heights = [0.5, 1.0] if smoke else [0.2, 0.4, 0.6, 0.8, 1.0]
    for name in hills.HILLS:
        for s in heights:
            r = hills.analyse(name, s, charts=False, n_born=10 if smoke else 40)
            rows.append(_row(r["bed"], 2, s, r))
    dims = [3] if smoke else [3, 4]
    for d in dims:
        for name in graphs.BUMPS:
            for s in heights:
                r = graphs.analyse(name, d, s, n=13 if smoke else None, n_born=10 if smoke else 40)
                rs = graphs.analyse(name, d, s, n=13 if smoke else None, n_born=2, scaled=True)
                r["scale"], r["rho_scaled"], r["max_dev_scaled"] = rs["scale"], rs["rho"], rs["max_dev"]
                rows.append(_row(r["bed"], d, s, r))
    for row in rows:
        write_block_result(BLOCK, row["bed"], f"s{row['s']:g}", 0, row, label=label)
        print(
            f"{row['bed']:22s} s={row['s']:.1f} rho {row['rho']:.3f} max_dev {row['max_dev']:.3f} "
            f"ratio {row['rho_over_maxdev']:.2f} depth meas {row['depth_1e-3']} pred {row['depth_pred_1e-3']:.1f} "
            f"naive {row['naive_error']:.3f}",
            flush=True,
        )
    return rows


def _row(bed: str, d: int, s: float, r: dict[str, Any]) -> dict[str, Any]:
    return {
        "bed": bed,
        "d": d,
        "s": s,
        "kind": "analytic",
        "rho": r["rho"],
        "max_dev": r["max_dev"],
        "rho_over_maxdev": r["rho"] / r["max_dev"] if r["max_dev"] > 0 else float("nan"),
        "scale": r.get("scale"),
        "rho_scaled": r.get("rho_scaled"),
        "max_dev_scaled": r.get("max_dev_scaled"),
        "rho_over_maxdev_scaled": (r["rho_scaled"] / r["max_dev_scaled"]) if r.get("max_dev_scaled") else float("nan"),
        "depth_1e-3": r["depth"]["1e-3"],
        "depth_pred_1e-3": predicted_depth(r["rho"], 1e-3, r["naive_error"]),
        "depth_pred_1e-3_no_naive": predicted_depth(r["rho"], 1e-3),
        "naive_error": r["naive_error"],
        "share_stretch": r["share_stretch"],
        "share_drift": r["share_drift"],
    }


def torus_chart_distortion(
    d: int, kind: str, eps: float, *, n_points: int = 2000, n_modes: int = 12, seed: int = 0
) -> dict[str, float]:
    """``max |A - I|`` (operator norm, after the best global scale) for a random chart of the flat torus.

    The chart Jacobian is ``J = I + eps E(theta)`` with ``E`` built from random Fourier fields of unit
    per-entry standard deviation: ``diagonal`` (``E`` diagonal), ``incoherent`` (every entry an
    independent field: its own frequencies, phases and amplitudes), ``coherent`` (``E = a b^T phi(theta)``,
    a rank-one shear). ``A = J J^T``.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-math.pi, math.pi, size=(n_points, d))

    def field() -> np.ndarray:  # [n_points], an independent random Fourier field of unit variance over theta
        freqs = rng.integers(-2, 3, size=(n_modes, d)).astype(float)
        phases = rng.uniform(0, 2 * math.pi, size=n_modes)
        amp = rng.standard_normal(n_modes)
        amp /= np.linalg.norm(amp) / math.sqrt(n_modes)
        return np.sqrt(2.0 / n_modes) * np.cos(theta @ freqs.T + phases) @ amp

    eye = np.eye(d)
    if kind == "diagonal":
        e = np.zeros((n_points, d, d))
        for i in range(d):
            e[:, i, i] = field()
    elif kind == "incoherent":
        e = np.zeros((n_points, d, d))
        for i in range(d):
            for j in range(d):
                e[:, i, j] = field()
    elif kind == "coherent":
        a = rng.standard_normal(d)
        b = rng.standard_normal(d)
        a /= np.linalg.norm(a) / math.sqrt(d)
        b /= np.linalg.norm(b) / math.sqrt(d)
        phi = field()
        e = phi[:, None, None] * np.outer(a, b)[None]
    else:
        raise ValueError(kind)
    jac = eye[None] + eps * e
    a_mat = jac @ np.transpose(jac, (0, 2, 1))
    tr = np.trace(a_mat, axis1=1, axis2=2).mean() / d  # best global scale
    dev = a_mat / tr - eye[None]
    norms = np.linalg.norm(dev, ord=2, axis=(1, 2))
    return {"max_dev": float(norms.max()), "mean_dev": float(norms.mean()), "scale": float(tr)}


def part_b(*, smoke: bool, label: str | None) -> list[dict[str, Any]]:
    """The ``sqrt d`` law on torus charts."""
    rows: list[dict[str, Any]] = []
    dims = [2, 4, 8] if smoke else [2, 4, 8, 16]
    eps = 0.05
    for kind in ("diagonal", "incoherent", "coherent"):
        base = None
        for d in dims:
            r = torus_chart_distortion(d, kind, eps, n_points=500 if smoke else 4000)
            base = base or (r["max_dev"], r["mean_dev"])
            row = {
                "bed": f"T{d}-chart",
                "d": d,
                "kind": "analytic",
                "distortion": kind,
                "eps": eps,
                **r,
                "growth_vs_d2": r["max_dev"] / base[0],
                "growth_mean_vs_d2": r["mean_dev"] / base[1],
            }
            rows.append(row)
            write_block_result(BLOCK, row["bed"], kind, 0, row, label=label)
            print(
                f"torus chart d={d:2d} {kind:10s} mean|A-I| {r['mean_dev']:.4f} max {r['max_dev']:.4f} "
                f"growth x{row['growth_vs_d2']:.2f}",
                flush=True,
            )
    return rows


def figure(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], label: str | None) -> None:
    """Two panels: measured vs predicted depth across d, and ``rho`` growth with d per distortion kind."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for d, marker in ((2, "o"), (3, "s"), (4, "^")):
        pts = [r for r in rows_a if r["d"] == d and r["depth_1e-3"] is not None]
        ax1.scatter([r["depth_pred_1e-3"] for r in pts], [r["depth_1e-3"] for r in pts], marker=marker, label=f"d = {d}")
    lim = max([r["depth_1e-3"] or 0 for r in rows_a] + [1]) * 1.1
    ax1.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax1.set_xlabel("predicted depth  log(1/ε)/log(1/ρ)")
    ax1.set_ylabel("measured Born depth to 1e-3")
    ax1.legend()
    for kind, ls in (("diagonal", ":"), ("incoherent", "-"), ("coherent", "--")):
        pts = sorted([r for r in rows_b if r["distortion"] == kind], key=lambda r: r["d"])
        ax2.plot([r["d"] for r in pts], [r["mean_dev"] for r in pts], ls, marker="o", label=kind)
    ds = sorted({r["d"] for r in rows_b})
    ref = next(r["mean_dev"] for r in rows_b if r["distortion"] == "incoherent" and r["d"] == ds[0])
    ax2.plot(ds, [ref * math.sqrt(d / ds[0]) for d in ds], "k:", lw=0.8, label="√d")
    ax2.plot(ds, [ref * d / ds[0] for d in ds], "k-.", lw=0.8, label="d")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("d")
    ax2.set_ylabel("mean ‖A − I‖ at fixed per-entry ε")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    out = figure_dir(BLOCK, label) / "g1_rho_law.png"
    fig.savefig(out, dpi=150)
    print("figure:", out)


def main() -> None:
    """Run the block."""
    args = parser(__doc__ or "").parse_args()
    gate(args)
    t0 = time.perf_counter()
    rows_a = part_a(smoke=args.smoke, label=args.label)
    rows_b = part_b(smoke=args.smoke, label=args.label)
    figure(rows_a, rows_b, args.label)
    print(f"G1 done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
