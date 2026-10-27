"""G2: the conformally flat gauge on a grid: discretisation error and the sensitivity to the conformal factor (no training).

For ``g = lambda delta`` in ``d = 3, 4`` with an additive and a non-additive ``log lambda``:

- solve ``-Delta_g u = f`` with the curved operator; form ``v = lambda^s u``, ``s = (d - 2)/4``, and the
  potential ``V = s (Delta log lambda + s |grad log lambda|^2)``; the residual of ``(-Delta + V) v - lambda^{1+s} f``
  is the finite-difference error of an exact identity (it must shrink with the grid);
- the sensitivity: replace ``lambda`` by ``lambda (1 + delta eta)`` with ``eta`` a smooth unit field, solve the
  potential-form problem with the perturbed factor and un-gauge; the relative error of the recovered ``u``
  against ``delta`` is the amplification the deck predicts to be ``s`` to first order.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from benchmarks.geometry._harness import write_block_result
from benchmarks.geometry.blocks._cli import figure_dir, gate, parser
from benchmarks.geometry.testbeds import graphs

Array = NDArray[np.float64]
BLOCK = "g2_gauge"
POINTS = {3: 25, 4: 15}


def log_lambda(coords: list[Array], kind: str) -> Array:
    """``log lambda`` on the grid: ``additive`` (a sum of one-variable terms) or ``mixed`` (a product term)."""
    if kind == "additive":
        return sum(0.3 * np.sin(math.pi * c / 2) for c in coords)
    return 0.3 * np.sin(math.pi * coords[0] * coords[1] / 2) + 0.2 * np.cos(math.pi * coords[-1] / 2)


def potential(loglam: Array, h: float, s: float) -> Array:
    """``V = s (Delta log lambda + s |grad log lambda|^2)`` by central differences."""
    grads = [np.gradient(loglam, h, axis=i) for i in range(loglam.ndim)]
    lap = sum(np.gradient(gi, h, axis=i) for i, gi in enumerate(grads))
    return s * (lap + s * sum(g**2 for g in grads))


def solve_potential_form(v_pot: Array, rhs: Array, h: float) -> Array:
    """``(-Delta + V) v = rhs`` with zero Dirichlet data, matrix-free, GMRES with the flat solve as preconditioner."""
    d, n = rhs.ndim, rhs.shape[0]
    inner = tuple(slice(1, -1) for _ in range(d))
    shape = rhs[inner].shape
    size = rhs[inner].size

    def embed(v: Array) -> Array:
        full = np.zeros((n,) * d)
        full[inner] = v.reshape(shape)
        return full

    def mv(v: Array) -> Array:
        full = embed(v)
        return (graphs.apply_flat(full, h) + (v_pot * full)[inner]).ravel()

    op = spla.LinearOperator((size, size), matvec=mv, dtype=float)
    pre = spla.LinearOperator((size, size), matvec=lambda v: graphs.flat_solve(v.reshape(shape), h).ravel(), dtype=float)
    sol, info = spla.gmres(op, rhs[inner].ravel(), M=pre, rtol=1e-10, restart=50, maxiter=400)
    if info != 0:
        raise RuntimeError(f"potential-form solve did not converge (info={info})")
    return embed(np.asarray(sol))


def one_case(d: int, kind: str, *, n: int, deltas: list[float], seed: int = 0) -> dict[str, Any]:
    """Residual of the gauge identity and the amplification of a relative error in ``lambda``."""
    coords, h = graphs.make_coords(d, n)
    s = (d - 2) / 4
    loglam = log_lambda(coords, kind)
    lam = np.exp(loglam)
    ginv = [[(1.0 / lam if i == j else np.zeros_like(lam)) for j in range(d)] for i in range(d)]
    sqrtg = lam ** (d / 2)
    f = graphs.heater(coords, graphs.HEATER_WIDTH[d])
    prob = graphs.from_metric(f"cf-{kind}-d{d}", coords, h, ginv, sqrtg, f)
    u = prob.u
    inner = prob.inner
    v = lam**s * u
    v_pot = potential(loglam, h, s)
    rhs = lam ** (1 + s) * f
    lhs = np.zeros_like(u)
    lhs[inner] = graphs.apply_flat(v, h) + (v_pot * v)[inner]
    core = tuple(slice(2, -2) for _ in range(d))
    residual = float(np.linalg.norm((lhs - rhs)[core]) / np.linalg.norm(rhs[core]))
    # potential-form solve with the exact lambda: the round trip u -> v -> u
    v_exact = solve_potential_form(v_pot, rhs, h)
    u_round = v_exact / lam**s
    round_trip = float(np.linalg.norm((u_round - u)[inner]) / np.linalg.norm(u[inner]))
    rng = np.random.default_rng(seed)
    eta = sum(
        np.cos(math.pi * (sum(rng.integers(1, 3) * c for c in coords)) / 2 + rng.uniform(0, 2 * math.pi)) for _ in range(4)
    )
    eta = eta / np.sqrt(np.mean(eta**2))  # unit RMS: the ratio below is error per unit RMS relative error of lambda
    amp: list[dict[str, float]] = []
    for delta in deltas:
        lam_p = lam * (1 + delta * eta)
        loglam_p = np.log(lam_p)
        v_pot_p = potential(loglam_p, h, s)
        v_p = solve_potential_form(v_pot_p, lam_p ** (1 + s) * f, h)
        u_p = v_p / lam_p**s
        err = float(np.linalg.norm((u_p - u)[inner]) / np.linalg.norm(u[inner]))
        amp.append({"delta": delta, "error": err, "ratio": err / delta})
    return {
        "bed": f"cf-{kind}-d{d}",
        "d": d,
        "n": n,
        "kind": "analytic",
        "s": s,
        "identity_fd_residual": residual,
        "round_trip_error": round_trip,
        "rho_curved": graphs.spectral_radius(prob, iters=20, scale=graphs.best_scale(prob)),
        "amplification": amp,
        "lambda_range": [float(lam.min()), float(lam.max())],
    }


def figure(rows: list[dict[str, Any]], label: str | None) -> None:
    """Amplification ratio ``error / delta`` against ``d`` with the line ``s = (d - 2)/4``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for kind, marker in (("additive", "o"), ("mixed", "s")):
        pts = [r for r in rows if kind in r["bed"]]
        ax.scatter([r["d"] for r in pts], [r["amplification"][-1]["ratio"] for r in pts], marker=marker, label=f"{kind} log λ")
    ds = sorted({r["d"] for r in rows})
    ax.plot(ds, [(d - 2) / 4 for d in ds], "k--", lw=0.8, label="s = (d−2)/4")
    ax.set_xlabel("d")
    ax.set_ylabel("relative error of u per unit relative error of λ")
    ax.legend()
    fig.tight_layout()
    out = figure_dir(BLOCK, label) / "g2_gauge.png"
    fig.savefig(out, dpi=150)
    print("figure:", out)


def main() -> None:
    """Run the block."""
    args = parser(__doc__ or "").parse_args()
    gate(args)
    t0 = time.perf_counter()
    dims = [3] if args.smoke else [3, 4]
    deltas = [0.01] if args.smoke else [0.005, 0.01, 0.02]
    rows: list[dict[str, Any]] = []
    for d in dims:
        for kind in ("additive", "mixed"):
            n = 11 if args.smoke else POINTS[d]
            r = one_case(d, kind, n=n, deltas=deltas)
            rows.append(r)
            write_block_result(BLOCK, r["bed"], "analytic", 0, r, label=args.label)
            a = r["amplification"][-1]
            print(
                f"{r['bed']:16s} n={n} s={r['s']:.2f} fd residual {r['identity_fd_residual']:.2e} "
                f"round trip {r['round_trip_error']:.2e} "
                f"rho {r['rho_curved']:.3f} | delta {a['delta']:g} -> error {a['error']:.4f} (ratio {a['ratio']:.2f})",
                flush=True,
            )
    figure(rows, args.label)
    print(f"G2 done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
