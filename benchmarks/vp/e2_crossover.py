"""E2 — inner-solver scalability crossover.

Pure microbenchmark of the multi-RHS ridge solve
``min_w ||phi w - y||^2 + l2_eff ||w||^2`` across dictionary sizes p, row
counts n, and output width D, over the solver zoo. No training.

Pre-registered predictions (from the VP solver discussion):

- P3  Flops crossover: direct stays cheaper than (matrix-free) CG until
      p in roughly [15k, 25k] at n=4096, D=768; pass if the empirical
      crossover p* lands in [8k, 40k] (factor-2 band).
- P4  Memory: direct peak memory ~ p^2 (Gram) or (n+p)p (stacked QR);
      extrapolated 48 GB wall in [30k, 80k].
- P5  Iteration scaling: CG iterations within x2 of sqrt(kappa)*log(1/eps)
      on controlled-spectrum problems, while GD epochs scale ~kappa.
- P6  Warm-started CG (x0 from a perturbed prior solution, emulating the
      outer loop) needs >=3x fewer iterations than cold at matched tol.
- P11 Nystrom-PCG iteration counts vary by less than x2 across the kappa
      grid (near-kappa-independence).

Run with::

    python -m benchmarks.vp.e2_crossover [--smoke] [--skip-real-phi]

Outputs ``results/benchmarks/vp/e2_crossover/<sha>/result.json``.
"""

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from math import log, sqrt
from typing import Any

import numpy as np
import torch

from benchmarks.vp._harness import Timing, machine_state, set_seed, time_median, write_result
from benchmarks.vp.solvers import (
    SolveResult,
    block_cg,
    cholesky_normal,
    direct_lstsq,
    effective_l2,
    nystrom_pcg,
    sgd_to_tol,
)

D = 768
L2 = 1e-3
CG_TOL = 1e-5
CG_MAX_ITER = 600  # cost model uses measured time-per-iteration, so a cap is safe
SVRG_MAX_P = 4096
ITERATIVE_SINGLE_REPEAT_P = 16384  # iterative arms at large p: 1 timed repeat keeps the sweep in budget
WARM_START_SCALES = [1e-3, 1e-2]  # relative phi perturbation: ~one outer step, ~one outer epoch

FULL_P_GRID = [256, 1024, 2048, 4096, 8192, 16384, 24576, 32768]
FULL_N_GRID = [4096, 16384]
SMOKE_P_GRID = [256, 1024, 2048]
SMOKE_N_GRID = [1024]
KAPPA_GRID = [1e2, 1e3, 1e4, 1e6]
GD_KAPPA_MAX = 1e4


def controlled_problem(n: int, p: int, d: int, *, kappa: float | None, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A ridge problem with an iid-gaussian or prescribed-spectrum phi."""
    generator = torch.Generator().manual_seed(seed)
    if kappa is None:
        phi = torch.randn(n, p, generator=generator)
    else:
        r = min(n, p)
        u, _ = torch.linalg.qr(torch.randn(n, r, generator=generator))
        v, _ = torch.linalg.qr(torch.randn(p, r, generator=generator))
        s = torch.logspace(0, -0.5 * float(np.log10(kappa)), r)
        phi = (u * s) @ v.T * sqrt(n)
    y = torch.randn(n, d, generator=generator)
    return phi, y


def repeats_for(p: int) -> int:
    if p <= 8192:
        return 5
    return 3 if p <= 16384 else 2


def timed_arm(fn: Callable[[], SolveResult], *, repeats: int, warmup: int = 1) -> tuple[Timing, SolveResult]:
    timing = time_median(fn, repeats=repeats, warmup=warmup)
    result = timing.result
    assert isinstance(result, SolveResult)
    return timing, result


def subprocess_peak_rss_mb(n: int, p: int, d: int, arm: str) -> float | None:
    """Peak RSS of one isolated solve (ru_maxrss is a process high-water mark)."""
    code = (
        "import torch, resource, sys\n"
        "from benchmarks.vp.solvers import direct_lstsq, cholesky_normal\n"
        f"g = torch.Generator().manual_seed(0)\n"
        f"phi = torch.randn({n}, {p}, generator=g)\n"
        f"y = torch.randn({n}, {d}, generator=g)\n"
        f"({arm})(phi, y)\n"
        "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)\n"
    )
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=1800, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return float(out.stdout.strip().splitlines()[-1])


def sweep(p_grid: list[int], n_grid: list[int], *, measure_rss: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in n_grid:
        for p in p_grid:
            phi, y = controlled_problem(n, p, D, kappa=None, seed=42)
            reps = repeats_for(p)
            case: dict[str, Any] = {"n": n, "p": p, "D": D, "repeats": reps, "arms": {}}
            arms: dict[str, Callable[[], SolveResult]] = {
                "direct_lstsq": lambda phi=phi, y=y: direct_lstsq(phi, y, l2=L2),
                "cholesky_normal": lambda phi=phi, y=y: cholesky_normal(phi, y, l2=L2),
                "cg_gram": lambda phi=phi, y=y: block_cg(phi, y, l2=L2, tol=CG_TOL, max_iter=CG_MAX_ITER),
                "cg_matrix_free": lambda phi=phi, y=y: block_cg(
                    phi, y, l2=L2, tol=CG_TOL, max_iter=CG_MAX_ITER, matrix_free=True
                ),
                "nystrom_pcg": lambda phi=phi, y=y: nystrom_pcg(phi, y, l2=L2, tol=CG_TOL, max_iter=CG_MAX_ITER),
            }
            if p <= SVRG_MAX_P:
                arms["svrg"] = lambda phi=phi, y=y: sgd_to_tol(phi, y, l2=L2, tol=1e-4, batch_size=512, max_epochs=1000)
            reference: torch.Tensor | None = None
            for name, fn in arms.items():
                arm_reps = 1 if (name not in ("direct_lstsq", "cholesky_normal") and p >= ITERATIVE_SINGLE_REPEAT_P) else reps
                if name == "svrg":  # epoch-bounded already; one measured run, no warmup
                    arm_reps = 1
                timing, result = timed_arm(fn, repeats=arm_reps, warmup=0 if name == "svrg" else 1)
                if name == "direct_lstsq":
                    reference = result.w
                error = float("nan")
                if reference is not None and name != "direct_lstsq":
                    error = float((result.w - reference).norm() / reference.norm().clamp_min(1e-30))
                case["arms"][name] = {
                    "median_s": timing.median_s,
                    "samples_s": timing.samples_s,
                    "iterations": result.iterations,
                    "residual": result.residual,
                    "converged": result.converged,
                    "rel_error_vs_direct": error,
                }
                print(
                    f"n={n} p={p} {name:16s} {timing.median_s:9.3f}s iters={result.iterations:5d} "
                    f"resid={result.residual:.2e} err={error:.2e}",
                    flush=True,
                )
            # warm-start measurement (P6): perturb phi at outer-step / outer-epoch scale, reuse w
            cold = block_cg(phi, y, l2=L2, tol=CG_TOL, max_iter=CG_MAX_ITER, matrix_free=True)
            case["warm_start"] = {}
            for scale in WARM_START_SCALES:
                phi_next = phi + scale * phi.std() * torch.randn_like(phi)
                cold_next = block_cg(phi_next, y, l2=L2, tol=CG_TOL, max_iter=CG_MAX_ITER, matrix_free=True)
                warm_next = block_cg(phi_next, y, l2=L2, tol=CG_TOL, max_iter=CG_MAX_ITER, matrix_free=True, x0=cold.w)
                case["warm_start"][f"{scale:g}"] = {
                    "cold_iters": cold_next.iterations,
                    "warm_iters": warm_next.iterations,
                }
            if measure_rss and p >= 8192 and n == n_grid[0]:
                case["peak_rss_mb"] = {
                    "direct_lstsq": subprocess_peak_rss_mb(n, p, D, "direct_lstsq"),
                    "cholesky_normal": subprocess_peak_rss_mb(n, p, D, "cholesky_normal"),
                }
            rows.append(case)
    return rows


def iteration_scaling(*, smoke: bool) -> list[dict[str, Any]]:
    """P5/P11: iteration counts vs controlled kappa at fixed p."""
    n, p, tol = (1024, 512, 1e-3) if smoke else (4096, 1024, 1e-3)
    rows: list[dict[str, Any]] = []
    for kappa in KAPPA_GRID if not smoke else KAPPA_GRID[:2]:
        phi, y = controlled_problem(n, p, 64, kappa=kappa, seed=123)
        row: dict[str, Any] = {"n": n, "p": p, "kappa_target": kappa, "tol": tol, "arms": {}}
        gram = phi.T @ phi
        lam = effective_l2(phi, L2)
        eigs = torch.linalg.eigvalsh(gram + lam * torch.eye(p))
        kappa_eff = float(eigs[-1] / eigs[0].clamp_min(1e-30))
        row["kappa_effective"] = kappa_eff
        row["cg_iters_predicted"] = 0.5 * sqrt(kappa_eff) * log(1.0 / tol)
        cg = block_cg(phi, y, l2=L2, tol=tol, max_iter=100_000)
        row["arms"]["cg"] = {"iterations": cg.iterations, "converged": cg.converged}
        ny = nystrom_pcg(phi, y, l2=L2, tol=tol, max_iter=100_000)
        row["arms"]["nystrom_pcg"] = {"iterations": ny.iterations, "converged": ny.converged}
        if kappa <= GD_KAPPA_MAX:
            gd = sgd_to_tol(phi, y, l2=L2, tol=tol, max_epochs=200_000)
            row["arms"]["gd"] = {"iterations": gd.iterations, "converged": gd.converged}
            svrg = sgd_to_tol(phi, y, l2=L2, tol=tol, batch_size=256, max_epochs=20_000)
            row["arms"]["svrg"] = {"iterations": svrg.iterations, "converged": svrg.converged}
        print(f"kappa={kappa:.0e} (eff {kappa_eff:.1e}): {row['arms']}", flush=True)
        rows.append(row)
    return rows


def real_phi_anchor() -> dict[str, Any]:
    """One design matrix from an actual (untrained) IGL module, for spectral realism."""
    from igl import IGLModule

    set_seed(42)
    module = IGLModule(input_dim=16, max_dim=8, output_dim=D, n_anchors=1024, n_scales=4)
    x = torch.randn(4096, 16)
    with torch.no_grad():
        phi = module.design_matrix(x)
    y = torch.randn(4096, D)
    s = torch.linalg.svdvals(phi)
    cg = block_cg(phi, y, l2=L2, tol=CG_TOL, max_iter=CG_MAX_ITER, matrix_free=True)
    direct = direct_lstsq(phi, y, l2=L2)
    return {
        "n": 4096,
        "p": int(phi.shape[1]),
        "singular_values_head": s[:5].tolist(),
        "singular_values_tail": s[-5:].tolist(),
        "cg_iterations": cg.iterations,
        "cg_converged": cg.converged,
        "cg_error_vs_direct": float((cg.w - direct.w).norm() / direct.w.norm()),
    }


def fit_cost_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Least-squares fit of the predicted cost forms; crossover and memory-wall extrapolation."""
    direct_feats, direct_times = [], []
    cg_feats, cg_times = [], []
    for case in rows:
        n, p = case["n"], case["p"]
        if "direct_lstsq" in case["arms"]:
            direct_feats.append([n * p**2, p**3, p**2 * D])
            direct_times.append(case["arms"]["direct_lstsq"]["median_s"])
        cg = case["arms"].get("cg_matrix_free")
        if cg and cg["iterations"] > 0:
            cg_feats.append([n * p * D * cg["iterations"]])
            cg_times.append(cg["median_s"])
    coef_direct, *_ = np.linalg.lstsq(np.array(direct_feats), np.array(direct_times), rcond=None)
    coef_cg, *_ = np.linalg.lstsq(np.array(cg_feats), np.array(cg_times), rcond=None)
    n_ref = 4096
    iters_ref = max((case["arms"]["cg_matrix_free"]["iterations"] for case in rows if case["n"] == n_ref), default=200)
    p_star = None
    for p in range(1024, 200_000, 512):
        t_direct = coef_direct @ np.array([n_ref * p**2, p**3, p**2 * D])
        t_cg = coef_cg[0] * n_ref * p * D * iters_ref
        if t_direct > t_cg:
            p_star = p
            break
    rss_points = [
        (case["p"], case["peak_rss_mb"]["direct_lstsq"])
        for case in rows
        if "peak_rss_mb" in case and case["peak_rss_mb"].get("direct_lstsq")
    ]
    memory_wall_p = None
    if len(rss_points) >= 2:
        ps = np.array([float(p) for p, _ in rss_points])
        mbs = np.array([float(m) for _, m in rss_points])
        # dominant term is quadratic in p: fit mb = a * p^2 + c
        a, c = np.polyfit(ps**2, mbs, 1)
        if a > 0:
            memory_wall_p = float(np.sqrt((48_000 - c) / a))
    return {
        "direct_coefficients": coef_direct.tolist(),
        "cg_coefficient": coef_cg.tolist(),
        "cg_iters_assumed": iters_ref,
        "crossover_p_star": p_star,
        "memory_wall_p_at_48gb": memory_wall_p,
        "rss_points_mb": rss_points,
    }


def verdicts(rows: list[dict[str, Any]], scaling: list[dict[str, Any]], model: dict[str, Any]) -> dict[str, Any]:
    p3 = model["crossover_p_star"] is not None and 8_000 <= model["crossover_p_star"] <= 40_000
    p4 = model["memory_wall_p_at_48gb"] is not None and 30_000 <= model["memory_wall_p_at_48gb"] <= 80_000
    p5_cg = all(
        row["arms"]["cg"]["iterations"] <= 2.0 * max(row["cg_iters_predicted"], 1.0)
        for row in scaling
        if row["arms"].get("cg", {}).get("converged")
    )
    warm_ratios = [
        entry["cold_iters"] / max(entry["warm_iters"], 1)
        for case in rows
        if "warm_start" in case
        for scale, entry in case["warm_start"].items()
        if scale == "0.001"  # P6 verdict from the per-outer-step perturbation scale
    ]
    p6 = bool(warm_ratios) and float(np.median(warm_ratios)) >= 3.0
    ny_iters = [row["arms"]["nystrom_pcg"]["iterations"] for row in scaling if "nystrom_pcg" in row["arms"]]
    p11 = bool(ny_iters) and max(ny_iters) <= 2 * max(min(ny_iters), 1)
    return {
        "P3_crossover_in_band": p3,
        "P4_memory_wall_in_band": p4,
        "P5_cg_iters_within_2x_sqrt_kappa": p5_cg,
        "P6_warm_start_median_ratio": float(np.median(warm_ratios)) if warm_ratios else None,
        "P6_pass": p6,
        "P11_nystrom_iter_spread_under_2x": p11,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-real-phi", action="store_true")
    args = parser.parse_args()
    state = machine_state(gate=not args.smoke)
    set_seed(42)
    start = time.time()
    p_grid = SMOKE_P_GRID if args.smoke else FULL_P_GRID
    n_grid = SMOKE_N_GRID if args.smoke else FULL_N_GRID
    rows = sweep(p_grid, n_grid, measure_rss=not args.smoke)
    scaling = iteration_scaling(smoke=args.smoke)
    anchor = None if args.skip_real_phi else real_phi_anchor()
    model = fit_cost_model(rows)
    payload = {
        "config": {"p_grid": p_grid, "n_grid": n_grid, "D": D, "l2": L2, "cg_tol": CG_TOL, "smoke": args.smoke},
        "sweep": rows,
        "iteration_scaling": scaling,
        "real_phi_anchor": anchor,
        "cost_model": model,
        "verdicts": verdicts(rows, scaling, model),
    }
    path = write_result("e2_crossover", payload, state=state, wall_clock_s=time.time() - start)
    print(f"\nwrote {path}")
    print(json.dumps(payload["verdicts"], indent=2))


if __name__ == "__main__":
    main()
