"""E3 — conditioning of the inner problem and tolerance transfer under whitening.

Part A measures the effective condition number kappa(phi^T phi + l2_eff I)
of real IGL design matrices across kernel configurations (anchor count,
row normalization, null space) and ridge strengths, and checks that CG
iteration counts track sqrt(kappa).

Part B pins down the honest version of "whitened targets are not
well-conditioned": target whitening changes the right-hand side, not the
operator, so kappa is untouched — what changes is how much *outer-gradient*
error a fixed inner tolerance leaves. The envelope gradient of the reduced
functional L(phi) = 0.5 ||phi w*(phi) - y||^2 is dL/dphi = (phi w* - y) w*^T;
with an eps-suboptimal inner solution the gradient error grows with the
target's dynamic range. Fisher-whitened targets stretch low-variance
directions and therefore demand a tighter inner tolerance for the same
outer-gradient accuracy; the trace-damped metric should recover most of
that gap.

Pre-registered predictions:

- E3a  CG iterations within x2 of 0.5*sqrt(kappa_eff)*ln(1/tol) across the
       kernel-config grid.
- E3b  For a fixed CG tolerance, the outer-gradient error ordering is
       fisher > logit > raw; the tolerance needed for 1e-3 gradient error
       is tighter for fisher than raw by a reported factor.
- E3c  damped_metric(fisher, logit) recovers at least half of the
       raw-vs-fisher tolerance gap (in log10-tolerance terms).

Run with::

    python -m benchmarks.vp.e3_conditioning [--smoke]
"""

import argparse
import json
import math
import time
from math import log, sqrt
from typing import Any

import torch

from benchmarks.vp._harness import machine_state, set_seed, write_result
from benchmarks.vp.solvers import block_cg, effective_l2
from igl import IGLModule
from igl.whitening import TargetWhitener, damped_metric, logit_metric
from igl.whitening.metrics import fisher_pullback

L2_GRID = [1e-2, 1e-3, 1e-4]
CG_TOL = 1e-5
GRAD_TOLS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
GRAD_ERROR_TARGET = 1e-3


def build_phi(n_anchors: int, normalize: str, null_space: str, *, n: int = 4096) -> torch.Tensor:
    """Design matrix of an (untrained) IGL module on synthetic gaussian input."""
    set_seed(42)
    module = IGLModule(
        input_dim=16,
        max_dim=8,
        output_dim=4,
        n_anchors=n_anchors,
        n_scales=4,
        normalize=normalize,  # pyright: ignore[reportArgumentType]
    )
    x = torch.randn(n, 16)
    with torch.no_grad():
        phi = module.design_matrix(x)
    if null_space == "constant":
        phi = torch.cat([phi, torch.ones(n, 1)], dim=1)
    elif null_space == "polynomial":
        z = module.latent(x).detach()
        phi = torch.cat([phi, torch.ones(n, 1), z], dim=1)
    return phi


def part_a(*, smoke: bool) -> list[dict[str, Any]]:
    anchor_grid = [64, 1024] if smoke else [64, 1024, 8192]
    rows: list[dict[str, Any]] = []
    for n_anchors in anchor_grid:
        for normalize in ("nw", "softmax"):
            for null_space in ("none", "constant", "polynomial"):
                phi = build_phi(n_anchors, normalize, null_space)
                s = torch.linalg.svdvals(phi.double())
                row: dict[str, Any] = {
                    "n_anchors": n_anchors,
                    "normalize": normalize,
                    "null_space": null_space,
                    "p": int(phi.shape[1]),
                    "sigma_max": float(s[0]),
                    "sigma_min": float(s[-1]),
                    "kappa_by_l2": {},
                }
                for l2 in L2_GRID:
                    lam = effective_l2(phi, l2)
                    row["kappa_by_l2"][f"{l2:g}"] = float((s[0] ** 2 + lam) / (s[-1] ** 2 + lam))
                lam = effective_l2(phi, 1e-3)
                kappa_eff = row["kappa_by_l2"]["0.001"]
                y = torch.randn(phi.shape[0], 64)
                cg = block_cg(phi, y, l2=1e-3, tol=CG_TOL, max_iter=50_000)
                row["cg_iterations"] = cg.iterations
                row["cg_converged"] = cg.converged
                row["cg_iters_predicted"] = 0.5 * sqrt(kappa_eff) * log(1.0 / CG_TOL)
                print(
                    f"anchors={n_anchors:5d} {normalize:7s} null={null_space:10s} p={row['p']:5d} "
                    f"kappa(1e-3)={kappa_eff:.3e} cg_iters={cg.iterations} (pred {row['cg_iters_predicted']:.0f})",
                    flush=True,
                )
                rows.append(row)
    return rows


def _envelope_grad(phi: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (phi @ w - y) @ w.T


def part_b(*, smoke: bool) -> dict[str, Any]:
    set_seed(123)
    n, d, vocab = (1024, 32, 200) if smoke else (4096, 64, 1000)
    states = torch.randn(n, d) @ torch.diag(torch.logspace(0, -2, d))  # anisotropic "hidden states"
    # An iid-gaussian head concentrates to a near-isotropic metric (kappa_G ~ 2-3),
    # which cannot discriminate whitening arms; real LLM unembeddings are strongly
    # anisotropic, so the synthetic head gets a matching log-spaced column scaling.
    head = (torch.randn(vocab, d) / sqrt(d)) * torch.logspace(0, -1.5, d).unsqueeze(0)
    metrics: dict[str, torch.Tensor | None] = {
        "raw": None,
        "logit": logit_metric(head),
        "fisher": fisher_pullback(head, states, n_sub=min(n, 2048)),
    }
    metrics["damped"] = damped_metric(metrics["fisher"], metrics["logit"])  # pyright: ignore[reportArgumentType]
    phi = build_phi(256, "nw", "constant", n=n)
    result: dict[str, Any] = {"n": n, "d": d, "vocab": vocab, "metrics": {}}
    for name, metric in metrics.items():
        whitener = TargetWhitener(metric).fit(states)
        y = whitener.transform(states).double()  # float64 isolates the whitening effect from the fp32 floor
        phi64 = phi.double()
        w_star = torch.linalg.lstsq(
            torch.cat([phi64, sqrt(effective_l2(phi, 1e-3)) * torch.eye(phi.shape[1], dtype=torch.float64)]),
            torch.cat([y, torch.zeros(phi.shape[1], y.shape[1], dtype=torch.float64)]),
        ).solution
        reference = _envelope_grad(phi64, y, w_star)
        kappa_metric = None
        if metric is not None:
            eigs = torch.linalg.eigvalsh(metric.double())
            kappa_metric = float(eigs[-1] / eigs[0].clamp_min(1e-30))
        errors: dict[str, float] = {}
        errors_aggregate: dict[str, float] = {}
        tol_for_target = None
        tol_for_target_aggregate = None
        for tol in GRAD_TOLS:
            for aggregate in (False, True):
                solved = block_cg(phi64, y, l2=1e-3, tol=tol, max_iter=100_000, aggregate_stop=aggregate)
                grad = _envelope_grad(phi64, y, solved.w)
                err = float((grad - reference).norm() / reference.norm().clamp_min(1e-30))
                if aggregate:
                    errors_aggregate[f"{tol:g}"] = err
                    if tol_for_target_aggregate is None and err <= GRAD_ERROR_TARGET:
                        tol_for_target_aggregate = tol
                else:
                    errors[f"{tol:g}"] = err
                    if tol_for_target is None and err <= GRAD_ERROR_TARGET:
                        tol_for_target = tol
        result["metrics"][name] = {
            "kappa_metric": kappa_metric,
            "target_dynamic_range": float(y.std(dim=0).max() / y.std(dim=0).min().clamp_min(1e-30)),
            "grad_error_by_tol": errors,
            "grad_error_by_tol_aggregate_stop": errors_aggregate,
            "tol_for_1e-3_grad_error": tol_for_target,
            "tol_for_1e-3_grad_error_aggregate_stop": tol_for_target_aggregate,
        }
        print(f"metric={name:7s} kappa_G={kappa_metric} tol_for_1e-3={tol_for_target} errors={errors}", flush=True)
    return result


def verdicts(rows_a: list[dict[str, Any]], b: dict[str, Any]) -> dict[str, Any]:
    e3a = all(row["cg_iterations"] <= 2.0 * max(row["cg_iters_predicted"], 1.0) for row in rows_a if row["cg_converged"])

    def tol_of(name: str, key: str = "tol_for_1e-3_grad_error") -> float | None:
        return b["metrics"][name][key] if b else None

    ordering = None
    raw_tol, fisher_tol, damped_tol = tol_of("raw"), tol_of("fisher"), tol_of("damped")
    agg = "tol_for_1e-3_grad_error_aggregate_stop"
    raw_agg, fisher_agg = tol_of("raw", agg), tol_of("fisher", agg)
    if raw_tol is not None and fisher_tol is not None:
        ordering = fisher_tol <= raw_tol
    e3c = None
    if raw_tol and fisher_tol and damped_tol and fisher_tol < raw_tol:
        gap = math.log10(raw_tol) - math.log10(fisher_tol)
        recovered = math.log10(damped_tol) - math.log10(fisher_tol)
        e3c = recovered >= 0.5 * gap
    return {
        "E3a_cg_iters_within_2x_prediction": e3a,
        "E3b_per_column_stop_tol_raw_vs_fisher": (raw_tol, fisher_tol),
        "E3b_aggregate_stop_tol_raw_vs_fisher": (raw_agg, fisher_agg),
        "E3b_fisher_needs_tighter_tol": ordering,
        "E3b_tol_raw": raw_tol,
        "E3b_tol_fisher": fisher_tol,
        "E3b_tol_damped": damped_tol,
        "E3c_damped_recovers_half_gap": e3c,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only", choices=["a", "b", "all"], default="all")
    args = parser.parse_args()
    state = machine_state(gate=not args.smoke)
    start = time.time()
    if args.only == "b":
        from pathlib import Path

        prior = sorted(Path("results/benchmarks/vp/e3_conditioning").glob("*/result.json"))[-1]
        rows_a = json.loads(prior.read_text())["part_a_kernel_conditioning"]
    else:
        rows_a = part_a(smoke=args.smoke)
    b = part_b(smoke=args.smoke) if args.only != "a" else {}
    payload = {
        "config": {"l2_grid": L2_GRID, "cg_tol": CG_TOL, "grad_tols": GRAD_TOLS, "smoke": args.smoke, "only": args.only},
        "part_a_kernel_conditioning": rows_a,
        "part_b_tolerance_transfer": b,
        "verdicts": verdicts(rows_a, b),
    }
    path = write_result("e3_conditioning", payload, state=state, wall_clock_s=time.time() - start)
    print(f"\nwrote {path}")
    print(json.dumps(payload["verdicts"], indent=2))


if __name__ == "__main__":
    main()
