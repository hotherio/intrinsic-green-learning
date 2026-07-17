"""E6 — end-metric invariance across inner solvers.

Whatever the inner solver, the *instrument readings* must not change: the
dimension curve, the detected effective dimension, and the greedy-knockout
knee are the quantities papers report, and a solver swap that moved them
would disqualify the swap regardless of its speed.

Pre-registered predictions (P9):

- P9a  Detected effective dimension (elbow) identical whether each per-k
       readout is refit by the direct solve or by CG(1e-8), on ONE fitted
       module, for every (testbed, seed).
- P9b  The two per-k dimension curves are pointwise within 2% relative.
- P9c  Greedy-knockout knee identical between the two refit solvers.

Protocol: fit ONE module per (testbed, seed) with the package's direct
solve, then read the dimension curve twice on that SAME module -- once
refitting each truncation via the package (direct) and once via CG(1e-8).
This isolates the *solver's* effect on the instrument reading; comparing
two independently-trained models instead (as an earlier version did) only
re-measures E1's trajectory chaos on the flat tail of the curve. Seeds
{42, 123, 456}.

Run with::

    python -m benchmarks.vp.e6_quality [--smoke]
"""

import argparse
import json
import time
from typing import Any

import torch

from benchmarks.vp._harness import machine_state, set_seed, write_result
from benchmarks.vp.data import moons_classification, split, swiss_roll_regression
from benchmarks.vp.solvers import block_cg
from benchmarks.vp.vp_loop import VPLoop, VPLoopConfig
from igl import CrossEntropyLoss, IGLModule, MSELoss, detect_elbow, normalize_phi
from igl.matryoshka.knockout import detect_knockout_knee

SEEDS = [42, 123, 456]
CG_TOL = 1e-8
CURVE_REL_BAND = 0.02


def curve_two_solvers(module: IGLModule, x_val: torch.Tensor, y_val: torch.Tensor, loss: Any) -> dict[str, dict[int, float]]:
    """Dimension curve of ONE fitted module, each per-k readout refit by direct vs CG."""
    from igl import direct_solve_weights

    target = loss.target(y_val)
    d_max = module.max_dim
    z_full = module.encoder(x_val)
    out: dict[str, dict[int, float]] = {"direct": {}, "cg": {}}
    for k in range(1, d_max + 1):
        mask = torch.zeros(d_max)
        mask[:k] = 1.0
        phi = normalize_phi(module.green(z_full * mask.unsqueeze(0), gate_mask=mask), module.normalize)
        ones = torch.ones(phi.shape[0], 1, dtype=phi.dtype)
        phi_aug = torch.cat([phi, ones], dim=-1)
        w_direct = direct_solve_weights(phi_aug, target, l2=1e-3, on_nonfinite="raise")
        w_cg = block_cg(phi_aug, target, l2=1e-3, tol=CG_TOL, max_iter=50_000).w
        out["direct"][k] = loss.curve_score(phi_aug @ w_direct, target)
        out["cg"][k] = loss.curve_score(phi_aug @ w_cg, target)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    state = machine_state(gate=not args.smoke)
    start = time.time()
    epochs = 15 if args.smoke else 300
    n_moons, n_roll = (400, 600) if args.smoke else (2000, 4000)
    x_m, y_m = moons_classification(n_moons)
    x_r, y_r = swiss_roll_regression(n_roll)
    beds: dict[str, dict[str, Any]] = {
        "moons": {"data": split(x_m, y_m), "loss": CrossEntropyLoss(n_classes=2), "output_dim": 2},
        "swiss_roll": {"data": split(x_r, y_r), "loss": MSELoss(), "output_dim": x_r.shape[1]},
    }
    results: dict[str, Any] = {}
    all_elbow_match, all_curve_close, all_knee_match = True, True, True
    for name, bed in beds.items():
        x_train, y_train, x_val, y_val = bed["data"]
        per_seed: list[dict[str, Any]] = []
        for seed in SEEDS:
            set_seed(seed)
            module = IGLModule(input_dim=x_train.shape[1], max_dim=8, output_dim=bed["output_dim"], n_anchors=48, n_scales=4)
            VPLoop(loss=bed["loss"], config=VPLoopConfig(epochs=epochs, seed=seed, inner_batch_size=1024)).fit(
                module, x_train, y_train, x_val, y_val
            )
            curves = curve_two_solvers(module, x_val, y_val, bed["loss"])
            elbow_d, elbow_c = int(detect_elbow(curves["direct"])), int(detect_elbow(curves["cg"]))
            knee_d, knee_c = int(detect_knockout_knee(curves["direct"])), int(detect_knockout_knee(curves["cg"]))
            curve_rel = max(abs(curves["cg"][k] - v) / max(abs(v), 1e-12) for k, v in curves["direct"].items())
            row = {
                "seed": seed,
                "elbow_direct": elbow_d,
                "elbow_cg": elbow_c,
                "elbow_match": elbow_d == elbow_c,
                "max_curve_rel_gap": curve_rel,
                "curve_within_band": curve_rel <= CURVE_REL_BAND,
                "knee_direct": knee_d,
                "knee_cg": knee_c,
                "knee_match": knee_d == knee_c,
                "curves": {
                    "direct": {int(k): v for k, v in curves["direct"].items()},
                    "cg": {int(k): v for k, v in curves["cg"].items()},
                },
            }
            all_elbow_match = all_elbow_match and bool(row["elbow_match"])
            all_curve_close = all_curve_close and bool(row["curve_within_band"])
            all_knee_match = all_knee_match and bool(row["knee_match"])
            per_seed.append(row)
            print(
                f"{name:10s} seed={seed} elbow {elbow_d}=={elbow_c} knee {knee_d}=={knee_c} curve_gap={curve_rel:.5f}",
                flush=True,
            )
        results[name] = per_seed
    payload = {
        "config": {"seeds": SEEDS, "cg_tol": CG_TOL, "curve_rel_band": CURVE_REL_BAND, "smoke": args.smoke},
        "beds": results,
        "verdicts": {
            "P9a_elbow_identical": all_elbow_match,
            "P9b_curves_within_2pct": all_curve_close,
            "P9c_knee_identical": all_knee_match,
        },
    }
    path = write_result("e6_quality", payload, state=state, wall_clock_s=time.time() - start)
    print(f"\nwrote {path}")
    print(json.dumps(payload["verdicts"], indent=2))


if __name__ == "__main__":
    main()
