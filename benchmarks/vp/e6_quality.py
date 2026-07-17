"""E6 — end-metric invariance across inner solvers.

Whatever the inner solver, the *instrument readings* must not change: the
dimension curve, the detected effective dimension, and the greedy-knockout
knee are the quantities papers report, and a solver swap that moved them
would disqualify the swap regardless of its speed.

Pre-registered predictions (P9):

- P9a  Detected effective dimension (elbow) identical between the direct
       and CG(1e-6) fits for every (testbed, seed).
- P9b  Dimension curves pointwise within 2% relative of each other.
- P9c  Greedy-knockout knee identical.

Protocol: free-running fits per (arm, seed) as in E1, then the package's
own ``eval_dimension_curve`` / ``detect_elbow`` / ``greedy_knockout`` on
each fitted module. Seeds {42, 123, 456}.

Run with::

    python -m benchmarks.vp.e6_quality [--smoke]
"""

import argparse
import json
import time
from functools import partial
from typing import Any

from benchmarks.vp._harness import machine_state, set_seed, write_result
from benchmarks.vp.data import moons_classification, split, swiss_roll_regression
from benchmarks.vp.e1_equivalence import cg_inner
from benchmarks.vp.vp_loop import VPLoop, VPLoopConfig
from igl import CrossEntropyLoss, IGLModule, MSELoss, detect_elbow, eval_dimension_curve, greedy_knockout

SEEDS = [42, 123, 456]
CG_TOL = 1e-6
CURVE_REL_BAND = 0.02


def fitted_module(bed: dict[str, Any], *, seed: int, epochs: int, solver: Any) -> IGLModule:
    x_train, y_train, x_val, y_val = bed["data"]
    set_seed(seed)
    module = IGLModule(input_dim=x_train.shape[1], max_dim=8, output_dim=bed["output_dim"], n_anchors=48, n_scales=4)
    loop = VPLoop(loss=bed["loss"], config=VPLoopConfig(epochs=epochs, seed=seed, inner_batch_size=1024), inner_solver=solver)
    loop.fit(module, x_train, y_train, x_val, y_val)
    return module


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
        _, _, x_val, y_val = bed["data"]
        per_seed: list[dict[str, Any]] = []
        for seed in SEEDS:
            readings: dict[str, dict[str, Any]] = {}
            for arm, solver in (("direct", None), ("cg", partial(cg_inner, tol=CG_TOL))):
                module = fitted_module(bed, seed=seed, epochs=epochs, solver=solver)
                curve = eval_dimension_curve(module, x_val, y_val, loss=bed["loss"])
                knockout = greedy_knockout(module, x_val, y_val.float(), loss=bed["loss"])
                readings[arm] = {
                    "curve": {int(k): float(v) for k, v in curve.items()},
                    "elbow": int(detect_elbow(curve)),
                    "knockout_knee": int(knockout.knee),
                }
            direct, cg = readings["direct"], readings["cg"]
            curve_rel = max(abs(cg["curve"][k] - v) / max(abs(v), 1e-12) for k, v in direct["curve"].items())
            row = {
                "seed": seed,
                "elbow_direct": direct["elbow"],
                "elbow_cg": cg["elbow"],
                "elbow_match": direct["elbow"] == cg["elbow"],
                "max_curve_rel_gap": curve_rel,
                "curve_within_band": curve_rel <= CURVE_REL_BAND,
                "knee_direct": direct["knockout_knee"],
                "knee_cg": cg["knockout_knee"],
                "knee_match": direct["knockout_knee"] == cg["knockout_knee"],
                "curves": {"direct": direct["curve"], "cg": cg["curve"]},
            }
            all_elbow_match = all_elbow_match and bool(row["elbow_match"])
            all_curve_close = all_curve_close and bool(row["curve_within_band"])
            all_knee_match = all_knee_match and bool(row["knee_match"])
            per_seed.append(row)
            print(
                f"{name:10s} seed={seed} elbow {direct['elbow']}=={cg['elbow']} "
                f"knee {direct['knockout_knee']}=={cg['knockout_knee']} curve_gap={curve_rel:.4f}",
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
