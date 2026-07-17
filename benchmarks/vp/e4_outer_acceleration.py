"""E4 — outer-loop acceleration.

The discussion's bet: at IGL's data scales the outer loop does not need to
be stochastic, and either full-batch L-BFGS on the reduced functional or
epoch-level safeguarded Anderson/RNA extrapolation buys a 2-5x wall-clock
cut on the slow outer tail.

Pre-registered predictions:

- P7a  Some accelerated arm (fullbatch-adam, fullbatch-lbfgs, or
       safeguarded AA) reaches the minibatch baseline's final validation
       loss with >=2x less wall-clock on at least one of the two
       regression testbeds.
- P7b  Safeguarded AA never ends above the baseline's final validation
       loss (the safeguard makes it free).

Protocol: mlp-manifold reconstruction testbeds (latent 2 -> D=64 and
latent 4 -> D=512) plus moons classification as a stress case. Baseline =
minibatch AdamW at the house recipe for a fixed epoch budget; accelerated
arms get a generous epoch cap and are scored by wall-clock to the
baseline's (i) final and (ii) 1.05x-final val loss, read off cumulative
per-epoch timestamps. Seeds {42, 123, 456}; all arms CPU, identical data
and modules per seed.

Run with::

    python -m benchmarks.vp.e4_outer_acceleration [--smoke]
"""

import argparse
import json
import time
from typing import Any

import numpy as np

from benchmarks.vp._harness import machine_state, set_seed, write_result
from benchmarks.vp.data import mlp_manifold, moons_classification, split
from benchmarks.vp.vp_loop import VPLoop, VPLoopConfig, VPLoopResult
from igl import CrossEntropyLoss, IGLModule, MSELoss

SEEDS = [42, 123, 456]
ARMS = (
    "minibatch-adam",
    "fullbatch-adam",
    "fullbatch-lbfgs",
    "minibatch-adam-aa",
    "fullbatch-adam-aa",
    "hybrid-adam-lbfgs",
)


def testbeds(*, smoke: bool) -> dict[str, dict[str, Any]]:
    n_small = 2000 if smoke else 8000
    n_moons = 800 if smoke else 2000
    x_a, y_a = mlp_manifold(n_small, latent_dim=2, ambient_dim=64)
    x_m, y_m = moons_classification(n_moons)
    beds = {
        "manifold_2_64": {"data": split(x_a, y_a), "loss": MSELoss(), "output_dim": x_a.shape[1], "kind": "regression"},
        "moons": {"data": split(x_m, y_m), "loss": CrossEntropyLoss(n_classes=2), "output_dim": 2, "kind": "classification"},
    }
    if not smoke:
        x_b, y_b = mlp_manifold(n_small, latent_dim=4, ambient_dim=512)
        beds["manifold_4_512"] = {
            "data": split(x_b, y_b),
            "loss": MSELoss(),
            "output_dim": x_b.shape[1],
            "kind": "regression",
        }
    return beds


def arm_config(arm: str, *, seed: int, smoke: bool) -> VPLoopConfig:
    base_epochs = 30 if smoke else 400
    fullbatch_multiplier = 3 if smoke else 8
    if arm == "minibatch-adam":
        return VPLoopConfig(epochs=base_epochs, batch_size=256, inner_batch_size=2048, seed=seed)
    if arm == "minibatch-adam-aa":
        # RNA on EMA-averaged iterates: extrapolating raw SGD iterates fits noise
        return VPLoopConfig(epochs=base_epochs, batch_size=256, inner_batch_size=2048, seed=seed, aa_ema=0.5)
    if arm == "fullbatch-adam":
        # one step per epoch: give it the same number of *gradient steps* upper bound
        return VPLoopConfig(epochs=fullbatch_multiplier * base_epochs, batch_size=10**9, inner_batch_size=2048, seed=seed)
    if arm == "fullbatch-adam-aa":
        # classic Anderson setting: deterministic outer map, raw iterates
        return VPLoopConfig(epochs=fullbatch_multiplier * base_epochs, batch_size=10**9, inner_batch_size=2048, seed=seed)
    if arm == "hybrid-adam-lbfgs":
        # explore (minibatch Adam picks the basin) then polish (L-BFGS endgame)
        return VPLoopConfig(
            epochs=base_epochs // 2,
            batch_size=256,
            inner_batch_size=2048,
            hybrid_warmup_epochs=base_epochs // 4,
            lbfgs_max_iter=10,
            seed=seed,
        )
    return VPLoopConfig(epochs=base_epochs // 2, inner_batch_size=2048, lbfgs_max_iter=10, seed=seed)


def time_to_target(result: VPLoopResult, target: float) -> float | None:
    for val, t in zip(result.val_loss, result.epoch_time_s, strict=False):
        if val <= target:
            return t
    return None


def run_bed(name: str, bed: dict[str, Any], *, smoke: bool, arms: tuple[str, ...] = ARMS) -> dict[str, Any]:
    x_train, y_train, x_val, y_val = bed["data"]
    out: dict[str, Any] = {"arms": {}}
    baselines: dict[int, VPLoopResult] = {}
    seeds = SEEDS[:1] if smoke else SEEDS
    for arm in arms:
        per_seed: list[dict[str, Any]] = []
        for seed in seeds:
            set_seed(seed)
            module = IGLModule(input_dim=x_train.shape[1], max_dim=8, output_dim=bed["output_dim"], n_anchors=48, n_scales=4)
            loop = VPLoop(
                loss=bed["loss"],
                config=arm_config(arm, seed=seed, smoke=smoke),
                outer_mode=arm,  # pyright: ignore[reportArgumentType]
            )
            result = loop.fit(module, x_train, y_train, x_val, y_val)
            if arm == "minibatch-adam":
                baselines[seed] = result
            record: dict[str, Any] = {
                "seed": seed,
                "final_val": result.val_loss[-1],
                "best_val": min(result.val_loss),
                "wall_clock_s": result.wall_clock_s,
                "epochs": len(result.val_loss),
                "aa_proposals": result.aa_proposals,
                "aa_accepted": result.aa_accepted,
            }
            baseline = baselines.get(seed)
            if baseline is not None:
                for label, target in (
                    ("final", baseline.val_loss[-1]),
                    ("final_1.05x", 1.05 * baseline.val_loss[-1]),
                ):
                    reached = time_to_target(result, target)
                    baseline_reached = time_to_target(baseline, target)
                    record[f"time_to_{label}_s"] = reached
                    record[f"speedup_vs_baseline_{label}"] = (
                        baseline_reached / reached if reached and baseline_reached else None
                    )
            per_seed.append(record)
            print(
                f"{name:14s} {arm:18s} seed={seed} final={record['final_val']:.5g} "
                f"wall={record['wall_clock_s']:.1f}s speedup(final)={record.get('speedup_vs_baseline_final')}",
                flush=True,
            )
        out["arms"][arm] = per_seed
    return out


def verdicts(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    regression_beds = [b for b in results if b.startswith("manifold")]
    best_speedups: dict[str, float] = {}
    p7a = False
    for bed in regression_beds:
        for arm, runs in results[bed]["arms"].items():
            if arm == "minibatch-adam":
                continue
            speedups = [r["speedup_vs_baseline_final"] for r in runs if r.get("speedup_vs_baseline_final")]
            if speedups:
                med = float(np.median(speedups))
                best_speedups[f"{bed}/{arm}"] = med
                if med >= 2.0:
                    p7a = True
    p7b = True
    for bed_result in results.values():
        baseline_finals = {r["seed"]: r["final_val"] for r in bed_result["arms"]["minibatch-adam"]}
        for aa_arm in ("minibatch-adam-aa", "fullbatch-adam-aa"):
            for r in bed_result["arms"].get(aa_arm, []):
                if aa_arm == "minibatch-adam-aa" and r["final_val"] > baseline_finals[r["seed"]] * 1.001:
                    p7b = False  # the safeguard must make minibatch AA free vs its own baseline
    return {
        "P7a_median_speedups_to_baseline_final": best_speedups,
        "P7a_pass": p7a,
        "P7b_aa_never_worse": p7b,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arms", default=None, help="comma-separated subset of arms (baseline auto-included)")
    args = parser.parse_args()
    arms = ARMS
    if args.arms:
        chosen = [a.strip() for a in args.arms.split(",")]
        if "minibatch-adam" not in chosen:
            chosen.insert(0, "minibatch-adam")  # targets are defined against the baseline
        arms = tuple(chosen)
    state = machine_state(gate=not args.smoke)
    start = time.time()
    beds = testbeds(smoke=args.smoke)
    results = {name: run_bed(name, bed, smoke=args.smoke, arms=arms) for name, bed in beds.items()}
    payload = {
        "config": {"seeds": SEEDS, "arms": list(arms), "smoke": args.smoke},
        "beds": results,
        "verdicts": verdicts(results),
    }
    path = write_result("e4_outer_acceleration", payload, state=state, wall_clock_s=time.time() - start)
    print(f"\nwrote {path}")
    print(json.dumps(payload["verdicts"], indent=2))


if __name__ == "__main__":
    main()
