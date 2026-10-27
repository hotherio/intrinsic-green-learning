"""O1: the oracle brackets. IGL with the true conformal chart, the parameter chart or the embedded ambient data as inputs.

No penalty in any arm (``none``): the bracket measures what the input chart alone is worth, which bounds the gain
any chart penalty can deliver on that bed. The grey band of the X1 figures is the gap between ``ambient`` and
``conformal``.
"""

from __future__ import annotations

import time
from dataclasses import replace

from benchmarks.geometry._harness import machine_state, resolve_device, write_block_result
from benchmarks.geometry.blocks._cli import gate, parser, seeds_of
from benchmarks.geometry.blocks.x1_penalty import bed_by_name
from benchmarks.geometry.fitting import FitSpec, fit_arm, make_dataset

BLOCK = "o1_oracle_bracket"
BEDS: list[tuple[str, int | None]] = [
    ("hill-round", 8),
    ("hill-round", 32),
    ("S4cap", None),
    ("S2xR2", None),
    ("SPD2-airm", None),
    ("S2cap", None),
    ("S2cap", 256),
    ("S8cap", None),
    ("S8cap", 256),
    ("S3xR", None),
]


def main() -> None:
    """Run the block."""
    args = parser(__doc__ or "").parse_args()
    gate(args)
    device = resolve_device(args.device)
    state = machine_state(device, gate=not args.no_gate and not args.smoke)
    seeds = seeds_of(args)
    smoke = args.smoke
    beds = BEDS[:1] if smoke else BEDS
    spec = FitSpec(
        epochs=6 if smoke else 400, n_train=512 if smoke else 4096, n_val=128 if smoke else 1024, n_test=256 if smoke else 2048
    )
    t0 = time.perf_counter()
    for bed_name, amb in beds:
        bed = bed_by_name(bed_name)
        for seed in seeds:
            s = replace(spec, ambient=amb, n_train=8192 if (bed.d >= 8 and not smoke) else spec.n_train)
            dataset = make_dataset(bed, s, seed=seed)
            for inputs in ("conformal", "param", "ambient"):
                if inputs == "conformal" and dataset.conformal_chart is None:
                    continue
                t1 = time.perf_counter()
                row = fit_arm(dataset, s, "none", seed=seed, device=device, inputs=inputs, prune=not smoke)
                row["ambient"] = amb
                tag = f"{bed_name}-D{amb or bed.ambient}"
                write_block_result(BLOCK, tag, inputs, seed, row, device=device, state=state, started_at=t1, label=args.label)
                print(
                    f"{tag:14s} inputs {inputs:9s} seed {seed} nrmse {row['nrmse_test_dmax']:.4f} "
                    f"rank(u) {row['chart_cost']['rank']} bumps2% {row['chart_cost']['bumps']['0.02']} | {row['wall_s']:.0f}s",
                    flush=True,
                )
    print(f"O1 done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
