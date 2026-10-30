"""S3: chart-limited or data-limited (plan v2, H5). Kind: learned chart.

``none`` and ``conf`` at ``N_train ∈ {4k, 16k, full}`` (``--sizes``), the binding anchor count (``--anchors``,
one value) and latent width ``--k`` (plan: 32), three seeds, with the constant step budget (so the smaller sets
see more epochs) and the conformality weight recalibrated per (set, N) by the v2 rule. The report plots the
learning curves of the two arms and the paired gain against ``N``: the penalty's gain, if any, should be largest
where the data are scarcest.

Results under ``results/benchmarks/geometry/<label>/s3_learning_curve/<set>-R<R>-k<k>/<arm>-N<n>/seed<s>.json``.
"""

from __future__ import annotations

import time
from dataclasses import replace

from benchmarks.geometry.blocks._cli import gate, parser, seeds_of
from benchmarks.geometry.blocks._scale import (
    add_scale_args,
    base_spec,
    calibrated_weight,
    load_set,
    resolve_device,
    run_fit,
    subsample,
)

BLOCK = "s3_learning_curve"
SIZES = (4096, 16384, 0)  # 0: the full training split


def main() -> None:
    p = add_scale_args(parser("S3: none and conf along the training-set size at the binding anchor count"))
    p.add_argument("--sizes", default=",".join(str(s) for s in SIZES), help="comma-separated training sizes (0 = full)")
    args = p.parse_args()
    gate(args)
    device, state = resolve_device(args)
    smoke = bool(args.smoke)
    seeds = seeds_of(args)
    anchors = [int(r) for r in args.anchors.split(",")] if args.anchors else [32]
    ks = [int(v) for v in str(args.k).split(",")]
    sizes = [int(s) for s in args.sizes.split(",")]
    if smoke:
        anchors, ks, sizes = anchors[:1], ks[:1], [256, 0]
    t_start = time.perf_counter()
    for set_name in args.sets.split(","):
        for k in ks:
            full = load_set(set_name, k=k, smoke=smoke)
            for r in anchors:
                for n in sizes:
                    for s in seeds:
                        ds = subsample(full, n, seed=s) if n else full
                        spec = base_spec(ds, n_anchors=r, k=k, smoke=smoke, args=args)
                        cal = calibrated_weight(ds, spec, seed=s, device=device)
                        spec_w = replace(spec, penalty_weight=cal["weight"])
                        n_label = int(ds.x["train"].shape[0])
                        for arm in ("none", "conf"):
                            run_fit(
                                BLOCK,
                                ds,
                                spec_w,
                                arm,
                                seed=s,
                                device=device,
                                state=state,
                                label=args.label,
                                set_name=set_name,
                                result_arm=f"{arm}-N{n_label}",
                                extra={"calibration": cal, "n_train": n_label},
                                resume=bool(args.resume),
                            )
    print(f"S3 done in {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
