"""S2: the penalty ablation where anchors bind (plan v2; H4 primary, H8, H9). Kind: learned chart.

Arms ``none, orth-today, conf, conf@0.1, conf@0.03`` at ``R ∈ {R_bind, 256}`` (``--anchors``; ``R_bind`` is read
off S1 and passed explicitly), latent width ``--k`` (plan: 32), k-means anchors, five seeds; the conformality
weight is calibrated once per (set, R) at 10 % of the budget of a penalty-free fit (seed 0) and the realised
penalty/task gradient ratio is logged at 10 / 50 / 100 % of every penalised fit. ``--control`` adds the 2× budget
control for ``none`` and ``conf`` (two seeds). ``--lr-readout`` scores every chart also with a multinomial
logistic regression on Φ (the readout-robustness check; one sklearn fit per model).

Results under ``results/benchmarks/geometry/<label>/s2_penalty_at_scale/<set>-R<R>-k<k>/<arm>/seed<s>.json``.
"""

from __future__ import annotations

import time
from dataclasses import replace

from benchmarks.geometry.blocks._cli import gate, parser, seeds_of
from benchmarks.geometry.blocks._scale import add_scale_args, base_spec, calibrated_weight, load_set, resolve_device, run_fit

BLOCK = "s2_penalty_at_scale"
ARMS = ("none", "orth-today", "conf", "conf@0.1", "conf@0.03")


def main() -> None:
    p = add_scale_args(parser("S2: penalty arms at the binding and a generous anchor count, calibrated weights"))
    p.add_argument("--arms", default=",".join(ARMS), help="comma-separated arms")
    p.add_argument("--control", action="store_true", help="add the 2x budget control for none and conf (two seeds)")
    p.add_argument("--lr-readout", action="store_true", help="also score with a logistic regression on Phi")
    p.set_defaults(seeds="0,1,2,3,4")
    args = p.parse_args()
    gate(args)
    device, state = resolve_device(args)
    smoke = bool(args.smoke)
    seeds = seeds_of(args)
    anchors = [int(r) for r in args.anchors.split(",")] if args.anchors else [32, 256]
    ks = [int(v) for v in str(args.k).split(",")]
    arms = args.arms.split(",")
    if smoke:
        anchors, ks, arms = anchors[:1], ks[:1], [a for a in arms if a in ("none", "conf")][:2]
    t_start = time.perf_counter()
    for set_name in args.sets.split(","):
        for k in ks:
            ds = load_set(set_name, k=k, smoke=smoke)
            common = {
                "device": device,
                "state": state,
                "label": args.label,
                "set_name": set_name,
                "lr_readout": bool(args.lr_readout),
                "resume": bool(args.resume),
            }
            for r in anchors:
                spec = base_spec(ds, n_anchors=r, k=k, smoke=smoke, args=args)
                cal = calibrated_weight(ds, spec, seed=seeds[0], device=device)
                print(f"{set_name} R={r} k={k}: calibrated weight {cal['weight']:.4g} ({cal})", flush=True)
                spec_w = replace(spec, penalty_weight=cal["weight"])
                for s in seeds:
                    for arm in arms:
                        run_fit(BLOCK, ds, spec_w, arm, seed=s, extra={"calibration": cal}, **common)
                if args.control:
                    double = replace(spec_w, steps_budget=2 * (spec_w.steps_budget or 0))
                    for s in seeds[:2]:
                        for arm in ("none", "conf"):
                            run_fit(
                                BLOCK,
                                ds,
                                double,
                                arm,
                                seed=s,
                                result_arm=f"{arm}-2x",
                                extra={"calibration": cal, "control": "2x budget"},
                                **common,
                            )
    print(f"S2 done in {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
