"""S1: the anchor budget on the real sets (plan v2, H1, H2, H7). Kind: learned chart.

The ``none`` arm over the true anchor ladder ``R ∈ {16, 32, 64, 128, 256, 512}``, latent widths ``--k`` (plan:
16 and 64) and the kernel arms ``--arms``, each ``<placement>[+per-scale]``: the anchor placement (``random``,
``kmeans-keep`` = k-means anchors with the default scale ladder, ``kmeans-median`` = k-means anchors with the
ladder recentred on the latent spread) and, with ``+per-scale``, one design column per (anchor, scale) so the
readout weights the scales per anchor instead of through the global ``γ_k``. The default order runs the
bandwidth arms first: W1 found the effective rank of Φ stuck at 8–23 whatever ``R`` (MNIST), which points at
the ladder, not the anchor count. ``inner_batch_size = max(4096, 16 × columns)``. The report reads off whether
the readout is anchor-limited (``err(32) − err(512)``), what placement and bandwidth buy and where the gap
closes, and the margin over the capacity-matched MLP head of S0 at every ``R``.

Results under ``results/benchmarks/geometry/<label>/s1_anchor_budget/<set>-R<R>-k<k>/<arm>/seed<s>.json``.
"""

from __future__ import annotations

import time

from benchmarks.geometry.blocks._cli import gate, parser, seeds_of
from benchmarks.geometry.blocks._scale import add_scale_args, base_spec, load_set, resolve_device, run_fit

BLOCK = "s1_anchor_budget"
R_LADDER = (16, 32, 64, 128, 256, 512)
ARMS = ("kmeans-median", "kmeans-median+per-scale", "kmeans-keep+per-scale", "kmeans-keep", "random")
"""Decision order: the two bandwidth levers first, then the placement-only arms of the plan."""
PER_SCALE = "+per-scale"


def parse_arm(arm: str) -> tuple[str, bool]:
    """``<placement>[+per-scale]`` → (anchor_init, per_scale_columns)."""
    per_scale = arm.endswith(PER_SCALE)
    return (arm[: -len(PER_SCALE)] if per_scale else arm), per_scale


def main() -> None:
    p = add_scale_args(parser("S1: the none arm over the anchor ladder, latent widths, placements and kernel columns"))
    p.add_argument("--arms", default=",".join(ARMS), help="comma-separated <placement>[+per-scale] arms, run in order")
    args = p.parse_args()
    gate(args)
    device, state = resolve_device(args)
    smoke = bool(args.smoke)
    seeds = seeds_of(args)
    anchors = [int(r) for r in args.anchors.split(",")] if args.anchors else list(R_LADDER)
    ks = [int(v) for v in str(args.k).split(",")]
    arms = args.arms.split(",")
    if smoke:
        anchors, ks, arms = anchors[:2], ks[:1], arms[:2]
    t_start = time.perf_counter()
    for set_name in args.sets.split(","):
        for k in ks:
            ds = load_set(set_name, k=k, smoke=smoke)
            for r in anchors:
                for arm in arms:
                    init, per_scale = parse_arm(arm)
                    spec = base_spec(
                        ds, n_anchors=r, k=k, smoke=smoke, args=args, anchor_init=init, per_scale_columns=per_scale
                    )
                    for s in seeds:
                        run_fit(
                            BLOCK,
                            ds,
                            spec,
                            "none",
                            seed=s,
                            device=device,
                            state=state,
                            label=args.label,
                            set_name=set_name,
                            result_arm=arm,
                            resume=bool(args.resume),
                        )
    print(f"S1 done in {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
