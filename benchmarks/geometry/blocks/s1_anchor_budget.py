"""S1: the anchor budget on the real sets (plan v2, H1, H2, H7). Kind: learned chart.

The ``none`` arm over the true anchor ladder ``R ∈ {16, 32, 64, 128, 256, 512}``, latent widths ``--k`` (plan:
16 and 64), anchor placement ``--inits`` (random, k-means keeping the scale ladder, k-means recentring it) and
the seeds, with ``inner_batch_size = max(4096, 16 R)``. The report reads off whether the readout is
anchor-limited (``err(32) − err(512)``), what data anchors buy and where the gap closes, and the margin over
the capacity-matched MLP head of S0 at every ``R``.

Results under ``results/benchmarks/geometry/<label>/s1_anchor_budget/<set>-R<R>-k<k>/<init>/seed<s>.json``.
"""

from __future__ import annotations

import time

from benchmarks.geometry.blocks._cli import gate, parser, seeds_of
from benchmarks.geometry.blocks._scale import add_scale_args, base_spec, load_set, resolve_device, run_fit

BLOCK = "s1_anchor_budget"
R_LADDER = (16, 32, 64, 128, 256, 512)
INITS = ("random", "kmeans-keep", "kmeans-median")


def main() -> None:
    p = add_scale_args(parser("S1: the none arm over the anchor ladder, latent widths and anchor placements"))
    p.add_argument("--inits", default=",".join(INITS), help="comma-separated anchor placements")
    args = p.parse_args()
    gate(args)
    device, state = resolve_device(args)
    smoke = bool(args.smoke)
    seeds = seeds_of(args)
    anchors = [int(r) for r in args.anchors.split(",")] if args.anchors else list(R_LADDER)
    ks = [int(v) for v in str(args.k).split(",")]
    inits = args.inits.split(",")
    if smoke:
        anchors, ks, inits = anchors[:2], ks[:1], inits[:2]
    t_start = time.perf_counter()
    for set_name in args.sets.split(","):
        for k in ks:
            ds = load_set(set_name, k=k, smoke=smoke)
            for r in anchors:
                for init in inits:
                    spec = base_spec(ds, n_anchors=r, k=k, smoke=smoke, args=args, anchor_init=init)
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
                            result_arm=init,
                            resume=bool(args.resume),
                        )
    print(f"S1 done in {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
