"""W1: the week-1 gate on the first real set (plan v2). Kind: learned chart.

Four parts, each a flag of ``--parts`` (default all), on each ``--sets`` entry at latent width ``--k``:

- ``noise``: the noise floor, ``none`` at ``R ∈ {32, 512}`` over five seeds with k-means anchors; its seed SD
  sets every later threshold (MDE ≈ 3.6 SD at three seeds, 1.9 SD at five).
- ``h0``: the learned chart against the **frozen PCA-k chart** through the same kernel and readout at the same
  two ``R`` (three seeds): if the frozen chart matches at both, the chart buys nothing on this set and the
  penalty program is moot here; if it matches at 512 and loses at 32, H4 is live.
- ``inner``: the inner-solve confound, ``none`` at ``R = 512`` with ``inner_batch_size = 4096`` (the package
  default) against the protocol's ``16 R`` (three seeds).
- ``l2``: the ridge sweep, ``none`` at both ``R`` with ``source_l2 ∈ {1e-4, 1e-2}`` (one seed), to show the
  ranking is insensitive to it.

Results under ``results/benchmarks/geometry/<label>/w1_gate/<set>-R<R>-k<k>/<arm>/seed<s>.json``.
"""

from __future__ import annotations

import time

from benchmarks.geometry.blocks._cli import gate, parser, seeds_of
from benchmarks.geometry.blocks._scale import add_scale_args, base_spec, load_set, resolve_device, run_fit

BLOCK = "w1_gate"
R_PAIR = (32, 512)
NOISE_SEEDS = (0, 1, 2, 3, 4)


def main() -> None:
    p = add_scale_args(parser("W1: noise floor, frozen-PCA chart vs learned chart, inner-solve and ridge checks"))
    p.add_argument("--parts", default="noise,h0,inner,l2", help="comma-separated subset of noise,h0,inner,l2")
    args = p.parse_args()
    gate(args)
    device, state = resolve_device(args)
    smoke = bool(args.smoke)
    parts = set(args.parts.split(","))
    seeds = seeds_of(args)
    anchors = [int(r) for r in args.anchors.split(",")] if args.anchors else list(R_PAIR)
    if smoke:
        anchors, noise_seeds = anchors[:1], seeds
    else:
        noise_seeds = list(NOISE_SEEDS)
    t_start = time.perf_counter()
    for set_name in args.sets.split(","):
        for k in (int(v) for v in str(args.k).split(",")):
            ds = load_set(set_name, k=k, smoke=smoke)
            common = {"device": device, "state": state, "label": args.label, "set_name": set_name}
            for r in anchors:
                spec = base_spec(ds, n_anchors=r, k=k, smoke=smoke, args=args)
                if "noise" in parts:
                    for s in noise_seeds:
                        run_fit(BLOCK, ds, spec, "none", seed=s, **common)
                if "h0" in parts:
                    for s in seeds:
                        run_fit(BLOCK, ds, spec, "none", seed=s, inputs="param", result_arm="frozen-pca", **common)
                if "inner" in parts and r == max(anchors):
                    for s in seeds:
                        run_fit(
                            BLOCK,
                            ds,
                            base_spec(ds, n_anchors=r, k=k, smoke=smoke, args=args, inner_batch_size=4096),
                            "none",
                            seed=s,
                            result_arm="none-inner4096",
                            **common,
                        )
                if "l2" in parts:
                    for l2 in (1e-4, 1e-2):
                        run_fit(
                            BLOCK,
                            ds,
                            base_spec(ds, n_anchors=r, k=k, smoke=smoke, args=args, source_l2=l2),
                            "none",
                            seed=seeds[0],
                            result_arm=f"none-l2-{l2:g}",
                            **common,
                        )
    print(f"W1 done in {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
