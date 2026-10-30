"""S0: the baselines of every real set (plan v2). Kind: baselines and frozen charts, no learned chart.

Per set: the linear probe and kNN once; the capacity-matched MLP head and the frozen PCA-k IGL chart at
``R ∈ {32, 128, 512}`` (``--anchors``) and ``k ∈ --k`` over the seeds; the Nyström RBF reference line at the same
``R`` and ``k`` (one seed). The headroom table of the report (``kNN or MLP − linear``) comes from here: a set
with no headroom cannot show a chart effect.

Results under ``results/benchmarks/geometry/<label>/s0_baselines/<set>-R<R>-k<k>/<baseline>/seed<s>.json`` (the
linear probe and kNN under ``<set>-R0-k0``).
"""

from __future__ import annotations

import time

from benchmarks.geometry import baselines
from benchmarks.geometry._harness import write_block_result
from benchmarks.geometry.blocks._cli import gate, parser, seeds_of
from benchmarks.geometry.blocks._scale import add_scale_args, base_spec, load_set, resolve_device, run_fit, tag

BLOCK = "s0_baselines"
R_LADDER = (32, 128, 512)


def main() -> None:
    p = add_scale_args(parser("S0: linear probe, kNN, capacity-matched MLP head, frozen PCA chart, Nyström reference"))
    p.add_argument("--parts", default="linear,knn,mlp,frozen,nystroem", help="comma-separated subset")
    args = p.parse_args()
    gate(args)
    device, state = resolve_device(args)
    smoke = bool(args.smoke)
    parts = set(args.parts.split(","))
    seeds = seeds_of(args)
    anchors = [int(r) for r in args.anchors.split(",")] if args.anchors else list(R_LADDER)
    ks = [int(v) for v in str(args.k).split(",")]
    if smoke:
        anchors, ks = anchors[:1], ks[:1]
    t_start = time.perf_counter()
    for set_name in args.sets.split(","):
        ds0 = load_set(set_name, k=ks[0], smoke=smoke)
        if "linear" in parts:
            t0 = time.perf_counter()
            row = baselines.linear_probe(ds0)
            write_block_result(
                BLOCK,
                tag(set_name, 0, 0),
                "linear",
                0,
                {**row, "set": set_name, "task": ds0.task},
                device="cpu",
                state=state,
                started_at=t0,
                label=args.label,
            )
            print(
                f"{set_name:18s} linear probe   {row.get('err_test', row.get('nrmse_test')):.4f} | {row['wall_s']:.0f}s",
                flush=True,
            )
        if "knn" in parts:
            t0 = time.perf_counter()
            metric = "cosine" if "minilm" in set_name else "euclidean"
            row = baselines.knn(ds0, metric=metric)
            write_block_result(
                BLOCK,
                tag(set_name, 0, 0),
                "knn",
                0,
                {**row, "set": set_name, "task": ds0.task},
                device="cpu",
                state=state,
                started_at=t0,
                label=args.label,
            )
            print(
                f"{set_name:18s} kNN ({metric})  {row.get('err_test', row.get('nrmse_test')):.4f} | {row['wall_s']:.0f}s",
                flush=True,
            )
        for k in ks:
            ds = ds0 if k == ks[0] else load_set(set_name, k=k, smoke=smoke)
            for r in anchors:
                spec = base_spec(ds, n_anchors=r, k=k, smoke=smoke, args=args)
                if "mlp" in parts:
                    for s in seeds:
                        t0 = time.perf_counter()
                        row = baselines.mlp_head(ds, spec, seed=s, device=device)
                        write_block_result(
                            BLOCK,
                            tag(set_name, r, k),
                            "mlp-head",
                            s,
                            {**row, "set": set_name, "task": ds.task},
                            device=device,
                            state=state,
                            started_at=t0,
                            label=args.label,
                        )
                        v = row.get("err_test", row.get("nrmse_test"))
                        print(
                            f"{set_name:18s} R={r:<4d} k={k:<3d} mlp-head        seed {s} {v:.4f} | "
                            f"best {row['best_epoch']}/{row['epochs']} | {row['wall_s']:.0f}s",
                            flush=True,
                        )
                if "frozen" in parts:
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
                            inputs="param",
                            result_arm="frozen-pca",
                        )
                if "nystroem" in parts:
                    t0 = time.perf_counter()
                    row = baselines.nystroem_reference(ds, k=k, n_components=r, seed=seeds[0])
                    write_block_result(
                        BLOCK,
                        tag(set_name, r, k),
                        "nystroem",
                        seeds[0],
                        {**row, "set": set_name, "task": ds.task},
                        device="cpu",
                        state=state,
                        started_at=t0,
                        label=args.label,
                    )
                    v = row.get("err_test", row.get("nrmse_test"))
                    print(f"{set_name:18s} R={r:<4d} k={k:<3d} nystroem        {v:.4f} | {row['wall_s']:.0f}s", flush=True)
    print(f"S0 done in {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
