"""X1: the penalty ablation on learned charts, against IGL today.

Arms: ``none``; ``orth-today`` (today's ``OrthogonalityPenalty(0.1, every=20)``); ``orth-matched`` (``beta = 0``,
axis-normalised, all prefixes, ``every = 2``); ``conf`` (``beta = 1``, same schedule); ``conf-plain`` (not
axis-normalised); ``beta-0.5``;
``conf-tan`` (the on-manifold metric through ``metric_inv``). Beds: the core set, or the dimension ladder with the reduced
arm set. Every run records both floors, the geometry report, the NRMSE of the readout solved on train and scored on
test, the pruning curve, the chart-only costs and ``d_eff``; the paired deltas against ``none`` are in the report.
"""

from __future__ import annotations

import time
from dataclasses import replace

import igl
from benchmarks.geometry._harness import machine_state, resolve_device, write_block_result
from benchmarks.geometry.blocks._cli import gate, parser, seeds_of
from benchmarks.geometry.fitting import FitSpec, calibrate_weight, fit_arm, make_dataset
from benchmarks.geometry.testbeds import manifolds, spd_beds

BLOCK = "x1_penalty"

CORE = ["hill-lop", "hill-round", "S4cap", "S2xR2", "SPD2-airm", "SPD2-logeuclid", "T2"]
CORE_ARMS = ["none", "orth-today", "orth-matched", "conf", "conf-tan"]
LADDER_ARMS = ["none", "orth-today", "conf"]


def bed_by_name(name: str) -> manifolds.Bed:
    """A bed of the matrix by its name (catalogue, SPD, graphs)."""
    cat = manifolds.catalogue()
    if name in cat:
        return cat[name]
    if name.startswith("SPD"):
        n = int(name[3])
        return spd_beds.spd_bed(n, metric=name.split("-")[1])
    if name.startswith("graph-"):
        _, bump, dd = name.split("-")
        return manifolds.graph_hypersurface(int(dd[1:]), bump)
    raise KeyError(name)


def ladder() -> list[tuple[str, int | None]]:
    """``(bed, ambient)`` pairs of the dimension ladder."""
    rows: list[tuple[str, int | None]] = []
    for d in (2, 4, 8):
        for amb in (None, 32, 256):
            rows.append((f"S{d}cap", amb))
            if d > 2:
                rows.append((f"S{d // 2}xS{d // 2}", amb))
    for d in (3, 4, 8):
        for amb in (None, 256):
            rows.append((f"graph-generic-d{d}", amb))
    for n in (3, 4):
        for amb in (None, 256):
            rows.append((f"SPD{n}-airm", amb))
    rows += [("T8", None), ("T8", 256), ("S16cap", 256)]
    return rows


def main() -> None:
    """Run the block."""
    p = parser(__doc__ or "")
    p.add_argument("--beds", default="core", help="core | ladder | comma-separated bed names")
    p.add_argument("--arms", default=None, help="comma-separated arm names (default by bed set)")
    p.add_argument("--ambient", default=None, help="ambient dimension for the core beds (default: native, hills 8)")
    args = p.parse_args()
    gate(args)
    device = resolve_device(args.device)
    state = machine_state(device, gate=not args.no_gate and not args.smoke)
    seeds = seeds_of(args)
    smoke = args.smoke
    if args.beds == "core":
        pairs = [(b, 8 if b.startswith("hill") else None) for b in CORE]
        arms = CORE_ARMS
    elif args.beds == "ladder":
        pairs = ladder()
        arms = LADDER_ARMS
    else:
        pairs = [(b, int(args.ambient) if args.ambient else None) for b in args.beds.split(",")]
        arms = CORE_ARMS
    if args.arms:
        arms = args.arms.split(",")
    if smoke:
        pairs, arms, seeds = pairs[:1], arms[:2], seeds[:1]
    spec = FitSpec(
        epochs=6 if smoke else 400, n_train=512 if smoke else 4096, n_val=128 if smoke else 1024, n_test=256 if smoke else 2048
    )
    t0 = time.perf_counter()
    for bed_name, amb in pairs:
        bed = bed_by_name(bed_name)
        weight: float | None = None  # calibrated once per bed on the first seed's penalty-free chart
        calibration: dict[str, float] = {}
        for seed in seeds:
            s = replace(spec, ambient=amb, n_train=8192 if (bed.d >= 8 and not smoke) else spec.n_train)
            try:
                dataset = make_dataset(bed, s, seed=seed)
            except (RuntimeError, ValueError) as exc:
                print(f"{bed_name} D={amb}: dataset failed ({exc})", flush=True)
                continue
            for arm in ["none", *[a for a in arms if a != "none"]]:
                if arm == "conf-tan" and dataset.metric_inv_train is None:
                    continue
                if arm not in arms and weight is not None:
                    continue  # ``none`` is run only for the calibration when it is not an arm
                t1 = time.perf_counter()
                stash: dict[str, igl.IGLModule] = {}
                s_arm = s if weight is None else replace(s, penalty_weight=weight)
                row = fit_arm(dataset, s_arm, arm, seed=seed, device=device, prune=not smoke, modules=stash)
                if arm == "none" and weight is None:
                    x_cal = dataset.x["train"].to(device)
                    y_cal = dataset.y["train"].to(device)
                    calibration = calibrate_weight(stash["none"], x_cal, y_cal, k=bed.d)
                    weight = calibration["weight"]
                    print(f"{bed_name} D={amb}: calibrated weight {weight:.4g} ({calibration})", flush=True)
                row["calibration"] = calibration
                if arm not in arms:
                    continue
                row["ambient"] = amb
                tag = f"{bed_name}-D{row['D']}"
                write_block_result(BLOCK, tag, arm, seed, row, device=device, state=state, started_at=t1, label=args.label)
                rt = row["report_test"]["full"]
                print(
                    f"{tag:18s} {arm:12s} seed {seed} nrmse {row['nrmse_test_dmax']:.4f} | floors orth {rt['sqrt_orth']:.3f} "
                    f"conf {rt['sqrt_conf']:.3f} | norm_inf {rt['norm_inf']:.2f} rank(u) {row['chart_cost']['rank']} "
                    f"bumps2% {row['chart_cost']['bumps']['0.02']} d_eff {row['d_eff_elbow']} | {row['wall_s']:.0f}s",
                    flush=True,
                )
    print(f"X1 done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
