"""G0: the geometry table of every analytic bed (no training).

Per bed and setting: intrinsic and ambient dimension, the obstruction to a conformal chart
(Weyl share in ``d >= 4``, Cotton share in ``d = 3``, nothing in ``d = 2``), whether a conformal
chart is known in closed form, and for the grid beds (hills, graphs) ``max |g^-1 - I|`` and the
separable ranks of the coefficient fields. This is the x-axis of the floor-vs-obstruction plots
and the truth the geometry report is checked against.
"""

from __future__ import annotations

import time
from typing import Any

from benchmarks.geometry._harness import write_block_result
from benchmarks.geometry.blocks._cli import gate, parser
from benchmarks.geometry.testbeds import graphs, hills, manifolds, spd_beds

BLOCK = "g0_geometry"


def analytic_beds(*, smoke: bool) -> list[manifolds.Bed]:
    """The embedded beds of the manifold matrix (a short list in smoke mode)."""
    cat = manifolds.catalogue()
    names = ["S4cap", "S2xR2", "S2xS2", "S3xR"] if smoke else list(cat)
    beds = [cat[n] for n in names]
    beds += [spd_beds.spd_bed(2), spd_beds.spd_bed(3)] + ([] if smoke else [spd_beds.spd_bed(2, metric="logeuclid")])
    beds += [manifolds.sphere_cap(3), manifolds.sphere_cap(3, axes=(1.0, 1.0, 2.0, 3.0)), manifolds.cylinder(3)]
    return beds


def main() -> None:
    """Run the block."""
    args = parser(__doc__ or "").parse_args()
    gate(args)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    n_points = 3 if args.smoke else 8
    for bed in analytic_beds(smoke=args.smoke):
        t0 = time.perf_counter()
        obs = manifolds.obstruction(bed, n_points=n_points)
        row = {
            "bed": bed.name,
            "d": bed.d,
            "D": bed.ambient,
            "kind": "analytic",
            "obstruction_kind": obs["kind"],
            "obstruction_mean": obs["mean"],
            "obstruction_max": obs["max"],
            "conformal_chart_known": bed.conformal_chart is not None,
            "blocks": list(bed.blocks),
            "notes": bed.notes,
            "seconds": time.perf_counter() - t0,
        }
        rows.append(row)
        write_block_result(BLOCK, bed.name, "analytic", 0, row, started_at=t0, label=args.label)
        print(
            f"{bed.name:24s} d={bed.d:2d} D={bed.ambient:3d} {obs['kind']:6s} share {obs['mean']:.4f} (max {obs['max']:.4f})",
            flush=True,
        )
    heights = [0.5, 1.0] if args.smoke else [0.2, 0.4, 0.6, 0.8, 1.0]
    for name in hills.HILLS:
        for s in heights:
            r = hills.analyse(name, s, charts=not args.smoke)
            row = {
                "bed": r["bed"],
                "d": 2,
                "D": 3,
                "kind": "analytic",
                "s": s,
                "max_dev": r["max_dev"],
                "rho": r["rho"],
                "r_g": r["r_g"],
                "coeff_ranks": r["coeff_ranks"],
                "residuals_xy": r["residuals_xy"],
                "conformal_chart_known": True,
                "obstruction_kind": "none",
                "obstruction_mean": 0.0,
            }
            rows.append(row)
            write_block_result(BLOCK, r["bed"], f"s{s:g}", 0, row, label=args.label)
            print(f"{r['bed']:24s} s={s:.1f} rho {r['rho']:.3f} max_dev {r['max_dev']:.3f} r_g {r['r_g']}", flush=True)
    dims = [3] if args.smoke else [3, 4]
    for d in dims:
        for name in graphs.BUMPS:
            r = graphs.analyse(name, d, 1.0, n=13 if args.smoke else None, n_born=5 if args.smoke else 40)
            row = {
                "bed": r["bed"],
                "d": d,
                "D": d + 1,
                "kind": "analytic",
                "s": 1.0,
                "max_dev": r["max_dev"],
                "rho": r["rho"],
                "obstruction_kind": "cotton" if d == 3 else "weyl",
                "obstruction_mean": None if name == "generic" else 0.0,
                "conformal_chart_known": False,
            }
            rows.append(row)
            write_block_result(BLOCK, r["bed"], "s1", 0, row, label=args.label)
            print(f"{r['bed']:24s} rho {r['rho']:.3f} max_dev {r['max_dev']:.3f}", flush=True)
    print(f"G0 done: {len(rows)} rows in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
