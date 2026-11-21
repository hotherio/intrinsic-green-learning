"""R0: the go/no-go on the rank story (no training).

On beds where the charts are known, measure what a target ``u`` costs **in each chart**, by two chart-only
measures (``charts.chart_rank``, ``charts.bump_count``) and, where the source is available, the separable
rank of the compensated source ``fhat``:

- hills: the data chart ``(x, y)``, the deck's orthogonal non-conformal chart ``(p, Q(q))`` and the
  conformal chart ``(p, q)``; the deck's claim is that the conformal chart cuts the products of the source
  from 4/9 to 3/5 while the orthogonal one costs 5/7;
- a ``d = 3`` graph, SPD(2) under AIRM and ``S^2 x R^2``: the parameter chart against the chart the
  oracle optimisation (G4) finds for conformality.

Decision rule of the plan: if the conformal chart does not cut the bump count of ``u`` by at least 1.3x
on the hill, proposals A and B cannot show through the kernel; if ``rank(u)`` does not track
``rank(fhat)`` across charts, the rank bound leaves the deck.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from benchmarks.geometry import charts
from benchmarks.geometry._harness import write_block_result
from benchmarks.geometry.blocks import g4_oracle_floor as g4
from benchmarks.geometry.blocks._cli import figure_dir, gate, parser
from benchmarks.geometry.testbeds import hills, manifolds, spd_beds, targets

BLOCK = "r0_rank_go_nogo"


def hill_rows(name: str, *, smoke: bool) -> list[dict[str, Any]]:
    """The three charts of a hill at full height: costs of ``u`` and ranks of the source in each."""
    prob = hills.make_problem(name, 1.0)
    grid, met = prob.grid, prob.met
    chart = hills.isothermal_chart(grid, hills.HILLS[name], 1.0, met, prob.curved)
    core = (slice(2, -2), slice(2, -2))
    u = torch.from_numpy(prob.u[core].ravel())
    xy = torch.from_numpy(np.column_stack([grid.X[core].ravel(), grid.Y[core].ravel()]))
    pq = torch.from_numpy(np.column_stack([chart.p[core].ravel(), chart.q[core].ravel()]))
    (lam_pq, f_pq, lamf_pq, u_pq), box = hills.resample(grid, chart.p, chart.q, [chart.lam, prob.f, chart.lam * prob.f, prob.u])
    fhat_orth, rank_orth, _ = hills.orthogonal_chart(grid, chart, prob.f, prob.u, box)
    q = chart.q[core]
    q0, half = 0.5 * (box[2] + box[3]), 0.5 * (box[3] - box[2])
    big_q = q + hills.A_WARP * (half / np.pi) * np.sin(np.pi * (q - q0) / half)
    p_bigq = torch.from_numpy(np.column_stack([chart.p[core].ravel(), big_q.ravel()]))
    kw = {"max_atoms": 64 if smoke else 256}
    rows = []
    for label, z, rank_fhat in (
        ("xy", xy, hills.separable_rank(prob.fhat)),
        ("orth-pQ", p_bigq, rank_orth),
        ("conf-pq", pq, hills.separable_rank(lamf_pq)),
    ):
        rows.append(
            {
                "bed": f"hill-{name}",
                "chart": label,
                "d": 2,
                "kind": "analytic",
                "rank_fhat": rank_fhat,
                "rank_u_chart": charts.chart_rank(z, u)["rank"],
                "bump_count": charts.bump_count(z, u, **kw),
            }
        )
    return rows


def bed_rows(bed: manifolds.Bed, *, smoke: bool) -> list[dict[str, Any]]:
    """Parameter chart against the oracle conformal chart on a bed with grid targets."""
    n_points = 512 if smoke else 2048
    t, x = bed.sample(n_points, seed=0)
    tg = targets.grid_poisson(bed, n=11 if smoke else None)
    u = tg.u_at(t)
    chart, info = g4.fit_chart(bed, beta=1.0, n_points=min(n_points, 1024), steps=150 if smoke else 1500)
    with torch.no_grad():
        z_conf = chart(t)
    kw = {"max_atoms": 64 if smoke else 256}
    rows = []
    for label, z in (("param", t), ("oracle-conf", z_conf)):
        rows.append(
            {
                "bed": bed.name,
                "chart": label,
                "d": bed.d,
                "kind": "analytic",
                "oracle_sqrt_floor": info["sqrt_floor"],
                "rank_u_chart": charts.chart_rank(z, u)["rank"],
                "tt_ranks_u": charts.chart_rank(z, u)["tt_ranks"],
                "bump_count": charts.bump_count(z, u, **kw),
            }
        )
    return rows


def figure(rows: list[dict[str, Any]], label: str | None) -> None:
    """Bump count of ``u`` per chart, grouped by bed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    beds = sorted({r["bed"] for r in rows})
    fig, ax = plt.subplots(figsize=(1.6 * len(beds) + 3, 4))
    width = 0.25
    for i, bed in enumerate(beds):
        pts = [r for r in rows if r["bed"] == bed]
        for j, r in enumerate(pts):
            c = r["bump_count"]["count"]
            ax.bar(
                i + (j - 1) * width,
                c if c is not None else len(r["bump_count"]["curve"]),
                width,
                label=r["chart"] if i == 0 else None,
                hatch="//" if c is None else None,
            )
    ax.set_xticks(range(len(beds)), beds, rotation=20)
    ax.set_ylabel("Gaussian atoms for 1 % (hatched: censored)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = figure_dir(BLOCK, label) / "r0_bump_counts.png"
    fig.savefig(out, dpi=150)
    print("figure:", out)


def main() -> None:
    """Run the block."""
    args = parser(__doc__ or "").parse_args()
    gate(args)
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for name in hills.HILLS:
        rows += hill_rows(name, smoke=args.smoke)
    beds = [manifolds.graph_hypersurface(3, "generic"), spd_beds.spd_bed(2), manifolds.product_sphere_flat(2, 2)]
    for bed in beds[: 1 if args.smoke else 3]:
        rows += bed_rows(bed, smoke=args.smoke)
    for r in rows:
        write_block_result(BLOCK, r["bed"], r["chart"], 0, r, label=args.label)
        bc = r["bump_count"]
        bumps = " ".join(
            f"{k}%:{v if v is not None else '>' + str(len(bc['curve']))}"
            for k, v in ((5, bc["counts"]["0.05"]), (2, bc["counts"]["0.02"]), (1, bc["counts"]["0.01"]))
        )
        print(
            f"{r['bed']:18s} {r['chart']:12s} rank(u) {r['rank_u_chart']:2d} "
            f"rank(fhat) {r.get('rank_fhat', '-'):>2} bumps {bumps}"
            + (f" (oracle floor {r['oracle_sqrt_floor']:.3f})" if "oracle_sqrt_floor" in r else ""),
            flush=True,
        )
    figure(rows, args.label)
    print(f"R0 done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
