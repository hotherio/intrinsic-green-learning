"""Assemble the JSON results of the geometry suite into the Markdown tables of ``REPORT.md``.

``python -m benchmarks.geometry.report --label tier0`` prints the tables of every block found under
``results/benchmarks/geometry/<label>/``; paste them into ``REPORT.md`` under the matching headings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(label: str, block: str) -> list[dict[str, Any]]:
    """All result rows of a block under a label."""
    base = Path.cwd() / "results" / "benchmarks" / "geometry" / label / block
    rows: list[dict[str, Any]] = []
    for path in sorted(base.glob("*/*/seed*.json")):
        rows.append(json.loads(path.read_text()))
    return rows


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    """A Markdown table."""
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    return "\n".join(out)


def _fmt(v: Any) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        if v == 0:
            return "0"
        return f"{v:.3g}" if abs(v) < 1e-2 or abs(v) >= 1e3 else f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def table_r0(rows: list[dict[str, Any]]) -> str:
    """R0: chart by chart, ranks and atom counts."""
    body = []
    for r in sorted(rows, key=lambda r: (r["bed"], r["chart"])):
        c = r["bump_count"]["counts"]
        body.append(
            [
                r["bed"],
                r["chart"],
                r.get("rank_fhat", "–"),
                r["rank_u_chart"],
                c["0.05"],
                c["0.02"],
                c["0.01"],
                _fmt(r.get("oracle_sqrt_floor")),
            ]
        )
    return md_table(
        ["bed", "chart", "rank f̂ (1 %)", "rank u (1 %)", "atoms 5 %", "atoms 2 %", "atoms 1 %", "oracle √L_conf"], body
    )


def table_g0(rows: list[dict[str, Any]]) -> str:
    """G0: the obstruction of every bed."""
    body = []
    for r in sorted(
        (r for r in rows if "obstruction_kind" in r and r.get("arm") == "analytic"), key=lambda r: (r["d"], r["bed"])
    ):
        body.append(
            [
                r["bed"],
                r["d"],
                r["D"],
                r["obstruction_kind"],
                r.get("obstruction_mean"),
                r.get("obstruction_max"),
                "yes" if r.get("conformal_chart_known") else "no",
            ]
        )
    return md_table(["bed", "d", "D", "obstruction", "share (mean)", "share (max)", "conformal chart known"], body)


def table_g1(rows: list[dict[str, Any]]) -> str:
    """G1: contraction against the proxy, and the growth with d."""
    a = [r for r in rows if "rho" in r]
    b = [r for r in rows if "distortion" in r]
    body = [
        [
            r["bed"],
            r["s"],
            r["rho"],
            r["max_dev"],
            r["rho_over_maxdev"],
            r["depth_1e-3"],
            r["depth_pred_1e-3"],
            r["naive_error"],
        ]
        for r in sorted(a, key=lambda r: (r["d"], r["bed"], r["s"]))
    ]
    t1 = md_table(["bed", "height s", "ρ", "max|A−I|", "ρ / max|A−I|", "depth 1e-3", "predicted", "naive error"], body)
    body2 = [
        [r["distortion"], r["d"], r["mean_dev"], r["max_dev"], r["growth_vs_d2"]]
        for r in sorted(b, key=lambda r: (r["distortion"], r["d"]))
    ]
    t2 = md_table(["distortion", "d", "mean ‖A−I‖", "max ‖A−I‖", "growth vs d = 2"], body2)
    return t1 + "\n\n" + t2


def table_g2(rows: list[dict[str, Any]]) -> str:
    """G2: gauge residuals and sensitivity."""
    body = []
    for r in sorted(rows, key=lambda r: (r["d"], r["bed"])):
        amp = r["amplification"][-1]
        body.append(
            [
                r["bed"],
                r["d"],
                r["s"],
                r["identity_fd_residual"],
                r["round_trip_error"],
                amp["delta"],
                amp["error"],
                amp["ratio"],
            ]
        )
    return md_table(["bed", "d", "s", "FD residual of the identity", "round trip", "δλ/λ (RMS)", "error of u", "ratio"], body)


def table_g4(rows: list[dict[str, Any]]) -> str:
    """G4: oracle floors."""
    body = []
    for r in sorted(rows, key=lambda r: (r["d"], r["bed"])):
        c, o = r["conf"], r["orth"]
        body.append(
            [
                r["bed"],
                r["d"],
                r["obstruction_kind"],
                r["obstruction_mean"],
                c["start"] ** 0.5,
                c["sqrt_floor"],
                c.get("known_chart_floor_sqrt"),
                c["removed_fraction"],
                "" if c["converged"] else "no",
                o["sqrt_floor"],
            ]
        )
    return md_table(
        [
            "bed",
            "d",
            "obstruction",
            "share",
            "√L_conf param chart",
            "oracle √L_conf",
            "known chart",
            "removed",
            "converged",
            "oracle √L_orth",
        ],
        body,
    )


def _group_o1(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["bed"], int(r["D"]), str(r["inputs"])), []).append(r)
    return groups


def _at_anchors(r: dict[str, Any], n: int) -> float:
    if n == 64:
        return float(r["nrmse_test_dmax"])
    return next((float(c["error"]) for c in r.get("prune_curve", []) if c["anchors"] == n), float("nan"))


def table_o1(rows: list[dict[str, Any]]) -> str:
    """O1: oracle brackets, median over seeds along the pruning curve, with the seed range at 64 anchors."""
    import statistics as st

    body = []
    for (bed, dim, inputs), rs in sorted(_group_o1(rows).items(), key=lambda kv: (kv[1][0]["d"], kv[0])):
        e64 = [float(r["nrmse_test_dmax"]) for r in rs]
        atoms = [r["chart_cost"]["bumps"]["0.05"] for r in rs]
        body.append(
            [
                bed,
                rs[0]["d"],
                dim,
                inputs if inputs != "param" else ("param (" + str(rs[0].get("param_chart", "")).split(" (")[0] + ")"),
                rs[0].get("encoder", ""),
                f"{st.median(e64):.4f} ({min(e64):.4f}–{max(e64):.4f})",
                st.median(_at_anchors(r, 32) for r in rs),
                st.median(_at_anchors(r, 16) for r in rs),
                st.median(_at_anchors(r, 8) for r in rs),
                "/".join("–" if a is None else str(a) for a in atoms),
                st.median(float(r["report_test"]["full"]["sqrt_conf"]) for r in rs),
                st.median(int(r["d_eff_elbow"]) for r in rs),
                len(rs),
            ]
        )
    return md_table(
        [
            "bed",
            "d",
            "D",
            "inputs",
            "encoder",
            "NRMSE @64 (range)",
            "@32",
            "@16",
            "@8",
            "atoms 5 % (seeds)",
            "√L_conf",
            "d_eff",
            "n",
        ],
        body,
    )


def figure_o1(rows: list[dict[str, Any]], label: str) -> None:
    """NRMSE against anchors per bed: the frozen conformal chart, the frozen parameter chart and the learned chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = _group_o1(rows)
    beds = sorted(
        {(k[0], k[1]) for k in groups if k[2] == "ambient"}, key=lambda b: (groups[(b[0], b[1], "ambient")][0]["d"], b)
    )
    n = len(beds)
    cols = 4
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(3.4 * cols, 2.9 * ((n + cols - 1) // cols)), squeeze=False)
    anchors = [64, 48, 32, 24, 16, 12, 8, 6, 4]
    style = {
        "conformal": ("tab:green", "frozen conformal chart"),
        "param": ("tab:orange", "frozen parameter chart"),
        "ambient": ("tab:blue", "learned chart (IGL today)"),
    }
    for ax, (bed, dim) in zip(axes.flat, beds, strict=False):
        for inputs, (color, name) in style.items():
            key = next((k for k in groups if k[0] == bed and k[2] == inputs and (inputs != "ambient" or k[1] == dim)), None)
            if key is None:
                continue
            rs = groups[key]
            med = [sorted(_at_anchors(r, a) for r in rs)[len(rs) // 2] for a in anchors]
            lo = [min(_at_anchors(r, a) for r in rs) for a in anchors]
            hi = [max(_at_anchors(r, a) for r in rs) for a in anchors]
            ax.plot(anchors, med, "-o", ms=3, color=color, label=name)
            ax.fill_between(anchors, lo, hi, color=color, alpha=0.15)
        ax.set_title(f"{bed}  d = {groups[(bed, dim, 'ambient')][0]['d']}, D = {dim}", fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("anchors kept")
        ax.set_ylabel("test NRMSE")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    out = Path.cwd() / "results" / "benchmarks" / "geometry" / label / "o1_oracle_bracket" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "o1_oracle_bracket.png", dpi=150)
    print("figure:", out / "o1_oracle_bracket.png")


def _paired(rows: list[dict[str, Any]], base: list[dict[str, Any]], key: str = "nrmse_test_dmax") -> dict[str, Any]:
    """Per-seed deltas of ``key`` against the ``none`` arm: median, min, max and the sign agreement."""
    b = {r["seed"]: float(r[key]) for r in base}
    deltas = [float(r[key]) - b[r["seed"]] for r in rows if r["seed"] in b]
    if not deltas:
        return {"median": float("nan"), "min": float("nan"), "max": float("nan"), "worse": 0, "n": 0}
    return {
        "median": sorted(deltas)[len(deltas) // 2],
        "min": min(deltas),
        "max": max(deltas),
        "worse": sum(d > 0 for d in deltas),
        "n": len(deltas),
    }


def table_x1(rows: list[dict[str, Any]]) -> str:
    """X1: penalty arms per bed, medians over seeds, paired deltas against ``none``."""
    import statistics as st

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["bed"], r["arm"]), []).append(r)
    beds = sorted({b for b, _ in groups}, key=lambda b: (groups[(b, "none")][0]["d"], b))
    order = ["none", "orth-today", "orth-matched", "conf", "conf@0.1", "conf@0.03", "conf-tan", "conf-plain", "beta-0.5"]
    body = []
    for b in beds:
        base = groups.get((b, "none"), [])
        for a in sorted({a for bb, a in groups if bb == b}, key=lambda a: order.index(a) if a in order else 99):
            rs = groups[(b, a)]
            e64 = [float(r["nrmse_test_dmax"]) for r in rs]
            pd = _paired(rs, base)
            rt = [r["report_test"]["full"] for r in rs]
            w = rs[0].get("penalty_weight")
            body.append(
                [
                    b,
                    rs[0]["d"],
                    rs[0]["D"],
                    a,
                    "–" if w is None else f"{w:.3g}",
                    f"{st.median(e64):.4f} ({min(e64):.4f}–{max(e64):.4f})",
                    st.median(_at_anchors(r, 16) for r in rs),
                    st.median(_at_anchors(r, 8) for r in rs),
                    "–" if a == "none" else f"{pd['median']:+.4f} [{pd['min']:+.4f}, {pd['max']:+.4f}] {pd['worse']}/{pd['n']}",
                    st.median(float(x["sqrt_orth"]) for x in rt),
                    st.median(float(x["sqrt_conf"]) for x in rt),
                    st.median(float(x["norm_inf"]) for x in rt),
                    st.median(int(r["d_eff_elbow"]) for r in rs),
                    len(rs),
                ]
            )
    return md_table(
        [
            "bed",
            "d",
            "D",
            "arm",
            "weight",
            "NRMSE @64 (range)",
            "@16",
            "@8",
            "Δ vs none @64 (median [min, max], worse/n)",
            "√L_orth",
            "√L_conf",
            "‖A−I‖∞",
            "d_eff",
            "n",
        ],
        body,
    )


def figure_x1(rows: list[dict[str, Any]], label: str) -> None:
    """Paired deltas against ``none`` per bed and arm (dots per seed), and the conformality floor per arm."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["bed"], r["arm"]), []).append(r)
    beds = sorted({b for b, _ in groups}, key=lambda b: (groups[(b, "none")][0]["d"], b))
    arms = [
        a
        for a in ["orth-today", "orth-matched", "conf", "conf@0.1", "conf@0.03", "conf-tan"]
        if any(a == aa for _, aa in groups)
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 0.6 * len(beds) + 2.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, b in enumerate(beds):
        base = {r["seed"]: float(r["nrmse_test_dmax"]) for r in groups[(b, "none")]}
        for j, a in enumerate(arms):
            rs = groups.get((b, a), [])
            ys = [i + (j - (len(arms) - 1) / 2) * 0.12] * len(rs)
            xs = [float(r["nrmse_test_dmax"]) / base[r["seed"]] - 1.0 for r in rs if r["seed"] in base]
            ax1.scatter(xs[: len(ys)], ys[: len(xs)], color=colors[j % len(colors)], s=18, label=a if i == 0 else None)
            fl = [float(r["report_test"]["full"]["sqrt_conf"]) for r in rs]
            ax2.scatter(fl, ys[: len(fl)], color=colors[j % len(colors)], s=18, label=a if i == 0 else None)
        fl0 = [float(r["report_test"]["full"]["sqrt_conf"]) for r in groups[(b, "none")]]
        ax2.scatter(fl0, [i] * len(fl0), color="k", s=18, marker="x", label="none" if i == 0 else None)
    for ax in (ax1, ax2):
        ax.set_yticks(range(len(beds)))
        ax.set_yticklabels(beds, fontsize=8)
        ax.grid(alpha=0.3)
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_xlabel("relative change of test NRMSE at 64 anchors against the none arm (same seed)")
    ax2.set_xlabel("√L_conf of the learned chart (test set)")
    ax1.legend(fontsize=7)
    fig.tight_layout()
    out = Path.cwd() / "results" / "benchmarks" / "geometry" / label / "x1_penalty" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "x1_penalty.png", dpi=150)
    print("figure:", out / "x1_penalty.png")


FIGURES = {"o1_oracle_bracket": figure_o1, "x1_penalty": figure_x1}

TABLES = {
    "o1_oracle_bracket": table_o1,
    "x1_penalty": table_x1,
    "r0_rank_go_nogo": table_r0,
    "g0_geometry": table_g0,
    "g1_rho_law": table_g1,
    "g2_gauge": table_g2,
    "g4_oracle_floor": table_g4,
}


def main() -> None:
    """Print the tables."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--blocks", default=",".join(TABLES))
    p.add_argument("--figures", action="store_true", help="also draw the figures of the blocks that have one")
    args = p.parse_args()
    for block in args.blocks.split(","):
        rows = load(args.label, block)
        if not rows:
            continue
        print(f"\n### {block}\n")
        print(TABLES[block](rows))
        if args.figures and block in FIGURES:
            FIGURES[block](rows, args.label)


__all__ = ["load", "md_table"]

if __name__ == "__main__":
    main()
