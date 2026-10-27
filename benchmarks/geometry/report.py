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


TABLES = {
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
    args = p.parse_args()
    for block in args.blocks.split(","):
        rows = load(args.label, block)
        if not rows:
            continue
        print(f"\n### {block}\n")
        print(TABLES[block](rows))


__all__ = ["load", "md_table"]

if __name__ == "__main__":
    main()
