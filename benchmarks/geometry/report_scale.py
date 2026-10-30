"""Tables and figures of the scale blocks (W1, S0–S3), registered into :mod:`benchmarks.geometry.report`.

Paired comparisons are made on the same test rows: two arms sharing a seed were trained on the same minibatches
and scored on the same split, so their per-row correctness vectors are paired. :func:`paired_tests` pools the
seeds (each seed contributes its rows), bootstraps the difference of error rates over rows and runs McNemar's
exact test on the discordant rows. Thresholds come from the measured noise floor: the seed SD of the ``none``
arm, with the minimum detectable paired effect at 80 % power quoted as ``3.6 SD`` for three seeds and ``1.9 SD``
for five (paired t, two-sided 5 %).
"""

from __future__ import annotations

import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["FIGURES_SCALE", "TABLES_SCALE", "paired_tests"]

MDE_FACTOR = {3: 3.6, 5: 1.9}
_RNG_SEED = 0


def _fmt(v: Any) -> str:
    from benchmarks.geometry.report import _fmt as base

    return base(v)


def _md(headers: list[str], rows: list[list[Any]]) -> str:
    from benchmarks.geometry.report import md_table

    return md_table(headers, rows)


def _metric(r: dict[str, Any]) -> float:
    return float(r["err_test"]) if r.get("task") == "classification" else float(r["nrmse_test_dmax"])


def _wrong(r: dict[str, Any]) -> list[int] | None:
    top = (r.get("score_test") or {}).get(str(r.get("k") or r.get("d")))
    if top is None:
        top = next(iter((r.get("score_test") or {}).values()), None)
    if top is None:
        return r.get("wrong")
    return top.get("wrong")


def paired_tests(
    pairs: list[tuple[list[int], list[int], int]], *, n_boot: int = 2000, seed: int = _RNG_SEED
) -> dict[str, float]:
    """Pooled paired comparison of two arms from their wrong-row lists ``(wrong_a, wrong_b, n_test)`` per seed.

    Returns the difference of error rates ``a − b`` (positive: ``a`` worse), its 95 % bootstrap interval over the
    pooled rows, and the two-sided exact McNemar p-value on the discordant rows.
    """
    from scipy.stats import binomtest

    a_vec, b_vec = [], []
    for wa, wb, n in pairs:
        ca = np.ones(n, dtype=bool)
        ca[np.asarray(wa, dtype=int)] = False
        cb = np.ones(n, dtype=bool)
        cb[np.asarray(wb, dtype=int)] = False
        a_vec.append(ca)
        b_vec.append(cb)
    ca, cb = np.concatenate(a_vec), np.concatenate(b_vec)
    diff = cb.astype(float) - ca.astype(float)  # +1 where a is wrong and b right
    delta = float(diff.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    b = int(((~ca) & cb).sum())  # a wrong, b right
    c = int((ca & (~cb)).sum())  # a right, b wrong
    p = float(binomtest(b, b + c, 0.5).pvalue) if b + c > 0 else 1.0
    return {"delta": delta, "ci_lo": lo, "ci_hi": hi, "mcnemar_p": p, "discordant": b + c, "rows": int(len(diff))}


def _paired_vs(rows: list[dict[str, Any]], base: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-seed deltas and the pooled paired tests of ``rows`` against ``base`` (same seeds)."""
    b = {r["seed"]: r for r in base}
    deltas: list[float] = []
    pairs: list[tuple[list[int], list[int], int]] = []
    for r in rows:
        if r["seed"] not in b:
            continue
        deltas.append(_metric(r) - _metric(b[r["seed"]]))
        wa, wb = _wrong(r), _wrong(b[r["seed"]])
        n = r.get("n_test_rows")
        if wa is not None and wb is not None and n:
            pairs.append((wa, wb, int(n)))
    out: dict[str, Any] = {
        "median": st.median(deltas) if deltas else float("nan"),
        "min": min(deltas) if deltas else float("nan"),
        "max": max(deltas) if deltas else float("nan"),
        "worse": sum(d > 0 for d in deltas),
        "n": len(deltas),
    }
    if pairs:
        out.update(paired_tests(pairs))
    return out


def _delta_str(d: dict[str, Any]) -> str:
    if not d.get("n"):
        return "–"
    if "ci_lo" in d:
        # The pooled paired difference over the shared test rows, its bootstrap interval and McNemar; then the
        # per-seed view (median delta and how many seeds are worse).
        return (
            f"{d['delta']:+.4f} CI [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}], McNemar p {d['mcnemar_p']:.2g}; "
            f"seeds {d['worse']}/{d['n']} worse, median {d['median']:+.4f}"
        )
    return f"{d['median']:+.4f} ({d['worse']}/{d['n']} worse)"


def _mean_sd(rows: list[dict[str, Any]]) -> tuple[float, float, int]:
    v = [_metric(r) for r in rows]
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v)) if v else (float("nan"), float("nan"), 0)


def _cell(r: dict[str, Any]) -> tuple[str, int, int]:
    return (str(r.get("set", r.get("bed"))), int(r.get("k") or r.get("d")), int(r.get("R") or 0))


def _with_n_test(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The number of test rows of every row, from its envelope when the block wrote it, else from the wrong list."""
    for r in rows:
        if r.get("n_test_rows") is None:
            r["n_test_rows"] = r.get("n_test") or len(r.get("wrong") or []) or None
    return rows


# ------------------------------------------------------------------------------------------------- W1


def table_w1(rows: list[dict[str, Any]]) -> str:
    """W1: the noise floor and its thresholds, the frozen PCA chart, the inner-solve and ridge checks."""
    rows = _with_n_test(rows)
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[_cell(r)].append(r)
    body: list[list[Any]] = []
    for cell in sorted(groups):
        g = groups[cell]
        none = [r for r in g if r["arm"] == "none"]
        m, sd, n = _mean_sd(none)
        body.append(
            [*cell, "none", n, f"{m:.4f} ± {sd:.4f}", f"MDE₃ {MDE_FACTOR[3] * sd:.4f}, MDE₅ {MDE_FACTOR[5] * sd:.4f}", "–"]
        )
        for arm in sorted({r["arm"] for r in g} - {"none"}):
            sub = [r for r in g if r["arm"] == arm]
            m2, sd2, n2 = _mean_sd(sub)
            body.append([*cell, arm, n2, f"{m2:.4f} ± {sd2:.4f}", "", _delta_str(_paired_vs(sub, none))])
    return _md(["set", "k", "R", "arm", "n", "error (mean ± sd)", "thresholds", "Δ vs none (paired)"], body)


# ------------------------------------------------------------------------------------------------- S0


def table_s0(rows: list[dict[str, Any]]) -> str:
    """S0: baselines per set; the headroom over the linear probe."""
    rows = _with_n_test(rows)
    by_set: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_set[str(r.get("set"))].append(r)
    body: list[list[Any]] = []
    for s in sorted(by_set):
        g = by_set[s]
        single = {r["arm"]: _metric_any(r) for r in g if r["arm"] in ("linear", "knn")}
        best_mlp = min((_metric_any(r) for r in g if r["arm"] == "mlp-head"), default=float("nan"))
        headroom = single.get("linear", float("nan")) - min(single.get("knn", float("inf")), best_mlp)
        body.append([s, "linear probe", "–", "–", _fmt(single.get("linear")), f"headroom {headroom:+.4f}"])
        body.append([s, "kNN", "–", "–", _fmt(single.get("knn")), ""])
        cells: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
        for r in g:
            if r["arm"] in ("mlp-head", "frozen-pca", "nystroem"):
                cells[(int(r.get("k") or 0), int(r.get("R") or 0), r["arm"])].append(r)
        for k, rr, arm in sorted(cells):
            v = [_metric_any(r) for r in cells[(k, rr, arm)]]
            body.append([s, arm, k, rr, f"{st.mean(v):.4f} ± {st.stdev(v) if len(v) > 1 else 0:.4f}", f"n {len(v)}"])
    return _md(["set", "baseline", "k", "R", "error", "note"], body)


def _metric_any(r: dict[str, Any]) -> float:
    if "err_test" in r and r["err_test"] is not None:
        return float(r["err_test"])
    v = r.get("nrmse_test")
    return float(v) if isinstance(v, int | float) else float(r.get("nrmse_test_dmax", float("nan")))


# ------------------------------------------------------------------------------------------------- S1


def _r_eq(curve: dict[int, tuple[float, float]], target: float, tol: float) -> int | None:
    """The smallest ``R`` whose mean error is within ``tol`` of ``target``."""
    for r in sorted(curve):
        if curve[r][0] <= target + tol:
            return r
    return None


def table_s1(rows: list[dict[str, Any]]) -> str:
    """S1: error over the anchor ladder per (set, k, placement); the anchor-limited gap; data anchors vs random."""
    rows = _with_n_test(rows)
    groups: dict[tuple[str, int, str], dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        s, k, rr = _cell(r)
        groups[(s, k, r["arm"])][rr].append(r)
    all_r = sorted({rr for g in groups.values() for rr in g})
    # the common target of R_eq: the best mean error at the largest R over the placements of the same (set, k)
    best_at_max: dict[tuple[str, int], tuple[float, float]] = {}
    for (s, k, _init), g in groups.items():
        if all_r and all_r[-1] in g:
            m, sd, _ = _mean_sd(g[all_r[-1]])
            if (s, k) not in best_at_max or m < best_at_max[(s, k)][0]:
                best_at_max[(s, k)] = (m, sd)
    body: list[list[Any]] = []
    for s, k, init in sorted(groups):
        g = groups[(s, k, init)]
        curve = {rr: _mean_sd(g[rr])[:2] for rr in g}
        cells = [f"{curve[rr][0]:.4f}" if rr in curve else "–" for rr in all_r]
        gap = "–"
        if 32 in g and all_r and all_r[-1] in g:
            gap = _delta_str(_paired_vs(g[32], g[all_r[-1]]))
        tgt = best_at_max.get((s, k))
        r_eq = _r_eq(curve, tgt[0], MDE_FACTOR[5] * tgt[1]) if tgt else None
        vs_random = "–"
        if init != "random" and (s, k, "random") in groups:
            at = [rr for rr in (32, 64) if rr in g and rr in groups[(s, k, "random")]]
            if at:
                vs_random = "; ".join(f"R{rr}: " + _delta_str(_paired_vs(g[rr], groups[(s, k, "random")][rr])) for rr in at)
        body.append([s, k, init, *cells, gap, r_eq, vs_random])
    return _md(["set", "k", "anchors", *[f"R={rr}" for rr in all_r], "err(32) − err(R_max) paired", "R_eq", "vs random"], body)


# ------------------------------------------------------------------------------------------------- S2


def _ratio_at(r: dict[str, Any]) -> str:
    g = r.get("grad_ratios") or []
    return " / ".join(f"{x['ratio']:.2f}" for x in g) if g else "–"


def table_s2(rows: list[dict[str, Any]]) -> str:
    """S2: arms against ``none`` per (set, k, R): paired deltas with CIs and McNemar, ratios, floors, stability."""
    rows = _with_n_test(rows)
    groups: dict[tuple[str, int, int], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[_cell(r)][r["arm"]].append(r)
    body: list[list[Any]] = []
    for cell in sorted(groups):
        g = groups[cell]
        none = g.get("none", [])
        for arm in sorted(g, key=lambda a: (a != "none", a)):
            sub = g[arm]
            m, sd, n = _mean_sd(sub)
            base = none if arm != "none" else []
            if arm.endswith("-2x"):
                base = g.get("none-2x", []) if arm != "none-2x" else []
            d = _paired_vs(sub, base) if base else {"n": 0}
            conf = st.mean(r["report_test"]["full"]["sqrt_conf"] for r in sub)
            unstable = sum(bool((r.get("stability") or {}).get("flag")) for r in sub)
            ratios = _ratio_at(sub[0]) if sub else "–"
            lr = [r.get("err_test_lr_readout") for r in sub if r.get("err_test_lr_readout") is not None]
            lr_s = f"{st.mean(lr):.4f}" if lr else "–"
            w = sub[0].get("penalty_weight")
            body.append(
                [*cell, arm, _fmt(w), n, f"{m:.4f} ± {sd:.4f}", _delta_str(d), lr_s, ratios, f"{conf:.3f}", f"{unstable}/{n}"]
            )
    return _md(
        [
            "set",
            "k",
            "R",
            "arm",
            "weight",
            "n",
            "error",
            "Δ vs none (paired)",
            "LR-readout err",
            "ratio @10/50/100 %",
            "√L_conf",
            "unstable",
        ],
        body,
    )


# ------------------------------------------------------------------------------------------------- S3


def table_s3(rows: list[dict[str, Any]]) -> str:
    """S3: ``none`` and ``conf`` along the training-set size; the paired gain of the penalty per N."""
    rows = _with_n_test(rows)
    groups: dict[tuple[str, int, int, int], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        s, k, rr = _cell(r)
        n_tr = int(r.get("n_train") or r.get("N"))
        arm = str(r["arm"]).split("-N")[0]
        groups[(s, k, rr, n_tr)][arm].append(r)
    body: list[list[Any]] = []
    for cell in sorted(groups):
        g = groups[cell]
        m0, sd0, n0 = _mean_sd(g.get("none", []))
        m1, sd1, n1 = _mean_sd(g.get("conf", []))
        d = _paired_vs(g.get("conf", []), g.get("none", []))
        w = (g.get("conf") or [{}])[0].get("penalty_weight")
        body.append([*cell, f"{m0:.4f} ± {sd0:.4f} (n {n0})", f"{m1:.4f} ± {sd1:.4f} (n {n1})", _fmt(w), _delta_str(d)])
    return _md(["set", "k", "R", "N_train", "none", "conf", "weight", "Δ conf − none (paired)"], body)


# ---------------------------------------------------------------------------------------------- figures


def _fig_dir(block: str, label: str) -> Path:
    from benchmarks.geometry.blocks._cli import figure_dir

    return figure_dir(block, label)


def figure_s1(rows: list[dict[str, Any]], label: str) -> None:
    """Log error against log R per set and k, one line per anchor placement, with the S0 baselines when present."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from benchmarks.geometry.report import load

    s0 = load(label, "s0_baselines")
    sets = sorted({str(r.get("set")) for r in rows})
    ks = sorted({int(r.get("k") or r.get("d")) for r in rows})
    fig, axes = plt.subplots(len(ks), len(sets), figsize=(4.5 * len(sets), 3.6 * len(ks)), squeeze=False)
    for i, k in enumerate(ks):
        for j, s in enumerate(sets):
            ax = axes[i][j]
            sub = [r for r in rows if str(r.get("set")) == s and int(r.get("k") or r.get("d")) == k]
            for init in sorted({r["arm"] for r in sub}):
                pts: dict[int, list[float]] = defaultdict(list)
                for r in sub:
                    if r["arm"] == init:
                        pts[int(r["R"])].append(_metric(r))
                xs = sorted(pts)
                ax.errorbar(
                    xs,
                    [st.mean(pts[x]) for x in xs],
                    yerr=[st.pstdev(pts[x]) for x in xs],
                    marker="o",
                    label=f"IGL {init}",
                    capsize=2,
                )
            for base_name, style in (("mlp-head", "s--"), ("frozen-pca", "^:"), ("nystroem", "x-.")):
                pts = defaultdict(list)
                for r in s0:
                    if str(r.get("set")) == s and r["arm"] == base_name and int(r.get("k") or 0) == k:
                        pts[int(r["R"])].append(_metric_any(r))
                if pts:
                    xs = sorted(pts)
                    ax.plot(xs, [st.mean(pts[x]) for x in xs], style, label=base_name, alpha=0.8)
            for base_name, ls in (("linear", "-"), ("knn", "--")):
                v = [_metric_any(r) for r in s0 if str(r.get("set")) == s and r["arm"] == base_name]
                if v:
                    ax.axhline(v[0], color="grey", linestyle=ls, linewidth=1, label=base_name)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(f"{s}, k = {k}", fontsize=9)
            ax.set_xlabel("anchors R")
            ax.set_ylabel("test error")
            ax.grid(alpha=0.3, which="both")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    out = _fig_dir("s1_anchor_budget", label) / "s1_anchor_budget.png"
    fig.savefig(out, dpi=130)
    print(f"figure: {out}")


def figure_s2(rows: list[dict[str, Any]], label: str) -> None:
    """Paired deltas against ``none`` per arm, with the pooled bootstrap interval, one panel per (set, R)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _with_n_test(rows)
    groups: dict[tuple[str, int, int], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[_cell(r)][r["arm"]].append(r)
    cells = sorted(groups)
    fig, axes = plt.subplots(1, max(1, len(cells)), figsize=(3.6 * max(1, len(cells)), 3.8), squeeze=False)
    for ax, cell in zip(axes[0], cells, strict=False):
        g = groups[cell]
        arms = [a for a in sorted(g) if a not in ("none", "none-2x")]
        for i, arm in enumerate(arms):
            base = g.get("none-2x", []) if arm.endswith("-2x") else g.get("none", [])
            d = _paired_vs(g[arm], base)
            if not d.get("n"):
                continue
            deltas = [_metric(r) - _metric(b) for r in g[arm] for b in base if b["seed"] == r["seed"]]
            ax.scatter([i] * len(deltas), deltas, s=14, alpha=0.7)
            if "ci_lo" in d:
                ax.errorbar(
                    [i],
                    [d["delta"]],
                    yerr=[[d["delta"] - d["ci_lo"]], [d["ci_hi"] - d["delta"]]],
                    fmt="_",
                    color="k",
                    capsize=4,
                )
        ax.axhline(0, color="grey", linewidth=1)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms, rotation=30, fontsize=7)
        ax.set_title(f"{cell[0]}, k = {cell[1]}, R = {cell[2]}", fontsize=9)
        ax.set_ylabel("Δ error vs none (per seed; bar: pooled 95 % CI)")
    fig.tight_layout()
    out = _fig_dir("s2_penalty_at_scale", label) / "s2_penalty_at_scale.png"
    fig.savefig(out, dpi=130)
    print(f"figure: {out}")


def figure_s3(rows: list[dict[str, Any]], label: str) -> None:
    """Learning curves of ``none`` and ``conf`` per set."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = sorted({str(r.get("set")) for r in rows})
    fig, axes = plt.subplots(1, len(sets), figsize=(4.2 * len(sets), 3.6), squeeze=False)
    for ax, s in zip(axes[0], sets, strict=False):
        for arm, style in (("none", "o-"), ("conf", "s--")):
            pts: dict[int, list[float]] = defaultdict(list)
            for r in rows:
                if str(r.get("set")) == s and str(r["arm"]).startswith(arm):
                    pts[int(r.get("n_train") or r.get("N"))].append(_metric(r))
            xs = sorted(pts)
            if xs:
                ax.errorbar(
                    xs, [st.mean(pts[x]) for x in xs], yerr=[st.pstdev(pts[x]) for x in xs], fmt=style, label=arm, capsize=2
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("training rows")
        ax.set_ylabel("test error")
        ax.set_title(s, fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = _fig_dir("s3_learning_curve", label) / "s3_learning_curve.png"
    fig.savefig(out, dpi=130)
    print(f"figure: {out}")


TABLES_SCALE = {
    "w1_gate": table_w1,
    "s0_baselines": table_s0,
    "s1_anchor_budget": table_s1,
    "s2_penalty_at_scale": table_s2,
    "s3_learning_curve": table_s3,
}
FIGURES_SCALE = {"s1_anchor_budget": figure_s1, "s2_penalty_at_scale": figure_s2, "s3_learning_curve": figure_s3}
