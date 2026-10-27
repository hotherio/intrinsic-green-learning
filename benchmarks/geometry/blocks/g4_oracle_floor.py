"""G4: the oracle floor of the chart penalties, by optimising a chart against the analytic metric (no readout, no encoder).

A chart ``psi_theta: patch -> R^d`` (identity plus a small MLP, zero at initialisation) is trained on
sampled parameter points to minimise the axis-normalised conformality loss of the inverse metric it
induces, ``A = D psi g^{-1} D psi^T`` (the matrix IGL's penalty sees), and separately the orthogonality
target (``beta = 0``) and the half-way ``beta = 0.5``. The floor reached, with its training curve,
separates **obstruction** (the geometry forbids a conformal chart: Weyl or Cotton non-zero) from
**optimisation** (an encoder that does not find the chart). Predictions: zero in ``d = 2``; zero in
``d = 3`` iff the Cotton tensor vanishes; zero in ``d >= 4`` iff the Weyl tensor vanishes; and the
fraction of ``|A - cI|`` removed against the count ``2/(d+1)``.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn
from torch.func import jacrev, vmap

from benchmarks.geometry._harness import write_block_result
from benchmarks.geometry.blocks._cli import figure_dir, gate, parser
from benchmarks.geometry.testbeds import manifolds, spd_beds
from igl.spd import conformality_loss

BLOCK = "g4_oracle_floor"


class Chart(nn.Module):
    """``psi(t) = t + MLP(t)`` with the last layer zero at initialisation: the identity chart, free to deform."""

    def __init__(self, d: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, d))
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.double()

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return t + self.net(t)


def inverse_metric_in_chart(chart: Chart, t: torch.Tensor, ginv_param: torch.Tensor) -> torch.Tensor:
    """``A = D psi g^{-1} D psi^T`` at every point: the inverse metric in the new chart, ``[n, d, d]``."""
    jac = vmap(jacrev(chart))(t)  # [n, d, d]
    return jac @ ginv_param @ jac.transpose(1, 2)


def fit_chart(
    bed: manifolds.Bed, *, beta: float, n_points: int, steps: int, seed: int = 0, lr: float = 3e-3
) -> tuple[Chart, dict[str, Any]]:
    """Train the chart; return it with the floor, the training curve and the fraction of the initial deviation removed."""
    torch.manual_seed(seed)
    t, _ = bed.sample(n_points, seed=seed)
    g_param = torch.stack([bed.metric(ti) for ti in t])
    ginv_param = torch.linalg.inv(g_param)
    chart = Chart(bed.d)
    opt = torch.optim.Adam(chart.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr / 100)
    start = float(conformality_loss(ginv_param, beta=beta))
    curve: list[float] = []
    best_loss, best_step = start, 0  # the identity chart (step 0) is a valid chart: the floor never exceeds ``start``
    best_state = {k: v.detach().clone() for k, v in chart.state_dict().items()}
    for step in range(steps):
        a = inverse_metric_in_chart(chart, t, ginv_param)
        loss = conformality_loss(a, beta=beta)
        value = float(loss.detach())
        if value < best_loss:
            best_loss, best_step = value, step
            best_state = {k: v.detach().clone() for k, v in chart.state_dict().items()}
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % max(1, steps // 50) == 0 or step == steps - 1:
            curve.append(value)
    chart.load_state_dict(best_state)
    with torch.no_grad():
        a = inverse_metric_in_chart(chart, t, ginv_param)
        floor = float(conformality_loss(a, beta=beta))
        conf_floor = float(conformality_loss(a, beta=1.0))
        orth_floor = float(conformality_loss(a, beta=0.0))
    tail = curve[-5:]
    converged = (max(tail) - min(tail)) < 0.05 * max(tail[-1], 1e-12) + 1e-9
    info = {
        "beta": beta,
        "start": start,
        "floor": floor,
        "sqrt_floor": floor**0.5,
        "conf_floor_sqrt": conf_floor**0.5,
        "orth_floor_sqrt": orth_floor**0.5,
        "removed_fraction": 1.0 - (floor / start) ** 0.5 if start > 1e-8 else 1.0,
        "already_conformal": start <= 1e-8,
        "curve": curve,
        "converged": converged,
        "best_step": best_step,
        "optimiser_drifted": best_step < steps - steps // 10,
    }
    return chart, info


def optimise(
    bed: manifolds.Bed, *, beta: float, n_points: int, steps: int, seed: int = 0, lr: float = 5e-3, restarts: int = 2
) -> dict[str, Any]:
    """The best of ``restarts`` runs of :func:`fit_chart` (JSON-serialisable), bounded by the known conformal chart.

    Where the bed exposes its conformal chart, the floor of that chart is computed too and the reported floor
    is the smaller of the two: the optimiser can only over-estimate the true floor.
    """
    best: dict[str, Any] | None = None
    for r in range(restarts):
        info = fit_chart(bed, beta=beta, n_points=n_points, steps=steps, seed=seed + r, lr=lr)[1]
        if best is None or info["floor"] < best["floor"]:
            best = info
    assert best is not None
    best["restarts"] = restarts
    known = known_chart_floor(bed, beta=beta, n_points=n_points, seed=seed)
    best["known_chart_floor_sqrt"] = known
    if known is not None and known**2 < best["floor"]:
        best["optimiser_gap_sqrt"] = best["sqrt_floor"] - known
        best["floor"] = known**2
        best["sqrt_floor"] = known
        best["removed_fraction"] = 1.0 - (best["floor"] / best["start"]) ** 0.5 if best["start"] > 1e-8 else 1.0
    return best


def known_chart_floor(bed: manifolds.Bed, *, beta: float, n_points: int, seed: int = 0) -> float | None:
    """``sqrt`` of the penalty at the bed's known conformal chart (``None`` when the bed has none)."""
    if bed.conformal_chart is None:
        return None
    torch.manual_seed(seed)
    t, _ = bed.sample(n_points, seed=seed)
    ginv_param = torch.linalg.inv(torch.stack([bed.metric(ti) for ti in t]))
    chart = bed.conformal_chart
    try:
        jac = torch.stack([jacrev(chart)(ti) for ti in t])  # [n, d, d]
    except (RuntimeError, TypeError):
        # a chart given by interpolation (the hills): finite-difference Jacobian
        eps = 1e-4
        cols = []
        for i in range(bed.d):
            e = torch.zeros(bed.d, dtype=t.dtype)
            e[i] = eps
            cols.append((chart(t + e) - chart(t - e)) / (2 * eps))
        jac = torch.stack(cols, dim=-1)
    a = jac @ ginv_param @ jac.transpose(1, 2)
    return float(conformality_loss(a, beta=beta)) ** 0.5


def beds_for(*, smoke: bool) -> list[manifolds.Bed]:
    """Analytic beds of the matrix for the oracle floor."""
    cat = manifolds.catalogue()
    if smoke:
        return [manifolds.hill_surface("lop"), manifolds.sphere_cap(3, axes=(1.0, 1.0, 2.0, 3.0)), cat["S2xR2"]]
    names = ["hill-round", "hill-lop", "S2cap", "S4cap", "S3xR", "S2xR2", "S2xS2", "T4", "hillxhill", "S8cap", "S4xS4"]
    beds = [cat[n] for n in names] + [b for b in cat.values() if "axes" in b.name]
    beds += [manifolds.sphere_cap(3), manifolds.sphere_cap(3, axes=(1.0, 1.0, 2.0, 3.0)), manifolds.cylinder(3)]
    beds += [manifolds.graph_hypersurface(d, b) for d in (2, 3, 4) for b in ("radial", "generic")]
    beds += [spd_beds.spd_bed(2), spd_beds.spd_bed(3), spd_beds.spd_bed(2, metric="logeuclid")]
    return beds


def figure(rows: list[dict[str, Any]], label: str | None) -> None:
    """Oracle conformality floor against the obstruction share, and removed fraction against ``d`` with ``2/(d+1)``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    markers = {2: "o", 3: "s", 4: "^", 6: "D", 8: "v"}
    for r in rows:
        ax1.scatter(r["obstruction_mean"], r["conf"]["sqrt_floor"], marker=markers.get(r["d"], "x"), label=f"d = {r['d']}")
        ax1.annotate(r["bed"], (r["obstruction_mean"], r["conf"]["sqrt_floor"]), fontsize=6, alpha=0.7)
    handles, labels = ax1.get_legend_handles_labels()
    uniq = dict(zip(labels, handles, strict=False))
    ax1.legend(uniq.values(), uniq.keys(), fontsize=8)
    ax1.set_xlabel("obstruction share (Weyl in d ≥ 4, Cotton in d = 3)")
    ax1.set_ylabel("oracle √L_conf floor")
    for r in rows:
        ax2.scatter(r["d"], r["conf"]["removed_fraction"], marker=markers.get(r["d"], "x"))
    ds = sorted({r["d"] for r in rows})
    ax2.plot(ds, [2 / (d + 1) for d in ds], "k--", lw=0.8, label="2/(d+1)")
    ax2.set_xlabel("d")
    ax2.set_ylabel("fraction of √L_conf removed by the best chart")
    ax2.legend()
    fig.tight_layout()
    out = figure_dir(BLOCK, label) / "g4_oracle_floor.png"
    fig.savefig(out, dpi=150)
    print("figure:", out)


def main() -> None:
    """Run the block."""
    p = parser(__doc__ or "")
    p.add_argument("--beds", default=None, help="comma-separated bed names to (re)run; default: every bed of the block")
    args = p.parse_args()
    gate(args)
    t0 = time.perf_counter()
    steps = 150 if args.smoke else 3000
    n_points = 256 if args.smoke else 2048
    rows: list[dict[str, Any]] = []
    beds = beds_for(smoke=args.smoke)
    if args.beds:
        wanted = {n.strip() for n in str(args.beds).split(",") if n.strip()}
        beds = [b for b in beds if b.name in wanted]
    for bed in beds:
        t1 = time.perf_counter()
        obs = manifolds.obstruction(bed, n_points=3 if args.smoke else 6)
        res = {
            name: optimise(bed, beta=b, n_points=n_points, steps=steps, restarts=1 if (args.smoke or name != "conf") else 2)
            for name, b in (("conf", 1.0), ("orth", 0.0), ("half", 0.5))
        }
        row = {
            "bed": bed.name,
            "d": bed.d,
            "D": bed.ambient,
            "kind": "analytic",
            "obstruction_kind": obs["kind"],
            "obstruction_mean": obs["mean"],
            "conformal_chart_known": bed.conformal_chart is not None,
            **res,
            "seconds": time.perf_counter() - t1,
        }
        rows.append(row)
        write_block_result(BLOCK, bed.name, "oracle", 0, row, label=args.label)
        c, o = res["conf"], res["orth"]
        flag = "" if c["converged"] else " (not converged)"
        print(
            f"{bed.name:22s} d={bed.d} {obs['kind']:6s} {obs['mean']:.3f} | conf: start {c['start'] ** 0.5:.3f} "
            f"floor {c['sqrt_floor']:.4f} removed {c['removed_fraction']:.2f}{flag} | orth floor {o['sqrt_floor']:.4f} "
            f"(its conf {o['conf_floor_sqrt']:.3f}) | {row['seconds']:.0f}s",
            flush=True,
        )
    if args.beds:
        # a partial run: the figure covers every bed saved under the label, not only the ones rerun
        from benchmarks.device._harness import git_short_sha
        from benchmarks.geometry.report import load

        rows = load(args.label or git_short_sha(), BLOCK)
    figure(rows, args.label)
    print(f"G4 done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
