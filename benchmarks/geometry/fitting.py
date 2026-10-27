"""The learned-chart protocol shared by every tier 1+ block: one IGL fit per (bed, arm, seed), scored the same way.

Fixed by the plan: the default encoder with Stiefel init, 64 anchors x 4 scales, 400 epochs, batch 256,
``source_l2 = 1e-3``, no scheduler, no early stopping, no gradient clipping; the same data tensors for every
arm; the readout solved on the full training set and scored on the test set (never ``eval_dimension_curve``
on test); both penalty floors, the geometry report, ``d_eff`` and the chart-only costs recorded for every run.
A paired design: arms sharing a seed see the same minibatches, because the sampler draws from the global RNG
that ``torch.manual_seed`` resets before each fit.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

import igl
from benchmarks.geometry import charts, geometry_report
from benchmarks.geometry.testbeds import manifolds, targets
from benchmarks.geometry.testbeds.package_sets import to_inputs
from igl.core.normalization import normalize_phi
from igl.core.solver import solve_with_intercept
from igl.spd import ConformalityPenalty, OrthogonalityPenalty, conformality_loss, init_encoder_orthogonal_, jacobian
from igl.types import ExtraLoss

__all__ = ["ARMS", "Dataset", "FitSpec", "arm_penalty", "fit_arm", "make_dataset", "score_readout"]


@dataclass(frozen=True, slots=True)
class FitSpec:
    """Sizes and schedule of one learned-chart experiment."""

    ambient: int | None = None
    noise: float = 0.005
    n_train: int = 4096
    n_val: int = 1024
    n_test: int = 2048
    epochs: int = 400
    batch_size: int = 256
    n_anchors: int = 64
    n_scales: int = 4
    source_l2: float = 1e-3
    max_dim: int | None = None
    penalty_weight: float = 1.0
    penalty_every: int = 2
    grad_clip: float = 1.0  # the package default; 0 disables clipping (unstable on some initialisations)
    encoder_lr: float = 1e-3  # the package default
    weight_decay: float | None = None
    noise_std: float = 0.0  # the trainer's input-noise augmentation
    scheduler: str = "none"
    frozen_chart_inputs: bool = True  # chart inputs (param, conformal): identity encoder, kernel and readout trained


@dataclass(slots=True)
class Dataset:
    """Inputs, targets and the bed's geometry for one bed, split and standardised once."""

    bed_name: str
    d: int
    x: dict[str, torch.Tensor]
    y: dict[str, torch.Tensor]
    t: dict[str, torch.Tensor]
    embed: torch.Tensor | None  # [D0, D] linear map of the ambient embedding, or None
    metric_inv_train: torch.Tensor | None  # [n_train, D, D] in the input coordinates
    conformal_chart: dict[str, torch.Tensor] | None
    target_family: str
    extra: dict[str, Any] = field(default_factory=dict)


def _embedding_matrix(d0: int, ambient: int | None, seed: int) -> torch.Tensor | None:
    if ambient is None or ambient <= d0:
        return None
    eye = torch.eye(d0)
    return to_inputs(eye, ambient=ambient, noise=0.0, seed=seed) - to_inputs(
        torch.zeros(1, d0), ambient=ambient, noise=0.0, seed=seed
    )


def make_dataset(bed: manifolds.Bed, spec: FitSpec, *, seed: int = 0, target_family: str = "auto") -> Dataset:
    """Sample the bed, build the targets, embed, standardise and split (train / val / test)."""
    n = spec.n_train + spec.n_val + spec.n_test
    t, x0 = bed.sample(n, seed=seed)
    tg = targets.targets_for(bed, t, x0, family=target_family)
    u = tg["u"].to(torch.float32)
    u = (u - u.mean()) / u.std().clamp_min(1e-12)
    x = to_inputs(x0, ambient=spec.ambient, noise=spec.noise, seed=seed)
    embed = _embedding_matrix(bed.ambient, spec.ambient, seed)
    idx = {
        "train": slice(0, spec.n_train),
        "val": slice(spec.n_train, spec.n_train + spec.n_val),
        "test": slice(spec.n_train + spec.n_val, n),
    }
    m_train = None
    try:
        m = bed.metric_inv_batch(t[idx["train"]])
        if embed is not None:
            e = embed.double()
            m = e.T @ m @ e
        m_train = m.to(torch.float32)
    except (RuntimeError, ValueError):
        m_train = None
    conformal = None
    param_chart = "parameter chart"
    t_param = t.to(torch.float32)
    if bed.conformal_chart is not None:
        conformal = {k: torch.stack([bed.conformal_chart(ti) for ti in t[v]]).to(torch.float32) for k, v in idx.items()}
        if torch.allclose(conformal["train"], t_param[idx["train"]]):
            # the bed is parametrised by its conformal chart (sphere caps): the non-conformal reference chart
            # is then the graph chart, the first d ambient coordinates (orthographic projection on a cap)
            t_param = x0[:, : bed.d].to(torch.float32)
            param_chart = "graph chart (first d ambient coordinates)"
    return Dataset(
        bed_name=bed.name,
        d=bed.d,
        x={k: x[v] for k, v in idx.items()},
        y={k: u[v] for k, v in idx.items()},
        t={k: t_param[v] for k, v in idx.items()},
        embed=embed,
        metric_inv_train=m_train,
        conformal_chart=conformal,
        target_family=str(tg["family"]),
        extra={"param_chart": param_chart},
    )


def arm_penalty(arm: str, spec: FitSpec, dataset: Dataset, trainer_ref: dict[str, igl.MatryoshkaTrainer]) -> ExtraLoss | None:
    """The penalty of an arm; ``trainer_ref['trainer']`` is read lazily for the batch indices of the tangent arms."""
    w, every = spec.penalty_weight, spec.penalty_every
    matched = {"weight": w, "every": every, "all_prefixes": True}
    if arm == "conf-tan":
        if dataset.metric_inv_train is None:
            raise ValueError("conf-tan needs the bed's metric_inv")
        m_all = dataset.metric_inv_train

        def metric_inv(xb: torch.Tensor) -> torch.Tensor:
            idx = trainer_ref["trainer"].current_batch_indices
            if idx is None:
                raise RuntimeError("batch indices unavailable")
            return m_all[idx.to(m_all.device)].to(device=xb.device, dtype=xb.dtype)

        return ConformalityPenalty(beta=1.0, metric_inv=metric_inv, **matched)
    table: dict[str, Callable[[], ExtraLoss | None]] = {
        "none": lambda: None,
        "orth-today": lambda: OrthogonalityPenalty(weight=0.1, every=20),
        "orth-matched": lambda: ConformalityPenalty(beta=0.0, **matched),
        "conf": lambda: ConformalityPenalty(beta=1.0, **matched),
        "conf-plain": lambda: ConformalityPenalty(beta=1.0, axis_normalize=False, **matched),
        "beta-0.5": lambda: ConformalityPenalty(beta=0.5, **matched),
    }
    if arm not in table:
        raise ValueError(f"unknown arm {arm!r}")
    return table[arm]()


def calibrate_weight(
    module: igl.IGLModule,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    k: int,
    beta: float = 1.0,
    target_ratio: float = 0.3,
    batch: int = 256,
) -> dict[str, float]:
    """The penalty weight at which ``‖∇penalty‖ ≈ target_ratio · ‖∇task‖`` over the encoder parameters.

    Measured on one batch of a module trained without penalty (the chart IGL today reaches), so the
    calibrated arms start from a known pull relative to the task, the same rule on every bed.
    """
    params = [p for p in module.encoder.parameters() if p.requires_grad]
    xb, yb = x[:batch], y[:batch]
    module.zero_grad(set_to_none=True)
    task = ((module(xb).squeeze(-1) - yb) ** 2).mean()
    g_task = torch.autograd.grad(task, params, allow_unused=True)
    n_task = torch.sqrt(sum((g**2).sum() for g in g_task if g is not None)).item()
    module.zero_grad(set_to_none=True)
    j = jacobian(module.encoder, xb, output_dim=module.max_dim)[:, :k, :]
    pen = conformality_loss(j @ j.transpose(1, 2), beta=beta)
    g_pen = torch.autograd.grad(pen, params, allow_unused=True)
    n_pen = torch.sqrt(sum((g**2).sum() for g in g_pen if g is not None)).item()
    weight = target_ratio * n_task / max(n_pen, 1e-12)
    return {"weight": float(weight), "grad_task": n_task, "grad_penalty_unit": n_pen, "target_ratio": target_ratio}


ARMS = ("none", "orth-today", "orth-matched", "conf", "conf-plain", "beta-0.5", "conf-tan")


def score_readout(
    module: igl.IGLModule, x_tr: torch.Tensor, y_tr: torch.Tensor, x_te: torch.Tensor, y_te: torch.Tensor, *, k: int, l2: float
) -> float:
    """NRMSE on the test set of the readout solved on the full training set at prefix ``k``."""
    mask = torch.zeros(module.max_dim, device=x_tr.device)
    mask[:k] = 1.0
    with torch.no_grad():
        phi_tr = normalize_phi(module.design_matrix(x_tr, gate_mask=mask), module.normalize)
        phi_te = normalize_phi(module.design_matrix(x_te, gate_mask=mask), module.normalize)
        w, b = solve_with_intercept(phi_tr, y_tr.unsqueeze(1), l2=l2)
        pred = (phi_te @ w.to(phi_te.device) + b.to(phi_te.device)).squeeze(1)
    return float(((pred - y_te) ** 2).mean().sqrt() / y_te.std())


def _prune_curve(
    module: igl.IGLModule, x_tr: torch.Tensor, y_tr: torch.Tensor, x_te: torch.Tensor, y_te: torch.Tensor, *, k: int, l2: float
) -> list[dict[str, float]]:
    mask = torch.zeros(module.max_dim, device=x_tr.device)
    mask[:k] = 1.0
    with torch.no_grad():  # the greedy loop runs on the CPU in float64 (MPS has no float64)
        phi_tr = normalize_phi(module.design_matrix(x_tr, gate_mask=mask), module.normalize).cpu().double()
        phi_te = normalize_phi(module.design_matrix(x_te, gate_mask=mask), module.normalize).cpu().double()
    n_anchor = module.green.n_anchors
    alive = list(range(n_anchor))
    y_tr64, y_te64 = y_tr.cpu().double(), y_te.cpu().double()

    def err(cols: list[int]) -> float:
        a, b_ = phi_tr[:, cols], phi_te[:, cols]
        w, b = solve_with_intercept(a, y_tr64.unsqueeze(1), l2=l2)
        pred = (b_ @ w.to(b_.device, b_.dtype) + b.to(b_.device, b_.dtype)).squeeze(1)
        return float(((pred - y_te64) ** 2).mean().sqrt() / y_te64.std())

    curve = [{"anchors": len(alive), "error": err(alive)}]
    while len(alive) > 4:
        # remove the anchor whose deletion costs least (greedy; the train error decides)
        best = min(alive, key=lambda i: _train_err(phi_tr, y_tr64, [j for j in alive if j != i], l2))
        alive.remove(best)
        if len(alive) in (48, 32, 24, 16, 12, 8, 6, 4):
            curve.append({"anchors": len(alive), "error": err(alive)})
    return curve


def _train_err(phi: torch.Tensor, y: torch.Tensor, cols: list[int], l2: float) -> float:
    a = phi[:, cols]
    w, b = solve_with_intercept(a, y.unsqueeze(1), l2=l2)
    return float(((a @ w.to(a.device, a.dtype) + b.to(a.device, a.dtype)).squeeze(1) - y).norm())


def fit_arm(
    dataset: Dataset,
    spec: FitSpec,
    arm: str,
    *,
    seed: int,
    device: torch.device | str = "cpu",
    inputs: str = "ambient",
    prune: bool = True,
    extra_penalties: Callable[[igl.MatryoshkaTrainer], list[ExtraLoss]] | None = None,
    modules: dict[str, igl.IGLModule] | None = None,
) -> dict[str, Any]:
    """Train one arm on one dataset and seed; return the scored row of the reporting schema.

    ``inputs`` chooses what the encoder sees: ``ambient`` (the embedded, noisy inputs), ``param`` (the bed's
    parameter chart) or ``conformal`` (the true conformal chart): the oracle brackets of block O1.
    """
    dev = torch.device(device)
    src = {"ambient": dataset.x, "param": dataset.t, "conformal": dataset.conformal_chart}[inputs]
    if src is None:
        raise ValueError(f"{dataset.bed_name} has no {inputs} inputs")
    x_tr, x_va, x_te = (src[k].to(dev) for k in ("train", "val", "test"))
    if inputs != "ambient":
        # chart inputs are standardised per column on the train split (a constant affine map, free for the
        # geometry); ambient inputs are left as the package receives them
        mu, sd = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-6)
        x_tr, x_va, x_te = (x_tr - mu) / sd, (x_va - mu) / sd, (x_te - mu) / sd
    y_tr, y_va, y_te = (dataset.y[k].to(dev) for k in ("train", "val", "test"))
    d = dataset.d
    max_dim = spec.max_dim or d
    torch.manual_seed(seed)
    module = igl.IGLModule(x_tr.shape[1], max_dim, 1, n_anchors=spec.n_anchors, n_scales=spec.n_scales)
    frozen = inputs != "ambient" and spec.frozen_chart_inputs and x_tr.shape[1] == max_dim
    if frozen:
        # the oracle chart is given: an encoder trained on top of it only distorts it (pilots on the hills),
        # so the chart is frozen and the kernel and readout alone are trained
        module.encoder = nn.Identity()
    else:
        init_encoder_orthogonal_(module.encoder)
    module = module.to(dev)
    config = igl.MatryoshkaConfig(
        epochs=spec.epochs,
        batch_size=spec.batch_size,
        source_l2=spec.source_l2,
        early_stop_patience=spec.epochs,  # never fires within the budget; the best-validation state is restored
        grad_clip=spec.grad_clip,
        encoder_lr=spec.encoder_lr,
        weight_decay=spec.weight_decay,
        noise_std=spec.noise_std,
        scheduler=spec.scheduler,
        log_every=10**9,
    )
    trainer = igl.MatryoshkaTrainer(loss=igl.MSELoss(), config=config)
    ref = {"trainer": trainer}
    penalty = (
        arm_penalty(arm, spec, dataset, ref)
        if inputs == "ambient" or arm in ("none", "orth-today", "orth-matched", "conf")
        else None
    )
    losses: list[ExtraLoss] = [penalty] if penalty is not None else []
    if extra_penalties is not None:
        losses += extra_penalties(trainer)
    snapshots: list[dict[str, float]] = []
    last: dict[str, Any] = {}

    def on_epoch(stats: igl.EpochStats) -> None:
        last["best_epoch"] = stats.best_epoch
        if stats.epoch in {spec.epochs // 4, spec.epochs // 2} or stats.epoch % max(1, spec.epochs // 8) == 0:
            snapshots.append(
                {"epoch": stats.epoch, "train_loss": stats.train_loss, "val_loss": float(stats.val_loss or float("nan"))}
            )

    torch.manual_seed(seed)
    t0 = time.perf_counter()
    history = trainer.fit(
        module, x_tr, y_tr.unsqueeze(1), x_val=x_va, y_val=y_va.unsqueeze(1), extra_losses=losses, on_epoch=on_epoch
    )
    wall = time.perf_counter() - t0
    module.eval()
    nrmse = {str(k): score_readout(module, x_tr, y_tr, x_te, y_te, k=k, l2=spec.source_l2) for k in sorted({d, max_dim})}
    # the test-set metric_inv is not stored; the report on the train set carries the tangent comparison
    rep_test = geometry_report.report(module.encoder, x_te, k=d, output_dim=max_dim)
    m_train = None
    if inputs == "ambient" and dataset.metric_inv_train is not None:
        m_train = dataset.metric_inv_train[:1024].to(dev)
    rep_train = geometry_report.report(module.encoder, x_tr[:1024], k=d, output_dim=max_dim, metric_inv=m_train)
    with torch.no_grad():
        z_te = module.latent(x_te)[:, :d].cpu().double()
    chart_cost = {"rank": charts.chart_rank(z_te, y_te.cpu())["rank"] if d == 2 else None}  # a 2-D grid measure
    bc = charts.bump_count(z_te[:1024], y_te.cpu()[:1024], max_atoms=128)
    chart_cost["bumps"] = bc["counts"]
    curve = igl.eval_dimension_curve(module, x_va, y_va.unsqueeze(1), loss=igl.MSELoss(), source_l2=spec.source_l2)
    try:
        d_eff = int(igl.detect_elbow(curve))
    except Exception:  # noqa: BLE001 (a flat curve has no elbow; report it as the full width)
        d_eff = max_dim
    row: dict[str, Any] = {
        "bed": dataset.bed_name,
        "d": d,
        "D": int(x_tr.shape[1]),
        "inputs": inputs,
        "arm": arm,
        "seed": seed,
        "kind": "learned",
        "epochs": spec.epochs,
        "grad_clip": spec.grad_clip,
        "wall_s": wall,
        "converged": bool(history.converged),
        "best_epoch": last.get("best_epoch"),
        "param_chart": dataset.extra.get("param_chart"),
        "encoder": "identity (frozen chart)" if frozen else "mlp",
        "train_loss_last": float(history.train_loss[-1]),
        "nrmse_test": nrmse,
        "nrmse_test_dmax": nrmse[str(max_dim)],
        "report_test": rep_test,
        "report_train": rep_train,
        "chart_cost": chart_cost,
        "d_eff_elbow": d_eff,
        "dimension_curve": {str(k): float(v) for k, v in curve.items()},
        "snapshots": snapshots,
        "target_family": dataset.target_family,
        "nrmse_val_history": [float(v) for v in history.val_loss[-3:]] if history.val_loss else [],
    }
    if prune:
        row["prune_curve"] = _prune_curve(module, x_tr, y_tr, x_te, y_te, k=max_dim, l2=spec.source_l2)
    row["penalty_weight"] = spec.penalty_weight if arm not in ("none", "orth-today") else None
    row["penalty_every"] = spec.penalty_every if arm not in ("none", "orth-today") else None
    if modules is not None:
        modules[arm] = module
    return row
