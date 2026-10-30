"""The learned-chart protocol shared by every learned block: one IGL fit per (data set, arm, seed), scored the same way.

Fixed by the plan (v1, revised by the O1 pilots; v2 for the real sets): the default encoder with Stiefel init,
``n_anchors`` x 4 scales, a constant step budget (``epochs`` for the synthetic beds, ``steps_budget`` for the real
sets), batch 256, ``source_l2 = 1e-3``, constant learning rate, the package's gradient clipping, no early stop
within the budget but the best-validation state restored; the same data tensors for every arm; the readout solved
on the full training split and scored on the test split (never ``eval_dimension_curve`` on test); the geometry
report, ``d_eff`` and (where ``d`` is small) the chart-only costs recorded for every run. Chart inputs (the oracle
arms, the frozen PCA chart of the real sets) are standardised and fed to a frozen identity encoder. A paired
design: arms sharing a seed see the same minibatches, because the sampler draws from the global RNG that
``torch.manual_seed`` resets before each fit; data anchors are taken from the epoch-0 latents, identical across
arms under the same seed.

v2 additions: :meth:`Dataset.from_arrays` (a plain point cloud with labels), classification through the package's
``CrossEntropyLoss`` (error rate, the wrong test rows kept for paired tests), ``inner_batch_size = max(4096, 16 R)``
(the training-time readout is solved on that many rows whatever ``R`` is), k-means data anchors, a penalty weight
calibrated against the trainer's own step (inner ridge + task loss) with the realised penalty/task gradient ratio
logged along the run, the Φ column statistics of the EEG campaign, and the geometry report on a PCA tangent proxy
for inputs whose projectors would not fit in memory.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
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

__all__ = [
    "ARMS",
    "Dataset",
    "FitSpec",
    "arm_penalty",
    "calibrate_weight",
    "epochs_for",
    "fit_arm",
    "grad_ratio",
    "loss_for",
    "make_dataset",
    "phi_stats",
    "score",
    "score_readout",
]

Loss = igl.MSELoss | igl.CrossEntropyLoss
_CHART_COST_MAX_D = 8
_PRUNE_MAX_N = 16_384
_REPORT_POINTS = 1024


@dataclass(frozen=True, slots=True)
class FitSpec:
    """Sizes and schedule of one learned-chart experiment."""

    ambient: int | None = None
    noise: float = 0.005
    n_train: int = 4096
    n_val: int = 1024
    n_test: int = 2048
    epochs: int = 400
    steps_budget: int | None = None  # when set, epochs = ceil(steps_budget * batch_size / n_train)
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
    inner_batch_size: int | None = None  # None: max(4096, 16 * n_anchors)
    anchor_init: str = "random"  # "random" | "kmeans-keep" | "kmeans-median" (data anchors from the epoch-0 latents)
    anchor_sample: int = 16_384  # rows of the training split the data anchors are fitted on
    task: str = "regression"  # "regression" | "classification"
    n_classes: int = 0


@dataclass(slots=True)
class Dataset:
    """Inputs, targets and (for a synthetic bed) the geometry, split and standardised once."""

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
    task: str = "regression"
    n_classes: int = 0

    @classmethod
    def from_arrays(
        cls,
        name: str,
        x: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        *,
        d: int,
        task: str = "regression",
        n_classes: int = 0,
        pca_dim: int | None = None,
        tangent_dim: int = 64,
    ) -> Dataset:
        """A plain point cloud with targets (a real data set): no metric, no conformal chart.

        ``t`` holds the PCA-``pca_dim`` (default ``d``) projection fitted on the training split, which is the
        frozen chart of the control arm; ``extra["pca_basis"]`` the top-``tangent_dim`` directions, the tangent
        proxy of the geometry report.
        """
        if task == "classification" and n_classes < 2:
            raise ValueError("classification needs n_classes >= 2")
        k = pca_dim or d
        x_tr = x["train"].float()
        mu = x_tr.mean(0, keepdim=True)
        m = min(max(k, tangent_dim), x_tr.shape[1])
        _, _, v = torch.pca_lowrank(x_tr - mu, q=min(m + 8, x_tr.shape[1]), center=False, niter=4)
        basis = v[:, :m]  # [D, m], orthonormal columns
        t = {s: (w.float() - mu) @ basis[:, :k] for s, w in x.items()}
        return cls(
            bed_name=name,
            d=d,
            x={s: w.float() for s, w in x.items()},
            y=dict(y),
            t=t,
            embed=None,
            metric_inv_train=None,
            conformal_chart=None,
            target_family="labels" if task == "classification" else "target",
            extra={"param_chart": f"PCA-{k} chart", "ambient_metric_identity": True, "pca_basis": basis[:, :tangent_dim]},
            task=task,
            n_classes=n_classes,
        )


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
        if torch.allclose(conformal["train"], t_param[idx["train"]]) and bed.ambient == bed.d + 1:
            # a hypersurface parametrised by its conformal chart (sphere caps): the non-conformal reference
            # chart is then the graph chart, the first d ambient coordinates (orthographic projection on a cap).
            # Not for tori or flat beds: their first d ambient coordinates do not form a chart.
            t_param = x0[:, : bed.d].to(torch.float32)
            param_chart = "graph chart (first d ambient coordinates)"
        elif torch.allclose(conformal["train"], t_param[idx["train"]]):
            param_chart = "parameter chart (= conformal chart)"
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
        extra={"param_chart": param_chart, "ambient_metric_identity": bed.ambient_metric is None},
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
    if "@" in arm:  # "conf@0.1": the arm's penalty at a multiple of the calibrated weight
        base, factor = arm.split("@")
        return arm_penalty(base, replace(spec, penalty_weight=w * float(factor)), dataset, trainer_ref)
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


ARMS = ("none", "orth-today", "orth-matched", "conf", "conf-plain", "beta-0.5", "conf-tan")
_CONF_ARMS = ("conf", "conf-plain", "beta-0.5", "orth-matched", "conf-tan")


def loss_for(spec: FitSpec) -> Loss:
    """The package loss of the spec's task."""
    if spec.task == "classification":
        return igl.CrossEntropyLoss(n_classes=spec.n_classes)
    if spec.task == "regression":
        return igl.MSELoss()
    raise ValueError(f"task must be 'regression' or 'classification', got {spec.task!r}")


def epochs_for(spec: FitSpec, n_train: int) -> int:
    """Epochs of a fit: ``spec.epochs``, or the constant step budget converted at this ``n_train``."""
    if spec.steps_budget is None:
        return spec.epochs
    return max(1, math.ceil(spec.steps_budget * spec.batch_size / n_train))


def _targets(loss: Loss, y: torch.Tensor) -> torch.Tensor:
    """What the trainer expects as targets: ``[N, 1]`` floats for regression, ``[N]`` labels for classification."""
    return y if isinstance(loss, igl.CrossEntropyLoss) else y.unsqueeze(1)


def _solve_targets(loss: Loss, y: torch.Tensor) -> torch.Tensor:
    """What the ridge solve regresses on: one-hot ``[N, C]`` for classification, ``[N, 1]`` for regression."""
    return loss.target(y) if isinstance(loss, igl.CrossEntropyLoss) else y.unsqueeze(1)


def _grad_norm(grads: tuple[torch.Tensor | None, ...]) -> float:
    total = sum((g**2).sum() for g in grads if g is not None)
    return float(torch.sqrt(total).item()) if isinstance(total, torch.Tensor) else 0.0


def _task_gradient_norm(
    module: igl.IGLModule,
    loss: Loss,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch: int,
    inner: int,
    l2: float,
    params: list[torch.Tensor],
) -> float:
    """``‖∇_encoder task‖`` of one trainer step: inner ridge on ``inner`` rows (no grad), task loss on ``batch`` rows."""
    xb, yb = x[:batch], y[:batch]
    module.zero_grad(set_to_none=True)
    with torch.no_grad():
        n_inner = min(inner, x.shape[0])
        phi_in = normalize_phi(module.design_matrix(x[:n_inner]), module.normalize)
        w, b = solve_with_intercept(phi_in, _solve_targets(loss, y[:n_inner]), l2=l2)
    phi = normalize_phi(module.design_matrix(xb), module.normalize)
    out = phi @ w.to(phi.device, phi.dtype) + b.to(phi.device, phi.dtype)
    task = loss.loss(out, _solve_targets(loss, yb))
    return _grad_norm(torch.autograd.grad(task, params, allow_unused=True))


def _penalty_gradient_norm(
    module: igl.IGLModule, x: torch.Tensor, *, k: int, beta: float, batch: int, params: list[torch.Tensor]
) -> float:
    module.zero_grad(set_to_none=True)
    j = jacobian(module.encoder, x[:batch], output_dim=module.max_dim)[:, :k, :]
    pen = conformality_loss(j @ j.transpose(1, 2), beta=beta)
    return _grad_norm(torch.autograd.grad(pen, params, allow_unused=True))


def calibrate_weight(
    module: igl.IGLModule,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    k: int,
    beta: float = 1.0,
    target_ratio: float = 0.3,
    batch: int = 256,
    loss: Loss | None = None,
    inner: int = 4096,
    l2: float = 1e-3,
) -> dict[str, float]:
    """The penalty weight at which ``‖∇penalty‖ ≈ target_ratio · ‖∇task‖`` over the encoder parameters.

    The task gradient is that of the trainer's own step (readout solved on ``inner`` rows, task loss on one
    batch, regression or classification), measured on a module trained without penalty for a fraction of the
    budget, so the calibrated arms start from a known pull relative to the task, the same rule on every set.
    """
    loss = loss if loss is not None else igl.MSELoss()
    params = [p for p in module.encoder.parameters() if p.requires_grad]
    n_task = _task_gradient_norm(module, loss, x, y, batch=batch, inner=inner, l2=l2, params=params)
    n_pen = _penalty_gradient_norm(module, x, k=k, beta=beta, batch=batch, params=params)
    weight = target_ratio * n_task / max(n_pen, 1e-12)
    return {"weight": float(weight), "grad_task": n_task, "grad_penalty_unit": n_pen, "target_ratio": target_ratio}


def grad_ratio(
    module: igl.IGLModule,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    k: int,
    weight: float,
    beta: float = 1.0,
    batch: int = 256,
    loss: Loss | None = None,
    inner: int = 4096,
    l2: float = 1e-3,
) -> dict[str, float]:
    """The realised ``weight · ‖∇penalty‖ / ‖∇task‖`` of the current module (logged along the run)."""
    loss = loss if loss is not None else igl.MSELoss()
    params = [p for p in module.encoder.parameters() if p.requires_grad]
    n_task = _task_gradient_norm(module, loss, x, y, batch=batch, inner=inner, l2=l2, params=params)
    n_pen = _penalty_gradient_norm(module, x, k=k, beta=beta, batch=batch, params=params)
    return {"ratio": weight * n_pen / max(n_task, 1e-12), "grad_task": n_task, "grad_penalty": weight * n_pen}


def score(
    module: igl.IGLModule,
    loss: Loss,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    y_te: torch.Tensor,
    *,
    k: int,
    l2: float,
    lr_readout: bool = False,
) -> dict[str, Any]:
    """Test score of the readout solved on the full training split at prefix ``k``.

    Regression: ``nrmse``. Classification: ``err`` (1 − accuracy), ``nll`` and ``wrong`` (the indices of the
    misclassified test rows, for paired bootstrap and McNemar tests against another arm). ``lr_readout`` adds
    the error of a multinomial logistic regression on the same Φ (the readout-robustness check).
    """
    mask = torch.zeros(module.max_dim, device=x_tr.device)
    mask[:k] = 1.0
    with torch.no_grad():
        phi_tr = normalize_phi(module.design_matrix(x_tr, gate_mask=mask), module.normalize)
        phi_te = normalize_phi(module.design_matrix(x_te, gate_mask=mask), module.normalize)
        # The scoring solve runs on the CPU (QR least squares): the on-device float32 Cholesky fails on the
        # near-singular Gram of a wide Φ (MNIST at R = 512) and would return zero weights.
        w, b = solve_with_intercept(phi_tr.cpu(), _solve_targets(loss, y_tr).cpu(), l2=l2)
        pred = phi_te @ w.to(phi_te.device, phi_te.dtype) + b.to(phi_te.device, phi_te.dtype)
    if isinstance(loss, igl.CrossEntropyLoss):
        labels = pred.argmax(dim=-1)
        wrong = (labels != y_te).nonzero(as_tuple=True)[0].cpu().tolist()
        out: dict[str, Any] = {
            "err": len(wrong) / y_te.shape[0],
            "nll": float(torch.nn.functional.cross_entropy(pred, y_te.long())),
            "wrong": wrong,
        }
        if lr_readout:
            out["err_lr_readout"] = _lr_readout_error(phi_tr, y_tr, phi_te, y_te)
        return out
    y_pred = pred.squeeze(1)
    return {"nrmse": float(((y_pred - y_te) ** 2).mean().sqrt() / y_te.std())}


def _lr_readout_error(phi_tr: torch.Tensor, y_tr: torch.Tensor, phi_te: torch.Tensor, y_te: torch.Tensor) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(phi_tr.cpu().numpy())
    clf = LogisticRegression(max_iter=300).fit(sc.transform(phi_tr.cpu().numpy()), y_tr.cpu().numpy())
    return float((clf.predict(sc.transform(phi_te.cpu().numpy())) != y_te.cpu().numpy()).mean())


def score_readout(
    module: igl.IGLModule, x_tr: torch.Tensor, y_tr: torch.Tensor, x_te: torch.Tensor, y_te: torch.Tensor, *, k: int, l2: float
) -> float:
    """NRMSE on the test set of the regression readout solved on the full training set at prefix ``k`` (v1 name)."""
    return float(score(module, igl.MSELoss(), x_tr, y_tr, x_te, y_te, k=k, l2=l2)["nrmse"])


def phi_stats(module: igl.IGLModule, x: torch.Tensor, *, n: int = 4096) -> dict[str, float]:
    """Column statistics of Φ on ``n`` rows: mean column CV and the effective rank (entropy of the spectrum)."""
    with torch.no_grad():
        phi = normalize_phi(module.design_matrix(x[:n]), module.normalize).cpu().double()  # MPS has no float64
    sd, mean = phi.std(0), phi.mean(0).abs().clamp_min(1e-12)
    centred = phi - phi.mean(0, keepdim=True)
    s = torch.linalg.svdvals(centred)
    p = (s / s.sum().clamp_min(1e-12)).clamp_min(1e-30)
    return {
        "column_cv": float((sd / mean).mean()),
        "dead_columns": int((sd < 1e-6).sum()),
        "eff_rank": float(torch.exp(-(p * p.log()).sum())),
    }


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


def _place_anchors(module: igl.IGLModule, spec: FitSpec, x_tr: torch.Tensor, *, seed: int) -> dict[str, Any]:
    """Data anchors from the epoch-0 latents (identical across arms sharing a seed); random anchors untouched."""
    if spec.anchor_init == "random":
        return {"anchor_init": "random"}
    if spec.anchor_init not in ("kmeans-keep", "kmeans-median"):
        raise ValueError(f"anchor_init must be random, kmeans-keep or kmeans-median, got {spec.anchor_init!r}")
    n = min(spec.anchor_sample, x_tr.shape[0])
    with torch.no_grad():
        z0 = module.latent(x_tr[:n])
    gen = torch.Generator().manual_seed(seed)
    sigma = "median" if spec.anchor_init == "kmeans-median" else "keep"
    module.green.reanchor_(z0, method="kmeans", sigma=sigma, generator=gen)
    return {"anchor_init": spec.anchor_init, "anchor_rows": n}


def fit_arm(  # noqa: PLR0912, PLR0915 (one protocol, read top to bottom)
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
    lr_readout: bool = False,
    ratio_epochs: tuple[float, ...] = (0.1, 0.5, 1.0),
) -> dict[str, Any]:
    """Train one arm on one dataset and seed; return the scored row of the reporting schema.

    ``inputs`` chooses what the encoder sees: ``ambient`` (the inputs as given), ``param`` (the bed's parameter
    chart, or the PCA chart of a real set) or ``conformal`` (the true conformal chart). The pruning curve runs only
    when ``prune`` and the training split is small; the chart-only costs only when ``d`` is small.
    """
    dev = torch.device(device)
    loss = loss_for(replace(spec, task=dataset.task, n_classes=dataset.n_classes))
    classification = isinstance(loss, igl.CrossEntropyLoss)
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
    n_train = int(x_tr.shape[0])
    epochs = epochs_for(spec, n_train)
    inner = spec.inner_batch_size or max(4096, 16 * spec.n_anchors)
    output_dim = dataset.n_classes if classification else 1
    torch.manual_seed(seed)
    module = igl.IGLModule(x_tr.shape[1], max_dim, output_dim, n_anchors=spec.n_anchors, n_scales=spec.n_scales)
    frozen = inputs != "ambient" and spec.frozen_chart_inputs and x_tr.shape[1] == max_dim
    if frozen:
        # the chart is given: an encoder trained on top of it only distorts it (pilots on the hills),
        # so the chart is frozen and the kernel and readout alone are trained
        module.encoder = nn.Identity()
    else:
        init_encoder_orthogonal_(module.encoder)
    module = module.to(dev)
    anchors = _place_anchors(module, spec, x_tr, seed=seed)
    basis = dataset.extra.get("pca_basis")
    tangent = basis.to(dev) if isinstance(basis, torch.Tensor) and not frozen else None
    rep0 = None if frozen else geometry_report.report(module.encoder, x_te[:_REPORT_POINTS], k=d, output_dim=max_dim)
    config = igl.MatryoshkaConfig(
        epochs=epochs,
        batch_size=spec.batch_size,
        inner_batch_size=inner,
        source_l2=spec.source_l2,
        early_stop_patience=epochs,  # never fires within the budget; the best-validation state is restored
        grad_clip=spec.grad_clip,
        encoder_lr=spec.encoder_lr,
        weight_decay=spec.weight_decay,
        noise_std=spec.noise_std,
        scheduler=spec.scheduler,
        log_every=10**9,
    )
    trainer = igl.MatryoshkaTrainer(loss=loss, config=config)
    ref = {"trainer": trainer}
    penalty = (
        arm_penalty(arm, spec, dataset, ref)
        if inputs == "ambient" or arm in ("none", "orth-today", "orth-matched", "conf")
        else None
    )
    losses: list[ExtraLoss] = [penalty] if penalty is not None else []
    if extra_penalties is not None:
        losses += extra_penalties(trainer)
    base_arm = arm.split("@")[0]
    factor = float(arm.split("@")[1]) if "@" in arm else 1.0
    weight = spec.penalty_weight * factor
    track_ratio = base_arm in _CONF_ARMS and not frozen
    beta = {"orth-matched": 0.0, "beta-0.5": 0.5}.get(base_arm, 1.0)
    ratio_at = {max(1, round(f * epochs)) for f in ratio_epochs}
    snapshots: list[dict[str, float]] = []
    ratios: list[dict[str, float]] = []
    last: dict[str, Any] = {}

    def on_epoch(stats: igl.EpochStats) -> None:
        last["best_epoch"] = stats.best_epoch
        if stats.epoch in {epochs // 4, epochs // 2, epochs} or stats.epoch % max(1, epochs // 8) == 0:
            snapshots.append(
                {
                    "epoch": stats.epoch,
                    "train_loss": stats.train_loss,
                    "val_loss": float(stats.val_loss if stats.val_loss is not None else float("nan")),
                    "val_metric": float(stats.val_metric if stats.val_metric is not None else float("nan")),
                }
            )
        if track_ratio and stats.epoch in ratio_at:
            was_training = module.training
            r = grad_ratio(module, x_tr, y_tr, k=d, weight=weight, beta=beta, loss=loss, inner=inner, l2=spec.source_l2)
            module.train(was_training)
            ratios.append({"epoch": stats.epoch, **r})

    torch.manual_seed(seed)
    t0 = time.perf_counter()
    history = trainer.fit(
        module,
        x_tr,
        _targets(loss, y_tr),
        x_val=x_va,
        y_val=_targets(loss, y_va),
        extra_losses=losses,
        on_epoch=on_epoch,
    )
    wall = time.perf_counter() - t0
    module.eval()
    scores = {
        str(k): score(module, loss, x_tr, y_tr, x_te, y_te, k=k, l2=spec.source_l2, lr_readout=lr_readout and k == max_dim)
        for k in sorted({d, max_dim})
    }
    rep_test = geometry_report.report(module.encoder, x_te[:_REPORT_POINTS], k=d, output_dim=max_dim, tangent_basis=tangent)
    m_train = None
    if inputs == "ambient" and dataset.metric_inv_train is not None:
        m_train = dataset.metric_inv_train[:_REPORT_POINTS].to(dev)
    rep_train = geometry_report.report(
        module.encoder,
        x_tr[:_REPORT_POINTS],
        k=d,
        output_dim=max_dim,
        metric_inv=m_train,
        metric_inv_is_projector=bool(dataset.extra.get("ambient_metric_identity", True)),
        tangent_basis=None if m_train is not None else tangent,
    )
    chart_cost: dict[str, Any] = {}
    if d <= _CHART_COST_MAX_D and not classification:
        with torch.no_grad():
            z_te = module.latent(x_te)[:, :d].cpu().double()
        # chart-only costs on a robust [0, 1] normalisation of the latent (a learned chart can have outliers)
        chart_cost["rank"] = charts.chart_rank(z_te, y_te.cpu(), robust=True)["rank"] if d == 2 else None
        chart_cost["bumps"] = charts.bump_count(z_te[:1024], y_te.cpu()[:1024], max_atoms=128, robust=True)["counts"]
    curve = igl.eval_dimension_curve(module, x_va, _targets(loss, y_va), loss=loss, source_l2=spec.source_l2)
    try:
        d_eff = int(igl.detect_elbow(curve))
    except Exception:  # noqa: BLE001 (a flat curve has no elbow; report it as the full width)
        d_eff = max_dim
    stability = None
    if rep0 is not None:
        worse = rep_test["full"]["sqrt_conf"] > rep0["full"]["sqrt_conf"]
        stability = {
            "sqrt_conf_epoch0": rep0["full"]["sqrt_conf"],
            "sqrt_conf_final": rep_test["full"]["sqrt_conf"],
            "cond_median_final": rep_test["full"]["cond_median"],
            "flag": bool(worse or rep_test["full"]["cond_median"] > 1e4),
        }
    row: dict[str, Any] = {
        "bed": dataset.bed_name,
        "task": dataset.task,
        "d": d,
        "D": int(x_tr.shape[1]),
        "N": n_train,
        "n_val": int(x_va.shape[0]),
        "n_test": int(x_te.shape[0]),
        "R": spec.n_anchors,
        "inputs": inputs,
        "arm": arm,
        "seed": seed,
        "kind": "learned",
        "epochs": epochs,
        "steps": epochs * math.ceil(n_train / spec.batch_size),
        "inner_batch_size": inner,
        "grad_clip": spec.grad_clip,
        "wall_s": wall,
        "converged": bool(history.converged),
        "best_epoch": last.get("best_epoch"),
        "param_chart": dataset.extra.get("param_chart"),
        "encoder": "identity (frozen chart)" if frozen else "mlp",
        "train_loss_last": float(history.train_loss[-1]),
        "score_test": scores,
        "report_test": rep_test,
        "report_train": rep_train,
        "report_epoch0": rep0,
        "stability": stability,
        "chart_cost": chart_cost,
        "d_eff_elbow": d_eff,
        "dimension_curve": {str(k): float(v) for k, v in curve.items()},
        "snapshots": snapshots,
        "grad_ratios": ratios,
        "phi": phi_stats(module, x_tr),
        "target_family": dataset.target_family,
        "val_loss_last3": [float(v) for v in history.val_loss[-3:]] if history.val_loss else [],
        "val_metric_best": max(history.val_metric) if (classification and history.val_metric) else None,
        **anchors,
    }
    top = scores[str(max_dim)]
    if classification:
        row["err_test"] = top["err"]
        row["err_test_lr_readout"] = top.get("err_lr_readout")
    else:
        row["nrmse_test"] = {k: v["nrmse"] for k, v in scores.items()}
        row["nrmse_test_dmax"] = top["nrmse"]
    if prune and not classification and n_train <= _PRUNE_MAX_N:
        row["prune_curve"] = _prune_curve(module, x_tr, y_tr, x_te, y_te, k=max_dim, l2=spec.source_l2)
    row["penalty_weight"] = weight if arm not in ("none", "orth-today") else None
    row["penalty_every"] = spec.penalty_every if arm not in ("none", "orth-today") else None
    if modules is not None:
        modules[arm] = module
    return row
