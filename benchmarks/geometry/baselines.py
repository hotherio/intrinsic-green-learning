"""The baselines every IGL number of the scale blocks sits next to, on the same split and seeds.

- :func:`linear_probe`: multinomial logistic regression (or ridge) on the standardised inputs.
- :func:`knn`: k-nearest neighbours on the inputs (cosine for unit-norm embeddings).
- :func:`mlp_head`: the **capacity-matched MLP head**: IGL's own encoder (the same ``EncoderConfig`` and Stiefel
  init) followed by ``Linear(k, R) → SiLU → Linear(R, C)``, trained end to end with the task loss on the same
  step budget, batch size, learning rate, clipping and best-validation restore as the IGL fits. It isolates what
  the Green readout adds over an MLP of the same width ``R`` after the same bottleneck ``k``.
- :func:`nystroem_reference`: Nyström RBF features on PCA-``k`` inputs with ``R`` components and a ridge
  readout: a reference line only (a different readout, one isotropic scale), not a control.

The frozen-chart control (the same kernel and readout on PCA-``k`` inputs with an identity encoder) is
``fitting.fit_arm(dataset, spec, "none", inputs="param")`` and lives there.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from benchmarks.geometry.fitting import Dataset, FitSpec, epochs_for
from igl.config import EncoderConfig
from igl.core.encoder import build_mlp_encoder
from igl.spd import init_encoder_orthogonal_

__all__ = ["knn", "linear_probe", "mlp_head", "nystroem_reference"]


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _err_or_nrmse(dataset: Dataset, pred: np.ndarray) -> dict[str, Any]:
    y = _np(dataset.y["test"])
    if dataset.task == "classification":
        wrong = np.nonzero(pred != y)[0]
        return {"err_test": float(len(wrong) / len(y)), "wrong": wrong.tolist()}
    return {"nrmse_test": float(np.sqrt(np.mean((pred - y) ** 2)) / y.std())}


def linear_probe(dataset: Dataset, *, max_iter: int = 300) -> dict[str, Any]:
    """Multinomial logistic regression (classification) or ridge (regression) on the inputs."""
    from sklearn.linear_model import LogisticRegression, RidgeCV

    t0 = time.perf_counter()
    x_tr, x_te = _np(dataset.x["train"]), _np(dataset.x["test"])
    y_tr = _np(dataset.y["train"])
    if dataset.task == "classification":
        model = LogisticRegression(max_iter=max_iter, C=1.0).fit(x_tr, y_tr)
    else:
        model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(x_tr, y_tr)
    pred = model.predict(x_te)
    return {"baseline": "linear", **_err_or_nrmse(dataset, pred), "wall_s": time.perf_counter() - t0}


def knn(dataset: Dataset, *, k: int = 10, metric: str = "euclidean") -> dict[str, Any]:
    """k-nearest neighbours on the inputs."""
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    t0 = time.perf_counter()
    cls = KNeighborsClassifier if dataset.task == "classification" else KNeighborsRegressor
    model = cls(n_neighbors=k, metric=metric, algorithm="brute", n_jobs=-1).fit(
        _np(dataset.x["train"]), _np(dataset.y["train"])
    )
    pred = model.predict(_np(dataset.x["test"]))
    return {"baseline": f"knn{k}", "metric": metric, **_err_or_nrmse(dataset, pred), "wall_s": time.perf_counter() - t0}


class _Head(nn.Module):
    def __init__(self, encoder: nn.Module, k: int, width: int, out: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Linear(k, width), nn.SiLU(), nn.Linear(width, out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def mlp_head(
    dataset: Dataset,
    spec: FitSpec,
    *,
    seed: int,
    device: torch.device | str = "cpu",
    encoder_config: EncoderConfig | None = None,
) -> dict[str, Any]:
    """The capacity-matched MLP head: IGL's encoder → ``Linear(k, R) → SiLU → Linear(R, C)``, same budget and seed."""
    dev = torch.device(device)
    x_tr, x_va, x_te = (dataset.x[s].to(dev) for s in ("train", "val", "test"))
    y_tr, y_va, y_te = (dataset.y[s].to(dev) for s in ("train", "val", "test"))
    classification = dataset.task == "classification"
    k = spec.max_dim or dataset.d
    out = dataset.n_classes if classification else 1
    torch.manual_seed(seed)
    encoder = build_mlp_encoder(x_tr.shape[1], k, config=encoder_config or EncoderConfig())
    init_encoder_orthogonal_(encoder)
    model = _Head(encoder, k, spec.n_anchors, out).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=spec.encoder_lr, weight_decay=spec.weight_decay or 0.0)
    epochs = epochs_for(spec, int(x_tr.shape[0]))
    n = int(x_tr.shape[0])
    best_metric, best_state, best_epoch = -math.inf, copy.deepcopy(model.state_dict()), 0
    history: list[dict[str, float]] = []

    def task_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if classification:
            return nn.functional.cross_entropy(pred, y)
        return nn.functional.mse_loss(pred.squeeze(1), y)

    def val_metric() -> float:
        model.eval()
        with torch.no_grad():
            pv = model(x_va)
            m = float((pv.argmax(1) == y_va).float().mean()) if classification else -float(task_loss(pv, y_va))
        model.train()
        return m

    t0 = time.perf_counter()
    torch.manual_seed(seed)
    model.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=dev)
        total = 0.0
        for start in range(0, n, spec.batch_size):
            idx = perm[start : start + spec.batch_size]
            opt.zero_grad()
            loss = task_loss(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            if spec.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip)
            opt.step()
            total += float(loss.detach()) * idx.shape[0]
        m = val_metric()
        history.append({"epoch": epoch, "train_loss": total / n, "val_metric": m})
        if m > best_metric:
            best_metric, best_state, best_epoch = m, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(x_te)
    pred_np = _np(pred.argmax(1)) if classification else _np(pred.squeeze(1))
    return {
        "baseline": "mlp-head",
        "R": spec.n_anchors,
        "k": k,
        "seed": seed,
        "epochs": epochs,
        "steps": epochs * math.ceil(n / spec.batch_size),
        "best_epoch": best_epoch,
        "n_params": sum(p.numel() for p in model.parameters()),
        **_err_or_nrmse(dataset, pred_np),
        "history": history[:: max(1, epochs // 8)],
        "wall_s": time.perf_counter() - t0,
    }


def nystroem_reference(dataset: Dataset, *, k: int, n_components: int, seed: int, gamma_scale: float = 1.0) -> dict[str, Any]:
    """Nyström RBF features on the PCA-``k`` chart with a ridge (one-hot) readout; bandwidth by the median heuristic."""
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import Ridge, RidgeClassifier

    t0 = time.perf_counter()
    z_tr, z_te = _np(dataset.t["train"][:, :k]), _np(dataset.t["test"][:, :k])
    mu, sd = z_tr.mean(0), z_tr.std(0) + 1e-6
    z_tr, z_te = (z_tr - mu) / sd, (z_te - mu) / sd
    rng = np.random.default_rng(seed)
    sub = z_tr[rng.permutation(len(z_tr))[:2048]]
    d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
    med = float(np.median(d2[np.triu_indices(len(sub), 1)]))
    gamma = gamma_scale / max(med, 1e-12)
    feats = Nystroem(kernel="rbf", gamma=gamma, n_components=min(n_components, len(z_tr)), random_state=seed)
    f_tr = feats.fit_transform(z_tr)
    f_te = feats.transform(z_te)
    y_tr = _np(dataset.y["train"])
    model = (RidgeClassifier(alpha=1e-3) if dataset.task == "classification" else Ridge(alpha=1e-3)).fit(f_tr, y_tr)
    pred = model.predict(f_te)
    return {
        "baseline": "nystroem-rbf",
        "R": n_components,
        "k": k,
        "gamma_scale": gamma_scale,
        "seed": seed,
        **_err_or_nrmse(dataset, pred),
        "wall_s": time.perf_counter() - t0,
    }
