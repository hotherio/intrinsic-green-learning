"""Matryoshka Variable-Projection trainer.

Per training step:

1. Sample a truncation level ``k ~ Sampler(d_max)``.
2. Encode and truncate inputs to the first ``k`` latent dimensions.
3. Solve the readout weights closed-form via :func:`igl.direct_solve_weights`
   on an *inner* mini-batch (no gradient).
4. Compute the task loss from the gradient-tracking design matrix Φ_k and the
   detached ``w_k``; backprop through encoder + Green kernel + bias.

The trainer is agnostic to the task — pass any :class:`igl.types.LossStrategy`.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from igl.config import MatryoshkaConfig
from igl.core.solver import direct_solve_weights
from igl.exceptions import IGLConvergenceError
from igl.matryoshka.sampler import PowerLawSampler, UniformSampler
from igl.nn.module import IGLModule
from igl.types import LossStrategy, MatryoshkaSampler


@dataclass(slots=True)
class TrainingHistory:
    """Per-epoch training history returned by :meth:`MatryoshkaTrainer.fit`."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_metric: list[float] = field(default_factory=list)
    truncation_k: list[float] = field(default_factory=list)
    best_epoch: int | None = None
    best_metric: float | None = None
    stopped_epoch: int | None = None
    early_stopped: bool = False


def _build_sampler(config: MatryoshkaConfig) -> MatryoshkaSampler:
    if config.sampling == "uniform":
        return UniformSampler()
    return PowerLawSampler(alpha=config.alpha)


class MatryoshkaTrainer:
    """Train an :class:`IGLModule` with Matryoshka VP.

    Args:
        config: :class:`MatryoshkaConfig`. Optional — defaults are used when
            omitted.
        loss: :class:`LossStrategy` describing the task (classification,
            regression, …).
        sampler: Optional explicit :class:`MatryoshkaSampler`. When ``None``,
            one is built from ``config.sampling`` and ``config.alpha``.
    """

    config: MatryoshkaConfig
    loss: LossStrategy
    sampler: MatryoshkaSampler

    def __init__(
        self,
        *,
        loss: LossStrategy,
        config: MatryoshkaConfig | None = None,
        sampler: MatryoshkaSampler | None = None,
    ) -> None:
        self.config = config or MatryoshkaConfig()
        self.loss = loss
        self.sampler = sampler if sampler is not None else _build_sampler(self.config)

    def fit(
        self,
        module: IGLModule,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        *,
        x_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
    ) -> TrainingHistory:
        """Fit ``module`` on the training tensors and return the history.

        Args:
            module: An :class:`IGLModule`.
            x_train: Training inputs ``[N, D]``.
            y_train: Training targets.
            x_val: Optional validation inputs.
            y_val: Optional validation targets.

        Returns:
            A :class:`TrainingHistory` instance.

        Raises:
            IGLConvergenceError: If the training loss becomes non-finite.
        """
        config = self.config
        device = next(module.parameters()).device
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        if x_val is not None:
            x_val = x_val.to(device)
            assert y_val is not None, "y_val must be provided when x_val is provided"
            y_val = y_val.to(device)

        d_max = module.max_dim
        n_samples = x_train.shape[0]

        params: list[nn.Parameter] = list(module.encoder.parameters()) + list(module.green.parameters()) + [module.bias]
        optimizer = (
            AdamW(params, lr=config.encoder_lr, weight_decay=config.weight_decay)
            if config.weight_decay is not None
            else AdamW(params, lr=config.encoder_lr)
        )
        scheduler = (
            CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=1) if config.scheduler == "cosine_warm_restarts" else None
        )

        use_early_stop = config.early_stop_patience is not None and x_val is not None
        best_metric: float = -float("inf") if self.loss.higher_is_better else float("inf")
        best_epoch: int = -1
        epochs_since_improvement: int = 0
        best_state: dict[str, object] | None = None
        history = TrainingHistory()

        for epoch in range(config.epochs):
            epoch_loss = self._train_one_epoch(
                module=module,
                optimizer=optimizer,
                params=params,
                x_train=x_train,
                y_train=y_train,
                d_max=d_max,
                n_samples=n_samples,
                history=history,
                epoch=epoch,
            )
            if not torch.isfinite(torch.tensor(epoch_loss)):
                raise IGLConvergenceError(epoch=epoch + 1, last_loss=epoch_loss)
            history.train_loss.append(epoch_loss)

            if scheduler is not None:
                scheduler.step()

            val_loss, val_metric = self._validate_and_refresh(
                module=module,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
            )
            if x_val is not None:
                history.val_loss.append(val_loss)
                history.val_metric.append(val_metric)

            if use_early_stop:
                improved = val_metric > best_metric if self.loss.higher_is_better else val_metric < best_metric
                if improved:
                    best_metric = val_metric
                    best_epoch = epoch
                    epochs_since_improvement = 0
                    best_state = self._snapshot(module)
                else:
                    epochs_since_improvement += 1
                if (
                    epoch + 1 >= config.early_stop_min_epochs
                    and config.early_stop_patience is not None
                    and epochs_since_improvement >= config.early_stop_patience
                ):
                    history.stopped_epoch = epoch + 1
                    history.early_stopped = True
                    break

        if use_early_stop and best_state is not None:
            self._restore(module, best_state)
            history.best_epoch = best_epoch
            history.best_metric = best_metric
            # Refresh source weights from the restored encoder.
            self._validate_and_refresh(
                module=module,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
            )

        return history

    def _train_one_epoch(
        self,
        *,
        module: IGLModule,
        optimizer: AdamW,
        params: Sequence[nn.Parameter],
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        d_max: int,
        n_samples: int,
        history: TrainingHistory,
        epoch: int,  # noqa: ARG002
    ) -> float:
        config = self.config
        module.train()
        device = x_train.device
        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0.0
        k_sum = 0
        n_batches = 0

        for i in range(0, n_samples, config.batch_size):
            idx = perm[i : i + config.batch_size]
            x_batch = x_train[idx]
            y_batch = y_train[idx]
            if config.noise_std > 0.0:
                x_batch = x_batch + config.noise_std * torch.randn_like(x_batch)

            k = self.sampler(d_max)
            k_sum += k
            n_batches += 1

            optimizer.zero_grad()
            mask = torch.zeros(d_max, device=device)
            mask[:k] = 1.0

            z = module.encoder(x_batch)
            z_trunc = z * mask.unsqueeze(0)
            phi = module.green(z_trunc, gate_mask=mask)
            from igl.core.normalization import normalize_phi  # noqa: PLC0415

            phi = normalize_phi(phi, module.normalize)

            with torch.no_grad():
                lstsq_n = min(config.inner_batch_size, n_samples)
                lstsq_idx = torch.randperm(n_samples, device=device)[:lstsq_n]
                z_lstsq = module.encoder(x_train[lstsq_idx]) * mask.unsqueeze(0)
                phi_lstsq = module.green(z_lstsq, gate_mask=mask)
                phi_lstsq = normalize_phi(phi_lstsq, module.normalize)
                target_lstsq = self.loss.target(y_train[lstsq_idx]) - module.bias.detach()
                w_k = direct_solve_weights(phi_lstsq, target_lstsq, l2=config.source_l2).to(device)

            output = phi @ w_k + module.bias
            target_batch = self.loss.target(y_batch)
            task_loss = self.loss.loss(output, target_batch)
            # torch's autograd entry points have partial stubs in this version.
            task_loss.backward()  # pyright: ignore[reportUnknownMemberType]

            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]

            epoch_loss += float(task_loss.item()) * int(idx.shape[0])

        history.truncation_k.append(k_sum / max(n_batches, 1))
        return epoch_loss / max(n_samples, 1)

    def _validate_and_refresh(
        self,
        *,
        module: IGLModule,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor | None,
        y_val: torch.Tensor | None,
    ) -> tuple[float, float]:
        config = self.config
        device = next(module.parameters()).device
        module.eval()

        with torch.no_grad():
            inner_n = min(config.inner_batch_size, x_train.shape[0])
            inner_idx = torch.randperm(x_train.shape[0], device=device)[:inner_n]
            z_full = module.encoder(x_train[inner_idx])
            phi_full = module.green(z_full)
            from igl.core.normalization import normalize_phi  # noqa: PLC0415

            phi_full = normalize_phi(phi_full, module.normalize)
            target_full = self.loss.target(y_train[inner_idx]) - module.bias.detach()
            w_full = direct_solve_weights(phi_full, target_full, l2=config.source_l2).to(device)
            module.set_source_weights(w_full)

        if x_val is None or y_val is None:
            return 0.0, 0.0

        with torch.no_grad():
            output = module(x_val)
            target_val = self.loss.target(y_val)
            val_loss = float(self.loss.loss(output, target_val).item())
            val_metric = self.loss.metric(output, target_val)
        return val_loss, val_metric

    @staticmethod
    def _snapshot(module: IGLModule) -> dict[str, object]:
        return {
            "encoder": {k: v.detach().cpu().clone() for k, v in module.encoder.state_dict().items()},
            "green": {k: v.detach().cpu().clone() for k, v in module.green.state_dict().items()},
            "bias": module.bias.detach().cpu().clone(),
        }

    @staticmethod
    def _restore(module: IGLModule, snapshot: dict[str, object]) -> None:
        encoder_state = cast(dict[str, torch.Tensor], snapshot["encoder"])
        green_state = cast(dict[str, torch.Tensor], snapshot["green"])
        bias = cast(torch.Tensor, snapshot["bias"])
        module.encoder.load_state_dict(encoder_state)
        module.green.load_state_dict(green_state)
        module.bias.data.copy_(bias.to(module.bias.device))


__all__ = ["MatryoshkaTrainer", "TrainingHistory"]
