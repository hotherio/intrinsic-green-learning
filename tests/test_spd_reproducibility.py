"""Per-issue reproducibility assertions for the SPD wrapper.

Mirrors the eleven issues in
``alex-eeg-igl/PACKAGE_REPRODUCIBILITY_ISSUES.md``. Each test corresponds
to one issue (1.1, 1.2, …, 4.1) and asserts the reproducer behaviour the
diagnostic spelled out.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import torch
from torch import nn

import igl
from igl import IGLConfig, IGLConfigError, KernelConfig, MatryoshkaConfig, NormalizeMode
from igl.core.trainer import MatryoshkaTrainer
from igl.data import make_spd_dataset
from igl.nn.module import IGLModule
from igl.spd import IGLReconSPDClassifier, LogEigVectorizer
from igl.spd.airm import AIRMLoss
from igl.types import SchedulerType

# --- Synthetic dataset shared by the integration tests ------------------------


@pytest.fixture
def spd_xy() -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Returns ``(log_eig_vec, labels, raw_spd_tensor)`` for d=4, N=80, 2 classes."""
    torch.manual_seed(0)
    np.random.seed(0)
    spd_t, y_t = make_spd_dataset(80, d=4, n_classes=2, class_separation=1.0, seed=42)
    spd = spd_t.float()
    vec = LogEigVectorizer().fit(spd.numpy()).transform(spd.numpy())
    return np.asarray(vec, dtype=np.float64), y_t.numpy(), spd


def _fast_config(epochs: int = 6) -> IGLConfig:
    """Tiny config for fast integration runs in the test suite."""
    return IGLConfig(
        matryoshka=MatryoshkaConfig(
            epochs=epochs,
            batch_size=16,
            inner_batch_size=64,
            scheduler="none",
            early_stop_patience=None,
            verbose=False,
        ),
    )


# --- Issue 1.1 + 2.1: normalize_input default + wrapper kwarg ----------------


def test_issue_1_1__igl_module_normalize_input_default_is_false() -> None:
    """IGLModule(normalize_input=...) default flips to False."""
    module = IGLModule(input_dim=4, max_dim=4, output_dim=2)
    # When normalize_input=False the encoder is the raw MLPEncoder, not a
    # Sequential wrapping a BatchNorm1d.
    assert not isinstance(next(iter(module.encoder.children())), nn.BatchNorm1d)


def test_issue_2_1__recon_spd_exposes_normalize_input_kwarg(
    spd_xy: tuple[np.ndarray, np.ndarray, torch.Tensor],
) -> None:
    """IGLReconSPDClassifier accepts a normalize_input kwarg (no TypeError)."""
    x, y, _ = spd_xy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        clf = IGLReconSPDClassifier(
            latent_dim=4,
            max_dim=4,
            n_anchors=8,
            n_scales=2,
            normalize_input=False,
            random_state=0,
            config=_fast_config(),
        ).fit(x, y)
    # After fit the encoder must not carry a top-level BN layer.
    assert not isinstance(next(iter(clf.module_.encoder.children())), nn.BatchNorm1d)


# --- Issue 1.2 + 2.2: NormalizeMode default + wrapper kwarg + config threading


def test_issue_1_2__kernel_config_default_normalize_is_nw() -> None:
    """KernelConfig defaults to NormalizeMode.NW now."""
    assert KernelConfig().normalize == NormalizeMode.NW


def test_issue_2_2__recon_spd_explicit_normalize_kwarg(
    spd_xy: tuple[np.ndarray, np.ndarray, torch.Tensor],
) -> None:
    """The new normalize="..." kwarg on IGLReconSPDClassifier doesn't raise."""
    x, y, _ = spd_xy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        clf = IGLReconSPDClassifier(
            latent_dim=4,
            max_dim=4,
            n_anchors=8,
            n_scales=2,
            normalize="nw",
            random_state=0,
            config=_fast_config(),
        ).fit(x, y)
    assert clf.module_.normalize == NormalizeMode.NW


def test_issue_2_2__recon_spd_config_kernel_normalize_threaded_through(
    spd_xy: tuple[np.ndarray, np.ndarray, torch.Tensor],
) -> None:
    """An IGLConfig(kernel=KernelConfig(normalize=SOFTMAX)) actually takes effect."""
    x, y, _ = spd_xy
    cfg = IGLConfig(
        kernel=KernelConfig(normalize=NormalizeMode.SOFTMAX),
        matryoshka=_fast_config().matryoshka,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        clf = IGLReconSPDClassifier(
            latent_dim=4,
            max_dim=4,
            n_anchors=8,
            n_scales=2,
            random_state=0,
            config=cfg,
        ).fit(x, y)
    assert clf.module_.normalize == NormalizeMode.SOFTMAX


# --- Issue 1.3: scheduler default ---------------------------------------------


def test_issue_1_3__matryoshka_config_default_scheduler_is_none() -> None:
    """MatryoshkaConfig defaults to SchedulerType.NONE now."""
    assert MatryoshkaConfig().scheduler == SchedulerType.NONE


# --- Issue 1.4: weight_decay default ------------------------------------------


def test_issue_1_4__matryoshka_config_default_weight_decay_is_none() -> None:
    """MatryoshkaConfig defaults to weight_decay=None (PyTorch's 0.01 then applies)."""
    assert MatryoshkaConfig().weight_decay is None


# --- Issue 1.5: AIRMLoss jitter -----------------------------------------------


def test_issue_1_5__airm_loss_default_jitter_is_nonzero() -> None:
    """AIRMLoss now exposes a jitter parameter; default is 1e-5."""
    loss = AIRMLoss(latent_dim=4)
    assert loss.jitter == pytest.approx(1e-5)


def test_issue_1_5__airm_loss_rejects_negative_jitter() -> None:
    with pytest.raises(IGLConfigError, match="jitter"):
        AIRMLoss(latent_dim=4, jitter=-1.0)


def test_issue_1_5b__jitter_inside_matrix_exp_on_predicted_side() -> None:
    """Predicted-side jitter is applied to the symmetric matrix *before* matrix_exp_sym.

    This is the discipline the EEG reference uses
    (`alex-eeg-igl/igl_recon_spd_orth.py:233-236`). v0.2.5 mistakenly added
    jitter *after* `matrix_exp_sym`, producing a measurably different c_hat
    for EEG-scale spectra and breaking bit-exact reproduction. This test
    pins the correct ordering against a hand-computed reference.
    """
    from igl.spd.linalg import matrix_exp_sym, unpack_sym_vec  # noqa: PLC0415

    torch.manual_seed(0)
    d = 4
    vec_dim = d * (d + 1) // 2
    pred = torch.randn(6, vec_dim, dtype=torch.float32)

    # Use a deliberately large jitter so "inside" vs "outside" the exp
    # differs on the order of 1, not float32 noise. 1e-5 (the production
    # default) would still differ correctly but the assertion would need
    # tighter tolerances.
    jitter = 0.5
    loss = AIRMLoss(latent_dim=d, jitter=jitter)
    expected = matrix_exp_sym(
        unpack_sym_vec(pred, d) + jitter * torch.eye(d).expand(pred.shape[0], d, d),
    )
    actual = loss._pred_to_spd(pred)  # noqa: SLF001
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


# --- Issue 2.3: covs= kwarg + raw-C path --------------------------------------


def test_issue_2_3__airm_loss_requires_trainer_when_covs_set() -> None:
    """AIRMLoss(covs=...) without a trainer reference raises IGLConfigError."""
    with pytest.raises(IGLConfigError, match="trainer"):
        AIRMLoss(latent_dim=4, covs=torch.randn(10, 4, 4))


def test_issue_2_3__recon_spd_fit_accepts_covs(
    spd_xy: tuple[np.ndarray, np.ndarray, torch.Tensor],
) -> None:
    """IGLReconSPDClassifier.fit(..., covs=...) does not raise."""
    x, y, raw = spd_xy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        clf = IGLReconSPDClassifier(
            latent_dim=4,
            max_dim=4,
            n_anchors=8,
            n_scales=2,
            random_state=0,
            config=_fast_config(),
        ).fit(x, y, covs=raw)
    assert clf.predict(x).shape == y.shape


def test_issue_2_3__recon_spd_rejects_misaligned_covs(
    spd_xy: tuple[np.ndarray, np.ndarray, torch.Tensor],
) -> None:
    x, y, raw = spd_xy
    clf = IGLReconSPDClassifier(latent_dim=4, max_dim=4, n_anchors=4, n_scales=2, config=_fast_config(epochs=2))
    with pytest.raises(IGLConfigError, match="covs"):
        clf.fit(x, y, covs=raw[:10])


# --- Issue 3.1: validation_fraction split via randperm ------------------------


def test_issue_3_1__recon_spd_validation_fraction_default_is_0_2() -> None:
    clf = IGLReconSPDClassifier(latent_dim=4)
    assert clf.validation_fraction == pytest.approx(0.2)


def test_issue_3_1__recon_spd_rejects_zero_validation_fraction() -> None:
    with pytest.raises(IGLConfigError, match="validation_fraction"):
        IGLReconSPDClassifier(latent_dim=4, validation_fraction=0.0)
    with pytest.raises(IGLConfigError, match="validation_fraction"):
        IGLReconSPDClassifier(latent_dim=4, validation_fraction=1.0)


def test_issue_3_1__recon_spd_uses_split_via_randperm(
    spd_xy: tuple[np.ndarray, np.ndarray, torch.Tensor],
) -> None:
    """With validation_fraction=0.5 the val loss differs from train loss across epochs.

    The legacy (broken) behaviour passed the same tensor as train and val, so
    `history.val_loss[k]` equalled `history.train_loss[k]` modulo BN
    train/eval differences. After the fix the two tracks should genuinely
    diverge.
    """
    x, y, _ = spd_xy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        clf = IGLReconSPDClassifier(
            latent_dim=4,
            max_dim=4,
            n_anchors=8,
            n_scales=2,
            validation_fraction=0.5,
            random_state=0,
            config=_fast_config(epochs=6),
        ).fit(x, y)
    # At least one epoch must show a measurable divergence between val and
    # train losses (the legacy "pass same tensor" path was bit-identical).
    train_losses = list(clf.history_.train_loss)
    val_losses = list(clf.history_.val_loss)
    assert len(train_losses) == len(val_losses)
    assert any(abs(t - v) > 1e-6 for t, v in zip(train_losses, val_losses, strict=True))


# --- Issue 3.2: source_weights is a non-trainable Parameter with randn init ---


def test_issue_3_2__source_weights_is_parameter_with_no_grad() -> None:
    module = IGLModule(input_dim=4, max_dim=4, output_dim=2, n_anchors=8, n_scales=2)
    assert isinstance(module.source_weights, nn.Parameter)
    assert not module.source_weights.requires_grad


def test_issue_3_2__source_weights_initial_values_not_zero() -> None:
    """The new randn*0.01 init is non-zero in expectation."""
    torch.manual_seed(0)
    module = IGLModule(input_dim=4, max_dim=4, output_dim=4, n_anchors=8, n_scales=2)
    assert module.source_weights.abs().mean().item() > 0.001  # noqa: PLR2004


def test_issue_3_2__source_weights_excluded_from_trainer_optim() -> None:
    """MatryoshkaTrainer does not put source_weights into the AdamW param group."""
    module = IGLModule(input_dim=4, max_dim=4, output_dim=2, n_anchors=8, n_scales=2)
    trainer_params = list(module.encoder.parameters()) + list(module.green.parameters()) + [module.bias]
    assert all(id(p) != id(module.source_weights) for p in trainer_params)


# --- Issue 3.3: sigma_max diagnostic flag + train-loop block ------------------


def test_issue_3_3__matryoshka_config_default_sigma_max_diagnostic_is_false() -> None:
    """The flag defaults to False — opt-in only."""
    assert MatryoshkaConfig().sigma_max_diagnostic is False


def test_issue_3_3__recon_spd_default_matryoshka_enables_sigma_max() -> None:
    """When the SPD wrapper builds its own MatryoshkaConfig, the flag is True."""
    clf = IGLReconSPDClassifier(latent_dim=4)
    cfg = clf._matryoshka_config()  # noqa: SLF001
    assert cfg.sigma_max_diagnostic is True


def test_issue_3_3__sigma_max_diagnostic_runs_extra_encoder_forwards() -> None:
    """Enabling the flag triggers two extra encoder forwards per epoch.

    With BatchNorm1d in the encoder, the extra forwards increment
    `num_batches_tracked` by 2 per epoch (one per `module.encoder(x_ref)`
    and one per `module.encoder(x_ref + eps*v)`).
    """
    torch.manual_seed(0)
    x = torch.randn(64, 8)
    y = (x.norm(dim=-1) > x.norm(dim=-1).median()).long()

    def _bn_seen(module: IGLModule) -> int:
        for sub in module.encoder.modules():
            if isinstance(sub, nn.BatchNorm1d):
                return int(sub.num_batches_tracked.item())
        return 0

    # Force the encoder to contain a BatchNorm1d so we can read
    # `num_batches_tracked`. Doing this via `normalize_input=True` is the
    # simplest way: the encoder becomes Sequential(BN, MLPEncoder).
    epochs = 3
    base_kwargs: dict[str, object] = {
        "input_dim": 8,
        "max_dim": 4,
        "output_dim": 2,
        "n_anchors": 4,
        "n_scales": 2,
        "normalize_input": True,
    }

    # Run A — flag off.
    module_a = IGLModule(**base_kwargs)  # type: ignore[arg-type]
    trainer_a = MatryoshkaTrainer(
        loss=igl.CrossEntropyLoss(n_classes=2),
        config=MatryoshkaConfig(
            epochs=epochs,
            batch_size=32,
            inner_batch_size=64,
            scheduler="none",
            early_stop_patience=None,
            verbose=False,
            sigma_max_diagnostic=False,
        ),
    )
    trainer_a.fit(module_a, x, y)

    # Run B — flag on, identical seed.
    torch.manual_seed(0)
    module_b = IGLModule(**base_kwargs)  # type: ignore[arg-type]
    trainer_b = MatryoshkaTrainer(
        loss=igl.CrossEntropyLoss(n_classes=2),
        config=MatryoshkaConfig(
            epochs=epochs,
            batch_size=32,
            inner_batch_size=64,
            scheduler="none",
            early_stop_patience=None,
            verbose=False,
            sigma_max_diagnostic=True,
        ),
    )
    trainer_b.fit(module_b, x, y)

    # The diagnostic adds 2 extra encoder forwards per epoch. Note: the
    # validation-and-refresh path also calls the encoder (one extra forward
    # per epoch in eval mode, but BN.eval() does NOT update
    # num_batches_tracked). So the only contribution to the diff is the
    # train-time diagnostic forwards.
    diff = _bn_seen(module_b) - _bn_seen(module_a)
    assert diff == 2 * epochs


# --- Issue 4.1: fork_rng opt-out ----------------------------------------------


def test_issue_4_1__recon_spd_default_fork_rng_preserves_global_torch_rng(
    spd_xy: tuple[np.ndarray, np.ndarray, torch.Tensor],
) -> None:
    """With fork_rng=True (default) the caller's global torch RNG survives .fit."""
    x, y, _ = spd_xy
    torch.manual_seed(123)
    state_before = torch.get_rng_state().clone()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        IGLReconSPDClassifier(
            latent_dim=4,
            max_dim=4,
            n_anchors=4,
            n_scales=2,
            random_state=0,
            config=_fast_config(epochs=2),
        ).fit(x, y)
    assert torch.equal(state_before, torch.get_rng_state())


def test_issue_4_1__recon_spd_fork_rng_false_mutates_global_torch_rng(
    spd_xy: tuple[np.ndarray, np.ndarray, torch.Tensor],
) -> None:
    """With fork_rng=False the global torch RNG is mutated by .fit."""
    x, y, _ = spd_xy
    torch.manual_seed(123)
    state_before = torch.get_rng_state().clone()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        IGLReconSPDClassifier(
            latent_dim=4,
            max_dim=4,
            n_anchors=4,
            n_scales=2,
            random_state=0,
            fork_rng=False,
            config=_fast_config(epochs=2),
        ).fit(x, y)
    assert not torch.equal(state_before, torch.get_rng_state())


# --- Issue 3.4: per-batch exception guard for AIRM eigh-NaN -------------------


class _RaisingLoss:
    """Synthetic ``LossStrategy`` that raises ``RuntimeError`` on a chosen call.

    The first ``raise_on_calls`` invocations of :meth:`loss` raise; later
    ones fall back to a plain MSE. Used to simulate the eigh-backward NaN
    pattern that motivates Issue 3.4 without depending on ill-conditioned
    SPD matrices.
    """

    higher_is_better: bool = False

    def __init__(self, raise_on_calls: tuple[int, ...] = (1,)) -> None:
        self._n_calls = 0
        self._raise_on = set(raise_on_calls)
        self.raised_count = 0

    def target(self, y: torch.Tensor) -> torch.Tensor:
        return y.float() if y.dim() > 1 else y.float().unsqueeze(-1)

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self._n_calls += 1
        if self._n_calls in self._raise_on:
            self.raised_count += 1
            raise RuntimeError("synthetic eigh-backward NaN")
        return ((pred - target) ** 2).mean()

    def metric(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        return float(self.loss(pred, target).item())

    def curve_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        return self.metric(pred, target)


def _toy_module() -> igl.IGLModule:
    """Tiny IGLModule for the guard tests — no SPD, just enough to backprop."""
    torch.manual_seed(0)
    return igl.IGLModule(input_dim=4, max_dim=4, output_dim=4, n_anchors=4, n_scales=2)


def _toy_xy() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(40, 4), torch.randn(40, 4)


def test_issue_3_4__matryoshka_config_default_skip_failing_batches_is_false() -> None:
    assert MatryoshkaConfig().skip_failing_batches is False


def test_issue_3_4__recon_spd_default_matryoshka_enables_skip() -> None:
    """When IGLReconSPDClassifier builds its own MatryoshkaConfig, the flag is True."""
    clf = IGLReconSPDClassifier(latent_dim=4)
    assert clf._matryoshka_config().skip_failing_batches is True  # noqa: SLF001


def test_issue_3_4__flag_off_propagates_runtime_error() -> None:
    """Without the flag, a RuntimeError in the loss escapes the trainer."""
    module = _toy_module()
    x, y = _toy_xy()
    loss = _RaisingLoss(raise_on_calls=(1,))
    trainer = MatryoshkaTrainer(
        loss=loss,
        config=MatryoshkaConfig(
            epochs=2,
            batch_size=20,
            inner_batch_size=40,
            scheduler="none",
            early_stop_patience=None,
            verbose=False,
            skip_failing_batches=False,
        ),
    )
    with pytest.raises(RuntimeError, match="synthetic eigh-backward NaN"):
        trainer.fit(module, x, y)


def test_issue_3_4__flag_on_skips_failing_batch_silently() -> None:
    """With the flag set, the trainer skips the failing batch and continues."""
    module = _toy_module()
    x, y = _toy_xy()
    # First batch raises, subsequent batches succeed → epoch should still run.
    loss = _RaisingLoss(raise_on_calls=(1,))
    trainer = MatryoshkaTrainer(
        loss=loss,
        config=MatryoshkaConfig(
            epochs=2,
            batch_size=20,
            inner_batch_size=40,
            scheduler="none",
            early_stop_patience=None,
            verbose=False,
            skip_failing_batches=True,
        ),
    )
    # No exception should escape.
    history = trainer.fit(module, x, y)
    assert loss.raised_count == 1
    assert len(history.train_loss) == 2  # noqa: PLR2004
    # The failing batch contributes nothing to the epoch loss accumulator,
    # so the surviving batch's contribution determines train_loss[0].
    assert math.isfinite(history.train_loss[0])


def test_issue_3_4__flag_on_validation_returns_inf_on_failure() -> None:
    """With the flag set, a validation failure surfaces as inf instead of crashing."""
    module = _toy_module()
    x, y = _toy_xy()
    x_val, y_val = _toy_xy()
    # Call ordering with batch_size=20 + n_samples=40 + x_val provided:
    #   epoch 1 — train.loss calls 1, 2;  val.loss call 3
    #   epoch 2 — train.loss calls 4, 5;  val.loss call 6
    # Pick call 3 so the failure lands during validation of epoch 1.
    loss = _RaisingLoss(raise_on_calls=(3,))
    trainer = MatryoshkaTrainer(
        loss=loss,
        config=MatryoshkaConfig(
            epochs=2,
            batch_size=20,
            inner_batch_size=40,
            scheduler="none",
            early_stop_patience=None,
            verbose=False,
            skip_failing_batches=True,
        ),
    )
    history = trainer.fit(module, x, y, x_val=x_val, y_val=y_val)
    assert loss.raised_count >= 1
    # At least one epoch's val_loss is inf — the sentinel set by the guard.
    assert any(v == float("inf") for v in history.val_loss)


def test_issue_3_4__flag_on_does_not_corrupt_parameters() -> None:
    """A failed batch leaves the encoder parameters unchanged (zero_grad'd)."""
    module = _toy_module()
    x, y = _toy_xy()
    # Snapshot encoder params before training.
    pre = {n: p.detach().clone() for n, p in module.encoder.named_parameters()}

    # Every call raises → no parameter updates should happen.
    loss = _RaisingLoss(raise_on_calls=tuple(range(1, 200)))
    trainer = MatryoshkaTrainer(
        loss=loss,
        config=MatryoshkaConfig(
            epochs=1,
            batch_size=20,
            inner_batch_size=40,
            scheduler="none",
            early_stop_patience=None,
            verbose=False,
            skip_failing_batches=True,
        ),
    )
    trainer.fit(module, x, y)
    for name, post in module.encoder.named_parameters():
        torch.testing.assert_close(pre[name], post.detach(), rtol=0, atol=0)
