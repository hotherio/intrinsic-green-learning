"""Affine-Invariant Riemannian Metric (AIRM) loss as an :class:`igl.LossStrategy`.

The AIRM is the canonical SPD-manifold distance:

    AIRM(C, Ĉ)² = ‖log(C^{-1/2} Ĉ C^{-1/2})‖_F²

It is affine-invariant — ``AIRM(A C A^T, A Ĉ A^T) = AIRM(C, Ĉ)`` for any
invertible ``A`` — and respects the Riemannian geometry of the SPD cone.

This module exposes both the raw :func:`airm_loss` function and a
:class:`AIRMLoss` wrapper that implements :class:`igl.LossStrategy` so it can
be passed directly to :class:`igl.MatryoshkaTrainer`. The trainer's design
matrix Φ is interpreted as a log-Eig vector that ``unpack_sym_vec`` lifts
back to a symmetric matrix; ``matrix_exp_sym`` then sends it back onto the
SPD manifold for the AIRM comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from igl.exceptions import IGLConfigError
from igl.spd.linalg import matrix_exp_sym, matrix_log_sym, matrix_pow_sym, unpack_sym_vec

if TYPE_CHECKING:
    from igl.core.trainer import MatryoshkaTrainer


def airm_loss(
    c: torch.Tensor,
    c_hat: torch.Tensor,
    *,
    eps: float = 1e-8,
    reduction: str = "mean",
    c_inv_half: torch.Tensor | None = None,
) -> torch.Tensor:
    """Affine-Invariant Riemannian Metric² between batched SPD matrices.

    Args:
        c: ``[B, d, d]`` ground-truth SPD matrices.
        c_hat: ``[B, d, d]`` predicted SPD matrices.
        eps: Eigenvalue clamp forwarded to matrix log / power routines.
        reduction: One of ``"mean"`` (default), ``"sum"``, ``"none"``.
            ``"none"`` returns the per-sample AIRM² as a ``[B]`` tensor.
        c_inv_half: Optional precomputed ``C^{-1/2}`` aligned with ``c``. When
            ``c`` is a constant (the data covariance), ``C^{-1/2}`` is constant
            too, so a caller may hoist this eigendecomposition out of the
            training loop and pass it here — bit-identical to computing it
            inline (per-matrix ``eigh`` is independent of batching), but skips
            one of the three per-batch ``eigh`` calls.

    Returns:
        Scalar (mean/sum) or per-sample AIRM² values.

    Raises:
        IGLConfigError: For an unknown ``reduction`` value.
    """
    if c_inv_half is None:
        c_inv_half = matrix_pow_sym(c, -0.5, eps=eps)
    a = c_inv_half @ c_hat @ c_inv_half
    log_a = matrix_log_sym(a, eps=eps)
    sq_per_sample = (log_a**2).sum(dim=(-1, -2))
    if reduction == "mean":
        return sq_per_sample.mean()
    if reduction == "sum":
        return sq_per_sample.sum()
    if reduction == "none":
        return sq_per_sample
    raise IGLConfigError(f"unknown reduction: {reduction!r}; expected mean/sum/none")


class AIRMLoss:
    """:class:`igl.LossStrategy` for SPD reconstruction via the AIRM metric.

    The trainer feeds the strategy:

    - ``y``: log-Eig vectors of shape ``[B, D]`` where ``D = d (d + 1) / 2``.
    - ``pred``: lstsq-fit predictions of the same shape — interpreted as
      log-Eig vectors of predicted SPD matrices.

    :meth:`target` returns ``y`` unchanged (the lstsq operates in
    log-Euclidean tangent space, where Euclidean lstsq is geometry-respecting).
    :meth:`loss` lifts both vectors back to SPD via
    ``unpack_sym_vec → matrix_exp_sym`` and computes AIRM². :meth:`metric` is
    the same AIRM² value.

    Args:
        latent_dim: Side length ``d`` of the underlying SPD matrices.
        eps: Eigenvalue clamp for matrix log/exp/power.
        jitter: Tikhonov-style ridge added on both sides of the AIRM call
            (``jitter * I`` to the predicted SPD after ``matrix_exp_sym`` and
            to the target SPD). Real EEG covariances often span 6+ orders of
            magnitude on their eigenvalue spectrum; without a small ridge
            the ``eigh`` backward pass produces NaN gradients on
            ill-conditioned batches. Default ``1e-5`` matches the reference
            trainer's discipline. Set ``0.0`` to disable.
        covs: Optional ``[N, d, d]`` raw SPD targets aligned with the
            trainer's training tensor (``x_train``). When set together with a
            ``trainer`` reference, :meth:`loss` slices ``covs`` by the
            trainer's :attr:`current_batch_indices` and uses the raw
            covariance as the AIRM target instead of round-tripping the
            log-Eig vector through ``unpack_sym_vec → matrix_exp_sym``. This
            avoids ~1e-6 round-trip noise per element that compounds to
            measurable AUC drift over 1000 epochs.
        trainer: The :class:`MatryoshkaTrainer` instance the loss is attached
            to. Required when ``covs`` is set. Stored as a back-reference so
            the loss can read ``trainer.current_batch_indices`` per batch.

    Attributes:
        higher_is_better: Always ``False`` — AIRM² is a distance.
    """

    higher_is_better: bool = False
    latent_dim: int
    eps: float
    jitter: float
    covs: torch.Tensor | None
    trainer: MatryoshkaTrainer | None

    def __init__(
        self,
        *,
        latent_dim: int,
        eps: float = 1e-8,
        jitter: float = 1e-5,
        covs: torch.Tensor | None = None,
        trainer: MatryoshkaTrainer | None = None,
    ) -> None:
        if latent_dim < 1:
            raise IGLConfigError(f"latent_dim must be >= 1, got {latent_dim}")
        if jitter < 0:
            raise IGLConfigError(f"jitter must be >= 0, got {jitter}")
        if covs is not None and trainer is None:
            raise IGLConfigError(
                "AIRMLoss(covs=...) requires a trainer reference so per-batch indices can be read",
            )
        self.latent_dim = latent_dim
        self.eps = eps
        self.jitter = jitter
        self.covs = covs
        self.trainer = trainer
        # Lazily-filled cache of C^{-1/2} for the (constant) data covariances,
        # so the training path skips one of three per-batch eigh calls. Reset
        # whenever covs is relocated to a new device.
        self._cov_inv_half: torch.Tensor | None = None

    def target(self, y: torch.Tensor) -> torch.Tensor:
        """Pass-through: the log-Eig vector is already the lstsq target."""
        return y.float() if y.dim() > 1 else y.float().unsqueeze(-1)

    def _to_spd(self, vec: torch.Tensor) -> torch.Tensor:
        """Lift a log-Eig vector back to an SPD matrix (no jitter — round-trip path)."""
        sym = unpack_sym_vec(vec, self.latent_dim)
        return matrix_exp_sym(sym)

    def _pred_to_spd(self, vec: torch.Tensor) -> torch.Tensor:
        """Lift the predicted log-Eig vector back to an SPD matrix, jitter inside the exp.

        Matches the EEG reference's discipline at
        ``alex-eeg-igl/igl_recon_spd_orth.py:233-236``:
        ``c_hat = matrix_exp_sym(unpack(pred) + jitter * I)``. Adding jitter
        *after* ``matrix_exp_sym`` produces a measurably different
        ``c_hat`` for EEG-scale eigenvalue spectra and breaks bit-exact
        reproduction (see Issue 1.5b discovered post-v0.2.5).
        """
        sym = unpack_sym_vec(vec, self.latent_dim)
        if self.jitter > 0:
            sym = sym + self._jitter_eye(sym.device, sym.dtype)
        return matrix_exp_sym(sym)

    def _jitter_eye(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.jitter * torch.eye(self.latent_dim, device=device, dtype=dtype)

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """AIRM² between the predicted SPD and the target SPD.

        When :attr:`covs` and :attr:`trainer` are set AND the trainer has
        published a ``current_batch_indices`` tensor (i.e. we are inside the
        trainer's training batch loop), the AIRM target is
        ``covs[indices] + jitter * I`` (avoiding the log-Eig → matrix-exp
        round-trip). Otherwise — validation pass, dimension-curve eval, or
        any external caller — the target is reconstructed from ``target``
        via ``unpack_sym_vec``. The training-time path is the bit-exact one;
        the val/eval path is the original round-trip (its small numerical
        noise only affects monitoring, not gradients).

        On the predicted side, jitter is added to the symmetric matrix
        *before* ``matrix_exp_sym`` (see :meth:`_pred_to_spd`).
        """
        idx = self.trainer.current_batch_indices if self.trainer is not None else None
        if self.covs is not None and idx is not None:
            # Cache the covs on the prediction device ONCE. The previous
            # ``self.covs.to(pred.device)`` re-uploaded the full ``[N, d, d]``
            # tensor host→device every minibatch under CUDA; hoisting the move
            # makes it a one-time transfer. No-op on CPU (same device), so the
            # CPU path is bit-identical.
            if self.covs.device != pred.device:
                self.covs = self.covs.to(pred.device)
                self._cov_inv_half = None
            raw = self.covs.index_select(0, idx)
            c = raw + self._jitter_eye(raw.device, raw.dtype) if self.jitter > 0 else raw
            # C = covs + jitter*I is constant across epochs, so C^{-1/2} is too.
            # Precompute it once for all N samples and index per batch — skips
            # one of the three per-batch eigh. Bit-identical to inline
            # matrix_pow_sym(c, -0.5): per-matrix eigh is independent of batching.
            if self._cov_inv_half is None:
                c_full = (
                    self.covs + self._jitter_eye(self.covs.device, self.covs.dtype)
                    if self.jitter > 0
                    else self.covs
                )
                self._cov_inv_half = matrix_pow_sym(c_full, -0.5, eps=self.eps)
            c_inv_half = self._cov_inv_half.index_select(0, idx)
            c_hat = self._pred_to_spd(pred)
            return airm_loss(c, c_hat, eps=self.eps, reduction="mean", c_inv_half=c_inv_half)
        c = self._to_spd(target)
        if self.jitter > 0:
            c = c + self._jitter_eye(c.device, c.dtype)
        c_hat = self._pred_to_spd(pred)
        return airm_loss(c, c_hat, eps=self.eps, reduction="mean")

    def metric(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        return float(self.loss(pred, target).item())

    def curve_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Dimension-curve score = AIRM² (already lower-is-better and not saturating)."""
        return self.metric(pred, target)


__all__ = ["AIRMLoss", "airm_loss"]
