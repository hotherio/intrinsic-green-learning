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

import torch

from igl.exceptions import IGLConfigError
from igl.spd.linalg import matrix_exp_sym, matrix_log_sym, matrix_pow_sym, unpack_sym_vec


def airm_loss(
    c: torch.Tensor,
    c_hat: torch.Tensor,
    *,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """Affine-Invariant Riemannian Metric² between batched SPD matrices.

    Args:
        c: ``[B, d, d]`` ground-truth SPD matrices.
        c_hat: ``[B, d, d]`` predicted SPD matrices.
        eps: Eigenvalue clamp forwarded to matrix log / power routines.
        reduction: One of ``"mean"`` (default), ``"sum"``, ``"none"``.
            ``"none"`` returns the per-sample AIRM² as a ``[B]`` tensor.

    Returns:
        Scalar (mean/sum) or per-sample AIRM² values.

    Raises:
        IGLConfigError: For an unknown ``reduction`` value.
    """
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

    Attributes:
        higher_is_better: Always ``False`` — AIRM² is a distance.
    """

    higher_is_better: bool = False
    latent_dim: int
    eps: float

    def __init__(self, *, latent_dim: int, eps: float = 1e-8) -> None:
        if latent_dim < 1:
            raise IGLConfigError(f"latent_dim must be >= 1, got {latent_dim}")
        self.latent_dim = latent_dim
        self.eps = eps

    def target(self, y: torch.Tensor) -> torch.Tensor:
        """Pass-through: the log-Eig vector is already the lstsq target."""
        return y.float() if y.dim() > 1 else y.float().unsqueeze(-1)

    def _to_spd(self, vec: torch.Tensor) -> torch.Tensor:
        """Lift a log-Eig vector back to an SPD matrix."""
        sym = unpack_sym_vec(vec, self.latent_dim)
        return matrix_exp_sym(sym)

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """AIRM² between the predicted SPD and the target SPD."""
        c = self._to_spd(target)
        c_hat = self._to_spd(pred)
        return airm_loss(c, c_hat, eps=self.eps, reduction="mean")

    def metric(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        return float(self.loss(pred, target).item())

    def curve_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Dimension-curve score = AIRM² (already lower-is-better and not saturating)."""
        return self.metric(pred, target)


__all__ = ["AIRMLoss", "airm_loss"]
