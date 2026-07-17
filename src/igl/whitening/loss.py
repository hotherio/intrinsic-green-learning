"""Whitened-target loss strategy."""

import torch
import torch.nn.functional as F  # noqa: N812

from igl.exceptions import IGLNotFittedError
from igl.whitening.whitener import TargetWhitener

__all__ = ["WhitenedMSELoss"]


class WhitenedMSELoss:
    """MSE in a whitened target space: second-order distillation as least squares.

    Wraps a *fitted* :class:`TargetWhitener`; whitening happens in
    :meth:`target`, so it flows through every trainer touchpoint — the
    closed-form inner solve, the per-batch loss, validation, early stopping,
    and :func:`igl.eval_dimension_curve` — with no trainer changes. When the
    whitener's metric is the Fisher pullback of a softmax read-out, the
    optimized quantity is the second-order expansion of the downstream KL.

    Args:
        whitener: A fitted :class:`TargetWhitener`.

    Attributes:
        higher_is_better: Always ``False`` — the metric is whitened MSE.

    Raises:
        IGLNotFittedError: If ``whitener`` is not fitted.
    """

    higher_is_better: bool = False
    whitener: TargetWhitener

    def __init__(self, whitener: TargetWhitener) -> None:
        if not whitener.is_fitted:
            raise IGLNotFittedError("WhitenedMSELoss requires a fitted TargetWhitener")
        self.whitener = whitener

    def target(self, y: torch.Tensor) -> torch.Tensor:
        """Whiten raw targets ``[N, C]`` (1-D targets are lifted to ``[N, 1]``)."""
        target = y.float()
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        return self.whitener.transform(target)

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)

    def metric(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        return float(F.mse_loss(pred, target).item())

    def curve_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Whitened MSE — already non-saturating, so identical to :meth:`metric`."""
        return self.metric(pred, target)
