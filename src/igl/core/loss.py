"""Built-in :class:`igl.types.LossStrategy` implementations.

A loss strategy decouples the trainer from the task. Each strategy knows:

- how to turn raw labels into the lstsq target (e.g. one-hot for classification),
- how to compute a differentiable loss for the gradient step,
- how to compute a scalar metric for early-stopping decisions.

Add new strategies (AIRM for SPD reconstruction, log-Euclidean for general
SPD, etc.) by implementing :class:`igl.types.LossStrategy` and passing your
instance to :class:`igl.core.trainer.MatryoshkaTrainer`.
"""

import torch
import torch.nn.functional as F  # noqa: N812

from igl.exceptions import IGLConfigError

_MIN_CLASSES = 2


class CrossEntropyLoss:
    """Multiclass cross-entropy with one-hot lstsq targets.

    Args:
        n_classes: Number of classes ``C``.

    Attributes:
        higher_is_better: Always ``True`` — the metric is accuracy.
    """

    higher_is_better: bool = True
    n_classes: int

    def __init__(self, *, n_classes: int) -> None:
        if n_classes < _MIN_CLASSES:
            raise IGLConfigError(f"n_classes must be >= {_MIN_CLASSES}, got {n_classes}")
        self.n_classes = n_classes

    def target(self, y: torch.Tensor) -> torch.Tensor:
        """Convert integer class labels ``[N]`` to one-hot float targets ``[N, C]``."""
        return F.one_hot(y.long(), num_classes=self.n_classes).float()

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss; ``target`` is one-hot, taken back to class indices."""
        return F.cross_entropy(pred, target.argmax(dim=-1))

    def metric(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Top-1 classification accuracy."""
        return float((pred.argmax(dim=-1) == target.argmax(dim=-1)).float().mean().item())

    def curve_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """0/1 classification *error rate* — used for dimension-curve elbow detection.

        Cross-entropy saturates on easy tasks and smears the elbow; error
        rate is discrete and gives sharper transitions.
        """
        return float((pred.argmax(dim=-1) != target.argmax(dim=-1)).float().mean().item())


class MSELoss:
    """Mean squared error for scalar or multi-output regression.

    Attributes:
        higher_is_better: Always ``False`` — the metric is MSE.
    """

    higher_is_better: bool = False

    def target(self, y: torch.Tensor) -> torch.Tensor:
        """Ensure the target is at least 2-D ``[N, C]``."""
        target = y.float()
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        return target

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)

    def metric(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        return float(F.mse_loss(pred, target).item())

    def curve_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """MSE — already lower-is-better and non-saturating; same as ``metric``."""
        return self.metric(pred, target)


__all__ = ["CrossEntropyLoss", "MSELoss"]
