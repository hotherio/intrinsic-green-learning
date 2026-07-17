"""Metric estimation and target whitening.

Whitening a reconstruction target by a metric square root makes plain least
squares optimize the second-order expansion of the geometry the metric
encodes — distillation as a metric change. The pieces compose::

    from igl.whitening import TargetWhitener, WhitenedMSELoss, fisher_pullback

    g = fisher_pullback(w_unembed, states)          # metric of the softmax read-out
    whitener = TargetWhitener(g).fit(states)        # centering + G^{1/2} + unit scale
    loss = WhitenedMSELoss(whitener)                # plugs into MatryoshkaTrainer

The loss whitens in ``target()``, so the trainer's closed-form inner solve,
validation, early stopping, and the dimension curve all operate in the
whitened geometry with no trainer changes.
"""

from igl.whitening.linalg import psd_sqrt_inv
from igl.whitening.loss import WhitenedMSELoss
from igl.whitening.metrics import damped_metric, fisher_pullback, logit_metric, tail_metric
from igl.whitening.whitener import TargetWhitener

__all__ = [
    "TargetWhitener",
    "WhitenedMSELoss",
    "damped_metric",
    "fisher_pullback",
    "logit_metric",
    "psd_sqrt_inv",
    "tail_metric",
]
