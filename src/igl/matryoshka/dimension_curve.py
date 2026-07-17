"""Post-training dimension-curve evaluation and elbow detection.

After Matryoshka training, every truncation level ``k ∈ {1, …, d_max}`` is a
valid model. :func:`eval_dimension_curve` sweeps ``k`` and reports the
validation loss at each. :func:`detect_elbow` identifies the smallest ``k``
beyond which further dimensions stop providing substantial improvement —
that's the discovered effective dimension ``d_eff``.
"""

import math

import torch

from igl.core.normalization import normalize_phi
from igl.core.solver import direct_solve_weights
from igl.exceptions import IGLConfigError
from igl.nn.module import IGLModule
from igl.types import DimensionCurve, LossStrategy

_MIN_CURVE_POINTS = 2


@torch.no_grad()
def eval_dimension_curve(
    module: IGLModule,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    loss: LossStrategy,
    source_l2: float = 1e-3,
) -> DimensionCurve:
    """Evaluate the trained module at every truncation level ``k``.

    For each ``k``, freshly solves the readout weights via lstsq using only the
    first ``k`` latent dimensions, then computes the validation metric. Returns
    a ``{k: metric}`` mapping. The mapping iterates ``k = 1, 2, …, d_max`` in
    insertion order.

    Args:
        module: Trained :class:`IGLModule`.
        x_val: Validation inputs ``[N, D]``.
        y_val: Validation targets.
        loss: Loss strategy used to compute the per-``k`` metric.
        source_l2: Tikhonov regularisation forwarded to
            :func:`igl.direct_solve_weights`.

    Returns:
        A dict mapping ``k → curve_score`` where ``curve_score`` is whatever
        :meth:`LossStrategy.curve_score` reports — error rate for
        :class:`igl.CrossEntropyLoss`, MSE for :class:`igl.MSELoss`, etc.
        The curve score is always lower-is-better so :func:`detect_elbow`
        can locate the knee.
    """
    module.eval()
    device = next(module.parameters()).device
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    target = loss.target(y_val)

    d_max = module.max_dim
    z_full = module.encoder(x_val)

    results: dict[int, float] = {}
    for k in range(1, d_max + 1):
        mask = torch.zeros(d_max, device=device)
        mask[:k] = 1.0
        z_trunc = z_full * mask.unsqueeze(0)
        phi = module.green(z_trunc, gate_mask=mask)
        phi = normalize_phi(phi, module.normalize)
        # Bias column so each k gets its own intercept.
        ones_col = torch.ones(phi.shape[0], 1, device=device, dtype=phi.dtype)
        phi_aug = torch.cat([phi, ones_col], dim=-1)
        weights = direct_solve_weights(phi_aug, target, l2=source_l2, on_nonfinite="raise").to(device)
        pred = phi_aug @ weights
        results[k] = loss.curve_score(pred, target)

    return results


def detect_elbow(curve: DimensionCurve, *, ratio: float = 2.0) -> int:
    """Locate the elbow of a dimension/loss curve in log-space.

    Operates on ``log(loss)`` so a 5× loss reduction has the same log-delta
    whether it occurs at ``loss=0.1`` or ``loss=0.001``. Returns the largest
    ``k`` whose log-reduction exceeds ``max_log_delta / ratio``.

    Args:
        curve: ``{k: loss}`` mapping (from :func:`eval_dimension_curve`).
            Must contain at least one entry.
        ratio: A reduction must be at least ``max_log_delta / ratio`` to count
            as substantial. Default ``2.0``.

    Returns:
        The estimated intrinsic dimension ``d_eff``.

    Raises:
        IGLConfigError: If ``curve`` is empty or ``ratio <= 0``.
    """
    if not curve:
        raise IGLConfigError("curve must contain at least one entry")
    if ratio <= 0:
        raise IGLConfigError(f"ratio must be > 0, got {ratio}")

    ks = sorted(curve)
    if len(ks) < _MIN_CURVE_POINTS:
        return ks[0]

    losses = [curve[k] for k in ks]

    # Floor very small / non-positive values to avoid log(0); use the smallest
    # *positive* loss × 1e-3 as the floor.
    pos_losses = [v for v in losses if v > 0]
    if not pos_losses:
        return ks[0]
    eps = min(pos_losses) * 1e-3
    log_losses = [math.log(max(v, eps)) for v in losses]
    log_deltas = [a - b for a, b in zip(log_losses, log_losses[1:], strict=False)]

    max_log_delta = max(log_deltas)
    if max_log_delta <= 0:
        return ks[0]

    cutoff = max_log_delta / ratio
    last_substantial = ks[0]
    for i, delta in enumerate(log_deltas):
        if delta > cutoff:
            last_substantial = ks[i + 1]
    return last_substantial


__all__ = [
    "detect_elbow",
    "eval_dimension_curve",
]
