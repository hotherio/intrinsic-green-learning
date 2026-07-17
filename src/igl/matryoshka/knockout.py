"""Greedy knockout: a stricter dimension certificate than the elbow.

The dimension curve keeps latent *prefixes*; the knockout instead deletes
whichever coordinate hurts the score least, one at a time, refitting the
readout in closed form after each deletion. The number of coordinates that
survive before the score degrades certifies how many the task actually
uses — including non-prefix subsets the Matryoshka ordering would miss.
"""

from dataclasses import dataclass

import torch

from igl.core.normalization import normalize_phi
from igl.core.solver import direct_solve_weights
from igl.exceptions import IGLConfigError
from igl.nn.module import IGLModule
from igl.types import LossStrategy

__all__ = ["KnockoutResult", "detect_knockout_knee", "greedy_knockout"]

_MIN_KNEE_POINTS = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class KnockoutResult:
    """Outcome of :func:`greedy_knockout`.

    Attributes:
        curve: ``{n_active: curve_score}`` for each greedy step, from
            ``max_dim`` active coordinates down to 1.
        removal_order: Coordinate indices in the order they were removed.
        knee: The certified dimension, per :func:`detect_knockout_knee`.
    """

    curve: dict[int, float]
    removal_order: list[int]
    knee: int


@torch.no_grad()
def greedy_knockout(
    module: IGLModule,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    loss: LossStrategy,
    source_l2: float = 1e-3,
    ratio: float = 2.0,
) -> KnockoutResult:
    """Delete coordinates greedily, least-harmful first, refitting each time.

    Scores use :meth:`LossStrategy.curve_score` (always lower-is-better:
    error rate for classification, MSE for regression), so the certificate
    is task-scored rather than reconstruction-scored.

    Args:
        module: Trained :class:`IGLModule`.
        x_val: Validation inputs ``[N, D]``.
        y_val: Validation targets.
        loss: Loss strategy providing targets and the curve score.
        source_l2: Tikhonov regularisation for the per-step readout refit.
        ratio: Knee threshold forwarded to :func:`detect_knockout_knee`.

    Returns:
        A :class:`KnockoutResult` with the greedy curve, the removal order,
        and the certified knee.
    """
    module.eval()
    device = next(module.parameters()).device
    x_val = x_val.to(device)
    target = loss.target(y_val.to(device))
    d_max = module.max_dim
    z_full = module.encoder(x_val)

    def score_with(mask: torch.Tensor) -> float:
        phi = module.green(z_full * mask.unsqueeze(0), gate_mask=mask)
        phi = normalize_phi(phi, module.normalize)
        ones_col = torch.ones(phi.shape[0], 1, device=device, dtype=phi.dtype)
        phi_aug = torch.cat([phi, ones_col], dim=-1)
        weights = direct_solve_weights(phi_aug, target, l2=source_l2, on_nonfinite="raise").to(device)
        return loss.curve_score(phi_aug @ weights, target)

    active = torch.ones(d_max, device=device)
    curve: dict[int, float] = {d_max: score_with(active)}
    removal_order: list[int] = []
    while int(active.sum()) > 1:
        best_score, best_idx = None, -1
        for idx in range(d_max):
            if active[idx] == 0.0:
                continue
            candidate = active.clone()
            candidate[idx] = 0.0
            score = score_with(candidate)
            if best_score is None or score < best_score:
                best_score, best_idx = score, idx
        active[best_idx] = 0.0
        removal_order.append(best_idx)
        assert best_score is not None
        curve[int(active.sum())] = best_score

    return KnockoutResult(curve=curve, removal_order=removal_order, knee=detect_knockout_knee(curve, ratio=ratio))


def detect_knockout_knee(curve: dict[int, float], *, ratio: float = 2.0) -> int:
    """Locate the smallest number of active coordinates before the score blows up.

    Walking from few to many active coordinates, the knee is the first count
    whose score is within ``ratio`` of the best score over the curve. A
    single-point curve returns 1 only when it genuinely holds the best score
    — the detector never fires unconditionally at ``n = 1``.

    Args:
        curve: ``{n_active: curve_score}`` (lower is better).
        ratio: Blow-up threshold relative to the best score.

    Returns:
        The certified dimension.
    """
    if not curve:
        raise IGLConfigError("cannot detect a knee on an empty curve")
    if len(curve) < _MIN_KNEE_POINTS:
        return next(iter(curve))
    counts = sorted(curve)
    best = min(curve.values())
    floor = best * ratio if best > 0 else ratio - 1.0
    for count in counts:
        if curve[count] <= floor:
            return count
    return counts[-1]
