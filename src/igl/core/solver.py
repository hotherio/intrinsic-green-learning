"""Closed-form Variable-Projection solve for the readout weights.

Given a design matrix Φ and targets y, the readout weights are obtained by

    w* = argmin ‖Φ w − y‖² + λ_eff ‖w‖²

with λ_eff auto-scaled to Φ's mean column norm so that a fixed user-facing
``l2`` translates into the same regularization strength regardless of how
Φ's scale drifts during training. The solve runs on CPU because
``torch.linalg.lstsq`` is unreliable on MPS for some shapes; the round-trip
is negligible for the sizes IGL typically uses (R ≤ 256, N ≤ 10K).
"""

import warnings
from typing import cast

import torch


@torch.no_grad()
def direct_solve_weights(
    phi: torch.Tensor,
    y: torch.Tensor,
    *,
    l2: float = 1e-3,
) -> torch.Tensor:
    """Solve the Tikhonov-regularised least-squares problem.

    The solution is computed via QR (``torch.linalg.lstsq``) on the stacked
    system ``[Φ; √λ I] w ≈ [y; 0]`` so the regularisation is correctly applied
    even when Φ is rank-deficient.

    Args:
        phi: Design matrix ``[N, R]``.
        y: Targets ``[N, C]`` or ``[N, 1]``.
        l2: Tikhonov coefficient relative to Φ's mean column norm (default
            ``1e-3``).

    Returns:
        ``w`` of shape ``[R, C]``. If the solve produces non-finite entries
        (e.g. catastrophic conditioning), the function emits a
        :class:`RuntimeWarning` and returns a zero matrix of the right shape.
    """
    phi_cpu = phi.detach().cpu().float()
    y_cpu = y.detach().cpu().float()
    if y_cpu.dim() == 1:
        y_cpu = y_cpu.unsqueeze(-1)

    n_anchors = phi_cpu.shape[1]
    # torch's column-norm chain is partially typed; cast to recover the known type.
    col_scale = cast(torch.Tensor, phi_cpu.norm(dim=0).mean().clamp_min(1e-6))  # pyright: ignore[reportUnknownMemberType]
    col_scale_value = float(cast(float, col_scale.item()))
    l2_eff = l2 * col_scale_value**2

    eye = torch.eye(n_anchors, dtype=phi_cpu.dtype)
    phi_aug = torch.cat([phi_cpu, (l2_eff**0.5) * eye], dim=0)
    y_aug = torch.cat([y_cpu, torch.zeros(n_anchors, y_cpu.shape[1], dtype=y_cpu.dtype)], dim=0)

    # torch.linalg.lstsq has no stubs; cast the solution back to Tensor.
    weights = cast(torch.Tensor, torch.linalg.lstsq(phi_aug, y_aug).solution)  # pyright: ignore[reportUnknownMemberType]

    if not torch.isfinite(weights).all():
        warnings.warn(
            f"direct_solve_weights produced non-finite weights "
            f"(col_scale={col_scale_value:.3g}, l2_eff={l2_eff:.3g}); "
            f"falling back to zero weights.",
            RuntimeWarning,
            stacklevel=2,
        )
        weights = torch.zeros_like(weights)
    return weights


__all__ = ["direct_solve_weights"]
