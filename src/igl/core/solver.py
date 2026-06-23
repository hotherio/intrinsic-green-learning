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


def _svd_pinv_solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Minimum-norm least-squares solve of ``a w = b`` via a float64 truncated
    SVD pseudoinverse, returned in ``a``'s dtype and on ``a``'s device.

    Backend-agnostic (CPU LAPACK, cuSOLVER): avoids ``torch.linalg.lstsq``,
    which is unreliable on MPS and on the linux x86 CPU (MKL) build for wide
    right-hand sides (INTERNAL ASSERT "Argument 4 has illegal value"). When
    ``a`` has full column rank — the IGL stacked system ``[phi; sqrt(l2) I]``
    always does — this is exactly the solution a working ``lstsq`` returns.
    """
    a64 = a.double()
    u, s, vh = torch.linalg.svd(a64, full_matrices=False)
    tol = s.max() * max(a64.shape) * torch.finfo(s.dtype).eps
    s_inv = torch.where(s > tol, s.reciprocal(), torch.zeros_like(s))
    return (vh.mT @ (s_inv.unsqueeze(-1) * (u.mT @ b.double()))).to(a.dtype)


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
    # On CUDA, keep the whole solve on-device (cuSOLVER) to avoid a per-call
    # host round-trip — under GPU training this is called once per minibatch.
    # On CPU/MPS, pin to CPU exactly as before: lstsq is unreliable on MPS, and
    # this path is bit-identical to the validated CPU results.
    on_cuda = phi.device.type == "cuda"
    if on_cuda:
        phi_w = phi.detach().float()
        y_w = y.detach().float()
    else:
        phi_w = phi.detach().cpu().float()
        y_w = y.detach().cpu().float()
    if y_w.dim() == 1:
        y_w = y_w.unsqueeze(-1)

    n_anchors = phi_w.shape[1]
    # torch's column-norm chain is partially typed; cast to recover the known type.
    col_scale = cast(torch.Tensor, phi_w.norm(dim=0).mean().clamp_min(1e-6))  # pyright: ignore[reportUnknownMemberType]
    col_scale_value = float(cast(float, col_scale.item()))
    l2_eff = l2 * col_scale_value**2

    eye = torch.eye(n_anchors, dtype=phi_w.dtype, device=phi_w.device)
    phi_aug = torch.cat([phi_w, (l2_eff**0.5) * eye], dim=0)
    y_aug = torch.cat(
        [y_w, torch.zeros(n_anchors, y_w.shape[1], dtype=y_w.dtype, device=y_w.device)],
        dim=0,
    )

    if on_cuda:
        # cuSOLVER: solve directly via the float64 truncated-SVD pseudoinverse —
        # robust for wide RHS at d >= 64 and keeping the result on-device.
        weights = _svd_pinv_solve(phi_aug, y_aug)
    else:
        # torch.linalg.lstsq has no stubs; cast the solution back to Tensor.
        try:
            weights = cast(torch.Tensor, torch.linalg.lstsq(phi_aug, y_aug).solution)  # pyright: ignore[reportUnknownMemberType]
        except RuntimeError:
            # lstsq is unreliable on MPS and on the linux x86 CPU (MKL) build,
            # which raises an INTERNAL ASSERT ("Argument 4 has illegal value")
            # for wide right-hand sides (SPD targets at d >= 64, D = d(d+1)/2).
            # Fall back to the same float64 SVD pseudoinverse used on CUDA.
            weights = _svd_pinv_solve(phi_aug, y_aug)

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
