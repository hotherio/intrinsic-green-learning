"""Closed-form Variable-Projection solve for the readout weights.

Given a design matrix Φ and targets y, the readout weights are obtained by

    w* = argmin ‖Φ w − y‖² + λ_eff ‖w‖²

with λ_eff auto-scaled to Φ's mean column norm so that a fixed user-facing
``l2`` translates into the same regularization strength regardless of how
Φ's scale drifts during training. The solve runs on CPU because
``torch.linalg.lstsq`` is unreliable on MPS for some shapes; the round-trip
is negligible for the sizes IGL typically uses (R ≤ 256, N ≤ 10K).
"""

import contextlib
import warnings
from collections.abc import Generator
from typing import Literal, cast

import torch

from igl.exceptions import IGLConvergenceError


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
    # torch.linalg.svd has no stubs; cast the result to recover known types.
    u, s, vh = cast(
        "tuple[torch.Tensor, torch.Tensor, torch.Tensor]",
        torch.linalg.svd(a64, full_matrices=False),  # pyright: ignore[reportUnknownMemberType]
    )
    tol = s.max() * max(a64.shape) * torch.finfo(s.dtype).eps
    s_inv = torch.where(s > tol, s.reciprocal(), torch.zeros_like(s))
    return (vh.mT @ (s_inv.unsqueeze(-1) * (u.mT @ b.double()))).to(a.dtype)


@torch.no_grad()
def direct_solve_weights(
    phi: torch.Tensor,
    y: torch.Tensor,
    *,
    l2: float = 1e-3,
    on_nonfinite: Literal["warn", "raise"] = "warn",
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
        on_nonfinite: Policy when inputs or the solution are non-finite.
            ``"warn"`` (default) emits a :class:`RuntimeWarning` and returns a
            zero matrix — the right behaviour inside training, where the
            trainer's convergence guards take over. ``"raise"`` raises
            :class:`igl.exceptions.IGLConvergenceError` instead — the right
            behaviour at evaluation time, where a silent zero readout would
            poison downstream metrics.

    Returns:
        ``w`` of shape ``[R, C]``.

    Raises:
        IGLConvergenceError: When non-finite values are encountered and
            ``on_nonfinite="raise"``.
    """
    # Non-finite INPUT guard. A diverged encoder feeds NaN/Inf into phi; the
    # lstsq/SVD backends then crash with an opaque LinAlgError ("svd: input
    # matrix contained non-finite values") at whichever call site hit it first
    # (the per-batch solve and the validation refresh are both outside the
    # trainer's skip_failing_batches guard). Mirror the existing non-finite
    # *output* behaviour: warn and return zeros, so training fails through the
    # clean IGLConvergenceError path instead of an opaque crash. No-op on
    # healthy runs (inputs are finite), so the validated path is bit-identical.
    n_cols = y.shape[1] if y.dim() > 1 else 1
    if not (torch.isfinite(phi).all() and torch.isfinite(y).all()):
        message = "direct_solve_weights received non-finite inputs (diverged training)"
        if on_nonfinite == "raise":
            raise IGLConvergenceError(epoch=0, last_loss=float("nan"), message=message)
        warnings.warn(f"{message}; returning zero weights.", RuntimeWarning, stacklevel=2)
        return torch.zeros(phi.shape[1], n_cols, dtype=torch.float32, device=phi.device)

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

    if on_cuda:  # pragma: no cover  # CUDA only
        # The device branch: Cholesky + one refinement step, no float64 SVD.
        weights, bad = ridge_solve_device(phi_w, y_w, l2=l2)
        if bool(bad):  # one host sync: this public entry point promises a warning or an exception
            message = "direct_solve_weights: the device solve failed (non-finite inputs or factorisation)"
            if on_nonfinite == "raise":
                raise IGLConvergenceError(epoch=0, last_loss=float("nan"), message=message)
            warnings.warn(f"{message}; returning zero weights.", RuntimeWarning, stacklevel=2)
        return weights
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
        message = f"direct_solve_weights produced non-finite weights (col_scale={col_scale_value:.3g}, l2_eff={l2_eff:.3g})"
        if on_nonfinite == "raise":
            raise IGLConvergenceError(epoch=0, last_loss=float("nan"), message=message)
        warnings.warn(f"{message}; falling back to zero weights.", RuntimeWarning, stacklevel=2)
        weights = torch.zeros_like(weights)
    return weights


@torch.no_grad()
def solve_with_intercept(
    phi: torch.Tensor,
    target: torch.Tensor,
    *,
    l2: float = 1e-3,
    on_nonfinite: Literal["warn", "raise"] = "warn",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ridge solve with a free intercept, by centring.

    Fits ``target ≈ Φ w + b`` with the ridge on ``w`` only: both sides are
    centred, ``w`` solves the centred system through
    :func:`direct_solve_weights`, and ``b = ȳ − Φ̄ w``. Appending a column of
    ones instead has two defects: the column is collinear with a
    row-normalised Φ (rows summing to one), and its norm dominates the mean
    column norm the ridge is scaled by, so the anchors were shrunk harder at
    evaluation than during training.

    Args:
        phi: Design matrix ``[N, R]``.
        target: Targets ``[N, C]`` or ``[N]``.
        l2: Tikhonov coefficient, forwarded to :func:`direct_solve_weights`.
        on_nonfinite: Forwarded to :func:`direct_solve_weights`.

    Returns:
        ``(w, b)`` with ``w`` of shape ``[R, C]`` and ``b`` of shape ``[C]``,
        on the device :func:`direct_solve_weights` returns.
    """
    target_2d = target if target.dim() > 1 else target.unsqueeze(-1)
    phi_mean = phi.detach().float().mean(dim=0, keepdim=True)
    target_mean = target_2d.detach().float().mean(dim=0, keepdim=True)
    if phi.device.type == "cpu":
        weights = direct_solve_weights(phi - phi_mean, target_2d - target_mean, l2=l2, on_nonfinite=on_nonfinite)
    else:
        # On-device branch (MPS, CUDA): one host read of the failure flag is the
        # price of this entry point's warn-or-raise contract; the trainer never
        # comes through here.
        weights, bad = ridge_solve_device(phi - phi_mean, target_2d - target_mean, l2=l2)
        if bool(bad):
            message = "solve_with_intercept: the device solve failed (non-finite inputs or factorisation)"
            if on_nonfinite == "raise":
                raise IGLConvergenceError(epoch=0, last_loss=float("nan"), message=message)
            warnings.warn(f"{message}; returning zero weights.", RuntimeWarning, stacklevel=2)
    intercept = (target_mean.to(weights.device) - phi_mean.to(weights.device) @ weights).reshape(-1)
    return weights, intercept


@contextlib.contextmanager
def _full_precision_matmul(device: torch.device) -> Generator[None]:
    """Disable TF32 for the duration of a solve on CUDA (no-op elsewhere).

    The Gram matrix and the refinement residual must not run through TF32:
    with cond(ΦᵀΦ+λI) up to 1e5 the 10-bit mantissa doubles the solve error.
    """
    if device.type != "cuda":
        yield
        return
    before = torch.backends.cuda.matmul.allow_tf32  # pragma: no cover  # CUDA only
    torch.backends.cuda.matmul.allow_tf32 = False  # pragma: no cover  # CUDA only
    try:  # pragma: no cover  # CUDA only
        yield
    finally:  # pragma: no cover  # CUDA only
        torch.backends.cuda.matmul.allow_tf32 = before


@torch.no_grad()
def ridge_solve_device(
    phi: torch.Tensor,
    y: torch.Tensor,
    *,
    l2: float = 1e-3,
    check_errors: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tikhonov least squares entirely on ``phi``'s device, without host synchronisation.

    Solves the normal equations ``(ΦᵀΦ + λ_eff I) w = Φᵀy`` by a Cholesky
    factorisation followed by ONE step of iterative refinement. Plain fp32
    normal equations lose three digits at the conditioning of real design
    matrices (cond 6e4-3e5 measured); the refinement step brings the
    prediction error back to the level of the QR-based ``lstsq`` at a fraction
    of its cost (4-20x faster on MPS and CUDA, measured). ``λ_eff`` is scaled
    by Φ's mean column norm exactly as in :func:`direct_solve_weights`.

    Nothing here reaches the host: the ridge scale stays a 0-d tensor,
    non-finite inputs are neutralised with ``nan_to_num``, and failure is
    reported through the returned flag instead of an exception.

    Args:
        phi: Design matrix ``[N, R]`` on any device.
        y: Targets ``[N, C]`` or ``[N]``.
        l2: Tikhonov coefficient relative to Φ's mean column norm.
        check_errors: Forwarded to ``torch.linalg.cholesky_ex``; ``True`` makes
            a failed factorisation raise (and synchronise) instead of flagging.

    Returns:
        ``(w, bad)``: ``w`` of shape ``[R, C]`` on ``phi``'s device (zeros when
        the solve failed) and ``bad``, a 0-d boolean tensor on the same device
        that is true when the inputs were non-finite, the factorisation
        failed, or the solution was non-finite.
    """
    phi32 = phi.detach().float()
    y32 = y.detach().float()
    if y32.dim() == 1:
        y32 = y32.unsqueeze(-1)
    with _full_precision_matmul(phi32.device):
        ok = torch.isfinite(phi32).all() & torch.isfinite(y32).all()
        phi_c = torch.nan_to_num(phi32)
        y_c = torch.nan_to_num(y32)
        col_scale = cast(torch.Tensor, phi_c.norm(dim=0).mean().clamp_min(1e-6))  # pyright: ignore[reportUnknownMemberType]
        l2_eff = l2 * col_scale * col_scale
        eye = torch.eye(phi_c.shape[1], dtype=phi_c.dtype, device=phi_c.device)
        gram = phi_c.T @ phi_c + l2_eff * eye
        rhs = phi_c.T @ y_c
        chol, info = cast(
            "tuple[torch.Tensor, torch.Tensor]",
            torch.linalg.cholesky_ex(gram, check_errors=check_errors),  # pyright: ignore[reportUnknownMemberType]
        )
        chol = torch.nan_to_num(chol)
        weights = torch.cholesky_solve(rhs, chol)
        residual = rhs - (phi_c.T @ (phi_c @ weights) + l2_eff * weights)
        weights = weights + torch.cholesky_solve(residual, chol)
        bad = (~ok) | (info > 0) | ~torch.isfinite(weights).all()
        weights = torch.where(bad, torch.zeros_like(weights), torch.nan_to_num(weights))
    return weights, bad


__all__ = ["direct_solve_weights", "ridge_solve_device", "solve_with_intercept"]
