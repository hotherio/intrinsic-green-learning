"""SPD-side preconditioning transforms for AIRM-based reconstruction.

A small dispatch over the four modes catalogued in
``alex-eeg-igl/MAINTAINER_MEMO_lwf_tikh_rules.md``:

- ``NONE`` — passthrough.
- ``TIKHONOV`` — ``C + epsilon * I``. Bit-identical to ``NONE`` at
  ``d ≤ 64`` because the encoder ``BatchNorm`` absorbs the constant
  offset, and rescues ``torch.linalg.eigh`` from LAPACK error 8481 at
  ``d ≥ 128``.
- ``TRACE`` — per-matrix trace-normalisation ``C / trace(C) * d``.
- ``TIKHONOV_TRACE`` — trace then tikhonov.

The function is pure, stateless, and accepts batched ``[..., d, d]``
SPD inputs; the wrapping ``IGLReconSPDClassifier`` calls it on every
``covs`` tensor at fit time.
"""

from __future__ import annotations

import torch

from igl.exceptions import IGLConfigError
from igl.types import PreconditionMode, PreconditionModeLike


def precondition(
    c: torch.Tensor,
    mode: PreconditionModeLike = PreconditionMode.TIKHONOV,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Apply SPD-side preconditioning to a batch of covariance matrices.

    Args:
        c: ``[..., d, d]`` batched SPD matrices. The last two dims must be
            the matrix dimensions; any leading dims are batched.
        mode: Which preconditioner to apply. Strings are accepted via the
            :class:`PreconditionModeLike` alias and coerced through the
            enum at entry.
        epsilon: Ridge magnitude for the Tikhonov branches. Ignored for
            ``NONE`` / ``TRACE``. Default ``1e-6`` is the value
            characterised in the memo as universally safe.

    Returns:
        ``[..., d, d]`` preconditioned SPD matrices, on the same device
        and dtype as ``c``.

    Raises:
        IGLConfigError: For an unknown ``mode``.
    """
    # Coerce at entry so the enum is the canonical reference; the rest of
    # the function dispatches on the enum value.
    try:
        resolved = PreconditionMode(mode)
    except ValueError as exc:  # unknown string from the Literal alias
        valid = ", ".join(m.value for m in PreconditionMode)
        raise IGLConfigError(
            f"unknown precondition mode {mode!r}; expected one of: {valid}",
        ) from exc

    if resolved is PreconditionMode.NONE:
        return c

    d = c.shape[-1]

    if resolved is PreconditionMode.TIKHONOV:
        eye = torch.eye(d, dtype=c.dtype, device=c.device)
        return c + epsilon * eye

    if resolved is PreconditionMode.TRACE:
        return _trace_normalise(c, d)

    # TIKHONOV_TRACE — trace then tikhonov, matching the memo's stated
    # composition order. The trace branch already moves the spectrum to
    # ``trace == d``; the tikhonov branch then nudges off the floor by
    # ``epsilon``.
    return precondition(
        _trace_normalise(c, d),
        mode=PreconditionMode.TIKHONOV,
        epsilon=epsilon,
    )


def _trace_normalise(c: torch.Tensor, d: int) -> torch.Tensor:
    """Per-matrix trace normalisation: ``C / trace(C) * d``.

    ``trace`` is computed along the last two dims and broadcast against
    the batch shape.
    """
    # ``diagonal`` returns ``[..., d]``; sum along the last axis gives
    # ``[..., 1, 1]`` after two unsqueezes — broadcasts cleanly with the
    # ``[..., d, d]`` input.
    trace = c.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
    return c * d / trace


__all__ = ["precondition"]
