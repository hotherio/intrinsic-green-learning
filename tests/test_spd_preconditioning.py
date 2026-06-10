"""Unit tests for the SPD-side preconditioning helper.

Covers the four modes catalogued in
``alex-eeg-igl/MAINTAINER_MEMO_lwf_tikh_rules.md``: ``none``, ``tikhonov``,
``trace``, ``tikhonov+trace``.
"""

from __future__ import annotations

import pytest
import torch

from igl import IGLConfigError, PreconditionMode
from igl.spd.preconditioning import precondition


def _random_spd_batch(n: int = 4, d: int = 6, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, d, d, generator=g)
    return a @ a.transpose(-2, -1) + 0.1 * torch.eye(d)


def test_precondition_none_is_passthrough() -> None:
    """``precondition(C, "none")`` returns C byte-identically."""
    c = _random_spd_batch()
    out = precondition(c, PreconditionMode.NONE)
    assert torch.equal(out, c)


def test_precondition_tikhonov_adds_epsilon_i() -> None:
    """``precondition(C, "tikhonov", eps)`` equals ``C + eps * I`` exactly."""
    c = _random_spd_batch()
    d = c.shape[-1]
    eps = 1e-3
    out = precondition(c, "tikhonov", epsilon=eps)
    expected = c + eps * torch.eye(d).expand_as(c)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_precondition_trace_normalises_to_d() -> None:
    """After ``"trace"``, every matrix has ``trace(C') == d``."""
    c = _random_spd_batch()
    d = c.shape[-1]
    out = precondition(c, "trace")
    traces = out.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    torch.testing.assert_close(traces, torch.full((c.shape[0],), float(d)), rtol=1e-5, atol=1e-5)


def test_precondition_tikhonov_trace_is_trace_then_tikhonov() -> None:
    """``"tikhonov+trace"`` is the trace-normalised SPD shifted by ``eps * I``."""
    c = _random_spd_batch()
    d = c.shape[-1]
    eps = 1e-3
    out = precondition(c, "tikhonov+trace", epsilon=eps)
    expected = precondition(c, "trace") + eps * torch.eye(d).expand_as(c)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_precondition_rejects_unknown_mode() -> None:
    """An unknown mode string raises :class:`IGLConfigError`."""
    c = _random_spd_batch()
    with pytest.raises(IGLConfigError, match="unknown precondition mode"):
        precondition(c, "bogus")  # type: ignore[arg-type]


def test_precondition_accepts_strenum_member_and_string() -> None:
    """Both enum members and the matching strings are accepted via the Like alias."""
    c = _random_spd_batch()
    d = c.shape[-1]
    out_enum = precondition(c, PreconditionMode.TIKHONOV, epsilon=1e-3)
    out_str = precondition(c, "tikhonov", epsilon=1e-3)
    torch.testing.assert_close(out_enum, out_str, rtol=0, atol=0)
    expected = c + 1e-3 * torch.eye(d).expand_as(c)
    torch.testing.assert_close(out_enum, expected, rtol=0, atol=0)


def test_precondition_preserves_device_and_dtype() -> None:
    """The output shares device + dtype with the input (no implicit promotion)."""
    c = _random_spd_batch().to(dtype=torch.float64)
    out = precondition(c, "tikhonov", epsilon=1e-6)
    assert out.dtype == torch.float64
    assert out.device == c.device
