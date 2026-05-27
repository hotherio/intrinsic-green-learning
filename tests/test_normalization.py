"""Tests for :func:`igl.normalize_phi`."""

import pytest
import torch

from igl import IGLConfigError, normalize_phi


def test_none_returns_input_unchanged() -> None:
    phi = torch.randn(3, 5)
    out = normalize_phi(phi, "none")
    torch.testing.assert_close(out, phi, rtol=0, atol=0)


def test_softmax_rows_sum_to_one() -> None:
    phi = torch.randn(4, 6)
    out = normalize_phi(phi, "softmax")
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(4), rtol=1e-5, atol=1e-6)


def test_l2_rows_have_unit_norm() -> None:
    phi = torch.randn(4, 6)
    out = normalize_phi(phi, "l2")
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(4), rtol=1e-5, atol=1e-6)


def test_nw_rows_sum_to_one_for_positive_phi() -> None:
    phi = torch.rand(4, 6)  # all positive
    out = normalize_phi(phi, "nw")
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(4), rtol=1e-5, atol=1e-6)


def test_unknown_mode_raises() -> None:
    with pytest.raises(IGLConfigError, match="unknown normalize mode"):
        normalize_phi(torch.zeros(2, 2), "bogus")  # type: ignore[arg-type]
