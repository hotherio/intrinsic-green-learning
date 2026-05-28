"""Tests for ``igl.spd.linalg``."""

import pytest
import torch

from igl import IGLConfigError
from igl.spd import matrix_exp_sym, matrix_log_sym, matrix_pow_sym, unpack_sym_vec


def _random_spd(batch: int, d: int) -> torch.Tensor:
    torch.manual_seed(0)
    a = torch.randn(batch, d, d)
    return a @ a.transpose(-1, -2) + d * torch.eye(d)


def test_matrix_log_exp_round_trip_for_spd() -> None:
    c = _random_spd(5, 4)
    c_round = matrix_exp_sym(matrix_log_sym(c))
    torch.testing.assert_close(c, c_round, rtol=1e-4, atol=1e-4)


def test_matrix_log_of_identity_is_zero() -> None:
    identity = torch.eye(3).unsqueeze(0).expand(2, 3, 3)
    log_i = matrix_log_sym(identity)
    torch.testing.assert_close(log_i, torch.zeros_like(log_i), rtol=1e-5, atol=1e-5)


def test_matrix_exp_of_zero_is_identity() -> None:
    zero = torch.zeros(2, 3, 3)
    exp_zero = matrix_exp_sym(zero)
    expected = torch.eye(3).unsqueeze(0).expand(2, 3, 3)
    torch.testing.assert_close(exp_zero, expected, rtol=1e-5, atol=1e-5)


def test_matrix_pow_with_one_is_identity() -> None:
    c = _random_spd(3, 4)
    torch.testing.assert_close(matrix_pow_sym(c, 1.0), c, rtol=1e-4, atol=1e-4)


def test_matrix_pow_neg_half_then_squared_is_inverse() -> None:
    """``C^{-1/2} · C^{-1/2} = C^{-1}`` for SPD ``C``."""
    c = _random_spd(2, 3)
    c_inv_half = matrix_pow_sym(c, -0.5)
    c_inv = c_inv_half @ c_inv_half
    identity_check = c @ c_inv
    expected = torch.eye(3).unsqueeze(0).expand(2, 3, 3)
    torch.testing.assert_close(identity_check, expected, rtol=1e-3, atol=1e-3)


def test_unpack_sym_vec_round_trips_against_log_eig() -> None:
    """Pack a symmetric matrix with √2 off-diag scaling, then unpack."""
    import math  # noqa: PLC0415

    d = 4
    sym = torch.randn(3, d, d)
    sym = 0.5 * (sym + sym.transpose(-1, -2))
    rows, cols = torch.triu_indices(d, d, offset=0)
    on_diag = (rows == cols).float()
    scale = on_diag + (1.0 - on_diag) * math.sqrt(2.0)
    vec = sym[:, rows, cols] * scale
    unpacked = unpack_sym_vec(vec, d)
    torch.testing.assert_close(unpacked, sym, rtol=1e-5, atol=1e-5)


def test_unpack_sym_vec_rejects_wrong_size() -> None:
    with pytest.raises(IGLConfigError, match="d=4"):
        unpack_sym_vec(torch.zeros(2, 9), d=4)
