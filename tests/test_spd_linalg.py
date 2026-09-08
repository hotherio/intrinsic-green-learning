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


def _eeg_like(b: int, d: int, span: float) -> torch.Tensor:
    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(b, d, d, dtype=torch.float64))
    lam = torch.logspace(-span, 0, d, dtype=torch.float64)[None].expand(b, d) * (0.5 + torch.rand(b, d, dtype=torch.float64))
    return q @ torch.diag_embed(lam) @ q.transpose(-1, -2)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max() / b.double().abs().max())


def test_iterative_matrix_functions_match_eigh_on_well_conditioned_spectra() -> None:
    from igl.spd.linalg import matrix_exp_sym, matrix_log_sym, matrix_pow_sym

    c = _eeg_like(6, 8, span=2.0).float()
    assert _rel(matrix_log_sym(c, method="iterative"), matrix_log_sym(c.double())) < 1e-5
    assert _rel(matrix_pow_sym(c, -0.5, method="iterative"), matrix_pow_sym(c.double(), -0.5)) < 1e-5
    assert _rel(matrix_pow_sym(c, 0.5, method="iterative"), matrix_pow_sym(c.double(), 0.5)) < 1e-5
    g = torch.randn(6, 8, 8)
    s = 0.5 * (g + g.transpose(-1, -2))
    assert _rel(matrix_exp_sym(s, method="iterative"), matrix_exp_sym(s.double())) < 1e-5


def test_iterative_matrix_functions_sit_in_the_fp32_eigh_error_band_on_eeg_spectra() -> None:
    """Six orders of magnitude: fp32 eigh itself is 5e-4..8e-3 from fp64; the iterative path is no worse."""
    from igl.spd.linalg import matrix_log_sym, matrix_pow_sym

    c = _eeg_like(8, 16, span=6.0)
    ref_log = matrix_log_sym(c)
    err_eigh = _rel(matrix_log_sym(c.float()), ref_log)
    err_iter = _rel(matrix_log_sym(c.float(), method="iterative"), ref_log)
    assert err_iter < max(3 * err_eigh, 5e-3), (err_iter, err_eigh)
    ref_pow = matrix_pow_sym(c, -0.5)
    assert _rel(matrix_pow_sym(c.float(), -0.5, method="iterative"), ref_pow) < 1e-2


def test_iterative_power_rejects_unsupported_exponents() -> None:
    from igl import IGLConfigError
    from igl.spd.linalg import matrix_pow_sym

    with pytest.raises(IGLConfigError, match="±0.5"):
        matrix_pow_sym(torch.eye(3)[None], 0.25, method="iterative")


def test_unpack_sym_vec_is_bit_identical_to_the_masked_formulation() -> None:
    import math

    from igl.spd.linalg import unpack_sym_vec

    torch.manual_seed(0)
    d = 6
    vec = torch.randn(9, d * (d + 1) // 2)
    rows, cols = torch.triu_indices(d, d)
    on_diag = (rows == cols).to(vec.dtype)
    unscaled = vec * (on_diag + (1.0 - on_diag) / math.sqrt(2.0))
    expected = torch.zeros(9, d, d)
    expected[:, rows, cols] = unscaled
    mask = rows != cols
    expected[:, cols[mask], rows[mask]] = unscaled[:, mask]
    assert torch.equal(unpack_sym_vec(vec, d), expected)


def test_iterative_exp_matches_fp64_on_wide_symmetric_spectra() -> None:
    from igl.spd.linalg import matrix_exp_sym

    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(8, 16, 16, dtype=torch.float64))
    lam = -40.0 + 80.0 * torch.rand(8, 16, dtype=torch.float64)
    s64 = q @ torch.diag_embed(lam) @ q.transpose(-1, -2)
    ref = matrix_exp_sym(s64)
    assert _rel(matrix_exp_sym(s64.float(), method="iterative"), ref) < 1e-5
