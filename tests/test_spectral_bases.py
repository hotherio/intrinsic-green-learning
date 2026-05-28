"""Tests for the closed-form 1-D spectral bases."""

from __future__ import annotations

import math

import pytest
import torch

from igl import IGLConfigError
from igl.spectral import (
    ChebyshevBasis,
    FourierCosineBasis,
    FourierSineBasis,
    HermiteBasis,
    LaguerreBasis,
    LegendreBasis,
)

_RTOL = 1e-3
_ATOL = 1e-3


@pytest.mark.parametrize(
    ("cls", "z_range"),
    [
        (FourierSineBasis, (0.05, 0.95)),
        (FourierCosineBasis, (0.05, 0.95)),
        (ChebyshevBasis, (0.05, 0.95)),
        (LegendreBasis, (0.05, 0.95)),
        (HermiteBasis, (-2.0, 2.0)),
        (LaguerreBasis, (0.1, 4.0)),
    ],
)
def test_basis_shape_contract(cls: type, z_range: tuple[float, float]) -> None:
    basis = cls(n_modes=6)
    z = torch.linspace(z_range[0], z_range[1], 8)
    out = basis(z)
    assert out.shape == (8, 6)


def test_fourier_sine_orthonormality_on_unit_interval() -> None:
    """Numerical Riemann sum: ⟨φₙ, φₘ⟩ ≈ δₙₘ on a fine grid."""
    basis = FourierSineBasis(n_modes=6)
    z = torch.linspace(1e-4, 1.0 - 1e-4, 5000)
    phi = basis(z)
    inner = (phi.T @ phi) / 5000.0
    torch.testing.assert_close(inner, torch.eye(6), rtol=_RTOL, atol=_ATOL)


def test_fourier_sine_no_null_indices() -> None:
    basis = FourierSineBasis(n_modes=4)
    assert basis.null_indices == ()


def test_fourier_cosine_constant_is_null_mode() -> None:
    basis = FourierCosineBasis(n_modes=5)
    z = torch.linspace(0.0, 1.0, 10)
    phi = basis(z)
    # Mode 0 is the constant 1.
    assert torch.all(phi[:, 0] == 1.0)
    assert basis.null_indices == (0,)


def test_chebyshev_recurrence_first_few_modes() -> None:
    basis = ChebyshevBasis(n_modes=4)
    z = torch.tensor([0.25, 0.5, 0.75])
    out = basis(z)
    x = 2.0 * z - 1.0
    expected = torch.stack([torch.ones_like(x), x, 2 * x**2 - 1, 4 * x**3 - 3 * x], dim=-1)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_legendre_eigenvalues_match_formula() -> None:
    basis = LegendreBasis(n_modes=4)
    indices = torch.arange(0, 4, dtype=torch.float32)
    expected = indices * (indices + 1.0)
    expected[0] = 1e-4
    torch.testing.assert_close(basis.eigenvalues, expected, rtol=1e-5, atol=1e-5)


def test_hermite_first_mode_is_gaussian() -> None:
    basis = HermiteBasis(n_modes=3)
    z = torch.tensor([0.0, 0.5, 1.0])
    out = basis(z)
    # φ₀(z) = exp(-z²/2) / π^(1/4)
    expected = torch.exp(-0.5 * z**2) / (math.pi**0.25)
    torch.testing.assert_close(out[:, 0], expected, rtol=1e-4, atol=1e-5)


def test_laguerre_first_mode_decays_exponentially() -> None:
    basis = LaguerreBasis(n_modes=2)
    z = torch.tensor([0.0, 1.0, 2.0])
    out = basis(z)
    expected = torch.exp(-0.5 * z)
    torch.testing.assert_close(out[:, 0], expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "cls",
    [FourierSineBasis, FourierCosineBasis, ChebyshevBasis, LegendreBasis, HermiteBasis, LaguerreBasis],
)
def test_basis_rejects_zero_modes(cls: type) -> None:
    with pytest.raises(IGLConfigError, match="n_modes"):
        cls(n_modes=0)


@pytest.mark.parametrize(
    "cls",
    [FourierCosineBasis, ChebyshevBasis, LegendreBasis],
)
def test_basis_rejects_non_positive_epsilon(cls: type) -> None:
    with pytest.raises(IGLConfigError, match="epsilon"):
        cls(n_modes=4, epsilon=0.0)


@pytest.mark.parametrize(
    "cls",
    [ChebyshevBasis, LegendreBasis, HermiteBasis, LaguerreBasis],
)
def test_basis_with_one_mode_returns_constant_column(cls: type) -> None:
    basis = cls(n_modes=1)
    z = torch.tensor([0.5, 0.7, 0.9]) if cls is not HermiteBasis and cls is not LaguerreBasis else torch.tensor([0.0, 0.5, 1.0])
    out = basis(z)
    assert out.shape == (3, 1)


def test_basis_eigenvalues_sorted_ascending() -> None:
    for cls in (FourierSineBasis, FourierCosineBasis, ChebyshevBasis, LegendreBasis, HermiteBasis, LaguerreBasis):
        b = cls(n_modes=6)
        diffs = b.eigenvalues[1:] - b.eigenvalues[:-1]
        assert torch.all(diffs >= 0)
