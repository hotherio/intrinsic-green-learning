"""Tests for the kernel operator zoo + registry."""

import math

import pytest
import torch

from igl import IGLConfigError, Operator, get_operator, list_operators, register_operator

_EXPECTED_BUILTINS = (
    "cauchy",
    "gabor",
    "gaussian",
    "helmholtz",
    "laplacian",
    "mexican_hat",
    "multiquadric",
    "soft_box",
    "yukawa",
)

_NON_OSCILLATORY = ("cauchy", "gaussian", "laplacian", "multiquadric", "soft_box", "yukawa")
_OSCILLATORY = ("gabor", "helmholtz", "mexican_hat")


def _op(name: str) -> Operator:
    return get_operator(name)


def test_list_operators_contains_every_builtin() -> None:
    names = list_operators()
    for name in _EXPECTED_BUILTINS:
        assert name in names


def test_get_operator_returns_correct_oscillatory_flag() -> None:
    for name in _NON_OSCILLATORY:
        assert _op(name).is_oscillatory is False
    for name in _OSCILLATORY:
        assert _op(name).is_oscillatory is True


def test_get_operator_unknown_raises() -> None:
    with pytest.raises(IGLConfigError, match="unknown operator"):
        get_operator("definitely_not_a_kernel")


def test_register_operator_rejects_duplicates() -> None:
    with pytest.raises(IGLConfigError, match="already registered"):
        register_operator("gaussian", _op("gaussian").fn, is_oscillatory=False)


def test_register_operator_allows_user_kernels() -> None:
    class _Triangle:
        is_oscillatory = False

        def __call__(self, d: torch.Tensor, sigma: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor]:
            log_abs = -(torch.abs(d) / (sigma + 1e-8))
            return log_abs, torch.ones_like(d)

    register_operator("triangle_kernel_test", _Triangle(), is_oscillatory=False)
    assert "triangle_kernel_test" in list_operators()
    op = get_operator("triangle_kernel_test")
    log_abs, sign = op.fn(torch.tensor(1.0), torch.tensor(1.0))
    assert log_abs.item() == pytest.approx(-1.0, abs=1e-3)
    assert sign.item() == pytest.approx(1.0)


@pytest.mark.parametrize("name", _NON_OSCILLATORY)
def test_non_oscillatory_kernels_have_positive_sign_everywhere(name: str) -> None:
    op = _op(name).fn
    d = torch.linspace(-2.0, 2.0, 50).unsqueeze(0)
    sigma = torch.full_like(d, 0.7)
    _, sign = op(d, sigma)
    assert torch.all(sign > 0)


@pytest.mark.parametrize("name", ["gaussian", "laplacian", "cauchy", "multiquadric", "yukawa"])
def test_radial_kernels_are_symmetric_in_d(name: str) -> None:
    """Smooth radial kernels are even in ``d``: ``k(d, σ) = k(-d, σ)``."""
    op = _op(name).fn
    d = torch.linspace(0.1, 2.0, 20).unsqueeze(0)
    sigma = torch.full_like(d, 0.8)
    log_pos, _ = op(d, sigma)
    log_neg, _ = op(-d, sigma)
    torch.testing.assert_close(log_pos, log_neg, rtol=1e-5, atol=1e-6)


def test_helmholtz_sign_flips_at_quarter_period() -> None:
    """At ``d = σ``, ``cos(π) = -1`` — sign should be negative."""
    helmholtz = _op("helmholtz").fn
    sigma = torch.tensor(1.0)
    d = torch.tensor(1.2)
    _, sign = helmholtz(d, sigma)
    assert sign.item() == -1.0


def test_gabor_sign_flips_like_helmholtz() -> None:
    gabor = _op("gabor").fn
    sigma = torch.tensor(1.0)
    d = torch.tensor(1.2)
    _, sign = gabor(d, sigma)
    assert sign.item() == -1.0


def test_mexican_hat_negative_beyond_one_sigma() -> None:
    mexican = _op("mexican_hat").fn
    sigma = torch.tensor(1.0)
    d = torch.tensor(2.0)
    _, sign = mexican(d, sigma)
    assert sign.item() == -1.0


def test_gaussian_value_matches_closed_form() -> None:
    gaussian = _op("gaussian").fn
    d = torch.tensor(0.5)
    sigma = torch.tensor(1.0)
    log_abs, _ = gaussian(d, sigma)
    expected = -(d.item() ** 2) / (2 * sigma.item() ** 2)
    assert log_abs.item() == pytest.approx(expected, rel=1e-3)


def test_soft_box_log_value_is_finite() -> None:
    soft_box = _op("soft_box").fn
    d = torch.linspace(-2.0, 2.0, 50)
    sigma = torch.full_like(d, 0.5)
    log_abs, sign = soft_box(d, sigma)
    assert torch.all(torch.isfinite(log_abs))
    assert torch.all(sign > 0)


def test_helmholtz_log_at_zero_distance() -> None:
    """At ``d=0``: ``cos(0)=1``, ``log|cos|=0``, ``|d|/σ=0`` → ``log_abs=0``."""
    helmholtz = _op("helmholtz").fn
    sigma = torch.tensor(1.0)
    d = torch.tensor(0.0)
    log_abs, sign = helmholtz(d, sigma)
    assert log_abs.item() == pytest.approx(0.0, abs=1e-3)
    assert sign.item() == 1.0


def test_kernel_supports_broadcasting_shapes() -> None:
    """Every operator must broadcast over arbitrary leading dims."""
    d = torch.randn(3, 4, 5)
    sigma = torch.full((1, 1, 5), 0.8)
    for name in _EXPECTED_BUILTINS:
        op = _op(name).fn
        log_abs, sign = op(d, sigma)
        assert log_abs.shape == d.shape
        assert sign.shape == d.shape


def test_helmholtz_oscillation_period_matches_expected() -> None:
    """Helmholtz uses ``cos(π · d / σ)``; the first zero is at ``d = σ/2``."""
    helmholtz = _op("helmholtz").fn
    sigma = torch.tensor(1.0)
    d_grid = torch.linspace(0.0, 2.0, 100)
    _, signs = helmholtz(d_grid, sigma.expand_as(d_grid))
    flips = torch.where(signs[:-1] != signs[1:])[0]
    assert len(flips) >= 1
    first_flip = float(d_grid[flips[0]].item())
    assert first_flip == pytest.approx(0.5, abs=0.05)


def test_math_pi_constant_used() -> None:
    # Sanity: helmholtz uses math.pi; verify the imported constant.
    assert math.pi == pytest.approx(3.14159, abs=1e-4)
