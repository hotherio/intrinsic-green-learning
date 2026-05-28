"""Tests for the kernel-agnostic null-space augmentations."""

import pytest
import torch

from igl import IGLConfigError
from igl.spectral import (
    ConstantNullSpace,
    CustomNullSpace,
    PolynomialNullSpace,
    build_null_space,
)
from igl.types import NullSpaceKind


def test_constant_null_space_returns_ones() -> None:
    null = ConstantNullSpace()
    assert null.n_columns == 1
    z = torch.randn(7, 4)
    out = null.evaluate(z)
    assert out.shape == (7, 1)
    assert torch.all(out == 1.0)


def test_polynomial_null_space_degree_1_columns() -> None:
    null = PolynomialNullSpace(latent_dim=3, degree=1)
    # 1 constant + 3 linear monomials.
    assert null.n_columns == 4  # noqa: PLR2004
    z = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = null.evaluate(z)
    expected = torch.tensor([[1.0, 1.0, 2.0, 3.0], [1.0, 4.0, 5.0, 6.0]])
    torch.testing.assert_close(out, expected)


def test_polynomial_null_space_degree_2() -> None:
    null = PolynomialNullSpace(latent_dim=2, degree=2)
    # 1 + 2 (linear) + 2 (quadratic) = 5.
    assert null.n_columns == 5  # noqa: PLR2004
    z = torch.tensor([[1.0, 2.0]])
    out = null.evaluate(z)
    expected = torch.tensor([[1.0, 1.0, 2.0, 1.0, 4.0]])
    torch.testing.assert_close(out, expected)


def test_polynomial_null_space_degree_zero_is_constant() -> None:
    null = PolynomialNullSpace(latent_dim=4, degree=0)
    assert null.n_columns == 1
    z = torch.randn(3, 4)
    torch.testing.assert_close(null.evaluate(z), torch.ones(3, 1))


def test_polynomial_null_space_rejects_invalid_args() -> None:
    with pytest.raises(IGLConfigError, match="latent_dim"):
        PolynomialNullSpace(latent_dim=0)
    with pytest.raises(IGLConfigError, match="degree"):
        PolynomialNullSpace(latent_dim=3, degree=-1)


def test_polynomial_null_space_rejects_wrong_input_dim() -> None:
    null = PolynomialNullSpace(latent_dim=3)
    with pytest.raises(IGLConfigError, match="expected 3"):
        null.evaluate(torch.randn(5, 2))


def test_custom_null_space_wraps_callable() -> None:
    def two_columns(z: torch.Tensor) -> torch.Tensor:
        return torch.stack([z[:, 0], z[:, 0] ** 2], dim=-1)

    null = CustomNullSpace(two_columns, n_columns=2)
    z = torch.tensor([[3.0, 7.0], [1.0, 0.0]])
    out = null.evaluate(z)
    expected = torch.tensor([[3.0, 9.0], [1.0, 1.0]])
    torch.testing.assert_close(out, expected)


def test_custom_null_space_rejects_non_positive_n() -> None:
    with pytest.raises(IGLConfigError, match="n_columns"):
        CustomNullSpace(lambda z: z, n_columns=0)


def test_custom_null_space_validates_output_shape() -> None:
    null = CustomNullSpace(lambda z: z[:, :1], n_columns=3)
    with pytest.raises(IGLConfigError, match="must return"):
        null.evaluate(torch.randn(2, 4))


def test_build_null_space_dispatches_correctly() -> None:
    assert build_null_space(NullSpaceKind.NONE, latent_dim=3) is None
    assert isinstance(build_null_space("constant", latent_dim=3), ConstantNullSpace)
    poly = build_null_space("polynomial", latent_dim=3, degree=2)
    assert isinstance(poly, PolynomialNullSpace)
    assert poly.n_columns == 1 + 3 * 2
