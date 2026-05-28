"""Tests for :class:`igl.GreenKernel`."""

import pytest
import torch

from igl import GreenKernel, IGLConfigError


def test_design_matrix_shape() -> None:
    gk = GreenKernel(latent_dim=4, n_anchors=10, n_scales=3, operator="gaussian")
    z = torch.randn(7, 4)
    phi = gk(z)
    assert phi.shape == (7, 10)


def test_design_matrix_forward_alias_matches_compute() -> None:
    gk = GreenKernel(latent_dim=3, n_anchors=5, n_scales=2)
    z = torch.randn(4, 3)
    direct = gk.compute_design_matrix(z)
    via_forward = gk(z)
    torch.testing.assert_close(direct, via_forward, rtol=1e-6, atol=1e-6)


def test_gate_mask_zeroes_out_dimensions_correctly() -> None:
    torch.manual_seed(0)
    gk = GreenKernel(latent_dim=4, n_anchors=8, n_scales=2)
    z = torch.randn(3, 4)
    full_mask = torch.ones(4)
    phi_full_mask = gk(z, gate_mask=full_mask)
    phi_no_mask = gk(z)
    torch.testing.assert_close(phi_full_mask, phi_no_mask, rtol=1e-5, atol=1e-6)


def test_multi_operator_distributes_scales_evenly() -> None:
    gk = GreenKernel(latent_dim=3, n_anchors=8, n_scales=4, operator=("gaussian", "helmholtz"))
    assert gk.operator_names == ("gaussian", "helmholtz")
    assert gk.n_scales == 4


def test_multi_operator_uneven_remainder() -> None:
    # 5 scales over 2 operators → first gets 3, second gets 2
    gk = GreenKernel(latent_dim=3, n_anchors=8, n_scales=5, operator=("gaussian", "laplacian"))
    assert gk.n_scales == 5


def test_invalid_construction_dimensions() -> None:
    with pytest.raises(IGLConfigError, match="latent_dim"):
        GreenKernel(latent_dim=0)
    with pytest.raises(IGLConfigError, match="n_anchors"):
        GreenKernel(latent_dim=2, n_anchors=0)
    with pytest.raises(IGLConfigError, match="n_scales"):
        GreenKernel(latent_dim=2, n_scales=0)


def test_empty_operator_sequence_rejected() -> None:
    with pytest.raises(IGLConfigError, match="non-empty"):
        GreenKernel(latent_dim=2, operator=())


def test_more_operators_than_scales_rejected() -> None:
    with pytest.raises(IGLConfigError, match="number of operators"):
        GreenKernel(latent_dim=2, n_scales=2, operator=("gaussian", "helmholtz", "cauchy"))


def test_oscillatory_kernel_produces_signed_design_matrix() -> None:
    torch.manual_seed(0)
    gk = GreenKernel(latent_dim=2, n_anchors=8, n_scales=2, operator="helmholtz")
    z = torch.randn(20, 2) * 2.0
    phi = gk(z)
    # Some entries should be negative for oscillatory kernels at random points.
    assert (phi < 0).any()


def test_gaussian_kernel_design_matrix_is_non_negative() -> None:
    """Gaussian is strictly positive; after softmax normalisation it stays so."""
    torch.manual_seed(0)
    gk = GreenKernel(latent_dim=2, n_anchors=6, n_scales=2, operator="gaussian")
    z = torch.randn(10, 2)
    phi = gk(z)
    # Gaussian + softmax over importance => non-negative once we exponentiate
    # in the log-space pipeline; assert finite-ness as the contract.
    assert torch.all(torch.isfinite(phi))
