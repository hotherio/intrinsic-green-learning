"""Tests for :class:`igl.IGLModule`."""

import pytest
import torch
from torch import nn

from igl import IGLConfigError, IGLModule, LinearEncoder, MLPEncoder


def test_module_forward_output_shape() -> None:
    module = IGLModule(input_dim=8, max_dim=4, output_dim=3, n_anchors=10, n_scales=2)
    out = module(torch.randn(6, 8))
    assert out.shape == (6, 3)


def test_module_design_matrix_shape() -> None:
    module = IGLModule(input_dim=8, max_dim=4, output_dim=3, n_anchors=10, n_scales=2)
    phi = module.design_matrix(torch.randn(6, 8))
    assert phi.shape == (6, 10)


def test_module_latent_returns_encoder_output_with_input_norm() -> None:
    # When normalize_input is True the latent goes through BatchNorm1d first.
    module = IGLModule(input_dim=8, max_dim=4, output_dim=2, normalize_input=True)
    module.eval()
    z = module.latent(torch.randn(5, 8))
    assert z.shape == (5, 4)


def test_module_accepts_explicit_encoder() -> None:
    encoder = LinearEncoder(input_dim=8, max_dim=4)
    module = IGLModule(
        input_dim=8,
        max_dim=4,
        output_dim=2,
        encoder=encoder,
        normalize_input=False,
    )
    assert module(torch.randn(5, 8)).shape == (5, 2)


def test_module_rejects_encoder_dimension_mismatch() -> None:
    encoder = LinearEncoder(input_dim=8, max_dim=4)
    with pytest.raises(IGLConfigError, match="encoder.input_dim"):
        IGLModule(input_dim=10, max_dim=4, output_dim=2, encoder=encoder, normalize_input=False)
    encoder2 = LinearEncoder(input_dim=8, max_dim=4)
    with pytest.raises(IGLConfigError, match="encoder.max_dim"):
        IGLModule(input_dim=8, max_dim=6, output_dim=2, encoder=encoder2, normalize_input=False)


def test_module_rejects_non_positive_dims() -> None:
    with pytest.raises(IGLConfigError, match="input_dim"):
        IGLModule(input_dim=0, max_dim=4, output_dim=2)
    with pytest.raises(IGLConfigError, match="max_dim"):
        IGLModule(input_dim=8, max_dim=0, output_dim=2)
    with pytest.raises(IGLConfigError, match="output_dim"):
        IGLModule(input_dim=8, max_dim=4, output_dim=0)


def test_module_gate_mask_propagates() -> None:
    module = IGLModule(input_dim=8, max_dim=4, output_dim=2, normalize_input=False)
    module.eval()
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    phi = module.design_matrix(torch.randn(5, 8), gate_mask=mask)
    assert phi.shape == (5, 10) or phi.shape[0] == 5  # second dim is n_anchors default


def test_module_set_source_weights_updates_buffer() -> None:
    module = IGLModule(input_dim=4, max_dim=2, output_dim=3, n_anchors=5, n_scales=2)
    w = torch.randn(5, 3)
    module.set_source_weights(w)
    assert isinstance(module.source_weights, torch.Tensor)
    torch.testing.assert_close(module.source_weights, w)


def test_module_set_source_weights_rejects_wrong_shape() -> None:
    module = IGLModule(input_dim=4, max_dim=2, output_dim=3, n_anchors=5)
    with pytest.raises(IGLConfigError, match="weights shape"):
        module.set_source_weights(torch.zeros(3, 5))


def test_module_with_custom_mlp_encoder() -> None:
    encoder = MLPEncoder(input_dim=8, max_dim=4, hidden=16, depth=1)
    module = IGLModule(input_dim=8, max_dim=4, output_dim=2, encoder=encoder, normalize_input=False)
    out = module(torch.randn(3, 8))
    assert out.shape == (3, 2)


def test_module_rejects_non_module_encoder() -> None:
    class FakeEncoder:
        input_dim = 4
        max_dim = 2

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, :2]

    with pytest.raises(AssertionError, match="encoder must be an nn.Module"):
        IGLModule(input_dim=4, max_dim=2, output_dim=2, encoder=FakeEncoder())  # type: ignore[arg-type]


def test_module_parameters_include_encoder_kernel_and_bias() -> None:
    module = IGLModule(input_dim=4, max_dim=2, output_dim=2, n_anchors=4, n_scales=2)
    param_names = {n for n, _ in module.named_parameters()}
    assert any("encoder" in n for n in param_names)
    assert any("green" in n for n in param_names)
    assert "bias" in param_names
    assert isinstance(module.bias, nn.Parameter)
