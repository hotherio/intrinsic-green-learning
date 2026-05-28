"""Tests for :mod:`igl.core.encoder`."""

import pytest
import torch

from igl import IGLConfigError, LinearEncoder, MLPEncoder


def test_linear_encoder_output_shape() -> None:
    enc = LinearEncoder(input_dim=8, max_dim=4)
    out = enc(torch.randn(5, 8))
    assert out.shape == (5, 4)
    assert enc.input_dim == 8
    assert enc.max_dim == 4


def test_linear_encoder_rejects_non_positive_dims() -> None:
    with pytest.raises(IGLConfigError, match="input_dim"):
        LinearEncoder(input_dim=0, max_dim=4)
    with pytest.raises(IGLConfigError, match="max_dim"):
        LinearEncoder(input_dim=4, max_dim=0)


def test_mlp_encoder_default_output_shape() -> None:
    enc = MLPEncoder(input_dim=8, max_dim=4)
    out = enc(torch.randn(5, 8))
    assert out.shape == (5, 4)
    assert enc.input_dim == 8
    assert enc.max_dim == 4


@pytest.mark.parametrize("norm", ["layer", "batch", "none"])
@pytest.mark.parametrize("activation", ["silu", "tanh", "relu", "gelu"])
def test_mlp_encoder_norm_and_activation_options(norm: str, activation: str) -> None:
    enc = MLPEncoder(
        input_dim=4,
        max_dim=2,
        hidden=8,
        depth=2,
        norm=norm,  # type: ignore[arg-type]
        activation=activation,  # type: ignore[arg-type]
    )
    out = enc(torch.randn(6, 4))
    assert out.shape == (6, 2)


@pytest.mark.parametrize(("field", "value"), [("input_dim", 0), ("max_dim", 0), ("depth", 0), ("hidden", 0)])
def test_mlp_encoder_rejects_non_positive(field: str, value: int) -> None:
    kwargs = {"input_dim": 4, "max_dim": 2, "hidden": 8, "depth": 2}
    kwargs[field] = value
    with pytest.raises(IGLConfigError, match=field):
        MLPEncoder(**kwargs)  # type: ignore[arg-type]


def test_mlp_encoder_deeper_models_have_more_parameters() -> None:
    shallow = MLPEncoder(input_dim=8, max_dim=4, hidden=16, depth=1)
    deep = MLPEncoder(input_dim=8, max_dim=4, hidden=16, depth=3)
    n_shallow = sum(p.numel() for p in shallow.parameters())
    n_deep = sum(p.numel() for p in deep.parameters())
    assert n_deep > n_shallow
