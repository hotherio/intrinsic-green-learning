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


def test_mlp_encoder_per_layer_widths_sequence() -> None:
    from torch import nn  # noqa: PLC0415

    enc = MLPEncoder(input_dim=10, max_dim=2, hidden=(64, 32, 16))
    assert enc.hidden_widths == (64, 32, 16)
    linear_widths = [layer.out_features for layer in enc.net if isinstance(layer, nn.Linear)]
    assert linear_widths == [64, 32, 16, 2]


def test_mlp_encoder_single_element_sequence() -> None:
    enc = MLPEncoder(input_dim=4, max_dim=2, hidden=(128,))
    assert enc.hidden_widths == (128,)


def test_mlp_encoder_explicit_depth_with_matching_sequence_ok() -> None:
    enc = MLPEncoder(input_dim=4, max_dim=2, hidden=(64, 32), depth=2)
    assert enc.hidden_widths == (64, 32)


def test_mlp_encoder_rejects_contradictory_depth_and_sequence() -> None:
    import pytest  # noqa: PLC0415

    with pytest.raises(IGLConfigError, match="does not match hidden sequence length"):
        MLPEncoder(input_dim=4, max_dim=2, hidden=(64, 32), depth=3)


def test_mlp_encoder_rejects_empty_sequence() -> None:
    import pytest  # noqa: PLC0415

    with pytest.raises(IGLConfigError, match="must be non-empty"):
        MLPEncoder(input_dim=4, max_dim=2, hidden=())


def test_mlp_encoder_rejects_non_positive_width_in_sequence() -> None:
    import pytest  # noqa: PLC0415

    with pytest.raises(IGLConfigError, match=r"hidden\[1\] must be >= 1"):
        MLPEncoder(input_dim=4, max_dim=2, hidden=(64, 0))


def test_mlp_encoder_rejects_non_positive_depth_with_int_hidden() -> None:
    import pytest  # noqa: PLC0415

    with pytest.raises(IGLConfigError, match="depth must be >= 1"):
        MLPEncoder(input_dim=4, max_dim=2, hidden=16, depth=0)


def test_mlp_encoder_accepts_string_norm_and_activation() -> None:
    enc = MLPEncoder(input_dim=4, max_dim=2, hidden=8, norm="batch", activation="relu")
    assert enc.hidden_widths == (8, 8)


def test_build_mlp_encoder_from_encoder_config() -> None:
    from igl import EncoderConfig  # noqa: PLC0415
    from igl.core.encoder import build_mlp_encoder  # noqa: PLC0415

    cfg = EncoderConfig(hidden=(48, 24), depth=2, norm="layer", activation="gelu")
    enc = build_mlp_encoder(input_dim=6, max_dim=3, config=cfg)
    assert enc.hidden_widths == (48, 24)


def test_build_mlp_encoder_rejects_non_mlp_kind() -> None:
    import pytest  # noqa: PLC0415

    from igl import EncoderConfig, EncoderKind  # noqa: PLC0415
    from igl.core.encoder import build_mlp_encoder  # noqa: PLC0415

    cfg = EncoderConfig(kind=EncoderKind.LINEAR, hidden=8)
    with pytest.raises(IGLConfigError, match="kind=MLP"):
        build_mlp_encoder(input_dim=6, max_dim=3, config=cfg)
