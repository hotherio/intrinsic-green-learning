"""Tests for the dataclass configs."""

import pytest

from igl import EncoderConfig, IGLConfig, KernelConfig, MatryoshkaConfig


def test_default_iglconfig_is_constructable() -> None:
    config = IGLConfig()
    assert config.max_dim == 16
    assert isinstance(config.encoder, EncoderConfig)
    assert isinstance(config.kernel, KernelConfig)
    assert isinstance(config.matryoshka, MatryoshkaConfig)


def test_iglconfig_round_trip_through_to_dict_and_from_dict() -> None:
    original = IGLConfig(
        max_dim=24,
        encoder=EncoderConfig(hidden=128, depth=3),
        kernel=KernelConfig(n_anchors=80, n_scales=5),
        matryoshka=MatryoshkaConfig(epochs=100, batch_size=64),
    )
    data = original.to_dict()
    rebuilt = IGLConfig.from_dict(data)
    assert rebuilt == original


def test_configs_are_frozen() -> None:
    config = IGLConfig()
    with pytest.raises(AttributeError):
        config.max_dim = 32  # type: ignore[misc]


def test_configs_are_hashable() -> None:
    config = IGLConfig()
    # Frozen dataclasses with hashable fields should hash.
    assert hash(config) == hash(IGLConfig())


def test_matryoshka_config_defaults_match_reference() -> None:
    config = MatryoshkaConfig()
    assert config.encoder_lr == 1e-3
    assert config.source_l2 == 1e-3
    assert config.grad_clip == 1.0
    assert config.sampling == "uniform"
