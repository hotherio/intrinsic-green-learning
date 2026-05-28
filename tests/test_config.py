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


def test_encoder_config_coerces_string_norm_and_activation() -> None:
    from igl import ActivationType, EncoderKind, NormType  # noqa: PLC0415

    cfg = EncoderConfig(kind="mlp", norm="batch", activation="relu")
    assert cfg.kind is EncoderKind.MLP
    assert cfg.norm is NormType.BATCH
    assert cfg.activation is ActivationType.RELU


def test_encoder_config_accepts_tuple_hidden() -> None:
    cfg = EncoderConfig(hidden=(256, 128, 64))
    assert cfg.hidden == (256, 128, 64)


def test_encoder_config_coerces_list_hidden_to_tuple() -> None:
    # __post_init__ coerces sequences to tuples for hashability.
    cfg = EncoderConfig(hidden=[128, 64])  # type: ignore[arg-type]
    assert isinstance(cfg.hidden, tuple)
    assert cfg.hidden == (128, 64)


def test_encoder_config_is_hashable_with_tuple_hidden() -> None:
    cfg = EncoderConfig(hidden=(128, 64))
    # Frozen + hashable fields → usable as a dict key.
    d = {cfg: "value"}
    assert d[EncoderConfig(hidden=(128, 64))] == "value"


def test_kernel_config_coerces_string_operator() -> None:
    from igl import NormalizeMode, OperatorName  # noqa: PLC0415

    cfg = KernelConfig(operator="laplacian", normalize="l2")
    assert cfg.operator is OperatorName.LAPLACIAN
    assert cfg.normalize is NormalizeMode.L2


def test_kernel_config_coerces_sequence_operator() -> None:
    from igl import OperatorName  # noqa: PLC0415

    cfg = KernelConfig(operator=("gaussian", "helmholtz"))
    assert cfg.operator == (OperatorName.GAUSSIAN, OperatorName.HELMHOLTZ)


def test_kernel_config_coerces_list_operator() -> None:
    from igl import OperatorName  # noqa: PLC0415

    cfg = KernelConfig(operator=["gaussian", "cauchy"])  # type: ignore[arg-type]
    assert cfg.operator == (OperatorName.GAUSSIAN, OperatorName.CAUCHY)


def test_matryoshka_config_coerces_scheduler_string() -> None:
    from igl import SamplingMode, SchedulerType  # noqa: PLC0415

    cfg = MatryoshkaConfig(sampling="power_law", scheduler="none")
    assert cfg.sampling is SamplingMode.POWER_LAW
    assert cfg.scheduler is SchedulerType.NONE


def test_iglconfig_round_trip_includes_tuple_hidden() -> None:
    original = IGLConfig(
        max_dim=24,
        encoder=EncoderConfig(hidden=(192, 96), depth=2),
        kernel=KernelConfig(n_anchors=80, n_scales=5, operator=("gaussian", "laplacian")),
        matryoshka=MatryoshkaConfig(epochs=100, batch_size=64),
    )
    data = original.to_dict()
    rebuilt = IGLConfig.from_dict(data)
    assert rebuilt == original
    assert isinstance(rebuilt.encoder.hidden, tuple)


def test_iglconfig_from_dict_rejects_non_mapping_encoder() -> None:
    import pytest  # noqa: PLC0415

    from igl import IGLConfigError  # noqa: PLC0415

    with pytest.raises(IGLConfigError, match="encoder must be a mapping"):
        IGLConfig.from_dict({"encoder": "not a mapping"})


def test_iglconfig_from_dict_rejects_non_mapping_kernel() -> None:
    import pytest  # noqa: PLC0415

    from igl import IGLConfigError  # noqa: PLC0415

    with pytest.raises(IGLConfigError, match="kernel must be a mapping"):
        IGLConfig.from_dict({"kernel": "bad"})


def test_iglconfig_from_dict_rejects_non_mapping_matryoshka() -> None:
    import pytest  # noqa: PLC0415

    from igl import IGLConfigError  # noqa: PLC0415

    with pytest.raises(IGLConfigError, match="matryoshka must be a mapping"):
        IGLConfig.from_dict({"matryoshka": "bad"})


def test_iglconfig_from_dict_rejects_non_int_max_dim() -> None:
    import pytest  # noqa: PLC0415

    from igl import IGLConfigError  # noqa: PLC0415

    with pytest.raises(IGLConfigError, match="max_dim"):
        IGLConfig.from_dict({"max_dim": "bad"})


def test_iglconfig_from_dict_rejects_bad_hidden_type() -> None:
    import pytest  # noqa: PLC0415

    from igl import IGLConfigError  # noqa: PLC0415

    with pytest.raises(IGLConfigError, match="encoder.hidden"):
        IGLConfig.from_dict({"encoder": {"hidden": "256"}})


def test_iglconfig_from_dict_rejects_bad_operator() -> None:
    import pytest  # noqa: PLC0415

    from igl import IGLConfigError  # noqa: PLC0415

    with pytest.raises(IGLConfigError, match="kernel.operator"):
        IGLConfig.from_dict({"kernel": {"operator": 42}})


def test_iglconfig_from_dict_rejects_bad_sigma_log_range() -> None:
    import pytest  # noqa: PLC0415

    from igl import IGLConfigError  # noqa: PLC0415

    with pytest.raises(IGLConfigError, match="sigma_log_range"):
        IGLConfig.from_dict({"kernel": {"sigma_log_range": "bad"}})


def test_iglconfig_from_dict_round_trips_list_operator() -> None:
    from igl import OperatorName  # noqa: PLC0415

    data: dict[str, object] = {"kernel": {"operator": ["gaussian", "cauchy"]}}
    cfg = IGLConfig.from_dict(data)
    assert cfg.kernel.operator == (OperatorName.GAUSSIAN, OperatorName.CAUCHY)


def test_iglconfig_from_dict_accepts_tuple_operator() -> None:
    from igl import OperatorName  # noqa: PLC0415

    data: dict[str, object] = {"kernel": {"operator": ("gaussian", "helmholtz")}}
    cfg = IGLConfig.from_dict(data)
    assert cfg.kernel.operator == (OperatorName.GAUSSIAN, OperatorName.HELMHOLTZ)


def test_iglconfig_to_dict_serialises_int_hidden() -> None:
    cfg = IGLConfig(encoder=EncoderConfig(hidden=256))
    assert cfg.to_dict()["encoder"]["hidden"] == 256  # type: ignore[call-overload, index]


def test_iglconfig_to_dict_serialises_single_operator_as_string() -> None:
    cfg = IGLConfig(kernel=KernelConfig(operator="gaussian"))
    assert cfg.to_dict()["kernel"]["operator"] == "gaussian"  # type: ignore[call-overload, index]
