"""Tests for the Matryoshka samplers."""

import pytest
import torch

from igl import IGLConfigError, PowerLawSampler, UniformSampler


def test_uniform_sampler_support_set() -> None:
    sampler = UniformSampler()
    d_max = 5
    samples = {sampler(d_max) for _ in range(200)}
    assert samples == set(range(1, d_max + 1))


def test_uniform_sampler_is_roughly_uniform() -> None:
    torch.manual_seed(0)
    sampler = UniformSampler()
    d_max = 4
    counts = [0] * (d_max + 1)
    for _ in range(4000):
        counts[sampler(d_max)] += 1
    expected = 1000
    for k in range(1, d_max + 1):
        # Loose tolerance — empirical, not a property test.
        assert abs(counts[k] - expected) < 200, f"k={k} count={counts[k]}"


def test_uniform_sampler_rejects_invalid_d_max() -> None:
    sampler = UniformSampler()
    with pytest.raises(IGLConfigError, match="d_max"):
        sampler(0)


def test_power_law_sampler_biases_toward_small_k() -> None:
    torch.manual_seed(0)
    sampler = PowerLawSampler(alpha=2.0)
    d_max = 6
    counts = [0] * (d_max + 1)
    for _ in range(2000):
        counts[sampler(d_max)] += 1
    # k=1 should be hit more than k=d_max.
    assert counts[1] > counts[d_max] * 3


def test_power_law_sampler_rejects_invalid_alpha() -> None:
    with pytest.raises(IGLConfigError, match="alpha"):
        PowerLawSampler(alpha=0.0)
    with pytest.raises(IGLConfigError, match="alpha"):
        PowerLawSampler(alpha=-1.0)


def test_power_law_sampler_rejects_invalid_d_max() -> None:
    sampler = PowerLawSampler(alpha=1.0)
    with pytest.raises(IGLConfigError, match="d_max"):
        sampler(0)


def test_power_law_sampler_support_set() -> None:
    sampler = PowerLawSampler(alpha=1.0)
    d_max = 4
    samples = {sampler(d_max) for _ in range(500)}
    assert samples == set(range(1, d_max + 1))
