"""Tests for the public Protocols and type aliases."""

import torch

from igl.types import (
    DimensionCurve,
    EncoderProtocol,
    LossStrategy,
    MatryoshkaSampler,
    OperatorFn,
)


class _StubEncoder:
    input_dim = 4
    max_dim = 2

    def __call__(self, x: torch.Tensor, /) -> torch.Tensor:
        return x[:, : self.max_dim]


class _StubOperator:
    is_oscillatory = False

    def __call__(self, d: torch.Tensor, sigma: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor]:
        log_abs = -(d / sigma).pow(2)
        sign = torch.ones_like(log_abs)
        return log_abs, sign


class _StubLoss:
    higher_is_better = False

    def target(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(pred, target)

    def metric(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        return self.loss(pred, target).item()


class _StubSampler:
    def __call__(self, d_max: int, /) -> int:
        return d_max


def test_encoder_protocol_is_satisfied_by_a_callable_with_dims() -> None:
    enc: EncoderProtocol = _StubEncoder()
    out = enc(torch.zeros(3, 4))
    assert out.shape == (3, 2)


def test_operator_fn_protocol_returns_log_abs_and_sign() -> None:
    op: OperatorFn = _StubOperator()
    d = torch.tensor([[1.0, 2.0]])
    sigma = torch.tensor([[1.0, 1.0]])
    log_abs, sign = op(d, sigma)
    assert log_abs.shape == d.shape
    assert sign.shape == d.shape
    assert not op.is_oscillatory


def test_loss_strategy_protocol_has_minimisation_metric() -> None:
    loss: LossStrategy = _StubLoss()
    pred = torch.tensor([1.0, 2.0])
    target = loss.target(torch.tensor([1.0, 2.5]))
    value = loss.metric(pred, target)
    assert value > 0
    assert loss.higher_is_better is False


def test_matryoshka_sampler_protocol_returns_an_int() -> None:
    sampler: MatryoshkaSampler = _StubSampler()
    d_max = 8
    k = sampler(d_max)
    assert isinstance(k, int)
    assert k == d_max


def test_dimension_curve_alias_accepts_mapping() -> None:
    expected = {1: 0.5, 2: 0.4, 3: 0.42}
    curve: DimensionCurve = expected
    assert list(curve.keys()) == [1, 2, 3]
    assert curve[2] == expected[2]
