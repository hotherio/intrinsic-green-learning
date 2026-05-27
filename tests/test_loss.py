"""Tests for the built-in :class:`igl.types.LossStrategy` implementations."""

import pytest
import torch

from igl import CrossEntropyLoss, IGLConfigError, MSELoss


def test_cross_entropy_target_is_one_hot() -> None:
    loss = CrossEntropyLoss(n_classes=3)
    y = torch.tensor([0, 2, 1])
    one_hot = loss.target(y)
    expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    torch.testing.assert_close(one_hot, expected)


def test_cross_entropy_metric_is_accuracy() -> None:
    loss = CrossEntropyLoss(n_classes=2)
    target = loss.target(torch.tensor([0, 1, 1, 0]))
    pred = torch.tensor([[10.0, 0.0], [0.0, 10.0], [0.0, 10.0], [10.0, 0.0]])  # all correct
    assert loss.metric(pred, target) == pytest.approx(1.0)


def test_cross_entropy_metric_decreases_with_errors() -> None:
    loss = CrossEntropyLoss(n_classes=2)
    target = loss.target(torch.tensor([0, 1, 1, 0]))
    pred = torch.tensor([[10.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 0.0]])  # 1 wrong
    assert loss.metric(pred, target) == pytest.approx(0.75)


def test_cross_entropy_higher_is_better() -> None:
    assert CrossEntropyLoss(n_classes=2).higher_is_better is True


def test_cross_entropy_rejects_one_class() -> None:
    with pytest.raises(IGLConfigError, match="n_classes"):
        CrossEntropyLoss(n_classes=1)


def test_mse_target_promotes_to_2d() -> None:
    loss = MSELoss()
    y = torch.tensor([1.0, 2.0, 3.0])
    target = loss.target(y)
    assert target.shape == (3, 1)


def test_mse_target_preserves_2d() -> None:
    loss = MSELoss()
    y = torch.randn(5, 3)
    target = loss.target(y)
    assert target.shape == (5, 3)


def test_mse_metric_is_loss_value() -> None:
    loss = MSELoss()
    pred = torch.tensor([[1.0], [2.0], [3.0]])
    target = torch.tensor([[1.5], [2.5], [3.5]])
    assert loss.metric(pred, target) == pytest.approx(0.25, abs=1e-5)


def test_mse_lower_is_better() -> None:
    assert MSELoss().higher_is_better is False
