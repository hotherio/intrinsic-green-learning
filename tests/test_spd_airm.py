"""Tests for ``igl.spd.AIRMLoss`` and ``airm_loss``."""

import pytest
import torch

from igl import IGLConfigError
from igl.spd import AIRMLoss, airm_loss


def _random_spd(batch: int, d: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    a = torch.randn(batch, d, d)
    return a @ a.transpose(-1, -2) + d * torch.eye(d)


def test_airm_loss_zero_when_same_matrix() -> None:
    c = _random_spd(3, 4)
    out = airm_loss(c, c)
    assert out.item() == pytest.approx(0.0, abs=1e-4)


def test_airm_loss_positive_for_different_matrices() -> None:
    c = _random_spd(2, 4, seed=1)
    c_hat = _random_spd(2, 4, seed=2)
    assert airm_loss(c, c_hat).item() > 0.0


def test_airm_loss_reduction_modes() -> None:
    c = _random_spd(4, 3, seed=0)
    c_hat = _random_spd(4, 3, seed=5)
    per_sample = airm_loss(c, c_hat, reduction="none")
    assert per_sample.shape == (4,)
    mean = airm_loss(c, c_hat, reduction="mean")
    total = airm_loss(c, c_hat, reduction="sum")
    torch.testing.assert_close(mean, per_sample.mean(), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(total, per_sample.sum(), rtol=1e-4, atol=1e-5)


def test_airm_loss_rejects_unknown_reduction() -> None:
    c = _random_spd(2, 3)
    with pytest.raises(IGLConfigError, match="unknown reduction"):
        airm_loss(c, c, reduction="bogus")


def test_airm_loss_is_affine_invariant() -> None:
    """``AIRM(A C A^T, A Ĉ A^T) = AIRM(C, Ĉ)`` for invertible ``A``."""
    torch.manual_seed(0)
    c = _random_spd(3, 4, seed=1)
    c_hat = _random_spd(3, 4, seed=2)
    a = torch.randn(4, 4) + 4 * torch.eye(4)  # well-conditioned
    c_t = a @ c @ a.T
    c_hat_t = a @ c_hat @ a.T
    torch.testing.assert_close(airm_loss(c, c_hat), airm_loss(c_t, c_hat_t), rtol=1e-3, atol=1e-3)


def test_airm_strategy_implements_loss_strategy_methods() -> None:
    """:class:`AIRMLoss` must satisfy the protocol — target/loss/metric/curve_score."""
    loss = AIRMLoss(latent_dim=4)
    assert loss.higher_is_better is False
    d = 4
    vec_dim = d * (d + 1) // 2
    y = torch.randn(8, vec_dim) * 0.1
    target = loss.target(y)
    assert target.shape == y.shape
    pred = y + 0.01 * torch.randn_like(y)
    out_loss = loss.loss(pred, target)
    assert out_loss.dim() == 0  # scalar
    assert isinstance(loss.metric(pred, target), float)
    assert isinstance(loss.curve_score(pred, target), float)


def test_airm_strategy_rejects_zero_latent_dim() -> None:
    with pytest.raises(IGLConfigError, match="latent_dim"):
        AIRMLoss(latent_dim=0)


def test_airm_strategy_promotes_1d_targets() -> None:
    """A 1-D target is unsqueezed to a 2-D vector."""
    loss = AIRMLoss(latent_dim=1)
    y = torch.tensor([0.5, 0.3, 0.1])
    target = loss.target(y)
    assert target.shape == (3, 1)
