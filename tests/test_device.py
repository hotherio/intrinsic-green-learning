"""Tests for :func:`igl.get_device` and the estimator ``device`` parameter."""

import torch
from sklearn.base import clone

from igl import IGLRegressor, get_device
from igl.data import embed_in_high_dim, make_moons


def test_get_device_returns_torch_device() -> None:
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"mps", "cuda", "cpu"}


def test_estimator_device_default_is_cpu() -> None:
    x_2d, y = make_moons(80, noise=0.1, seed=42)
    x = embed_in_high_dim(x_2d, target_dim=8, seed=123)
    est = IGLRegressor(max_dim=4, random_state=0)
    est.fit(x.numpy(), y.numpy().astype(float))
    assert next(est.module_.parameters()).device.type == "cpu"


def test_estimator_honors_device_param() -> None:
    device = get_device()
    x_2d, y = make_moons(80, noise=0.1, seed=42)
    x = embed_in_high_dim(x_2d, target_dim=8, seed=123)
    est = IGLRegressor(max_dim=4, random_state=0, device=device)
    est.fit(x.numpy(), y.numpy().astype(float))
    assert next(est.module_.parameters()).device.type == device.type
    preds = est.predict(x.numpy())
    assert preds.shape == (80,)


def test_estimator_device_survives_get_params_and_clone() -> None:
    est = IGLRegressor(max_dim=4, device="cpu")
    assert est.get_params()["device"] == "cpu"
    cloned = clone(est)
    assert cloned.device == "cpu"
