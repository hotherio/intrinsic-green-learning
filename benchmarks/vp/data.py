"""Testbeds for the VP training experiments."""

from math import sqrt

import torch

from igl.data import embed_in_high_dim, make_moons, make_swiss_roll


def moons_classification(n: int = 2000, *, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Two moons embedded in 10 dimensions; integer labels."""
    x_2d, y = make_moons(n, noise=0.08, seed=seed)
    return embed_in_high_dim(x_2d, target_dim=10, seed=seed + 81), y


def swiss_roll_regression(n: int = 4000, *, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Swiss roll embedded in 12 dimensions; reconstruction target = the states themselves."""
    x_3d, _ = make_swiss_roll(n, seed=seed)
    x = embed_in_high_dim(x_3d, target_dim=12, seed=seed + 81)
    return x, x


def mlp_manifold(
    n: int = 20_000, *, latent_dim: int = 2, ambient_dim: int = 64, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """A random-MLP image of a low-dimensional latent; reconstruction target = states.

    The E4 testbed shape: latent ``U[-1, 1]^d`` pushed through a fixed random
    two-layer tanh MLP into ``ambient_dim`` dimensions (the probe series'
    mlp-manifold construction at benchmark scale).
    """
    generator = torch.Generator().manual_seed(seed)
    z = 2.0 * torch.rand(n, latent_dim, generator=generator) - 1.0
    hidden = 128
    w1 = torch.randn(latent_dim, hidden, generator=generator) / sqrt(latent_dim)
    w2 = torch.randn(hidden, ambient_dim, generator=generator) / sqrt(hidden)
    x = torch.tanh(torch.tanh(z @ w1) @ w2 * 2.0)
    x = x + 0.01 * torch.randn(n, ambient_dim, generator=generator)
    return x, x


def split(
    x: torch.Tensor, y: torch.Tensor, *, fraction: float = 0.2, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic train/val split: (x_train, y_train, x_val, y_val)."""
    n = x.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = int(n * fraction)
    val, train = perm[:n_val], perm[n_val:]
    return x[train], y[train], x[val], y[val]
