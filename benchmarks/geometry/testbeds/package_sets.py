"""The package's own synthetic sets and the ambient-embedding step shared by every learned-chart block.

``to_inputs`` turns the points of any bed into IGL inputs: a random orthogonal embedding into
``D`` dimensions (``igl.data.embed_in_high_dim``, which keeps the induced metric exact) plus
isotropic Gaussian noise. ``mlp_manifold`` is the generic curved manifold of the ``bench/vp-suite``
branch (a random two-layer tanh MLP image of a latent cube), the one bed with no analytic metric.
"""

from __future__ import annotations

from math import sqrt

import torch

from igl.data.synthetic import embed_in_high_dim, make_swiss_roll


def to_inputs(x: torch.Tensor, *, ambient: int | None, noise: float = 0.005, seed: int = 0) -> torch.Tensor:
    """Embed ``x [n, D0]`` into ``ambient`` dimensions by a random rotation after zero-padding, then add noise."""
    x = x.to(torch.float32)
    if ambient is not None and ambient > x.shape[1]:
        x = embed_in_high_dim(x, target_dim=ambient, seed=seed)
    elif ambient is not None and ambient < x.shape[1]:
        raise ValueError(f"ambient {ambient} is smaller than the bed's native dimension {x.shape[1]}")
    if noise > 0:
        gen = torch.Generator().manual_seed(seed + 1)
        x = x + noise * torch.randn(x.shape, generator=gen)
    return x


def mlp_manifold(n: int, *, latent_dim: int = 2, ambient_dim: int = 64, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """A random-MLP image of ``U[-1, 1]^d``: ``(x, z)`` with ``x`` in ``R^ambient_dim`` and ``z`` the latent."""
    generator = torch.Generator().manual_seed(seed)
    z = 2.0 * torch.rand(n, latent_dim, generator=generator) - 1.0
    hidden = 128
    w1 = torch.randn(latent_dim, hidden, generator=generator) / sqrt(latent_dim)
    w2 = torch.randn(hidden, ambient_dim, generator=generator) / sqrt(hidden)
    x = torch.tanh(torch.tanh(z @ w1) @ w2 * 2.0)
    return x, z


def swiss_roll(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The package's Swiss roll: ``(x [n, 3], params [n, 2])`` with the unrolling parameters ``(t, h)``, a flat 2-manifold."""
    x, (t, h) = make_swiss_roll(n)
    return x, torch.stack([t, h], dim=1)


__all__ = ["mlp_manifold", "swiss_roll", "to_inputs"]
