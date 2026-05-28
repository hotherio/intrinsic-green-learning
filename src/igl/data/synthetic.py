"""Synthetic manifolds for end-to-end IGL experiments.

Three low-intrinsic-dimensional generators (flat torus, swiss roll, two moons)
plus a high-dimensional embedding helper that pads and rotates a low-D dataset
into ``D``-dimensional space via a random orthogonal matrix.
"""

import math
from typing import cast

import torch

from igl.exceptions import IGLConfigError
from igl.spd.linalg import matrix_exp_sym

_MIN_MOON_SAMPLES = 2


def _maybe_seed(seed: int | None) -> torch.Generator:
    """Return a freshly-seeded generator (avoids touching the global RNG)."""
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.seed()
    return generator


def make_flat_torus(
    n_samples: int,
    *,
    noise: float = 0.0,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat torus ``T²`` embedded in ``R⁴`` via ``(cos θ₁, sin θ₁, cos θ₂, sin θ₂)``.

    Intrinsic dimension: 2 (the two angles).

    Args:
        n_samples: Number of points to sample.
        noise: Additive Gaussian noise std in the ambient ``R⁴``. Default ``0``.
        seed: Optional RNG seed (does not affect the global RNG state).

    Returns:
        ``(X, theta)`` where ``X`` is ``[n_samples, 4]`` ambient coordinates
        and ``theta`` is ``[n_samples, 2]`` intrinsic angles in ``[0, 2π)``.
    """
    if n_samples < 1:
        raise IGLConfigError(f"n_samples must be >= 1, got {n_samples}")
    gen = _maybe_seed(seed)
    theta = torch.rand(n_samples, 2, generator=gen) * (2 * math.pi)
    x = torch.stack(
        [
            torch.cos(theta[:, 0]),
            torch.sin(theta[:, 0]),
            torch.cos(theta[:, 1]),
            torch.sin(theta[:, 1]),
        ],
        dim=1,
    )
    if noise > 0:
        x = x + torch.randn(x.shape, generator=gen) * noise
    return x, theta


def make_flat_torus_labels(
    theta: torch.Tensor,
    *,
    task: str = "regression_smooth",
) -> torch.Tensor:
    """Build labels for a flat-torus task from the intrinsic angles.

    Args:
        theta: ``[N, 2]`` intrinsic coordinates from :func:`make_flat_torus`.
        task: One of:

            - ``"regression_smooth"`` (default): ``sin/cos`` of both angles,
              shape ``[N, 4]``. No 0/2π seam discontinuity.
            - ``"hemisphere"``: binary classification on ``θ₁ > π``.
            - ``"xor"``: binary XOR of the two quadrant indicators.

    Returns:
        Tensor of labels (float for regression, long for classification).

    Raises:
        IGLConfigError: For an unknown ``task``.
    """
    if task == "regression_smooth":
        return torch.stack(
            [
                torch.sin(theta[:, 0]),
                torch.cos(theta[:, 0]),
                torch.sin(theta[:, 1]),
                torch.cos(theta[:, 1]),
            ],
            dim=1,
        )
    if task == "hemisphere":
        return (theta[:, 0] > math.pi).long()
    if task == "xor":
        q1 = (theta[:, 0] > math.pi).long()
        q2 = (theta[:, 1] > math.pi).long()
        return (q1 ^ q2).long()
    raise IGLConfigError(f"unknown task: {task!r}")


def make_swiss_roll(
    n_samples: int,
    *,
    noise: float = 0.0,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Swiss roll in ``R³``: ``x(t, h) = (t cos t, h, t sin t)``.

    Intrinsic dimension: 2 (``t`` and ``h``).

    Args:
        n_samples: Number of points to sample.
        noise: Additive Gaussian noise std in ambient space.
        seed: Optional RNG seed.

    Returns:
        ``(X, params)`` where ``X`` is ``[n_samples, 3]`` and ``params`` is
        ``[n_samples, 2]`` intrinsic ``(t, h)`` coordinates.
    """
    if n_samples < 1:
        raise IGLConfigError(f"n_samples must be >= 1, got {n_samples}")
    gen = _maybe_seed(seed)
    t = torch.rand(n_samples, generator=gen) * (4.5 * math.pi - 1.5 * math.pi) + 1.5 * math.pi
    h = torch.rand(n_samples, generator=gen)
    x = torch.stack([t * torch.cos(t), h, t * torch.sin(t)], dim=1)
    if noise > 0:
        x = x + torch.randn(x.shape, generator=gen) * noise
    return x, torch.stack([t, h], dim=1)


def make_moons(
    n_samples: int,
    *,
    noise: float = 0.1,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two interleaving half-moons in ``R²``.

    Args:
        n_samples: Total number of points (split evenly between moons).
        noise: Additive Gaussian noise std.
        seed: Optional RNG seed.

    Returns:
        ``(X, y)`` where ``X`` is ``[n_samples, 2]`` and ``y`` is ``[n_samples]``
        binary class labels (``long`` dtype).
    """
    if n_samples < _MIN_MOON_SAMPLES:
        raise IGLConfigError(f"n_samples must be >= {_MIN_MOON_SAMPLES}, got {n_samples}")
    gen = _maybe_seed(seed)
    n_half = n_samples // 2
    n_other = n_samples - n_half

    angles_upper = torch.linspace(0, math.pi, n_half)
    upper = torch.stack([torch.cos(angles_upper), torch.sin(angles_upper)], dim=1)

    angles_lower = torch.linspace(0, math.pi, n_other)
    lower = torch.stack([1 - torch.cos(angles_lower), -torch.sin(angles_lower) - 0.2], dim=1)

    x = torch.cat([upper, lower], dim=0)
    y = torch.cat([torch.zeros(n_half, dtype=torch.long), torch.ones(n_other, dtype=torch.long)])

    if noise > 0:
        x = x + torch.randn(x.shape, generator=gen) * noise

    perm = torch.randperm(n_samples, generator=gen)
    return x[perm], y[perm]


def embed_in_high_dim(
    x: torch.Tensor,
    *,
    target_dim: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Embed low-D points in ``target_dim`` via padding + random orthogonal rotation.

    Args:
        x: Low-D data ``[N, d]``.
        target_dim: Ambient dimension ``D``. Must satisfy ``D >= d``.
        seed: Optional RNG seed.

    Returns:
        ``[N, D]`` embedded points.

    Raises:
        IGLConfigError: If ``target_dim < x.shape[1]``.
    """
    n_samples, low_dim = x.shape
    if target_dim < low_dim:
        raise IGLConfigError(f"target_dim ({target_dim}) must be >= x.shape[1] ({low_dim})")
    if target_dim == low_dim:
        return x.clone()
    gen = _maybe_seed(seed)
    padded = torch.zeros(n_samples, target_dim, dtype=x.dtype)
    padded[:, :low_dim] = x
    random_matrix = torch.randn(target_dim, target_dim, generator=gen)
    # torch.linalg.qr returns a NamedTuple with partial stubs; cast to recover types.
    qr_result = cast(tuple[torch.Tensor, torch.Tensor], torch.linalg.qr(random_matrix))  # pyright: ignore[reportUnknownMemberType]
    rotation = qr_result[0]
    return padded @ rotation.T


def make_spd_dataset(
    n_samples: int,
    *,
    d: int = 4,
    n_classes: int = 2,
    class_separation: float = 1.0,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate batches of SPD matrices labelled by class.

    Each class is a mixture of two ingredients:

    1. A class-specific mean SPD ``C_k = exp(S_k)`` where ``S_k`` is a
       symmetric matrix sampled once per class (controlled by
       ``class_separation``).
    2. Per-sample noise added in log-Euclidean tangent space, then mapped
       back to the manifold via :func:`igl.spd.matrix_exp_sym`.

    Useful as a labelled toy dataset for :class:`igl.spd.IGLReconSPDClassifier`
    smoke tests and synthetic experiments — no MOABB / mne dependency.

    Args:
        n_samples: Total number of SPD matrices.
        d: Side length of each SPD matrix.
        n_classes: Number of distinct class means.
        class_separation: Std of the per-class log-Euclidean mean. ``0`` →
            all classes collapse to ``I``; ``1.0`` → well-separated.
        seed: Optional RNG seed.

    Returns:
        ``(X, y)`` where ``X`` is ``[n_samples, d, d]`` SPD tensors and ``y``
        is a ``[n_samples]`` long-tensor of class labels.
    """
    if n_samples < 1:
        raise IGLConfigError(f"n_samples must be >= 1, got {n_samples}")
    if d < 1:
        raise IGLConfigError(f"d must be >= 1, got {d}")
    if n_classes < 2:  # noqa: PLR2004
        raise IGLConfigError(f"n_classes must be >= 2, got {n_classes}")

    gen = _maybe_seed(seed)
    class_means: list[torch.Tensor] = []
    for _ in range(n_classes):
        sym = torch.randn(d, d, generator=gen) * class_separation
        sym = 0.5 * (sym + sym.T)
        class_means.append(sym)

    per_class = n_samples // n_classes
    remainder = n_samples - per_class * n_classes

    samples: list[torch.Tensor] = []
    labels: list[int] = []
    for class_idx, base_sym in enumerate(class_means):
        n_here = per_class + (1 if class_idx < remainder else 0)
        noise = torch.randn(n_here, d, d, generator=gen) * 0.1
        noise = 0.5 * (noise + noise.transpose(-1, -2))
        batch_sym = base_sym.unsqueeze(0) + noise
        samples.append(matrix_exp_sym(batch_sym))
        labels.extend([class_idx] * n_here)

    x = torch.cat(samples, dim=0)
    y = torch.tensor(labels, dtype=torch.long)
    perm = torch.randperm(x.shape[0], generator=gen)
    return x[perm], y[perm]


__all__ = [
    "embed_in_high_dim",
    "make_flat_torus",
    "make_flat_torus_labels",
    "make_moons",
    "make_spd_dataset",
    "make_swiss_roll",
]
