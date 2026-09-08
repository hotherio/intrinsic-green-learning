"""Shared synthetic workloads for the device suite (deterministic, device-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

import igl


@dataclass(frozen=True, slots=True)
class Problem:
    """A training problem size: rows, ambient dim, anchors, scales, latent dim, batch sizes."""

    name: str
    n: int
    input_dim: int
    n_anchors: int
    n_scales: int
    max_dim: int
    batch_size: int
    inner_batch_size: int


PROBLEMS: dict[str, Problem] = {
    "small": Problem("small", 1024, 8, 32, 3, 4, 128, 1024),
    "medium": Problem("medium", 4096, 16, 64, 4, 8, 256, 4096),
    "large": Problem("large", 16384, 64, 128, 4, 16, 512, 4096),
}


def make_data(
    problem: Problem, device: torch.device, *, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gaussian inputs with binary labels; the same tensors on every device for a given seed."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(problem.n, problem.input_dim, generator=generator)
    y = torch.randint(0, 2, (problem.n,), generator=generator)
    x_val = torch.randn(max(256, problem.n // 4), problem.input_dim, generator=generator)
    y_val = torch.randint(0, 2, (x_val.shape[0],), generator=generator)
    return x.to(device), y.to(device), x_val.to(device), y_val.to(device)


def make_module(
    problem: Problem, device: torch.device, *, output_dim: int = 2, seed: int = 0, **kwargs: object
) -> igl.IGLModule:
    torch.manual_seed(seed)
    module = igl.IGLModule(
        input_dim=problem.input_dim,
        max_dim=problem.max_dim,
        output_dim=output_dim,
        n_anchors=problem.n_anchors,
        n_scales=problem.n_scales,
        **kwargs,  # type: ignore[arg-type]
    )
    return module.to(device)


def make_config(problem: Problem, *, epochs: int, **overrides: object) -> igl.MatryoshkaConfig:
    base: dict[str, object] = {
        "epochs": epochs,
        "batch_size": problem.batch_size,
        "inner_batch_size": problem.inner_batch_size,
        "early_stop_patience": None,
        "scheduler": "none",
        "verbose": False,
        "final_refresh": "subset",
    }
    base.update(overrides)
    return igl.MatryoshkaConfig(**base)  # type: ignore[arg-type]
