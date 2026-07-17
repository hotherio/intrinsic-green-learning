"""Metric estimators for target whitening.

Each function returns a symmetric PSD matrix ``G`` of shape ``[C, C]`` over
the target space. Whitening a reconstruction target by ``G^{1/2}`` makes
plain least squares equal the second-order expansion of the loss geometry
``G`` encodes — for :func:`fisher_pullback`, the KL divergence of the
downstream softmax read-out.
"""

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812

from igl.exceptions import IGLConfigError

__all__ = ["damped_metric", "fisher_pullback", "logit_metric", "tail_metric"]

_MATRIX_NDIM = 2


def logit_metric(w: torch.Tensor) -> torch.Tensor:
    """Pull the Euclidean logit geometry back to the target space.

    Args:
        w: Read-out matrix ``[V, C]`` mapping targets to logits.

    Returns:
        ``G = w^T w`` of shape ``[C, C]``: every logit direction counts
        equally, kernel directions of ``w`` are free.
    """
    if w.dim() != _MATRIX_NDIM:
        raise IGLConfigError(f"w must be 2-D [V, C], got shape {tuple(w.shape)}")
    w = w.detach().float()
    return w.T @ w


def fisher_pullback(
    w: torch.Tensor,
    states: torch.Tensor,
    *,
    n_sub: int = 4096,
    batch: int = 256,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Estimate the Fisher pullback of KL through a softmax read-out.

    ``G = w^T E[diag(p) - p p^T] w`` with ``p = softmax(states @ w^T)``,
    estimated on a subsample of ``states``. This is the second-order
    expansion of ``KL(p(h) || p(h_hat))`` in ``h_hat - h``, averaged over
    the state distribution: softmax saturation annihilates the logit-shift
    direction exactly and down-weights directions that only move improbable
    logits.

    Args:
        w: Read-out matrix ``[V, C]``.
        states: States ``[N, C]`` defining the empirical distribution.
        n_sub: Subsample size for the expectation.
        batch: Batch size for the accumulation.
        generator: Optional RNG for the subsample, for reproducibility.

    Returns:
        The estimated metric ``[C, C]``, symmetric PSD up to sampling noise.
    """
    if w.dim() != _MATRIX_NDIM or states.dim() != _MATRIX_NDIM or w.shape[1] != states.shape[1]:
        raise IGLConfigError(
            f"shape mismatch: w {tuple(w.shape)} (expected [V, C]) vs states {tuple(states.shape)} (expected [N, C])"
        )
    w = w.detach().float()
    states = states.detach().float()
    n = min(n_sub, states.shape[0])
    idx = torch.randperm(states.shape[0], generator=generator)[:n]
    sub = states[idx]

    p_bar = torch.zeros(w.shape[0], dtype=torch.float64)
    second = torch.zeros(w.shape[1], w.shape[1], dtype=torch.float64)
    for start in range(0, n, batch):
        chunk = sub[start : start + batch]
        p = F.softmax(chunk @ w.T, dim=-1).double()
        p_bar += p.sum(dim=0)
        m = p @ w.double()
        second += m.T @ m
    p_bar /= n
    second /= n
    g = (w.double() * p_bar.unsqueeze(1)).T @ w.double() - second
    return (0.5 * (g + g.T)).float()


def damped_metric(g: torch.Tensor, m: torch.Tensor, *, lam: float = 0.1) -> torch.Tensor:
    """Blend a sampled metric with a trace-scaled damping metric.

    ``g + lam * (trace(g) / trace(m)) * m`` — the Tikhonov damping of
    natural-gradient practice. A sampled Fisher estimate leaves the
    near-null spectrum ill-conditioned; a floor metric ``m`` (typically
    :func:`logit_metric`) regularises it without changing the leading
    geometry.

    Args:
        g: The metric to damp ``[C, C]``.
        m: The damping metric ``[C, C]``.
        lam: Damping weight relative to the trace ratio.

    Returns:
        The damped metric ``[C, C]``.
    """
    if g.shape != m.shape:
        raise IGLConfigError(f"shape mismatch: g {tuple(g.shape)} vs m {tuple(m.shape)}")
    scale = lam * float(torch.trace(g)) / max(float(torch.trace(m)), torch.finfo(torch.float32).tiny)
    return g + scale * m


def tail_metric(
    fn: Callable[[torch.Tensor], torch.Tensor],
    states: torch.Tensor,
    g: torch.Tensor,
    *,
    n_probes: int = 64,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pull a metric back through a differentiable map by Hutchinson probing.

    Estimates ``E[J^T g J]`` with ``J = d fn / d x`` averaged over
    ``states``, without ever forming ``J``: for probe vectors
    ``u ~ N(0, I)``, ``E_u[(J^T g^{1/2} u)(J^T g^{1/2} u)^T] = J^T g J``.
    One backward pass per probe. In the paper use case ``fn`` runs the
    remaining transformer blocks and ``g`` is the head's Fisher metric,
    giving each layer the metric of its own consumer.

    Args:
        fn: Differentiable map from ``[N, C_in]`` to ``[N, C_out]``.
        states: Points ``[N, C_in]`` at which to average the pullback.
        g: Metric on the output space ``[C_out, C_out]``.
        n_probes: Number of Hutchinson probes.
        generator: Optional RNG for the probes, for reproducibility.

    Returns:
        The mean-field pullback metric ``[C_in, C_in]``.
    """
    if n_probes < 1:
        raise IGLConfigError(f"n_probes must be >= 1, got {n_probes}")
    from igl.whitening.linalg import psd_sqrt_inv  # local import: avoid cycle at package init

    a, _ = psd_sqrt_inv(g.detach().float())
    x = states.detach().float().requires_grad_(True)
    out = fn(x)
    if out.dim() != _MATRIX_NDIM or out.shape[1] != g.shape[0]:
        raise IGLConfigError(f"fn output shape {tuple(out.shape)} does not match metric shape {tuple(g.shape)}")
    result = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64)
    for probe_index in range(n_probes):
        u = torch.randn(out.shape[1], generator=generator)
        retain = probe_index < n_probes - 1
        (grads,) = cast(
            tuple[torch.Tensor],
            torch.autograd.grad(out @ (a @ u), x, grad_outputs=torch.ones(out.shape[0]), retain_graph=retain),
        )
        result += (grads.double().T @ grads.double()) / out.shape[0]
    return (result / n_probes).float()
