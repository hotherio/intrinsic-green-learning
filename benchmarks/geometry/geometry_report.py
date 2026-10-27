"""The geometry report: what the Jacobian of a chart says before and after training (prototype of proposal D).

From an encoder (or any differentiable chart map) and a batch of inputs, for the leading ``k`` latent
dimensions: the inverse metric ``A_full = J J^T`` that today's penalty sees and, when the bed supplies
``metric_inv``, the on-manifold ``A_tan = J M J^T``; for each: both penalty floors (orthogonality and
conformality, the latter axis-normalised and plain), the best global scale ``c = mean tr(A)/k``,
``max |A/c - I|`` in operator norm and the Born depth ``n(eps)`` it predicts, the anisotropy of the
chart (coefficient of variation of the log-eigenvalues), the conformal factor ``lambda = k / tr(A)``
and its spread, and the off-diagonal mass fraction that sets the error of a Rademacher trace probe.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from igl.spd import conformality_loss, jacobian, orthogonality_loss

__all__ = ["metric_summary", "report"]


def metric_summary(a: torch.Tensor, *, eps_depth: float = 1e-3) -> dict[str, float]:
    """Summary of a batch of inverse metrics ``a`` ``[n, k, k]``."""
    k = a.shape[-1]
    tr = torch.diagonal(a, dim1=-2, dim2=-1).sum(-1)
    scale = float(tr.mean() / k)
    dev = a / scale - torch.eye(k, dtype=a.dtype)
    op_norms = torch.linalg.matrix_norm(dev, ord=2)
    norm_inf = float(op_norms.max())
    eig = torch.linalg.eigvalsh(0.5 * (a + a.transpose(-1, -2))).clamp_min(1e-12).log()
    aniso_cv = float((eig.std(dim=-1) / eig.mean(dim=-1).abs().clamp_min(1e-12)).mean()) if k > 1 else 0.0
    lam = k / tr
    off = a - torch.diag_embed(torch.diagonal(a, dim1=-2, dim2=-1))
    offdiag_mass = float(((off**2).sum((-2, -1)) / (a**2).sum((-2, -1))).mean())
    return {
        "k": k,
        "scale": scale,
        "norm_inf": norm_inf,
        "norm_mean": float(op_norms.mean()),
        "n_eps": math.log(1 / eps_depth) / math.log(1 / norm_inf) if 0 < norm_inf < 1 else float("inf"),
        "sqrt_orth": float(orthogonality_loss(a)) ** 0.5,
        "sqrt_conf": float(conformality_loss(a)) ** 0.5,
        "sqrt_conf_plain": float(conformality_loss(a, axis_normalize=False)) ** 0.5,
        "aniso_cv": aniso_cv,
        "lambda_mean": float(lam.mean()),
        "lambda_cv": float(lam.std() / lam.mean()),
        "offdiag_mass": offdiag_mass,
    }


def report(
    encoder: nn.Module,
    x: torch.Tensor,
    *,
    k: int,
    output_dim: int,
    metric_inv: torch.Tensor | None = None,
    batch_size: int = 512,
) -> dict[str, Any]:
    """The geometry report of ``encoder`` on inputs ``x`` for the leading ``k`` latent dimensions.

    ``metric_inv`` ``[n, D, D]`` gives the on-manifold metric ``A_tan = J M J^T`` next to ``A_full = J J^T``.
    """
    encoder_was_training = encoder.training
    encoder.eval()
    chunks: list[torch.Tensor] = []
    chunks_tan: list[torch.Tensor] = []
    with torch.enable_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size]
            j = jacobian(encoder, xb, output_dim=output_dim)[:, :k, :].detach()
            chunks.append(j @ j.transpose(1, 2))
            if metric_inv is not None:
                m = metric_inv[start : start + batch_size].to(j.dtype)
                chunks_tan.append(j @ m @ j.transpose(1, 2))
    if encoder_was_training:
        encoder.train()
    a_full = torch.cat(chunks).double()
    out: dict[str, Any] = {"full": metric_summary(a_full), "n": int(x.shape[0])}
    if chunks_tan:
        a_tan = torch.cat(chunks_tan).double()
        out["tan"] = metric_summary(a_tan)
        out["off_manifold_share"] = 1.0 - float(
            torch.linalg.matrix_norm(a_tan).mean() / torch.linalg.matrix_norm(a_full).mean()
        )
    return out
