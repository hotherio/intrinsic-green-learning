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
    # anisotropy of the chart at a point: the spread of the log-eigenvalues of A (0 iff A is a multiple of I;
    # a coefficient of variation of log-eigenvalues would be meaningless, their mean can sit at 0)
    aniso_logstd = float(eig.std(dim=-1).mean()) if k > 1 else 0.0
    lam = k / tr
    cond = (eig.max(dim=-1).values - eig.min(dim=-1).values).exp()  # lambda_max / lambda_min per point
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
        "aniso_logstd": aniso_logstd,
        "lambda_mean": float(lam.mean()),
        "lambda_cv": float(lam.std() / lam.mean()),
        "offdiag_mass": offdiag_mass,
        "cond_median": float(cond.median()),
        "cond_max": float(cond.max()),
    }


def report(
    encoder: nn.Module,
    x: torch.Tensor,
    *,
    k: int,
    output_dim: int,
    metric_inv: torch.Tensor | None = None,
    metric_inv_is_projector: bool = True,
    tangent_basis: torch.Tensor | None = None,
    batch_size: int = 512,
) -> dict[str, Any]:
    """The geometry report of ``encoder`` on inputs ``x`` for the leading ``k`` latent dimensions.

    ``metric_inv`` ``[n, D, D]`` gives the on-manifold metric ``A_tan = J M J^T`` next to ``A_full = J J^T``.
    When ``M`` is the tangent projector (ambient metric the identity), ``off_manifold_share`` is the fraction
    of the Jacobian's energy that leaves the tangent space, ``1 - tr(A_tan)/tr(A_full)``; for another ambient
    metric the two matrices are not comparable and the share is omitted. ``tangent_basis`` ``[D, m]`` (orthonormal
    columns, the same at every point: for instance the top PCA directions of a real data set) is the cheap form of
    a projector, ``A_tan = (J V)(J V)^T``, for inputs whose ``[n, D, D]`` projectors would not fit in memory.
    """
    if metric_inv is not None and tangent_basis is not None:
        raise ValueError("give metric_inv or tangent_basis, not both")
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
            elif tangent_basis is not None:
                jv = j @ tangent_basis.to(device=j.device, dtype=j.dtype)  # [B, k, m]
                chunks_tan.append(jv @ jv.transpose(1, 2))
    if encoder_was_training:
        encoder.train()
    a_full = torch.cat(chunks).cpu().double()  # MPS has no float64: summarise on the CPU
    out: dict[str, Any] = {"full": metric_summary(a_full), "n": int(x.shape[0])}
    if chunks_tan:
        a_tan = torch.cat(chunks_tan).cpu().double()
        out["tan"] = metric_summary(a_tan)
        if metric_inv_is_projector or tangent_basis is not None:
            tr_tan = torch.diagonal(a_tan, dim1=-2, dim2=-1).sum(-1)
            tr_full = torch.diagonal(a_full, dim1=-2, dim2=-1).sum(-1)
            out["off_manifold_share"] = 1.0 - float((tr_tan / tr_full.clamp_min(1e-12)).mean())
    return out
