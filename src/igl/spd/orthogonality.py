"""Pullback-metric penalties (orthogonality, conformality) + Stiefel projection helpers.

The pullback metric of an encoder ``Ψ`` at input ``x`` is the symmetric
positive-semidefinite matrix ``g(x) = J(x) J(x)^T`` where ``J = ∂Ψ/∂x``.
When ``g(x)`` is diagonal, the latent coordinates are orthogonal to first
order around ``x`` — the Stäckel condition; geodesics separate cleanly per
coordinate. The :class:`OrthogonalityPenalty` regularizer drives the
off-diagonal mass toward zero so the encoder learns
geometrically-disentangled coordinates.

:class:`ConformalityPenalty` is the stricter, opt-in target: ``g(x)`` a
multiple of the identity (right angles *and* equal stretching in every
direction). It is computed on the axis-normalised metric because the Green
kernel's per-axis widths absorb any constant per-axis rescaling of the
chart; only spatially varying anisotropy costs anchors.

Three complementary tools:

- :func:`init_encoder_orthogonal_` initialises each ``nn.Linear`` weight in
  the encoder to a Stiefel matrix via PyTorch's QR-orthogonal init. Cheap
  and applies before training.
- :class:`OrthogonalityPenalty` implements :class:`igl.types.ExtraLoss` and
  plugs into :meth:`igl.MatryoshkaTrainer.fit` via the ``extra_losses``
  parameter. Active during training; skipped on ``k < 2`` (the metric is
  trivially diagonal for a single dimension).
- :class:`ConformalityPenalty` has the same shape and plumbing; ``beta``
  interpolates between the two targets, ``all_prefixes`` penalises every
  nested Matryoshka block, and ``metric_inv`` lets the metric be taken on the
  data manifold (a tangent projector) or under another ambient metric.
"""

from collections.abc import Callable

import torch
from torch import nn
from torch.func import functional_call, jacrev, vmap

from igl.exceptions import IGLConfigError

_MIN_K_FOR_ORTH = 2
_BATCHNORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def _batchnorm_in_training(module: nn.Module) -> bool:
    """True when a BatchNorm layer of ``module`` is in training mode.

    A per-sample ``jacrev`` sees each sample alone, so it drops the
    cross-sample terms BatchNorm's batch statistics create in training mode;
    the loop over output dimensions keeps them.
    """
    return any(isinstance(m, _BATCHNORM_TYPES) and m.training for m in module.modules())


def jacobian(encoder: nn.Module, x: torch.Tensor, *, output_dim: int, vectorized: bool | None = None) -> torch.Tensor:
    """Compute ``J[b, j, i] = ∂encoder(x_b)_j / ∂x_b_i`` keeping the autograd graph.

    Used to drive the orthogonality penalty: the resulting Jacobian gradient
    flows back to the encoder parameters so the penalty actually updates them.

    Two implementations give the same Jacobian: a loop of ``output_dim``
    backward passes (the reference, bit-identical on CPU) and a vectorised
    ``vmap(jacrev)`` that is 2-5x faster (measured on MPS and an H100) and
    launches far fewer kernels. The vectorised form is taken off-CPU unless a
    BatchNorm layer is training, in which case the loop is required.

    Args:
        encoder: The encoder module.
        x: Input batch ``[B, D]``.
        output_dim: Number of output dimensions ``d`` of the encoder.
        vectorized: Force one implementation; ``None`` picks by device.

    Returns:
        Jacobian tensor of shape ``[B, d, D]``.
    """
    if vectorized is None:
        vectorized = x.device.type != "cpu"
    if vectorized and not _batchnorm_in_training(encoder):
        state: dict[str, torch.Tensor] = {**dict(encoder.named_parameters()), **dict(encoder.named_buffers())}

        def single(xi: torch.Tensor) -> torch.Tensor:
            return functional_call(encoder, state, (xi.unsqueeze(0),)).squeeze(0)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        return vmap(jacrev(single))(x.detach())  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    x = x.detach().requires_grad_(True)
    z = encoder(x)
    rows: list[torch.Tensor] = []
    for j in range(output_dim):
        grad = torch.autograd.grad(z[:, j].sum(), x, create_graph=True, retain_graph=True)[0]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        rows.append(grad)  # pyright: ignore[reportUnknownArgumentType]
    return torch.stack(rows, dim=1)


def pullback_metric(j: torch.Tensor) -> torch.Tensor:
    """Pullback metric ``g = J J^T`` of shape ``[B, d, d]``.

    Args:
        j: Jacobian tensor ``[B, d, D]``.
    """
    return torch.bmm(j, j.transpose(1, 2))


def orthogonality_loss(g: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Sum of squared off-diagonals divided by sum of squared diagonals.

    Args:
        g: Pullback metric ``[B, d, d]``.
        eps: Numerical floor on the diagonal-squared denominator.

    Returns:
        Scalar penalty (mean across the batch).
    """
    diag = torch.diagonal(g, dim1=-2, dim2=-1)
    off = g - torch.diag_embed(diag)
    off_sq = (off**2).sum(dim=(-2, -1))
    diag_sq = (diag**2).sum(dim=-1)
    return (off_sq / (diag_sq + eps)).mean()


def conformality_loss(
    g: torch.Tensor,
    *,
    beta: float = 1.0,
    axis_normalize: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Traceless part of the pullback metric, as a scale-free penalty.

    For ``beta = 1`` the contribution per sample is
    ``‖ĝ − tr(ĝ)/k · I‖²_F / (‖ĝ‖²_F + eps)``, zero exactly when the metric is
    a multiple of the identity (a conformal chart). The numerator splits into
    the off-diagonal mass and the spread of the diagonal; ``beta`` weights the
    second term, so ``beta = 0`` keeps only the orthogonality target (with the
    full metric as denominator).

    With ``axis_normalize`` the metric is first rescaled per axis by the
    batch-mean diagonal, ``ĝ = D^{-1/2} g D^{-1/2}`` (``D`` detached), which
    makes the penalty blind to a constant per-axis rescaling of the latent:
    the kernel's per-axis widths absorb that rescaling for free, so only
    spatially varying anisotropy should be penalised.

    Args:
        g: Pullback metric ``[B, k, k]``.
        beta: Weight of the diagonal-spread term; ``1`` is conformality,
            ``0`` orthogonality.
        axis_normalize: Rescale each axis by the batch-mean diagonal first.
        eps: Numerical floor on the denominator and on the diagonal used for
            the normalisation.

    Returns:
        Scalar penalty (mean across the batch).
    """
    if axis_normalize:
        scale = torch.diagonal(g, dim1=-2, dim2=-1).mean(dim=0).clamp_min(eps).rsqrt().detach()
        g = g * scale.unsqueeze(-1) * scale.unsqueeze(-2)
    diag = torch.diagonal(g, dim1=-2, dim2=-1)
    off = g - torch.diag_embed(diag)
    off_sq = (off**2).sum(dim=(-2, -1))
    spread = ((diag - diag.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1)
    total = (g**2).sum(dim=(-2, -1))
    return ((off_sq + beta * spread) / (total + eps)).mean()


def conformal_factor(j: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Conformal factor ``λ = k / tr(J Jᵀ)`` per sample, shape ``[B]``.

    Under the conformally flat hypothesis ``g = λ I`` in the chart, the inverse
    metric is ``J Jᵀ = λ⁻¹ I`` and its trace is ``k / λ``; the trace needs no
    matrix product, only the squared Frobenius norm of each sample's Jacobian.

    Args:
        j: Jacobian tensor ``[B, k, D]`` of the active latent dimensions.
        eps: Numerical floor on the trace.
    """
    k = j.shape[1]
    return k / (j**2).sum(dim=(-2, -1)).clamp_min(eps)


@torch.no_grad()
def init_encoder_orthogonal_(encoder: nn.Module) -> int:
    """Initialize every ``nn.Linear`` weight in ``encoder`` with QR-orthogonal init.

    Operates in-place. Returns the number of linear layers initialised.
    """
    count = 0
    for module in encoder.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)  # pyright: ignore[reportUnknownMemberType]
            count += 1
    return count


class OrthogonalityPenalty:
    """:class:`igl.types.ExtraLoss` driving the pullback metric toward diagonality.

    The contribution is

        weight × ‖off-diag(g)‖² / (‖diag(g)‖² + ε)

    where ``g = J J^T`` and ``J`` is the encoder Jacobian at the truncated
    output. Skipped for ``k < 2`` (the metric is trivially diagonal for a
    single dimension).

    Args:
        weight: Multiplier applied by the trainer. ``0.1`` is the value
            validated on EEG data.
        every: Call frequency in *batches*. ``20`` is a reasonable default —
            the Jacobian computation is O(d · D) so applying it on every
            batch is expensive.
        eps: Diagonal-squared floor for numerical stability.
    """

    weight: float
    every: int
    eps: float

    def __init__(self, *, weight: float = 0.1, every: int = 20, eps: float = 1e-6) -> None:
        if weight < 0:
            raise IGLConfigError(f"weight must be non-negative, got {weight}")
        if every < 1:
            raise IGLConfigError(f"every must be >= 1, got {every}")
        self.weight = weight
        self.every = every
        self.eps = eps

    def __call__(
        self,
        *,
        encoder: nn.Module,
        x_batch: torch.Tensor,
        gate_mask: torch.Tensor,
        k: int,
        epoch: int,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor | None:
        if k < _MIN_K_FOR_ORTH:
            return None
        # Compute the Jacobian only for the active latent dims.
        # The trainer's mask is a prefix and k is known on the host: no mask read.
        j_full = jacobian(encoder, x_batch, output_dim=gate_mask.shape[0])
        j_active = j_full[:, :k, :]
        g = pullback_metric(j_active)
        return orthogonality_loss(g, eps=self.eps)


class ConformalityPenalty:
    """:class:`igl.types.ExtraLoss` driving the pullback metric toward a multiple of the identity.

    Same plumbing as :class:`OrthogonalityPenalty` (``extra_losses`` of
    :meth:`igl.MatryoshkaTrainer.fit`, skipped for ``k < 2``), with the
    :func:`conformality_loss` target instead. Opt-in: nothing in the trainer
    changes.

    Args:
        weight: Multiplier applied by the trainer.
        every: Call frequency in *batches*; the index resets every epoch, so
            at 16 batches per epoch ``every=20`` fires once per epoch.
        beta: Weight of the diagonal-spread term, see :func:`conformality_loss`;
            ``1`` is conformality, ``0`` orthogonality on the normalised metric.
        axis_normalize: Rescale each axis by the batch-mean diagonal first.
        all_prefixes: Average the loss over every nested leading block
            ``k' = 2 … k`` instead of the sampled prefix alone. The Jacobian is
            computed for all latent dimensions anyway, so this is free.
        metric_inv: Optional map from the input batch ``[B, D]`` to a matrix
            ``M(x)`` of shape ``[B, D, D]``; the metric becomes ``J M Jᵀ``. Use a
            tangent projector to take the metric on the data manifold only, or
            the inverse of another ambient metric (e.g. the affine-invariant
            metric on SPD inputs), or their product. ``None`` keeps ``J Jᵀ``.
        eps: Numerical floor passed to :func:`conformality_loss`.
    """

    weight: float
    every: int
    beta: float
    axis_normalize: bool
    all_prefixes: bool
    metric_inv: Callable[[torch.Tensor], torch.Tensor] | None
    eps: float

    def __init__(
        self,
        *,
        weight: float = 0.1,
        every: int = 20,
        beta: float = 1.0,
        axis_normalize: bool = True,
        all_prefixes: bool = False,
        metric_inv: Callable[[torch.Tensor], torch.Tensor] | None = None,
        eps: float = 1e-6,
    ) -> None:
        if weight < 0:
            raise IGLConfigError(f"weight must be non-negative, got {weight}")
        if every < 1:
            raise IGLConfigError(f"every must be >= 1, got {every}")
        if beta < 0:
            raise IGLConfigError(f"beta must be non-negative, got {beta}")
        if eps <= 0:
            raise IGLConfigError(f"eps must be positive, got {eps}")
        self.weight = weight
        self.every = every
        self.beta = beta
        self.axis_normalize = axis_normalize
        self.all_prefixes = all_prefixes
        self.metric_inv = metric_inv
        self.eps = eps

    def _loss(self, g: torch.Tensor) -> torch.Tensor:
        return conformality_loss(g, beta=self.beta, axis_normalize=self.axis_normalize, eps=self.eps)

    def __call__(
        self,
        *,
        encoder: nn.Module,
        x_batch: torch.Tensor,
        gate_mask: torch.Tensor,
        k: int,
        epoch: int,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor | None:
        if k < _MIN_K_FOR_ORTH:
            return None
        j_full = jacobian(encoder, x_batch, output_dim=gate_mask.shape[0])
        j_active = j_full[:, :k, :]
        if self.metric_inv is None:
            g = pullback_metric(j_active)
        else:
            g = torch.bmm(torch.bmm(j_active, self.metric_inv(x_batch)), j_active.transpose(1, 2))
        if not self.all_prefixes:
            return self._loss(g)
        blocks = [self._loss(g[:, :kk, :kk]) for kk in range(_MIN_K_FOR_ORTH, k + 1)]
        return torch.stack(blocks).mean()


__all__ = [
    "ConformalityPenalty",
    "OrthogonalityPenalty",
    "conformal_factor",
    "conformality_loss",
    "init_encoder_orthogonal_",
    "jacobian",
    "orthogonality_loss",
    "pullback_metric",
]
