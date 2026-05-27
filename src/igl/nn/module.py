"""The end-to-end IGL ``nn.Module``: encoder → Green kernel → readout.

This is the bare PyTorch entry point for users who want to write a custom
training loop. The high-level scikit-learn wrappers (M3) compose this module
with :class:`igl.MatryoshkaTrainer` internally.
"""

import torch
from torch import nn

from igl.core.encoder import MLPEncoder
from igl.core.kernel import GreenKernel
from igl.core.normalization import normalize_phi
from igl.exceptions import IGLConfigError
from igl.types import EncoderProtocol, NormalizeMode


class IGLModule(nn.Module):
    """End-to-end IGL model: ``x → z → Φ → Φ w + b``.

    The readout weights ``source_weights`` are *not* learned by gradient
    descent — they are refreshed in closed form by
    :func:`igl.direct_solve_weights` (called by :class:`MatryoshkaTrainer`).
    The bias is a learnable parameter, the encoder and Green-kernel parameters
    are too; only the readout is closed-form.

    Args:
        input_dim: Ambient input dimension ``D``.
        max_dim: Latent dimension ``d_max``.
        output_dim: Output dimension (``C`` classes or regression outputs).
        n_anchors: Number of anchors ``R`` in the Green kernel.
        n_scales: Number of kernel scales ``K``.
        operator: Single operator name or a sequence of names. Defaults to
            ``"gaussian"``.
        encoder: Optional pre-built encoder satisfying
            :class:`igl.types.EncoderProtocol`. When ``None``, an
            :class:`igl.MLPEncoder` of width 256 / depth 2 is used.
        normalize: Φ-normalization mode (default ``"softmax"``).
        normalize_input: If ``True`` (default), prepend an
            ``nn.BatchNorm1d(affine=False)`` to the encoder. Stabilises
            training when ambient input dimensions have wildly different
            scales.

    Raises:
        IGLConfigError: When any dimension is non-positive, or the provided
            encoder's ``input_dim`` / ``max_dim`` don't match the module's.
    """

    input_dim: int
    max_dim: int
    output_dim: int

    def __init__(
        self,
        input_dim: int,
        max_dim: int,
        output_dim: int,
        *,
        n_anchors: int = 64,
        n_scales: int = 4,
        operator: str = "gaussian",
        encoder: EncoderProtocol | None = None,
        normalize: NormalizeMode = "softmax",
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise IGLConfigError(f"input_dim must be >= 1, got {input_dim}")
        if max_dim < 1:
            raise IGLConfigError(f"max_dim must be >= 1, got {max_dim}")
        if output_dim < 1:
            raise IGLConfigError(f"output_dim must be >= 1, got {output_dim}")

        if encoder is None:
            inner_encoder: nn.Module = MLPEncoder(input_dim=input_dim, max_dim=max_dim)
        else:
            if encoder.input_dim != input_dim:
                raise IGLConfigError(
                    f"encoder.input_dim ({encoder.input_dim}) != input_dim ({input_dim})",
                )
            if encoder.max_dim != max_dim:
                raise IGLConfigError(
                    f"encoder.max_dim ({encoder.max_dim}) != max_dim ({max_dim})",
                )
            assert isinstance(encoder, nn.Module), "encoder must be an nn.Module"
            inner_encoder = encoder

        if normalize_input:
            self.encoder: nn.Module = nn.Sequential(
                nn.BatchNorm1d(input_dim, affine=False),
                inner_encoder,
            )
        else:
            self.encoder = inner_encoder

        self.green = GreenKernel(
            latent_dim=max_dim,
            n_anchors=n_anchors,
            n_scales=n_scales,
            operator=operator,
        )

        self.input_dim = input_dim
        self.max_dim = max_dim
        self.output_dim = output_dim
        self.normalize: NormalizeMode = normalize

        # Closed-form readout. Stored as a non-learnable buffer so it travels
        # with the module's device but doesn't pick up gradients.
        self.register_buffer("source_weights", torch.zeros(n_anchors, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent encoding ``z = Encoder(x)``."""
        return self.encoder(x)

    def design_matrix(self, x: torch.Tensor, *, gate_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the normalised design matrix Φ from raw input ``x``."""
        z = self.encoder(x)
        if gate_mask is not None:
            z = z * gate_mask.unsqueeze(0)
        phi = self.green(z, gate_mask=gate_mask)
        return normalize_phi(phi, self.normalize)

    def forward(self, x: torch.Tensor, *, gate_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass returning ``Φ w + b`` of shape ``[N, output_dim]``."""
        phi = self.design_matrix(x, gate_mask=gate_mask)
        weights = self.source_weights
        assert isinstance(weights, torch.Tensor), "source_weights must be a tensor"
        return phi @ weights + self.bias

    def set_source_weights(self, weights: torch.Tensor) -> None:
        """Replace the closed-form readout weights with ``weights`` ``[R, C]``."""
        if weights.shape != self.source_weights.shape:  # type: ignore[union-attr]
            raise IGLConfigError(
                f"weights shape {tuple(weights.shape)} != source_weights shape {tuple(self.source_weights.shape)}",  # type: ignore[union-attr]
            )
        self.source_weights.data.copy_(weights.detach().to(self.source_weights.device))  # type: ignore[union-attr]


__all__ = ["IGLModule"]
