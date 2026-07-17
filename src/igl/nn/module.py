"""The end-to-end IGL ``nn.Module``: encoder → Green kernel → readout.

This is the bare PyTorch entry point for users who want to write a custom
training loop. The high-level scikit-learn wrappers (M3) compose this module
with :class:`igl.MatryoshkaTrainer` internally.

Construction supports four mutually-exclusive paths for the encoder:

1. Pass a pre-built ``encoder`` instance satisfying
   :class:`igl.types.EncoderProtocol` — escape hatch for fully custom
   encoders.
2. Pass an ``encoder_config: EncoderConfig`` — IGLModule builds an
   :class:`igl.MLPEncoder` from the config.
3. Pass a top-level ``config: IGLConfig`` — IGLModule uses
   ``config.encoder`` and ``config.kernel`` to populate defaults. Explicit
   per-field kwargs (e.g. ``n_anchors=32``) override the config.
4. Pass nothing — IGLModule uses default :class:`EncoderConfig` /
   :class:`KernelConfig` values.

Combining paths 1 with 2/3, or 2 with 3 when the configs disagree, raises
:class:`igl.IGLConfigError`.
"""

from typing import Protocol, cast

import torch
from torch import nn

from igl.config import EncoderConfig, IGLConfig, KernelConfig
from igl.core.encoder import build_mlp_encoder
from igl.core.kernel import GreenKernel
from igl.core.normalization import normalize_phi
from igl.exceptions import IGLConfigError
from igl.spectral.null_space import build_null_space
from igl.types import EncoderProtocol, NormalizeMode, NormalizeModeLike, OperatorName, OperatorNameLike


class _HasOutputDim(Protocol):
    """Structural type for kernels exposing the number of design-matrix columns.

    Both :class:`igl.GreenKernel` and :class:`igl.spectral.SpectralKernel`
    declare ``output_dim: int`` as a class attribute, so they satisfy this
    Protocol structurally. Used to narrow ``self.green`` for buffer sizing.
    """

    output_dim: int


def _resolve_kernel_params(
    *,
    n_anchors: int | None,
    n_scales: int | None,
    operator: OperatorNameLike | None,
    normalize: NormalizeModeLike | None,
    kernel_cfg: KernelConfig,
) -> tuple[int, int, OperatorName | tuple[OperatorName, ...], NormalizeMode]:
    """Explicit kwargs win over the kernel config; otherwise use config values."""
    resolved_anchors = n_anchors if n_anchors is not None else kernel_cfg.n_anchors
    resolved_scales = n_scales if n_scales is not None else kernel_cfg.n_scales
    resolved_operator: OperatorName | tuple[OperatorName, ...] = (
        OperatorName(operator) if operator is not None else kernel_cfg.operator
    )
    # kernel_cfg.normalize is coerced to a NormalizeMode in __post_init__;
    # rewrap to satisfy the static type checker.
    resolved_normalize = NormalizeMode(normalize) if normalize is not None else NormalizeMode(kernel_cfg.normalize)
    return resolved_anchors, resolved_scales, resolved_operator, resolved_normalize


class IGLModule(nn.Module):
    """End-to-end IGL model: ``x → z → Φ → Φ w + b``.

    The readout weights ``source_weights`` are *not* learned by gradient
    descent — they are refreshed in closed form by
    :func:`igl.direct_solve_weights` (called by :class:`MatryoshkaTrainer`).
    The bias is a learnable parameter; the encoder and Green-kernel
    parameters are too — only the readout is closed-form.

    Args:
        input_dim: Ambient input dimension ``D``.
        max_dim: Latent dimension ``d_max``. Always wins over
            ``config.max_dim`` if both are explicit (raises on mismatch).
        output_dim: Output dimension (``C`` classes or regression outputs).
        n_anchors: Number of anchors ``R`` in the Green kernel. ``None``
            defers to ``config.kernel.n_anchors`` (or
            :class:`KernelConfig`'s default of ``64``).
        n_scales: Number of kernel scales ``K``. ``None`` defers to
            ``config.kernel.n_scales`` (default ``4``).
        operator: Single operator name. ``None`` defers to
            ``config.kernel.operator`` (default
            :data:`igl.OperatorName.GAUSSIAN`). For multi-operator setups
            (e.g. ``("gaussian", "helmholtz")``), build the kernel via
            :class:`KernelConfig` and pass through ``config``.
        encoder: Optional pre-built encoder satisfying
            :class:`igl.types.EncoderProtocol`.
        encoder_config: Optional :class:`EncoderConfig` from which to build
            an :class:`MLPEncoder`. Mutually exclusive with ``encoder``.
        normalize: Φ-normalization mode. ``None`` defers to
            ``config.kernel.normalize`` (default
            :data:`igl.NormalizeMode.NW`).
        normalize_input: If ``True``, prepend an
            ``nn.BatchNorm1d(affine=False)`` to the encoder. Stabilises
            training when ambient input dimensions have wildly different
            scales. Default ``False`` — appropriate for inputs that have
            been pre-normalised (e.g. log-Eig tangent vectors from
            :class:`igl.spd.LogEigVectorizer`).
        config: Optional top-level :class:`IGLConfig`. When provided,
            populates defaults for any field passed as ``None``. Explicit
            per-field kwargs always win.

    Raises:
        IGLConfigError: For dimension mismatches, conflicting parameters,
            non-positive dimensions, or an explicit ``max_dim`` that
            contradicts ``config.max_dim``.
    """

    input_dim: int
    max_dim: int
    output_dim: int

    def __init__(  # noqa: PLR0912
        self,
        input_dim: int,
        max_dim: int,
        output_dim: int,
        *,
        n_anchors: int | None = None,
        n_scales: int | None = None,
        operator: OperatorNameLike | None = None,
        encoder: EncoderProtocol | None = None,
        encoder_config: EncoderConfig | None = None,
        normalize: NormalizeModeLike | None = None,
        normalize_input: bool = False,
        config: IGLConfig | None = None,
        kernel: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise IGLConfigError(f"input_dim must be >= 1, got {input_dim}")
        if max_dim < 1:
            raise IGLConfigError(f"max_dim must be >= 1, got {max_dim}")
        if output_dim < 1:
            raise IGLConfigError(f"output_dim must be >= 1, got {output_dim}")

        if encoder is not None and encoder_config is not None:
            raise IGLConfigError(
                "pass either encoder or encoder_config (or rely on config), not both",
            )
        if config is not None and config.max_dim != max_dim:
            raise IGLConfigError(
                f"max_dim ({max_dim}) does not match config.max_dim ({config.max_dim})",
            )
        if encoder is not None and config is not None:
            # A pre-built encoder takes precedence, but we still want the kernel
            # defaults from config — that's fine, no conflict to flag.
            pass

        # Resolve encoder source.
        if encoder is not None:
            if encoder.input_dim != input_dim:
                raise IGLConfigError(
                    f"encoder.input_dim ({encoder.input_dim}) != input_dim ({input_dim})",
                )
            if encoder.max_dim != max_dim:
                raise IGLConfigError(
                    f"encoder.max_dim ({encoder.max_dim}) != max_dim ({max_dim})",
                )
            if not isinstance(encoder, nn.Module):
                raise IGLConfigError(
                    f"encoder must be an nn.Module, got {type(encoder).__name__}",
                )
            inner_encoder: nn.Module = encoder
        else:
            resolved_encoder_cfg = (
                encoder_config if encoder_config is not None else (config.encoder if config is not None else EncoderConfig())
            )
            inner_encoder = build_mlp_encoder(input_dim, max_dim, config=resolved_encoder_cfg)

        # Resolve kernel + normalize params.
        kernel_cfg = config.kernel if config is not None else KernelConfig()
        resolved_anchors, resolved_scales, resolved_operator, resolved_normalize = _resolve_kernel_params(
            n_anchors=n_anchors,
            n_scales=n_scales,
            operator=operator,
            normalize=normalize,
            kernel_cfg=kernel_cfg,
        )

        if normalize_input:
            self.encoder: nn.Module = nn.Sequential(
                nn.BatchNorm1d(input_dim, affine=False),
                inner_encoder,
            )
        else:
            self.encoder = inner_encoder

        if kernel is None:
            # `null_space`, `polynomial_degree`, `sigma_log_range` and
            # `anchor_init_std` have no per-field kwarg override, so they come
            # straight off the config. `KernelConfig()`'s defaults match
            # GreenKernel's own, keeping this branch a no-op for callers who
            # pass no config at all.
            self.green: nn.Module = GreenKernel(
                latent_dim=max_dim,
                n_anchors=resolved_anchors,
                n_scales=resolved_scales,
                operator=resolved_operator,
                sigma_log_range=kernel_cfg.sigma_log_range,
                anchor_init_std=kernel_cfg.anchor_init_std,
                null_space=build_null_space(
                    kernel_cfg.null_space,
                    latent_dim=max_dim,
                    degree=kernel_cfg.polynomial_degree,
                ),
            )
        else:
            # Pre-built kernel (e.g. SpectralKernel). It must expose
            # `output_dim` for sizing the source_weights buffer.
            if not hasattr(kernel, "output_dim"):
                raise IGLConfigError(
                    "kernel must expose an `output_dim` attribute (number of design-matrix columns)",
                )
            self.green = kernel

        # Size the readout buffer to the kernel's design-matrix width. Both
        # branches above guarantee `self.green` satisfies _HasOutputDim
        # (built-in GreenKernel declares it; user kernels are validated above).
        n_columns = int(cast(_HasOutputDim, self.green).output_dim)

        self.normalize_input = normalize_input
        self.input_dim = input_dim
        self.max_dim = max_dim
        self.output_dim = output_dim
        self.normalize: NormalizeMode = resolved_normalize

        # Closed-form readout. Initialised with small random values so the
        # RNG-consumption profile matches the EEG reference's `IGLRegressor`
        # (which uses `nn.Parameter(randn * 0.01)`). Registered as a
        # non-trainable Parameter — gradients never touch it because the
        # MatryoshkaTrainer builds its optimizer from the encoder and Green
        # kernel parameter groups only (see ``src/igl/core/trainer.py``).
        self.source_weights = nn.Parameter(torch.randn(n_columns, output_dim) * 0.01)
        self.source_weights.requires_grad_(False)
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
        return phi @ self.source_weights + self.bias

    def set_source_weights(self, weights: torch.Tensor) -> None:
        """Replace the closed-form readout weights with ``weights`` ``[R, C]``."""
        if weights.shape != self.source_weights.shape:
            raise IGLConfigError(
                f"weights shape {tuple(weights.shape)} != source_weights shape {tuple(self.source_weights.shape)}",
            )
        self.source_weights.data.copy_(weights.detach().to(self.source_weights.device))


__all__ = ["IGLModule"]
