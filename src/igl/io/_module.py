"""Rebuild an :class:`igl.IGLModule` from a serialized configuration."""

from igl.config import IGLConfig
from igl.nn.module import IGLModule
from igl.spectral._build import build_kernel_null_space, build_spectral_kernel

__all__ = ["build_module_from_config"]


def build_module_from_config(config: IGLConfig, *, input_dim: int, output_dim: int) -> IGLModule:
    """Construct an :class:`IGLModule` from a fully-resolved :class:`IGLConfig`.

    Mirrors the estimator construction path (spectral kernel when
    ``config.spectral`` is set, otherwise the local kernel with the
    configured null space). Loaded checkpoints overwrite every parameter via
    ``load_state_dict(strict=True)``, so any architectural drift between the
    saved config and this builder fails loudly at load time rather than
    silently.

    Args:
        config: The resolved configuration (kernel and encoder fields are
            authoritative; ctor-override style kwargs must already be folded
            in).
        input_dim: Ambient input dimension.
        output_dim: Output dimension.

    Returns:
        A freshly-initialized module with the configured architecture.
    """
    kernel: object = None
    if config.spectral is not None:
        kernel = build_spectral_kernel(latent_dim=config.max_dim, config=config.spectral)
    else:
        kernel = build_kernel_null_space(latent_dim=config.max_dim, config=config.kernel)
    return IGLModule(
        input_dim=input_dim,
        max_dim=config.max_dim,
        output_dim=output_dim,
        n_anchors=config.kernel.n_anchors,
        n_scales=config.kernel.n_scales,
        operator=config.kernel.operator,  # pyright: ignore[reportArgumentType]
        encoder_config=config.encoder,
        normalize=config.kernel.normalize,
        config=None,
        kernel=kernel,  # type: ignore[arg-type]
    )
