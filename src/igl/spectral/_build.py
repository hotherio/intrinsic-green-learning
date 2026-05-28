"""Internal factories: build SpectralKernel / GreenKernel from configs.

Kept separate from the public ``__init__`` to avoid pulling the heavy
spectral imports (numpy/scipy paths) into every estimator import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from igl.exceptions import IGLConfigError
from igl.spectral.bases.chebyshev import ChebyshevBasis
from igl.spectral.bases.fourier_cosine import FourierCosineBasis
from igl.spectral.bases.fourier_sine import FourierSineBasis
from igl.spectral.bases.hermite import HermiteBasis
from igl.spectral.bases.laguerre import LaguerreBasis
from igl.spectral.bases.learned_lb import LearnedLaplacianBasis
from igl.spectral.bases.legendre import LegendreBasis
from igl.spectral.kernel import SpectralKernel
from igl.spectral.null_space import build_null_space
from igl.types import NullSpaceKind, SpectralKind

if TYPE_CHECKING:
    from igl.config import KernelConfig, SpectralConfig


_BASIS_REGISTRY: dict[SpectralKind, object] = {
    SpectralKind.FOURIER_SINE: FourierSineBasis,
    SpectralKind.FOURIER_COSINE: FourierCosineBasis,
    SpectralKind.CHEBYSHEV: ChebyshevBasis,
    SpectralKind.LEGENDRE: LegendreBasis,
    SpectralKind.HERMITE: HermiteBasis,
    SpectralKind.LAGUERRE: LaguerreBasis,
}


def _make_basis(kind: SpectralKind, n_modes: int, *, k_nn: int) -> object:
    if kind is SpectralKind.LEARNED_LB:
        return LearnedLaplacianBasis(n_modes=n_modes, k_nn=k_nn)
    cls = _BASIS_REGISTRY.get(kind)
    if cls is None:
        raise IGLConfigError(
            f"SpectralKind {kind!r} cannot be built from a SpectralConfig; "
            "GRAPH_LAPLACIAN requires a user-supplied adjacency — instantiate "
            "GraphLaplacianBasis directly and pass it via SpectralKernel(bases=...).",
        )
    return cls(n_modes=n_modes)  # type: ignore[operator]


def build_spectral_kernel(
    *,
    latent_dim: int,
    config: SpectralConfig,
) -> SpectralKernel:
    """Construct a :class:`SpectralKernel` from a :class:`SpectralConfig`."""
    kind = config.kind
    if isinstance(kind, SpectralKind | str):
        basis: object = _make_basis(SpectralKind(kind), config.n_modes, k_nn=config.k_nn)
        bases: object = basis
    else:
        kinds = [SpectralKind(k) for k in kind]
        if len(kinds) != latent_dim:
            raise IGLConfigError(
                f"per-dim spectral kinds: expected {latent_dim} entries, got {len(kinds)}",
            )
        bases = [_make_basis(k, config.n_modes, k_nn=config.k_nn) for k in kinds]

    null = build_null_space(
        config.null_space,
        latent_dim=latent_dim,
        degree=config.polynomial_degree,
    )
    return SpectralKernel(
        latent_dim=latent_dim,
        bases=bases,  # type: ignore[arg-type]
        n_anchors=config.n_anchors,
        null_space=null,
        epsilon=config.epsilon,
        anchor_init_std=config.anchor_init_std,
    )


def build_kernel_null_space(*, latent_dim: int, config: KernelConfig) -> object:
    """When the local GreenKernel has a null-space config, build it pre-wired.

    Returns ``None`` (no pre-built kernel) when no null-space is configured —
    :class:`IGLModule` will construct the default GreenKernel itself.
    """
    from igl.core.kernel import GreenKernel  # noqa: PLC0415

    null = build_null_space(
        config.null_space,
        latent_dim=latent_dim,
        degree=config.polynomial_degree,
    )
    if null is None and isinstance(config.null_space, NullSpaceKind) and config.null_space is NullSpaceKind.NONE:
        return None
    return GreenKernel(
        latent_dim=latent_dim,
        n_anchors=config.n_anchors,
        n_scales=config.n_scales,
        operator=config.operator,  # type: ignore[arg-type]
        sigma_log_range=config.sigma_log_range,
        anchor_init_std=config.anchor_init_std,
        null_space=null,
    )


__all__ = ["build_kernel_null_space", "build_spectral_kernel"]
