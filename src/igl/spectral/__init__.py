"""Spectral formulation of IGL: eigendecomposition-based Green's kernels.

The local :class:`igl.GreenKernel` evaluates a product of fixed-shape 1-D
kernels (Gaussian, Helmholtz, …) at learnable anchor positions. The
spectral form replaces this with an eigendecomposition of the underlying
operator $L$:

::

    G(z, s) = Σₖ φₖ(z) · φₖ(s) / max(λₖ, ε)

This module provides:

- Closed-form bases (:class:`FourierSineBasis`, :class:`FourierCosineBasis`,
  :class:`ChebyshevBasis`, :class:`LegendreBasis`, :class:`HermiteBasis`,
  :class:`LaguerreBasis`) — orthonormal bases on standard 1-D domains
  with analytic eigenvalues.
- Data-driven bases (:class:`LearnedLaplacianBasis`,
  :class:`GraphLaplacianBasis`) — the Laplace–Beltrami spectrum of the
  *learned* manifold or a user-supplied graph, estimated via sparse
  eigendecomposition.
- Mixture wrapper (:class:`MultiSpectralBasis`) for combining multiple
  bases on the same dimension.
- :class:`SpectralKernel` — the peer of :class:`igl.GreenKernel`.
- :class:`NullSpaceBasis` implementations (:class:`ConstantNullSpace`,
  :class:`PolynomialNullSpace`, :class:`CustomNullSpace`) that augment
  any kernel's design matrix with un-regularised null-space columns —
  the algorithmic counterpart to the paper's "augment the null space".

Not flattened into the top-level ``igl`` namespace. Import explicitly::

    from igl.spectral import SpectralKernel, FourierSineBasis, ConstantNullSpace
"""

from igl.spectral.bases import (
    ChebyshevBasis,
    FourierCosineBasis,
    FourierSineBasis,
    GraphLaplacianBasis,
    HermiteBasis,
    LaguerreBasis,
    LearnedLaplacianBasis,
    LegendreBasis,
)
from igl.spectral.kernel import SpectralKernel
from igl.spectral.multi import MultiSpectralBasis
from igl.spectral.null_space import (
    ConstantNullSpace,
    CustomNullSpace,
    PolynomialNullSpace,
    build_null_space,
)
from igl.spectral.refresh import LearnedLBRefresh

__all__ = [
    "ChebyshevBasis",
    "ConstantNullSpace",
    "CustomNullSpace",
    "FourierCosineBasis",
    "FourierSineBasis",
    "GraphLaplacianBasis",
    "HermiteBasis",
    "LaguerreBasis",
    "LearnedLBRefresh",
    "LearnedLaplacianBasis",
    "LegendreBasis",
    "MultiSpectralBasis",
    "PolynomialNullSpace",
    "SpectralKernel",
    "build_null_space",
]
