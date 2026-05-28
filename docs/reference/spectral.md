# `igl.spectral`

Spectral formulation of IGL: eigendecomposition-based Green's kernels +
null-space augmentation.

Not flattened into the top-level ``igl`` namespace. Import explicitly::

    from igl.spectral import SpectralKernel, FourierSineBasis, ConstantNullSpace

## Mathematical setup

For an operator $L$ with eigendecomposition $L \phi_n = \lambda_n \phi_n$,
the Green's function is

$$
G(z, s) = \sum_n \frac{\phi_n(z)\, \phi_n(s)}{\max(\lambda_n, \varepsilon)}.
$$

Zero-$\lambda$ modes ARE the null space — modes the Green's function
cannot reach. The kernel-agnostic
`NullSpaceBasis` adds them back as extra
design-matrix columns whose coefficients are fitted by lstsq without
Tikhonov shrinkage.

## SpectralKernel

::: igl.spectral.kernel.SpectralKernel
    options:
      show_root_heading: true
      show_source: false

## Closed-form 1-D bases

::: igl.spectral.bases.fourier_sine.FourierSineBasis
    options:
      show_root_heading: true
      show_source: false

::: igl.spectral.bases.fourier_cosine.FourierCosineBasis
    options:
      show_root_heading: true
      show_source: false

::: igl.spectral.bases.chebyshev.ChebyshevBasis
    options:
      show_root_heading: true
      show_source: false

::: igl.spectral.bases.legendre.LegendreBasis
    options:
      show_root_heading: true
      show_source: false

::: igl.spectral.bases.hermite.HermiteBasis
    options:
      show_root_heading: true
      show_source: false

::: igl.spectral.bases.laguerre.LaguerreBasis
    options:
      show_root_heading: true
      show_source: false

## Data-driven bases

::: igl.spectral.bases.learned_lb.LearnedLaplacianBasis
    options:
      show_root_heading: true
      show_source: false

::: igl.spectral.bases.graph_laplacian.GraphLaplacianBasis
    options:
      show_root_heading: true
      show_source: false

## Mixtures

::: igl.spectral.multi.MultiSpectralBasis
    options:
      show_root_heading: true
      show_source: false

## Null-space augmentation

::: igl.spectral.null_space.ConstantNullSpace
    options:
      show_root_heading: true
      show_source: false

::: igl.spectral.null_space.PolynomialNullSpace
    options:
      show_root_heading: true
      show_source: false

::: igl.spectral.null_space.CustomNullSpace
    options:
      show_root_heading: true
      show_source: false

::: igl.spectral.null_space.build_null_space
    options:
      show_root_heading: true
      show_source: false

## Refresh hook

::: igl.spectral.refresh.LearnedLBRefresh
    options:
      show_root_heading: true
      show_source: false
