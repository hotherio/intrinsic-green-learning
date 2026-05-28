"""1-D spectral bases for the spectral IGL kernel.

Each basis is an ``nn.Module`` satisfying :class:`igl.types.SpectralBasis`.
"""

from igl.spectral.bases.chebyshev import ChebyshevBasis
from igl.spectral.bases.fourier_cosine import FourierCosineBasis
from igl.spectral.bases.fourier_sine import FourierSineBasis
from igl.spectral.bases.graph_laplacian import GraphLaplacianBasis
from igl.spectral.bases.hermite import HermiteBasis
from igl.spectral.bases.laguerre import LaguerreBasis
from igl.spectral.bases.learned_lb import LearnedLaplacianBasis
from igl.spectral.bases.legendre import LegendreBasis

__all__ = [
    "ChebyshevBasis",
    "FourierCosineBasis",
    "FourierSineBasis",
    "GraphLaplacianBasis",
    "HermiteBasis",
    "LaguerreBasis",
    "LearnedLaplacianBasis",
    "LegendreBasis",
]
