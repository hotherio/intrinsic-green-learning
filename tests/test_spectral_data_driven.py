"""Tests for the data-driven spectral bases (LearnedLB, GraphLaplacian) and Multi."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from igl import IGLConfigError
from igl.spectral import (
    ChebyshevBasis,
    FourierSineBasis,
    GraphLaplacianBasis,
    LearnedLaplacianBasis,
    LearnedLBRefresh,
    MultiSpectralBasis,
)


def _torus_latents(n: int = 300) -> torch.Tensor:
    torch.manual_seed(42)
    theta = torch.rand(n, 2) * 2 * math.pi
    return torch.stack(
        [torch.cos(theta[:, 0]), torch.sin(theta[:, 0]), torch.cos(theta[:, 1]), torch.sin(theta[:, 1])],
        dim=1,
    )


def test_learned_lb_refresh_sets_eigenvalues() -> None:
    z = _torus_latents(300)
    basis = LearnedLaplacianBasis(n_modes=8, k_nn=12)
    basis.refresh(z)
    assert basis.is_refreshed
    assert basis.eigenvalues.shape == (8,)
    # Eigenvalues monotone non-decreasing.
    assert torch.all(basis.eigenvalues[1:] >= basis.eigenvalues[:-1] - 1e-5)


def test_learned_lb_smallest_eigenvalue_is_near_zero() -> None:
    z = _torus_latents(300)
    basis = LearnedLaplacianBasis(n_modes=6, k_nn=12, epsilon=1e-4)
    basis.refresh(z)
    # The constant mode (λ ≈ 0) is floored to epsilon.
    assert basis.eigenvalues[0].item() <= 1e-3
    assert basis.null_indices == (0,)


def test_learned_lb_nystroem_extension_shape() -> None:
    z = _torus_latents(300)
    basis = LearnedLaplacianBasis(n_modes=6, k_nn=12)
    basis.refresh(z)
    new_z = z[:5]
    out = basis(new_z)
    assert out.shape == (5, 6)


def test_learned_lb_evaluate_before_refresh_raises() -> None:
    basis = LearnedLaplacianBasis(n_modes=6)
    with pytest.raises(IGLConfigError, match="refresh"):
        basis(torch.randn(3, 4))


def test_learned_lb_rejects_invalid_args() -> None:
    with pytest.raises(IGLConfigError, match="n_modes"):
        LearnedLaplacianBasis(n_modes=0)
    with pytest.raises(IGLConfigError, match="k_nn"):
        LearnedLaplacianBasis(n_modes=8, k_nn=0)
    with pytest.raises(IGLConfigError, match="epsilon"):
        LearnedLaplacianBasis(n_modes=8, epsilon=0.0)


def test_learned_lb_refresh_rejects_too_small_dataset() -> None:
    basis = LearnedLaplacianBasis(n_modes=4, k_nn=10)
    with pytest.raises(IGLConfigError, match="z of shape"):
        basis.refresh(torch.randn(5, 3))


def test_learned_lb_refresh_hook_runs_under_trainer() -> None:
    """LearnedLBRefresh is invoked per its `every` and refreshes the basis."""
    basis = LearnedLaplacianBasis(n_modes=4, k_nn=5)
    encoder = nn.Linear(4, 3)  # toy encoder
    x = torch.randn(40, 4)
    hook = LearnedLBRefresh(basis, every=1)
    contribution = hook(
        encoder=encoder,
        x_batch=x,
        gate_mask=torch.ones(3),
        k=3,
        epoch=0,
        batch_idx=0,
    )
    assert contribution is None  # purely scheduling
    assert basis.is_refreshed


def test_learned_lb_refresh_rejects_invalid_every() -> None:
    basis = LearnedLaplacianBasis(n_modes=4)
    with pytest.raises(IGLConfigError, match="every"):
        LearnedLBRefresh(basis, every=0)


def _path_adjacency(n: int) -> np.ndarray:
    a = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        a[i, i + 1] = 1.0
        a[i + 1, i] = 1.0
    return a


def test_graph_laplacian_path_graph_eigenvalues() -> None:
    """Path graph P_n has known eigenvalues 2(1 - cos(kπ/n))."""
    n = 12
    basis = GraphLaplacianBasis(_path_adjacency(n), n_modes=4, normalization="unnormalized")
    # First eigenvalue is 0 (path graph is connected), floored to epsilon.
    assert basis.eigenvalues[0].item() <= 1e-3
    # The rest should be monotone-non-decreasing.
    assert torch.all(basis.eigenvalues[1:] >= basis.eigenvalues[:-1])


def test_graph_laplacian_evaluate_returns_per_node_values() -> None:
    n = 8
    basis = GraphLaplacianBasis(_path_adjacency(n), n_modes=3)
    nodes = torch.arange(n)
    out = basis(nodes)
    assert out.shape == (n, 3)


def test_graph_laplacian_accepts_torch_tensor() -> None:
    n = 6
    adj = torch.from_numpy(_path_adjacency(n))
    basis = GraphLaplacianBasis(adj, n_modes=3, normalization="symmetric")
    assert basis.n_nodes == n


def test_graph_laplacian_random_walk_normalization() -> None:
    n = 8
    basis = GraphLaplacianBasis(_path_adjacency(n), n_modes=3, normalization="rw")
    assert basis.eigenvalues.shape == (3,)


def test_graph_laplacian_rejects_non_square_adjacency() -> None:
    with pytest.raises(IGLConfigError, match="square"):
        GraphLaplacianBasis(np.zeros((3, 4)), n_modes=2)


def test_graph_laplacian_rejects_invalid_args() -> None:
    adj = _path_adjacency(6)
    with pytest.raises(IGLConfigError, match="n_modes"):
        GraphLaplacianBasis(adj, n_modes=0)
    with pytest.raises(IGLConfigError, match="epsilon"):
        GraphLaplacianBasis(adj, n_modes=2, epsilon=0.0)
    with pytest.raises(IGLConfigError, match="< n_nodes"):
        GraphLaplacianBasis(adj, n_modes=10)


def test_multi_spectral_basis_concatenates_outputs() -> None:
    multi = MultiSpectralBasis([FourierSineBasis(n_modes=4), ChebyshevBasis(n_modes=3)])
    z = torch.linspace(0.05, 0.95, 6)
    out = multi(z)
    assert out.shape == (6, 7)
    assert multi.n_modes == 7  # noqa: PLR2004


def test_multi_spectral_null_indices_are_re_numbered() -> None:
    # Fourier sine: no null; Chebyshev: index 0 (constant) is the null mode.
    multi = MultiSpectralBasis([FourierSineBasis(n_modes=4), ChebyshevBasis(n_modes=3)])
    # The chebyshev null index 0 → offset 4 in the concatenated layout.
    assert multi.null_indices == (4,)


def test_multi_spectral_eigenvalues_concatenated() -> None:
    a = FourierSineBasis(n_modes=3)
    b = FourierSineBasis(n_modes=2)
    multi = MultiSpectralBasis([a, b])
    expected = torch.cat([a.eigenvalues, b.eigenvalues], dim=0)
    torch.testing.assert_close(multi.eigenvalues, expected, rtol=1e-5, atol=1e-5)


def test_multi_spectral_rejects_singleton() -> None:
    with pytest.raises(IGLConfigError, match=">= 2"):
        MultiSpectralBasis([FourierSineBasis(n_modes=4)])


def test_multi_spectral_rejects_non_module() -> None:
    class _NotModule:
        n_modes = 4
        eigenvalues = torch.ones(4)
        null_indices = ()
        domain = (0.0, 1.0)

        def evaluate(self, z: torch.Tensor, /) -> torch.Tensor:
            return torch.zeros(z.shape[0], 4)

    with pytest.raises(IGLConfigError, match="nn.Module"):
        MultiSpectralBasis([FourierSineBasis(n_modes=4), _NotModule()])  # type: ignore[list-item]


def test_multi_spectral_learnable_weights_have_gradient() -> None:
    multi = MultiSpectralBasis(
        [FourierSineBasis(n_modes=3), FourierSineBasis(n_modes=3)],
        learnable=True,
    )
    z = torch.linspace(0.05, 0.95, 5)
    out = multi(z)
    out.sum().backward()  # type: ignore[no-untyped-call]
    assert multi.log_weights.grad is not None


def test_multi_spectral_frozen_weights_are_buffer_not_parameter() -> None:
    """With ``learnable=False`` the mixing weights are a buffer."""
    multi = MultiSpectralBasis(
        [FourierSineBasis(n_modes=3), FourierSineBasis(n_modes=3)],
        learnable=False,
    )
    assert not isinstance(multi.log_weights, nn.Parameter)
    # Still evaluate-able and produces the right shape.
    out = multi(torch.linspace(0.05, 0.95, 4))
    assert out.shape == (4, 6)  # noqa: PLR2004
