"""Tests for ``igl.spectral.SpectralKernel``."""

import pytest
import torch

import igl
from igl import IGLConfigError
from igl.spectral import (
    ChebyshevBasis,
    ConstantNullSpace,
    FourierCosineBasis,
    FourierSineBasis,
    SpectralKernel,
)


def test_spectral_kernel_forward_shape() -> None:
    sk = SpectralKernel(latent_dim=3, bases=FourierSineBasis(n_modes=8), n_anchors=10)
    z = torch.rand(5, 3)
    out = sk(z)
    assert out.shape == (5, 10)
    assert sk.output_dim == 10


def test_spectral_kernel_with_null_space_concatenates_columns() -> None:
    sk = SpectralKernel(
        latent_dim=3,
        bases=FourierCosineBasis(n_modes=6),
        n_anchors=12,
        null_space=ConstantNullSpace(),
    )
    z = torch.rand(4, 3)
    out = sk(z)
    assert out.shape == (4, 13)
    assert sk.output_dim == 13
    # Last column should be all ones (constant).
    assert torch.all(out[:, -1] == 1.0)


def test_spectral_kernel_per_dim_bases() -> None:
    sk = SpectralKernel(
        latent_dim=2,
        bases=[FourierSineBasis(n_modes=6), ChebyshevBasis(n_modes=4)],
        n_anchors=8,
    )
    z = torch.rand(6, 2)
    out = sk(z)
    assert out.shape == (6, 8)


def test_spectral_kernel_gate_mask_neutralizes_dim() -> None:
    sk = SpectralKernel(latent_dim=2, bases=FourierSineBasis(n_modes=6), n_anchors=8)
    z = torch.rand(4, 2)
    mask = torch.tensor([1.0, 0.0])
    out_masked = sk(z, gate_mask=mask)
    # When the second dim is masked, the kernel value should equal the
    # contribution from dim 0 only — verifiable by computing it manually.
    basis = sk._bases[0]  # noqa: SLF001
    z_mapped = sk.map_to_domain(z)
    anchors_mapped = sk.map_to_domain(sk.anchor_positions)
    phi_z = basis(z_mapped[:, 0])
    phi_s = basis(anchors_mapped[:, 0])
    eigvals = basis.eigenvalues.clamp(min=1e-4)
    expected = phi_z @ (phi_s / eigvals.unsqueeze(0)).T
    torch.testing.assert_close(out_masked, expected, rtol=1e-4, atol=1e-4)


def test_spectral_kernel_rejects_invalid_dims() -> None:
    with pytest.raises(IGLConfigError, match="latent_dim"):
        SpectralKernel(latent_dim=0, bases=FourierSineBasis(n_modes=4))
    with pytest.raises(IGLConfigError, match="n_anchors"):
        SpectralKernel(latent_dim=2, bases=FourierSineBasis(n_modes=4), n_anchors=0)
    with pytest.raises(IGLConfigError, match="epsilon"):
        SpectralKernel(latent_dim=2, bases=FourierSineBasis(n_modes=4), epsilon=0.0)


def test_spectral_kernel_rejects_mismatched_per_dim_bases() -> None:
    with pytest.raises(IGLConfigError, match="length"):
        SpectralKernel(
            latent_dim=3,
            bases=[FourierSineBasis(n_modes=4), ChebyshevBasis(n_modes=4)],
        )


def test_spectral_kernel_rejects_non_module_in_sequence() -> None:
    class _Fake:
        n_modes = 4
        eigenvalues = torch.ones(4)
        null_indices = ()

        def evaluate(self, z: torch.Tensor, /) -> torch.Tensor:
            return torch.zeros(z.shape[0], 4)

    with pytest.raises(IGLConfigError, match="nn.Module"):
        SpectralKernel(latent_dim=2, bases=[_Fake(), FourierSineBasis(n_modes=4)])  # type: ignore[list-item]


def test_spectral_kernel_rejects_wrong_input_shape() -> None:
    sk = SpectralKernel(latent_dim=3, bases=FourierSineBasis(n_modes=4), n_anchors=6)
    with pytest.raises(IGLConfigError, match=r"\[N, 3\]"):
        sk(torch.rand(4, 2))


def test_spectral_kernel_trains_end_to_end() -> None:
    """End-to-end IGLModule with a SpectralKernel — at least one optimiser step works."""
    import math  # noqa: PLC0415

    torch.manual_seed(0)
    theta = torch.rand(200, 2) * 2 * math.pi
    x = torch.stack(
        [torch.cos(theta[:, 0]), torch.sin(theta[:, 0]), torch.cos(theta[:, 1]), torch.sin(theta[:, 1])],
        dim=1,
    )
    y = torch.stack(
        [torch.sin(theta[:, 0]), torch.cos(theta[:, 0]), torch.sin(theta[:, 1]), torch.cos(theta[:, 1])],
        dim=1,
    )
    sk = SpectralKernel(
        latent_dim=4,
        bases=FourierCosineBasis(n_modes=6),
        n_anchors=12,
        null_space=ConstantNullSpace(),
    )
    module = igl.IGLModule(input_dim=4, max_dim=4, output_dim=4, kernel=sk, normalize_input=False)
    trainer = igl.MatryoshkaTrainer(
        loss=igl.MSELoss(),
        config=igl.MatryoshkaConfig(
            epochs=3,
            batch_size=32,
            inner_batch_size=200,
            scheduler="none",
            early_stop_patience=None,
            verbose=False,
        ),
    )
    history = trainer.fit(module, x, y)
    assert len(history.train_loss) == 3  # noqa: PLR2004


# ----- null modes, domain map, fixed anchors -----


def test_spectral_kernel_excludes_null_modes_so_the_design_matrix_has_full_rank() -> None:
    """Flooring λ₀ at ε handed the constant a weight of 1/ε and made Φ rank 1."""
    from igl.spectral import FourierCosineBasis

    torch.manual_seed(0)
    with pytest.warns(RuntimeWarning, match="null modes"):
        sk = SpectralKernel(latent_dim=2, bases=FourierCosineBasis(n_modes=8), n_anchors=16)
    phi = sk(torch.randn(200, 2))
    singular = torch.linalg.svdvals(phi)
    # With the floor the null term was 1e4 x the rest: s1/s2 ~ 2e5 and rank 1 at 1e-3.
    assert singular[0] / singular[1] < 100  # noqa: PLR2004
    assert int(torch.linalg.matrix_rank(phi, rtol=1e-3)) >= 12  # noqa: PLR2004
    # Each factor is zero-mean on its domain: the constant is gone from the expansion.
    basis = sk._bases[0]  # noqa: SLF001
    grid = torch.linspace(0.0, 1.0, 2001)
    factor = basis(grid)[:, sk._keeps[0]]  # noqa: SLF001
    assert factor.mean(dim=0).abs().max() < 1e-2  # noqa: PLR2004


def test_spectral_kernel_null_space_silences_the_warning() -> None:
    import warnings

    from igl.spectral import ConstantNullSpace, FourierCosineBasis

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SpectralKernel(latent_dim=2, bases=FourierCosineBasis(n_modes=8), n_anchors=8, null_space=ConstantNullSpace())


@pytest.mark.parametrize("kind", ["fourier_sine", "fourier_cosine", "chebyshev", "legendre", "laguerre", "hermite"])
def test_spectral_kernel_domain_map_lands_in_the_basis_domain(kind: str) -> None:
    import warnings

    from igl.config import SpectralConfig
    from igl.spectral._build import build_spectral_kernel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sk = build_spectral_kernel(latent_dim=2, config=SpectralConfig(kind=kind, n_modes=6, n_anchors=8))  # type: ignore[arg-type]
    z = torch.randn(64, 2) * 5.0
    mapped = sk.map_to_domain(z)
    lo, hi = sk._bases[0].domain  # noqa: SLF001
    assert bool(torch.all(mapped >= lo)) and bool(torch.all(mapped <= hi))
    # Polynomial bases no longer explode on unbounded latents.
    assert torch.isfinite(sk(z)).all()
    assert sk(z).abs().max() < 1e3  # noqa: PLR2004


def test_spectral_kernel_domain_map_none_feeds_the_raw_latent() -> None:
    sk = SpectralKernel(latent_dim=2, bases=FourierSineBasis(n_modes=4), n_anchors=4, domain_map="none")
    z = torch.randn(5, 2)
    torch.testing.assert_close(sk.map_to_domain(z), z)


def test_spectral_kernel_fixed_anchors_are_a_buffer() -> None:
    anchors = torch.zeros(6, 2)
    sk = SpectralKernel(latent_dim=2, bases=FourierSineBasis(n_modes=4), anchors=anchors, learnable_anchors=False)
    assert sk.n_anchors == 6  # noqa: PLR2004
    assert not any(name == "anchor_positions" for name, _ in sk.named_parameters())
    assert sk(torch.randn(3, 2)).shape == (3, 6)


def test_spectral_kernel_graph_basis_needs_fixed_integer_anchors() -> None:
    import numpy as np

    from igl.spectral import GraphLaplacianBasis

    adjacency = np.zeros((6, 6))
    for i in range(5):
        adjacency[i, i + 1] = adjacency[i + 1, i] = 1.0
    basis = GraphLaplacianBasis(adjacency, n_modes=3)
    with pytest.raises(IGLConfigError, match="learnable_anchors=False"):
        SpectralKernel(latent_dim=1, bases=basis, n_anchors=4)
    with pytest.raises(IGLConfigError, match="node indices"):
        basis(torch.tensor([0.5, 1.0]))
    node_ids = torch.arange(6, dtype=torch.float32).unsqueeze(1)
    sk = SpectralKernel(latent_dim=1, bases=basis, anchors=node_ids, learnable_anchors=False)
    out = sk(torch.tensor([[0.0], [3.0], [5.0]]))
    assert out.shape == (3, 6)
    assert torch.isfinite(out).all()


def test_spectral_kernel_rejects_joint_basis_in_a_sequence() -> None:
    from igl.spectral import LearnedLaplacianBasis

    with pytest.raises(IGLConfigError, match="joint basis"):
        SpectralKernel(latent_dim=2, bases=[LearnedLaplacianBasis(n_modes=4), FourierSineBasis(n_modes=4)])


def test_spectral_kernel_gate_mask_never_reads_the_mask_on_the_host(mocker: object) -> None:
    import sys

    original = torch.Tensor.item
    calls: list[str] = []

    def spy(self: torch.Tensor) -> object:
        calls.append(sys._getframe(1).f_code.co_filename.rsplit("/", 1)[-1])  # noqa: SLF001
        return original(self)

    getattr(mocker, "patch").object(torch.Tensor, "item", new=spy)  # noqa: B009
    sk = SpectralKernel(latent_dim=3, bases=FourierSineBasis(n_modes=4), n_anchors=5)
    out = sk(torch.rand(6, 3), gate_mask=torch.tensor([1.0, 0.0, 1.0]))
    assert out.shape == (6, 5)
    assert "kernel.py" not in calls, calls


def test_spectral_kernel_keeps_its_mode_indices_as_non_persistent_buffers() -> None:
    from igl.spectral import FourierCosineBasis, SpectralKernel

    kernel = SpectralKernel(latent_dim=2, bases=FourierCosineBasis(n_modes=5), n_anchors=4)
    assert kernel._keep_index(0).tolist() == kernel._keeps[0]  # noqa: SLF001
    assert kernel._keep_index(0).dtype == torch.long  # noqa: SLF001
    assert not any(name.startswith("_keep_index_") for name in kernel.state_dict())
    z = torch.randn(7, 2)
    phi = kernel(z)
    assert phi.shape[0] == 7
    # Legacy checkpoints (no such buffers) still load.
    fresh = SpectralKernel(latent_dim=2, bases=FourierCosineBasis(n_modes=5), n_anchors=4)
    fresh.load_state_dict(kernel.state_dict())
    torch.testing.assert_close(fresh(z), phi)
