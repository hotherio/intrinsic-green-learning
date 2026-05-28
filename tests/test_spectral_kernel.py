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
    phi_z = basis(z[:, 0])
    phi_s = basis(sk.anchor_positions[:, 0])
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
