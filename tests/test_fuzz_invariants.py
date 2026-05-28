"""Property-based (Hypothesis) tests for IGL's numerical-core invariants.

These tests complement the unit suite by sampling input shapes and parameter
combinations the hand-written tests don't cover. They check structural
properties (shapes, finiteness, monotonicity, sign) rather than specific
numerical values, so they remain stable across refactors of the underlying
operators.

Profiles
--------
- Default (local/CI): bounded examples (``max_examples=30``) keep the suite
  under a couple seconds while still surfacing shape/finiteness regressions.
- ``--hypothesis-profile=nightly``: enables ``max_examples=200`` for deeper
  search. Used by ``.github/workflows/fuzz.yml`` on the weekly schedule.
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import igl
from igl import GreenKernel, IGLModule, MSELoss, direct_solve_weights
from igl.spectral import (
    ChebyshevBasis,
    ConstantNullSpace,
    FourierCosineBasis,
    FourierSineBasis,
    PolynomialNullSpace,
    SpectralKernel,
)

settings.register_profile(
    "ci",
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "nightly",
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("ci")


# --- Strategies ---------------------------------------------------------------

_latent_dims = st.integers(min_value=1, max_value=6)
_n_anchors = st.integers(min_value=2, max_value=32)
_n_scales = st.integers(min_value=1, max_value=8)
_polynomial_degrees = st.integers(min_value=0, max_value=3)
_n_samples = st.integers(min_value=2, max_value=128)


# --- GreenKernel output_dim invariant -----------------------------------------


@given(
    latent_dim=_latent_dims,
    n_anchors=_n_anchors,
    n_scales=_n_scales,
    add_null=st.booleans(),
    degree=_polynomial_degrees,
)
def test_green_kernel_output_dim_equals_anchors_plus_null_columns(
    latent_dim: int,
    n_anchors: int,
    n_scales: int,
    add_null: bool,
    degree: int,
) -> None:
    """``GreenKernel.output_dim`` is exactly ``n_anchors + null.n_columns``."""
    null = PolynomialNullSpace(latent_dim=latent_dim, degree=degree) if add_null else None
    kernel = GreenKernel(
        latent_dim=latent_dim,
        n_anchors=n_anchors,
        n_scales=n_scales,
        null_space=null,
    )
    expected = n_anchors + (null.n_columns if null is not None else 0)
    assert kernel.output_dim == expected
    # And the design matrix actually has that width.
    z = torch.randn(7, latent_dim) * 0.3
    phi = kernel(z)
    assert phi.shape == (7, expected)


# --- SpectralKernel finiteness on basis support -------------------------------


@given(
    n_modes=st.integers(min_value=2, max_value=12),
    n_anchors=st.integers(min_value=2, max_value=16),
    n_samples=_n_samples,
)
def test_spectral_kernel_fourier_sine_is_finite(
    n_modes: int,
    n_anchors: int,
    n_samples: int,
) -> None:
    """``SpectralKernel(FourierSineBasis)`` produces finite values on [0, 1]."""
    sk = SpectralKernel(
        latent_dim=2,
        bases=FourierSineBasis(n_modes=n_modes),
        n_anchors=n_anchors,
    )
    z = torch.rand(n_samples, 2)  # Fourier sine domain is [0, 1]
    out = sk(z)
    assert out.shape == (n_samples, n_anchors)
    assert torch.isfinite(out).all()


@given(
    n_modes=st.integers(min_value=2, max_value=10),
    n_samples=_n_samples,
)
def test_chebyshev_basis_evaluate_is_finite_on_domain(
    n_modes: int,
    n_samples: int,
) -> None:
    """Chebyshev basis returns finite values for inputs in [-1, 1]."""
    basis = ChebyshevBasis(n_modes=n_modes)
    z = torch.rand(n_samples) * 2 - 1  # [-1, 1]
    out = basis(z)
    assert out.shape == (n_samples, n_modes)
    assert torch.isfinite(out).all()


# --- direct_solve_weights L2 monotonicity -------------------------------------


@given(
    n_samples=st.integers(min_value=8, max_value=64),
    n_features=st.integers(min_value=2, max_value=16),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_direct_solve_weights_l2_increases_monotonically_shrinks_norm(
    n_samples: int,
    n_features: int,
    seed: int,
) -> None:
    """Increasing the Tikhonov coefficient never grows the weight norm.

    Mathematically: ``w(l2) = (XᵀX + n*l2*I)⁻¹ Xᵀy``; ``‖w‖`` is
    non-increasing in ``l2``. This is a structural property of ridge
    regression that should hold for any well-conditioned input.
    """
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(n_samples, n_features, generator=g)
    y = torch.randn(n_samples, 1, generator=g)

    w_small = direct_solve_weights(phi, y, l2=1e-4)
    w_med = direct_solve_weights(phi, y, l2=1e-2)
    w_large = direct_solve_weights(phi, y, l2=1.0)

    n_small = float(torch.linalg.norm(w_small).item())
    n_med = float(torch.linalg.norm(w_med).item())
    n_large = float(torch.linalg.norm(w_large).item())

    # Allow a tiny numerical slack — the ridge solution is monotone in
    # regularization but lstsq's solver can wobble at the ~1e-6 level.
    slack = 1e-4 * max(n_small, 1.0)
    assert n_med <= n_small + slack
    assert n_large <= n_med + slack
    # And the heavily-regularised norm is genuinely smaller than the weakly-
    # regularised one when there's signal to shrink.
    if n_small > 1e-2:
        assert n_large < n_small


# --- LossStrategy.curve_score sign --------------------------------------------


@given(
    n_samples=st.integers(min_value=2, max_value=64),
    n_dims=st.integers(min_value=1, max_value=8),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_mse_curve_score_is_non_negative(n_samples: int, n_dims: int, seed: int) -> None:
    """``MSELoss.curve_score`` (a squared error) is non-negative for any inputs."""
    g = torch.Generator().manual_seed(seed)
    pred = torch.randn(n_samples, n_dims, generator=g)
    target = torch.randn(n_samples, n_dims, generator=g)
    score = MSELoss().curve_score(pred, target)
    assert math.isfinite(score)
    assert score >= 0.0


# --- MatryoshkaConfig accepts valid hyperparameter triples --------------------


@given(
    epochs=st.integers(min_value=1, max_value=50),
    batch_size=st.integers(min_value=1, max_value=512),
    inner_batch_size=st.integers(min_value=1, max_value=2048),
)
def test_matryoshka_config_accepts_any_positive_triple(
    epochs: int,
    batch_size: int,
    inner_batch_size: int,
) -> None:
    """``MatryoshkaConfig`` constructs successfully for any positive triple."""
    cfg = igl.MatryoshkaConfig(
        epochs=epochs,
        batch_size=batch_size,
        inner_batch_size=inner_batch_size,
        scheduler="none",
        early_stop_patience=None,
        verbose=False,
    )
    # Frozen-slotted dataclass exposes the constructor values verbatim.
    assert cfg.epochs == epochs
    assert cfg.batch_size == batch_size
    assert cfg.inner_batch_size == inner_batch_size
    # And the full IGLConfig round-trip preserves them.
    wrapped = igl.IGLConfig(matryoshka=cfg)
    rt = igl.IGLConfig.from_dict(wrapped.to_dict())
    assert rt.matryoshka.epochs == epochs
    assert rt.matryoshka.batch_size == batch_size
    assert rt.matryoshka.inner_batch_size == inner_batch_size


# --- IGLModule.forward finiteness end-to-end ----------------------------------


@given(
    latent_dim=st.integers(min_value=1, max_value=4),
    n_anchors=st.integers(min_value=2, max_value=16),
    n_scales=st.integers(min_value=1, max_value=4),
    n_samples=st.integers(min_value=2, max_value=32),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_igl_module_forward_is_finite_on_normalized_input(
    latent_dim: int,
    n_anchors: int,
    n_scales: int,
    n_samples: int,
    seed: int,
) -> None:
    """``IGLModule(GreenKernel(...))(x)`` returns finite outputs for normalised input."""
    torch.manual_seed(seed)
    kernel = GreenKernel(
        latent_dim=latent_dim,
        n_anchors=n_anchors,
        n_scales=n_scales,
        null_space=ConstantNullSpace(),
    )
    module = IGLModule(
        input_dim=latent_dim,
        max_dim=latent_dim,
        output_dim=2,
        kernel=kernel,
        normalize_input=True,
    )
    x = torch.randn(n_samples, latent_dim)
    y = module(x)
    assert y.shape == (n_samples, 2)
    assert torch.isfinite(y).all()


# --- SpectralKernel + null space output_dim invariant -------------------------


@given(
    latent_dim=st.integers(min_value=1, max_value=3),
    n_modes=st.integers(min_value=2, max_value=8),
    n_anchors=st.integers(min_value=2, max_value=12),
    use_polynomial=st.booleans(),
    degree=st.integers(min_value=0, max_value=2),
)
def test_spectral_kernel_output_dim_matches_anchors_plus_null(
    latent_dim: int,
    n_modes: int,
    n_anchors: int,
    use_polynomial: bool,
    degree: int,
) -> None:
    """Same invariant as GreenKernel but for SpectralKernel + null space."""
    null = PolynomialNullSpace(latent_dim=latent_dim, degree=degree) if use_polynomial else ConstantNullSpace()
    sk = SpectralKernel(
        latent_dim=latent_dim,
        bases=FourierCosineBasis(n_modes=n_modes),
        n_anchors=n_anchors,
        null_space=null,
    )
    expected = n_anchors + null.n_columns
    assert sk.output_dim == expected
    z = torch.rand(5, latent_dim)
    assert sk(z).shape == (5, expected)


if __name__ == "__main__":  # pragma: no cover  # ad-hoc local invocation
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
