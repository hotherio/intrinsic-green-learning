"""End-to-end tests: GreenKernel.null_space, IGLModule(kernel=...), sklearn wrappers."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

import igl
from igl import (
    GreenKernel,
    IGLClassifier,
    IGLConfig,
    IGLConfigError,
    IGLModule,
    KernelConfig,
    MatryoshkaConfig,
    SpectralConfig,
)
from igl.spectral import (
    ConstantNullSpace,
    FourierCosineBasis,
    PolynomialNullSpace,
    SpectralKernel,
)


def test_green_kernel_null_space_adds_columns() -> None:
    gk = GreenKernel(latent_dim=4, n_anchors=12, n_scales=2, null_space=ConstantNullSpace())
    assert gk.output_dim == 13
    z = torch.randn(5, 4)
    out = gk(z)
    assert out.shape == (5, 13)


def test_green_kernel_null_space_polynomial() -> None:
    null = PolynomialNullSpace(latent_dim=3, degree=1)
    gk = GreenKernel(latent_dim=3, n_anchors=8, n_scales=2, null_space=null)
    assert gk.output_dim == 8 + null.n_columns
    z = torch.randn(4, 3)
    out = gk(z)
    assert out.shape == (4, 8 + null.n_columns)


def test_igl_module_accepts_pre_built_spectral_kernel() -> None:
    sk = SpectralKernel(latent_dim=4, bases=FourierCosineBasis(n_modes=6), n_anchors=8)
    module = IGLModule(input_dim=4, max_dim=4, output_dim=2, kernel=sk, normalize_input=False)
    assert module.source_weights.shape == (8, 2)


def test_igl_module_accepts_pre_built_green_kernel_with_null_space() -> None:
    gk = GreenKernel(latent_dim=3, n_anchors=10, n_scales=2, null_space=ConstantNullSpace())
    module = IGLModule(input_dim=4, max_dim=3, output_dim=2, kernel=gk, normalize_input=False)
    assert module.source_weights.shape == (11, 2)


def test_igl_module_rejects_kernel_without_output_dim() -> None:
    class _BadKernel(torch.nn.Module):
        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return z

    with pytest.raises(IGLConfigError, match="output_dim"):
        IGLModule(input_dim=4, max_dim=3, output_dim=2, kernel=_BadKernel())


def test_classifier_with_spectral_config() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    x = np.random.RandomState(0).randn(60, 6).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    config = IGLConfig(
        max_dim=4,
        spectral=SpectralConfig(kind="fourier_cosine", n_modes=6, n_anchors=10, null_space="constant"),
        matryoshka=MatryoshkaConfig(
            epochs=4,
            batch_size=20,
            inner_batch_size=60,
            scheduler="none",
            early_stop_patience=None,
            verbose=False,
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        clf = IGLClassifier(max_dim=4, random_state=0, config=config).fit(x, y)
    assert clf.module_.source_weights.shape[0] == 11  # 10 anchors + 1 null  # noqa: PLR2004


def test_classifier_with_local_kernel_null_space_config() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    x = np.random.RandomState(0).randn(60, 6).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    config = IGLConfig(
        max_dim=4,
        kernel=KernelConfig(n_anchors=12, n_scales=2, null_space="constant"),
        matryoshka=MatryoshkaConfig(
            epochs=3,
            batch_size=20,
            inner_batch_size=60,
            scheduler="none",
            early_stop_patience=None,
            verbose=False,
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        clf = IGLClassifier(max_dim=4, random_state=0, config=config).fit(x, y)
    assert clf.module_.source_weights.shape[0] == 13  # 12 + 1  # noqa: PLR2004


def test_spectral_config_round_trip_through_to_dict() -> None:
    original = IGLConfig(
        max_dim=8,
        spectral=SpectralConfig(
            kind=(
                "fourier_sine",
                "chebyshev",
                "fourier_sine",
                "chebyshev",
                "fourier_sine",
                "chebyshev",
                "fourier_sine",
                "chebyshev",
            ),
            n_modes=12,
            n_anchors=32,
            null_space="polynomial",
            polynomial_degree=2,
        ),
    )
    data = original.to_dict()
    rebuilt = IGLConfig.from_dict(data)
    assert rebuilt.spectral is not None
    assert rebuilt.spectral.n_modes == 12  # noqa: PLR2004
    assert rebuilt.spectral.n_anchors == 32  # noqa: PLR2004


def test_kernel_config_null_space_round_trip() -> None:
    original = IGLConfig(kernel=KernelConfig(null_space="constant"))
    data = original.to_dict()
    rebuilt = IGLConfig.from_dict(data)
    assert rebuilt.kernel.null_space == "constant"


def test_spectral_config_default_serialises_to_none() -> None:
    cfg = IGLConfig()
    data = cfg.to_dict()
    assert data["spectral"] is None


def test_spectral_config_rejects_unsupported_basis_in_factory() -> None:
    """``GRAPH_LAPLACIAN`` requires a user-supplied adjacency — config can't build it."""
    config = IGLConfig(
        max_dim=4,
        spectral=SpectralConfig(kind="graph_laplacian", n_anchors=8),
    )
    x = np.random.RandomState(0).randn(20, 4).astype(np.float32)
    y = np.random.RandomState(0).randint(0, 2, 20)
    clf = IGLClassifier(max_dim=4, random_state=0, config=config)
    with pytest.raises(IGLConfigError, match="GRAPH_LAPLACIAN"):
        clf.fit(x, y)


def test_spectral_per_dim_kind_mismatch_raises() -> None:
    config = IGLConfig(
        max_dim=4,
        spectral=SpectralConfig(kind=("fourier_sine", "chebyshev"), n_anchors=8),
    )
    x = np.random.RandomState(0).randn(20, 4).astype(np.float32)
    y = np.random.RandomState(0).randint(0, 2, 20)
    clf = IGLClassifier(max_dim=4, random_state=0, config=config)
    with pytest.raises(IGLConfigError, match="per-dim"):
        clf.fit(x, y)


def test_iglconfig_from_dict_rejects_non_mapping_spectral() -> None:
    with pytest.raises(IGLConfigError, match="spectral must be a mapping"):
        IGLConfig.from_dict({"spectral": "bad"})


def test_iglconfig_from_dict_with_string_spectral_kind() -> None:
    """Single-string ``kind`` branch in :func:`_make_spectral_config`."""
    data: dict[str, object] = {"spectral": {"kind": "chebyshev", "n_modes": 4, "n_anchors": 6}}
    rebuilt = IGLConfig.from_dict(data)
    assert rebuilt.spectral is not None
    assert rebuilt.spectral.kind == "chebyshev"


def test_iglconfig_from_dict_with_tuple_spectral_kind() -> None:
    """Tuple ``kind`` branch (rare; JSON would normally give a list)."""
    data: dict[str, object] = {
        "spectral": {"kind": ("fourier_sine", "chebyshev"), "n_anchors": 8},
    }
    rebuilt = IGLConfig.from_dict(data)
    assert rebuilt.spectral is not None
    assert rebuilt.spectral.kind == ("fourier_sine", "chebyshev")


def test_iglconfig_from_dict_rejects_bad_spectral_kind_type() -> None:
    data: dict[str, object] = {"spectral": {"kind": 42, "n_anchors": 6}}
    with pytest.raises(IGLConfigError, match="spectral.kind"):
        IGLConfig.from_dict(data)


@pytest.mark.parametrize(
    "kind",
    ["fourier_sine", "fourier_cosine", "chebyshev", "legendre", "hermite", "laguerre"],
)
def test_spectral_config_each_closed_form_basis(kind: str) -> None:
    """Every closed-form basis can be built via SpectralConfig."""
    from igl.spectral._build import build_spectral_kernel  # noqa: PLC0415

    kernel = build_spectral_kernel(
        latent_dim=3,
        config=SpectralConfig(kind=kind, n_modes=4, n_anchors=6),
    )
    assert kernel.output_dim == 6  # noqa: PLR2004


def test_spectral_config_learned_lb_can_be_built() -> None:
    from igl.spectral._build import build_spectral_kernel  # noqa: PLC0415

    kernel = build_spectral_kernel(
        latent_dim=3,
        config=SpectralConfig(kind="learned_lb", n_modes=4, n_anchors=6),
    )
    assert kernel.output_dim == 6  # noqa: PLR2004


def test_spectral_config_per_dim_kind_matching_latent_dim() -> None:
    from igl.spectral._build import build_spectral_kernel  # noqa: PLC0415

    kernel = build_spectral_kernel(
        latent_dim=2,
        config=SpectralConfig(kind=("fourier_sine", "chebyshev"), n_modes=6, n_anchors=8),
    )
    assert kernel.output_dim == 8  # noqa: PLR2004


def test_spectral_kind_list_round_trip_via_to_dict() -> None:
    """Round-trip a per-dim kinds tuple through to_dict / from_dict."""
    cfg = IGLConfig(
        max_dim=2,
        spectral=SpectralConfig(kind=("fourier_sine", "chebyshev"), n_modes=4, n_anchors=6),
    )
    rebuilt = IGLConfig.from_dict(cfg.to_dict())
    assert rebuilt.spectral is not None
    assert rebuilt.spectral.kind == ("fourier_sine", "chebyshev")


def test_graph_laplacian_basis_from_scipy_sparse() -> None:
    """``_to_sparse`` accepts scipy sparse matrices directly."""
    import scipy.sparse  # noqa: PLC0415

    n = 8
    rows = list(range(n - 1)) + list(range(1, n))
    cols = list(range(1, n)) + list(range(n - 1))
    data = [1.0] * (2 * (n - 1))
    adj = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    from igl.spectral import GraphLaplacianBasis  # noqa: PLC0415

    basis = GraphLaplacianBasis(adj, n_modes=3)
    assert basis.n_nodes == n


def test_null_space_in_lstsq_recovers_constant_target() -> None:
    """A constant target should be fit exactly via the null-space column."""
    torch.manual_seed(0)
    x = torch.randn(60, 4)
    y_const = torch.full((60, 1), 3.14)

    gk = GreenKernel(latent_dim=4, n_anchors=8, n_scales=2, null_space=ConstantNullSpace())
    phi = gk(x)
    weights = igl.direct_solve_weights(phi, y_const, l2=1e-3)
    pred = phi @ weights
    assert (pred - y_const).abs().max().item() < 0.05  # noqa: PLR2004
