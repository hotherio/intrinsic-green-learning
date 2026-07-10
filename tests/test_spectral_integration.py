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


# --- Config path: IGLModule must honour every KernelConfig field ---------------
#
# When no `kernel=` is passed, IGLModule builds its own GreenKernel. That branch
# used to drop null_space, polynomial_degree, sigma_log_range and
# anchor_init_std silently (issue #50).


def _config_module(**kernel_kwargs: object) -> IGLModule:
    """Build an IGLModule through the `config=` path only (no `kernel=`)."""
    cfg = IGLConfig(max_dim=4, kernel=KernelConfig(n_anchors=8, n_scales=2, **kernel_kwargs))  # type: ignore[arg-type]
    return IGLModule(input_dim=3, max_dim=4, output_dim=1, config=cfg)


def test_igl_module_config_null_space_polynomial_widens_design_matrix() -> None:
    """The reproducer from issue #50: 8 anchors + (1 + max_dim) null columns."""
    module = _config_module(null_space="polynomial", polynomial_degree=1)
    assert module.design_matrix(torch.randn(64, 3)).shape[1] == 8 + 1 + 4


def test_igl_module_config_null_space_degree_2() -> None:
    """Degree 2 contributes ``1 + 2 * max_dim`` columns."""
    module = _config_module(null_space="polynomial", polynomial_degree=2)
    assert module.green.output_dim == 8 + 1 + 2 * 4


def test_igl_module_config_null_space_constant() -> None:
    """The constant null space contributes exactly one DC column."""
    module = _config_module(null_space="constant")
    assert module.green.output_dim == 8 + 1


def test_igl_module_config_null_space_none_is_the_default() -> None:
    """`NONE` stays the default and adds no columns — guards the SPD contract."""
    assert KernelConfig().null_space is igl.NullSpaceKind.NONE
    assert _config_module().green.output_dim == 8


def test_igl_module_config_forwards_sigma_log_range() -> None:
    module = _config_module(sigma_log_range=(-9.0, -8.0))
    log_sigma = module.green.log_sigma.detach()
    assert float(log_sigma.min()) >= -9.0
    assert float(log_sigma.max()) <= -8.0


def test_igl_module_config_forwards_anchor_init_std() -> None:
    torch.manual_seed(0)
    wide = _config_module(anchor_init_std=50.0).green.anchor_positions.detach()
    torch.manual_seed(0)
    narrow = _config_module(anchor_init_std=0.5).green.anchor_positions.detach()
    # Same seed, same draws — the config value only rescales them.
    torch.testing.assert_close(wide, narrow * 100.0, rtol=1e-5, atol=1e-4)


def test_igl_module_explicit_kernel_overrides_config_null_space() -> None:
    """A pre-built `kernel=` wins over `config.kernel.null_space`."""
    gk = GreenKernel(latent_dim=4, n_anchors=10, n_scales=2, null_space=ConstantNullSpace())
    cfg = IGLConfig(max_dim=4, kernel=KernelConfig(n_anchors=8, n_scales=2, null_space="polynomial"))
    module = IGLModule(input_dim=3, max_dim=4, output_dim=1, config=cfg, kernel=gk)
    assert module.green.output_dim == 10 + 1  # the kernel's, not the config's


def test_igl_module_default_config_is_rng_and_shape_identical() -> None:
    """Forwarding ``KernelConfig()`` defaults must be a strict no-op.

    ``IGLReconSPDClassifier`` builds ``IGLModule`` with neither ``config=`` nor
    ``kernel=``, so it relies on the self-built branch. If forwarding the config
    defaults moved a single torch RNG draw, the EEG bit-exact reproducibility
    contract in ``test_spd_reproducibility.py`` would silently break.
    """

    def state(**kwargs: object) -> dict[str, torch.Tensor]:
        torch.manual_seed(1234)
        return IGLModule(input_dim=6, max_dim=8, output_dim=3, **kwargs).state_dict()  # type: ignore[arg-type]

    bare = state()
    with_default_config = state(config=IGLConfig(max_dim=8))

    assert bare.keys() == with_default_config.keys()
    for name, tensor in bare.items():
        assert torch.equal(tensor, with_default_config[name]), name


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
