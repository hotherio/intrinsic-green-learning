"""Opt-in readout hooks: per-scale design columns, re-anchoring and the geometry report."""

import math

import pytest
import torch
from torch import nn

from igl.config import IGLConfig, KernelConfig
from igl.core.kernel import GreenKernel, ReanchorMethod, ReanchorSigma
from igl.exceptions import IGLConfigError
from igl.nn import IGLModule
from igl.spd import geometry_report


def _pair(d: int = 3, r: int = 5, k: int = 4, **kw: object) -> tuple[GreenKernel, GreenKernel]:
    torch.manual_seed(0)
    blended = GreenKernel(d, n_anchors=r, n_scales=k, **kw)  # pyright: ignore[reportArgumentType]
    per_scale = GreenKernel(d, n_anchors=r, n_scales=k, per_scale_columns=True, **kw)  # pyright: ignore[reportArgumentType]
    per_scale.load_state_dict(blended.state_dict())
    return blended, per_scale


@pytest.mark.parametrize("path", ["reference", "fast"])
def test_per_scale_columns_sum_to_the_blended_matrix(path: str) -> None:
    blended, per_scale = _pair()
    z = torch.randn(17, 3)
    phi_b = blended(z, path=path)  # pyright: ignore[reportArgumentType]
    phi_s = per_scale(z, path=path)  # pyright: ignore[reportArgumentType]
    assert per_scale.output_dim == 20
    assert phi_s.shape == (17, 20)
    torch.testing.assert_close(phi_s.view(17, 5, 4).sum(-1), phi_b, atol=1e-6, rtol=1e-6)


def test_per_scale_columns_respect_the_gate_mask() -> None:
    blended, per_scale = _pair()
    z = torch.randn(9, 3)
    mask = torch.tensor([1.0, 0.0, 1.0])
    phi_b = blended(z, gate_mask=mask)
    phi_s = per_scale(z, gate_mask=mask)
    torch.testing.assert_close(phi_s.view(9, 5, 4).sum(-1), phi_b, atol=1e-6, rtol=1e-6)


def test_per_scale_columns_with_two_operators_keep_scale_order() -> None:
    blended, per_scale = _pair(k=5, operator=("gaussian", "laplacian"))
    z = torch.randn(6, 3)
    phi_s = per_scale(z, path="reference")
    assert per_scale.output_dim == 25
    torch.testing.assert_close(phi_s.view(6, 5, 5).sum(-1), blended(z, path="reference"), atol=1e-6, rtol=1e-6)


def test_per_scale_columns_reach_the_module_through_the_config() -> None:
    cfg = IGLConfig(max_dim=4, kernel=KernelConfig(n_anchors=6, n_scales=3, per_scale_columns=True))
    module = IGLModule(input_dim=5, max_dim=4, output_dim=2, config=cfg)
    assert module.green.output_dim == 18
    assert module.source_weights.shape[0] == 18


def test_reanchor_sample_puts_anchors_on_the_data_deterministically() -> None:
    torch.manual_seed(0)
    z = torch.randn(50, 2)
    kernels = [GreenKernel(2, n_anchors=4, n_scales=3) for _ in range(2)]
    idx = [k.reanchor_(z, generator=torch.Generator().manual_seed(1)) for k in kernels]
    assert torch.equal(idx[0], idx[1])
    torch.testing.assert_close(kernels[0].anchor_positions.detach(), z[idx[0]])
    assert len(set(idx[0].tolist())) == 4


def test_reanchor_samples_with_replacement_when_the_batch_is_small() -> None:
    k = GreenKernel(2, n_anchors=8, n_scales=2)
    idx = k.reanchor_(torch.randn(3, 2), generator=torch.Generator().manual_seed(0))
    assert idx.shape == (8,)
    assert int(idx.max()) < 3


def test_reanchor_kmeans_finds_two_clusters() -> None:
    g = torch.Generator().manual_seed(2)
    a = torch.randn(40, 2, generator=g) * 0.05 + torch.tensor([3.0, 0.0])
    b = torch.randn(40, 2, generator=g) * 0.05 + torch.tensor([-3.0, 0.0])
    z = torch.cat([a, b])
    k = GreenKernel(2, n_anchors=2, n_scales=2)
    k.reanchor_(z, method=ReanchorMethod.KMEANS, generator=torch.Generator().manual_seed(0))
    centres = k.anchor_positions.detach()
    xs = sorted(centres[:, 0].tolist())
    assert abs(xs[0] + 3.0) < 0.1
    assert abs(xs[1] - 3.0) < 0.1


def test_reanchor_median_centres_the_scale_ladder() -> None:
    torch.manual_seed(0)
    z = torch.randn(64, 4)
    k = GreenKernel(4, n_anchors=5, n_scales=3, sigma_log_range=(-1.0, 1.0))
    k.reanchor_(z, sigma=ReanchorSigma.MEDIAN, generator=torch.Generator().manual_seed(0))
    ls = k.log_sigma.detach()
    expected_middle = math.log(float(torch.pdist(z).median()))
    torch.testing.assert_close(ls[1], torch.full((4,), expected_middle), atol=1e-5, rtol=0)
    assert ls[0, 0] < ls[1, 0] < ls[2, 0]
    assert torch.allclose(ls[:, 0:1].expand(3, 4), ls)


def test_reanchor_rejects_a_wrong_width() -> None:
    k = GreenKernel(3, n_anchors=2, n_scales=2)
    with pytest.raises(IGLConfigError):
        k.reanchor_(torch.randn(5, 2))


class _Chart(nn.Module):
    input_dim: int
    max_dim: int

    def __init__(self, w: torch.Tensor) -> None:
        super().__init__()
        self.input_dim = int(w.shape[1])
        self.max_dim = int(w.shape[0])
        self.linear = nn.Linear(self.input_dim, self.max_dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_geometry_report_is_flat_on_an_orthogonal_linear_chart() -> None:
    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(6, 6))
    chart = _Chart(2.0 * q[:3])  # rows orthogonal with a common scale: conformal, not orthonormal
    x = torch.randn(32, 6)
    rep = geometry_report(chart, x, k=3, output_dim=3)
    full = rep["full"]
    assert rep["n"] == 32
    assert full["sqrt_orth"] < 1e-5
    assert full["sqrt_conf"] < 1e-5
    assert full["norm_inf"] < 1e-5
    assert full["lambda_cv"] < 1e-6
    assert full["offdiag_mass"] < 1e-10
    assert full["n_eps"] < 1.0  # a flat chart needs no Born correction
    assert abs(full["scale"] - 4.0) < 1e-5


def test_geometry_report_sees_anisotropy() -> None:
    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(5, 5))
    chart = _Chart(q[:2] * torch.tensor([[1.0], [3.0]]))
    rep = geometry_report(chart, torch.randn(16, 5), k=2, output_dim=2)
    full = rep["full"]
    assert full["sqrt_orth"] < 1e-5  # orthogonal rows
    assert full["sqrt_conf"] < 1e-5  # a constant per-axis scale is what the axis normalisation removes
    assert full["sqrt_conf_plain"] > 0.1  # the plain floor sees the unequal scales
    assert full["aniso_logstd"] > 0.5


def test_reanchor_median_keeps_the_design_matrix_alive() -> None:
    torch.manual_seed(3)
    z = torch.randn(256, 32) @ torch.diag(torch.linspace(0.5, 2.0, 32))
    kernel = GreenKernel(32, n_anchors=16, n_scales=4)
    kernel.reanchor_(z, method=ReanchorMethod.KMEANS, sigma=ReanchorSigma.MEDIAN, generator=torch.Generator().manual_seed(0))
    phi = kernel(z)
    best = phi.max(dim=1).values / torch.sigmoid(kernel.rank_importance).max()
    assert float(best.median()) > 0.2
    assert float(phi.std(dim=0).min()) > 1e-4
