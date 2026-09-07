"""Reference vs fast Green-kernel paths: values, gradients, masks, mixed operators, null space."""

from __future__ import annotations

import pytest
import torch

from igl import GreenKernel, IGLConfigError
from igl.kernels._registry import get_operator
from igl.spectral import ConstantNullSpace

_SEPARABLE = ("gaussian", "laplacian", "yukawa")


def _reference_with_sign_tracking(green: GreenKernel, z: torch.Tensor, gate_mask: torch.Tensor | None) -> torch.Tensor:
    """The pre-change formulation (sign parity computed for every block), for the exactness test."""
    dist = z.unsqueeze(1) - green.anchor_positions.unsqueeze(0)
    sigma = torch.exp(green.log_sigma)
    gamma = torch.softmax(green.log_gamma, dim=0)
    phi = torch.zeros(z.shape[0], green.n_anchors, dtype=z.dtype)
    k_offset = 0
    for operator, k_count in zip(green._operators, green._op_counts, strict=True):  # noqa: SLF001
        sigma_op = sigma[k_offset : k_offset + k_count]
        gamma_op = gamma[k_offset : k_offset + k_count]
        log_kvals, signs = operator.fn(dist.unsqueeze(2), sigma_op.unsqueeze(0).unsqueeze(0))
        if gate_mask is not None:
            mask4 = gate_mask[None, None, None, :]
            log_kvals = log_kvals * mask4
            signs = torch.where(mask4.bool(), signs, torch.ones_like(signs))
        neg_count = (signs < 0).sum(dim=-1)
        total_sign = torch.where(neg_count % 2 == 1, -1.0, 1.0)
        prod_k = total_sign * torch.exp(log_kvals.sum(dim=-1))
        phi = phi + (prod_k * gamma_op[None, None, :]).sum(dim=-1)
        k_offset += k_count
    phi = phi * torch.sigmoid(green.rank_importance).unsqueeze(0)
    if green._null_space is not None:  # noqa: SLF001
        phi = torch.cat([phi, green._null_space.evaluate(z)], dim=-1)  # noqa: SLF001
    return phi


@pytest.mark.parametrize("operator", [*_SEPARABLE, "cauchy", "helmholtz", ("gaussian", "helmholtz")])
def test_reference_path_is_bit_identical_to_sign_tracking_everywhere(operator: str | tuple[str, ...]) -> None:
    torch.manual_seed(0)
    green = GreenKernel(latent_dim=5, n_anchors=7, n_scales=4, operator=operator, null_space=ConstantNullSpace())
    z = torch.randn(30, 5)
    mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])
    for gate in (None, mask):
        assert torch.equal(green(z, gate_mask=gate, path="reference"), _reference_with_sign_tracking(green, z, gate))


@pytest.mark.parametrize("operator", _SEPARABLE)
def test_fast_path_matches_reference_values_and_gradients(operator: str) -> None:
    torch.manual_seed(0)
    green = GreenKernel(latent_dim=6, n_anchors=9, n_scales=3, operator=operator)
    z = torch.randn(40, 6, requires_grad=True)
    mask = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    for gate in (None, mask):
        ref = green(z, gate_mask=gate, path="reference")
        fast = green(z, gate_mask=gate, path="fast")
        torch.testing.assert_close(fast, ref, rtol=1e-5, atol=1e-6)
        grads_ref = torch.autograd.grad(ref.sum(), [z, green.anchor_positions, green.log_sigma], retain_graph=True)
        grads_fast = torch.autograd.grad(fast.sum(), [z, green.anchor_positions, green.log_sigma], retain_graph=True)
        for a, b in zip(grads_fast, grads_ref, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_fast_path_mixes_separable_and_general_blocks() -> None:
    torch.manual_seed(0)
    green = GreenKernel(
        latent_dim=4, n_anchors=6, n_scales=4, operator=("gaussian", "helmholtz"), null_space=ConstantNullSpace()
    )
    z = torch.randn(20, 4)
    torch.testing.assert_close(green(z, path="fast"), green(z, path="reference"), rtol=1e-5, atol=1e-6)
    assert get_operator("helmholtz").separable is None
    assert get_operator("gaussian").separable is not None


def test_fast_path_gradcheck_in_float64() -> None:
    torch.manual_seed(0)
    green = GreenKernel(latent_dim=3, n_anchors=4, n_scales=2, operator="gaussian").double()
    z = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: green(t, path="fast"), (z,), eps=1e-6, atol=1e-5)


def test_auto_path_is_reference_on_cpu_and_rejects_unknown_names() -> None:
    green = GreenKernel(latent_dim=2, n_anchors=3, n_scales=2)
    assert green._resolve_path(torch.zeros(1, 2), None) == "reference"  # noqa: SLF001
    assert green._resolve_path(torch.zeros(1, 2), "fast") == "fast"  # noqa: SLF001
    with pytest.raises(IGLConfigError, match="kernel_path"):
        GreenKernel(latent_dim=2, n_anchors=3, n_scales=2, kernel_path="quick")  # type: ignore[arg-type]
