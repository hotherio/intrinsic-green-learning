"""Row-chunked evaluation of the design matrix: the same matrix on every path, block by block."""

from __future__ import annotations

import pytest
import torch

from igl import GreenKernel, IGLConfigError, IGLModule
from igl.spectral import ConstantNullSpace

_MASK = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])


def _kernel(*, chunk_elements: int, **kwargs: object) -> GreenKernel:
    torch.manual_seed(0)
    return GreenKernel(latent_dim=5, n_anchors=6, n_scales=3, chunk_elements=chunk_elements, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("path", ["reference", "fast"])
@pytest.mark.parametrize("operator", ["gaussian", "helmholtz", ("gaussian", "yukawa")])
@pytest.mark.parametrize("per_scale_columns", [False, True])
def test_chunked_evaluation_equals_the_single_block(
    path: str, operator: str | tuple[str, ...], per_scale_columns: bool
) -> None:
    whole = _kernel(
        chunk_elements=2**26, operator=operator, per_scale_columns=per_scale_columns, null_space=ConstantNullSpace()
    )
    chunked = _kernel(
        chunk_elements=7 * 6 * 5, operator=operator, per_scale_columns=per_scale_columns, null_space=ConstantNullSpace()
    )
    chunked.load_state_dict(whole.state_dict())
    z = torch.randn(45, 5)  # 45 rows in blocks of 7: six full blocks and a remainder of 3
    for gate in (None, _MASK):
        expected = whole(z, gate_mask=gate, path=path)  # type: ignore[arg-type]
        got = chunked(z, gate_mask=gate, path=path)  # type: ignore[arg-type]
        assert got.shape == expected.shape
        if path == "reference":
            assert torch.equal(got, expected)  # element-wise ops: bit-identical on the CPU branch
        else:
            torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-7)


def test_chunking_keeps_gradients() -> None:
    whole = _kernel(chunk_elements=2**26)
    chunked = _kernel(chunk_elements=4 * 6 * 5)
    chunked.load_state_dict(whole.state_dict())
    z = torch.randn(10, 5)
    grads = []
    for green in (whole, chunked):
        green.zero_grad()
        green(z).square().sum().backward()
        grads.append([p.grad.clone() for p in green.parameters() if p.grad is not None])
    for a, b in zip(grads[0], grads[1], strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


def test_one_row_per_chunk_when_the_budget_is_smaller_than_a_row() -> None:
    green = _kernel(chunk_elements=1)
    z = torch.randn(3, 5)
    assert green(z).shape == (3, 6)


def test_chunk_budget_must_be_positive() -> None:
    with pytest.raises(IGLConfigError, match="chunk_elements"):
        GreenKernel(latent_dim=2, chunk_elements=0)


def test_module_forward_goes_through_the_chunked_kernel() -> None:
    torch.manual_seed(0)
    module = IGLModule(input_dim=4, max_dim=3, output_dim=1, n_anchors=5, n_scales=2)
    x = torch.randn(33, 4)
    expected = module(x)
    module.green.chunk_elements = 4 * 5 * 3
    torch.testing.assert_close(module(x), expected)
