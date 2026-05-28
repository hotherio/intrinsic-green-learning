"""Tests for ``igl.spd.orthogonality``."""

import pytest
import torch
from torch import nn

from igl import IGLConfigError
from igl.spd import (
    OrthogonalityPenalty,
    init_encoder_orthogonal_,
    jacobian,
    orthogonality_loss,
    pullback_metric,
)


def test_jacobian_shape_matches_input_output() -> None:
    encoder = nn.Linear(8, 4)
    x = torch.randn(5, 8)
    j = jacobian(encoder, x, output_dim=4)
    assert j.shape == (5, 4, 8)


def test_pullback_metric_is_symmetric() -> None:
    j = torch.randn(3, 4, 8)
    g = pullback_metric(j)
    assert g.shape == (3, 4, 4)
    torch.testing.assert_close(g, g.transpose(-1, -2), rtol=1e-5, atol=1e-5)


def test_orthogonality_loss_zero_when_metric_is_diagonal() -> None:
    diag_values = torch.tensor([[1.0, 2.0, 3.0]])
    g = torch.diag_embed(diag_values)
    assert orthogonality_loss(g).item() == pytest.approx(0.0, abs=1e-6)


def test_orthogonality_loss_positive_when_metric_has_off_diag() -> None:
    g = torch.tensor([[[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    assert orthogonality_loss(g).item() > 0.0


def test_init_encoder_orthogonal_counts_linear_layers() -> None:
    encoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    count = init_encoder_orthogonal_(encoder)
    assert count == 2  # noqa: PLR2004 — counting linear layers in the test fixture


def test_orth_penalty_skipped_for_k_lt_2() -> None:
    encoder = nn.Linear(4, 2)
    penalty = OrthogonalityPenalty(weight=0.1, every=1)
    out = penalty(
        encoder=encoder,
        x_batch=torch.randn(3, 4),
        gate_mask=torch.tensor([1.0, 0.0]),
        k=1,
        epoch=0,
        batch_idx=0,
    )
    assert out is None


def test_orth_penalty_returns_tensor_for_k_geq_2() -> None:
    encoder = nn.Linear(6, 4)
    init_encoder_orthogonal_(encoder)
    penalty = OrthogonalityPenalty(weight=0.1, every=1)
    out = penalty(
        encoder=encoder,
        x_batch=torch.randn(3, 6),
        gate_mask=torch.tensor([1.0, 1.0, 1.0, 0.0]),
        k=3,
        epoch=0,
        batch_idx=0,
    )
    assert isinstance(out, torch.Tensor)
    assert out.dim() == 0


def test_orth_penalty_rejects_negative_weight() -> None:
    with pytest.raises(IGLConfigError, match="weight"):
        OrthogonalityPenalty(weight=-1.0)


def test_orth_penalty_rejects_non_positive_every() -> None:
    with pytest.raises(IGLConfigError, match="every"):
        OrthogonalityPenalty(every=0)


def test_orth_penalty_near_zero_after_orthogonal_init() -> None:
    """After QR-orthogonal init the pullback metric of a single Linear should be near-diagonal."""
    encoder = nn.Linear(6, 3)
    init_encoder_orthogonal_(encoder)
    penalty = OrthogonalityPenalty(weight=1.0, every=1)
    out = penalty(
        encoder=encoder,
        x_batch=torch.randn(8, 6),
        gate_mask=torch.tensor([1.0, 1.0, 1.0]),
        k=3,
        epoch=0,
        batch_idx=0,
    )
    assert out is not None
    # A QR-orthogonal Linear maps to orthogonal columns; J = W and W W^T ≈ I,
    # so off-diagonals should be small.
    assert out.item() < 1e-6
