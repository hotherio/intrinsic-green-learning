"""Tests for ``igl.spd.orthogonality``."""

import pytest
import torch
from torch import nn

from igl import IGLConfigError
from igl.spd import (
    ConformalityPenalty,
    OrthogonalityPenalty,
    conformal_factor,
    conformality_loss,
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


def test_vectorized_jacobian_matches_the_loop_and_its_gradients() -> None:
    from igl import MLPEncoder
    from igl.spd.orthogonality import jacobian, orthogonality_loss, pullback_metric

    torch.manual_seed(0)
    encoder = MLPEncoder(7, 3, hidden=16, depth=2)
    x = torch.randn(5, 7)
    j_loop = jacobian(encoder, x, output_dim=3, vectorized=False)
    j_vmap = jacobian(encoder, x, output_dim=3, vectorized=True)
    torch.testing.assert_close(j_vmap, j_loop, rtol=1e-5, atol=1e-6)
    grads = []
    for vectorized in (False, True):
        encoder.zero_grad()
        orthogonality_loss(pullback_metric(jacobian(encoder, x, output_dim=3, vectorized=vectorized))).backward()
        grads.append([p.grad.clone() for p in encoder.parameters() if p.grad is not None])
    for a, b in zip(grads[0], grads[1], strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_vectorized_jacobian_falls_back_to_the_loop_under_training_batchnorm() -> None:
    from igl import MLPEncoder
    from igl.spd.orthogonality import _batchnorm_in_training, jacobian

    encoder = MLPEncoder(4, 2, hidden=8, depth=1, norm="batch")
    encoder.train()
    assert _batchnorm_in_training(encoder)
    x = torch.randn(6, 4)
    j = jacobian(encoder, x, output_dim=2, vectorized=True)
    assert j.shape == (6, 2, 4)
    encoder.eval()
    assert not _batchnorm_in_training(encoder)


# --- conformality -----------------------------------------------------------


def _call(penalty: ConformalityPenalty, encoder: nn.Module, x: torch.Tensor, k: int) -> torch.Tensor | None:
    mask = torch.zeros(encoder(x[:1]).shape[-1])
    mask[:k] = 1.0
    return penalty(encoder=encoder, x_batch=x, gate_mask=mask, k=k, epoch=0, batch_idx=0)


def test_conformality_loss_zero_on_a_multiple_of_the_identity() -> None:
    g = 3.0 * torch.eye(4).expand(5, 4, 4)
    assert conformality_loss(g, axis_normalize=False).item() == pytest.approx(0.0, abs=1e-7)
    assert conformality_loss(g).item() == pytest.approx(0.0, abs=1e-7)


def test_conformality_loss_known_value_on_a_diagonal_metric() -> None:
    g = torch.diag(torch.tensor([1.0, 4.0])).unsqueeze(0)
    # ‖g − 2.5 I‖² = 4.5, ‖g‖² = 17
    assert conformality_loss(g, axis_normalize=False).item() == pytest.approx(4.5 / 17, rel=1e-5)
    # a diagonal metric normalises to the identity: no spatially varying anisotropy
    assert conformality_loss(g).item() == pytest.approx(0.0, abs=1e-7)


def test_conformality_loss_is_invariant_to_global_scale_and_per_axis_rescaling() -> None:
    torch.manual_seed(0)
    a = torch.randn(6, 3, 3)
    g = a @ a.transpose(-1, -2) + 0.1 * torch.eye(3)
    base = conformality_loss(g)
    assert conformality_loss(7.0 * g).item() == pytest.approx(base.item(), rel=1e-5)
    s = torch.diag(torch.tensor([0.5, 2.0, 3.0]))
    assert conformality_loss(s @ g @ s).item() == pytest.approx(base.item(), rel=1e-5)
    # without normalisation a per-axis rescaling changes the value
    assert conformality_loss(s @ g @ s, axis_normalize=False).item() != pytest.approx(
        conformality_loss(g, axis_normalize=False).item(), rel=1e-3
    )


def test_conformality_loss_beta_zero_is_the_orthogonality_target_over_the_full_metric() -> None:
    torch.manual_seed(1)
    a = torch.randn(4, 3, 3)
    g = a @ a.transpose(-1, -2)
    diag = torch.diagonal(g, dim1=-2, dim2=-1)
    per_sample = ((g - torch.diag_embed(diag)) ** 2).sum((-2, -1)) / (g**2).sum((-2, -1))
    got = conformality_loss(g, beta=0.0, axis_normalize=False, eps=0.0).item()
    assert got == pytest.approx(per_sample.mean().item(), rel=1e-6)
    # same numerator as orthogonality_loss, which divides by the diagonal mass instead
    assert got < orthogonality_loss(g, eps=0.0).item()


def test_conformality_loss_increases_with_beta_when_the_diagonal_is_spread() -> None:
    g = torch.diag(torch.tensor([1.0, 4.0])).unsqueeze(0)
    low = conformality_loss(g, beta=0.0, axis_normalize=False).item()
    high = conformality_loss(g, beta=1.0, axis_normalize=False).item()
    assert low == pytest.approx(0.0, abs=1e-7)
    assert high > low


def test_conf_penalty_skipped_for_k_lt_2() -> None:
    encoder = nn.Linear(4, 2)
    assert _call(ConformalityPenalty(every=1), encoder, torch.randn(3, 4), k=1) is None


def test_conf_penalty_zero_for_an_orthogonal_linear_encoder() -> None:
    encoder = nn.Linear(6, 3)
    init_encoder_orthogonal_(encoder)
    out = _call(ConformalityPenalty(weight=1.0, every=1), encoder, torch.randn(8, 6), k=3)
    assert out is not None
    assert out.item() < 1e-6  # W Wᵀ = I: conformal (and isometric)


def test_conf_penalty_all_prefixes_averages_the_nested_blocks() -> None:
    torch.manual_seed(2)
    encoder = nn.Linear(5, 4)
    x = torch.randn(6, 5)
    single = ConformalityPenalty(every=1)
    nested = ConformalityPenalty(every=1, all_prefixes=True)
    j = jacobian(encoder, x, output_dim=4)[:, :4, :]
    g = pullback_metric(j)
    expected = torch.stack([conformality_loss(g[:, :kk, :kk]) for kk in (2, 3, 4)]).mean()
    out = _call(nested, encoder, x, k=4)
    assert out is not None
    assert out.item() == pytest.approx(expected.item(), rel=1e-5)
    plain = _call(single, encoder, x, k=4)
    assert plain is not None
    assert plain.item() == pytest.approx(conformality_loss(g).item(), rel=1e-5)


def test_conf_penalty_metric_inv_identity_matches_the_plain_metric() -> None:
    torch.manual_seed(3)
    encoder = nn.Linear(5, 3)
    x = torch.randn(4, 5)
    plain = _call(ConformalityPenalty(every=1), encoder, x, k=3)
    with_m = _call(
        ConformalityPenalty(every=1, metric_inv=lambda xb: torch.eye(5).expand(xb.shape[0], 5, 5)),
        encoder,
        x,
        k=3,
    )
    assert plain is not None and with_m is not None
    assert with_m.item() == pytest.approx(plain.item(), rel=1e-6)


def test_conf_penalty_metric_inv_projector_restricts_the_metric_to_the_tangent_space() -> None:
    # Encoder reads two input directions; the "manifold" is the first axis only.
    encoder = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        encoder.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))  # anisotropic over both axes
    x = torch.randn(3, 2)
    full = _call(ConformalityPenalty(every=1, axis_normalize=False), encoder, x, k=2)
    projector = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    on_manifold = _call(
        ConformalityPenalty(every=1, axis_normalize=False, metric_inv=lambda xb: projector.expand(xb.shape[0], 2, 2)),
        encoder,
        x,
        k=2,
    )
    assert full is not None and on_manifold is not None
    assert full.item() > 0.0  # diag(1, 4) is not conformal
    assert on_manifold.item() != pytest.approx(full.item())  # the projector changes the metric seen


def test_conf_penalty_gradient_reaches_the_encoder() -> None:
    torch.manual_seed(4)
    from igl import MLPEncoder

    encoder = MLPEncoder(6, 3, hidden=16, depth=2)
    out = _call(ConformalityPenalty(every=1, all_prefixes=True), encoder, torch.randn(8, 6), k=3)
    assert out is not None
    out.backward()
    grads = [p.grad for p in encoder.parameters() if p.grad is not None]
    assert grads
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_conf_penalty_rejects_bad_arguments() -> None:
    with pytest.raises(IGLConfigError, match="weight"):
        ConformalityPenalty(weight=-1.0)
    with pytest.raises(IGLConfigError, match="every"):
        ConformalityPenalty(every=0)
    with pytest.raises(IGLConfigError, match="beta"):
        ConformalityPenalty(beta=-0.1)
    with pytest.raises(IGLConfigError, match="eps"):
        ConformalityPenalty(eps=0.0)


def test_conformal_factor_matches_the_trace_and_the_isometric_case() -> None:
    torch.manual_seed(5)
    j = torch.randn(4, 3, 7)
    expected = 3.0 / torch.einsum("bij,bij->b", j, j)
    torch.testing.assert_close(conformal_factor(j), expected)
    q, _ = torch.linalg.qr(torch.randn(7, 3))
    j_iso = 2.0 * q.T.unsqueeze(0)  # rows orthonormal, scaled by 2: J Jᵀ = 4 I → λ = 1/4
    assert conformal_factor(j_iso)[0].item() == pytest.approx(0.25, rel=1e-6)
