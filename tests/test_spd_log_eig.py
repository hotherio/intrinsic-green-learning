"""Tests for ``igl.spd.LogEigVectorizer``."""

import numpy as np

from igl.spd import LogEigVectorizer


def _random_spd_np(batch: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((batch, d, d))
    return np.asarray(a @ np.swapaxes(a, -1, -2) + d * np.eye(d), dtype=np.float64)


def test_log_eig_output_shape() -> None:
    x = _random_spd_np(5, 4)
    out = LogEigVectorizer().fit(x).transform(x)
    assert out.shape == (5, 4 * 5 // 2)


def test_log_eig_identity_maps_to_zero_vector() -> None:
    x = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    out = LogEigVectorizer().fit(x).transform(x)
    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_log_eig_preserves_frobenius_norm() -> None:
    """Off-diagonal entries get a √2 boost so ``‖vec‖₂ = ‖log(X)‖_F``."""
    x = _random_spd_np(4, 5, seed=1)
    vec = LogEigVectorizer().fit(x).transform(x)
    # Compute log(X) ourselves and check its Frobenius norm matches ‖vec‖.
    vals, vecs = np.linalg.eigh(x)
    log_x = (vecs * np.log(np.clip(vals, 1e-8, None))[:, None, :]) @ np.swapaxes(vecs, -1, -2)
    fro = np.linalg.norm(log_x.reshape(4, -1), axis=1)
    l2 = np.linalg.norm(vec, axis=1)
    np.testing.assert_allclose(fro, l2, rtol=1e-4)


def test_log_eig_records_size_and_indices() -> None:
    x = _random_spd_np(2, 6)
    transformer = LogEigVectorizer().fit(x)
    assert transformer.n_features_in_ == 6
    assert transformer.triu_idx_[0].shape == transformer.triu_idx_[1].shape
