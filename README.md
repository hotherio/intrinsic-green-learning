# Intrinsic Green Learning

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/hotherio/intrinsic-green-learning/badge)](https://scorecard.dev/viewer/?uri=github.com/hotherio/intrinsic-green-learning)

High-dimensional inputs — pixel grids, EEG channels, embedding vectors —
almost never *use* all the dimensions they appear to. The handful that
actually matter depends on the question you ask: a binary classifier
may need only one or two latent axes, a regressor to a continuous
target may need a few more, and a full reconstruction needs whatever
dimension the data manifold genuinely has.

**Intrinsic Green Learning (IGL)** discovers that task-conditioned
effective dimension while it fits the model. A learned encoder maps the
ambient input to a low-dimensional latent space; a multi-scale
**Green's-function kernel** computes a structured design matrix on that
latent space; and **Variable Projection with random Matryoshka
truncation** trains the encoder and reads off the smallest dimension
that still solves the task. There's no separate "dimensionality
reduction" step and no fixed bottleneck — the dimension you should use
falls out of training.

The key difference from PCA, UMAP, t-SNE, or any other purely-geometric
manifold-learning method: the effective dimension IGL reports is a
property of *(input, task)*, not of the input alone. The same dataset
will resolve into different `d_eff` values for a classifier, a
regressor, and an autoencoder — and the hierarchy
$d_{\text{cls}} \le d_{\text{reg}} \le d_{\text{recon}}$ holds out of
the box.

What ships:

- **scikit-learn-compatible estimators** (`IGLClassifier`,
  `IGLRegressor`, `IGLAutoencoder`) for drop-in use in existing
  pipelines.
- **Bare PyTorch building blocks** (`IGLModule`, `GreenKernel`,
  `MatryoshkaTrainer`, …) for custom training loops, novel kernels,
  and research extensions.
- **Spectral formulation** (`SpectralKernel` + closed-form Fourier /
  Chebyshev / Legendre / Hermite / Laguerre bases, plus learned
  Laplace–Beltrami and user-supplied graph bases) with kernel-agnostic
  null-space augmentation for operators with non-trivial $\ker(L)$.
- **Riemannian / SPD extension** (`igl.spd`) for covariance-valued
  data — EEG, fMRI-derived connectivity, financial covariances — with
  an AIRM-based loss plugged in through the same `ExtraLoss` seam used
  by every other training-time regulariser.

> **Note on the import name.** The distribution is
> `intrinsic-green-learning`; the import name is `igl`. This collides
> with [`libigl`](https://github.com/libigl/libigl-python-bindings);
> if you need both in the same env, install one of them under a
> different module name.

## Why IGL?

For the same input data, a classifier usually needs fewer latent
dimensions than a regressor, which in turn needs fewer than a full
autoencoder. IGL discovers this hierarchy automatically:

$$
d_{\text{eff}}(\text{classification}) \;\le\; d_{\text{eff}}(\text{regression}) \;\le\; d_{\text{eff}}(\text{reconstruction})
$$

The library ships an `examples/synthetic/moons_xor.py` script that fits
all three estimators on the same data and reports the discovered
dimensions — the hierarchy holds out of the box.

## Installation

```bash
pip install intrinsic-green-learning
```

Optional extras:

| Extra | Adds | Use case |
|---|---|---|
| `[viz]` | matplotlib | Plot dimension curves via `igl.viz.plot_dimension_curve`. |
| `[eeg]` | mne + moabb + pyriemann | Future EEG / clinical loaders (placeholder for v0.2). |
| `[nlp]` | transformers + datasets | Future NLP loaders. |
| `[elbow]` | kneed | Alternative elbow detector. |
| `[all]` | all of the above | One-shot install for development. |

## Quickstart

The library exposes three sklearn-compatible estimators plus a SPD
extension. All accept numpy arrays at the API boundary.

### Classification

```python
import numpy as np
import igl
from igl.data import embed_in_high_dim, make_moons

x_2d, y = make_moons(400, noise=0.1, seed=0)
x = embed_in_high_dim(x_2d, target_dim=16, seed=0).numpy()

clf = igl.IGLClassifier(max_dim=8, random_state=0).fit(x, y.numpy())
print(f"accuracy = {clf.score(x, y.numpy()):.3f}")
print(f"discovered d_eff = {clf.effective_dimension_}")  # ~ 1 on moons
```

### Regression and reconstruction

```python
from igl.data import make_swiss_roll

x, params = make_swiss_roll(800, seed=0)
x_np = x.numpy(); params_np = params.numpy()

reg = igl.IGLRegressor(max_dim=8, random_state=0).fit(x_np, params_np)
ae = igl.IGLAutoencoder(max_dim=8, random_state=0).fit(x_np)

print(reg.effective_dimension_)   # ~ 2 on swiss roll (intrinsic dim)
print(ae.effective_dimension_)    # ~ 2 on swiss roll
```

### Cross-task hierarchy check

```python
report = igl.compare_d_eff(
    cls=clf.dimension_curve_,
    reg=reg.dimension_curve_,
    recon=ae.dimension_curve_,
)
print(report.d_effs)            # {'cls': 1, 'reg': 2, 'recon': 2}
print(report.hierarchy_holds)   # True
```

### SPD / Riemannian extension

For covariance-valued data (EEG, clinical signals, …), `igl.spd` ships
an AIRM-based reconstruction classifier:

```python
from igl.data import make_spd_dataset
from igl.spd import IGLReconSPDClassifier, LogEigVectorizer

spd, y = make_spd_dataset(400, d=8, n_classes=3, seed=0)
x = LogEigVectorizer().fit(spd.numpy()).transform(spd.numpy())

clf = IGLReconSPDClassifier(
    latent_dim=8, max_dim=12,
    orthogonality_weight=0.1,   # plug-in via the ExtraLoss seam
    random_state=0,
).fit(x, y.numpy())
print(clf.effective_dimension_)
```

For EEG (raw signals → covariances → AIRM), the `make_igl_airm`
factory (in the `[eeg]` extra) composes Ledoit-Wolf vs sample-cov
auto-selection with Tikhonov-preconditioned IGL-AIRM into a single
sklearn pipeline:

```python
import igl  # requires: pip install intrinsic-green-learning[eeg]

pipe = igl.make_igl_airm(latent_dim=22)
pipe.fit(X_raw, y)   # X_raw: [N, channels, time]
```

Tikhonov ε = 10⁻⁶ is applied to every input SPD by default — bit-near
identical to no preconditioning at d ≤ 64 (with a BatchNorm encoder)
and rescues `torch.linalg.eigh` from LAPACK error 8481 at d ≥ 128.

### Custom training loop

If sklearn's surface is too high-level, use the bare PyTorch entry
points directly:

```python
import torch
import igl

module = igl.IGLModule(
    input_dim=16, max_dim=8, output_dim=2,
    config=igl.IGLConfig(
        encoder=igl.EncoderConfig(hidden=(128, 64)),  # pyramidal MLP
        kernel=igl.KernelConfig(n_anchors=64, operator=igl.OperatorName.GAUSSIAN),
    ),
)

trainer = igl.MatryoshkaTrainer(
    loss=igl.CrossEntropyLoss(n_classes=2),
    config=igl.MatryoshkaConfig(epochs=500),
)
history = trainer.fit(module, x_train_t, y_train_t, x_val=x_val_t, y_val=y_val_t)
curve = igl.eval_dimension_curve(module, x_val_t, y_val_t, loss=igl.CrossEntropyLoss(n_classes=2))
print("d_eff =", igl.detect_elbow(curve))
```

## Documentation

Local build:

```bash
uv sync --group doc
uv run mkdocs serve
```

Published at <https://hotherio.github.io/intrinsic-green-learning/latest/>
after the first release.

## Examples

Three runnable scripts under `examples/synthetic/`:

| Script | Manifold | Tasks | Expected `d_eff` |
|---|---|---|---|
| `torus_classification.py` | T² ⊂ R⁴ → R³² | XOR cls + sin/cos reg | ≈ 2 |
| `moons_xor.py` | Moons ⊂ R² → R¹⁶ | cls + reg + recon | d_cls ≤ d_reg ≤ d_recon |
| `swiss_roll_recon.py` | Swiss roll ⊂ R³ | autoencoder + reg | ≈ 2 |

Run with `python -m examples.synthetic.<name>`; outputs land in
`results/<name>/<git_short_sha>/`. Install `[viz]` for PNG plots.

## Development

```bash
uv sync --all-groups
uv run lefthook install
```

Verify your environment:

```bash
uv run pytest                                # tests + 100% coverage
uv run basedpyright src                      # strict type check
uv run lefthook run pre-commit --all-files   # full pre-commit pass
```

### Conventions

Style, typing, exceptions, commit-message format, and the rest are
documented in [`CONTRIBUTING.md`](.github/CONTRIBUTING.md) and
[`docs/guidelines/`](docs/guidelines/).

### Release process

Releases are fully automated by
[python-semantic-release](https://python-semantic-release.readthedocs.io/)
on every push to `main` via `.github/workflows/semantic-release.yml`. See
[`docs/security.md`](docs/security.md) for the supply-chain posture
(OIDC, sigstore attestations, GPG-signed checksums, `pip-audit`).

## Bibliography

If you use IGL in academic work, please cite the paper this library
implements:

> Quemy, A. (2026). *Intrinsic Green's Learning: Supervised Learning on
> Manifolds via Inverse PDE*. ICLR 2026 Workshop on AI and PDE.
> <https://openreview.net/forum?id=Y6RpdS98l8>

```bibtex
@inproceedings{quemy2026igl,
  title     = {{Intrinsic Green's Learning: Supervised Learning on Manifolds via Inverse PDE}},
  author    = {Quemy, Alexandre},
  booktitle = {ICLR 2026 Workshop on AI and PDE},
  year      = {2026},
  month     = {3},
  url       = {https://openreview.net/forum?id=Y6RpdS98l8}
}
```

For a citation to this exact software version, GitHub's *"Cite this
repository"* widget reads [`CITATION.cff`](CITATION.cff); its
`preferred-citation` block points back to the paper above.

## License

MIT. See [`LICENSE`](LICENSE) and [`REUSE.toml`](REUSE.toml).
