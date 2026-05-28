# Intrinsic Green Learning

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/hotherio/intrinsic-green-learning/badge)](https://scorecard.dev/viewer/?uri=github.com/hotherio/intrinsic-green-learning)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/intrinsic-green-learning/badge)](https://www.bestpractices.dev/projects/intrinsic-green-learning)
[![REUSE compliant](https://api.reuse.software/badge/github.com/hotherio/intrinsic-green-learning)](https://api.reuse.software/info/github.com/hotherio/intrinsic-green-learning)

**Task-conditioned intrinsic-dimensionality discovery** for high-dimensional
data. IGL pairs a learned encoder with a multi-scale Green's-function
kernel and trains the system end-to-end via Variable Projection with random
Matryoshka truncation. The model fits the task and *simultaneously* reveals
how many dimensions the task actually needs — usually far fewer than the
ambient input.

> **Note on the import name.** The distribution is
> `intrinsic-green-learning`; the import name is `igl`. This collides with
> [`libigl`](https://github.com/libigl/libigl-python-bindings); if you need
> both in the same env, install one of them with a different module name.

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

The library follows the Hother Python guidelines under
[`docs/guidelines/`](docs/guidelines/):

- **basedpyright strict** type checking; `Any` is not allowed in public
  signatures.
- **`__all__` exhaustive** at every module surface.
- **Google-style docstrings** on every public symbol.
- **Single base exception** `igl.IGLError`, one level deep.
- **Conventional Commits**: commit subjects drive `python-semantic-release`
  (`feat:` → minor, `fix:` / `perf:` / `refactor:` → patch, `BREAKING
  CHANGE:` → major).
- **String-valued type aliases** are `enum.StrEnum` classes with a
  companion `Literal` mirror; public APIs accept either form.

### Release process

Releases are fully automated by
[python-semantic-release](https://python-semantic-release.readthedocs.io/)
on every push to `main` via `.github/workflows/semantic-release.yml`. See
[`docs/security.md`](docs/security.md) for the supply-chain posture
(OIDC, sigstore attestations, GPG-signed checksums, `pip-audit`).

## License

MIT. See [`LICENSE`](LICENSE) and [`REUSE.toml`](REUSE.toml).
