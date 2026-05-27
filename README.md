# Intrinsic Green Learning

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/hotherio/intrinsic-green-learning/badge)](https://scorecard.dev/viewer/?uri=github.com/hotherio/intrinsic-green-learning)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/intrinsic-green-learning/badge)](https://www.bestpractices.dev/projects/intrinsic-green-learning)
[![REUSE compliant](https://api.reuse.software/badge/github.com/hotherio/intrinsic-green-learning)](https://api.reuse.software/info/github.com/hotherio/intrinsic-green-learning)

> Badges resolve after the first scorecard.yml run, after registering at https://www.bestpractices.dev/, and after `pipx run reuse lint` passes against this repo.

**Task-conditioned intrinsic-dimensionality discovery** for high-dimensional
data. IGL pairs a learned encoder with a multi-scale Green's-function kernel
and trains the system end-to-end via Variable Projection with random
Matryoshka truncation. The result is a model that simultaneously
fits the task *and* reveals how many dimensions the task actually needs.

> **Note on the import name.** The library is distributed as
> `intrinsic-green-learning` and imported as `igl`. The name collides with
> the well-known [`libigl`](https://github.com/libigl/libigl-python-bindings)
> geometry-processing package. If you need both in the same environment, see
> the install matrix below for namespace-isolation options.

## Installation

The library is in pre-release; install from source:

```bash
pip install intrinsic-green-learning
```

With optional extras:

```bash
pip install "intrinsic-green-learning[viz]"   # matplotlib plotting helpers
pip install "intrinsic-green-learning[eeg]"   # mne + moabb + pyriemann
pip install "intrinsic-green-learning[nlp]"   # transformers + datasets
pip install "intrinsic-green-learning[all]"   # everything above
```

## Quickstart

```python
import numpy as np
import igl  # see "Note on the import name" above

# coming in 0.1.0
# from igl import IGLClassifier, IGLConfig, compare_d_eff
```

The 0.1.0 milestone is the first public release; until then the surface is
limited to the exception hierarchy and version metadata.

## Documentation

Built locally with:

```bash
uv sync --group doc
uv run mkdocs serve
```

Published at https://hotherio.github.io/intrinsic-green-learning/latest/ once
the first release ships.

## Development

```bash
uv sync --all-groups
uv run lefthook install
```

Verify your environment:

```bash
uv run pytest                                    # tests + 100% coverage
uv run basedpyright src                          # strict type check
uv run lefthook run pre-commit --all-files       # full pre-commit pass
```

### Conventions

The library follows the Hother Python guidelines published under
[`docs/guidelines/`](docs/guidelines/):

- **basedpyright strict** type checking; `Any` is not allowed in public
  signatures.
- **`__all__` exhaustive** at every module surface.
- **Google-style docstrings** on every public symbol.
- **Single base exception** `igl.IGLError`, one level deep.
- **Conventional Commits**: commit subjects drive `python-semantic-release`
  (`feat:` → minor, `fix:`/`perf:`/`refactor:` → patch, `BREAKING CHANGE:` →
  major).

### Release process

Releases are fully automated by
[python-semantic-release](https://python-semantic-release.readthedocs.io/)
running on every push to `main` via `.github/workflows/semantic-release.yml`.
See [`docs/security.md`](docs/security.md) for the supply-chain posture
(OIDC, sigstore attestations, GPG-signed checksums, `pip-audit`).

## License

MIT. See [`LICENSE`](LICENSE) and [`REUSE.toml`](REUSE.toml).
