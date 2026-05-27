# CHANGE-ME | Python Library Template

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/hotherio/CHANGE-ME/badge)](https://scorecard.dev/viewer/?uri=github.com/hotherio/CHANGE-ME)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/CHANGE-ME/badge)](https://www.bestpractices.dev/projects/CHANGE-ME)
[![REUSE compliant](https://api.reuse.software/badge/github.com/hotherio/CHANGE-ME)](https://api.reuse.software/info/github.com/hotherio/CHANGE-ME)

> Badges resolve after the first scorecard.yml run, after registering at https://www.bestpractices.dev/ (paste your project URL, fill the questionnaire to "passing"), and after `pipx run reuse lint` passes against this repo.

## Installation



## Documentation

To build and serve the documentation locally:

1. Install the dependencies:
```
uv sync --group doc
source .venv/bin/activate
```

2. Serve the documentation:
```
mkdocs serve
```

## Development

### Installation

The only command that should be necessary is:
```
uv sync --group dev
source .venv/bin/activate
lefthook install
```

It creates a virtual environment, install all dependencies required for development and install the library in editable mode.
It also installs the Lefthook git hooks manager.

### Git Hooks with Lefthook

This project uses Lefthook for managing git hooks. Hooks are automatically installed when you run `make install-dev`.

To run hooks manually:
```
# Run all pre-commit hooks
lefthook run pre-commit

# Run specific hook
lefthook run pre-commit --commands ruff-check

# Skip hooks for a single commit
git commit --no-verify -m "emergency fix"
```

For local customization, copy `.lefthook-local.yml.example` to `.lefthook-local.yml` and modify as needed.


### Tests

uv run -m pytest

### Coverage

uv run python -m pytest src --cov=hother

### Building the package

```
uv build
```

### Release process

Releases are fully automated by [python-semantic-release](https://python-semantic-release.readthedocs.io/) running on every push to `main` via `.github/workflows/semantic-release.yml`. The workflow:

1. Analyzes conventional-commit history since the last tag (`feat:` → minor, `fix:`/`perf:`/`refactor:` → patch, `BREAKING CHANGE:` → major).
2. Updates `version` in `pyproject.toml` and writes the entry to `CHANGELOG.md`.
3. Commits the bump as `chore(release): bump to <version>` (signed with the Hother Bot GPG key) and pushes the matching `v<version>` tag.
4. Builds the wheel + sdist, publishes to PyPI via OIDC **Trusted Publishing** (no API token), generates `SHA256SUMS` + GPG signature, creates a GitHub Release with signed assets and git-cliff-rendered release notes, then attests build provenance.
5. Dispatches a `new_release` event to the private package registry.

#### Preview the next release locally

```bash
# Compute what version PSR would bump to (no side effects)
uvx python-semantic-release version --noop --print-version

# Preview the changelog entry PSR would write
uvx python-semantic-release changelog --noop

# Preview unreleased changes via git-cliff (same source the release-notes step uses)
make changelog-unreleased
```

#### Trigger a release manually

The workflow can be triggered via the GitHub UI ("Run workflow" on `Semantic Release`) or `gh workflow run semantic-release.yml`. This is useful when you want to release without pushing new commits.

#### Skip a release

PSR only releases when there's a relevant commit. Push only `chore:`/`docs:`/`style:`/`test:`/`ci:`/`build:` commits to land changes without bumping the version.

### Changelog Management

This project uses [git-cliff](https://git-cliff.org/) to automatically generate changelogs from conventional commits.

```
# Generate/update CHANGELOG.md
make changelog

# Preview unreleased changes
make changelog-unreleased

# Get changelog for latest tag (used in releases)
make changelog-tag
```

The changelog is automatically updated and included in GitHub releases when you push a version tag.

Generate the licenses:
```
uvx pip-licenses --from=mixed --order count -f md --output-file licenses.md
uvx pip-licenses --from=mixed --order count -f csv --output-file licenses.csv
```

Build the new documentation:
```
uv run mike deploy --push --update-aliases <version> latest
mike set-default latest
mike list
```
Checking the documentation locally
```
mike serve
```


## Development practices

### Branching & Pull-Requests

Each git branch should have the format `<tag>/item_<id>` with eventually a descriptive suffix.

We us a **Squash & Merge** approach.

### Conventional Commits

We use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

Format: `<type>(<scope>): <subject>`

`<scope>` is optional

#### Example

```
feat: add hat wobble
^--^  ^------------^
|     |
|     +-> Summary in present tense.
|
+-------> Type: chore, docs, feat, fix, refactor, style, or test.
```

More Examples:

- `feat`: (new feature for the user, not a new feature for build script)
- `fix`: (bug fix for the user, not a fix to a build script)
- `docs`: (changes to the documentation)
- `style`: (formatting, missing semi colons, etc; no production code change)
- `refactor`: (refactoring production code, eg. renaming a variable)
- `test`: (adding missing tests, refactoring tests; no production code change)
- `chore`: (updating grunt tasks etc; no production code change)
- `build`: (changes in the build system)
- `ci`: (changes in the CI/CD and deployment pipelines)
- `perf`: (significant performance improvement)
- `revert`: (revert a previous change)
