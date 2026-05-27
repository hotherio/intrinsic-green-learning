# Dependencies

How third-party packages enter, are pinned, and are audited in Hother Python libraries.

## What's enforced

- **`uv.lock`** is committed and authoritative — `uv-lock` pre-push hook verifies it's in sync with `pyproject.toml`.
- **`pip-audit`** runs on every PR + weekly schedule against `uv.lock` (blocking).
- **Renovate** auto-opens PRs for dep updates (extends `hotherio/config-renovate`).
- **basedpyright** flags missing or untyped imports — adding a dep without stubs surfaces immediately.

## What we expect

### When to add a dependency

Ask in order:

1. **Is it in the standard library?** `pathlib`, `dataclasses`, `tomllib`, `itertools`, `asyncio` cover a lot.
2. **Is it a one-pager?** If we'd vendor 30 lines instead of a 50-MB transitive tree, vendor.
3. **Do we already pull it transitively?** Then declare it explicitly so we own the pin.
4. **Is the maintainer active?** Check PyPI release cadence and GitHub last-commit-date — abandoned deps are tomorrow's CVEs.
5. **Is the license compatible?** See "License compatibility" below.

If yes to "add it," add to the right place:

```toml
[project]
dependencies = [
    "anyio>=4.9.0",     # runtime
]

[project.optional-dependencies]
fastapi = [
    "fastapi>=0.115",   # opt-in integration
]

[dependency-groups]
dev = [
    "pytest>=9.0.3",    # dev-only
]
doc = [
    "mkdocs-material>=9.5.49",  # docs-only
]
```

### Pinning strategy

| Location | Operator | Why |
|---|---|---|
| `[project].dependencies` | `>=` lower bound | Lets downstream resolvers find a compatible version. Don't `==`-pin — that breaks downstream compatibility. |
| `[project].dependencies` | Add `<X` upper bound only if you know that version breaks you | Avoid speculative caps; they cause resolver pain later. |
| `[project.optional-dependencies]` | `>=` lower bound | Same reasoning. |
| `[dependency-groups]` (dev, doc) | `>=` lower bound | We don't ship these; flexibility is fine. The `uv.lock` pins the actual version we test against. |
| `uv.lock` | Auto-pinned by uv | Authoritative for reproducible builds. **Commit it.** |

> **Hother decision needed:** historically some pinned-with-`==` tool versions (like `ruff==0.8.6`) sat in dev. Block 4 of the modernization loosened most to `>=`. Decide if any deserve the strict pin (e.g., ruff format-stability across versions) and document why.

### Upper bounds — when?

Only when you have evidence the next major breaks you:

```python
"pydantic>=2.11,<3"        # OK: Pydantic 3 is a known future incompat
"anyio>=4.9"               # better: no upper bound until breakage is known
```

Adding `<X` "just in case" makes your library uninstallable alongside packages that need newer X. Don't do it.

### Optional / integration dependencies

Use `[project.optional-dependencies]` for opt-in integrations (FastAPI adapter, OpenAI provider, etc.):

```toml
[project.optional-dependencies]
fastapi = ["fastapi>=0.115"]
gemini = ["google-generativeai>=0.8"]
```

Then consumers install with `pip install hother-package[fastapi]`. Document the extras in the README.

### License compatibility

The template ships **MIT**. When adding a dep, check its license:

| OK to depend on | Caution |
|---|---|
| MIT, BSD, Apache-2.0, ISC, Unlicense, PSF | Compatible with MIT. |
| LGPL | OK for dynamic linking, but document the constraint in `LICENSE-3RD-PARTY.md` if applicable. |
| GPL | **Avoid in runtime deps** — would force the library to relicense as GPL. |
| AGPL | **Avoid everywhere** — even dev deps can create obligations. |
| Custom / unclear | **Don't.** |

Use `uvx pip-licenses --from=mixed -f md` (the existing `make licenses` target) to audit after non-trivial dep changes.

### Vulnerability response

`pip-audit` runs weekly. When it surfaces a CVE:

1. **High / critical** — fix within 7 days. Renovate-bump the affected dep; merge ASAP.
2. **Medium** — fix within the next minor release cycle.
3. **Low / informational** — bundle with next dep refresh.

If a fix isn't available, `pip-audit --ignore-vuln <ID>` with a comment explaining the situation (in `pyproject.toml` or a dedicated `.pip-audit-ignore.toml`).

### Renovate

The template extends `hotherio/config-renovate`. Renovate batches dep PRs by ecosystem and opens them weekly. Review and merge them as part of normal PR flow — don't let them stack.

If a Renovate PR fails CI, treat it as a real failure: either pin the prior version with a `<X` upper bound (and a comment) or fix the incompatibility.

### `uv.lock` discipline

- **Always commit `uv.lock` changes** alongside the `pyproject.toml` change that produced them.
- **Never edit `uv.lock` by hand.** Use `uv lock` / `uv lock --upgrade <pkg>`.
- **Re-lock when bumping `requires-python`** so platform-specific wheels resolve correctly.

### Removing a dependency

When deleting an import, also remove from `pyproject.toml` and re-lock:

```bash
uv remove <pkg>             # or edit pyproject.toml manually
uv lock
git add pyproject.toml uv.lock
```

`vulture` (via `make vulture`) helps catch unused imports lurking in the codebase.

## Examples

### Good

```toml
[project]
dependencies = [
    "anyio>=4.9.0",
    "pydantic>=2.11.7",
]

[project.optional-dependencies]
fastapi = ["fastapi>=0.115"]

[dependency-groups]
dev = [
    "pytest>=9.0.3",
    "basedpyright==1.39.6",  # pinned: per-version diagnostics drift
]
```

### Bad

```toml
[project]
dependencies = [
    "anyio==4.9.0",         # exact pin in a library — breaks downstream
    "requests<3",           # speculative cap
    "some-fork-of-x",       # no version, no constraint
    "abandoned-pkg",        # last release: 2019
]
```

## See also

- [`releases.md`](releases.md) — how dep bumps appear in the changelog
- [PEP 631 — Dependency specification](https://peps.python.org/pep-0631/)
- [PEP 735 — Dependency Groups](https://peps.python.org/pep-0735/)
- [SPDX license list](https://spdx.org/licenses/) — for the `[project].license` field
