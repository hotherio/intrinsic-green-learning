# AGENTS.md

This file is the entry point for AI coding agents (Claude Code, Cursor, Continue, Aider, …) working in this repo. The same content is the recommended reading for human contributors on their first day.

> **Why am I in `.agents/`?** This is the canonical location. The same content is reachable via several paths so each tool's discovery rules just work:
>
> - `/AGENTS.md` — agents.md spec discovery (symlink → this file)
> - `/CLAUDE.md` — Claude Code root discovery (symlink → this file)
> - `/.claude/CLAUDE.md` — Claude Code project-dir discovery (`.claude/` is itself a symlink to `.agents/`, and `.agents/CLAUDE.md` symlinks back to this file)
>
> Edit this one file; the others follow.

## Read before you change code

Project-wide guidance lives in **[`docs/guidelines/`](../docs/guidelines/)** and is also published to the docs site. Each file maps to a category of work — consult the relevant one before touching the code:

| Before you… | Read |
|---|---|
| Add or refactor type annotations | [`docs/guidelines/typing.md`](../docs/guidelines/typing.md) |
| Add or restructure tests | [`docs/guidelines/testing.md`](../docs/guidelines/testing.md) |
| Raise, catch, or design an exception | [`docs/guidelines/errors.md`](../docs/guidelines/errors.md) |
| Add or modify the public API surface | [`docs/guidelines/api-design.md`](../docs/guidelines/api-design.md) |
| Document a public symbol | [`docs/guidelines/docstrings.md`](../docs/guidelines/docstrings.md) |
| Write a commit message or plan a release | [`docs/guidelines/releases.md`](../docs/guidelines/releases.md) |
| Add a dependency, pin, or upper bound | [`docs/guidelines/dependencies.md`](../docs/guidelines/dependencies.md) |
| Mix sync and async code | [`docs/guidelines/async.md`](../docs/guidelines/async.md) |

The guidelines are definitive — they encode the conventions extracted from `cancelable` and `streamblocks`. If a guideline forbids something, discuss the guideline change rather than silently working around it.

## Project commands

| Goal | Command |
|---|---|
| Install dev deps | `uv sync --all-groups` |
| Install hooks | `uv run lefthook install` |
| Run tests | `uv run pytest` |
| Type check | `uv run basedpyright src` |
| Lint + format | `uv run lefthook run pre-commit --all-files` |
| Build wheel + sdist | `uv build` |
| Serve docs locally | `uv run mkdocs serve` |
| Audit deps | `uvx pip-audit --requirement <(uv export --frozen --no-emit-project)` |
| Find dead code | `uv run vulture src/ --min-confidence 80` |

For the full developer setup, see [`README.md`](../README.md).

## Forge portability

This template runs on **GitHub** and **Forgejo** with the same workflow files. Where the two diverge (PyPI Trusted Publishing, attestations, OpenSSF Scorecard), the workflows env-gate on `github.server_url`. See [`FORGEJO.md`](../FORGEJO.md) for the full divergence list.

## Don't invent conventions

If something isn't covered by the guidelines, look at how the sibling repos (`cancelable`, `streamblocks`) handle it before inventing a new pattern. Both repos are linked from the Hother org page. When in doubt, open a discussion before changing the guidelines themselves — they're a contract.
