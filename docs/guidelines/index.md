# Guidelines

These pages encode the Hother conventions for Python libraries. They're aimed equally at human contributors and AI coding agents — read the relevant guideline before changing the code, and cite it in PR reviews when something doesn't conform.

| File | When to consult |
|---|---|
| [`typing.md`](typing.md) | Before writing or refactoring type annotations |
| [`testing.md`](testing.md) | Before adding or restructuring tests |
| [`errors.md`](errors.md) | Before raising, catching, or designing an exception |
| [`api-design.md`](api-design.md) | Before adding or modifying public-API surface |
| [`docstrings.md`](docstrings.md) | Before documenting (or removing) a public symbol |
| [`releases.md`](releases.md) | Before crafting a commit message or planning a release |
| [`dependencies.md`](dependencies.md) | Before adding a dep, pinning, or relaxing a pin |
| [`async.md`](async.md) | Before mixing sync/async code |

Each guideline:

- States what's **enforced** automatically (basedpyright, ruff, pip-audit, etc.).
- States what's **expected** in review (the team's conventions).

If you find yourself wanting to do something a guideline forbids, the right move is to discuss the guideline change — don't silently work around it.
