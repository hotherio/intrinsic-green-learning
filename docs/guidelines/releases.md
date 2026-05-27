# Releases

How versions, commits, and releases work in Hother Python libraries. Closely mirrors the ClickUp KB doc `📚 Python Versioning & Release Automation Best Practices`.

## What's enforced

- **Conventional Commits** validated by `lefthook` (commit-msg hook + pre-push hook) and by `amannn/action-semantic-pull-request` on PR titles.
- **python-semantic-release** (PSR) reads commits since the last tag and decides the next version.
- **GPG-signed commits + tags** from the Hother Bot — verified by GitHub once the bot's key is registered on the org.
- **`pre-push`** runs `pytest` + `uv build` so a broken push never reaches `main` (Block 14).

## What we expect

### Commit message grammar

```
<type>(<scope>): <subject>

<optional body>

<optional footer>
```

Allowed `<type>` values (enforced):

| Type | Triggers | Bumps |
|---|---|---|
| `feat` | New feature | MINOR |
| `fix` | Bug fix | PATCH |
| `perf` | Performance improvement | PATCH |
| `refactor` | Internal refactor, no behavior change | PATCH |
| `docs` | Documentation only | none |
| `style` | Formatting / whitespace, no logic | none |
| `test` | Test changes only | none |
| `build` | Build system / packaging | none |
| `ci` | CI/CD config | none |
| `chore` | Misc maintenance | none |
| `revert` | Revert of prior commit | matches type of reverted |

**Breaking changes** bump MAJOR (or, for `0.x`, MINOR). Two ways to signal:

1. **Exclamation in subject**: `feat!: redesign the registry API`
2. **`BREAKING CHANGE:` footer in body** (preferred for libraries — gives room to explain):

   ```
   feat: redesign the registry API

   The Registry class now requires an explicit `scope` argument. Migration:
   `Registry()` → `Registry(scope=Scope.PROCESS)`.

   BREAKING CHANGE: Registry constructor signature changed.
   ```

### Subject line rules

- Imperative mood: "add", "fix", "remove" — not "added", "adds".
- Lowercase first letter (the PR-title checker is configured otherwise per `amannn/action-semantic-pull-request` but commit-msg is lowercase-first).
- No trailing period.
- Under 72 characters.
- Concrete: `fix: stop dropping events on backpressure` > `fix: bug fix`.

### Scope (optional)

When the change touches a clear sub-area, use a scope:

```
feat(registry): add per-operation timeout
fix(ci): pin actions/checkout to SHA
docs(typing): clarify Protocol guidance
```

Don't invent new scopes just to be specific. Existing scopes used across siblings: `ci`, `release`, `deps`, `api`, `core`. Match what's there.

### What the body is for

The body is the **PR description rendered in the changelog**. Use it to explain:

- Why the change was needed (linked issue, incident, customer request).
- Migration steps for breaking changes.
- Performance numbers ("p99 dropped from 230ms to 12ms").
- Anything a future reader scanning `git log` would thank you for.

Markdown is supported. Reference issues with `Closes #123` / `Refs #456`.

### What goes in `chore(release):` commits

PSR generates these automatically. Don't write one yourself unless backfilling a manual release. They contain the version bump in `pyproject.toml` and the new CHANGELOG entry — nothing else.

### Skip-release commits

Push only `chore:` / `docs:` / `style:` / `test:` / `ci:` / `build:` commits to land changes without triggering a release. PSR sees no `feat` / `fix` / `perf` / `refactor` / breaking-change footer and decides "no version bump needed."

### Pre-release tags

The `[tool.semantic_release].allow_zero_version = true` setting permits 0.x releases. Pre-release tags follow PEP 440:

- `1.2.0a1` — alpha
- `1.2.0b1` — beta
- `1.2.0rc1` — release candidate
- `1.2.0` — final

The template does **not** ship a dedicated RC workflow (cancelable and streamblocks release directly from main). If you need an RC flow, see KB §`rc-release.yml`.

> **Hother decision needed:** the KB documents a dev-release pattern (every-push-to-main → `1.2.3.dev10+gSHA`) that requires hatch-vcs. The template removed hatch-vcs when migrating to PSR. Decide whether to (a) skip dev releases (current default), (b) re-introduce hatch-vcs alongside PSR for dev builds, (c) use PSR `--prerelease --prerelease-token dev` on every push (noisy).

### How to preview a release

```bash
# Show what version PSR would compute next (no side effects)
uvx python-semantic-release version --noop --print-version

# Show the changelog entry PSR would write
uvx python-semantic-release changelog --noop

# Preview unreleased changes via git-cliff (used for the GitHub Release body)
make changelog-unreleased
```

### How to trigger a release manually

```bash
gh workflow run semantic-release.yml
```

Or via the GitHub UI: Actions → "Semantic Release" → "Run workflow".

## Examples

### Good

```
feat(registry): add per-operation timeout argument

Operations can now opt into a deadline by passing `timeout_s` to
`Registry.register()`. When elapsed, the operation's CancellationToken is
fired and the operation handler raises TimeoutCancellationError.

Closes #142.
```

```
fix(ci): pin actions/checkout to SHA

zizmor `unpinned-uses` finding (high). Renovate will continue to bump
via the `# v4` comment.
```

```
feat!: drop Python 3.12 support

Python 3.13 is now the minimum to match `cancelable` and `streamblocks`.

BREAKING CHANGE: Requires Python >=3.13. 3.12 users must stay on 1.x.
```

### Bad

```
update stuff           ← no type, vague subject
Fix Bug                ← title case, no detail
chore: misc            ← no useful description
feat(MyScope): added a new thing  ← past tense, capitalized scope, vague
fix: things            ← what things?
```

## See also

- [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/)
- [python-semantic-release docs](https://python-semantic-release.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)
- [PEP 440 — Version Identification](https://peps.python.org/pep-0440/)
- ClickUp KB doc `2bdjt-7352` — the canonical Hother source
