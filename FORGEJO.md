# Forgejo portability

This template is designed to work on both **GitHub** and **Forgejo** (Codeberg, self-hosted, etc.) with the same workflow files. This document records the divergences and what you need to do when running on a Forgejo instance.

## What works out of the box

- **Workflows live in `.github/workflows/`** — Forgejo 1.21+ reads this path natively. No `.forgejo/workflows/` mirror is required.
- **Action references** use bare `actions/checkout@<sha>` form. Forgejo resolves these via its default action mirror (`data.forgejo.org`), which is itself a mirror of GitHub Marketplace. Admins can override `DEFAULT_ACTIONS_URL` to point straight at GitHub if desired.
- **Issue & PR templates** are mirrored at `.forgejo/pull_request_template.md` (symlinked to `.github/PULL_REQUEST_TEMPLATE.md`). Add `.forgejo/ISSUE_TEMPLATE/` symlinks the same way if you add issue templates later.
- **Renovate** supports Forgejo natively — set `platform: forgejo` in `renovate.json` when deploying there.

## What differs

### 1. PyPI publishing — Trusted Publishing vs API token

| | GitHub | Forgejo |
|---|---|---|
| Mechanism | OIDC Trusted Publishing | API token (`UV_PUBLISH_TOKEN`) |
| Configuration | One-time setup at pypi.org → Manage → Publishing | Add `UV_PUBLISH_TOKEN` secret to the repo |
| Attestations | Automatic (PEP 740, sigstore) | None |
| Step | `pypa/gh-action-pypi-publish@release/v1` | `uv publish --token "$UV_PUBLISH_TOKEN"` |

The release workflow (`.github/workflows/semantic-release.yml`) auto-detects via `github.server_url` and uses the appropriate path.

**PyPI does not currently trust Forgejo OIDC tokens** — the immutable-ID claims required to prevent token-resurrection attacks aren't yet emitted by Forgejo. Track the [Forgejo issue](https://codeberg.org/forgejo/forgejo/issues/2389) for progress.

### 2. Build-provenance attestations

The `gh attestation attest` step uses the GitHub API and is GitHub-only. On Forgejo this step is skipped; consumers cannot verify the package via `gh attestation verify` or `pip install --verify-attestations` for Forgejo-published releases. GPG-signed `SHA256SUMS` remains the verification path.

### 3. OpenSSF Best Practices Badge

[OpenSSF Scorecard](https://scorecard.dev/) only runs on GitHub. Forgejo-only forks of this template **cannot earn the badge**. The Scorecard workflow (when added in Phase 4) will skip on non-GitHub forges via the same `github.server_url` gate.

### 4. App-token-based commits

`actions/create-github-app-token` doesn't work on Forgejo. The release workflow uses it to mint a short-lived token for PSR's release commits. On Forgejo, you'd need to:

- Provide a long-lived bot deploy key in `secrets.RELEASE_BOT_PRIVATE_KEY`, OR
- Use a personal access token stored in `secrets.GITHUB_TOKEN`

This template hasn't yet wired the Forgejo path for the release commit step — open a follow-up if you need Forgejo releases to work.

## Required secrets per platform

### GitHub
- `vars.RELEASE_BOT_APP_ID`, `secrets.RELEASE_BOT_PRIVATE_KEY` — GitHub App for signed release commits
- `secrets.HOTHER_BOT_GPG_KEY`, `secrets.HOTHER_BOT_GPG_PASSPHRASE` — GPG signing
- `vars.PKG_NAME`, `vars.PKG_REGISTRY`, `secrets.GH_TOKEN` — private registry dispatch
- (no PyPI secret — Trusted Publishing)

### Forgejo
- `secrets.UV_PUBLISH_TOKEN` — PyPI API token (https://pypi.org/manage/account/token/)
- `secrets.HOTHER_BOT_GPG_KEY`, `secrets.HOTHER_BOT_GPG_PASSPHRASE` — same as GitHub
- App-token path: TBD (see above)

## Runner labels

GitHub workflows use `runs-on: ubuntu-latest` and `ubuntu-24.04`. Forgejo's bundled `act_runner` recognizes the same labels via its `labels:` config (typically `ubuntu-latest:docker://node:24-bookworm` and `ubuntu-24.04:docker://node:24-bookworm`). Adjust your runner's `config.yaml` if you self-host.
