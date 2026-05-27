# Security Policy

We take security seriously. Thank you for helping keep this library and its users safe.

## Supported versions

Only the latest minor release receives security updates. Older lines are not back-patched.

| Version | Supported |
|---|---|
| latest | ✅ |
| older  | ❌ |

If you're on an unsupported version, the fix is to upgrade.

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please report privately via one of:

- **GitHub Security Advisories**: open a [private vulnerability report](https://github.com/hotherio/CHANGE-ME/security/advisories/new) directly on this repo. Preferred — it gives us a private space to coordinate the fix and the CVE.
- **Email**: `security@hother.io`. Please include:
  - A description of the issue
  - Steps to reproduce (a minimal test case if possible)
  - The version(s) affected
  - Any suggested mitigation

We aim to:

- Acknowledge receipt within **3 business days**
- Provide an initial assessment within **7 business days**
- Release a fix or workaround within **30 days** for high/critical severity, longer for lower severity

If you don't hear back within those windows, please nudge — mail can be lost.

## Disclosure policy

We coordinate disclosure with the reporter:

1. We confirm the issue and assess severity.
2. We develop a fix on a private branch.
3. We agree on a public disclosure date with the reporter.
4. We release the fix and publish a GitHub Security Advisory with the CVE (if applicable).
5. The reporter is credited in the advisory unless they prefer to remain anonymous.

## Supply-chain posture

This library is built and published with:

- **OIDC Trusted Publishing** to PyPI (no API tokens that can leak)
- **PEP 740 sigstore attestations** linking each release artifact to its source commit
- **GPG-signed `SHA256SUMS`** for asset verification
- **GitHub build-provenance attestation** in addition to sigstore
- **`pip-audit`** running on every PR and weekly schedule
- **`zizmor`** auditing GitHub Actions workflows for credential and permission risks
- **Renovate** keeping dependencies up to date

See [`docs/security.md`](docs/security.md) for the consumer-side verification flows.

## Out of scope

The following are not considered security vulnerabilities:

- Issues only reproducible against unsupported Python versions
- Denial-of-service via deliberately crafted input that exceeds documented limits
- Bugs in third-party dependencies (report to them; we'll bump after their fix lands)
- Reports from automated scanners without a demonstrated impact
