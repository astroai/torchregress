# Releasing torchregress

This guide describes how to publish a new version of **torchregress** to PyPI.

Publishing is fully automated in GitHub Actions after you push an annotated release tag.
Local scripts handle version bumps, validation, and tagging.

## Overview

1. Run `./scripts/release/prepare_release.sh` locally.
2. Push `main` and the new tag `vX.Y.Z`.
3. GitHub Actions builds the package and publishes to PyPI via Trusted Publishing.

See also [`scripts/release/README.md`](https://github.com/astroai/torchregress/blob/main/scripts/release/README.md) for command-level reference.

## One-time setup

### 1. Configure PyPI Trusted Publishing

In the torchregress project settings on PyPI:

`https://pypi.org/manage/project/torchregress/settings/publishing/`

Add a **GitHub** trusted publisher with:

| Field | Value |
|-------|-------|
| Owner | `sfabbro` |
| Repository | `torchregress` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

No long-lived PyPI API token is required in GitHub secrets when Trusted Publishing is configured.

### 2. Create the GitHub `pypi` environment

In the GitHub repository:

**Settings → Environments → New environment → `pypi`**

Recommended protection rules:

- Required reviewers before deployment (optional but recommended for production releases)
- Restrict deployment branches to `main` if you use branch protections

The workflow job in [`.github/workflows/release.yml`](../.github/workflows/release.yml) uses
`environment: pypi` and `permissions.id-token: write`, which are required for OIDC-based
publishing with `uv publish`.

## Release checklist

### 1. Prepare the release locally

From a clean working tree on `main`:

```bash
./scripts/release/prepare_release.sh patch
```

Other forms:

```bash
./scripts/release/prepare_release.sh minor
./scripts/release/prepare_release.sh major
./scripts/release/prepare_release.sh --version 1.0.0
./scripts/release/prepare_release.sh --dry-run patch
```

`prepare_release.sh` will:

1. Bump `version` in [`pyproject.toml`](../pyproject.toml)
2. Run [`scripts/ci_local.sh`](../scripts/ci_local.sh) (CI parity checks)
3. Build and validate artifacts with [`scripts/release/build_package.sh`](../scripts/release/build_package.sh)
4. Verify tag/version alignment
5. Commit `Release vX.Y.Z`
6. Create annotated tag `vX.Y.Z`

### 2. Push main and the tag

```bash
git push origin main
git push origin vX.Y.Z
```

Pushing the tag triggers the release workflow.

### 3. Verify the release

1. Open **Actions → Release** and confirm the workflow succeeded.
2. Download the uploaded `dist` artifact if you want to inspect the built files.
3. Confirm the new version on PyPI: https://pypi.org/project/torchregress/

## Version source of truth

- Canonical version: `version` in [`pyproject.toml`](../pyproject.toml)
- Runtime version: `torchregress.__version__` from installed package metadata
- Release tags must match exactly: tag `v1.2.3` ↔ `version = "1.2.3"`

[`scripts/release/verify_version.py`](../scripts/release/verify_version.py) enforces this in CI.

## Troubleshooting

### Trusted Publishing failed

Common causes:

- PyPI trusted publisher workflow/environment mismatch
- Missing `id-token: write` permission on the publish job
- Tag/version mismatch between git tag and `pyproject.toml`

For clearer OIDC errors during setup, CI uses:

```bash
uv publish --trusted-publishing always
```

### Tag pushed but version not bumped on main

The tag must point to a commit whose `pyproject.toml` version matches the tag. Use
`prepare_release.sh` so the commit and tag are created together.

### Local build validation

```bash
./scripts/release/build_package.sh
./scripts/release/build_package.sh --check-only
```

### Manual version bump only

```bash
uv run python scripts/release/bump_version.py patch
uv run python scripts/release/verify_version.py --tag vX.Y.Z
```

## What is intentionally not automated here

- CHANGELOG generation
- GitHub Release notes
- TestPyPI dry-run publishing

These can be added later if needed.
