# Releasing torchregress

This guide describes how to publish a new version of **torchregress** to PyPI.

Publishing is fully automated in GitHub Actions after you push an annotated release tag.
Local scripts handle version bumps, validation, and tagging.

## Overview

1. Run `./scripts/release/prepare_release.sh` locally.
2. Push `main` and the new tag `vX.Y.Z`.
3. GitHub Actions builds the package, checks its contents and metadata, installs
   the wheel with core dependencies only on Linux and macOS for Python 3.12–3.14,
   and then publishes to PyPI via Trusted Publishing.

See also [`scripts/release/README.md`](https://github.com/astroai/torchregress/blob/main/scripts/release/README.md) for command-level reference.

## One-time setup

### 1. Configure PyPI Trusted Publishing

In the torchregress project settings on PyPI:

`https://pypi.org/manage/project/torchregress/settings/publishing/`

Add a **GitHub** trusted publisher with:

| Field | Value |
|-------|-------|
| Owner | `astroai` |
| Repository | `torchregress` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

No long-lived PyPI API token is required in GitHub secrets when Trusted Publishing is configured.

Before the first upload the project does not exist on PyPI yet: add the same values as a
**pending publisher** under *Your account → Publishing*. Do the same on TestPyPI
(`https://test.pypi.org`) for release-candidate dry runs.

### 2. Create the GitHub `pypi` and `testpypi` environments

In the GitHub repository:

**Settings → Environments → New environment → `pypi`** (and again for `testpypi`)

Tags `vX.Y.ZrcN` publish to TestPyPI through the `testpypi` environment; tags `vX.Y.Z`
publish to PyPI through `pypi`. A TestPyPI trusted publisher with the same owner,
repository and workflow, and environment `testpypi`, is needed for the dry run.

Recommended protection rules:

- Required reviewers before deployment (optional but recommended for production releases)
- Deployment branches and tags: allow tags matching `v*` (the release job runs on tag pushes, not on `main`)

The workflow job in [`.github/workflows/release.yml`](../.github/workflows/release.yml) uses
`environment: pypi` and `permissions.id-token: write`, which are required for OIDC-based
publishing with GitHub Actions OIDC (`pypa/gh-action-pypi-publish`).

### 3. Turn on GitHub Pages for the docs

**Settings → Pages → Source: GitHub Actions.** The `Docs` workflow
(`.github/workflows/docs.yml`) then publishes https://astroai.github.io/torchregress/
on every push to `main`. The README, `pyproject.toml` and the PyPI page link there.

## First release (0.3.0): dry run on TestPyPI first

1. Finish the one-time setup above, including the TestPyPI pending publisher.
2. Move the `[Unreleased]` notes in `CHANGELOG.md` under `## [0.3.0] - <date>`.
3. Prepare a release candidate and push it:

   ```bash
   ./scripts/release/prepare_release.sh --version 0.3.0rc1
   git push origin main && git push origin v0.3.0rc1
   ```

4. Watch **Actions → Release**: `build`, the six `install-smoke` legs, then
   `publish-testpypi`.
5. Install from TestPyPI in a clean environment (dependencies come from PyPI):

   ```bash
   python -m venv /tmp/tr && . /tmp/tr/bin/activate
   pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ torchregress==0.3.0rc1
   torchregress-health
   ```

6. Run the local CUDA tests on the candidate (`pixi run -e cuda test-cuda` on a GPU machine).
7. Release: `./scripts/release/prepare_release.sh --version 0.3.0`, push `main` and `v0.3.0`.

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

For clearer OIDC errors during setup, the release workflow uses
`pypa/gh-action-pypi-publish` with `id-token: write`.

### Tag pushed but version not bumped on main

The tag must point to a commit whose `pyproject.toml` version matches the tag. Use
`prepare_release.sh` so the commit and tag are created together.

### Local build validation

`build_package.sh` checks the archives with `scripts/release/check_dist.py`
(no build or tooling output, `py.typed` and `LICENSE` present, matching versions).
To repeat the CI install check by hand:

```bash
python -m venv /tmp/tr && /tmp/tr/bin/pip install dist/*.whl
cd /tmp && /tmp/tr/bin/python <repo>/scripts/release/smoke_install.py
```

```bash
./scripts/release/build_package.sh
./scripts/release/build_package.sh --check-only
```

### Manual version bump only

```bash
pixi run python scripts/release/bump_version.py patch
pixi run python scripts/release/verify_version.py --tag vX.Y.Z
```

## What is intentionally not automated here

- CHANGELOG generation
- GitHub Release notes
- TestPyPI dry-run publishing

These can be added later if needed.
