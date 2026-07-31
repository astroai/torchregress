# Release scripts

Local helpers for preparing torchregress PyPI releases. Publishing is automated in CI when a
release tag is pushed.

## Quick start

```bash
# Patch release (0.1.0 -> 0.1.1)
./scripts/release/prepare_release.sh patch

# Minor or major release
./scripts/release/prepare_release.sh minor
./scripts/release/prepare_release.sh major

# Explicit version
./scripts/release/prepare_release.sh --version 1.0.0

# Preview without writing
./scripts/release/prepare_release.sh --dry-run patch
```

Then push:

```bash
git push origin main
git push origin vX.Y.Z
```

Pushing `vX.Y.Z` triggers [`.github/workflows/release.yml`](../.github/workflows/release.yml),
which builds and publishes to PyPI using Trusted Publishing.

## Scripts

| Script | Purpose |
|--------|---------|
| [`prepare_release.sh`](prepare_release.sh) | End-to-end local release prep |
| [`bump_version.py`](bump_version.py) | Update `pyproject.toml` version |
| [`verify_version.py`](verify_version.py) | Ensure tag `vX.Y.Z` matches `pyproject.toml` |
| [`build_package.sh`](build_package.sh) | Build and validate `dist/` artifacts |

## Individual commands

```bash
pixi run python scripts/release/bump_version.py patch
pixi run python scripts/release/verify_version.py --tag v0.1.0
./scripts/release/build_package.sh
./scripts/release/build_package.sh --check-only
```

## Notes

- Canonical version lives in [`pyproject.toml`](../../pyproject.toml).
- Runtime `torchregress.__version__` is read from installed package metadata.
- Local publish is intentionally not used; CI publishes via PyPI Trusted Publishing.
- Full maintainer setup and troubleshooting: [`docs/RELEASING.md`](../../docs/RELEASING.md).
