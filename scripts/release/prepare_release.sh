#!/usr/bin/env bash
# Prepare a release: bump version, run CI checks, build artifacts, commit, and tag.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DRY_RUN=0
FORCE=0
BUMP_ARGS=()

usage() {
  cat <<'EOF'
Usage: prepare_release.sh [--dry-run] [--force] <patch|minor|major|--version X.Y.Z>

Prepare a torchregress release locally. Publishing happens in CI when the tag is pushed.

Steps:
  1. Bump version in pyproject.toml
  2. Run ./scripts/ci_local.sh
  3. Build and validate dist/ artifacts
  4. Commit "Release vX.Y.Z"
  5. Create annotated tag vX.Y.Z

Options:
  --dry-run  Print actions without modifying git state or pyproject.toml.
  --force    Allow version bump on a dirty git tree.
  --help     Show this help message.

After running, push with:
  git push origin main
  git push origin vX.Y.Z
EOF
}

run() {
  if (( DRY_RUN == 1 )); then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

read_version() {
  uv run python - <<'PY'
import re
from pathlib import Path

text = Path("pyproject.toml").read_text(encoding="utf-8")
match = re.search(r'^version\s*=\s*"(\d+\.\d+\.\d+)"$', text, re.MULTILINE)
if match is None:
    raise SystemExit("Could not read version from pyproject.toml")
print(match.group(1))
PY
}

while (($# > 0)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --version)
      if (($# < 2)); then
        echo "Missing value for --version" >&2
        exit 2
      fi
      BUMP_ARGS=(--version "$2")
      shift 2
      ;;
    patch|minor|major)
      BUMP_ARGS=("$1")
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if (( ${#BUMP_ARGS[@]} == 0 )); then
  usage >&2
  exit 2
fi

BUMP_CMD=(uv run python scripts/release/bump_version.py)
if (( FORCE == 1 )); then
  BUMP_CMD+=(--force)
fi
BUMP_CMD+=("${BUMP_ARGS[@]}")

if (( DRY_RUN == 1 )); then
  BUMP_CMD+=(--dry-run)
  VERSION="$("${BUMP_CMD[@]}" | awk '{print $NF}')"
else
  "${BUMP_CMD[@]}"
  VERSION="$(read_version)"
fi

TAG="v${VERSION}"

echo "== CI parity checks =="
if (( DRY_RUN == 1 )); then
  echo "[dry-run] ./scripts/ci_local.sh"
else
  ./scripts/ci_local.sh
fi

echo "== Build package =="
if (( DRY_RUN == 1 )); then
  echo "[dry-run] ./scripts/release/build_package.sh"
else
  ./scripts/release/build_package.sh
fi

echo "== Verify tag/version alignment =="
if (( DRY_RUN == 1 )); then
  echo "[dry-run] uv run python scripts/release/verify_version.py --tag ${TAG}"
else
  uv run python scripts/release/verify_version.py --tag "${TAG}"
fi

COMMIT_MESSAGE="Release ${TAG}"
echo "== Commit and tag =="
run git add pyproject.toml
run git commit -m "${COMMIT_MESSAGE}"
run git tag -a "${TAG}" -m "${COMMIT_MESSAGE}"

cat <<EOF

Release prepared locally as ${TAG}.

Next steps:
  git push origin main
  git push origin ${TAG}

Pushing ${TAG} triggers .github/workflows/release.yml and publishes to PyPI.
EOF
