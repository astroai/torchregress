#!/usr/bin/env bash
# Build and validate torchregress distribution artifacts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CHECK_ONLY=0

usage() {
  cat <<'EOF'
Usage: build_package.sh [--check-only] [--help]

Build sdist/wheel with `python -m build` (via pixi) and validate artifacts under dist/.

Options:
  --check-only  Validate existing dist/ artifacts without rebuilding.
  --help        Show this help message.
EOF
}

validate_artifacts() {
  shopt -s nullglob
  local sdists=(dist/*.tar.gz)
  local wheels=(dist/*.whl)

  if (( ${#sdists[@]} == 0 )); then
    echo "ERROR: no sdist found in dist/" >&2
    exit 1
  fi
  if (( ${#wheels[@]} == 0 )); then
    echo "ERROR: no wheel found in dist/" >&2
    exit 1
  fi

  for artifact in "${sdists[@]}" "${wheels[@]}"; do
    echo "Validating ${artifact}"
    case "${artifact}" in
      *.tar.gz)
        tar -tzf "${artifact}" >/dev/null
        ;;
      *.whl)
        pixi run --environment release python -m zipfile -t "${artifact}" >/dev/null
        ;;
      *)
        echo "ERROR: unsupported artifact type: ${artifact}" >&2
        exit 1
        ;;
    esac
  done

  echo "OK: validated ${#sdists[@]} sdist(s) and ${#wheels[@]} wheel(s) in dist/"
}

while (($# > 0)); do
  case "$1" in
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if (( CHECK_ONLY == 0 )); then
  rm -rf dist/
  pixi run --environment release build
fi

validate_artifacts
