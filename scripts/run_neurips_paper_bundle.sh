#!/usr/bin/env bash
# Run SPT-Reg then SAGE-Reg NeurIPS one-shot evidence scripts with the same CLI args.
# Do not pass --run-root here (both scripts would write manifests into the same tree).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
for a in "$@"; do
  if [[ "${a}" == "--run-root" ]]; then
    echo "error: ${0##*/} does not support --run-root (outputs would collide)." >&2
    echo "Run scripts/run_neurips_spt_reg_full.py and scripts/run_neurips_sage_reg_full.py separately with custom paths." >&2
    exit 2
  fi
done
echo "== NeurIPS paper bundle: SPT-Reg full runner ==" >&2
uv run python scripts/run_neurips_spt_reg_full.py "$@"
echo "== NeurIPS paper bundle: SAGE-Reg full runner ==" >&2
uv run python scripts/run_neurips_sage_reg_full.py "$@"
