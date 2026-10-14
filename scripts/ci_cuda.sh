#!/usr/bin/env bash
# Local CUDA test gate (GitHub CI stays CPU-only).
#
# Run on a GPU machine (workstation, CANFAR session, ...) for every release candidate:
#
#   pixi run test-cuda            # or: bash scripts/ci_cuda.sh
#
# Steps:
#   1. Verify torch.cuda.is_available() (exit 2 with a message otherwise).
#   2. pytest -m cuda -v                         (CPU<->CUDA parity of losses and metrics)
#   3. pytest <full suite> with TORCHREGRESS_TEST_DEVICE=cuda   (the ``device`` fixture -> cuda)
#   4. Write reports/cuda/<YYYYMMDD>_<short-sha>.txt with the commit, torch/CUDA versions,
#      GPU name and both pytest summaries.  Keep that file with the release evidence.
#
# Exit status: 0 all green, 1 test failures, 2 CUDA unavailable.
#
# Optional environment variables (mainly to exercise the script on a CPU-only machine):
#   PYTHON                      interpreter to use (default: python)
#   TORCHREGRESS_CI_DEVICE      device under test (default: cuda; e.g. "cpu" for a dry run,
#                               which skips the CUDA availability check)
#   TORCHREGRESS_CI_FULL_TARGET pytest target of the full-suite step (default: tests)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

PYTHON="${PYTHON:-python}"
DEVICE="${TORCHREGRESS_CI_DEVICE:-cuda}"
FULL_TARGET="${TORCHREGRESS_CI_FULL_TARGET:-tests}"

if [ "$DEVICE" = "cuda" ]; then
  if ! "$PYTHON" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "ci_cuda.sh: torch.cuda.is_available() is False; no usable CUDA device here." >&2
    echo "ci_cuda.sh: run this on a GPU machine with a CUDA build of torch" >&2
    echo "ci_cuda.sh: (to exercise the parity tests on CPU: TORCHREGRESS_PARITY_DEVICE=cpu pytest -m cuda)." >&2
    exit 2
  fi
else
  echo "ci_cuda.sh: WARNING: TORCHREGRESS_CI_DEVICE=$DEVICE (dry run, not a CUDA validation)" >&2
fi

if SHA_FULL="$(git rev-parse HEAD 2>/dev/null)"; then
  SHA_SHORT="$(git rev-parse --short=9 HEAD)"
  if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    DIRTY="yes (uncommitted changes in tracked files)"
  else
    DIRTY="no"
  fi
else
  SHA_FULL="unknown"
  SHA_SHORT="nogit"
  DIRTY="unknown"
fi

REPORT_DIR="$ROOT/reports/cuda"
REPORT="$REPORT_DIR/$(date -u +%Y%m%d)_${SHA_SHORT}.txt"
mkdir -p "$REPORT_DIR"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

export TORCHREGRESS_PARITY_DEVICE="$DEVICE"

# Last line of a pytest run ("=== 12 passed, 3 skipped in 4.2s ===") plus the failure list.
summarize() {
  local log="$1"
  grep -E '^(FAILED|ERROR) ' "$log" | sort -u
  grep -E '(passed|failed|error|skipped|deselected|no tests ran).* in [0-9.]+s( \([0-9:]+\))?( =+)?$' "$log" | tail -n 1
}

echo "== [1/2] pytest -m cuda -v (device: $DEVICE) =="
"$PYTHON" -m pytest -m cuda -v -rfEs -p no:cacheprovider 2>&1 | tee "$WORK/parity.log"
PARITY_RC=${PIPESTATUS[0]}

echo "== [2/2] full suite with TORCHREGRESS_TEST_DEVICE=$DEVICE ($FULL_TARGET) =="
# FULL_TARGET is intentionally unquoted so that it may hold several paths / pytest options.
# shellcheck disable=SC2086
TORCHREGRESS_TEST_DEVICE="$DEVICE" "$PYTHON" -m pytest $FULL_TARGET -q -rfE -p no:cacheprovider \
  2>&1 | tee "$WORK/full.log"
FULL_RC=${PIPESTATUS[0]}

{
  echo "torchregress CUDA test report"
  echo "============================="
  echo "date (UTC)    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "commit        : $SHA_FULL"
  echo "dirty tree    : $DIRTY"
  echo "device tested : $DEVICE"
  echo "host          : $(hostname 2>/dev/null || echo unknown)"
  "$PYTHON" - <<'PY'
import platform
import sys

import torch

print(f"python        : {sys.version.split()[0]} ({platform.platform()})")
print(f"torch         : {torch.__version__}")
print(f"torch CUDA    : {torch.version.cuda}")
print(f"cuDNN         : {torch.backends.cudnn.version()}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            f"GPU {i}         : {props.name} "
            f"(capability {props.major}.{props.minor}, {props.total_memory / 2**30:.1f} GiB)"
        )
PY
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi    : $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | paste -sd ';' -)"
  else
    echo "nvidia-smi    : not found"
  fi
  echo
  echo "-- pytest -m cuda -v (exit code $PARITY_RC) --"
  summarize "$WORK/parity.log"
  echo
  echo "-- full suite, TORCHREGRESS_TEST_DEVICE=$DEVICE: pytest $FULL_TARGET (exit code $FULL_RC) --"
  summarize "$WORK/full.log"
  echo
  if [ "$PARITY_RC" -eq 0 ] && [ "$FULL_RC" -eq 0 ]; then
    echo "RESULT: PASS"
  else
    echo "RESULT: FAIL"
  fi
} >"$REPORT"

echo
echo "ci_cuda.sh: report written to ${REPORT#"$ROOT"/}"
tail -n 1 "$REPORT"

if [ "$PARITY_RC" -eq 0 ] && [ "$FULL_RC" -eq 0 ]; then
  exit 0
fi
exit 1
