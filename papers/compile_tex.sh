#!/usr/bin/env bash
# Build NeurIPS-track manuscripts with vendored neurips_2026.sty.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TEXINPUTS="${ROOT}/papers/neurips_tex//:${TEXINPUTS:-}"

usage() {
  echo "usage: ${0##*/} neurips_sage_reg|neurips_spt_reg" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

paper="$1"
case "${paper}" in
  neurips_sage_reg|neurips_spt_reg) ;;
  *)
    usage
    echo "error: unknown paper '${paper}'" >&2
    exit 2
    ;;
esac

for tool in pdflatex bibtex; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "error: required TeX command not found on PATH: ${tool}" >&2
    exit 127
  fi
done

dir="${ROOT}/papers/${paper}"
if [[ ! -f "${dir}/main.tex" ]]; then
  echo "missing ${dir}/main.tex" >&2
  exit 1
fi
(
  cd "${dir}"
  pdflatex -interaction=nonstopmode main.tex
  if grep -q '^\\citation' main.aux 2>/dev/null; then
    bibtex main
  fi
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
)
echo "Wrote ${dir}/main.pdf"
