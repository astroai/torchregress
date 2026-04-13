#!/usr/bin/env bash
# Build NeurIPS-track manuscripts with vendored neurips_2026.sty.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TEXINPUTS="${ROOT}/papers/neurips_tex//:${TEXINPUTS:-}"
paper="${1:?usage: compile_tex.sh neurips_sage_reg|neurips_spt_reg}"
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
