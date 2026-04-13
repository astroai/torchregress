# Shift-Factored Predictive Transport for Probabilistic Regression

NeurIPS-track manuscript and artifact map for SPT-Reg in **torchregress**.

## Layout

- `main.tex`: single-file manuscript source (NeurIPS 2026 style)
- `refs.bib`: bibliography
- `reproducibility.md`: map from tables/figures to generated artifacts
- `status.md`: canonical claims, empirical position, next steps

## Working rules

- Keep `main.tex` as the only build path for the manuscript.
- Tie paper claims to generated artifacts under `reports/neurips_spt_reg/` (see `.gitignore` allowlist for what may be committed).
- Update `reproducibility.md` whenever a cited figure, table, or benchmark summary changes.
- Keep SPT-Reg files separate from the sibling track: [SAGE-Reg](../neurips_sage_reg/).

## Sister track

Semi-supervised distributional self-agreement: [papers/neurips_sage_reg/](../neurips_sage_reg/).

## Parent index

See [papers/README.md](../README.md) for both NeurIPS workspaces.

## Build

From repo root: `./papers/compile_tex.sh neurips_spt_reg` (see [neurips_tex/README.md](../neurips_tex/README.md)).
