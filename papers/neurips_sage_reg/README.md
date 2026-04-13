# SAGE-Reg: Self-Agreement Distributional Self-Training for Regression

NeurIPS-track manuscript and artifact map for SAGE-Reg in **torchregress**.

## Layout

- `main.tex`: single-file manuscript source (NeurIPS 2026 style)
- `refs.bib`: bibliography
- `reproducibility.md`: map from tables/figures to scripts and result paths
- `status.md`: canonical claims, experiment stage, stop/go criteria

## Working rules

- Keep `main.tex` as the only build path for the manuscript.
- Tie submission-facing claims to dated outputs under `docs/research/sage_reg_results/` (and summaries you explicitly commit).
- Update `reproducibility.md` when benchmark outputs or script names change.
- Keep SAGE-Reg files separate from the sibling track: [SPT-Reg](../neurips_spt_reg/).

## Sister track

Shift-factored predictive transport (test-time): [papers/neurips_spt_reg/](../neurips_spt_reg/).

## Parent index

See [papers/README.md](../README.md) for both NeurIPS workspaces.

## Build

From repo root: `./papers/compile_tex.sh neurips_sage_reg` (see [neurips_tex/README.md](../neurips_tex/README.md)).
