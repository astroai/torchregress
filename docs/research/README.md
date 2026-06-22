# Research notes (`docs/research/`)

Internal memos, dated result trees, and experiment plans for torchregress research tracks.

**Paper workspaces (symmetric layout):** see `papers/README.md` in the sibling `torchregress-research` repository.

## NeurIPS-scale tracks

| Track | Paper folder | Status | Key library modules |
|-------|----------------|--------|---------------------|
| **SPT-Reg** (shift-factored predictive transport) | `papers/neurips_spt_reg/` (in `torchregress-research`) | `status.md` (in `torchregress-research`) | `torchregress.test_time.transport`, `torchregress.prediction` |
| **SAGE-Reg** (self-agreement distributional self-training) | `papers/neurips_sage_reg/` (in `torchregress-research`) | `status.md` (in `torchregress-research`) | `torchregress.semi_supervised`, `torchregress.prediction` |

**Cross-track empirical roadmap** (stronger real-data evidence on both): `joint_empirical_priorities.md` (in `torchregress-research`).

**Long-run competitiveness suite** (baselines, multi-dataset, tiers): `paper_strong_experiment_suite.md` (in `torchregress-research`).

**Draft implementation plans** (not product specs): `plans/` (in `torchregress-research`).

**CURE rejected paper → datasets / metrics / baselines crosswalk** (TabReD, Wild-Tab, CatBoost(RaC) vs Reg): `cure_rejected_paper_crosswalk.md` (in `torchregress-research`).

## SAGE-Reg pointers

- Experiment staging: `sage_reg_experiment_plan.md` (in `torchregress-research`)
- Working outline / math sketch: `sage_reg_paper_outline.md` (in `torchregress-research`)
- Dated runs: `sage_reg_results/` (large files may be local-only). If **parquet / zip / misplaced ``openml_year.csv``** land here (tens of GB), remove with ``uv run python tools/clean_sage_reg_results_heavy_artifacts.py`` (``--dry-run`` first).
- Legacy redirect stub: `sage_reg_status.md` (in `torchregress-research`)

## SPT-Reg pointers

- Artifact render tool: `tools/render_spt_reg_paper_artifacts.py`
- Default generated summaries (local unless whitelisted in `.gitignore`): `reports/neurips_spt_reg/`
- Shared dataset notes: `papers/neurips_dataset_registry.md` (in `torchregress-research`)

## Examples (benchmarks)

SAGE-Reg paper scripts: `examples/benchmarks/README.md` (in `torchregress-research`)
SPT-Reg entrypoints: `examples/spt_reg_*.py`
