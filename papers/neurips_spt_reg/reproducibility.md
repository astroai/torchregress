# Reproducibility Map

This file links each manuscript artifact to generated outputs on the branch.

## Planned Main Artifacts

- Main synthetic table: `reports/neurips_spt_reg/synthetic_competing_methods_<profile>.json`
  - generator: `examples/spt_reg_synthetic_comparison.py`
- Main small-tabular table: `reports/neurips_spt_reg/tabular_competing_methods_<profile>.json`
  - generator: `examples/spt_reg_realdata_comparison.py`
- Main large-tabular table: `reports/neurips_spt_reg/year_competing_methods_<profile>.json`
  - generator: `examples/spt_reg_year_comparison.py`
- Optional photo-z compatibility table in `torchregress`: `reports/neurips_spt_reg/photoz_competing_methods_<profile>.json`
  - generator: `examples/spt_reg_photoz_comparison.py`
  - render only with `--include-photoz`
- Calibration/coverage figure: `reports/neurips_spt_reg/fig_calibration_coverage.png`
- PPI efficiency figure: `reports/neurips_spt_reg/fig_ppi_efficiency.png`
- Artifact manifest: `reports/neurips_spt_reg/artifact_manifest_latest.json`

## Notes

- Primary render entrypoint:

```bash
uv run python tools/render_spt_reg_paper_artifacts.py --profile smoke
```

- Optional photo-z compatibility render:

```bash
uv run python tools/render_spt_reg_paper_artifacts.py --profile smoke --include-photoz
```

- Profiles currently supported: `smoke`, `audit`, and `full`.
- Shared dataset/source registry: `papers/neurips_dataset_registry.md`
- Keep filenames stable after they are cited in the paper.
- The default render path now covers one synthetic track plus two real-data tracks in `torchregress`.
