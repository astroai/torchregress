# Benchmark scripts (`examples/benchmarks/`)

## SAGE-Reg (self-agreement / semi-supervised)

These drive the SAGE-Reg NeurIPS track; see [papers/neurips_sage_reg/](../../papers/neurips_sage_reg/).

| Script | Purpose |
|--------|---------|
| `self_agreement_synthetic.py` | Synthetic confidence-trap and agreement stress tests |
| `self_agreement_backbone_comparison.py` | Gaussian vs quantile vs bar heads |
| `self_agreement_realdata_year.py` | OpenML Year real-data benchmark |
| `self_agreement_higgs_ood.py` | FAIR Universe Higgs public OOD-style benchmark |
| `self_agreement_supervised_gap_tuning.py` | Hyperparameter sweep for supervised-gap experiments |
| `self_agreement_supervised_gap_confirm.py` | Fixed-config confirmation from a sweep CSV |
| `self_agreement_supervised_gap_multiseed.py` | Multi-seed aggregation over confirms |
| `tabred_sage_ssl_probe.py` | TabReD (preprocessed `.npy`) × SAGE vs baselines; loads wide ``X`` with **Polars** (requires `torchregress[tabred]`); morning script runs [fetch](../../tools/fetch_tabred_data.py) then [launcher](../../scripts/morning_tabred_bundle.sh) |

Reproducibility and data paths: [papers/neurips_sage_reg/reproducibility.md](../../papers/neurips_sage_reg/reproducibility.md).
Paper index: [papers/README.md](../../papers/README.md).

## Other benchmarks

Additional `examples/benchmarks/*.py` files may serve CI smoke tests or library comparisons; read each module docstring for scope.

## Conventions

- Run from repo root with `uv run python examples/benchmarks/<script>.py …` so imports and relative data paths resolve predictably.
- Prefer **repository-relative** paths in docs and shell history (not machine-specific absolute paths).
