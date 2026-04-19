# SAGE/SPT claim-to-code matrix (NeurIPS 2026 cycle)

This matrix locks paper claims to concrete code paths and generated artifacts so
submission language stays evidence-bounded.

## SAGE-Reg

| Claim bucket | Allowed claim text | Code path(s) | Artifact path(s) |
|---|---|---|---|
| Method existence | Distributional self-agreement SSL is implemented for regression in `torchregress` | `torchregress/semi_supervised.py`; `examples/benchmarks/self_agreement_realdata_year.py`; `examples/benchmarks/self_agreement_higgs_ood.py` | `docs/research/sage_reg_results/*/neurips_sage_reg_full/sage/year_direct/summary.json` |
| Safety vs scalar confidence | Safer unlabeled signal than scalar confidence-weighted pseudo-labeling | `examples/benchmarks/self_agreement_realdata_year.py` (`ConfidenceWeightedPseudoLabel` vs `SAGE-Reg`) | `.../sage/year_direct/summary.json`; `.../sage/multiseed/multiseed_summary.json` |
| Label-scarce IID regime | Benefits are strongest in low-label regime (not universal superiority) | `scripts/run_neurips_sage_reg_full.py` (labeled sweep phase) | `.../year_labeled_sweep/year_labeled_sweep_collated.json` |
| Variance-aware reporting | Seed variance is explicitly reported for SAGE vs supervised | `examples/benchmarks/self_agreement_supervised_gap_multiseed.py`; `tools/aggregate_sage_paper_report.py` | `.../sage/multiseed/multiseed_summary.json`; `.../sage_paper_report.json` |
| OOD stress support | Higgs is a stress-test/robustness support track, not main IID proof | `examples/benchmarks/self_agreement_higgs_ood.py`; `examples/benchmarks/self_agreement_supervised_gap_multiseed.py` | `.../sage/multiseed/*/higgs_confirm_summary.json` |

## SPT-Reg

| Claim bucket | Allowed claim text | Code path(s) | Artifact path(s) |
|---|---|---|---|
| Method existence | Shift-factored predictive transport is implemented end-to-end | `torchregress/test_time/transport.py`; `examples/spt_reg_year_comparison.py`; `tools/render_spt_reg_paper_artifacts.py` | `reports/neurips_spt_reg/*competing_methods*.json` |
| Decomposition transparency | Transport-only vs conformal-only vs full SPT are separated in benchmark rows | `examples/spt_reg_year_comparison.py` (rows: `PriorTransportGaussian`, `RawSplitConformalGaussian`, `SPTTransportGaussian`, `SPTRegGaussian`) | `reports/neurips_spt_reg/year_competing_methods_*.json` |
| Submission gating | Do not claim primary-paper readiness unless matched-validity efficiency wins on real data are shown | `papers/neurips_spt_reg/status.md`; `docs/research/paper_strong_experiment_suite.md` | `reports/neurips_spt_reg/README.md`; dated run trees under `reports/neurips_spt_reg/runs/` |

## Latest-commit impact boundary

- Commit `8f76ddc` expands method/tooling surface (new losses, OT-conformal,
  bayesian head, packed ensemble) but does not directly modify core SAGE method
  code in `torchregress/semi_supervised.py` or core SPT transport in
  `torchregress/test_time/transport.py`.
- Therefore manuscript claims should treat this as capability expansion for
  comparator depth and robustness checks, not as direct central-claim evidence.

## Submission-boundary rule

- SAGE primary paper: claim low-label, variance-aware, safety-first evidence.
- SPT remains research-only unless real-data matched-validity efficiency wins
  are robust and multi-seed supported.
