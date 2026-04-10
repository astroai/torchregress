# SAGE-Reg Results (2026-04-09)

This directory stores benchmark artifacts produced during the current SAGE-Reg
paper-prototype pass.

## Synthetic confidence-trap run

- `synthetic_confidence_trap.csv`
- `synthetic_confidence_trap_perf.png`
- `synthetic_confidence_trap_calib.png`
- `synthetic_confidence_trap_diag.png`
- `synthetic_confidence_trap_summary.json`

This run is the tightened Stage 3 synthetic stress test where confidence can
mis-rank pseudo-label quality and disagreement weighting has to matter.

## Higgs-inspired OOD proxy run

- `higgs_ood_proxy.csv`
- `higgs_ood_proxy_perf.png`
- `higgs_ood_proxy_calib.png`
- `higgs_ood_proxy_summary.json`

This run is the first OOD benchmark step after the real-data `year` benchmark.
It uses the built-in Higgs-like proxy and records whether SAGE-Reg downweights
OOD unlabeled samples more strongly than ID unlabeled samples.
