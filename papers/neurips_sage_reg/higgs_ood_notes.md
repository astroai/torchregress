# SAGE-Reg Higgs OOD Notes

## Why Higgs

The FAIR Universe Higgs Uncertainty Challenge is a strong fit for the next
SAGE-Reg benchmark step because it explicitly centers:

- uncertainty in the training inputs
- credible confidence intervals
- systematic and statistical uncertainty together

That makes it a better long-run OOD target than a generic tabular covariate
shift benchmark.

## What Is Implemented Now

`examples/benchmarks/self_agreement_higgs_ood.py` is a narrow prototype:

- Gaussian head only
- supervised only
- confidence-weighted pseudo-labeling
- SAGE-Reg

It supports:

- a local `dataset_path` for later real Higgs-style tabular data
- a built-in Higgs-like proxy when no local dataset is provided

The key diagnostics are not just ID/OOD predictive metrics, but also:

- `MeanWeightID` vs `MeanWeightOOD`
- `MeanDisagreementID` vs `MeanDisagreementOOD`

Those are the quantities that test the SAGE-Reg story directly:

- OOD-like unlabeled samples should exhibit larger disagreement
- therefore they should receive smaller agreement weights

## Current Role

This is a **proxy benchmark**, not yet a claim on the official Higgs challenge
data. The proxy exists so the code path, metrics, and artifact layout are in
place before the heavier real-data integration work.

## Next Integration Step

Once a local Higgs challenge table or derived tabular subset is available, use
the same benchmark script with `--dataset-path ...` and an OOD score column or
feature-based shift rule. That keeps the benchmark logic stable while swapping
in the real challenge data.
