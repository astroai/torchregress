# Real-Data Recommendation Guide

Use this page after the [Comparative Evidence Matrix](comparative_evidence_matrix.md) when you need
an adoption-oriented answer to:

- what we can claim today from measured evidence,
- which method family is the default starting point per task,
- what data evidence is still missing before decision-grade recommendations.

This guide is evidence-first and claim-bounded. It is generated from committed comparative artifacts.

<!-- REALDATA-RECOMMENDATION-GENERATED:START -->

_Generated provenance_: `tools/render_realdata_recommendation_guide.py:render_generated_section`
_Source artifact_: `reports/comparative_evidence_matrix_latest.json`
_Generated date_: `2026-06-11`

## Evidence Band Summary

- Synthetic only: `7`
- Real proxy: `9`
- Decision-grade real-data: `0`

## Claim Policy

- `Synthetic only`: claim feasibility and relative behavior under controlled synthetic assumptions.
- `Real proxy`: claim task-fit plausibility on at least one real-data proxy benchmark.
- `Decision-grade real-data`: claim deployment-facing recommendation within validated scope.

## Task-to-Method Recommendations

| Task | Start Methods | Evidence Band | Claim Boundary | Next Data Step |
|---|---|---|---|---|
| Robust regression / outliers | `HuberLoss`, `CauchyLoss`, `WeightedMSELoss` | `Synthetic only` | Algorithmic feasibility only; no domain transfer claims. | Needs broader real-domain coverage beyond synthetic and tabular comparison tasks. |
| Imbalanced / rare-target regression | `DensityWeightedLoss`, `PropensityWeightedLoss`, `LDSLoss` | `Synthetic only` | Algorithmic feasibility only; no domain transfer claims. | Needs additional real-data long-tail benchmarks beyond synthetic selection proxies. |
| Selection bias / long-tail with missing labels | `PropensityWeightedLoss`, `DensityWeightedLoss`, `WeightedMSELoss` | `Synthetic only` | Algorithmic feasibility only; no domain transfer claims. | Needs real-data selection-bias benchmarks beyond synthetic generation. |
| Output constraints + post-hoc calibration transforms | `BoundedHead`, `NonCrossingSort`, `VarianceTemperatureScaler` | `Synthetic only` | Algorithmic feasibility only; no domain transfer claims. | Needs additional domain benchmarks beyond synthetic stress tests. |
| Target transforms for skewed regression | `WeightedMSELoss`, `LogTransformLoss`, `BoxCoxTransformLoss` | `Synthetic only` | Algorithmic feasibility only; no domain transfer claims. | Needs real-data positive-target benchmarks beyond synthetic multiplicative-noise tasks. |
| Semi-supervised regression / limited labels | `PseudoLabelConsistencyLoss`, `PseudoLabelNLL`, `WeightedMSELoss` | `Synthetic only` | Algorithmic feasibility only; no domain transfer claims. | Current evidence is one real-data proxy benchmark; add domain-native SSL regression tracks. |
| Uncertain ground-truth + density-aware conformal | `SplitConformal`, `DensityConformal`, `PrevalenceAdjustedCP` | `Real proxy` | Task-fit and transfer plausibility; avoid production-readiness claims. | Includes one real-data proxy benchmark; needs domain-native uncertain-label datasets for stronger external validity. |
| Causal inference regression (DR ATE/CATE) | `dr_ate`, `dr_cate`, `naive difference-in-means` | `Real proxy` | Task-fit and transfer plausibility; avoid production-readiness claims. | Includes real-covariate proxy benchmarks; needs external treatment-effect datasets for stronger external validity. |
| Calibrated intervals / coverage | `ConformalLoss`, `QuantileLoss`, `GaussianNLLLoss` | `Real proxy` | Task-fit and transfer plausibility; avoid production-readiness claims. | Coverage evidence now spans ensemble/SWAG/BNN base models; still needs multi-domain real-data calibration benchmarks under stronger shift. |
| Ordinal regression / ordered targets | `OrdinalCrossEntropyLoss`, `CumulativeLinkLoss`, `CORALLoss` | `Real proxy` | Task-fit and transfer plausibility; avoid production-readiness claims. | Includes one quantile-binned real-data benchmark; needs domain-native ordinal-label datasets for stronger external validity. |
| Censored / interval-censored regression | `CensoredGaussianNLLLoss`, `CensoredQuantileLoss`, `AFTLoss` | `Real proxy` | Task-fit and transfer plausibility; avoid production-readiness claims. | Includes one real-data synthetic-censoring benchmark; needs naturally censored datasets for stronger external validity. |
| OOD robustness / selective prediction | `DeepEnsemble`, `HeteroscedasticEnsembleModel`, `MCDropoutWrapper` | `Real proxy` | Task-fit and transfer plausibility; avoid production-readiness claims. | Needs multiple real-data OOD/selective benchmarks (beyond one covariate-shift proxy) for stronger external validity and regression tracking. |
| Multimodal / multi-target non-Gaussian | `GaussianNLLLoss`, `MDNLoss`, `NormalizingFlowLoss` | `Real proxy` | Task-fit and transfer plausibility; avoid production-readiness claims. | Needs domain-specific real-data multimodal benchmark(s) (beyond synthetic multimodal targets on real covariates) and optional-dependency CI coverage for zuko flow runs. |
| Noisy features / EIV | `FunctionalEIVLoss`, `StructuralEIVLoss`, `OrthogonalDistanceRegressionLoss` | `Real proxy` | Task-fit and transfer plausibility; avoid production-readiness claims. | Needs additional larger-scale/nonlinear real-data benchmarks (beyond Diabetes and one astronomical domain benchmark) for stronger external validity. |
| Noisy labels / corruption | `WeightedHuberLoss`, `CauchyLoss`, `TukeyBiweightLoss` | `Real proxy` | Task-fit and transfer plausibility; avoid production-readiness claims. | Needs comparisons against explicit noisy-label algorithms (co-teaching / sample-weight meta-learning) if/when implemented, plus more than one real dataset for stronger external validity. |
| Low-shot linear adaptation on fixed features (last layer) | `BayesianLinearHead`, `RecursiveBayesianHead`, `WeightedMSELoss` | `Synthetic only` | Algorithmic feasibility only; no domain transfer claims. | Synthetic Gaussian linear benchmarks only; needs real frozen-backbone protocols before deployment-grade ranking against ensembles or SWAG. |
<!-- REALDATA-RECOMMENDATION-GENERATED:END -->

## Usage

- Start with [Task-First Method Selection](../guide/method-selection.md) to shortlist viable APIs.
- Use this guide to set claim boundaries (`synthetic`, `real proxy`, `decision-grade real-data`).
- For deployment-facing decisions, require at least `Decision-grade real-data` evidence.

## Regeneration

```bash
uv run python -m tools.render_realdata_recommendation_guide \
  --doc docs/reports/real_data_recommendation_guide.md \
  --comparative-json reports/comparative_evidence_matrix_latest.json
```
