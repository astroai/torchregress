# Comparative Evidence Matrix

_Generated provenance_: `tools/render_method_catalog.py:render_comparative_evidence_markdown`
_Source artifacts_: `reports/comparative_evidence_matrix_latest.json`, `reports/method_catalog_latest.json`
_Generated date_: `2026-08-24`

## Summary

- Task rows: `15`
- Decision-grade rows: `2`
- Strong-or-better rows: `15`

## Coverage Matrix

| Task | Evidence Grade | Examples | Fairness Controls | Metrics Coverage | Peer Methods Visible | Gaps |
|---|---|---|---|---|---|---|
| Robust regression / outliers | `Decision-grade` | `examples/comprehensive_comparison.py`, `examples/comprehensive_loss_comparison.py` | fixed seeds, shared scenarios / domain split, runtime summaries | MSE, MAE, R2, runtime | `WeightedHuberLoss`, `PseudoHuberLoss`, `LogCoshLoss`, `CharbonnierLoss`, `CauchyLoss`, `TukeyBiweightLoss`, `BarronLoss`, `AdaptiveRobustLoss`, `CVaRLoss`, `WeightedMSELoss` | Needs broader real-domain coverage beyond synthetic and tabular comparison tasks. |
|  |  |  |  |  |  | _Note_: Robust/probabilistic/EIV comparisons run under shared budgets; results are available in the reports directory. |
| Imbalanced / rare-target regression | `Strong` | `examples/imbalanced_regression.py` | shared split, summary tables | tail vs dense metrics, tail MAE/RMSE, native interval coverage/width, runtime | `DensityWeightedLoss`, `PropensityWeightedLoss`, `LDSLoss`, `GaussianNLLLoss`, `MultiQuantileLoss` | Needs additional real-data long-tail benchmarks beyond synthetic selection proxies. |
| Output constraints + post-hoc calibration transforms | `Strong` | `examples/constraints_calibration_comparison.py` | fixed seed, shared synthetic split, shared calibrator fit budget | MAE, NLL, PIT chi-square, crossing/bound violations, runtime | `BoundedHead`, `NonCrossingSort`, `VarianceTemperatureScaler`, `IsotonicMeanCalibrator`, `PITCalibrator` | Needs additional domain benchmarks beyond synthetic stress tests. |
| Target transforms for skewed regression | `Strong` | `examples/transformed_target_regression_comparison.py` | fixed seed, shared architecture, shared optimizer/epoch budget | MSE, MAE, R2, MAPE, upper-tail MAE, runtime | `WeightedMSELoss`, `LogTransformLoss`, `BoxCoxTransformLoss`, `SqrtTransformLoss` | Needs real-data positive-target benchmarks beyond synthetic multiplicative-noise tasks. |
|  |  |  |  |  |  | _Note_: Focused evidence for target-transform choice under skewed positive targets. |
| Semi-supervised regression / limited labels | `Strong` | `examples/semi_supervised_regression_comparison.py` | fixed seed, shared architecture, shared labeled/unlabeled split, shared teacher bootstrap | MSE, MAE, R2, pseudo-label acceptance, runtime | `PseudoLabelConsistencyLoss`, `PseudoLabelNLL`, `WeightedMSELoss` | Current evidence is one real-data proxy benchmark; add domain-native SSL regression tracks. |
|  |  |  |  |  |  | _Note_: Real-data proxy benchmark uses Diabetes with masked labels and confidence-gated pseudo labels. |
| Uncertain ground-truth + density-aware conformal | `Strong` | `examples/semi_supervised_regression_comparison.py`, `examples/uncertain_gt_density_conformal_comparison.py`, `examples/uncertain_gt_density_conformal_realdata_comparison.py` | fixed seed, shared synthetic/real-data split, shared calibration budget | coverage, interval width, noisy-target NLL, consistency loss, pseudo-label NLL, runtime | `SplitConformal`, `DensityConformal`, `PrevalenceAdjustedCP`, `MonteCarloConformal`, `NoisyTargetGaussianNLL`, `PseudoLabelNLL`, `ConsistencyRegLoss`, `PseudoLabelConsistencyLoss` | Includes one real-data proxy benchmark; needs domain-native uncertain-label datasets for stronger external validity. |
|  |  |  |  |  |  | _Note_: Evidence now includes synthetic conformal tracks plus a real-data masked-label SSL proxy benchmark and real-data (Diabetes) uncertain-label evaluation. |
| Causal inference regression (DR ATE/CATE) | `Strong` | `examples/causal_dr_uplift_comparison.py`, `examples/causal_dr_realdata_comparison.py` | fixed seed, shared synthetic/real-covariate scenarios, matched nuisance models | ATE absolute error, CI contains true ATE, CI width, overlap rate, effective sample size, runtime | `dr_ate`, `dr_cate` | Includes real-covariate proxy benchmarks; needs external treatment-effect datasets for stronger external validity. |
|  |  |  |  |  |  | _Note_: Evidence includes synthetic uplift/astronomy-style scenarios plus real-covariate Diabetes proxy scenarios, all with cross-fitted DR estimators and overlap diagnostics. A naive difference-in-means baseline is reported alongside ``dr_ate``/``dr_cate`` in both comparison examples (see ``_naive_ate`` helper) for reference. |
| Calibrated intervals / coverage | `Strong` | `examples/evaluate_conformal_methods.py`, `examples/ood_selective_prediction_comparison.py`, `examples/ood_selective_prediction_realdata_comparison.py` | shared evaluation budget, coverage/width summaries, shared train budget | coverage, interval width, runtime, domain error metrics | `ConformalLoss`, `QuantileLoss`, `GaussianNLLLoss`, `BaseEnsembleModel`, `SWAG`, `BayesianNeuralNetwork` | Coverage evidence now spans ensemble/SWAG/BNN base models; still needs multi-domain real-data calibration benchmarks under stronger shift. |
|  |  |  |  |  |  | _Note_: OOD/selective comparisons now include split-conformal interval diagnostics across ensemble/MCDropout/SWAG/BNN. |
| Ordinal regression / ordered targets | `Strong` | `examples/ordinal_regression_comparison.py`, `examples/ordinal_regression_realdata_comparison.py`, `examples/ordinal_uncertain_ground_truth_comparison.py` | fixed seed, shared synthetic/real-data split, matched model capacity | accuracy, ordinal class MAE, QWK, true-class NLL, runtime | `OrdinalCrossEntropyLoss`, `CumulativeLinkLoss`, `CORALLoss` | Includes one quantile-binned real-data benchmark; needs domain-native ordinal-label datasets for stronger external validity. |
|  |  |  |  |  |  | _Note_: Comparison examples evaluate hard-label, soft-plausibility, and cumulative objectives under shared training budgets and ordinal-aware metrics. |
| Censored / interval-censored regression | `Strong` | `examples/censored_regression_comparison.py`, `examples/censored_regression_realdata_comparison.py` | fixed seed, shared synthetic/real-data censoring split, matched model capacity | true-target MAE, observed MAE, concordance index, runtime | `CensoredGaussianNLLLoss`, `CensoredQuantileLoss`, `AFTLoss` | Includes one real-data synthetic-censoring benchmark; needs naturally censored datasets for stronger external validity. |
|  |  |  |  |  |  | _Note_: Examples include right/left censoring plus explicit interval-censored samples under shared training budgets on synthetic and real-data tracks. |
| OOD robustness / selective prediction | `Decision-grade` | `examples/ood_selective_prediction_comparison.py`, `examples/ood_selective_prediction_realdata_comparison.py`, `examples/comprehensive_comparison.py` | fixed seeds, shared synthetic ID/OOD splits, shared train budget | ID/OOD MSE, AURC, rejection policy, OOD uncertainty gap, runtime | `BaseEnsembleModel`, `HeteroscedasticEnsembleModel`, `MCDropoutWrapper`, `SWAG`, `BayesianNeuralNetwork` | Needs multiple real-data OOD/selective benchmarks (beyond one covariate-shift proxy) for stronger external validity and regression tracking. |
|  |  |  |  |  |  | _Note_: Dedicated OOD/selective examples now include synthetic and real-data (Diabetes covariate-shift proxy) comparisons with SWAG and BNN under shared budgets; committed smoke/audit/full JSON summary artifacts are available. |
| Multimodal / multi-target non-Gaussian | `Strong` | `examples/multimodal_method_comparison.py`, `examples/multimodal_method_realdata_comparison.py`, `examples/normalizing_flows_multitarget.py` | fixed seeds, shared synthetic / real-covariate multimodal splits, shared train budget | point metrics, NLL, energy score, marginal calibration error, runtime | `GaussianNLLLoss`, `MDNLoss`, `NormalizingFlowLoss` | Needs domain-specific real-data multimodal benchmark(s) (beyond synthetic multimodal targets on real covariates) and optional-dependency CI coverage for zuko flow runs. |
|  |  |  |  |  |  | _Note_: Dedicated comparisons now include synthetic and real-covariate tracks; flow path is optional and may be skipped in minimal environments without zuko. |
| Noisy features / EIV | `Strong` | `examples/eiv_method_comparison.py`, `examples/eiv_method_realdata_comparison.py` | fixed seeds, shared synthetic/real-data measurement-error splits, shared train budget | clean test MSE, observed test MSE, stress MSE, runtime | `FunctionalEIVLoss`, `StructuralEIVLoss`, `OrthogonalDistanceRegressionLoss`, `EnsembleEIVLoss` | Needs additional larger-scale/nonlinear real-data benchmarks (beyond Diabetes and one astronomical domain benchmark) for stronger external validity. |
|  |  |  |  |  |  | _Note_: EIV comparisons now include synthetic and real-data (Diabetes with synthetic measurement-error injection) tracks, plus an astronomical benchmark with SDSS-style feature errors. |
| Noisy labels / corruption | `Strong` | `examples/noisy_label_comparison.py`, `examples/noisy_label_realdata_comparison.py`, `examples/comprehensive_loss_comparison.py` | fixed seeds, shared corrupted train/cal/test splits, shared train budget | clean/observed point metrics, split-conformal coverage/width, interval score, runtime | `WeightedHuberLoss`, `PseudoHuberLoss`, `LogCoshLoss`, `CharbonnierLoss`, `CauchyLoss`, `TukeyBiweightLoss`, `BarronLoss`, `AdaptiveRobustLoss`, `CVaRLoss`, `GaussianNLLLoss`, `MultiQuantileLoss` | Needs comparisons against explicit noisy-label algorithms (co-teaching / sample-weight meta-learning) if/when implemented, plus more than one real dataset for stronger external validity. |
|  |  |  |  |  |  | _Note_: Dedicated runnable comparisons now include synthetic and real-data (Diabetes with synthetic corruption) calibration-aware evaluations via shared split-conformal metrics. |
| Low-shot linear adaptation on fixed features (last layer) | `Strong` | `examples/benchmarks/bayesian_linear_head_lowshot_adaptation.py`, `examples/benchmarks/bayesian_linear_head_online_drift.py` | fixed seeds, matched ridge L2 vs conjugate prior in low-shot script, tabular stdout summaries (RMSE / NLL / streaming RMSE) | RMSE, Gaussian NLL, streaming RMSE under forgetting | `BayesianLinearHead`, `RecursiveBayesianHead`, `WeightedMSELoss` | Synthetic Gaussian linear benchmarks only; needs real frozen-backbone protocols before deployment-grade ranking against ensembles or SWAG. |
|  |  |  |  |  |  | _Note_: CPU-only benchmarks: low-shot path aligns BLR posterior mean with ridge MAP under matched L2 and contrasts predictive NLL with an oracle homoscedastic ridge baseline; drift path evaluates recursive partial_fit versus a phase-only batch oracle. |

## Programmatic Access

Use the public catalog metadata to inspect comparative evidence coverage from code:

```python
import torchregress as tr

rows = tr.method_catalog.list_comparative_evidence_rows()
for row in rows:
    print(row['task'], row['comparison_grade'])
```

## Usage

- Use this page with the [Task-First Method Selection Matrix](../guide/method-selection.md) to distinguish API support from decision-grade comparative evidence.
- Prioritize filling `Missing` and `Demo-only` rows before making strong performance or method-selection claims in docs.
