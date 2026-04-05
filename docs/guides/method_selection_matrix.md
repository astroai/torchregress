# Task-First Method Selection Matrix

This guide is the user-facing version of the adoption audit capability matrix.

Design intent:

- Start from the problem you have, not a modeling ideology
- Keep `SWAG`, `BNN`, and `MDN` as peer methods in the catalog
- Use evidence-based maturity labels (tests/docs/examples), not family-based labels
- Keep Bayesian concepts optional for onboarding

## How To Use This Page

1. Find your primary problem in the task matrix.
2. Start with the recommended defaults.
3. Use the method matrix to shortlist alternatives when you need:
   - multimodal outputs
   - explicit decomposition (epistemic vs aleatoric)
   - stronger tail/imbalance handling
   - calibrated intervals / conformal guarantees
   - OOD scoring

```python
import torchregress as tr

# Start from the task/problem you have:
# - robust regression
# - calibration / conformal intervals
# - multimodal targets (MDN / flows)
# - OOD / selective prediction
#
# Then choose a method family and validate with metrics/examples.
loss = tr.losses.HuberLoss()
```

## Task Matrix (Recommended Starting Points)

<!-- BEGIN:TASK_MATRIX_GENERATED -->
_Generated provenance_: `tools/render_method_catalog.py:render_task_matrix_table`
_Source artifacts_: `reports/method_catalog_latest.json`
_Generated date_: `2026-04-05`

| Task / Constraint | Recommended Start | Strong Alternatives | Notes |
|---|---|---|---|
| Clean regression baseline | `WeightedMSELoss` | `HuberLoss` | Start simple; add UQ only if needed. |
| Outliers / robust regression | `HuberLoss` | `CauchyLoss`, `TukeyBiweightLoss`, `CharbonnierLoss` | Huber is the best default tradeoff. |
| Heteroscedastic noise (aleatoric UQ) | `GaussianCRPSLoss` | `GaussianNLLLoss`, `HeteroscedasticEnsembleModel`, `MDNLoss` | Photo-z benchmarks favor CRPS-trained Gaussian heads as the safest calibrated Gaussian baseline. |
| Epistemic uncertainty | `DeepEnsemble` | `HeteroscedasticBatchEnsembleModel`, `BinnedPDFEnsembleModel`, `MDNEnsembleModel`, `SWAG`, `BayesianNeuralNetwork`, `MCDropoutWrapper` | Deep ensembles are easiest operationally. |
| Epistemic + aleatoric decomposition | `HeteroscedasticEnsembleModel` | `HeteroscedasticBNN`, `MDNLoss`, `NormalizingFlowLoss` | Requires variance/distribution modeling. |
| Multimodal targets | `MDNLoss` | `MDNEnsembleModel`, `BinnedPDFEnsembleModel`, `NormalizingFlowLoss` | MDN is usually easier to debug first; ensembles of MDN or ordered-bin heads are the next move when mode averaging matters. |
| Non-Gaussian / skewed tails | `QuantileLoss` / `ExpectileLoss` / `TweedieLoss` | `MDNLoss`, `NormalizingFlowLoss` | Choose by target support and evaluation metric. |
| Multi-target correlated outputs | `MultivariateGaussianLoss` / `LowRankGaussianLoss` | `MDNLoss`, `NormalizingFlowLoss` | Prefer low-rank/full covariance when Gaussian is enough. |
| Noisy features / measurement error | `InputNoiseMarginalizationLoss + GaussianCRPSLoss` / `MDNLoss` / `BinnedPDF` | `FunctionalEIVLoss`, `StructuralEIVLoss`, `OrthogonalDistanceRegressionLoss` | Start with explicit input-noise marginalization and test-time predictive averaging, then escalate to Jacobian-based EIV losses only if they clearly help. |
| Noisy labels / label corruption | `HuberLoss` | `DeepEnsemble`, `ConformalLoss` | Prefer robust baselines before heavier methods. |
| Imbalanced / rare-target regression | `GaussianCRPSLoss` / `QuantileLoss + tail-slice evaluation` | `DensityConformal` | Photo-z benchmarks do not justify density weighting as default. Advanced research methods (DensityWeightedLoss, LDSLoss) should only be tried if coverage/calibration allow for tail gains. |
| Selection bias / covariate-dependent missing labels | `PropensityWeightedLoss` | `DensityWeightedLoss` | Estimate p(observed|x) and apply IPW to reduce selection bias. |
| Output constraints / monotonicity | `BoundedHead` / `NonNegativeHead` / `NonCrossingSort` | `SimplexHead`, `SpectralNormWrapper` | Apply structural constraints in the head before post-hoc calibration. |
| Post-hoc calibration transforms | `VarianceTemperatureScaler` | `IsotonicMeanCalibrator`, `PITCalibrator` | Fit transforms on a held-out calibration split. |
| Calibrated intervals with coverage guarantees | `ConformalLoss on top of a strong probabilistic backbone` | `QuantileLoss`, `MonteCarloConformal`, `DensityConformal` | Conformal gives coverage, not density estimation; keep CRPS/NLL reporting from the underlying predictive model. |
| Density-aware conformal under long-tail targets | `DensityConformal` | `PrevalenceAdjustedCP`, `MonteCarloConformal` | Prefer density/prevalence variants when tail-region coverage is a key objective; point-accuracy gains are not guaranteed. |
| Uncertain ground-truth / weak labels | `NoisyTargetGaussianNLL` | `OrdinalCrossEntropyLoss`, `PseudoLabelConsistencyLoss`, `PseudoLabelNLL` | Use NoisyTargetGaussianNLL for Gaussian label uncertainty and soft-bin / PMF supervision when target PDFs or intervals are the natural label form. |
| Semi-supervised regression | `PseudoLabelConsistencyLoss` | `PseudoLabelNLL`, `NoisyTargetGaussianNLL` | Use confidence-gated pseudo labels and keep a clean held-out evaluation split. |
| Target transforms for skewed / multiplicative-noise regression | `LogTransformLoss` | `BoxCoxTransformLoss`, `SqrtTransformLoss`, `YeoJohnsonTransformLoss` | Match transform support to target support before tuning model complexity. |
| Causal inference regression (ATE/CATE) | `dr_ate` / `dr_cate` | `PredictionPoweredInference` | Use cross-fitting and overlap diagnostics before interpreting treatment effects. |
| Population inference with few labels | `PredictionPoweredInference` | `ConformalLoss`, `QuantileLoss` | Use PPI for means/quantiles/regression coefficients with limited labels. |
| Ordinal / ordered targets | `CumulativeLinkLoss` | `CORALLoss`, `OrdinalCrossEntropyLoss` | Prefer cumulative objectives when rank-distance errors matter. |
| Censored / interval-censored regression | `CensoredGaussianNLLLoss` | `AFTLoss`, `CensoredQuantileLoss` | Use censoring code 0/1/-1 and explicit interval bounds when available. |
| OOD scoring / selective prediction | `DeepEnsemble + OOD metrics` | `HeteroscedasticBatchEnsembleModel + OOD metrics`, `SWAG + OOD metrics`, `BayesianNeuralNetwork + OOD metrics` | Use multiple signals; no single OOD score is sufficient. |
<!-- END:TASK_MATRIX_GENERATED -->

## Method Capability Matrix (Peer Framing)

Legend:

- `Yes`: directly supported
- `Partial`: supported with conditions / composition
- `No`: not a natural fit

<!-- BEGIN:FAMILY_CAPABILITY_MATRIX_GENERATED -->
_Generated provenance_: `tools/render_method_catalog.py:render_family_capability_matrix_table`
_Source artifacts_: `reports/method_catalog_latest.json`
_Generated date_: `2026-04-05`

| Method Family | Multi-target | Multimodal | Non-Gaussian | Epistemic | Aleatoric | Decomposition | Calibration | OOD Support | Imbalance | Noisy Features (EIV) |
|---|---|---|---|---|---|---|---|---|---|---|
| `bnn` (2) | yes | no | partial | yes | yes | yes | partial | partial | no | no |
| `calibration_transform` (3) | yes | no | partial | no | no | no | yes | partial | no | no |
| `causal` (2) | yes | no | partial | no | no | no | partial | partial | no | no |
| `censored` (3) | yes | no | yes | no | no | no | partial | partial | no | no |
| `conformal` (4) | yes | no | yes | partial | no | no | yes | partial | yes | no |
| `constraints` (2) | yes | no | partial | no | no | no | partial | partial | no | no |
| `eiv` (7) | yes | yes | partial | no | no | no | partial | partial | no | yes |
| `ensemble` (6) | yes | no | yes | yes | yes | yes | partial | yes | no | no |
| `flow` (1) | yes | yes | yes | no | yes | partial | partial | partial | no | no |
| `imbalanced_loss` (3) | yes | no | partial | no | no | no | partial | partial | yes | no |
| `inference` (1) | yes | no | partial | no | no | no | partial | partial | no | no |
| `mc_dropout` (1) | yes | no | partial | yes | partial | partial | partial | partial | no | no |
| `mdn` (1) | yes | yes | yes | no | yes | yes | partial | partial | no | no |
| `ordinal` (3) | yes | no | yes | no | no | no | partial | partial | no | no |
| `point_loss` (1) | yes | no | no | no | no | no | partial | partial | no | no |
| `probabilistic_loss` (3) | yes | no | partial | no | yes | no | partial | partial | no | no |
| `quantile` (1) | yes | no | yes | no | no | no | yes | partial | no | no |
| `robust_loss` (1) | yes | no | partial | no | no | no | partial | partial | no | no |
| `swag` (2) | yes | no | partial | yes | partial | partial | partial | partial | no | no |
| `target_transform` (4) | yes | no | partial | no | no | no | partial | partial | no | no |
| `uncertain_gt` (4) | yes | no | partial | no | yes | no | partial | partial | no | no |
<!-- END:FAMILY_CAPABILITY_MATRIX_GENERATED -->

## Code-Driven Catalog (Discovery API)

The tables above are task guidance. For scriptable filtering (and to reduce docs drift), use the
public catalog metadata API:

```python
import torchregress as tr

# Example: shortlist peer methods for OOD + epistemic signals
candidates = tr.method_catalog.list_methods(
    task_tag="ood",
    capability_filters={"epistemic": "yes"},
)

for row in candidates:
    print(row["name"], row["family"], row["maturity"])
```

Generated snapshot (from code): [`method_catalog_generated.md`](method_catalog_generated.md)

Next step for operational tradeoffs: [Choosing Methods by Constraint](choosing_by_constraint.md)

<!-- BEGIN:METHOD_CATALOG_GENERATED_SECTION -->
### Catalog-Backed Peer Method Snapshot (Generated)

_Generated provenance_: `tools/render_method_catalog.py:render_method_matrix_generated_section`
_Source artifacts_: `docs/guides/method_catalog_generated.md`, `reports/method_catalog_latest.json`
_Generated date_: `2026-04-05`

| Method | Family | Maturity | Multi-target | Multimodal | Epistemic | Aleatoric | Decomposition | Calibration | OOD |
|---|---|---|---|---|---|---|---|---|---|
| `BayesianNeuralNetwork` | `bnn` | `Available` | yes | no | yes | partial | partial | partial | partial |
| `HeteroscedasticBNN` | `bnn` | `Available` | yes | no | yes | yes | yes | partial | partial |
| `IsotonicMeanCalibrator` | `calibration_transform` | `Available` | yes | no | no | no | no | yes | partial |
| `PITCalibrator` | `calibration_transform` | `Available` | yes | no | no | no | no | yes | partial |
| `VarianceTemperatureScaler` | `calibration_transform` | `Available` | yes | no | no | no | no | yes | partial |
| `dr_ate` | `causal` | `Available` | yes | no | no | no | no | partial | partial |
| `dr_cate` | `causal` | `Available` | yes | no | no | no | no | partial | partial |
| `AFTLoss` | `censored` | `Available` | yes | no | no | no | no | partial | partial |
| `CensoredGaussianNLLLoss` | `censored` | `Available` | yes | no | no | no | no | partial | partial |
| `CensoredQuantileLoss` | `censored` | `Available` | yes | no | no | no | no | partial | partial |
| `ConformalLoss` | `conformal` | `Core` | yes | no | no | no | no | yes | partial |
| `DensityConformal` | `conformal` | `Available` | yes | no | no | no | no | yes | partial |
| `MonteCarloConformal` | `conformal` | `Available` | yes | no | partial | no | no | yes | partial |
| `PrevalenceAdjustedCP` | `conformal` | `Available` | yes | no | no | no | no | yes | partial |
| `BoundedHead` | `constraints` | `Available` | yes | no | no | no | no | partial | partial |
| `NonCrossingSort` | `constraints` | `Available` | yes | no | no | no | no | partial | partial |
| `FunctionalEIVLoss` | `eiv` | `Available` | yes | no | no | no | no | partial | partial |
| `InputNoiseBinnedPDFLoss` | `eiv` | `Available` | yes | yes | no | no | no | partial | partial |
| `InputNoiseMDNLoss` | `eiv` | `Available` | yes | yes | no | no | no | partial | partial |
| `InputNoiseMarginalizationLoss` | `eiv` | `Strong` | yes | no | no | no | no | partial | partial |
| `NoisyInputPredictor` | `eiv` | `Strong` | yes | no | no | no | no | partial | partial |
| `OrthogonalDistanceRegressionLoss` | `eiv` | `Available` | yes | no | no | no | no | partial | partial |
| `StructuralEIVLoss` | `eiv` | `Available` | yes | no | no | no | no | partial | partial |
| `BinnedPDFEnsembleModel` | `ensemble` | `Available` | yes | no | yes | partial | partial | partial | partial |
| `CumulativeLinkEnsembleModel` | `ensemble` | `Available` | yes | no | yes | partial | partial | partial | partial |
| `DeepEnsemble` | `ensemble` | `Core` | yes | no | yes | partial | partial | partial | yes |
| `HeteroscedasticBatchEnsembleModel` | `ensemble` | `Strong` | yes | no | yes | yes | yes | partial | partial |
| `HeteroscedasticEnsembleModel` | `ensemble` | `Strong` | yes | no | yes | yes | yes | partial | yes |
| `MDNEnsembleModel` | `ensemble` | `Available` | yes | no | yes | yes | yes | partial | partial |
| `NormalizingFlowLoss` | `flow` | `Available` | yes | yes | no | yes | partial | partial | partial |
| `DensityWeightedLoss` | `imbalanced_loss` | `Available` | yes | no | no | no | no | partial | partial |
| `LDSLoss` | `imbalanced_loss` | `Available` | yes | no | no | no | no | partial | partial |
| `PropensityWeightedLoss` | `imbalanced_loss` | `Available` | yes | no | no | no | no | partial | partial |
| `PredictionPoweredInference` | `inference` | `Available` | yes | no | no | no | no | partial | partial |
| `MCDropoutWrapper` | `mc_dropout` | `Strong` | yes | no | yes | partial | partial | partial | partial |
| `MDNLoss` | `mdn` | `Available` | yes | yes | no | yes | yes | partial | partial |
| `CORALLoss` | `ordinal` | `Available` | yes | no | no | no | no | partial | partial |
| `CumulativeLinkLoss` | `ordinal` | `Available` | yes | no | no | no | no | partial | partial |
| `OrdinalCrossEntropyLoss` | `ordinal` | `Available` | yes | no | no | no | no | partial | partial |
| `WeightedMSELoss` | `point_loss` | `Core` | yes | no | no | no | no | partial | partial |
| `GaussianNLLLoss` | `probabilistic_loss` | `Core` | yes | no | no | yes | no | partial | partial |
| `LowRankGaussianLoss` | `probabilistic_loss` | `Strong` | yes | no | no | yes | no | partial | partial |
| `MultivariateGaussianLoss` | `probabilistic_loss` | `Strong` | yes | no | no | yes | no | partial | partial |
| `QuantileLoss` | `quantile` | `Core` | yes | no | no | no | no | yes | partial |
| `HuberLoss` | `robust_loss` | `Core` | yes | no | no | no | no | partial | partial |
| `MultiSWAG` | `swag` | `Available` | yes | no | yes | partial | partial | partial | partial |
| `SWAG` | `swag` | `Available` | yes | no | yes | partial | partial | partial | partial |
| `BoxCoxTransformLoss` | `target_transform` | `Available` | yes | no | no | no | no | partial | partial |
| `LogTransformLoss` | `target_transform` | `Available` | yes | no | no | no | no | partial | partial |
| `SqrtTransformLoss` | `target_transform` | `Available` | yes | no | no | no | no | partial | partial |
| `YeoJohnsonTransformLoss` | `target_transform` | `Available` | yes | no | no | no | no | partial | partial |
| `ConsistencyRegLoss` | `uncertain_gt` | `Available` | yes | no | no | no | no | partial | partial |
| `NoisyTargetGaussianNLL` | `uncertain_gt` | `Available` | yes | no | no | yes | no | partial | partial |
| `PseudoLabelConsistencyLoss` | `uncertain_gt` | `Available` | yes | no | no | no | no | partial | partial |
| `PseudoLabelNLL` | `uncertain_gt` | `Available` | yes | no | no | no | no | partial | partial |

Peer-method check: `SWAG`, `BayesianNeuralNetwork`, `MDNLoss`

### Generated Family Capability Matrix

| Family | # Methods | Multi-target | Multimodal | Non-Gaussian | Epistemic | Aleatoric | Decomposition | Calibration | OOD | Imbalance | EIV |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| `bnn` | 2 | yes | no | partial | yes | yes | yes | partial | partial | no | no |
| `calibration_transform` | 3 | yes | no | partial | no | no | no | yes | partial | no | no |
| `causal` | 2 | yes | no | partial | no | no | no | partial | partial | no | no |
| `censored` | 3 | yes | no | yes | no | no | no | partial | partial | no | no |
| `conformal` | 4 | yes | no | yes | partial | no | no | yes | partial | yes | no |
| `constraints` | 2 | yes | no | partial | no | no | no | partial | partial | no | no |
| `eiv` | 7 | yes | yes | partial | no | no | no | partial | partial | no | yes |
| `ensemble` | 6 | yes | no | yes | yes | yes | yes | partial | yes | no | no |
| `flow` | 1 | yes | yes | yes | no | yes | partial | partial | partial | no | no |
| `imbalanced_loss` | 3 | yes | no | partial | no | no | no | partial | partial | yes | no |
| `inference` | 1 | yes | no | partial | no | no | no | partial | partial | no | no |
| `mc_dropout` | 1 | yes | no | partial | yes | partial | partial | partial | partial | no | no |
| `mdn` | 1 | yes | yes | yes | no | yes | yes | partial | partial | no | no |
| `ordinal` | 3 | yes | no | yes | no | no | no | partial | partial | no | no |
| `point_loss` | 1 | yes | no | no | no | no | no | partial | partial | no | no |
| `probabilistic_loss` | 3 | yes | no | partial | no | yes | no | partial | partial | no | no |
| `quantile` | 1 | yes | no | yes | no | no | no | yes | partial | no | no |
| `robust_loss` | 1 | yes | no | partial | no | no | no | partial | partial | no | no |
| `swag` | 2 | yes | no | partial | yes | partial | partial | partial | partial | no | no |
| `target_transform` | 4 | yes | no | partial | no | no | no | partial | partial | no | no |
| `uncertain_gt` | 4 | yes | no | partial | no | yes | no | partial | partial | no | no |

### Generated Hard-Task Shortlists

| Need | Catalog Filter (conceptual) | Suggested Methods |
|---|---|---|
| OOD + epistemic signals | `task_tag='ood'` + `epistemic=yes` | `BayesianNeuralNetwork`, `HeteroscedasticBNN`, `DeepEnsemble`, `HeteroscedasticBatchEnsembleModel`, `HeteroscedasticEnsembleModel`, `MultiSWAG`, `SWAG` |
| Coverage / calibration | `calibration=yes` | `IsotonicMeanCalibrator`, `PITCalibrator`, `VarianceTemperatureScaler`, `ConformalLoss`, `DensityConformal`, `MonteCarloConformal`, `PrevalenceAdjustedCP`, `QuantileLoss` |
| Multimodal targets | `multimodal=yes` | `InputNoiseBinnedPDFLoss`, `InputNoiseMDNLoss`, `NormalizingFlowLoss`, `MDNLoss` |
| Imbalanced / rare targets | `imbalance=yes` | `DensityConformal`, `PrevalenceAdjustedCP`, `DensityWeightedLoss`, `LDSLoss`, `PropensityWeightedLoss` |
| Noisy features / EIV | `noisy_features_eiv=yes` | `FunctionalEIVLoss`, `InputNoiseBinnedPDFLoss`, `InputNoiseMDNLoss`, `InputNoiseMarginalizationLoss`, `NoisyInputPredictor`, `OrthogonalDistanceRegressionLoss`, `StructuralEIVLoss` |
<!-- END:METHOD_CATALOG_GENERATED_SECTION -->

## Evidence-Based Maturity Labels (Current Audit Guidance)

These labels are intentionally evidence-based and implementation-specific.

| Label | Meaning |
|---|---|
| `Core` | Stable API pattern, direct tests, docs coverage, examples |
| `Strong` | Good coverage and examples, some edge-case/test debt remains |
| `Available` | Implemented and documented, but fewer examples/benchmarks or less direct test depth |
| `Advanced` | Powerful but needs stronger decision-grade examples/benchmarks for adoption confidence |

Current high-level guidance (family-level, not absolute):

- `Core`: robust point losses, Gaussian losses, basic metrics, conformal basics, deep ensembles
- `Strong`: calibration/OOD metrics, imbalanced losses, multi-target Gaussian variants
- `Available`: `SWAG`, `BNN`, `MDN`, normalizing-flow losses, EIV losses
- `Advanced`: flow-based multi-target comparisons, combined decomposition + calibration benchmarks

## Evidence and Tradeoffs (What To Check Before Committing)

Use this matrix to shortlist methods, then validate the shortlist against examples and
benchmarks:

- Comparison examples: `examples/comprehensive_comparison.py`,
  `examples/comprehensive_loss_comparison.py`
- Hard-task examples: `examples/imbalanced_regression.py`,
  `examples/evaluate_conformal_methods.py`,
  `examples/normalizing_flows_multitarget.py`
- Comparative coverage tracker: [Comparative Evidence Matrix](comparative_evidence_matrix.md)
- Performance smoke/sweep baselines: `tools/benchmark_smoke.py` and reports under
  `reports/benchmark_thresholds/`

Key tradeoffs and common failure modes:

- `Conformal`:
  strong coverage guarantees, but interval width can become large under shift or poor
  base models; does not provide epistemic/aleatoric decomposition.
- `MDN`:
  strong multimodal baseline, but can be sensitive to component count and optimization;
  compare interval/calibration behavior, not just NLL.
- `Normalizing flows`:
  flexible and expressive, but higher implementation/runtime complexity and optional
  dependency requirements; use when Gaussian/MDN miss important structure.
- `SWAG` / `BNN`:
  useful epistemic signals, but calibration and runtime quality depend heavily on training
  protocol and posterior approximation choices; validate on your OOD and calibration
  metrics, not only in-distribution error.
- `Deep / heteroscedastic ensembles`:
  strong defaults for uncertainty, but training and inference costs scale with ensemble
  size; benchmark with your latency budget.

## Decision Workflow (Fast Path)

<!-- BEGIN:DECISION_WORKFLOW_GENERATED -->
_Generated provenance_: `tools/render_method_catalog.py:render_decision_workflow`
_Source artifacts_: `reports/method_catalog_latest.json`
_Generated date_: `2026-04-05`

1. Need coverage guarantees?
   Use `ConformalLoss (split / CQR / ACI patterns)`.
   Caveat: Coverage guarantees are not uncertainty decomposition.

2. Need epistemic + aleatoric decomposition?
   Use `HeteroscedasticEnsembleModel`.
   Alternatives: `HeteroscedasticBNN`, `MDNLoss`, `NormalizingFlowLoss (+ ensemble)`.
   Caveat: Requires variance/distribution modeling and stronger compute budget.

3. Need multimodal outputs?
   Use `MDNLoss`.
   Alternatives: `NormalizingFlowLoss`.
   Caveat: Move to flows when MDN component count/training stability is the bottleneck.

4. Have noisy features / measurement error?
   Use `InputNoiseMarginalizationLoss + GaussianCRPSLoss / MDNLoss / BinnedPDF`.
   Alternatives: `FunctionalEIVLoss / StructuralEIVLoss / OrthogonalDistanceRegressionLoss`.
   Caveat: Use the simpler explicit input-noise path first, including test-time predictive averaging; Jacobian-style EIV losses are more fragile and need careful benchmarking.

5. Have imbalanced tails / rare targets?
   Use `GaussianCRPSLoss / QuantileLoss + tail-slice evaluation`.
   Alternatives: `DensityConformal`, `DensityWeightedLoss`, `LDSLoss`.
   Caveat: Density-aware weighting is not yet a universally strong default on photo-z benchmarks.

6. Need OOD scoring / selective prediction under a latency budget?
   Use `DeepEnsemble + OOD metrics`.
   Alternatives: `HeteroscedasticBatchEnsembleModel + OOD metrics`, `SWAG + OOD metrics`, `BayesianNeuralNetwork + OOD metrics`, `MCDropoutWrapper`.
   Caveat: Use multiple signals and benchmark runtime against deployment latency targets.

7. Are labels uncertain or weak (noisy targets, pseudo-labels, partial trust)?
   Use `NoisyTargetGaussianNLL`.
   Alternatives: `OrdinalCrossEntropyLoss`, `PseudoLabelNLL`, `ConsistencyRegLoss`.
   Caveat: Retain held-out clean-label evaluation where available to avoid self-confirming loops.

8. Need treatment-effect estimation under confounding (ATE/CATE)?
   Use `dr_ate / dr_cate`.
   Alternatives: `PredictionPoweredInference`.
   Caveat: Check overlap/ESS diagnostics and avoid causal claims under severe positivity violations.
<!-- END:DECISION_WORKFLOW_GENERATED -->

## Important Clarification: Conformal vs Uncertainty Decomposition

Conformal prediction provides interval coverage guarantees.

It does **not** separate epistemic and aleatoric uncertainty.

Use conformal for calibrated intervals.
Use heteroscedastic/ensemble/MDN/flow approaches when you need decomposition or richer predictive distributions.
