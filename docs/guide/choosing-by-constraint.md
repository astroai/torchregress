# Choosing Methods by Constraint

Use this guide after the [Task-First Method Selection Matrix](method-selection.md)
when you already know the task category but need to choose based on practical constraints:

- latency / compute budget
- coverage guarantees vs uncertainty decomposition
- multimodality requirements
- noisy features (measurement error)
- calibration and OOD robustness requirements

This page is audit-driven: recommendations are organized around adoption concerns
(reliability, performance, interpretability, deployment fit), not modeling ideology.
For evidence-grade claim boundaries, pair this page with the
[Real-Data Recommendation Guide](../reports/real_data_recommendation_guide.md).

## Fast Constraint Triage

1. **Need coverage guarantees?**
   Start with conformal methods (`split`, `CQR`, `ACI`) and evaluate interval width.
2. **Need epistemic/aleatoric decomposition?**
   Start with heteroscedastic ensembles, then compare `HeteroscedasticBNN` / `MDN`.
3. **Need multimodal outputs?**
   Start with `MDN`, then move to normalizing flows if MDN is too rigid.
4. **Need low operational complexity?**
   Start with robust point losses (`WeightedHuberLoss`) or Gaussian NLL before Bayesian/flow methods.
5. **Have noisy features / measurement error?**
   Use EIV/ODR losses; these change the call pattern (`loss(x_obs, y_obs)` with model in loss).

## Constraint Profiles

### 1. Low Latency / Limited Compute

Recommended order:

- `WeightedHuberLoss` / robust point losses
- `GaussianNLLLoss` (single-model aleatoric)
- MC Dropout (`MCDropoutWrapper`)
- Small deep ensembles (if latency budget allows)

Tradeoffs:

- Deep ensembles often improve epistemic uncertainty, but inference cost scales with ensemble size.
- `SWAG` / `BNN` can be attractive for epistemic signals, but require careful protocol tuning.
- Flow-based methods usually add the most implementation/runtime complexity.

## 2. Need Coverage Guarantees (Intervals) vs Decomposition

If you need guaranteed coverage:

- Use conformal methods for intervals and report:
  - empirical coverage
  - interval width
  - coverage under shift (if relevant)

If you need decomposition (epistemic vs aleatoric):

- Use heteroscedastic ensembles first
- Then compare `HeteroscedasticBNN`, `MDN`, or flow ensembles

Important:

- Conformal prediction gives coverage guarantees.
- Conformal prediction does **not** provide epistemic/aleatoric decomposition.

### 3. Multimodal / Non-Gaussian Targets

Recommended order:

- `MDN` (good first multimodal baseline)
- Quantile/expectile methods (if interval-focused and decomposition not required)
- Normalizing flows (when MDN or Gaussian families miss structure)

What to compare:

- calibration and interval quality (not just NLL)
- sensitivity to component count (`MDN`)
- runtime and memory (`MDN` vs flow)

## 4. Noisy Features / Measurement Error

Use EIV losses when the input is noisy and that noise is part of the problem statement:

- `FunctionalEIVLoss`
- `StructuralEIVLoss`
- `OrthogonalDistanceRegressionLoss`

Practical notes:

- These losses require a different training pattern than standard supervised losses.
- Start with a simpler baseline (`Huber`, Gaussian NLL) to quantify lift before adopting EIV.

## 5. Calibration and OOD Robustness

For deployment-facing reliability checks, combine:

- calibration metrics (quantile/marginal calibration, coverage diagnostics)
- OOD metrics (Mahalanobis, typicality, entropy, density-based signals)
- decision metrics (risk-coverage curves, rejection policy)

No single score is sufficient; compare multiple signals and validate on your shift scenarios.

## What To Run (Evidence Path)

Use these examples first for decision-grade comparisons:

- `examples/comprehensive_comparison.py`
- `examples/comprehensive_loss_comparison.py`
- `examples/imbalanced_regression.py`
- `examples/evaluate_conformal_methods.py`
- `examples/normalizing_flows_multitarget.py`

Use these benchmark artifacts for performance guardrails:

- `reports/benchmark_thresholds/cpu/smoke.json`
- `reports/benchmark_thresholds/cpu/sweep.json`
- `tools/benchmark_smoke.py`
- `tools/benchmark_report_summary.py`

## Scriptable Shortlisting (Method Catalog)

```python
import torchregress as tr

# Example: methods that support decomposition and are at least "Available"
rows = tr.method_catalog.list_methods(
    capability_filters={"decomposition": "yes"},
    maturity=("Core", "Strong", "Available"),
)

for row in rows:
    print(row["name"], row["family"], row["maturity"], row["task_tags"])
```

## When To Escalate Complexity

Escalate from simpler to more complex methods only when the simpler method fails on a metric
you care about (coverage, calibration, tail error, OOD selectivity, or multimodal fit).

This keeps comparisons credible and reduces accidental overfitting to fashionable methods.
