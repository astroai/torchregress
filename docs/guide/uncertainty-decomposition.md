# Uncertainty Decomposition

Uncertainty language is easy to overstate. In `torchregress`, keep these four
contracts separate:

| Contract | What it means | Typical tools |
|---|---|---|
| Predictive spread | The predictive distribution or interval is wide. | `GaussianNLLLoss`, `MDNLoss`, `NormalizingFlowLoss`, `QuantileLoss` |
| Coverage guarantee | A calibrated interval covers future labels at a target rate under conformal assumptions. | `ConformalLoss`, CQR/UACQR-style wrappers |
| Epistemic signal | Different plausible models disagree. | `DeepEnsemble`, `PackedEnsembleRegressor`, `MCDropoutWrapper`, `SWAG`, `BNN` |
| Epistemic + aleatoric variance decomposition | Total variance is split into model disagreement and expected per-model noise. | heteroscedastic ensembles, heteroscedastic BNNs, ensembles of probabilistic heads |

The standard ensemble variance identity is:

```text
total_variance = epistemic_variance + aleatoric_variance
epistemic_variance = variance_across_member_means
aleatoric_variance = mean_member_predicted_variance
```

Use `torchregress.metrics.uncertainty_decomposition(means, variances)` or
`ensemble_variance_decomposition(...)` when you already have per-member means
and per-member variances.

## Quantile Ensembles

Yes, you can deep-ensemble quantile regressors.

An ensemble of quantile heads gives useful epistemic information through
disagreement among predicted quantile functions. For example, if independently
trained models disagree strongly about the 0.9 quantile, that is a model
uncertainty signal.

That is not the same thing as a clean variance decomposition. Quantile
regression predicts conditional quantiles, not per-member Gaussian variances.
You can summarize ensemble disagreement across quantiles, and you can estimate
interval width from the quantile band, but calling those two numbers
`epistemic_variance` and `aleatoric_variance` requires extra modeling choices.

Use this framing:

- Single quantile model: calibrated/non-Gaussian intervals, no epistemic split.
- Quantile ensemble: epistemic-style disagreement across quantile functions.
- Quantile ensemble + conformal calibration: stronger interval coverage story.
- Quantile ensemble + explicit distributional/variance model: only then consider
  formal variance decomposition.

## Method Semantics

| Method/API | Epistemic | Aleatoric / spread | Decomposition status |
|---|---|---|---|
| `HeteroscedasticEnsembleModel`, `HeteroscedasticBatchEnsembleModel` | yes | yes | Full variance decomposition; returns `epistemic_variance`, `aleatoric_variance`, and total `variance`. |
| `HeteroscedasticBNN` | yes | yes | Full variance decomposition via `predict_with_decomposition()`. |
| `MDNEnsembleModel` | yes | yes | Ensemble disagreement plus mixture predictive spread. |
| `DeepEnsemble` | yes | partial | Full only if members also predict variances or distributions. Plain point ensembles expose epistemic disagreement only. |
| `PackedEnsembleRegressor` | yes | partial | Full for heteroscedastic heads; homoscedastic heads expose no aleatoric component. |
| `BinnedPDFEnsembleModel`, `CumulativeLinkEnsembleModel` | yes | partial | Ensemble disagreement plus distributional/ordinal spread; decomposition is representation-specific. |
| `MCDropoutWrapper`, `SWAG`, `MultiSWAG`, `BayesianNeuralNetwork` | yes | partial | Weight/sample uncertainty is epistemic; aleatoric requires an explicit variance head or likelihood model. |
| `EvidentialRegressionLoss` | partial | yes | Analytic NIG-derived uncertainty; validate calibration before treating the epistemic term as model uncertainty. |
| `MDNLoss`, `NormalizingFlowLoss`, `GaussianNLLLoss`, `LowRankGaussianLoss`, `MultivariateGaussianLoss` | no | yes | Single-model predictive distributions model aleatoric or predictive spread, not epistemic uncertainty. |
| `QuantileLoss` | no | yes | Conditional quantile spread/intervals; no epistemic signal without an ensemble or sampling mechanism. |
| `ConformalLoss` and conformal predictors | no | no | Coverage guarantees and calibrated intervals, not uncertainty decomposition. |

## Catalog Rules

When updating `method_catalog.py`, use conservative capability labels:

- `yes`: the component is explicitly modeled and returned, or directly computed
  by a tested helper.
- `partial`: the component exists only for a specific head, sampling mode,
  ensemble construction, or modeling assumption.
- `no`: the method may produce intervals or spread, but not that uncertainty
  component.

Avoid listing single-model MDN, flow, or quantile losses as epistemic +
aleatoric decomposition methods. List their ensemble variants when the intended
claim is model-disagreement plus predictive spread.
