# torchregress examples

Runnable scripts that show the library in use. Run any of them from a scratch
directory so the figures and JSON summaries they write do not land in the repo:

```bash
mkdir -p /tmp/tr-examples && cd /tmp/tr-examples
MPLBACKEND=Agg python /path/to/torchregress/examples/basic_usage.py
```

There is no shared fast/smoke switch. Scripts either run at a fixed small budget
or expose their own flags (`--epochs`, `--steps`, `--seed`, `--summary-json-path`;
see `--help`). `tests/test_examples_smoke.py` and
`tests/test_examples_output_summaries.py` import or run the most important ones
with the training loops stubbed out or shrunk.

## How to read the tables

**Type**

- **tutorial**: teaches one API or loss on synthetic data.
- **comparison**: several methods under a shared budget on synthetic data (they usually write a JSON summary, see `--summary-json-path`).
- **real-data**: uses a real dataset. The `*_realdata_comparison.py` scripts and the `--dataset diabetes` runs of the merged comparison scripts use scikit-learn's bundled Diabetes data (no network). The two others download data.

**Runtime class** (measured on 2 shared CPU cores, torch CPU, no GPU)

- **fast**: finishes in under 60 s.
- **slow**: needs more than 60 s (evidence given where measured).
- **needs-download**: fetches a dataset or pretrained weights; fails offline.

Optional dependencies are noted where a script needs one (`zuko` for flows,
`botorch`, `mapie`, `torchcp`, `scikit-lego`, `crepes`, `tabpfn`).

## Tutorials and demos

| Script | Type | What it shows | Runtime |
|---|---|---|---|
| `basic_usage.py` | tutorial | Point, quantile and Gaussian-NLL losses on one toy problem, with a plot. | fast |
| `native_api_usage.py` | tutorial | Using the losses as plain PyTorch modules inside normal training loops. | fast |
| `ensemble_tutorial.py` | tutorial | Deep ensembles with aleatoric/epistemic uncertainty decomposition. | fast |
| `tabular_mlp_tutorial.py` | tutorial | `TabularMLP` + `fit_tabular` with MSE and `GaussianNLLLoss` (2 outputs) heads on bundled Diabetes, and `TabularPreprocessor` NaN imputation with missing indicators (`--epochs`, `--seed`). | fast (about 6 s) |
| `evidential_regression.py` | tutorial | Evidential (NIG) regression and its uncertainty split. | slow (about 80-110 s) |
| `gaussian_full_covariance_regression.py` | tutorial | Full-covariance Gaussian head for multi-output targets. | fast |
| `gaussian_low_rank_regression.py` | tutorial | Low-rank-plus-diagonal Gaussian head. | fast |
| `heteroscedastic_beta_nll_demo.py` | tutorial | Gaussian NLL vs Beta-NLL under heteroscedastic noise (`--epochs`, `--seed`, `--lr`). | fast |
| `heteroscedastic_laplace_demo.py` | tutorial | Laplace likelihood and its uncertainty decomposition. | fast |
| `expectile_regression_demo.py` | tutorial | Expectile vs quantile regression under heteroscedasticity. | fast |
| `count_regression_demo.py` | tutorial | `--family poisson`: Poisson deviance, likelihood-ratio, ZIP and negative-binomial losses on zero-inflated counts; `--family tweedie`: Tweedie and compound Poisson losses on zero-inflated continuous data (default `all`). | fast |
| `poisson_gaussian_contamination_mixture_demo.py` | tutorial | Poisson-Gaussian contamination-mixture losses (for shot plus read noise use `PoissonGaussianLoss`). | fast |
| `balanced_mse_demo.py` | tutorial | `BalancedMSELoss` / `BinReweightedMSELoss` on a long-tailed target (`--n`, `--steps`, `--seed`). | fast |
| `imbalanced_regression.py` | tutorial | Imbalanced regression with a calibration check. | fast |
| `bayesian_learning_rule_demo.py` | tutorial | Variational regression with the Bayesian Learning Rule (IVON). | fast |
| `normalizing_flows_multitarget.py` | tutorial | Multi-target conditional flows (needs `zuko`). | slow (about 70 s) |
| `conformal_regression_example.py` | tutorial + comparison | `--mode tour`: split, CQR, UACQR, per-dimension, CV+ and EnbPI conformal wrappers; `--mode sweep`: coverage and width of trained split/CQR/UACQR vs target level; `--mode mondrian`: group-conditional (Mondrian) intervals (default `all`). | fast |
| `ot_shift_conformal_demo.py` | tutorial | Score-CDF gap, OT-style reweighting, weighted split conformal (`--seed`). | fast |
| `eiv_algorithms_demo.py` | tutorial | Errors-in-variables input-noise correction algorithms. | fast (about 50 s) |
| `gaussian_wasserstein_bound_demo.py` | tutorial | `--mode loss`: `GaussianWassersteinBoundLoss` with mean plus covariance targets; `--mode hybrid`: Wasserstein-bound pretraining then Gaussian NLL fine-tuning (default `all`). | fast |
| `test_time_bayesian_linear_head_demo.py` | tutorial | `--mode exactness`: exact conjugate `BayesianLinearHead` vs `RecursiveBayesianHead` on fixed features; `--mode adapter`: `SupportsPredictiveBatch` adapter around `BayesianLinearHead` (default `all`). | fast |
| `test_time_adaptation_suite.py` | tutorial | Covariate-shift alignment, label-shift correction, sample selection. | fast |
| `ppi_calibrated_mean.py` | tutorial | Standard vs affine-calibrated prediction-powered inference for a mean. | fast |
| `ppi_mean_plus_split_conformal.py` | tutorial | Combining PPI mean inference with split-conformal intervals. | fast |
| `orthogonal_inference_rff_demo.py` | tutorial | `orthogonal_partially_linear` with a `random_fourier_features` basis (`median_heuristic_bandwidth`, `polynomial_degree=3`, `ridge="gcv"`) vs the default call and naive OLS: bias and 95% CI coverage over replications (`--replications`, `--n`, `--seed`). | fast (about 9 s) |
| `metrics_suite_showcase.py` | tutorial | Tour of the point, distribution, interval, OOD, decision, ensemble and weak-GT metrics. | fast |
| `viz_diagnostic_gallery.py` | tutorial | Every `torchregress.viz` diagnostic plot in one gallery. | fast |

## Comparisons (synthetic data)

| Script | Type | What it shows | Runtime |
|---|---|---|---|
| `loss_comparison.py` | comparison | Standard vs robust losses on data with outliers: `--suite basic` (linear model, 7 losses) and `--suite comprehensive` (MLP, 10 losses, point, distribution and calibration metrics) (default `all`). | fast |
| `comprehensive_comparison.py` | comparison | Robust losses, uncertainty heads and ensembles side by side. | fast |
| `causal_dr_uplift_comparison.py` | comparison | Doubly-robust ATE/CATE estimators. | fast |
| `censored_regression_comparison.py` | comparison | Censored-regression losses. | fast |
| `constraints_calibration_comparison.py` | comparison | Constrained heads and post-hoc calibrators (variance temperature, isotonic, PIT). | fast |
| `contrastive_flow_parameter_estimation.py` | comparison | `--mode demo`: contrastive normalizing flow for nuisance-aware parameter estimation; `--mode compare`: Gaussian summary, flow NLL, contrastive flow, NRE and NPE on the same pseudo-experiments (`--summary-json-path`; default `all`; needs `zuko`). | fast |
| `eiv_method_comparison.py` | comparison | EIV loss variants under measurement error. | fast |
| `multimodal_method_comparison.py` | comparison | Multimodal / multi-target regression methods. | fast |
| `noisy_label_comparison.py` | comparison | Noisy-label losses with calibration metrics (`--dataset synthetic\|diabetes`). | fast |
| `ood_selective_prediction_comparison.py` | comparison | OOD and selective-prediction signals across uncertainty methods (`--dataset synthetic\|diabetes`). | fast |
| `ordinal_regression_comparison.py` | comparison | Ordinal regression methods (`--dataset synthetic\|diabetes`). | fast |
| `ordinal_uncertain_ground_truth_comparison.py` | comparison | Uncertain ground truth in regression-as-classification. | fast |
| `sls_multimodal_regression.py` | comparison | SLS vs CQR vs CTI for multimodal targets. | fast |
| `transformed_target_regression_comparison.py` | comparison | Transformed-target losses. | fast |
| `uncertain_gt_density_conformal_comparison.py` | comparison | Uncertain labels with density-aware conformal methods (`--dataset synthetic\|diabetes`). | fast |
| `calibrated_regressor_comparison.py` | comparison | `CalibratedRegressor` (`vts`, `isotonic+vts`), `ConformalRegressor(method="split")` and `calibrated_deep_ensemble` vs an uncalibrated Gaussian `TabularMLP`: RMSE, 90% coverage, width, NLL, runtime (`--epochs`, `--seed`, `--n-members`). | fast (about 10 s) |
| `nsf_tail_bound_comparison.py` | comparison | NSF head with `create_flow_model(bound=recommended_tail_bound(y_std))` vs the fixed bound 5 on Student-t(3) noise: test NLL, 90% coverage and width (needs `zuko`; `--epochs`, `--seed`). | fast (about 11 s) |

## Real-data comparisons

| Script | Type | What it shows | Runtime |
|---|---|---|---|
| `causal_dr_uplift_comparison.py --dataset diabetes` | real-data | Doubly-robust causal estimators on Diabetes covariates. | fast |
| `censored_regression_comparison.py --dataset diabetes` | real-data | Censored losses on Diabetes. | fast |
| `eiv_method_comparison.py --dataset diabetes` | real-data | EIV losses on Diabetes with injected measurement error. | fast |
| `multimodal_method_comparison.py --dataset diabetes` | real-data | Multimodal / multi-target methods on Diabetes features. | fast |
| `noisy_label_comparison.py --dataset diabetes` | real-data | Noisy-label losses on Diabetes. | fast |
| `ood_selective_prediction_comparison.py --dataset diabetes` | real-data | OOD / selective prediction on Diabetes. | fast |
| `ordinal_regression_comparison.py --dataset diabetes` | real-data | Ordinal methods on binned Diabetes. | fast |
| `semi_supervised_regression_comparison.py` | real-data | Semi-supervised regression methods on a real-data proxy. | fast |
| `imdb_wiki_age_regression.py` | real-data | Age regression on IMDB-WIKI with several losses (ResNet backbone). | needs-download (about 1 GB dataset plus pretrained weights) |

## Library benchmarks (`benchmarks/`)

Small benchmarks of the library itself. Run from the repo root.

| Script | Type | What it shows | Runtime |
|---|---|---|---|
| `benchmarks/bayesian_linear_head_lowshot_adaptation.py` | comparison | Low-shot adaptation: conjugate BLR vs ridge MAP. | fast |
| `benchmarks/bayesian_linear_head_online_drift.py` | comparison | Online label drift: `RecursiveBayesianHead` vs batch refit. | fast |
| `benchmarks/ot_conformal_score_shift_benchmark.py` | comparison | Score shift, OT reweighting and weighted split conformal. | fast |
| `benchmarks/tail_extremes_benchmark.py` | comparison | Tail-focused losses on heavy-tailed targets. | fast |
| `benchmarks/tail_extremes_sweep.py` | comparison | Sweep over the tail benchmark configurations (imports `tail_extremes_benchmark`). | fast (about 50 s) |
| `benchmarks/foundation_model_comparison.py` | real-data | TabPFN vs torchregress Gaussian and mixture-density heads on California Housing (`--n_samples`). | needs-download (California Housing; TabPFN also needs weights and a Hugging Face token) |

### Moved out of this repository in 0.3.0

- The SAGE-Reg Higgs OOD benchmark and the NeurIPS SAGE/SPT reproduction scripts live in
  [torchregress-research](https://github.com/sfabbro/torchregress-research)
  (`benchmarks/neurips/sage/self_agreement_higgs_ood.py`).
- The external-library comparisons (MAPIE, crepes, torchcp, BoTorch, scikit-learn and
  statsmodels GLMs) are suites of
  [torchregress-harness](https://github.com/astroai/torchregress-harness)
  (`conformal`, `bayesian_linear`, `tweedie`); see
  [External comparison](../docs/examples/external-comparison-vs-mapie-botorch-sklego.md).

## Candidates to merge after 0.3.0

These overlap and could share one script:

- **OT-shift conformal**: `ot_shift_conformal_demo` and `benchmarks/ot_conformal_score_shift_benchmark` describe the same toy.
- **Intro**: `basic_usage` and `native_api_usage`.

Already merged (one script per topic, variants behind a flag): the synthetic / real-data
pairs (causal, censored, EIV, multimodal, noisy-label, OOD selective prediction, ordinal,
uncertain ground truth) behind `--dataset {synthetic,diabetes}`, `loss_comparison --suite`,
`conformal_regression_example --mode`, `contrastive_flow_parameter_estimation --mode`,
`count_regression_demo --family`, `test_time_bayesian_linear_head_demo --mode`,
`gaussian_wasserstein_bound_demo --mode`. Kept separate on purpose:
`comprehensive_comparison` (robustness, uncertainty heads and ensembles across three
scenarios, not a loss sweep), `poisson_gaussian_contamination_mixture_demo` (detector
noise mixtures, not count regression) and `benchmarks/bayesian_linear_head_*`
(library benchmarks, not tutorials).
