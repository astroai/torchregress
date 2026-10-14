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
- **real-data**: uses a real dataset. The `*_realdata_comparison.py` scripts use scikit-learn's bundled Diabetes data (no network). The two others download data.

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
| `evidential_regression.py` | tutorial | Evidential (NIG) regression and its uncertainty split. | slow (about 80-110 s) |
| `gaussian_full_covariance_regression.py` | tutorial | Full-covariance Gaussian head for multi-output targets. | fast |
| `gaussian_low_rank_regression.py` | tutorial | Low-rank-plus-diagonal Gaussian head. | fast |
| `heteroscedastic_beta_nll_demo.py` | tutorial | Gaussian NLL vs Beta-NLL under heteroscedastic noise (`--epochs`, `--seed`, `--lr`). | fast |
| `heteroscedastic_laplace_demo.py` | tutorial | Laplace likelihood and its uncertainty decomposition. | fast |
| `expectile_regression_demo.py` | tutorial | Expectile vs quantile regression under heteroscedasticity. | fast |
| `poisson_regression_demo.py` | tutorial | Poisson and zero-inflated count regression. | fast |
| `tweedie_regression_demo.py` | tutorial | Tweedie loss for zero-inflated continuous data. | fast |
| `poisson_gaussian_mixture_demo.py` | tutorial | Poisson-Gaussian mixture loss for shot plus readout noise. | fast |
| `balanced_mse_demo.py` | tutorial | `BalancedMSELoss` / `BinReweightedMSELoss` on a long-tailed target (`--n`, `--steps`, `--seed`). | fast |
| `imbalanced_regression.py` | tutorial | Imbalanced regression with a calibration check. | fast |
| `bayesian_learning_rule_demo.py` | tutorial | Variational regression with the Bayesian Learning Rule (IVON). | fast |
| `normalizing_flows_multitarget.py` | tutorial | Multi-target conditional flows (needs `zuko`). | slow (about 70 s) |
| `conformal_regression_example.py` | tutorial | Split, CQR, UACQR, per-dimension, CV+ and EnbPI conformal wrappers. | fast |
| `conformal_mondrian_demo.py` | tutorial | Group-conditional (Mondrian) conformal intervals. | fast |
| `ot_shift_conformal_demo.py` | tutorial | Score-CDF gap, OT-style reweighting, weighted split conformal (`--seed`). | fast |
| `eiv_algorithms_demo.py` | tutorial | Errors-in-variables input-noise correction algorithms. | fast (about 50 s) |
| `gaussian_wasserstein_bound_demo.py` | tutorial | `GaussianWassersteinBoundLoss` with mean plus covariance targets. | fast |
| `wasserstein_bound_hybrid_pretrain_demo.py` | tutorial | Wasserstein-bound pretraining then Gaussian NLL fine-tuning. | fast |
| `test_time_bayesian_linear_head_demo.py` | tutorial | Exact conjugate `BayesianLinearHead` on fixed features. | fast |
| `test_time_blr_predictive_adapter_demo.py` | tutorial | Predictive-batch adapter around `BayesianLinearHead`. | fast |
| `test_time_adaptation_suite.py` | tutorial | Covariate-shift alignment, label-shift correction, sample selection. | fast |
| `ppi_calibrated_mean.py` | tutorial | Standard vs affine-calibrated prediction-powered inference for a mean. | fast |
| `ppi_mean_plus_split_conformal.py` | tutorial | Combining PPI mean inference with split-conformal intervals. | fast |
| `metrics_suite_showcase.py` | tutorial | Tour of the point, distribution, interval, OOD, decision, ensemble and weak-GT metrics. | fast |
| `viz_diagnostic_gallery.py` | tutorial | Every `torchregress.viz` diagnostic plot in one gallery. | fast |
| `stellar_spectra_ensemble.py` | tutorial | Deep ensemble of heteroscedastic CNNs on synthetic stellar spectra, with a corner plot. | slow (about 8 min on 2 shared cores) |
| `stellar_spectra_ensemble_flows.py` | tutorial | Ensemble of normalizing flows on the same spectra (needs `zuko`). | slow (about 6 min on 2 shared cores) |
| `stellar_spectra_flow_corner.py` | tutorial | Conditional flow posterior and a 3x3 corner plot (needs `zuko`). | slow (about 100 s) |

## Comparisons (synthetic data)

| Script | Type | What it shows | Runtime |
|---|---|---|---|
| `loss_comparison.py` | comparison | Standard vs robust losses on noisy data with outliers. | fast |
| `comprehensive_loss_comparison.py` | comparison | Many losses on one MLP with point, distribution and calibration metrics. | fast |
| `comprehensive_comparison.py` | comparison | Robust losses, uncertainty heads and ensembles side by side. | fast |
| `evaluate_conformal_methods.py` | comparison | Coverage and width of the conformal predictors vs target level. | fast |
| `causal_dr_uplift_comparison.py` | comparison | Doubly-robust ATE/CATE estimators. | fast |
| `censored_regression_comparison.py` | comparison | Censored-regression losses. | fast |
| `constraints_calibration_comparison.py` | comparison | Constrained heads and post-hoc calibrators (variance temperature, isotonic, PIT). | fast |
| `contrastive_flow_parameter_estimation.py` | comparison | Contrastive normalizing flows for nuisance-aware parameter estimation. | fast |
| `contrastive_flow_parameter_estimation_comparison.py` | comparison | Parameter-estimation methods on synthetic pseudo-experiments. | fast |
| `eiv_method_comparison.py` | comparison | EIV loss variants under measurement error. | fast |
| `multimodal_method_comparison.py` | comparison | Multimodal / multi-target regression methods. | fast |
| `noisy_label_comparison.py` | comparison | Noisy-label losses with calibration metrics. | fast |
| `ood_selective_prediction_comparison.py` | comparison | OOD and selective-prediction signals across uncertainty methods. | fast |
| `ordinal_regression_comparison.py` | comparison | Ordinal regression methods. | fast |
| `ordinal_uncertain_ground_truth_comparison.py` | comparison | Uncertain ground truth in regression-as-classification. | fast |
| `sls_multimodal_regression.py` | comparison | SLS vs CQR vs CTI for multimodal targets. | fast |
| `transformed_target_regression_comparison.py` | comparison | Transformed-target losses. | fast |
| `uncertain_gt_density_conformal_comparison.py` | comparison | Uncertain labels with density-aware conformal methods. | fast |
| `external_comparison_bayesian_linear_vs_botorch.py` | comparison | `BayesianLinearHead` vs BoTorch `SingleTaskGP` (needs `botorch`). | fast |
| `external_comparison_conformal_vs_mapie.py` | comparison | Conformal intervals vs MAPIE >= 1.0, crepes, torchcp >= 1.2 (needs `mapie crepes torchcp`; ported to the current comparator APIs, see below). | fast |
| `external_comparison_tweedie_vs_sklego.py` | comparison | Tweedie losses vs a log-link GLM. scikit-lego 0.9.x has no Tweedie GLM, so the baseline is `sklearn.linear_model.TweedieRegressor`. | fast |

## Real-data comparisons

| Script | Type | What it shows | Runtime |
|---|---|---|---|
| `causal_dr_realdata_comparison.py` | real-data | Doubly-robust causal estimators on Diabetes covariates. | fast |
| `censored_regression_realdata_comparison.py` | real-data | Censored losses on Diabetes. | fast |
| `eiv_method_realdata_comparison.py` | real-data | EIV losses on Diabetes with injected measurement error. | fast |
| `multimodal_method_realdata_comparison.py` | real-data | Multimodal / multi-target methods on Diabetes features. | fast |
| `noisy_label_realdata_comparison.py` | real-data | Noisy-label losses on Diabetes. | fast |
| `ood_selective_prediction_realdata_comparison.py` | real-data | OOD / selective prediction on Diabetes. | fast |
| `ordinal_regression_realdata_comparison.py` | real-data | Ordinal methods on binned Diabetes. | fast |
| `semi_supervised_regression_comparison.py` | real-data | Semi-supervised regression methods on a real-data proxy. | fast |
| `uncertain_gt_density_conformal_realdata_comparison.py` | real-data | Uncertain-GT and density conformal on Diabetes. | fast |
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

### Research benchmarks (belong in torchregress-research)

| Script | What it shows | Runtime |
|---|---|---|
| `benchmarks/self_agreement_higgs_ood.py` | SAGE-Reg Higgs-inspired OOD benchmark: supervised, Mean Teacher, Pi-model, confidence-weighted pseudo-labels and SAGE-Reg on a shifted tabular task (synthetic unless `--dataset-path` is given). | fast (about 50 s) |

It still lives here and runs; it is not moved. The NeurIPS SAGE/SPT reproduction
scripts were already extracted to
[torchregress-research](https://github.com/sfabbro/torchregress-research).

### Helper module (not a script)

`contrastive_flow_benchmark_utils.py` holds helpers shared by the two
contrastive-flow scripts and has no `main`.

## Candidates to merge in 0.4

Nothing is deleted or merged for 0.3.0. These overlap and could share one
script with a `--dataset {synthetic,diabetes}` style switch:

- **Synthetic / real-data pairs** that share the same harness and differ only in data source:
  `causal_dr_uplift_comparison` + `causal_dr_realdata_comparison`,
  `censored_regression_comparison` + `censored_regression_realdata_comparison`,
  `eiv_method_comparison` + `eiv_method_realdata_comparison`,
  `multimodal_method_comparison` + `multimodal_method_realdata_comparison`,
  `noisy_label_comparison` + `noisy_label_realdata_comparison`,
  `ood_selective_prediction_comparison` + `ood_selective_prediction_realdata_comparison`,
  `ordinal_regression_comparison` + `ordinal_regression_realdata_comparison`,
  `uncertain_gt_density_conformal_comparison` + `uncertain_gt_density_conformal_realdata_comparison`.
- **Loss comparisons**: `loss_comparison`, `comprehensive_loss_comparison`, `comprehensive_comparison`.
- **Conformal**: `conformal_regression_example`, `evaluate_conformal_methods`, `conformal_mondrian_demo`.
- **OT-shift conformal**: `ot_shift_conformal_demo` and `benchmarks/ot_conformal_score_shift_benchmark` describe the same toy.
- **Contrastive flows**: `contrastive_flow_parameter_estimation` and `contrastive_flow_parameter_estimation_comparison`.
- **Stellar spectra**: the three `stellar_spectra_*` scripts repeat the same synthetic generator and CNN.
- **Count / zero-inflated data**: `poisson_regression_demo`, `tweedie_regression_demo`, `poisson_gaussian_mixture_demo`.
- **Bayesian linear head**: `test_time_bayesian_linear_head_demo`, `test_time_blr_predictive_adapter_demo`, `benchmarks/bayesian_linear_head_*`.
- **Wasserstein bound**: `gaussian_wasserstein_bound_demo` and `wasserstein_bound_hybrid_pretrain_demo`.
- **Intro**: `basic_usage` and `native_api_usage`.

## External-library comparators

The `external_comparison_*` scripts need their comparator packages installed
explicitly (there is no `torchregress[external]` extra):

```bash
uv pip install mapie crepes torchcp botorch gpytorch   # scikit-lego is optional, see below
```

They track the current comparator APIs (last checked with MAPIE 1.5.0 and
1.5.1.dev6, crepes 0.9.1, torchcp 1.2.1, scikit-learn 1.9.1, scikit-lego 0.9.10):

- `external_comparison_conformal_vs_mapie.py` uses MAPIE's `SplitConformalRegressor`
  and `ConformalizedQuantileRegressor` (`fit` / `conformalize` / `predict_interval`,
  `confidence_level = 1 - alpha`) and torchcp's `SplitPredictor` with the `ABS`
  score. MAPIE 0.x (`MapieRegressor`) and torchcp < 1.2 (`SplitCP`) are no longer supported.
- `external_comparison_tweedie_vs_sklego.py` compares against
  `sklearn.linear_model.TweedieRegressor` because scikit-lego 0.9.x has no
  Tweedie/GLM estimator (`GLMRegressor` was removed). scikit-lego is only probed
  so its version and the missing GLM are recorded in the summary.

A comparator that is not installed gives a `skipped: <lib> not installed` row. One that is
installed but whose API does not match gives `skipped: <lib> <version> incompatible`
(or `failed: <lib> <version> incompatible (...)` if it breaks while running), so API
drift is never reported as "not installed". Skipped rows keep `null` metrics so the JSON
schema is stable. Each script prints the installed comparator versions and records them in
the summary JSON `notes` and per-row `Version` field.
