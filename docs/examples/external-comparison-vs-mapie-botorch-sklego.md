# External Comparison: torchregress vs MAPIE / BoTorch / scikit-lego

This page documents three external-comparison benchmarks that pit `torchregress`
against widely-used third-party libraries on canonical regression tasks.

→ API: [`SplitConformal`](../api/losses.md), [`CQR`](../api/losses.md), [`BayesianLinearHead`](../api/test_time.md).

| Task | External library | Comparison focus | Artifact |
|---|---|---|---|
| Conformal prediction intervals | [MAPIE](https://mapie.readthedocs.io/), [crepes](https://crepes.readthedocs.io/), [torchcp](https://github.com/snap-stanford/torchcp) | Split + CQR coverage / width / interval score | `reports/external_comparison_conformal_vs_mapie_latest.json` |
| Low-shot Bayesian linear regression | [BoTorch](https://botorch.org/) | RMSE / NLL / 95% coverage | `reports/external_comparison_bayesian_linear_vs_botorch_latest.json` |
| Tweedie / compound-Poisson regression | [scikit-learn `TweedieRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html) (scikit-lego 0.9.x has no Tweedie GLM) | MAE / Tweedie deviance | `reports/external_comparison_tweedie_vs_sklego_latest.json` |

## Setup

The external libraries are optional dependencies and there is no
`torchregress[external]` extra. Install the comparators directly:

```bash
uv pip install mapie crepes torchcp botorch gpytorch   # or: pixi add mapie crepes torchcp botorch gpytorch
```

The Tweedie benchmark only needs scikit-learn; scikit-lego (0.9.x) is merely
probed, because it no longer ships a Tweedie GLM.

If an external library is not installed, the corresponding script still runs
and emits rows with the metrics set to `null` plus a `Notes` field that
documents the skip (`skipped: <lib> not installed`). A library that is
installed but whose API does not match the version this page targets gives
`skipped: <lib> <version> incompatible` (or `failed: ... incompatible` if it
breaks while running) instead, so API drift is never reported as "not
installed". The JSON artifacts stay schema-stable across environments, and each
script prints the installed comparator versions and records them in the
summary `notes` and in a per-row `Version` field.

Run the three benchmarks:

```bash
pixi run python examples/external_comparison_conformal_vs_mapie.py \
    --summary-json-path reports/external_comparison_conformal_vs_mapie_latest.json
pixi run python examples/external_comparison_bayesian_linear_vs_botorch.py \
    --summary-json-path reports/external_comparison_bayesian_linear_vs_botorch_latest.json
pixi run python examples/external_comparison_tweedie_vs_sklego.py \
    --summary-json-path reports/external_comparison_tweedie_vs_sklego_latest.json
```

## Task 1 — Conformal prediction intervals (vs MAPIE / crepes / torchcp)

**Task.** Heteroscedastic regression with a shared train/calibration/test
split. We compare eight rows across four library wrappers:

| Method | Library | Backbone |
|---|---|---|
| Split + MLP | torchregress | small MLP point head + `ConformalLoss(method="split")` |
| CQR + MLP | torchregress | small MLP two-output head + `ConformalLoss(method="cqr")` |
| Split + Linear | torchregress | single linear layer + `ConformalLoss(method="split")` (apples-to-apples wrapper comparison) |
| Split + Linear | MAPIE (>= 1.0) | sklearn `LinearRegression` + `SplitConformalRegressor(confidence_level=1 - alpha)` (`fit` → `conformalize` → `predict_interval`) |
| CQR + GBR | MAPIE (>= 1.0) | sklearn `GradientBoostingRegressor` + `ConformalizedQuantileRegressor(symmetric_correction=True)` |
| Split + Linear | crepes | sklearn `LinearRegression` + `crepes.ConformalRegressor` (residual-based calibration) |
| CQR + GBR | crepes | sklearn `GradientBoostingRegressor` quantile + `crepes.ConformalRegressor` on (shifted, non-negative) CQR scores |
| Split + Linear | torchcp (>= 1.2) | sklearn `LinearRegression` weights in an `nn.Linear` + `torchcp.regression.predictor.SplitPredictor` with the `ABS` score |

**Metrics.** Coverage vs `1 - alpha` (target 0.9), mean interval width,
**interval score** (proper scoring rule for predictive intervals), and
training/evaluation runtime. Capacity is **not** matched between libraries:
torchregress uses MLP backbones, the others wrap sklearn estimators by
design. The `torchregress/Split+Linear` row is included so the wrapper
itself is compared apples-to-apples against MAPIE/crepes/torchcp on the
same linear backbone.

**Latest run** (alpha = 0.1, seed 260612, MAPIE 1.5.0, crepes 0.9.1,
torchcp 1.2.1; exact numbers depend on library versions and BLAS):

| Method | Coverage | Width | Interval score |
|---|---:|---:|---:|
| torchregress / Split + MLP | 0.905 | 1.328 | 1.890 |
| torchregress / CQR + MLP | 0.933 | 1.480 | 1.759 |
| torchregress / Split + Linear | 0.918 | 2.299 | 2.869 |
| MAPIE / Split + Linear | 0.903 | 1.635 | 2.348 |
| MAPIE / CQR + GBR | 0.918 | 1.903 | 2.205 |
| crepes / Split + Linear | 0.903 | 1.635 | 2.348 |
| crepes / CQR + GBR | 0.918 | 1.903 | 2.205 |
| torchcp / Split + Linear | 0.903 | 1.635 | 2.348 |

The three external split rows share one backbone and one calibration set and so
agree exactly, as do the two CQR rows; the `torchregress/Split+Linear` row is
wider because its single linear layer is trained by 60 epochs of Adam
(`lr=1e-3`) and is not converged to the least-squares fit.

**Script:** [`examples/external_comparison_conformal_vs_mapie.py`](https://github.com/astroai/torchregress/blob/main/examples/external_comparison_conformal_vs_mapie.py)

**Note on `torch-uncertainty`:** the `torch-uncertainty` library (ENSTA-U2IS-AI)
focuses on classification UQ and does **not** ship an end-to-end regression
conformal prediction API as of mid-2026, so it is not in the comparison. Its
distribution-estimation heads can be wrapped externally if a regression CP
wrapper is added in a future release.

## Task 2 — Low-shot Bayesian linear regression (vs BoTorch)

**Task.** Linear regression with known `w_true`; `n_train=30`, `d=5`,
`noise=0.3`. We compare:

| Method | Library | Backbone |
|---|---|---|
| Bayesian linear head | torchregress | `BayesianLinearHead` (exact conjugate posterior) |
| SingleTask GP | BoTorch | `SingleTaskGP` fit with exact MLL |

**Metrics.** RMSE, Gaussian NLL, empirical 95% coverage, posterior-mean L2
error to `w_true` (torchregress only — GP weights are not directly
comparable), and runtime. Capacity is **not** matched: torchregress is an
exact-conjugate linear posterior; BoTorch fits a flexible GP with
hyperparameter marginal-likelihood optimization.

**Script:** [`examples/external_comparison_bayesian_linear_vs_botorch.py`](https://github.com/astroai/torchregress/blob/main/examples/external_comparison_bayesian_linear_vs_botorch.py)

## Task 3 — Tweedie / compound-Poisson regression (vs a log-link GLM)

**Task.** Synthetic zero-inflated continuous response drawn from a compound
Poisson-Gamma distribution with Tweedie power `p=1.5`. We compare:

| Method | Library | Backbone |
|---|---|---|
| Tweedie loss | torchregress | small MLP on log-mean + `TweedieLoss(p=1.5)` |
| Compound-Poisson loss | torchregress | small MLP on log-mean + `CompoundPoissonLoss(p=1.5)` |
| Tweedie GLM | scikit-learn | `TweedieRegressor(power=1.5, link="log", alpha=0)` |

**Metrics.** MAE, Tweedie unit deviance, predicted zero-fraction, training
runtime. Capacity is **not** matched: torchregress uses an MLP; the GLM is
linear in the features with a log link. The unit-deviance metric is the standard scoring rule
for Tweedie responses and is the closest thing to a likelihood-grounded
comparison across library boundaries.

**Latest run** (p = 1.5, seed 260614, scikit-learn 1.9.1):

| Method | MAE | Tweedie deviance |
|---|---:|---:|
| torchregress / Tweedie | 0.680 | 0.666 |
| torchregress / CompoundPoisson | 0.676 | 0.674 |
| scikit-learn / TweedieRegressor (log link, alpha=0) | 0.677 | 0.663 |

**Script:** [`examples/external_comparison_tweedie_vs_sklego.py`](https://github.com/astroai/torchregress/blob/main/examples/external_comparison_tweedie_vs_sklego.py)

## Decision Criteria

| If you need… | Start with… | Then consider… |
|---|---|---|
| Conformal intervals on a deep backbone with custom losses | `torchregress.ConformalLoss` (split / CQR / UACQR) | MAPIE for sklearn-estimator wrappers, crepes for Mondrian / class-conditional / APS variants, torchcp for research-grade CP with NN backbones |
| Few-shot linear regression with closed-form posteriors | `torchregress.BayesianLinearHead` / `RecursiveBayesianHead` | BoTorch `SingleTaskGP` for nonlinear features |
| Tweedie / compound-Poisson target with a flexible backbone | `torchregress.TweedieLoss` / `CompoundPoissonLoss` | scikit-learn `TweedieRegressor` for interpretable log-link models |

## Limitations

- Capacity is intentionally not matched between libraries. The numbers are
  meant to anchor an operational default, not a horse race.
- The benchmarks run on synthetic data. For external-data validation, see the
  `*_realdata_comparison.py` examples in the same directory.
- BoTorch adds a significant dependency footprint (gpytorch, botorch, pyro).
  Treat it as an opt-in, not a default.
- The comparator packages are deliberately not part of any torchregress extra,
  so default installs stay light; install them explicitly as shown above.
