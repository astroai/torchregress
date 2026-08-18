# External Comparison: torchregress vs MAPIE / BoTorch / scikit-lego

This page documents three external-comparison benchmarks that pit `torchregress`
against widely-used third-party libraries on canonical regression tasks.

→ API: [`SplitConformal`](../api/losses.md), [`CQR`](../api/losses.md), [`BayesianLinearHead`](../api/test_time.md).

| Task | External library | Comparison focus | Artifact |
|---|---|---|---|
| Conformal prediction intervals | [MAPIE](https://mapie.readthedocs.io/), [crepes](https://crepes.readthedocs.io/), [torchcp](https://github.com/snap-stanford/torchcp) | Split + CQR coverage / width / interval score | `reports/external_comparison_conformal_vs_mapie_latest.json` |
| Low-shot Bayesian linear regression | [BoTorch](https://botorch.org/) | RMSE / NLL / 95% coverage | `reports/external_comparison_bayesian_linear_vs_botorch_latest.json` |
| Tweedie / compound-Poisson regression | [scikit-lego](https://scikit-lego.netlify.app/) | MAE / Tweedie deviance | `reports/external_comparison_tweedie_vs_sklego_latest.json` |

## Setup

All three external libraries are optional dependencies. Install them directly:

```bash
uv pip install mapie scikit-lego botorch gpytorch crepes torchcp
```

If an external library is not installed, the corresponding script still runs
and emits rows with the metrics set to `null` plus a `Notes` field that
documents the skip — so the JSON artifacts stay schema-stable across
environments.

Run the three benchmarks:

```bash
uv run python examples/external_comparison_conformal_vs_mapie.py \
    --summary-json-path reports/external_comparison_conformal_vs_mapie_latest.json
uv run python examples/external_comparison_bayesian_linear_vs_botorch.py \
    --summary-json-path reports/external_comparison_bayesian_linear_vs_botorch_latest.json
uv run python examples/external_comparison_tweedie_vs_sklego.py \
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
| Split + Linear | MAPIE | sklearn `LinearRegression` + `MapieRegressor(method="base")` |
| CQR + GBR | MAPIE | sklearn `GradientBoostingRegressor` + `MapieQuantileRegressor` |
| Split + Linear | crepes | sklearn `LinearRegression` + `crepes.ConformalRegressor` (residual-based calibration) |
| CQR + GBR | crepes | sklearn `GradientBoostingRegressor` quantile + `crepes.ConformalRegressor` on CQR scores |
| Split + Linear | torchcp | sklearn `LinearRegression` + `torchcp.regression.SplitCP` (API-version tolerant) |

**Metrics.** Coverage vs `1 - alpha` (target 0.9), mean interval width,
**interval score** (proper scoring rule for predictive intervals), and
training/evaluation runtime. Capacity is **not** matched between libraries:
torchregress uses MLP backbones, the others wrap sklearn estimators by
design. The `torchregress/Split+Linear` row is included so the wrapper
itself is compared apples-to-apples against MAPIE/crepes/torchcp on the
same linear backbone.

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

## Task 3 — Tweedie / compound-Poisson regression (vs scikit-lego)

**Task.** Synthetic zero-inflated continuous response drawn from a compound
Poisson-Gamma distribution with Tweedie power `p=1.5`. We compare:

| Method | Library | Backbone |
|---|---|---|
| Tweedie loss | torchregress | small MLP on log-mean + `TweedieLoss(p=1.5)` |
| Compound-Poisson loss | torchregress | small MLP on log-mean + `CompoundPoissonLoss(p=1.5)` |
| Tweedie GLM | scikit-lego | `GLMRegressor(distribution="tweedie", power=1.5)` |

**Metrics.** MAE, Tweedie unit deviance, predicted zero-fraction, training
runtime. Capacity is **not** matched: torchregress uses an MLP; scikit-lego
fits a log-link GLM. The unit-deviance metric is the standard scoring rule
for Tweedie responses and is the closest thing to a likelihood-grounded
comparison across library boundaries.

**Script:** [`examples/external_comparison_tweedie_vs_sklego.py`](https://github.com/astroai/torchregress/blob/main/examples/external_comparison_tweedie_vs_sklego.py)

## Decision Criteria

| If you need… | Start with… | Then consider… |
|---|---|---|
| Conformal intervals on a deep backbone with custom losses | `torchregress.ConformalLoss` (split / CQR / UACQR) | MAPIE for sklearn-estimator wrappers, crepes for Mondrian / class-conditional / APS variants, torchcp for research-grade CP with NN backbones |
| Few-shot linear regression with closed-form posteriors | `torchregress.BayesianLinearHead` / `RecursiveBayesianHead` | BoTorch `SingleTaskGP` for nonlinear features |
| Tweedie / compound-Poisson target with a flexible backbone | `torchregress.TweedieLoss` / `CompoundPoissonLoss` | scikit-lego `GLMRegressor` for interpretable log-link models |

## Limitations

- Capacity is intentionally not matched between libraries. The numbers are
  meant to anchor an operational default, not a horse race.
- The benchmarks run on synthetic data. For external-data validation, see the
  `*_realdata_comparison.py` examples in the same directory.
- BoTorch adds a significant dependency footprint (gpytorch, botorch, pyro).
  Treat it as an opt-in, not a default.
- The `external` extra is intentionally separate from the core `all` extra so
  default installs stay light.
