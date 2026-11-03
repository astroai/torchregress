# External Comparison: torchregress vs MAPIE / crepes / torchcp / BoTorch / scikit-learn

These comparisons pit `torchregress` against widely used third-party libraries on
canonical regression tasks. They live in the external-software parity harness,
[torchregress-harness](https://github.com/astroai/torchregress-harness), which pins
the comparator versions, runs them on shared splits and several datasets, and
publishes the results in its
[`PARITY_REPORT.md`](https://github.com/astroai/torchregress-harness/blob/main/PARITY_REPORT.md).
Until 0.3.0 they were scripts in `examples/`; keeping them in the harness keeps
comparator dependencies out of this repository.

→ API: [`SplitConformal`](../api/losses.md), [`CQR`](../api/losses.md), [`BayesianLinearHead`](../api/test_time.md).

| Task | External library | Comparison focus | Harness suite |
|---|---|---|---|
| Conformal prediction intervals | [MAPIE](https://mapie.readthedocs.io/), [crepes](https://crepes.readthedocs.io/), [torchcp](https://github.com/snap-stanford/torchcp) | Split + CQR coverage / width / interval score | `conformal` |
| Low-shot Bayesian linear regression | [BoTorch](https://botorch.org/) | RMSE / NLL / 95% coverage | `bayesian_linear` |
| Tweedie / compound-Poisson regression | [scikit-learn `TweedieRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html), statsmodels GLM | MAE / Tweedie deviance | `tweedie` |

## Run the comparisons

From a checkout of torchregress-harness next to torchregress (see its README for the
pixi environment and the `conformal`, `bayesian` and `tweedie` extras):

```bash
pixi run python tools/run_all.py --suites conformal,bayesian_linear,tweedie \
    --datasets diabetes,synthetic_hetero,synthetic_sinusoidal,synthetic_tweedie \
    --output results/latest/external_comparisons.json
```

Each row records the comparator library and version. A comparator that is not
installed is reported as skipped, never as a torchregress result.

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


**Harness suite:** [`suites/tabular/conformal.py`](https://github.com/astroai/torchregress-harness/blob/main/suites/tabular/conformal.py)

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

**Harness suite:** [`suites/tabular/bayesian_linear.py`](https://github.com/astroai/torchregress-harness/blob/main/suites/tabular/bayesian_linear.py)

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


**Harness suite:** [`suites/tabular/tweedie.py`](https://github.com/astroai/torchregress-harness/blob/main/suites/tabular/tweedie.py) (also runs a statsmodels Tweedie GLM as ground truth)

## Decision Criteria

| If you need… | Start with… | Then consider… |
|---|---|---|
| Conformal intervals on a deep backbone with custom losses | `torchregress.ConformalLoss` (split / CQR / UACQR) | MAPIE for sklearn-estimator wrappers, crepes for Mondrian / class-conditional / APS variants, torchcp for research-grade CP with NN backbones |
| Few-shot linear regression with closed-form posteriors | `torchregress.BayesianLinearHead` / `RecursiveBayesianHead` | BoTorch `SingleTaskGP` for nonlinear features |
| Tweedie / compound-Poisson target with a flexible backbone | `torchregress.TweedieLoss` / `CompoundPoissonLoss` | scikit-learn `TweedieRegressor` for interpretable log-link models |

## Limitations

- Capacity is intentionally not matched between libraries. The numbers are
  meant to anchor an operational default, not a horse race.
- The harness also runs these suites on real datasets (California housing, Diabetes,
  insurance claims and others); see its parity report.
- BoTorch adds a significant dependency footprint (gpytorch, botorch, pyro).
  Treat it as an opt-in, not a default.
- The comparator packages are not part of any torchregress extra, so default
  installs stay light; the harness installs them.
