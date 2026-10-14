# Public API review for 0.3.0

Release plan step 2.4, "Freeze the public API". The maintainer decides from
this review what changes before the first PyPI release, if anything. Nothing
in it has been applied, and no code was changed.

- **Library state:** `torchregress` `main` at `e02d186`. The package metadata
  still says `0.2.0`. **Harness state:** `torchregress-harness` working tree on
  2026-10-07. Every count was recomputed on that date.
- **Method:** every count comes from a script run against the installed
  package (`inspect`, `ast`) or from `grep`. Nothing is estimated. A use is
  resolved from imports and from attribute access on `torchregress` objects,
  not from bare word matches. A local variable called `bias` therefore does not
  count as a use of `torchregress.metrics.bias`. Comments and docstrings do not
  count.
- **Where usage was searched:**
  - `src/torchregress`, outside the defining module.
  - `tests/` and `examples/`.
  - `docs/`. Narrative pages are counted apart from the `docs/api/` reference
    tables. `docs/reports/` and `docs/loss_test_coverage.md` are excluded.
  - Every `.py` file in `torchregress-harness`.
- **Conventions:** **Recommendation** marks a judgement. Everything else is a
  verified finding. "Deprecate in 0.4" means a `DeprecationWarning` that names
  the replacement in 0.4 and removal in 0.6. That is the policy in `ROADMAP.md`
  ("Deprecation policy (from 0.4)"), which does not cover 0.3.0.

## Summary

| Item | Count |
|:--|--:|
| Subpackages in `torchregress.__all__` | 15 |
| Names in the 15 subpackage `__all__` lists | 476 (457 declared, plus 19 from `method_catalog`, which has no `__all__`) |
| Objects exported under two names or from two subpackages | 17, plus 4 top-level re-exports |
| One name exported for two different functions | 1 (`quantile_loss`) |
| Names importable from a subpackage but missing from its `__all__` | 12 |
| Names in the docs that do not exist anywhere in the package | 25 |
| Keyword arguments in documentation code blocks and examples that the called function does not have | 16 keyword arguments, in 11 calls on 5 pages; 0 in `examples/` |
| Public signatures analysed for naming and argument order | 823 (214 functions, 218 `__init__`, 391 public methods) |
| `utils` exports: make private now / deprecate in 0.4 / keep public | 28 / 9 / 16 |
| Public functions whose argument order breaks their module's own order | 4. In addition, one module has its own order (`inference.ppi`), one function puts the quantile level first (`metrics.pinball_loss`), and one method returns samples on a different axis (`NormalizingFlowLoss.sample`) |
| P0 items in `PROPOSALS.md` | 3. Item 1 is closed (fixed); items 2 and 3 are deferred to 0.5 |

**Recommendation in brief:**

1. Before 0.3.0, fix the docs and remove only names that nothing uses:
   - the 25 names in the docs that do not exist in the package;
   - the 11 documented calls that pass a keyword the function does not accept;
   - one docs example whose arguments are reversed;
   - 2 unused aliases;
   - the 9 imported names that leak from `method_catalog`;
   - the `CORALLoss` docstring side effect;
   - optionally, 28 internal helpers removed from the `torchregress.utils`
     namespace. This was dry-run against the library tests and the harness
     export-coverage tests.
2. Leave every rename, every argument-order change and every new export for
   0.4, under the deprecation policy.
3. Close P0 #1 in `PROPOSALS.md` as fixed. Record #2 and #3 as deferred to 0.5,
   where `ROADMAP.md` already schedules them.

---

## 1. Inventory

### 1.1 Public names per subpackage

The source is `__all__` of each subpackage listed in `torchregress.__all__`.
`method_catalog` has no `__all__`, so its count is every name without an
underscore.

| Subpackage | `__all__` entries | Classes | Functions | Notes |
|:--|--:|--:|--:|:--|
| `losses` | 138 | 109 | 29 | 6 aliases (section 1.2) |
| `metrics` | 97 | 31 | 66 | 7 names also exported by `calibration` |
| `test_time` | 41 | 26 | 15 | |
| `viz` | 34 | 0 | 34 | |
| `utils` | 53 | 11 | 42 | Section 2 |
| `ensemble` | 24 | 22 | 2 | |
| `method_catalog` | 19 | | | No `__all__`: 10 of its own names (4 dataclasses, the `CapabilityValue` type alias, 5 functions) and 9 leaked imports (`Any`, `Dict`, `Iterable`, `List`, `Literal`, `Optional`, `annotations`, `asdict`, `dataclass`) |
| `algorithms` | 18 | 16 | 2 | |
| `calibration` | 13 | 8 | 5 | |
| `semi_supervised` | 10 | 4 | 6 | |
| `inference` | 10 | 2 | 8 | |
| `comparison` | 6 | 0 | 6 | |
| `constraints` | 5 | 5 | 0 | |
| `causal` | 4 | 0 | 4 | |
| `prediction` | 4 | 1 | 3 | |
| **Total** | **476** | | | |

`torchregress.__all__` itself has 20 entries:

- the 15 subpackages;
- `BaseLoss`, `RegressionLoss` and `DistributionLoss`, which are also in
  `losses`;
- `iteratively_reweighted_least_squares`, which is also in `algorithms`;
- `__version__`.

`torchregress.health` is documented as
`from torchregress.health import check_health`. It is not in `__all__` and is
not lazy-loaded. That is consistent with it being an explicit submodule.

**Guards that already pin the API.** Any change below has to update these:

- `tests/test_public_api_contracts.py` pins the exact `__all__` of
  `torchregress`, `metrics`, `ensemble`, `algorithms`, `causal`, `inference`,
  `constraints`, `comparison`, `calibration`, `test_time`, `prediction` and
  `semi_supervised`, plus a set of signatures. It does not pin `losses`,
  `utils`, `viz` or `method_catalog`.
- `tests/test_loss_forward_signature_contracts.py` requires
  `forward(y_pred, target, ..., mask=None, weights=None, **kwargs)` on every
  `BaseLoss` subclass in `losses`.
- `tests/metrics/test_reference_parity.py` requires a parity test or a reason
  for every name in `metrics.__all__`.
- `tests/audit/test_audit_losses_a.py::test_A_LOSS_018_all_exports` requires
  `StudentTLoss`, `ExpectileCrossover` and `QuantileCrossover` in
  `losses.__all__`.
- `tests/test_utils_exports.py` pins 11 `utils` names.
- `tests/audit/test_audit_batch5.py::test_UTL_009_*` checks the tables in
  `docs/api/utils.md` against the code, in both directions.
- In the harness, `tests/test_export_coverage.py` requires each of the 360
  names in the `__all__` of 10 modules to be referenced by a suite or listed in
  `tools/export_coverage_allowlist.py`. The 10 modules are `losses`, `metrics`,
  `calibration`, `ensemble`, `algorithms`, `test_time`, `inference`, `causal`,
  `semi_supervised` and `constraints`. The test also fails on stale allowlist
  entries. Removing or adding an export in those 10 modules therefore needs a
  harness edit. `utils`, `viz`, `prediction`, `comparison` and
  `method_catalog` are outside this gate.

### 1.2 Duplicates and aliases

**One object under two names in `losses`.** There are 6 such aliases, all
module-level assignments.

| Alias | Same object as | Used externally |
|:--|:--|:--|
| `AsymmetricLeastSquaresLoss` | `ExpectileLoss` (`losses/expectile.py:328`) | 3 narrative docs pages; harness allowlist; loss registry key `"als"` |
| `CORALLoss` | `CumulativeLinkLoss` (`losses/ordinal.py:213`) | 2 harness suites (`ordinal`, `r2c_conformal`); 2 examples; 8 docs pages |
| `JackknifePlus` | `CVPlus` (`losses/conformal.py:1145`) | harness allowlist; 2 docs pages |
| `ExpectileCrossover` | `ExpectileCrossoverLoss` (`losses/expectile.py:473`) | tests only (`test_A_LOSS_018` requires it) |
| `MDNLoss` | `MixtureDensityLoss` (`losses/mdn.py:545`) | `MDNLoss`: harness `sbi_parity`, 2 examples, 17 docs pages. `MixtureDensityLoss`: 11 harness files |
| `QuantileCrossover` | `QuantileCrossoverLoss` (`losses/quantile.py:405`) | tests only (`test_A_LOSS_018` requires it) |

**Finding, a docstring bug.** `losses/ordinal.py:214` assigns
`CORALLoss.__doc__`. `CORALLoss` is the same class object as
`CumulativeLinkLoss`, so the assignment overwrites the `CumulativeLinkLoss`
docstring. Verified: `CumulativeLinkLoss.__doc__` now starts with
"CORAL ordinal loss — identical to CumulativeLinkLoss."

**One object exported from two subpackages.** There are 11 such objects.

| Object (defining module) | Exported from | Paths the callers use |
|:--|:--|:--|
| `ExpectedCalibrationError`, `MarginalCalibrationError`, `bias`, `calibration_metrics_report`, `calibration_score`, `expected_calibration_error`, `marginal_calibration_error` (`calibration/metrics.py`) | `metrics` and `calibration` | Examples use `torchregress.metrics`. `docs/metrics/calibration.md` uses a third path, `torchregress.metrics.calibration`, a module whose docstring calls it a "Backward-compatible shim". The harness uses none of the 7 |
| `RepresentationShiftInflator` (`calibration/shift.py`) | `test_time` and `calibration` | The harness uses `torchregress.test_time` (`suites/tabular/test_time_adaptation.py`) |
| `parse_heteroscedastic_output` (`utils/gaussian_output.py`) | `ensemble` and `utils` | The docs use `torchregress.ensemble`. The `utils` path appears in tests only |
| `low_rank_output_dim`, `split_low_rank_gaussian_output` (`utils/gaussian_output.py`) | `losses` and `utils` | The example uses `torchregress.losses`. The `utils` path is not used outside tests |

The 4 top-level re-exports are `BaseLoss`, `RegressionLoss`,
`DistributionLoss` and `iteratively_reweighted_least_squares`. All 4 are
pinned by `test_public_api_contracts.py`.

**One name for two different functions:** `quantile_loss`.

- `losses.quantile_loss(y_pred, target, quantile=0.5, mask=None, weights=None, reduction="mean")`
  is defined in `losses/quantile.py:375`.
- `utils.quantile_loss(y_pred, y_true, quantile)` is defined in
  `utils/quantile.py:11`.

**Distinct objects with confusable names.** `VarioScore` and `vario_score`
implement Zamo and Naveau's score: univariate, higher is better.
`VariogramScore` and `variogram_score` implement Scheuerer and Hamill's score:
multivariate, lower is better. The `VariogramScore` docstring warns about the
confusion. No change is proposed.

### 1.3 Names importable but not declared

These names exist as attributes of a subpackage but are missing from its
`__all__`. That means `import *` skips them, and the harness export-coverage
gate cannot see them.

| Subpackage | Name | What it is | Used by |
|:--|:--|:--|:--|
| `losses` | `CQR`, `CTI` | conformal classes in `losses/conformal.py` | examples (`from torchregress.losses import CQR, CTI`). The harness imports `CTI` (`distributional_bins`). Both are listed in `docs/api/conformal.md` and `docs/api/losses.md` |
| `losses` | `MultivariateScoreConformal`, `NonExchangeableConformalRegressor` | conformal classes | harness suites `multivariate_intervals` and `conformal`, both via `torchregress.losses.conformal` |
| `losses` | `BaseEIVLoss` | base class | 1 example |
| `semi_supervised` | `SelfAgreementTrainer` | alias of `TeacherStudentTrainer` (`semi_supervised.py:1184`) | 1 example (`examples/benchmarks/self_agreement_higgs_ood.py`) |
| `semi_supervised` | `PredictiveBatch`, `update_ema_teacher_` | imported names | the harness imports `PredictiveBatch` from here in 1 file |
| `test_time` | `OTScoreWeightEstimator` | class re-exported with the `X as X` idiom | src (`joint_tta`), 1 test |
| `test_time` | `RepresentationShiftCalibrator`, `SignificantSubspaceAligner` | aliases of `RepresentationShiftInflator` and `WeightedSubspaceMomentAligner` (`test_time/__init__.py:59-60`) | **nothing**. A grep of both repositories finds only the two assignments |
| `comparison` | `set_all_seeds` | imported name | — |

The harness also imports from submodule paths and through private names:

- `finite_sample_quantile` from `torchregress.losses.conformal`;
- `fetch_openml_dataset` from `torchregress.utils.openml_relaxed`;
- `_poly_features` from `torchregress.inference.orthogonal`, in one test.

Most harness imports go through submodules: there are 21 imports from
`torchregress.losses.nflows` and 17 from `torchregress.losses.conformal`.
Submodule paths are therefore part of the API the harness relies on. No
recommendation below renames a module.

### 1.4 Docs that disagree with the package

**Names that exist nowhere in the package.** 20 of them are listed in
`docs/api/` tables:

- `MultiDimensionalConformalLoss`, `conformal_loss`, `ExplicitEIVAdapter`,
  `create_eiv_loss`, `enhanced_poisson_gaussian_loss`,
  `poisson_gaussian_likelihood_ratio_loss` and `poisson_gaussian_mixture_loss`,
  in `conformal.md` and `losses.md`;
- `PackedEnsembleRegressor`, `PackedEnsembleOutput`, `MCDropoutModel`,
  `BayesianModelAveraging` and `DynamicEnsembleWeighting`, in `ensemble.md`;
- `mse`, `mae`, `mean_absolute_percentage_error`, `mean_squared_log_error` and
  `outlier_fraction`, in `metrics.md`;
- `perturbation_instability_score`, in `semi_supervised.md`;
- `SupportsRepresentation` and `SupportsAdaptationParameters`, in
  `test_time.md`.

5 more appear only in narrative imports:

- `set_seed`, in `docs/examples/ensemble_methods.md`;
- `GradientAccumulation` and `compile_model`, in `docs/guide/performance.md`;
- `multivariate_rmse`, in `docs/metrics/multivariate.md`;
- `explained_variance_score`, in `docs/metrics/point.md`.

Overall, 15 imported names in 9 narrative pages do not resolve. Of 329
`from torchregress... import ...` statements in the narrative pages, every
other name does.

**Calls with keyword arguments the function does not have.** The docs code
blocks (392 blocks, of which 365 parse as Python) and the `examples/` scripts
were parsed with `ast`. Each call to a name imported from `torchregress` in
the same snippet was checked against `inspect.signature`. This finds 16
keyword arguments in 11 calls, all in the docs and none in `examples/`. The
check does not see calls through an object, such as
`loss.predict_interval(...)`, or names imported elsewhere.

| Page and line | Call | Problem |
|:--|:--|:--|
| `docs/metrics/point.md:24,39,54,86,334` | `mean_squared_error`, `mean_absolute_error`, `rmse`, `r2_score`, `regression_metrics_report` with `mask=`, `weights=` | The metrics take `sample_weight` and no `mask`. `r2_score` takes neither. 5 calls, 10 of the 16 keywords |
| `docs/metrics/point.md:286,287` | `tail_mae(..., mask=...)`, `tail_rmse(..., mask=...)` | No `mask` parameter |
| `docs/examples/ensemble_methods.md:302` | `prediction_interval_coverage(y_true, lower, upper, confidence=0.95)` | The signature is `(lower_bound, upper_bound, y_true, alpha=0.1)`. The call has the wrong argument order and the wrong keyword, and `0.95` would be read as a miscoverage |
| `docs/losses/quantile_expectile.md:171` | `AsymmetricLeastSquaresLoss(tau=0.75)` | The constructor is `(expectile=0.5, reduction="mean")` |
| `docs/methods/visualization.md:244` | `create_grid_figure(n_plots=6, n_cols=3, ...)` | The keyword is `ncols` |
| `docs/metrics/distribution.md:111` | `plot_pit_histogram(y_pred_dist, y_true, bins=20)` | The signature is `(y_pred, y_pred_std, y_true, n_bins=20, ...)` |

The `docs/api/` tables carry the same kind of error. Row text, not code blocks:

- `docs/api/losses.md` gives `AsymmetricLeastSquaresLoss(tau=0.5)` and
  `quantile_loss(y_pred, y, tau)`. The real signatures are
  `(expectile=0.5, ...)` and
  `quantile_loss(y_pred, target, quantile=0.5, mask=None, weights=None, reduction="mean")`.
  The same table calls `QuantileCrossover` a "helper dataclass", but it is an
  alias of `QuantileCrossoverLoss`.
- `docs/api/metrics.md:116` gives `calibration_metrics_report(y_pred, y_pred_std, y)`.
  The real signature starts `(dist_or_samples, y_true, y_pred_quantiles, ...)`.

**Other docs errors:**

- `docs/methods/visualization.md:120` calls
  `plot_target_density_error_overlap(y_pred, y_true)`. The signature is
  `(y_true, y_pred, ...)`, so a reader who copies the line swaps the two
  arguments silently. Section 4 covers the cause.
- `docs/api/semi_supervised.md` says `SelfAgreementTrainer` "exposes additional
  methods: `compute_agreement()`, `unsupervised_loss()`". It is a plain alias
  of `TeacherStudentTrainer`, which has neither method. Verified with
  `hasattr`.
- `docs/api/test_time.md` shows `flatten_adaptation_parameters` used with
  `SupportsAdaptationParameters`, which does not exist.

**Exports missing from `docs/api/`.** `AGENTS.md` says "Every exported
class/function in `__init__.py` must appear in the relevant docs page". These
exports are missing:

- `metrics.VariogramScore` and `metrics.variogram_score`, which appear only in
  `docs/metrics/distribution.md`;
- `inference.OrthogonalEstimate`, `inference.naive_linear_estimate` and
  `inference.orthogonal_partially_linear`;
- the 4 `method_catalog` dataclasses and `CapabilityValue`.

**Documented but not exported.** These names exist in the code but are not in
any `__all__`. Whether they are public is undecided:

- `CQR` and `CTI`;
- `huber_elementwise`, `log_cosh` and `tukey_biweight`
  (`losses/utils_robust.py`);
- `create_metric_result`, `metric_state_list`, `metric_state_tensor`,
  `prepare_functional_metric` and `validate_inputs` (`metrics/utils.py`);
- `float_dtype` (`utils/tensor_ops.py`);
- `add_zero_line` and `is_lower_better` (`viz/utils.py`);
- the two `fetch_openml_*` loaders;
- `_safe_denominator`, which is listed in `docs/api/utils.md` despite its
  leading underscore;
- `SelfAgreementTrainer`.

---

## 2. Privatisation candidates

**Rule used, which is conservative.** A name is **keep public** if any of the
following is true:

- the harness uses it;
- an example uses it;
- a narrative docs page presents it, or the code example on an API page uses
  it;
- it belongs to a documented family, such as a base class.

Appearing only as a row in a `docs/api/` table does not count, because
`AGENTS.md` makes that row mandatory for every export. A name with none of
these uses is **make private now** when it is plumbing or a second path to an
object exported elsewhere. It is **deprecate in 0.4** when it has user-level
meaning.

"Make private" for `utils` means removing the name from
`torchregress/utils/__init__.py` and from `utils.__all__`. The function stays
where it is, in its submodule, and every caller in `src` already imports it
from there. Verified: the only `src` import from the `torchregress.utils`
namespace is `semi_supervised.py`, which imports `update_ema_teacher_`, and that
name is kept. No module is renamed, because the harness imports submodule paths
(section 1.3).

The columns below have these meanings:

- **Test files:** the number of files in `tests/` that reference the name.
- **Docs:** narrative pages, then whether the name has a row in `docs/api/`.
- **Harness:** suite or file names. "allowlisted" means the name is in
  `tools/export_coverage_allowlist.py`.

### 2.1 `utils`: make private now (28)

| Name | Defined in `utils/` | Used in src outside own module | Test files | Examples | Docs | Harness | Reason |
|:--|:--|:--|--:|--:|:--|:--|:--|
| `normal_cdf` | `distributions.py:13` | `calibration/posthoc` | 3 | 0 | — / yes | — | One-line helper; `torch.special.ndtr` is the public equivalent |
| `parse_heteroscedastic_output` | `gaussian_output.py:87` | `ensemble/models`, `ensemble/utils` | 2 | 0 | — / yes | allowlisted | Second path to the same object. The docs present `torchregress.ensemble.parse_heteroscedastic_output`, which stays |
| `low_rank_output_dim` | `gaussian_output.py:61` | — | 2 | 0 via `utils` (1 via `losses`) | guide page uses the `losses` path / yes | allowlisted | Second path to the same object. `torchregress.losses.low_rank_output_dim` stays |
| `split_low_rank_gaussian_output` | `gaussian_output.py:68` | — | 2 | 0 via `utils` (1 via `losses`) | as above | allowlisted | Second path to the same object. The `losses` path stays |
| `subsample_rows` | `numpy_stats.py:8` | `test_time/subspace` | 3 | 0 | — / yes | — | NumPy plumbing |
| `winsorize` | `numpy_stats.py:39` | `test_time/subspace` | 3 | 0 | — / yes | — | NumPy plumbing |
| `labels_to_levels` | `ordinal.py:21` | `losses/ordinal` | 3 | 0 | — / yes | — | Encoding plumbing for ordinal losses |
| `normalize_class_probs` | `ordinal.py:36` | `losses/ordinal` | 1 | 0 | — / yes | — | Encoding plumbing |
| `class_probs_to_levels` | `ordinal.py:51` | `losses/ordinal` | 1 | 0 | — / yes | — | Encoding plumbing |
| `get_device` | `pytorch_compat.py:27` | — | 1 | 0 | — / yes | — | Not used anywhere, not even in `src` |
| `quantile_loss` | `quantile.py:11` | `losses/quantile` | 3 | 0 | — / yes | (the allowlist entry is for `losses.quantile_loss`) | A different function from `losses.quantile_loss` under the same name: no mask, weights or reduction, and `y_true` instead of `target` |
| `multi_quantile_loss` | `quantile.py:37` | `losses/quantile` | 1 | 0 | — / yes | — | Functional plumbing behind `MultiQuantileLoss`. Uses `y_true`, unlike the losses |
| `apply_mask` | `tensor_ops.py:74` | `losses/eiv` | 1 | 0 | — / yes | — | Plumbing for errors-in-variables (EIV) losses |
| `convert_to_tensor` | `tensor_ops.py:15` | 12 modules (`calibration/metrics`, `metrics/*`, ...) | 3 | 0 | — / yes | — | Input-coercion plumbing |
| `ensure_batch_dim` | `tensor_ops.py:67` | `metrics/ood`, `metrics/utils` | 1 | 0 | — / yes | — | Shape plumbing |
| `prepare_cross_covariance` | `tensor_ops.py:182` | `losses/eiv` | 1 | 0 | — / yes | — | EIV plumbing |
| `prepare_model_input_for_gradients` | `tensor_ops.py:207` | `losses/eiv` | 1 | 0 | — / yes | — | EIV plumbing |
| `compute_model_gradients` | `tensor_ops.py:216` | `losses/eiv` | 2 | 0 | — / yes | — | EIV plumbing |
| `calculate_gaussian_nll` | `tensor_ops.py:284` | `losses/eiv` | 4 | 0 | — / yes | — | EIV plumbing. Users have `GaussianNLLLoss` and `metrics.gaussian_nll` |
| `calculate_propagated_variance` | `tensor_ops.py:329` | `losses/eiv` | 1 | 0 | — / yes | — | EIV plumbing |
| `validate_reduction` | `validation.py:16` | `losses/base` | 2 | 0 | — / yes | — | Argument checks |
| `validate_range` | `validation.py:108` | `losses/censored`, `expectile`, `quantile`, `robust` | 2 | 0 | — / yes | — | Argument checks |
| `validate_quantile` | `validation.py:152` | `losses/quantile` | 2 | 0 | — / yes | — | Argument checks |
| `validate_weights` | `validation.py:166` | 5 loss modules | 1 | 0 | — / yes | — | Argument checks |
| `validate_metric_inputs` | `validation.py:229` | `calibration/metrics`, `metrics/utils` | 1 | 0 | — / yes | — | Argument checks |
| `validate_sample_weight` | `validation.py:255` | `metrics/point`, `metrics/utils` | 1 | 0 | — / yes | — | Argument checks |
| `check_tensor` | `validation.py:262` | 5 `algorithms` modules | 1 | 0 | — / yes | — | Argument checks |
| `validate_url` | `security.py:8` | `utils/openml_relaxed` | 1 | 0 | — / yes | — | Download plumbing for the OpenML loader |

### 2.2 `utils`: deprecate in 0.4 (9)

| Name | Defined in `utils/` | Used in src outside own module | Test files | Examples | Docs | Harness | Reason |
|:--|:--|:--|--:|--:|:--|:--|:--|
| `validate_positive` | `validation.py:52` | `losses/contrastive`, `losses/robust` | 2 | 0 | — / yes, in the API page's code example | — | Plumbing, but the `docs/api/utils.md` quick example calls it |
| `masked_mean` | `tensor_ops.py:149` | — | 2 | 0 | — / yes, in the code example | — | User-level helper with no external use. The quick example calls it |
| `masked_sum` | `tensor_ops.py:167` | — | 2 | 0 | — / yes | — | Sibling of `masked_mean` |
| `masked_reduction` | `tensor_ops.py:97` | — | 2 | 0 | — / yes | — | Sibling of `masked_mean` |
| `ipw_weights` | `propensity.py:10` | `losses/imbalanced`, `losses/uncertain_gt` | 3 | 0 | — / yes, in the code example | — | User-level statistic with no external use. The quick example calls it |
| `split_mean_log_variance` | `gaussian_output.py:22` | `losses/gaussian`, `losses/uncertain_gt` | 2 | 0 | — / yes | — | Gaussian-head helper with no external use |
| `variance_from_logvar` | `gaussian_output.py:11` | `ensemble/_variance` | 1 | 0 | — / yes | — | Gaussian-head helper with no external use |
| `cumulative_probs_to_pmf` | `ordinal.py:66` | — | 1 | 0 | — / yes | — | Its sibling `cumulative_logits_to_pmf` is kept, because an example uses it |
| `EnsemblePerturbationAugmenter` | `augment.py:106` | `losses/eiv` | 0 | 0 | — / yes | — | No tests and no external use |

**Recommendation:** decide each of these in 0.4. Make private the ones that
are plumbing, and either document the rest properly or keep them.

### 2.3 `utils`: keep public (16)

| Name | Reason |
|:--|:--|
| `BSplineDensityBasis` | Narrative docs (`losses/density_basis.md`) |
| `Augmentation`, `Adversarial` | `Adversarial` is in 4 narrative pages. `Augmentation` is its base class |
| `ordinal_predict` | Harness `suites/tabular/ordinal.py`, 3 examples and the docs |
| `CORALHead` | Harness `ordinal` and `r2c_conformal` |
| `cumulative_logits_to_pmf` | 1 example |
| `set_all_seeds` | 2 examples |
| `generate_pseudo_labels` | 1 example and narrative docs |
| `update_ema_teacher_` | Harness `stress_matrix`, 1 example, 2 docs pages. `semi_supervised.py` imports it from `torchregress.utils` |
| `LogTransform`, `BoxCoxTransform`, `SqrtTransform`, `YeoJohnsonTransform`, `make_target_transform` | Narrative docs (`losses/transforms.md`) |
| `TargetTransform`, `IdentityTransform` | Base class and member of the documented transform family |

### 2.4 Named candidates outside `utils`

| Name | Defined | Src outside module | Test files | Examples | Docs | Harness | Recommendation | Reason |
|:--|:--|:--|--:|--:|:--|:--|:--|:--|
| `losses.symmetric_spd_matrix_sqrt` | `losses/gaussian_wasserstein.py:82` | — | 5 | 0 | 1 narrative page / yes | allowlisted | **keep public** | `docs/losses/gaussian_wasserstein.md:50` says it is "exported from `torchregress.losses`" and documents its gradient |
| `test_time.flatten_adaptation_parameters` | `test_time/base.py:35` | — | 1 | 0 | — / yes, with a code snippet | allowlisted | **deprecate in 0.4** | No use in `src`, examples or the harness. Its docs snippet relies on a protocol that does not exist. Removing it also needs the harness allowlist entry deleted |
| `test_time.RepresentationShiftCalibrator`, `test_time.SignificantSubspaceAligner` | `test_time/__init__.py:59-60` | — | 0 | 0 | — / no | — | **make private now** (delete the two lines) | Undeclared aliases that nothing references |
| 9 leaked names in `method_catalog` | — | — | — | — | — | — | **make private now** (add `__all__` with the 10 real names) | Typing and `dataclasses` imports. Adding `__all__` changes only `import *`; `tools/render_method_catalog.py` and the tests use attribute access |

**Other exports with no use in examples, narrative docs or the harness.**
Outside `utils` and `method_catalog` there are 38 such exports.
**Recommendation: keep all of them public.** They fall into these groups:

- **Result and config containers** that public APIs return or consume (12):
  `EnsembleFitConfig`, `BatchEnsembleOutput`, `SAGERegAgreement`,
  `SAGERegOutput`, `AdaptationBatch`, `JointTTAResult`, `LabelShiftEstimate`,
  `ShiftFactoredTransportState`, `SubspaceAlignmentState`,
  `LocalConsistencyConfig`, `OrthogonalEstimate` and `AdaptivePriorGuide`.
- **Metrics awaiting the scoringrules parity tests** (16): `DawidSebastianiScore`,
  `PinballMetric`, `WassersteinGaussian`, `wasserstein_gaussian_p2`,
  `EntropyScore`, `KernelDensityScore`, `MahalanobisDistance`,
  `TypicalityScore`, `HuberMetric`, `MedianAbsoluteError`,
  `MedianAbsoluteDeviation`, `noisy_target_gaussian_nll`, `consistency_error`,
  `pseudo_label_acceptance_rate`, plus the two functional losses
  `discrete_wasserstein1` and `rank_n_contrast_loss`.
- **Helpers:** `bars_to_density_grid`, `quantiles_to_density_grid` and
  `samples_to_density_grid` (documented in `docs/api/utils.md`),
  `local_consistency_weights` (used by `src`),
  `gaussian_moments_from_binned_probabilities`, `ParameterEMA`,
  `SoftmaxModelCombiner`, `distributional_pseudo_loss`, and
  `flatten_adaptation_parameters` (covered above).
- `algorithms.sample_synthetic_environments` has no tests at all. Review it in
  0.4.

---

## 3. Naming inconsistencies

**Method.** `inspect.signature` was run over every public function, every
`__init__` and every public method defined on an exported class: 823
signatures, with `method_catalog` dataclass fields excluded. Parameter names
were then grouped by concept. "Signatures" counts each signature once. Where a
spelling means something else in a given place, that is noted, so the rows
compare like with like.

Some spellings from the release plan do not occur as parameters of any public
signature: `preds`, `num_samples` and `variance`. `num_samples` appears only
inside `algorithms/irls.py`.

| Concept | Spellings found (signatures) | Where the minority spellings are | Recommended canonical spelling |
|:--|:--|:--|:--|
| Ground truth | `target` 108 (losses 99, metrics 4, calibration 3, other 2); `y_true` 87 (metrics 69, viz 13, other 5); `y` 23 (estimator `fit`/`partial_fit`, `causal`, `inference.orthogonal`, transforms' `inverse`); `targets` 3; `labels` 1; `y_obs` 1 | `target` in metrics: `metrics/censored.py`: `observed_mae`, `concordance_index`; `metrics/uncertain.py`: `noisy_target_gaussian_nll`. `target` in calibrators: `calibration/posthoc.py`: `VarianceTemperatureScaler.fit`, `IsotonicMeanCalibrator.fit`, `PITCalibrator.pit_from_gaussian`. `targets`: `losses/imbalanced.py`: `FeatureDistributionSmoother.forward` and `.update_running_stats`; `test_time/label_shift.py`: `gaussian_bin_edges_from_targets`. `labels`: `losses/contrastive.py`: `rank_n_contrast_loss`. `y_obs`: `viz/results.py`: `plot_causal_uplift_qini` | **Two conventions by layer.** Losses use `target`, as `AGENTS.md` requires and a test enforces. Metrics, calibration and viz use `y_true`. Estimator `fit` uses `(X, y)` |
| Prediction | `y_pred` 169; `predictions` 6; `model_output` 3; `prediction` 1; `pred` 1; `output` 1 | `predictions`: `metrics/ensemble.py`: `ensemble_mean`, `ensemble_std`, `ensemble_statistics`; `metrics/interval.py`: `interval_metrics_report`; `viz/results.py`: `plot_model_ensemble_contributions`; field of `AdaptationBatch`. `model_output`: `metrics/ood.py`: `TypicalityScore.update`, `typicality_score`, `ood_metrics_report`. `prediction`: `utils/semisupervised.py`: `generate_pseudo_labels`. `pred`: `ensemble/bnn.py`: `BayesianNeuralNetwork.elbo_loss`. `output`: `utils/gaussian_output.py`: `parse_heteroscedastic_output` | **`y_pred`**. Container fields, such as `AdaptationBatch.predictions`, may keep their nouns |
| Predicted mean | `pred_mean` 9; `mean` 6 as a predicted mean; `y_pred_mean` 2; `loc` 1; `means` 8 (member stacks in `metrics/ensemble.py`) | `mean`: `metrics/distribution.py`: `gaussian_nll`, `crps_gaussian`; `test_time/label_shift.py`: `gaussian_bin_probabilities`, `correct_gaussian_predictions_for_label_shift`. `y_pred_mean`: `metrics/distribution.py`: `DawidSebastianiScore.update`, `dss_score`. `loc`: `WassersteinGaussian.update`. (`mean` in `metrics/ood.py` Mahalanobis is a reference feature mean, a different concept) | **`pred_mean`**. Dataclass fields (`PredictiveBatch.mean`, `BatchEnsembleOutput.mean`) keep `mean`, and member stacks keep `means` |
| Predicted variance | `pred_var` 2; `pred_variance` 2; `var` 2; `variances` 8 (member stacks); `predictive_variance` 1 (field) | `pred_variance`: `metrics/uncertain.py`: `noisy_target_gaussian_nll`, `uncertain_gt_metrics_report`. `var`: `metrics/distribution.py`: `gaussian_nll`; `utils/tensor_ops.py`: `calculate_gaussian_nll` | **`pred_var`** |
| Predicted standard deviation | `y_pred_std` 7; `std` 4 as a predicted std; `pred_std` 2; `scale` 1 as a predicted std | `y_pred_std`: `viz/diagnostic.py` ×5; `metrics/distribution.py`: `DawidSebastianiScore.update`, `dss_score`. `std`: `metrics/distribution.py`: `crps_gaussian`; `calibration/shift.py`: `RepresentationShiftInflator.calibrate_std`; `test_time/label_shift.py` ×2. `scale`: `WassersteinGaussian.update`. (`scale` in robust losses is a loss parameter, a different concept) | **`pred_std`**. The `pred_*` family (`pred_mean`, `pred_std`, `pred_var`) has 15 signatures against 9 for `y_pred_*` |
| Log-variance tensor | `log_var` 1 (plus the return names of `split_mean_log_variance`); `log_variance` 1 | `log_variance`: `utils/semisupervised.py`: `generate_pseudo_labels` | **`log_var`**. Note that `log_variance: bool` in `GaussianNLLLoss.__init__` and `CensoredGaussianNLLLoss.__init__` is a flag that shares the spelling. `log_input: bool` in 4 Poisson and Tweedie losses mirrors `torch.nn.PoissonNLLLoss`; keep it |
| Coverage level | `alpha` = miscoverage, 31 signatures: 11 conformal classes in `losses/conformal.py`; 8 in `metrics/interval.py` and `metrics/ensemble.py`; 6 in `test_time`; 2 in `inference/ppi.py`; 2 in `causal/dr.py`; 2 in `calibration/semicp.py`. `confidence` = nominal coverage, 10 signatures (an 11th, `utils.generate_pseudo_labels(confidence=...)`, is a per-sample confidence tensor, a different concept). `alpha` = nominal coverage, 1 signature, the opposite meaning. `credible_interval` 1 | `confidence`, default 0.95: `predict_interval` of `EvidentialRegressionLoss` (and `predict_interval_gaussian`), `MixtureDensityLoss`, `FullNetworkLaplace`, `SnapshotEnsemble`, `MCDropoutWrapper`, `BayesianNeuralNetwork`, `HeteroscedasticBNN`; `inference/orthogonal.py`: `naive_linear_estimate`, `orthogonal_partially_linear`. **Opposite meaning:** `metrics/distribution.py:304` `highest_posterior_density_coverage(alpha=0.1)` treats `alpha` as the nominal mass of the region. Verified on calibrated Gaussian predictions (20,000 draws, 3,201-point grid): `alpha=0.9` gives coverage 0.904 and `alpha=0.1` gives 0.100. `credible_interval`: `viz/diagnostic.py`: `plot_distribution_comparison`. Unrelated uses of `alpha`, which are left alone: `BarronLoss` (shape), `CVaRLoss` (tail fraction), `BatchEnsembleRegressor` (fast-weight scale), `Adversarial` (step size), and 4 `viz` functions (transparency) | **`alpha` = miscoverage everywhere** (31 against 10). In 0.4, add `alpha` to the 10 `confidence` signatures and deprecate `confidence`. The harness passes `confidence=` to `MixtureDensityLoss`/`EvidentialRegressionLoss.predict_interval` (`distributional_bins.py:995`) and to `naive_linear_estimate` (`orthogonal_inference.py:202`), so a plain rename would break it. For HPD, see section 6.2. Defaults differ too (0.1 against 0.95 coverage, and `PPIConfig.alpha=0.1` against `ppi_pp_mean_ci(alpha=0.05)`). Changing them would change behaviour, so keep them and document them |
| Per-sample weights | `weights` 98 (losses 96); `sample_weight` 9; `sample_weights` 3; `test_weights` 11; `calibration_weights` 2; `weights_cal` 1 | `sample_weight`: `metrics/point.py` ×6; `test_time/bayes.py`: `BayesianLinearHead.fit`, `RecursiveBayesianHead.partial_fit`. `sample_weights`: `semi_supervised.py`: `distributional_pseudo_loss` (a loss); `test_time/label_shift.py`: `PosteriorLabelShiftAdapter.estimate`, `estimate_target_prior_em`. `weights_cal`: `calibration/semicp.py`: `SemiConformalCalibrator.fit`. `test_weights` (11 conformal `predict_interval` methods) means test-point weights, a separate concept | **Losses use `weights`**, as `AGENTS.md` requires and a test enforces. **Metrics and estimators use `sample_weight`**, the scikit-learn spelling that `docs/api/metrics.md` already states. Fix `sample_weights` (to `weights` in the loss, `sample_weight` elsewhere) and `weights_cal` (to `calibration_weights`) |
| Monte Carlo draw count | `n_samples` 52; `mc_samples` 3; `n_mc_samples` 1; `num_samples` 0 | `mc_samples`: `losses/conformal.py`: `MonteCarloConformal.calibrate`, `.predict_interval`; `algorithms/ivon.py`: `IVON.__init__`. `n_mc_samples`: `algorithms/warmup_mc.py`: `WarmupMCTrainer.__init__`. (`sample_size` in `test_time/label_shift.py` is a subsample size, and `n_simulations` in `SIMEX` is a domain term; both are different concepts) | **`n_samples`** |
| Number of bins or classes | `n_bins` 18; `num_bins` 1; `num_classes` 4 | `num_bins`: `losses/balanced_mse.py`: `BinReweightedMSELoss`. `num_classes`: `metrics/ordinal.py`: `quadratic_weighted_kappa`; `utils/ordinal.py`: `labels_to_levels`, `ordinal_predict`, `CORALHead` | **`n_bins`**. Keep `num_classes`: it is the only spelling for classes, and torch and torchmetrics use it |
| Random seed | `seed` 9; `random_state` 8; `base_seed` 1; `generator` 4 (`torch.Generator`, a different type) | `random_state`: `test_time/subspace.py`: `FeatureStatNormalizer`, `WeightedSubspaceMomentAligner`; `test_time/label_shift.py`: `PosteriorLabelShiftAdapter`, `estimate_target_prior_em`; `test_time/transport.py`: `ShiftFactoredTransportConfig`; `test_time/selection.py`: `LocalConsistencyConfig`; `calibration/shift.py`: `RepresentationShiftInflator`; `utils/numpy_stats.py`: `subsample_rows`. `test_time` uses both spellings | **`seed`** for integer seeds |
| Numerical floors | `eps` 80; `jitter` 13 (a diagonal added to a covariance, used consistently); `min_variance` 14; `min_scale` 5; `variance_floor` 1; `var_clamp` 1; `min_std` 1; `min_sigma` 1; `min_uncertainty` 1; `epsilon` 2 | `variance_floor`: `calibration/posthoc.py`: `VarianceTemperatureScaler` (the harness benchmarks this option). `var_clamp`: `ensemble/swag.py`: `SWAG`. `min_std`: `losses/mdn.py`: `MixtureDensityLoss`. `min_sigma`: `losses/eiv.py`: `InputNoiseAugmentationLoss`. `min_uncertainty`: `MonteCarloConformal`. `epsilon`: `algorithms/irls.py`: `IRLSConfig` (a numerical eps). (`epsilon` in `Adversarial` is a perturbation radius; keep it) | **`eps`**, **`min_variance`** for variance floors, **`min_scale`** for standard-deviation floors |
| Width of inputs, outputs and hidden layers | Input: `in_features` 8, `input_dim` 5, `feature_dim` 2, `input_size` 1. Output: `out_features` 5, `output_dim` 4, `target_dim` 3, `output_size` 1, `n_features_y` 1. Hidden: `hidden_dim` 7, `hidden_dims` 3, `hidden_size` 1, `hidden_features` 1, `hidden` 1. **`n_features` 7 means target dimension**, not input features, as it would in scikit-learn. `d` 4 also means target dimension | `input_size`, `hidden_size`: `ensemble/models.py`: `BatchEnsembleMLPBackbone`. `output_size`: `HeteroscedasticBatchEnsembleModel`. `feature_dim`: `FeatureDistributionSmoother`, `BatchEnsembleRegressor`. `hidden`: `test_time/shift_weights.py`: `DomainClassifierRatioEstimator`. `n_features` as target dimension: `MixtureDensityLoss`, `MultivariateGaussianLoss`, `InputNoiseMDNLoss`, `create_flow_model`, `MDNEnsembleModel`, `low_rank_output_dim`, `split_low_rank_gaussian_output`. `d`: `losses/sls.py`: `SLSLoss`, `UnionFrontier`, `MahalanobisFrontier`, `VolumePreservingFlow` | **Layers** (`nn.Linear`-like) use `in_features`/`out_features`. **Models** use `input_dim`/`output_dim`/`hidden_dim(s)`. **Target dimensionality** uses `target_dim`. Keep `hidden_features` in `create_flow_model`, which passes it through to zuko. The `n_features` rename touches classes the harness uses heavily (`MixtureDensityLoss`, in 11 harness files), so the lowest-cost option is to document it |
| Optimiser settings | `lr` 10 / `learning_rate` 2; `max_iter` 4 / `max_iterations` 1 / `n_steps` 1; `tol` 3 / `tolerance` 1; `epochs` 9 / `total_epochs` 1 | `learning_rate`: `losses/eiv.py`: `OrthogonalDistanceRegressionLoss`; `test_time/ot_conformal.py`: `ScoreCDFReweighter`. `max_iterations`, `tolerance`: `OrthogonalDistanceRegressionLoss`. `n_steps`: `ScoreCDFReweighter`. `total_epochs`: `WarmupMCTrainer` | **`lr`, `max_iter`, `tol`, `epochs`** |
| Quantile levels | `quantile` 6 (scalar level); `quantiles` 6 (a sequence of levels in losses, but **predicted values** in `prediction`); `quantile_levels` 2; `q` 2; `levels` 1; `level` 1; `quantile_value` 1; `y_pred_quantiles` 8 (predicted values, in metrics) | `levels`: `losses/nflows.py`: `NormalizingFlowLoss.quantile`. `level` and `quantile_value`: `metrics/distribution.py`: `pinball_loss`. `q`: `inference/ppi.py`: `ppi_quantile_ci`; `test_time/transport.py`: `ShiftFactoredPredictiveTransport.ppi_target_ci`. `quantiles`/`quantile_levels` as values/levels: `prediction.py`: `PredictiveBatch`, `quantiles_to_density_grid` | **Levels use `quantile` (scalar) and `quantiles` (sequence). Predicted values use `y_pred_quantiles`.** `PredictiveBatch` uses `quantiles` for values, which contradicts the losses. It is a container the harness imports in 5 places, so the recommendation is to document the field names rather than rename them |
| Ensemble size | `ensemble_size` 9; `n_models` 1 | `n_models`: `ensemble/swag.py`: `MultiSWAG`. (`n_snapshots` and `max_num_models` are different concepts) | **`ensemble_size`** |
| Dropout | `dropout` 4; `dropout_rate` 1 | `dropout_rate`: `ensemble/mc_dropout.py`: `MCDropoutWrapper` | **`dropout`** |
| Mixture components | `n_components` 3; `K` 2 | `K`: `losses/sls.py`: `SLSLoss`, `UnionFrontier` (its docstring says "Number of mixture components") | **`n_components`** |
| Predicted interval bounds | `lower_bound`/`upper_bound` 7 as predicted bounds (`metrics/interval.py`), 4 as censoring bounds (`losses/censored.py`, `metrics/censored.py`); `pred_lower`/`pred_upper` 2; `y_lower`/`y_upper` 1; `lower`/`upper` 1 | `pred_lower`/`pred_upper`: `calibration/semicp.py`: `SemiConformalCalibrator.calibrate_interval`; `metrics/censored.py`: `interval_overlap_rate`, which takes both spellings because it compares a predicted interval with a censoring interval. `y_lower`/`y_upper`: `viz/diagnostic.py`: `plot_prediction_intervals`. `lower`/`upper`: `semi_supervised.py`: `conformal_width_to_weight` | **`lower_bound`/`upper_bound`**. Keep `interval_overlap_rate` as it is |
| Density grid size | `n_support` 10; `n_grid` 2; `grid_size` 1 | `n_grid`: `test_time/ot_conformal.py`: `OptimalTransportCoverageGap`, `ScoreCDFReweighter`. `grid_size`: `losses/conformal.py`: `SLSConformal`. (`n_points` in `RiskCoverageCurve` counts curve points, a different concept) | **`n_support`** |
| Device | `device` 17; `device_str` 1 | `device_str`: `utils/pytorch_compat.py`: `get_device` | **`device`**. Resolved by section 2.1 |
| Feature matrix | `x` 113; `X` 14; `features` 9 | `X` is used by scikit-learn-style `fit`/`predict` (`SIMEX`, `LatentNN`, `WarmupMCTrainer`, `DelayedLabelResidualAdapter`, `WeightedConformalRegressionAdapter`). `features` is used for representation inputs (`BayesianLinearHead`, `FeatureDistributionSmoother`, `rank_n_contrast_loss`) | Keep **`x`** for tensors going into `nn.Module`s and **`X`** for scikit-learn-style estimators. No change proposed |

---

## 4. Argument-order inconsistencies

**Method.** Among the 823 signatures, 197 take both a prediction-like and a
target-like positional argument. Each of these was classified by which of the
two comes first.

| Module group | Prediction first | Target first | Dominant |
|:--|--:|--:|:--|
| `losses` | 99 | 0 | prediction first. `test_loss_forward_signature_contracts.py` enforces `forward(y_pred, target, ...)` for `BaseLoss` subclasses |
| `metrics` + `calibration` | 71 | 1 | prediction first |
| `viz` | 12 | 2 | prediction first |
| `inference.ppi` | 0 | 6 | target first, consistently |
| `ensemble`, `test_time`, `comparison`, `utils` | 6 | 0 | prediction first |

**Public functions that break their module's dominant order:**

| Function | Signature | Module order | Callers |
|:--|:--|:--|:--|
| `calibration_score` (`calibration/metrics.py:333`; exported from `metrics` and `calibration`) | `(y_true, pred_mean, pred_std, n_levels=...)` | `(y_pred, y_true)` | `examples/benchmarks/self_agreement_higgs_ood.py:803` (positional); `docs/metrics/calibration.md:117`; `docs/examples/ensemble_methods.md:305`. The `docs/api/metrics.md` row for `calibration_metrics_report` shows yet another order, `(y_pred, y_pred_std, y)` |
| `plot_target_density_error_overlap` (`viz/diagnostic.py:1822`) | `(y_true, y_pred, n_bins, ...)` | `(y_pred, y_true)` | `examples/viz_diagnostic_gallery.py:124` (correct). **`docs/methods/visualization.md:120` passes `(y_pred, y_true)`, the wrong order** |
| `plot_risk_coverage_curve` (`viz/results.py:1002`) | `(y_true, y_pred, rejection_scores, ...)` | `(y_pred, y_true)` | `examples/viz_diagnostic_gallery.py:283`; `docs/methods/visualization.md:207` |
| `pinball_loss` (`metrics/distribution.py:1268`) | `(level, quantile_value, y_true)` | the data arguments first | `PinballMetric.update` and tests only. Neither the harness nor the examples call it |

**Documented exceptions, which need no change:**

- `GaussianWassersteinBoundLoss` and `gaussian_wasserstein_bound_loss` take
  `(pred_mean, target_mean, ...)`. The contract test lists this exception.
- EIV losses take `(x_obs, y_obs)`, as `AGENTS.md` documents.
- `conditional_density_estimation_loss`, `highest_posterior_density_level` and
  `highest_posterior_density_coverage` take `(support, density, y_true)`. The
  prediction is the `(support, density)` pair, so this is still prediction
  first, and it matches `PredictiveBatch`.

**A whole module with its own order.** All 6 functions in `inference/ppi.py`
take `(y_labeled, pred_labeled, pred_unlabeled, ...)`: `ppi_mean_ci`,
`ppi_calibrated_mean_ci`, `ppi_pp_mean_ci`, `ppi_quantile_ci`, `ppi_ols_ci`
and `ppi_diagnostics`. This mirrors the reference `ppi_py` API, which the
harness `ppi` suite calls side by side with the same order.
**Recommendation:** keep this order, and document it as the module convention.

**Sample axis of returned samples.** `NormalizingFlowLoss.sample`
(`losses/nflows.py:517`) returns `[batch, n_samples, D]`. Verified: batch 8 and
`n_samples=5` give shape `(8, 5, 1)`. With `n_samples=1` the sample axis is
dropped and the shape is `(8, 1)`, whereas `MixtureDensityLoss.sample` keeps
it, `(1, 8, 1)`. Everything else uses
`[n_samples, batch, ...]`:

- `MixtureDensityLoss.sample` returns `(5, 8, 1)`, and
  `EvidentialRegressionLoss.sample_predictions` returns `(5, 8, 1)`;
- the ensemble `sample` methods transpose to sample-first, and every
  `mc_forward` is sample-first;
- the EIV `sample_predictions` methods are sample-first;
- the `y_samples` input of every sample-based metric
  (`crps_from_samples`, `energy_score`, `vario_score`, ...) is sample-first.

Flow samples therefore cannot go into the sample metrics without a transpose.
The harness wrappers reshape them (PROPOSALS P1 #7). The harness's draft
`ConditionalFlow` also returns `(N, n_samples, D)`.

---

## 5. `PROPOSALS.md` P0 items

`PROPOSALS.md` is in `torchregress-harness`, not in this repository. The
"IMPLEMENTATION STATUS" table it contains is dated 2026-06-18 and is stale for
item 1.

| # | Item | Status, verified | Recommended decision |
|--:|:--|:--|:--|
| 1 | SLS counter fragility | **Fixed.** Commit `6f2813a` (2026-09-04), "SLS step bookkeeping decoupled and bounded (PROPOSALS.md P0 #1)", changed `losses/sls.py`, added 129 lines to `tests/losses/test_sls.py`, and touched `tests/losses/test_loss_fixes.py`. Audit finding B-LOSS-004 (`docs/reports/audit_0.3.0.md`, batch B, `6c63316`) made `SLSLoss.forward` side-effect free and added `evaluate_frontier`. In the current code, `evaluate_frontier`, `forward_frontier` and `forward_quantiles` accept an explicit `step=`. The step-less fallback counter saturates at `_SLS_STEP_CAP = 1_000_000` (`sls.py:24`, `:642`). No `step_counter -= 1` remains (checked with grep). `tests/losses/test_sls.py`: 16 passed | **Close as fixed in 0.3.0.** Update the status row in the harness `PROPOSALS.md`, which still says "Not implemented at harness level". The proposal's evidence (SLS intervals too wide on Gaussian data and worse on bimodal data) is about performance, not the counter. It belongs with the harness parity results, not with this item. `step_counter` stays a public attribute; keep it for 0.3.0 |
| 2 | Bundled `ConditionalFlow` estimator | **Not in the library.** A 550-line working draft is at `torchregress-harness/proposals/conditional_flow/conditional_flow.py`. `ROADMAP.md` schedules "Bundled `ConditionalFlow` estimator (fit / sample / log_prob / interval)" for **0.5**. The 0.3.0 algorithm freeze ("only fixes, tests, docs and packaging land") rules it out now | **Defer to 0.5; it stops being P0.** Before the port, settle two things from sections 3 and 4. The draft uses `predict_interval(confidence=0.9)` and returns samples batch-first, and both conflict with the recommended `alpha` and sample-first conventions |
| 3 | Scikit-learn `fit()`/`predict()` interface | **Not in the library.** The harness has working wrappers in `src/harness/wrappers.py` (`GaussianNLLRegressor`, `MDNRegressor`, `EvidentialRegressor`, `NormalizingFlowRegressor`, `DeepEnsembleRegressor`, `HeteroscedasticEnsembleRegressor`), with a contract test in `tests/test_api_contracts.py`. `ROADMAP.md` schedules "scikit-learn-style wrappers for the most-used method families" for **0.5** | **Defer to 0.5; it stops being P0.** The harness wrappers are the reference design |

---

## 6. Recommended plan

### 6.1 Before 0.3.0

**Recommendation:** make only these changes. None of them changes behaviour
for any caller in `src`, the tests, the examples or the harness.

1. **Docs only.** The freeze allows these changes.
   - Remove or repair the 25 names in the docs that do not exist (section 1.4).
   - Repair the 11 documented calls with unknown keyword arguments, and the
     wrong signatures in the `docs/api/losses.md` and `docs/api/metrics.md`
     rows (section 1.4). The metrics pages should show `sample_weight`, not
     `mask=` and `weights=`.
   - Correct `docs/methods/visualization.md:120` to
     `plot_target_density_error_overlap(y_true, y_pred)`.
   - Remove the false `SelfAgreementTrainer` method claim.
   - Add `docs/api/` rows for the 5 undocumented exports.
   - State that `highest_posterior_density_coverage(alpha=...)` takes the HPD
     mass, not the miscoverage.
   - State that `NormalizingFlowLoss.sample` is batch-first.
   - Add a short "Naming conventions" note: `target` against `y_true`,
     `weights` against `sample_weight`, `alpha`, and the `ppi` argument order.
   - Have the 0.3.0 CHANGELOG say that the deprecation policy starts in 0.4.
2. **Delete the two unused aliases** `RepresentationShiftCalibrator` and
   `SignificantSubspaceAligner` (`test_time/__init__.py:59-60`). Neither is in
   `__all__` and nothing references either one.
3. **Add `__all__` to `method_catalog.py`** with its 10 own names. This hides
   the 9 leaked imports from `import *`, and every import path keeps working.
4. **Delete the `CORALLoss.__doc__ = ...` assignment** at
   `losses/ordinal.py:214`, so that `CumulativeLinkLoss` keeps its own
   docstring. This changes only the docstring.
5. **Done (2026-10-07, maintainer approved).** Optional item: take the 28 names in section 2.1 out of the
   `torchregress.utils` namespace. They stay importable from their
   submodules. The edits are:
   - `src/torchregress/utils/__init__.py`;
   - imports in `tests/test_utils_ordinal.py` and
     `tests/audit/test_audit_batch5.py`;
   - the pin list in `tests/test_utils_exports.py`;
   - the corresponding rows of `docs/api/utils.md` (`test_UTL_009` checks
     them).

   **Dry run (repeated on 2026-10-07).** The change was made in a full scratch
   copy of the repository (including `.github/`, `scripts/` and `README.md`).
   `utils/__init__.py` was regenerated without the 28 names (53 to 25
   exports), and the imports of `tests/audit/test_audit_batch5.py` and
   `tests/test_utils_ordinal.py` were redirected to the submodules.
   - Library suite, `tests/test_examples_smoke.py` excluded, `-n 2`:
     - unmodified copy: 3377 passed, 724 skipped, 2 xfailed, 0 failed;
     - modified copy: 3375 passed, 724 skipped, 2 xfailed, **2 failed**:
       `tests/test_utils_exports.py::test_utils_exports_coherence_helpers`
       and `tests/audit/test_audit_batch5.py::test_UTL_009_no_phantom_symbols_in_utils_md`.
       These are the two pins the change has to update.
   - Harness, run against the modified copy: `tests/test_export_coverage.py`
     and `tests/test_api_contracts.py` passed (162 tests).
   - Static check of the harness, `examples/` and `docs/`: none of the 28
     names is imported from the `torchregress.utils` namespace, and none is
     reached by `utils.<name>` attribute access. The harness imports
     `update_ema_teacher_` from the namespace (kept) and reaches `CORALHead`,
     `ordinal_predict` and `fetch_openml_dataset` through submodule paths,
     which do not change.
   - The full harness suite was not run: its network-bound tests exceed
     10 minutes.

   **Applied.** The 28 names are out of `utils/__init__.py` (53 to 25
   exports). The functions stay in their submodules. The imports in
   `tests/test_utils_ordinal.py` and `tests/audit/test_audit_batch5.py` were
   redirected to the submodules. `test_utils_exports_coherence_helpers` now pins
   only the kept names, and a new test pins that the plumbing is not
   re-exported. `test_UTL_009_no_phantom_symbols_in_utils_md` also resolves
   names against the submodules. `docs/api/utils.md` lists the 28 under
   "Internal helpers (not re-exported)". A CHANGELOG bullet is under
   [Unreleased] / Changed. The harness allowlist needed no change: its entries
   for `low_rank_output_dim`, `split_low_rank_gaussian_output`,
   `quantile_loss` and `parse_heteroscedastic_output` are keyed to the
   `losses` and `ensemble` exports, which stay.

   **Decision recorded:** the `TeacherStudentTrainer` `tau` default stays
   `0.2` (`semi_supervised.py`); no default change in 0.3.0.

   **Why now:** once 0.3.0 is on PyPI, each of these 28 names needs a full
   deprecation cycle, and the earliest removal is 0.6. **Why not:** the
   algorithm freeze lists "fixes, tests, docs and packaging", and `ROADMAP.md`
   puts the API trim in 0.4. This is the maintainer's call.

**Not before 0.3.0:**

- Any rename.
- Any argument-order change.
- Any default change.
- Adding `CQR`, `CTI`, `MultivariateScoreConformal` and
  `NonExchangeableConformalRegressor` to `losses.__all__`. The harness
  export-coverage test would need `CQR` and `BaseEIVLoss` covered or
  allowlisted; `CTI`, `MultivariateScoreConformal`,
  `NonExchangeableConformalRegressor` and `finite_sample_quantile` are already
  referenced by suites.

### 6.2 In 0.4, under the deprecation policy

**Prerequisites:**

- Add a small helper for renamed keyword arguments and aliases. It should emit
  a `DeprecationWarning` that names the replacement and the removal version.
- Add a CHANGELOG "Deprecated" section.
- `pyproject.toml` sets `filterwarnings = ["ignore::DeprecationWarning", ...]`.
  Add tests asserting that each shim warns. Consider turning
  torchregress-originated `DeprecationWarning`s into errors in CI, so that the
  library never calls its own deprecated names.
- `FunctionalEIVLoss(monte_carlo=...)` is marked "deprecated" only in a comment
  (`losses/eiv.py:1011`) and emits no warning. The harness
  (`suites/tabular/eiv.py:514`), 2 examples and `docs/losses/eiv.md` pass it.
  Give it a real warning in 0.4 and remove it in 0.6.

**Deprecations, in order of value for the effort.** Each is to be deprecated in
0.4 and removed in 0.6.

1. **Coverage level.** Add `alpha` (miscoverage) to the 10 `confidence`
   signatures and deprecate `confidence`, mapping `alpha = 1 - confidence`.
   For `highest_posterior_density_coverage`, give `alpha` a deprecation
   warning saying that it currently means HPD mass and will mean miscoverage
   in 0.6. Add a temporary keyword for the mass, such as `mass=`, so callers
   can migrate. A silent flip would invert results: `examples/metrics_suite_showcase.py`
   passes `alpha=0.90` meaning 90% mass.
2. **Argument order.** Make `calibration_score`, `plot_target_density_error_overlap`
   and `plot_risk_coverage_curve` warn when their first two arguments are
   passed positionally, and switch them to prediction-first in 0.6.
   Reorder `pinball_loss` to `(y_pred, y_true, quantile)` the same way.
3. **Sample axis.** Give `NormalizingFlowLoss.sample` a keyword-only switch
   for the sample axis. Warn when it is left unset, and make sample-first the
   default in 0.6. Coordinate with the harness wrappers.
4. **Parameter spellings from section 3,** each with a warning that names the
   new spelling: `num_bins`, `mc_samples`/`n_mc_samples`, `learning_rate`,
   `max_iterations`/`n_steps`, `tolerance`, `total_epochs`, `dropout_rate`,
   `n_models`, `K`, `epsilon` (`IRLSConfig`), `variance_floor`/`var_clamp`,
   `min_std`/`min_sigma`, `random_state`, `sample_weights`, `weights_cal`,
   `y_pred_mean`/`y_pred_std`/`pred_variance`/`var`, `levels`/`q`,
   `input_size`/`hidden_size`/`output_size`/`feature_dim`/`hidden`, and
   `target` in the 6 metric and calibrator signatures. Leave `n_features`
   and the `PredictiveBatch` fields as documented exceptions, unless the
   maintainer accepts the harness churn.
5. **Exports:**
   - the 9 names in section 2.2, plus `flatten_adaptation_parameters`, which
     needs the harness allowlist entry removed;
   - `ExpectileCrossover` and `QuantileCrossover`, pure abbreviations with no
     use outside tests (update `test_A_LOSS_018`);
   - the `torchregress.metrics.calibration` shim module. Keep the 7 names in
     both `metrics` and `calibration`;
   - decide whether `CQR`, `CTI`, `MultivariateScoreConformal`,
     `NonExchangeableConformalRegressor`, `BaseEIVLoss` and
     `SelfAgreementTrainer` are public. If they are, declare them in
     `__all__` together with a harness allowlist update. If not, make them
     private.
6. **Keep as documented aliases** of established method names: `MDNLoss`,
   `CORALLoss`, `JackknifePlus` and `AsymmetricLeastSquaresLoss`.

### 6.3 For the harness owner

These are observations only. This review changed nothing in the harness.

- Update the status table in `PROPOSALS.md`: #1 is fixed in the library
  (`6f2813a`), and #2 and #3 are deferred to 0.5.
- `HPDCoverage` is computed with the default `alpha=0.1`
  (`src/harness/metrics.py:581`, `suites/tabular/pzflow_parity.py:563`). That
  is the coverage of a 10%-mass HPD region, which is about 0.10 when the
  predictions are calibrated. It is not the coverage of a 90% region. No
  report shows the column today, because grep finds `HPDCoverage` in no `.md`
  file. Pass the mass explicitly.
- Any 0.3.0 or 0.4 change to the `__all__` of the 10 gated modules needs a
  matching `tools/export_coverage_allowlist.py` edit, because the test fails on
  stale entries.
