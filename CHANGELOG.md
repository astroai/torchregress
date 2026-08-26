# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `BetaNLLLoss` preserves `[B,D]` elementwise before `_reduce` so partial masks `[[True,False]]` no longer discard whole rows (NEW-HIGH-02)
- `PoissonLikelihoodRatioLoss` clamps `exp(y_pred.clamp(max=30))` to avoid `inf` for `y_pred~100` (NEW-HIGH-04)
- `GEV` `gev_nll_elementwise` masks `1/xi` and `pow(-1/xi)` via `use_gev` so `xi→0` does not overflow before `where` (NEW-HIGH-03)
- `Tweedie p=1` Poisson term uses `ratio.clamp(min=eps)` not `log(+eps)` bias (NEW-MED-03)
- `prediction.quantiles_to_density_grid` forces `float32` levels for int quantiles, `n_support>=2` and `isfinite` guards (NEW-LOW-01/05)
- `losses/families.py` adds `unconstrained_inputs` flag (default `True`) to avoid `softplus(softplus(x))` (NEW-MED-01, pattern shown for SkewNormal/SkewT)
- `losses/conformal.py` unifies `_weighted_conformal_threshold` → `_weighted_quantile` augmented `+w_{n+1}=1` (NEW-HIGH-01) and CV+/Jackknife+ docs correct `>=1-2alpha` (NEW-HIGH-05)
- `ensemble/swag.py` skips BatchNorm `running_*` buffers in posterior (NEW-HIGH-06)
- `algorithms/simex.py` validates `sigma_u` PSD via cholesky, warns if `|w|>5` (NEW-HIGH-07)
- `algorithms/tictac.py` clamps `log_k` to `[-6,6]` and warns if Hessian `>50M` elems (~200MB) (NEW-HIGH-08)
- `algorithms/adaptive_prior_vi.py` `VIDSRegressor` KL and NLL both `mean()` so KL not dominated by `P` (NEW-HIGH-09)
- `losses/__init__.py` adds explicit `__all__` (139 symbols) for `audit_api_coverage.py` soundness (TR-API-01 latent)

### Changed
- `BetaNLLLoss` numerics: previously `sum(dim=-1)` then `mean` over `B`; now `mean` over `B*D` elementwise (breaking for `D>1`, semver MINOR, documented)
- `README` badge `python 3.12 | 3.13 | 3.14 | 3.15` now matches `pyproject.toml:14 requires-python <3.16` (NEW-LOW-09)

### Added
- `CHANGELOG.md` (Keep a Changelog) and release-script `prepare_release.sh` now enforces bump

## [0.1.0] - 2026-08-26

- Initial PyPI release candidate (unreleased). Pre-1.0 blockers TR-COR-01…09 (all fixed in `6ce4b9e`).
- Features: 19 loss families, `PredictiveBatch`, `torchmetrics` metrics, conformal (split/CQR/CTI/weighted/Mondrian/CV+), `SemiConformalCalibrator`, ensembles (DeepEnsemble/BatchEnsemble/SWAG/Laplace/BNN/MC-Dropout), algorithms (IRLS/SIMEX/RC/LatentNN/TICTAC/VIDS), causal `dr_*`, PPI, test-time (BLR/OT/COSA/subspace/transport).

[Unreleased]: https://github.com/astroai/torchregress/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/astroai/torchregress/releases/tag/v0.1.0
