# WALDO: Wald Confidence Regions

> ← [Calibration](../methods/calibration.md) | [Conformal Prediction](../losses/conformal.md) →

WALDO (Masserano et al., AISTATS 2023) constructs confidence regions for a
parameter from a conditional mean and covariance estimator.  It replaces the
classical Wald statistic's MLE and variance with predictive moments:

$$
\tau(y_0 \mid x)
=(\hat\mu(x)-y_0)^T\hat\Sigma(x)^{-1}(\hat\mu(x)-y_0).
$$

The confidence set for the observed $x$ is the Neyman inversion

$$
\mathcal R(x)=\{y_0:\tau(y_0\mid x)\le C(y_0)\},
$$

where $C(y_0)$ is a critical value estimated by quantile regression in the
original WALDO simulator setting.  A constant $C$ is the split-conformal
special case.

## `WaldoConformal`

```python
from torchregress.losses import WaldoConformal

cp = WaldoConformal(alpha=0.1, critical_bins=10)
cp.calibrate(mean_cal, y_cal, variance_cal)       # diagonal variances
lo, hi = cp.predict_interval(mean_test, std_test, y_min=-3, y_max=3)
inside = cp.contains(mean_test, y_test, variance_test)
```

- **Scalar targets**: intervals from Neyman inversion on a grid.
- **Multi-target targets**: use `contains` or `predict_region` for the joint
  Wald ellipsoid; the class deliberately does not manufacture per-axis intervals
  that would promise joint coverage they do not have.
- **Critical values**: `critical_bins` estimates a target-binned quantile of
  $\tau$; pass `critical_value_fn` for an externally fitted quantile regression.
  A constant threshold (no bins) is the safest split-conformal fallback.
- **Covariance type**: `"diagonal"` or `"full"`.

## Posterior/sample mode

`WaldoDensityConformal` is the simulator-free conditional-density wrapper: pass
a sampler `sample_fn(context, n)`, calibrate on held-out targets, and use
`contains`/`predict_region`/`predict_interval`.  It estimates conditional
moments from the density itself, so no simulator or parameter prior is required.

`WaldoPosteriorCalibrator` estimates the conditional mean/covariance from
predictive samples (e.g. a flow):

```python
from torchregress.losses import WaldoPosteriorCalibrator

cp = WaldoPosteriorCalibrator(alpha=0.1, covariance_type="full")
cp.calibrate(samples_cal, y_cal)          # samples_cal: [N, S, D]
inside = cp.contains(samples_test, y_test)
region = cp.predict_region(samples_test, candidate_grid)
```

This is the WALDO "posterior estimator" mode.  It is a **calibration** layer: a
bad posterior can still produce a valid but wide region; validate empirical
coverage by parameter region.

!!! warning "Regression adaptation, not the full SBI pipeline"
    The original WALDO uses simulator draws at each parameter and estimates
    conditional critical values with quantile regression.  This implementation
    uses observed calibration targets and a binned critical-value approximation
    unless you supply `critical_value_fn`; the finite-sample guarantee of the
    constant-threshold version is split-conformal, while the binned version
    depends on the quality of the critical-value regression.

## Critical-value regression without a simulator

The original WALDO critical-value function is fitted by quantile regression on
simulator draws.  When only observed calibration data are available,
`WaldoCriticalValueRegressor` fits a local quantile curve instead:

1. reduce the calibration features (usually the candidate target) to a scalar
   key;
2. split the key into quantile bins;
3. estimate the $(1-\alpha)$ quantile of the Wald statistic in each bin;
4. interpolate between bin centers, falling back to the global split-conformal
   quantile for sparse bins.

```python
from torchregress.losses import WaldoConformal, WaldoCriticalValueRegressor

reg = WaldoCriticalValueRegressor(alpha=0.1, n_bins=8, min_samples_per_bin=10)
cp = WaldoConformal(alpha=0.1, critical_value_regressor=reg)
cp.calibrate(mean_cal, y_cal, variance_cal)
inside = cp.contains(mean_test, y_test, variance_test)
```

This is a **simulator-free local approximation**.  It can tighten regions where
the Wald statistic is heteroscedastic, but it does not recover WALDO's original
simulator-based conditional guarantee; validate coverage by target region.

## References

| # | Reference |
|:-:|:----------|
| 1 | L. Masserano et al. ["Simulator-Based Inference with Waldo."](https://arxiv.org/abs/2205.15680) AISTATS 2023. |
| 2 | V. Vovk et al. *Algorithmic Learning in a Random World.* Springer 2005 (split conformal). |
