# Calibration Metrics

> ← [Distributional Metrics](distribution.md) | [OOD Metrics](ood.md) →

Calibration metrics evaluate whether predicted probabilistic distributions or intervals accurately reflect the true uncertainty in the data.

Calibration is the *honesty* half of forecast quality. For a predictive CDF $F_i$, the probability integral transform $u_i = F_i(y_i)$ should be uniform on $[0,1]$ when the forecast is correct; equivalently, the empirical frequency of $y_i \le F_i^{-1}(q)$ should equal $q$ at every level $q$. Calibration metrics measure the distance from that ideal, separately from sharpness (which the [distribution metrics](distribution.md) reward). They are necessary but not sufficient: the constant forecast $F = $ training marginal is often well calibrated and useless, which is why calibration is always reported with a score such as CRPS.

→ See [Calibration methods](../methods/calibration.md), [Visualization diagnostics](../methods/visualization.md#1-diagnostic-plots-torchregressvizdiagnostic), and the [calibration API](../api/metrics.md).

---

## Expected Calibration Error (ECE)

ECE measures the discrepancy between predicted quantile levels and their empirical observed coverage:

$$
\text{ECE} = \frac{1}{Q} \sum_{q \in \mathcal{Q}} |q - \hat{p}(q)|
$$

where $\mathcal{Q} = \{q_1, \dots, q_Q\}$ is the set of predicted quantile levels (e.g. $[0.1, 0.5, 0.9]$), and $\hat{p}(q)$ is the observed proportion of targets that fall below the predicted $q$-quantile $\hat{y}_{i, q}$:

$$
\hat{p}(q) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i \le \hat{y}_{i, q})
$$

The Maximum Calibration Error (MCE) is the maximum absolute discrepancy across quantiles:

$$
\text{MCE} = \max_{q \in \mathcal{Q}} |q - \hat{p}(q)|
$$

```python
from torchregress.metrics import expected_calibration_error

# Using a dictionary of quantile predictions
quantiles = {0.1: q10_pred, 0.5: q50_pred, 0.9: q90_pred}

# Calculate basic ECE metrics
ece_metrics = expected_calibration_error(quantiles, y_true)
print(f"Mean Absolute Calibration Error: {ece_metrics['mean_absolute_calibration_error']}")
print(f"Maximum Calibration Error: {ece_metrics['maximum_calibration_error']}")
```
See also: [expected_calibration_error](../api/metrics.md) and the stateful [`ExpectedCalibrationError`](../api/metrics.md) metric class.

---

## Marginal Calibration Error (MCE)

Marginal Calibration Error evaluates how well the predictive distribution's marginal ECDF matches the empirical ECDF of true observations.

For $B$ bins spanning the range of true and predicted targets with edges $b_1, \dots, b_{B+1}$, the observed target CDF at bin edge $k$ is:

$$
F_{\text{obs}}(b_{k+1}) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i < b_{k+1})
$$

Given $S$ Monte Carlo predictions per test point $\hat{y}_i^{(s)}$, the mean predicted CDF is:

$$
\bar{F}_{\text{pred}}(b_{k+1}) = \frac{1}{S \cdot N} \sum_{s=1}^S \sum_{i=1}^N \mathbb{I}(\hat{y}_i^{(s)} < b_{k+1})
$$

The Marginal Calibration Error is the average absolute CDF difference over bins:

$$
\text{MCE}_{\text{marginal}} = \frac{1}{B} \sum_{k=1}^B |F_{\text{obs}}(b_{k+1}) - \bar{F}_{\text{pred}}(b_{k+1})|
$$

The Maximum MCE is:

$$
\text{MaxMCE}_{\text{marginal}} = \max_{k} |F_{\text{obs}}(b_{k+1}) - \bar{F}_{\text{pred}}(b_{k+1})|
$$

```python
from torchregress.metrics import marginal_calibration_error

# y_pred_samples has shape [n_samples, batch_size]
mce_metrics = marginal_calibration_error(y_pred_samples, y_true)
print(f"Marginal Calibration Error: {mce_metrics['marginal_calibration_error']}")
print(f"Maximum MCE: {mce_metrics['maximum_marginal_calibration_error']}")
```
See also: [marginal_calibration_error](../api/metrics.md) and the stateful [`MarginalCalibrationError`](../api/metrics.md) metric class.

---

## Bias

Measures the mean signed prediction error (mean bias):

$$
\text{Bias}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)
$$

```python
from torchregress.metrics import bias

mean_bias = bias(y_pred, y_true)
```
See also: [bias](../api/metrics.md).

---

## Calibration Score

A convenience calibration score for Gaussian predictions. It builds a fine grid of 19 quantile levels from `pred_mean` and `pred_std` and computes the quantile ECE on that grid:

$$
\hat{y}_{i, q} = \hat{\mu}_i + z_q \cdot \hat{\sigma}_i
$$

where $z_q = \Phi^{-1}(q)$ is the inverse CDF of the standard normal distribution.

```python
from torchregress.metrics import calibration_score

# Get ECE computed over a fine normal quantile grid
score = calibration_score(y_true, pred_mean, pred_std)
```
See also: [calibration_score](../api/metrics.md).

---

## Comprehensive Reporting

### Calibration Metrics Report

Generate a comprehensive report of calibration metrics.

```python
from torchregress.metrics import calibration_metrics_report
from torch.distributions import Normal

# Using a PyTorch distribution
dist = Normal(mean_pred, std_pred)
report = calibration_metrics_report(dist, y_true)
```
See also: [calibration_metrics_report](../api/metrics.md).

---

## Limitations

1. **ECE bin sensitivity**: Quantile-based ECE depends on the set of quantile levels $\mathcal{Q}$. A model that is well-calibrated at $\{0.1, 0.5, 0.9\}$ may be poorly calibrated at $\{0.05, 0.25, 0.75, 0.95\}$. Use a fine grid (19+ levels) for thorough evaluation.
2. **Marginal vs conditional**: Both ECE and MCE measure marginal (average) calibration. They do not detect systematic miscalibration that varies with $X$ — a model can have zero ECE while being overconfident in some regions and underconfident in others.
3. **Bias conceals variance**: The `bias` metric (mean signed error) can be near zero while the model has large but symmetric errors. Always pair bias with RMSE or MAE.
4. **Calibration score assumes Gaussian**: `calibration_score` constructs quantiles assuming $\mathcal{N}(\mu, \sigma^2)$. For non-Gaussian predictive distributions (MDN, flows), use `expected_calibration_error` with model-generated quantiles instead.

## Recommendations

- **Default workflow**: Report ECE (quantile-based), bias, and a PIT histogram (`plot_pit_histogram` from `torchregress.viz`). These three together diagnose most calibration issues.
- **For fine-grained calibration**: Use `calibration_score` with 19 quantile levels to detect miscalibration at specific probability regions.
- **Post-hoc correction**: If calibration is poor, apply [Variance Temperature Scaling](../methods/calibration.md) or [Isotonic Calibration](../methods/calibration.md) before deployment.
- **Layer with distributional metrics**: Calibration metrics tell you if predicted probabilities are reliable. Distributional metrics (CRPS, NLL) tell you if the predictive distribution is sharp. Both are needed. See [Distributional metrics](distribution.md).
- **[calibration_metrics_report](../api/metrics.md)** consolidates all calibration metrics into one call.

## Next steps

- [Distributional metrics](distribution.md) — CRPS and PIT diagnostics that evaluate the full predictive distribution
- [Interval metrics](interval.md) — PICP and Winkler score for interval-focused evaluation
- [Post-hoc calibration](../methods/calibration.md) — fix miscalibration with temperature scaling or isotonic regression
- [Visualization diagnostics](../methods/visualization.md) — reliability diagrams and PIT histograms

---

## Simulation-based calibration (SBC) and TARP

The calibration metrics above compare a **predictive distribution** with observed
targets. In simulation-based inference (SBI) — approximate Bayesian computation,
neural posterior estimation, validating an MCMC or variational inference
implementation, or inverting a forward model — there is a prior question: does the
inference algorithm return the posterior that actually generated the data? When the
data are simulated from a known prior and likelihood, the ground-truth parameters
are known by construction, and posterior calibration can be tested directly. SBC
and TARP are the standard diagnostics for that setting.

Both diagnostics take posterior draws of shape `[N, S, D]`: `N` simulated
observations, `S` posterior draws per observation, `D` parameter dimensions. They
do not test whether the assumed prior/likelihood model matches reality — that is a
modelling question. They test whether the inference algorithm is calibrated for the
model you gave it.

### SBC ranks

For every simulated observation $n$ and parameter dimension $d$, SBC (Cook et al.
2006; Talts et al. 2018) computes the rank of the true value among the posterior
draws, crediting exact ties with half a draw:

$$
r_{nd} = \sum_{s=1}^{S} \mathbb{1}\!\left[\theta^{(s)}_{nd} < \theta^{\star}_{nd}\right] + \tfrac{1}{2} \sum_{s=1}^{S} \mathbb{1}\!\left[\theta^{(s)}_{nd} = \theta^{\star}_{nd}\right]
$$

For a calibrated posterior the true value is exchangeable with the posterior draws,
so the rank is discrete-uniform on the $S+1$ possible values:

$$
\Pr(r_{nd} = k) = \frac{1}{S+1}, \qquad k = 0, 1, \dots, S
$$

Deviations are diagnosed from the rank histogram: a U-shape means the posterior is
too narrow (over-confident), a central peak means it is too wide (under-confident),
and a monotone skew indicates a bias. Uniformity is quantified per dimension with a
chi-square test on $B$ integer bins over $[0, S+1]$:

$$
X^2_d = \sum_{b=1}^{B} \frac{(O_{db} - E_b)^2}{E_b}, \qquad E_b = N \, \frac{w_b}{S+1}
$$

where $O_{db}$ are the observed counts, $w_b$ is the number of integer rank values
in bin $b$ ($w_b = 1$ for the default $S+1$ one-bin-per-rank histogram), and
$X^2_d \sim \chi^2_{B-1}$ under the null hypothesis.

!!! note "Reading an SBC rejection"
    A small p-value rejects the combination "prior + likelihood + inference
    algorithm" — it does not say which of the three is wrong. SBC checks one
    marginal rank per parameter dimension, so it is a necessary but not sufficient
    condition: a posterior with correct marginals can still have wrong
    correlations. All simulated data must go through the exact same pipeline as the
    real data (including any data-dependent preprocessing).

### TARP coverage

The TARP test (Lemos et al. 2023) builds *random credible regions* from a reference
point $\tilde{\theta}_n$ and the posterior draws,

$$
d_{ns} = \lVert \theta^{(s)}_n - \tilde{\theta}_n \rVert_2, \qquad d^{\star}_n = \lVert \theta^{\star}_n - \tilde{\theta}_n \rVert_2
$$

and defines the region as the ball of radius equal to the $\alpha$-quantile
$q_n(\alpha)$ of the distances. The empirical coverage curve is

$$
\hat{c}(\alpha) = \frac{1}{N} \sum_{n=1}^{N} \mathbb{1}\!\left[d^{\star}_n \le q_n(\alpha)\right], \qquad \mathrm{TARP\ error} = \max_{\alpha} \left|\hat{c}(\alpha) - \alpha\right|
$$

For a calibrated posterior, and for reference points that are independent of the
true parameters and of the posterior draws,
$\mathbb{E}[\hat{c}(\alpha)] = \alpha$ at every nominal level. Unlike SBC, TARP
tests the **joint** posterior — dispersion and correlation structure included — so
it is sensitive to failures a per-parameter rank test cannot see. Systematic
deviations from the diagonal identify the failure mode: over-confident posteriors
typically under-cover at high nominal levels (while over-covering at low levels),
under-confident posteriors show the opposite pattern, and biased posteriors produce
asymmetric curves. The last two lines of the example below contrast a calibrated
and an over-confident posterior.

### Runnable example

```python
import math

import torch

from torchregress.metrics.sbi import (
    sbc_rank_histogram,
    sbc_uniformity_pvalue,
    simulation_based_calibration_ranks,
    tarp_coverage,
)

torch.manual_seed(0)
n_obs, n_samples, n_dims = 2000, 100, 2

# Simulate from an exact Gaussian posterior: prior theta ~ N(0, 1),
# likelihood y | theta ~ N(theta, 1), so the posterior is N(y / 2, 1 / 2).
truth = torch.randn(n_obs, n_dims)
y = truth + torch.randn(n_obs, n_dims)
posterior_mean = 0.5 * y
posterior_scale = math.sqrt(0.5)
calibrated = posterior_mean.unsqueeze(1) + posterior_scale * torch.randn(
    n_obs, n_samples, n_dims
)

ranks = simulation_based_calibration_ranks(calibrated, truth)
histogram = sbc_rank_histogram(ranks)
uniformity = sbc_uniformity_pvalue(ranks)
print("calibrated min p-value:", uniformity["min_p_value"])
print("calibrated edge-bin counts:", histogram["counts"][:, [0, -1]].tolist())

# TARP reference points: drawn independently from the prior.
reference_points = torch.randn(n_obs, n_dims)
calibrated_tarp = tarp_coverage(calibrated, truth, reference_points)
print("calibrated TARP error:", calibrated_tarp["tarp_error"])

# Over-confident posterior: same posterior mean, half the standard deviation.
overconfident = posterior_mean.unsqueeze(1) + 0.5 * posterior_scale * torch.randn(
    n_obs, n_samples, n_dims
)
overconfident_ranks = simulation_based_calibration_ranks(overconfident, truth)
overconfident_uniformity = sbc_uniformity_pvalue(overconfident_ranks)
overconfident_tarp = tarp_coverage(overconfident, truth, reference_points)
print("over-confident min p-value:", overconfident_uniformity["min_p_value"])
print("over-confident TARP error:", overconfident_tarp["tarp_error"])
```

With this fixed seed the calibrated posterior gives a minimum p-value around 0.36
and a TARP error of about 0.02, while the over-confident posterior gives a p-value
that underflows to zero and a TARP error of about 0.24.

### Caveats

!!! warning "Statistical diagnostics, not calibration certificates"
    - **Finite-sample stochasticity.** Under the null, SBC p-values are uniform on
      $[0, 1]$, so small values occur by chance; the calibration is only as good as
      the number of simulations $N$ and draws $S$. The TARP error is a maximum over
      the $\alpha$ grid, so it is positive even for a calibrated posterior (order
      $10^{-2}$ for $N \sim 10^3$) and is **not** a calibrated p-value. Compare
      estimators on the same simulated data and with the same grid.
    - **Multiple testing.** Testing $D$ parameter dimensions at level $\alpha$ gives
      a family-wise false-positive rate of roughly $D\alpha$. Use the returned
      `bonferroni_min_p_value` for a conservative correction, or treat the p-values
      as descriptive.
    - **Reference points for TARP.** They must be drawn independently from the
      prior, in the same parameter space and units as the posterior, and must never
      be the true generating parameters (that would trivially cover the truth). The
      default Euclidean metric depends on parameter scales: standardise or whiten
      parameters whose units differ before running TARP, otherwise the largest-scale
      dimension dominates the regions.
    - **What each test actually covers.** SBC tests the marginal rank of each
      dimension separately; TARP tests the joint coverage of random balls. TARP is
      not a mode-coverage or point-accuracy test: a multimodal posterior can have a
      correct coverage curve (the ball may swallow the true mode), and both tests
      can pass for a posterior with the right shape but the wrong mode. Neither
      replaces predictive calibration (ECE, PIT) or proper scores (CRPS, NLL)
      evaluated on real observations.
    - **Ties and sampler assumptions.** The tie-aware rank follows the randomised
      rank in expectation but is not exactly discrete-uniform for heavily discrete
      parameters, and the chi-square null assumes near-continuous ranks. The SBC
      derivation also assumes independent posterior draws: thin autocorrelated MCMC
      chains before computing ranks.
    - **Simulate what you analyse.** Every data-dependent step of the real pipeline
      (feature standardisation, cuts, selection functions, noise realisations) must
      be part of the simulation, otherwise the test validates a different problem.

### SBC and TARP references

| # | Reference |
|:-:|:----------|
| 1 | S. R. Cook, A. Gelman, D. B. Rubin. ["Validation of Software for Bayesian Models Using Posterior Quantiles."](https://doi.org/10.1198/106186006X136976) *Journal of Computational and Graphical Statistics*, 15(3):675–692, **2006**. |
| 2 | S. Talts, M. Betancourt, D. Simpson, A. Vehtari, A. Gelman. ["Validating Bayesian Inference Algorithms with Simulation-Based Calibration."](https://arxiv.org/abs/1804.06788) arXiv:1804.06788, **2018**. |
| 3 | P. Lemos, A. Coogan, Y. Hezaveh, L. Perreault-Levasseur. ["Sampling-Based Accuracy Testing of Posterior Estimators for General Inference."](https://arxiv.org/abs/2302.03026) *ICML*, **2023**. |
| 4 | J. Geweke. ["Getting It Right: Joint Distribution Tests of Posterior Simulators."](https://doi.org/10.1198/016214504000001132) *Journal of the American Statistical Association*, 99(467):799–804, **2004**. |

→ See the [metrics catalogue](index.md), [distribution metrics](distribution.md) for
PIT-based predictive calibration, and the [metrics API](../api/metrics.md) for the
function signatures.

---

## References

| # | Reference |
|:-:|:----------|
| 1 | M.P. Naeini, G.F. Cooper, M. Hauskrecht. ["Obtaining Well Calibrated Probabilities Using Bayesian Binning into Quantiles."](https://ojs.aaai.org/index.php/AAAI/article/view/9602) *AAAI*, **2015**. |
| 2 | V. Kuleshov, N. Fenner, S. Ermon. ["Accurate Uncertainties for Deep Learning Using Calibrated Regression."](https://arxiv.org/abs/1807.00263) *ICML*, **2018**. |
| 3 | A.P. Dawid. ["Statistical Theory: The Prequential Approach."](https://www.jstor.org/stable/2345714) *JRSS-A*, 147(2):278–292, **1984**. |
