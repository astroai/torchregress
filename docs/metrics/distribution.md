# Distribution Metrics

Distribution metrics evaluate the quality of probabilistic forecasts by comparing the predicted distributions with observed outcomes.

## Probability Integral Transform (PIT)

PIT evaluates calibration by transforming observations through the predicted CDF. For perfectly calibrated forecasts, PIT values follow a uniform distribution.

```python
from torchregress.metrics.distribution import probability_integral_transform
from torch.distributions import Normal

# Create a simple CDF function for a normal distribution
def cdf_fn(y):
    dist = Normal(loc=model_mean, scale=model_std)
    return dist.cdf(y)

# Get PIT values
pit = probability_integral_transform(cdf_fn, y_true)

# Get PIT values with histogram for assessing uniformity
pit_results = probability_integral_transform(cdf_fn, y_true, 
                                            n_bins=20, return_histogram=True)
print(f"Uniformity chi-squared: {pit_results['uniformity_chi2']}")
```

## Continuous Ranked Probability Score (CRPS)

CRPS measures the integrated squared difference between the predicted CDF and the empirical CDF of the observation. Lower values indicate better performance.

```python
from torchregress.metrics.distribution import continuous_ranked_probability_score

# Create a dictionary of quantile predictions
quantiles = {0.1: q10_pred, 0.5: q50_pred, 0.9: q90_pred}

# Calculate CRPS
crps = continuous_ranked_probability_score(quantiles, y_true)

# Calculate CRPS with custom reduction
crps_per_sample = continuous_ranked_probability_score(quantiles, y_true, reduction="none")
```

## Energy Score

Energy Score is a multivariate generalization of CRPS, suitable for evaluating joint distributions.

```python
from torchregress.metrics.distribution import energy_score

# y_samples has shape [n_samples, batch_size, n_dimensions]
# y_true has shape [batch_size, n_dimensions]
es = energy_score(y_samples, y_true)

# Use a different value of beta (default is 1.0)
es_beta_half = energy_score(y_samples, y_true, beta=0.5)

# For large sample sizes, limit computation
es_limited = energy_score(y_samples, y_true, max_pairs=1000)
```

## Comprehensive Reporting

### Distribution Metrics Report

Generate a comprehensive report of distribution evaluation metrics.

```python
from torchregress.metrics.distribution import distribution_metrics_report
from torch.distributions import Normal

# Using a PyTorch distribution
dist = Normal(mean_pred, std_pred)
report = distribution_metrics_report(dist, y_true)

# Using quantile predictions
quantiles = {0.1: q10_pred, 0.5: q50_pred, 0.9: q90_pred}
report = distribution_metrics_report(None, y_true, y_pred_quantiles=quantiles)

# Using samples from predictive distribution
report = distribution_metrics_report(None, y_true, samples=pred_samples)

print(f"CRPS: {report['crps']}")
```

## Advanced Usage

### Custom Quantile Levels

You can use custom quantile levels for more detailed evaluation:

```python
from torchregress.metrics.distribution import continuous_ranked_probability_score

# More detailed quantiles
detailed_quantiles = {
    0.01: q01_pred, 0.05: q05_pred, 0.1: q10_pred, 
    0.25: q25_pred, 0.5: q50_pred, 
    0.75: q75_pred, 0.9: q90_pred, 0.95: q95_pred, 0.99: q99_pred
}

# Calculate CRPS with detailed quantiles
detailed_crps = continuous_ranked_probability_score(detailed_quantiles, y_true)
```
