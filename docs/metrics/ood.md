# Out-of-Distribution Detection Metrics

Out-of-Distribution (OOD) detection metrics evaluate how well a model can identify inputs that differ significantly from the training distribution.

## Mahalanobis Distance

Measures how many standard deviations a point is from the mean of a distribution, useful for detecting outliers.

```python
from torchregression.metrics.ood import mahalanobis_distance

# x has shape [batch_size, n_features]
# mean has shape [n_features]
# cov has shape [n_features, n_features]
md = mahalanobis_distance(x, mean, cov)

# Calculate average Mahalanobis distance across batch
avg_md = mahalanobis_distance(x, mean, cov, reduction="mean")
```

## Typicality Score

Measures how typical a test sample is under the model's predictive distribution.

```python
from torchregression.metrics.ood import typicality_score

# Using a tuple of (mean, variance)
ts = typicality_score((mean_pred, var_pred), x_test)

# Using a dictionary with distribution parameters
ts = typicality_score({'mean': mean_pred, 'variance': var_pred}, x_test)

# With more Monte Carlo samples
ts = typicality_score((mean_pred, var_pred), x_test, n_samples=1000)
```

## Entropy Score

Calculates entropy of the predictive distribution as a measure of uncertainty.

```python
from torchregression.metrics.ood import entropy_score

# samples has shape [n_samples, batch_size, ...]
es = entropy_score(samples)

# With more bins for histogram estimation
es = entropy_score(samples, n_bins=20)
```

## Kernel Density Score

Measures similarity of test samples to a reference set using kernel density estimation.

```python
from torchregression.metrics.ood import kernel_density_score

# x_test has shape [batch_size, n_features]
# x_reference has shape [n_reference, n_features]
kds = kernel_density_score(x_test, x_reference)

# Adjust bandwidth for RBF kernel
kds = kernel_density_score(x_test, x_reference, bandwidth=0.5)
```

## Comprehensive Reporting

### OOD Metrics Report

Generate a comprehensive report of OOD detection metrics.

```python
from torchregression.metrics.ood import ood_metrics_report

# Basic usage with test data and reference statistics
report = ood_metrics_report(
    model_output=(mean_pred, var_pred),
    x_test=x_test,
    x_reference=x_train,
    mean=train_mean,
    cov=train_cov,
    samples=pred_samples
)

print(f"Mahalanobis distance: {report['mahalanobis_distance']}")
print(f"Typicality score: {report['typicality_score']}")
print(f"Kernel density: {report['kernel_density']}")
print(f"Entropy: {report['entropy']}")
```

## Advanced Usage

### Detecting Distribution Shift

Compare OOD scores between training and test data:

```python
from torchregression.metrics.ood import mahalanobis_distance
import torch

# Calculate scores for training data
train_md = mahalanobis_distance(x_train, mean, cov, reduction="none")

# Calculate scores for test data
test_md = mahalanobis_distance(x_test, mean, cov, reduction="none")

# Analyze the distribution of scores
train_quantiles = torch.quantile(train_md, torch.tensor([0.5, 0.9, 0.95, 0.99]))
threshold = train_quantiles[-1]  # 99th percentile

# Flag potential OOD samples
ood_flags = test_md > threshold
ood_proportion = torch.mean(ood_flags.float())
print(f"Proportion of potential OOD samples: {ood_proportion.item():.2%}")
```
