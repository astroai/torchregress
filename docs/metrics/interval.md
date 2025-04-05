# Interval Metrics

Interval metrics evaluate the quality of prediction intervals, focusing on coverage and width properties.

## Interval Score (Winkler Score)

Evaluates prediction intervals by rewarding narrow intervals and penalizing when observations fall outside the interval.

```python
from torchregress.metrics.interval import interval_score

# lower_bound and upper_bound typically represent a 90% prediction interval
score = interval_score(lower_bound, upper_bound, y_true, alpha=0.1)

# Get detailed metrics
detailed_score = interval_score(lower_bound, upper_bound, y_true, 
                                alpha=0.1, reduction="full")
print(f"Mean width: {detailed_score['mean_width']}")
print(f"Coverage: {detailed_score['mean_coverage']}")
print(f"Expected coverage: {detailed_score['expected_coverage']}")
```

## Prediction Interval Coverage Probability (PICP)

Measures the proportion of observations that fall within the prediction interval.

```python
from torchregress.metrics.interval import prediction_interval_coverage_probability

# Calculate basic PICP
picp = prediction_interval_coverage_probability(lower_bound, upper_bound, y_true)

# With detailed diagnostics
picp_detailed = prediction_interval_coverage_probability(
    lower_bound, upper_bound, y_true,
    expected_coverage=0.9, return_diagnostics=True
)

print(f"PICP: {picp_detailed['picp']}")
print(f"Coverage error: {picp_detailed['coverage_error']}")
print(f"Mean Prediction Interval Width: {picp_detailed['mpiw']}")
print(f"Normalized MPIW: {picp_detailed['nmpiw']}")
```

## Mean Prediction Interval Width (MPIW)

Measures the average width of prediction intervals. This is included in the detailed output of `prediction_interval_coverage_probability`.

```python
from torchregress.metrics.interval import prediction_interval_coverage_probability

# Get MPIW from detailed metrics
results = prediction_interval_coverage_probability(
    lower_bound, upper_bound, y_true,
    return_diagnostics=True
)
mpiw = results['mpiw']
```

## Comprehensive Reporting

### Interval Metrics Report

Compare interval quality across multiple models.

```python
from torchregress.metrics.interval import interval_metrics_report

# Create a dictionary of model predictions
predictions = {
    'model1': {'lower': model1_lower, 'upper': model1_upper},
    'model2': {'lower': model2_lower, 'upper': model2_upper},
    'model3': {'lower': model3_lower, 'upper': model3_upper}
}

# Generate comprehensive report
report = interval_metrics_report(predictions, y_true, alpha=0.1)

# Access results for specific models
model1_coverage = report['model1']['mean_coverage']
model2_interval_score = report['model2']['score']
model3_interval_width = report['model3']['mean_width']
```

## Advanced Usage

### Asymmetric Interval Evaluation

The interval score can reveal asymmetries in the prediction intervals:

```python
from torchregress.metrics.interval import interval_score

detailed_score = interval_score(lower_bound, upper_bound, y_true, 
                               alpha=0.1, reduction="full")

# Check for asymmetric errors
below_penalty = detailed_score['penalty_below']
above_penalty = detailed_score['penalty_above']
asymmetry_ratio = below_penalty / above_penalty if above_penalty > 0 else float('inf')
print(f"Asymmetry ratio: {asymmetry_ratio}")
```
