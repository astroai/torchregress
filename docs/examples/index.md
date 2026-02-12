# Examples

This section contains practical examples of using torchregress for various regression tasks. The examples are designed to demonstrate key features and use cases of the library.

## Getting Started

**New to torchregress?** Start with the [Concepts Guide](../guides/concepts.md) to learn key concepts.

## Basic Examples

### [Basic Usage](basic_usage.md)

This tutorial covers the fundamental usage patterns of torchregress:

- Setting up simple regression models
- Choosing and applying loss functions
- Evaluating models with appropriate metrics
- Basic visualization of results

```python
import torch
import torchregress as tr

# Example of basic torchregress usage
X_train, y_train = load_data()
model = MyRegressionModel()
loss_fn = tr.losses.HuberLoss()

# Train model
# ...

# Evaluate
predictions = model(X_test)
rmse = tr.metrics.rmse(predictions, y_test)
```

### [Loss Comparison](loss_comparison.md)

This example compares different loss functions on the same dataset:

- Comparing traditional vs. robust losses
- Analyzing how different losses handle outliers
- Visualizing the effect of loss choice on predictions
- Determining which loss is best for different data characteristics

```python
import torchregress as tr

# Dictionary of loss functions to compare
losses = {
    "MSE": tr.losses.MSELoss(),
    "MAE": tr.losses.L1Loss(),
    "Huber": tr.losses.HuberLoss(),
    "LogCosh": tr.losses.LogCoshLoss(),
}

# Compare losses and visualize results
# ...
```

### [Comprehensive Comparison](https://github.com/sfabbro/torchregress/blob/main/examples/comprehensive_comparison.py) 🆕

**All-in-one comparison** demonstrating the three main capabilities of torchregress:

- **Robust Regression** - Handling outliers (MSE vs. Huber vs. Cauchy)
- **Uncertainty Estimation** - Quantifying confidence (Gaussian NLL, ensembles)
- **Ensemble Methods** - Combining models (Deep Ensemble, Heteroscedastic Ensemble)

Three challenging scenarios:
1. Clean data (baseline comparison)
2. Data with outliers (robust losses)
3. Heteroscedastic data (uncertainty decomposition)

```python
# Compare robust losses on outlier data
losses = {
    "MSE": WeightedMSELoss(),      # Sensitive to outliers
    "Huber": HuberLoss(delta=1.0),  # Balanced
    "Cauchy": CauchyLoss(scale=0.5) # Very robust
}
```

## Advanced Examples

### [Ensemble Methods](ensemble_methods.md) 🆕

**Complete guide to uncertainty quantification with ensembles:**

- **Deep Ensemble** - Epistemic uncertainty from model disagreement
- **Heteroscedastic Ensemble** - Both epistemic and aleatoric uncertainty
- **Batch Ensemble** - Efficient alternative for limited compute
- **Uncertainty Decomposition** - Separating model vs. data uncertainty

Includes decision trees, comparison tables, and complete working example ([`ensemble_tutorial.py`](https://github.com/sfabbro/torchregress/blob/main/examples/ensemble_tutorial.py)).

```python
# Train heteroscedastic ensemble with uncertainty decomposition
ensemble_models = train_heteroscedastic_ensemble(n_models=5, ...)
epistemic, aleatoric = ensemble_variance_decomposition(means, log_vars)
```

### [Photometric Redshift Estimation](photoz.md)

A real-world application for astronomy:

- Implementing uncertainty-aware regression for photometric redshift estimation
- Using specialized loss functions for astronomical data
- Creating calibrated prediction intervals
- Evaluating results with domain-specific metrics

### [Conformal Regression](conformal_regression_example.md)

- Using conformal prediction to obtain prediction intervals with guaranteed coverage.

### [Evidential Regression](evidential_regression.md)

- Decomposing uncertainty into aleatoric and epistemic components.

### [Imbalanced Regression](imbalanced_regression.md)

- Handling imbalanced datasets in regression tasks.

### [Noisy Labels Regression](noisy_labels_regression.md)

- Training models on data with noisy labels.

### [Normalizing Flows](normalizing_flows_multitarget.md)

- Using normalizing flows for multi-target regression.

## Running the Examples

All examples can be run directly from the repository:

```bash
# Clone the repository
git clone https://github.com/username/torchregress.git
cd torchregress

# Install dependencies
pip install -e '.[examples]'

# Run a specific example
python examples/basic_usage.py
```

For questions or issues with the examples, please [open an issue](https://github.com/username/torchregress/issues) in the repository.
