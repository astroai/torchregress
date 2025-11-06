# torchregress

A comprehensive PyTorch library for regression and uncertainty estimation.

## What is torchregress?

**torchregress** is a PyTorch-based library that provides a comprehensive set of tools for regression tasks, with a special focus on uncertainty estimation. It offers a wide range of loss functions, metrics, and visualization tools to help you build, evaluate, and understand your regression models.

Whether you're a researcher exploring new uncertainty quantification techniques or a practitioner building a production-level regression pipeline, torchregress has the tools you need to get the job done.

## Key Features

- **Diverse Loss Functions**: Move beyond standard Mean Squared Error with a rich collection of loss functions, including robust losses for noisy data, quantile and expectile losses for distributional modeling, and advanced losses for evidential regression and normalizing flows.
- **Uncertainty Quantification**: Easily estimate and work with uncertainty in your predictions. torchregress provides built-in support for various uncertainty estimation techniques, from simple variance prediction to more advanced methods like ensembles and conformal prediction.
- **Comprehensive Metrics**: Evaluate your regression models with a wide range of metrics. Assess not only the accuracy of your point predictions but also the quality of your predictive distributions and the calibration of your uncertainty estimates.
- **Insightful Visualization**: Gain deeper insights into your models with a powerful set of visualization tools. Diagnose model behavior, analyze residuals, visualize prediction intervals, and assess uncertainty calibration with just a few lines of code.
- **Seamless PyTorch Integration**: torchregress is designed to integrate seamlessly into your existing PyTorch workflows. Use it with your favorite PyTorch models and training loops without any friction.

## Getting Started

### Installation

```bash
pip install torchregress
```

### Quickstart

Here's a simple, self-contained example of how to use torchregress to train a model and make predictions:

```python
import torch
import torch.nn as nn
from torchregress.losses import GaussianNLLLoss
from torchregress.metrics import rmse, gaussian_nll

# 1. Create a simple dataset
X = torch.linspace(0, 1, 100).unsqueeze(1)
y = 2 * X + torch.randn_like(X) * 0.1

# 2. Define a simple model with two outputs (mean and variance)
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

# 3. Choose a loss function
loss_fn = GaussianNLLLoss()

# 4. Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(100):
    optimizer.zero_grad()
    output = model(X)
    mean, var = output.chunk(2, dim=-1)
    loss = loss_fn(mean, y, var)
    loss.backward()
    optimizer.step()

# 5. Make predictions and evaluate
with torch.no_grad():
    output = model(X)
    mean, var = output.chunk(2, dim=-1)
    
    print(f"RMSE: {rmse(mean, y):.4f}")
    print(f"NLL: {gaussian_nll(mean, y, var):.4f}")
```

## Next Steps

### For Beginners
- **[Core Concepts Guide](guides/concepts.md)** 🆕 - Start here! Learn key concepts: uncertainty types, robustness, ensembles, and more
- **[Quick Start](usage/quickstart.md)** - Three runnable examples to get started quickly
- **[Basic Examples](examples/basic_usage.md)** - Four detailed tutorials covering common use cases

### For Practitioners
- **[Practical Usage Guide](usage/practical_usage.md)** - Decision trees for choosing losses and methods
- **[Best Practices](guides/best-practices.md)** - 7-phase development workflow and common pitfalls
- **[Comprehensive Comparison](examples/comprehensive_comparison.py)** 🆕 - All-in-one comparison of robustness, uncertainty, and ensembles

### Deep Dives
- **[Ensemble Methods](examples/ensemble_methods.md)** 🆕 - Complete guide to uncertainty quantification with ensembles
- **[Browse All Examples](examples/index.md)** - Real-world examples including astronomy, computer vision, and more
- **[API Reference](api/index.md)** - Detailed documentation of all available functions

## Citation

If you use torchregress in your research, please cite:

```bibtex
@software{torchregress,
  title = {{torchregress: A PyTorch Library for Regression and Uncertainty Estimation}},
  author = {Fabbro, Sébastien},
  url = {https://github.com/sfabbro/torchregress},
  version = {0.1.0},
  year = {2024},
}
```

## License

torchregress is released under the MIT License.