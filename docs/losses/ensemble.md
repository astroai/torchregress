# Ensemble Methods for Regression

Ensemble methods combine multiple models to achieve better performance and uncertainty estimation than any single model. TorchRegression provides ensemble model implementations that enable better uncertainty quantification and robust predictions.

## Mathematical Background

Ensemble methods leverage the idea that a collection of diverse models can outperform a single model. For regression tasks, an ensemble prediction is typically:

$$\hat{y}_{\text{ensemble}} = \frac{1}{M}\sum_{i=1}^M \hat{y}_i$$

The uncertainty can be estimated from the ensemble variance:

$$\sigma^2_{\text{ensemble}} = \frac{1}{M}\sum_{i=1}^M (\hat{y}_i - \hat{y}_{\text{ensemble}})^2$$

## Available Ensemble Models

### BaseEnsembleModel

```python
class BaseEnsembleModel(nn.Module)
```

Base class for ensemble models that provides common functionality.

**Parameters:**

- `base_model` (nn.Module or type): Base model class or instance to ensemble
- `ensemble_size` (int, optional): Number of ensemble members. Default: `5`
- `device` (str, optional): Device to use. Default: 'cpu'
- `**base_model_kwargs`: Additional arguments passed to the base model constructor

**Methods:**

- `forward(x)`: Computes predictions from all ensemble members
- `predict(x)`: Makes prediction with uncertainty estimates
- `predict_with_uncertainties(x)`: Makes prediction with epistemic/aleatoric uncertainties

**Example:**

```python
import torch
import torchregress as tr

# Create a base model class
class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# Create ensemble using the base model
ensemble = tr.ensemble.BaseEnsembleModel(
    base_model=SimpleModel,
    ensemble_size=5,
    input_dim=10,
    hidden_dim=32,
    output_dim=1
)

# Make predictions with uncertainty
x = torch.randn(10, 10)
predictions = ensemble.predict(x)

# Access mean prediction and variance
mean_pred = predictions['mean']
variance = predictions['variance']
```

### DeepEnsemble

```python
class DeepEnsemble(BaseEnsembleModel)
```

Implementation of deep ensembles for uncertainty estimation as described in "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" by Lakshminarayanan et al.

**Parameters:**

- `base_model` (nn.Module or type): Base model class or instance to ensemble
- `ensemble_size` (int, optional): Number of ensemble members. Default: `5`
- `device` (str, optional): Device to use. Default: 'cpu'
- `**base_model_kwargs`: Additional arguments passed to the base model constructor

**Methods:**

- `fit(train_loader, criterion, optimizer_class, epochs, lr, verbose, val_loader, **optimizer_kwargs)`: Trains all ensemble members independently
- All methods inherited from BaseEnsembleModel

**Example:**

```python
import torch
import torch.nn as nn
import torchregress as tr

# Create base model class
class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# Create deep ensemble
ensemble = tr.ensemble.DeepEnsemble(
    base_model=SimpleModel,
    ensemble_size=5,
    input_dim=10,
    output_dim=1
)

# Create data loaders
train_loader = torch.utils.data.DataLoader(dataset)

# Train the ensemble
history = ensemble.fit(
    train_loader=train_loader,
    criterion=nn.MSELoss(),
    optimizer_class=torch.optim.Adam,
    epochs=10,
    lr=0.001
)
```

### HeteroscedasticEnsembleModel

```python
class HeteroscedasticEnsembleModel(BaseEnsembleModel)
```

Ensemble model with heteroscedastic uncertainty estimation, where each ensemble member predicts both mean and variance.

**Parameters:**

- `base_model` (nn.Module or type): Base model class or instance to ensemble
- `ensemble_size` (int, optional): Number of ensemble members. Default: `5`
- `device` (str, optional): Device to use. Default: 'cpu'

**Methods:**

- `predict(x)`: Makes prediction with uncertainty estimates, separating epistemic and aleatoric uncertainty

**Example:**

```python
import torch
import torch.nn as nn
import torchregress as tr

# Create base heteroscedastic model that outputs (mean, log_var)
class HeteroscedasticModel(nn.Module):
    def __init__(self, input_dim=10, output_dim=1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(32, output_dim)
        self.logvar_head = nn.Linear(32, output_dim)
        
    def forward(self, x):
        features = self.shared(x)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        return mean, logvar

# Create heteroscedastic ensemble
ensemble = tr.ensemble.HeteroscedasticEnsembleModel(
    base_model=HeteroscedasticModel,
    ensemble_size=5,
    input_dim=10,
    output_dim=1
)

# Make predictions with uncertainty decomposition
x = torch.randn(10, 10)
predictions = ensemble.predict(x)

# Access uncertainties
mean = predictions['mean']
total_variance = predictions['variance']
epistemic_variance = predictions['epistemic_variance']
aleatoric_variance = predictions['aleatoric_variance']
```

### HeteroscedasticBatchEnsembleModel

```python
class HeteroscedasticBatchEnsembleModel(nn.Module)
```

Batch ensemble model with heteroscedastic uncertainty estimation, using parameter sharing for efficiency.

**Parameters:**

- `backbone` (nn.Module): Base model architecture (without output head)
- `input_size` (int): Size of input features
- `output_size` (int): Size of output features
- `ensemble_size` (int, optional): Number of ensemble members. Default: `4`
- `device` (str, optional): Device to use. Default: 'cpu'

**Methods:**

- `forward(x)`: Forward pass through the model
- `predict(x)`: Makes prediction with uncertainty estimates, separating epistemic and aleatoric uncertainty

**Example:**

```python
import torch
import torch.nn as nn
import torchregress as tr

# Create backbone network (without output layer)
backbone = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU()
)

# Create batch ensemble model
model = tr.ensemble.HeteroscedasticBatchEnsembleModel(
    backbone=backbone,
    input_size=32,  # Size of the backbone's output
    output_size=1,  # Target dimension
    ensemble_size=4
)

# Make predictions
x = torch.randn(10, 10)
predictions = model.predict(x)

# Access uncertainties
mean = predictions['mean']
total_variance = predictions['variance']
epistemic_variance = predictions['epistemic_variance']
aleatoric_variance = predictions['aleatoric_variance']
```

## Utility Functions

### run_ensemble_model

```python
run_ensemble_model(model, inputs, return_individual=False)
```

Run a model on multiple input variations and aggregate results.

**Parameters:**

- `model` (Callable): Model function to run
- `inputs` (torch.Tensor or List[torch.Tensor]): List of input tensors or batched tensor [n_samples, batch_size, ...]
- `return_individual` (bool, optional): Whether to return individual predictions. Default: False

**Example:**

```python
import torch
import torchregress as tr

# Create model with dropout
model = create_model_with_dropout()

# Generate multiple inputs with perturbations
inputs = [x + 0.01 * torch.randn_like(x) for _ in range(10)]

# Run with ensemble of inputs
results = tr.ensemble.utils.run_ensemble_model(model, inputs)

# Access mean and uncertainty
mean = results['mean']
variance = results['variance']
```

### run_heteroscedastic_ensemble_model

```python
run_heteroscedastic_ensemble_model(model, inputs)
```

Run a heteroscedastic model on multiple input variations and aggregate results.

**Parameters:**

- `model` (Callable): Heteroscedastic model function
- `inputs` (torch.Tensor or List[torch.Tensor]): List of input tensors or batched tensor [n_samples, batch_size, ...]

**Example:**

```python
import torch
import torchregress as tr

# Create heteroscedastic model that outputs mean and log_var
model = create_heteroscedastic_model()

# Generate multiple inputs with perturbations
inputs = [x + 0.01 * torch.randn_like(x) for _ in range(10)]

# Run with ensemble of inputs
results = tr.ensemble.utils.run_heteroscedastic_ensemble_model(model, inputs)

# Access different uncertainty components
mean = results['mean']
total_variance = results['variance'] 
epistemic_variance = results['epistemic_variance']
aleatoric_variance = results['aleatoric_variance']
```

### generate_prediction_samples

```python
generate_prediction_samples(model, x, n_samples=10, return_samples=False)
```

Generate multiple predictions using dropout at inference time (MC Dropout).

**Parameters:**

- `model` (Callable): Model with dropout layers
- `x` (torch.Tensor): Input tensor [batch_size, ...]
- `n_samples` (int, optional): Number of samples to generate. Default: 10
- `return_samples` (bool, optional): Whether to return individual samples. Default: False

**Example:**

```python
import torch
import torchregress as tr

# Create model with dropout
model = create_model_with_dropout(dropout_prob=0.2)

# Generate prediction samples using MC Dropout
x = torch.randn(10, input_dim)
results = tr.ensemble.utils.generate_prediction_samples(
    model, x, n_samples=30
)

# Access mean prediction and uncertainty
mean = results['mean']
variance = results['variance']
```

## Common Ensemble Architectures

### Deep Ensembles

Train multiple models independently with different random initializations:

```python
import torch
import torchregress as tr

# Create deep ensemble
ensemble = tr.ensemble.DeepEnsemble(
    base_model=YourModelClass,
    ensemble_size=5,
    **model_kwargs
)

# Train ensemble
history = ensemble.fit(
    train_loader=train_loader,
    criterion=torch.nn.MSELoss(),
    optimizer_class=torch.optim.Adam,
    epochs=10,
    lr=0.001
)

# Make predictions with uncertainty
predictions = ensemble.predict(x_test)
```

### MC Dropout Ensemble

Use dropout at inference time to create an ensemble:

```python
import torch
import torchregress as tr

# Create model with dropout
model = create_model_with_dropout(dropout_prob=0.2)

# Train normally
train_model(model, train_loader)

# Generate MC Dropout samples at inference time
results = tr.ensemble.utils.generate_prediction_samples(
    model=model,
    x=x_test,
    n_samples=30
)

# Extract mean and variance
mean = results['mean']
variance = results['variance']
```

## BatchEnsemble

Efficient parameter-sharing ensemble using fast weight vectors:

```python
import torch
import torch.nn as nn
import torchregress as tr

# Create a backbone network
backbone = nn.Sequential(
    nn.Linear(input_dim, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU()
)

# Create batch ensemble output layer
batch_ensemble_layer = tr.ensemble.BatchEnsembleLinear(
    in_features=32,
    out_features=output_dim,
    ensemble_size=4
)

# Forward pass
features = backbone(x)
ensemble_outputs = batch_ensemble_layer(features)  # [batch_size, ensemble_size, output_dim]

# Calculate mean and variance
mean = ensemble_outputs.mean(dim=1)
variance = ensemble_outputs.var(dim=1)
```

## SWAG: Stochastic Weight Averaging Gaussian

### SWAG

```python
class SWAG(nn.Module)
```

Stochastic Weight Averaging - Gaussian (SWAG) approximates the posterior over model weights using first and second moments collected during SGD training. This provides a cheap alternative to full Bayesian inference while still capturing epistemic uncertainty.

**Key Concept**: Instead of training multiple independent models (as in Deep Ensembles), SWAG collects snapshots of a single model during training and uses them to approximate a Gaussian posterior over weights.

**Parameters**:

- `base_model` (nn.Module): The model architecture to use
- `max_num_models` (int, optional): Maximum number of models to store for low-rank approximation. Default: 20
- `var_clamp` (float, optional): Minimum value to clamp variance to avoid numerical issues. Default: 1e-30

**Methods**:

- `collect_model(model)`: Collect a model snapshot for SWAG averaging
- `sample(scale=1.0, diag_noise=True)`: Sample weights from the SWAG approximate posterior
- `forward(*args, **kwargs)`: Forward pass through base model with current sampled weights

**Example**:

```python
import torch
import torch.nn as nn
from torchregress.ensemble import SWAG

# Create base model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Wrap with SWAG
swag_model = SWAG(model, max_num_models=20)

# Phase 1: Warmup training (75% of epochs)
warmup_epochs = 75
for epoch in range(warmup_epochs):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
    print(f"Warmup epoch {epoch+1}/{warmup_epochs}")

# Phase 2: SWAG collection (25% of epochs)
swag_epochs = 25
for epoch in range(swag_epochs):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()

    # Collect model snapshot after each epoch
    swag_model.collect_model(model)
    print(f"SWAG epoch {epoch+1}/{swag_epochs}")

# Phase 3: Inference with uncertainty
predictions = []
n_samples = 30

for _ in range(n_samples):
    swag_model.sample(scale=0.5)  # Sample weights from posterior
    with torch.no_grad():
        pred = swag_model(x_test)
    predictions.append(pred)

# Compute mean and uncertainty
preds_stacked = torch.stack(predictions)
mean = preds_stacked.mean(0)
variance = preds_stacked.var(0)
std = torch.sqrt(variance)

print(f"Mean prediction: {mean[:5]}")
print(f"Std deviation: {std[:5]}")
```

**When to Use SWAG**:

- ✅ Want Bayesian uncertainty with single model training cost
- ✅ Have limited computational budget (cheaper than deep ensembles)
- ✅ Need calibrated uncertainty estimates
- ✅ Can afford to collect snapshots during training

**Training Tips**:

1. **Warmup Phase**: Train normally for 75-80% of total epochs
2. **Collection Phase**: Use constant or cyclical learning rate
3. **Sampling Scale**: Tune `scale` parameter (0.5-1.5) on validation set
4. **Number of Samples**: 20-50 samples usually sufficient at inference

### MultiSWAG

```python
class MultiSWAG(nn.Module)
```

Multi-SWAG trains multiple independent SWAG models and combines their predictions. This provides better uncertainty estimates than a single SWAG by capturing both:
1. **Within-SWAG uncertainty**: From weight posterior sampling
2. **Between-SWAG uncertainty**: From different local optima

**Parameters**:

- `base_model` (nn.Module): Model architecture (will be copied n_models times)
- `n_models` (int, optional): Number of independent SWAG models. Default: 5
- `max_num_models` (int, optional): Maximum snapshots per SWAG. Default: 20

**Methods**:

- `predict_with_uncertainty(x, n_samples=30, scale=1.0)`: Predict with decomposed epistemic and aleatoric uncertainty
- `predict_with_samples(x, n_samples=30, scale=1.0)`: Generate multiple prediction samples
- `forward(x, n_samples=1, scale=1.0)`: Forward pass with sampling from all SWAG models

**Example**:

```python
import torch
import torch.nn as nn
from torchregress.ensemble import MultiSWAG

# Create MultiSWAG with 5 independent models
multi_swag = MultiSWAG(SimpleModel(), n_models=5, max_num_models=20)

# Train each SWAG independently with different seeds
for swag_idx in range(5):
    print(f"\n=== Training SWAG {swag_idx+1}/5 ===")

    # Create new model with different initialization
    model = SimpleModel()
    torch.manual_seed(swag_idx)  # Different seed for each SWAG
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Warmup training
    for epoch in range(warmup_epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

    # SWAG collection
    for epoch in range(swag_epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

        # Collect snapshot for this specific SWAG
        multi_swag.swag_models[swag_idx].collect_model(model)

# Predict with uncertainty decomposition
mean, epistemic_var, aleatoric_var = multi_swag.predict_with_uncertainty(
    x_test, n_samples=30, scale=0.5
)

total_var = epistemic_var + aleatoric_var
std = torch.sqrt(total_var)

print(f"Mean: {mean[:5]}")
print(f"Epistemic uncertainty: {torch.sqrt(epistemic_var[:5])}")
print(f"Aleatoric uncertainty: {torch.sqrt(aleatoric_var[:5])}")
print(f"Total uncertainty: {std[:5]}")
```

**Advantages of MultiSWAG**:

- Better uncertainty estimates than single SWAG
- Captures multi-modal posteriors (different local optima)
- Can decompose epistemic vs aleatoric uncertainty
- More robust predictions through diversity

**Cost Comparison**:

| Method | Training Cost | Inference Cost | Uncertainty Quality |
|--------|--------------|----------------|-------------------|
| Single Model | 1× | 1× | ❌ No uncertainty |
| SWAG | 1× | 30× (sampling) | ⭐ Good epistemic |
| MultiSWAG (5) | 5× | 150× (5 × 30 samples) | ⭐⭐ Better epistemic |
| Deep Ensemble (5) | 5× | 5× | ⭐⭐⭐ Best overall |

## Advanced Ensemble Methods

### BayesianModelAveraging

```python
class BayesianModelAveraging(RegressionLoss)
```

Combines predictions from multiple models using learnable Bayesian weighting. The model weights are learned parameters that adapt during training.

**Parameters**:

- `n_models` (int): Number of models in the ensemble
- `reduction` (str, optional): Loss reduction method. Default: 'mean'

**Methods**:

- `forward(y_pred, target, mask=None, weights=None)`: Calculate BMA loss using weighted average
- `get_model_weights()`: Get the current model weights (probabilities)
- `predict_with_uncertainty(y_pred)`: Get predictions with uncertainty estimates

**Example**:

```python
import torch
from torchregress.ensemble import BayesianModelAveraging

# Create 5 different models
models = [Model() for _ in range(5)]

# Create BMA loss
bma_loss = BayesianModelAveraging(n_models=5)
optimizer = torch.optim.Adam(
    list(models[0].parameters()) +
    list(models[1].parameters()) +
    list(models[2].parameters()) +
    list(models[3].parameters()) +
    list(models[4].parameters()) +
    list(bma_loss.parameters()),  # Include BMA weights
    lr=0.001
)

# Training: models and weights adapt together
for x_batch, y_batch in train_loader:
    # Get predictions from all models
    predictions = [model(x_batch) for model in models]

    # Stack predictions: [batch, n_models, features]
    y_pred_stacked = torch.stack(predictions, dim=1)

    # BMA loss (automatically weights models)
    loss = bma_loss(y_pred_stacked, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Check learned weights
weights = bma_loss.get_model_weights()
print(f"Learned model weights: {weights}")
# Example output: tensor([0.15, 0.30, 0.25, 0.20, 0.10])

# Prediction with uncertainty
predictions = [model(x_test) for model in models]
y_pred_stacked = torch.stack(predictions, dim=1)
mean, variance = bma_loss.predict_with_uncertainty(y_pred_stacked)
```

**When to Use**:

- ✅ Have multiple pre-trained models of varying quality
- ✅ Want to learn optimal model combination automatically
- ✅ Models have complementary strengths
- ❌ Don't use if all models are similar (use equal weighting instead)

### StackingEnsemble

```python
class StackingEnsemble(RegressionLoss)
```

Uses a meta-learner (neural network) to combine base model predictions. More flexible than simple averaging.

**Parameters**:

- `n_models` (int): Number of base models
- `n_features` (int): Number of output features
- `meta_learner` (nn.Module, optional): Custom meta-learner. Default: Linear layer
- `reduction` (str, optional): Loss reduction method. Default: 'mean'

**Methods**:

- `forward(y_pred, target, mask=None, weights=None)`: Calculate stacking ensemble loss
- `predict(y_pred)`: Get final predictions from the ensemble

**Example**:

```python
import torch
import torch.nn as nn
from torchregress.ensemble import StackingEnsemble

# Create 5 base models (e.g., different architectures)
base_models = [
    LinearModel(),
    SmallNN(),
    MediumNN(),
    LargeNN(),
    ResidualModel()
]

# Option 1: Default linear meta-learner
stacking = StackingEnsemble(n_models=5, n_features=1)

# Option 2: Custom meta-learner
custom_meta = nn.Sequential(
    nn.Linear(5 * 1, 32),  # 5 models × 1 feature
    nn.ReLU(),
    nn.Linear(32, 1)
)
stacking = StackingEnsemble(n_models=5, n_features=1, meta_learner=custom_meta)

# Training
optimizer = torch.optim.Adam(
    list(itertools.chain(*[m.parameters() for m in base_models])) +
    list(stacking.parameters()),
    lr=0.001
)

for x_batch, y_batch in train_loader:
    # Get predictions from all base models
    predictions = [model(x_batch) for model in base_models]

    # Stacking combines them via meta-learner
    loss = stacking(predictions, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Prediction
predictions = [model(x_test) for model in base_models]
final_pred = stacking.predict(predictions)
```

**When to Use**:

- ✅ Have diverse base models (different architectures)
- ✅ Want to learn complex combination strategies
- ✅ Base models have nonlinear interactions
- ⚠️ Requires careful tuning to avoid overfitting

### DynamicEnsembleWeighting

```python
class DynamicEnsembleWeighting(RegressionLoss)
```

Adjusts model weights dynamically based on recent performance. Useful when model quality varies over time or input regions.

**Parameters**:

- `n_models` (int): Number of models
- `window_size` (int, optional): Size of performance tracking window. Default: 100
- `learning_rate` (float, optional): Rate of weight adaptation. Default: 0.1
- `reduction` (str, optional): Loss reduction method. Default: 'mean'

**Example**:

```python
from torchregress.ensemble import DynamicEnsembleWeighting

# Create ensemble with dynamic weighting
dynamic_ensemble = DynamicEnsembleWeighting(
    n_models=5,
    window_size=100,  # Track last 100 predictions
    learning_rate=0.1  # How fast to adapt weights
)

# Models can have varying performance over time
models = [Model() for _ in range(5)]

for x_batch, y_batch in train_loader:
    predictions = [model(x_batch) for model in models]

    # Weights adapt based on recent performance
    loss = dynamic_ensemble(predictions, y_batch)

    # The weights change dynamically during training
    current_weights = dynamic_ensemble.get_model_weights()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**When to Use**:

- ✅ Model performance varies across input space
- ✅ Want automatic adaptation to data distribution shifts
- ✅ Some models are better at certain types of inputs
- ❌ Don't use if performance is uniform

## Ensemble Method Comparison

### Decision Guide: Which Ensemble Method?

```
┌─ Budget & Requirements ─────────────────────────────────┐
│                                                          │
│  Minimal Training Cost?                                  │
│  ├─ Yes → SWAG (1× training, sampling at inference)    │
│  └─ No → Continue below                                  │
│                                                          │
│  Need Epistemic/Aleatoric Decomposition?                 │
│  ├─ Yes → HeteroscedasticEnsembleModel or MultiSWAG    │
│  └─ No → Continue below                                  │
│                                                          │
│  Inference Speed Critical?                               │
│  ├─ Yes → BatchEnsemble (parameter sharing)            │
│  └─ No → Continue below                                  │
│                                                          │
│  Have Diverse Pre-trained Models?                        │
│  ├─ Yes → BayesianModelAveraging or Stacking           │
│  └─ No → Continue below                                  │
│                                                          │
│  Best Uncertainty Estimates (any cost)?                  │
│  └─ Yes → DeepEnsemble (5-10 members)                  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Method Comparison Table

| Method | Training Cost | Inference Cost | Uncertainty Type | Calibration | Best For |
|--------|--------------|----------------|-----------------|-------------|----------|
| **DeepEnsemble** | High (N×) | Medium (N×) | ⭐⭐⭐ Epistemic | ⭐⭐⭐ Best | Gold standard, best uncertainty |
| **SWAG** | Low (1×) | High (sampling) | ⭐⭐ Epistemic | ⭐⭐ Good | Budget-constrained, single model |
| **MultiSWAG** | Medium (N×) | Very High | ⭐⭐⭐ Both | ⭐⭐⭐ Best | Bayesian UQ, multi-modal posteriors |
| **HeteroscedasticEnsemble** | High (N×) | Medium (N×) | ⭐⭐⭐ Both | ⭐⭐⭐ Best | Decomposed uncertainty |
| **BatchEnsemble** | Low (1.2×) | Fast (1.2×) | ⭐⭐ Epistemic | ⭐⭐ Good | Speed critical, efficient training |
| **MC Dropout** | Low (1×) | Medium (sampling) | ⭐ Epistemic | ⭐ Fair | Quick baseline, model has dropout |
| **BayesianModelAveraging** | N/A | Low (N×) | ⭐ Mixed | ⭐⭐ Varying | Combining pre-trained models |
| **StackingEnsemble** | High (N×) | Low (N×) | ⭐⭐ Complex | ⭐⭐ Good | Diverse architectures, complex combos |

### Best Practices

**1. Training Deep Ensembles**:

```python
# Use different random seeds for diversity
ensemble_models = []
for i in range(5):
    torch.manual_seed(i)
    model = MyModel()
    model.apply(init_weights)
    ensemble_models.append(model)
```

**2. SWAG Tips**:

```python
# Typical schedule
total_epochs = 100
warmup_epochs = 75  # 75% warmup
swag_epochs = 25    # 25% collection

# Use cyclical or constant LR during SWAG collection
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=0.001, max_lr=0.01,
    step_size_up=5, mode='triangular'
)
```

**3. Uncertainty Calibration**:

```python
# Always validate calibration
from torchregress.metrics import expected_calibration_error

# For regression, check if std matches actual errors
predictions = ensemble.predict(x_val)
std = predictions['std']
errors = torch.abs(predictions['mean'] - y_val)

# Errors should fall within predicted confidence intervals
within_1std = (errors < std).float().mean()
print(f"Fraction within 1 std: {within_1std:.3f}")  # Should be ~0.68

# Use calibration metrics
ece = expected_calibration_error(predictions, y_val)
print(f"Calibration error: {ece:.4f}")
```

**4. Production Deployment**:

```python
# For production, save ensemble efficiently
torch.save({
    'ensemble_models': [m.state_dict() for m in ensemble.models],
    'config': ensemble_config
}, 'ensemble.pt')

# Load and run inference
checkpoint = torch.load('ensemble.pt')
ensemble = DeepEnsemble(MyModel, ensemble_size=5)
for i, state_dict in enumerate(checkpoint['ensemble_models']):
    ensemble.models[i].load_state_dict(state_dict)
```
