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
import torchregression as tr

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
import torchregression as tr

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
import torchregression as tr

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
import torchregression as tr

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
import torchregression as tr

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
import torchregression as tr

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
import torchregression as tr

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
import torchregression as tr

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
import torchregression as tr

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
import torchregression as tr

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
