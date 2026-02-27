# Training with Noisy Labels

This page covers loss functions designed to handle noisy or corrupted labels in regression tasks. These methods are adapted from classification literature but specifically designed for continuous target variables.

## Overview

Noisy labels are a common problem in real-world datasets. Standard training methods can be sensitive to these noisy labels, leading to poor model performance. The loss functions in this section are designed to be robust to noisy labels, allowing you to train accurate models even when your data is not perfectly clean.

## Noise-Adaptive Loss

This loss function learns a confidence weight for each training sample, automatically downweighting noisy labels during training. The weights are learned parameters that are updated based on the loss behavior.

### Mathematical Foundation

The Noise-Adaptive Loss adds a learnable weight $w_i$ to each sample $i$. The loss for a batch is then:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} w_i \cdot \mathcal{L}_{base}(y_i, \hat{y}_i)$$

Where $w_i$ is the learned weight for sample $i$, and $\mathcal{L}_{base}$ is a standard regression loss like MSE or MAE.

### When to Use Noise-Adaptive Loss

**Ideal scenarios:**
- When you have a dataset with a mix of clean and noisy labels.
- When you want to identify which samples are likely to be noisy.

**Example: Training with learnable sample weights**

```python
import torch
import torch.nn as nn
import torchregress as tr

# Create a model
model = nn.Sequential(nn.Linear(10, 1))

# Create loss for a dataset with 1000 samples
loss_fn = NoiseAdaptiveLoss(n_samples=1000)  # concept example (not currently in torchregress)

# Separate optimizer for the sample weights
weight_optimizer = torch.optim.SGD([loss_fn.sample_weight_logits], lr=1e-3)

# Training loop
for epoch in range(100):
    for x, y, indices in train_loader:
        # Forward pass
        y_pred = model(x)
        
        # Loss computation with sample indices
        loss = loss_fn(y_pred, y, sample_indices=indices)
        
        # Backward and optimize
        optimizer.zero_grad()
        weight_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        weight_optimizer.step()

# Get learned weights to identify noisy samples
sample_weights = loss_fn.get_sample_weights()
noisy_indices = torch.where(sample_weights < 0.5)[0]
```

## Co-Teaching Loss

Co-teaching trains two networks simultaneously, where each network selects small-loss samples (assumed clean) for the other network to learn from. The `forget_rate` determines what fraction of samples to exclude (assumed noisy).

### Mathematical Foundation

For each batch, two models, A and B, are trained. The loss for model A is computed on a subset of the batch selected by model B, and vice-versa. The subset is chosen by selecting the samples with the smallest loss according to the other model.

### When to Use Co-Teaching Loss

**Ideal scenarios:**
- When you have a high level of noise in your labels.
- When you can afford the computational cost of training two models.

**Example: Co-teaching with two models**

```python
import torch
import torch.nn as nn
import torchregress as tr

# Create two models
model1 = nn.Sequential(nn.Linear(10, 1))
model2 = nn.Sequential(nn.Linear(10, 1))

# Create loss function
loss_fn = CoTeachingLoss(forget_rate=0.2)  # concept example (not currently in torchregress)

# Optimizers for each model
optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

# Training loop
for epoch in range(100):
    for x, y in train_loader:
        # Forward pass for both networks
        y_pred1 = model1(x)
        y_pred2 = model2(x)
        
        # Co-teaching loss (each network teaches the other)
        loss1, loss2 = loss_fn(y_pred1, y_pred2, y, epoch=epoch)
        
        # Separate backward passes
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
```

## RENT Loss

Robust Ensemble Training (RENT) loss uses ensemble disagreement to identify noisy labels. Samples with high disagreement among ensemble members are downweighted, as they are likely to have noisy labels or be difficult/ambiguous examples.

### Mathematical Foundation

The RENT loss weights each sample by the inverse of the disagreement among the ensemble members. The disagreement is measured as the variance of the predictions from the ensemble members.

### When to Use RENT Loss

**Ideal scenarios:**
- When you are already using an ensemble of models.
- When you want a simple way to make your ensemble more robust to noisy labels.

**Example: RENT with a deep ensemble**

```python
import torch
import torch.nn as nn
import torchregress as tr
from torchregress.ensemble import DeepEnsemble

# Create an ensemble of 5 models
base_model = nn.Sequential(nn.Linear(10, 1))
ensemble = DeepEnsemble(base_model, ensemble_size=5)

# Create loss function
loss_fn = RENTLoss(ensemble_size=5)  # concept example (not currently in torchregress)

# Optimizer for the ensemble
optimizer = torch.optim.Adam(ensemble.parameters(), lr=1e-3)

# Training loop
for x, y in train_loader:
    # Get predictions from all ensemble members
    ensemble_preds = ensemble.forward_all(x)
    
    # Compute RENT loss
    loss = loss_fn(ensemble_preds, y)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
