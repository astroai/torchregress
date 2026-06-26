# Adaptive-Prior Bayesian Uncertainty (VIDS-style)

This algorithm implements a variational regressor that trains an input-conditional, context-aware adaptive prior to make model predictions robust and well-calibrated under covariate shifts.

---

## Mathematical Background

In standard Bayesian neural networks, the prior $p(\theta)$ is static and independent of the test distribution. Under covariate shift, this can lead to overconfident or poorly calibrated predictions.

VIDS (Variational Inference under Distribution Shifts) addresses this by learning an **adaptive prior** $p(\theta | X_{\text{context}}, x)$ conditioned on the global context of the feature distribution $X_{\text{context}}$ and the specific query point $x$.

### 1. Synthetic Environment Sampling

To train the adaptive prior, we generate $E$ synthetic environments $e = (X_e, Y_e)$ by bootstrapping subsets of our training data. Each bootstrap sample represents a simulated shifted training/validation environment.

### 2. Amortized Variational Inference (Guide)

For each environment $e$, we define a variational posterior guide $q(\theta | X_e, Y_e)$ parameterized by a neural network that maps context summaries of the environment's features and targets to posterior parameters:

$$q(\theta | X_e, Y_e) = \mathcal{N}\left(\mu_{\text{post}}(X_e, Y_e),\; \text{diag}(\sigma^2_{\text{post}}(X_e, Y_e))\right)$$

### 3. Adaptive Prior Network

The prior network $p(\theta | X_e, x)$ maps the environment context features $X_e$ and a query feature $x$ to prior parameters:

$$p(\theta_x | X_e, x) = \mathcal{N}\left(\mu_{\text{prior}}(X_e, x),\; \text{diag}(\sigma^2_{\text{prior}}(X_e, x))\right)$$

The parameters are optimized by maximizing the Evidence Lower Bound (ELBO) averaged across all synthetic environments:

$$\text{ELBO} = \sum_{e=1}^E \left[ \mathbb{E}_{q(\theta|X_e,Y_e)} \left[ \log p(Y_e | X_e, \theta) \right] - \beta \cdot \frac{1}{|X_e|}\sum_{x \in X_e} D_{KL}\left( q(\theta|X_e,Y_e) \;||\; p(\theta|X_e, x) \right) \right]$$

where:

- $\log p(Y_e | X_e, \theta)$ is the likelihood (negative log-likelihood loss w.r.t observation noise variance $\sigma_{\text{noise}}^2$).
- $D_{KL}(q || p)$ is the analytical KL divergence between the diagonal Gaussian guide and the diagonal Gaussian adaptive prior.
- $\beta$ is a KL regularization scaling factor.

### 4. Test-Time Inference

At test time, given the global training context $X_{\text{train}}$ and a new test point $x_{\text{query}}$, we evaluate the adaptive prior:

$$\theta^{(s)} \sim p(\theta | X_{\text{train}}, x_{\text{query}})$$

We draw $S$ samples from this prior to predict $\mu^{(s)} = \theta^{(s)} \cdot [x_{\text{query}}, 1]$. The predictive variance is decomposed into:

- **Epistemic Uncertainty** (context-shift variance):
  $$\sigma^2_{\text{epistemic}} = \text{Var}\left(\{\mu^{(s)}\}_{s=1}^S\right)$$
- **Aleatoric Uncertainty** (observation noise):
  $$\sigma^2_{\text{aleatoric}} = \sigma^2_{\text{noise}}$$
- **Total Uncertainty**: $\sigma^2_{\text{total}} = \sigma^2_{\text{epistemic}} + \sigma^2_{\text{aleatoric}}$.

---

## High-Level API: `VIDSRegressor`

```python
from torchregress.algorithms import VIDSRegressor

# 1. Instantiate the regressor
model = VIDSRegressor(
    in_features=5,
    target_dim=1,
    hidden_dim=64,
    noise_variance_init=0.1,
)

# 2. Fit the adaptive prior and guide using bootstrap environments
model.fit(
    x_train_features=x_train,
    y_train=y_train,
    n_environments=32,       # Number of synthetic environments
    bootstrap_fraction=0.3,   # Size of each bootstrap environment
    epochs=50,
    beta=1.0,                 # KL penalty multiplier
)

# 3. Predict distribution under potential shift
pred = model.predict_distribution(x_test, n_samples=30)
```

### Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `in_features` | `int` | — | Number of input features |
| `target_dim` | `int` | `1` | Dimension of the target space |
| `hidden_dim` | `int` | `64` | Number of hidden units in guide and prior networks |
| `prior_variance_init` | `float` | `1.0` | Reserved API parameter (not yet applied in `fit`) |
| `noise_variance_init` | `float` | `0.1` | Initial observation noise variance $\sigma_{\text{noise}}^2$ |
| `jitter` | `float` | `1e-6` | Stabilizer added to diagonal contexts |

---

## Complete Example

```python
import torch
from torchregress.algorithms import VIDSRegressor

# Generate synthetic train data (source distribution)
torch.manual_seed(42)
x_train = torch.randn(100, 3)
y_train = x_train[:, [0]] * 1.5 - x_train[:, [1]] * 0.8 + 0.3 * torch.randn(100, 1)

# Generate shifted test data (covariate shift)
x_test = torch.randn(20, 3) + 2.0  # shifted mean

model = VIDSRegressor(in_features=3, target_dim=1, hidden_dim=32)

# Train using VIDS ELBO
model.fit(
    x_train_features=x_train,
    y_train=y_train,
    n_environments=12,
    bootstrap_fraction=0.4,
    epochs=10,
    lr=1e-2,
)

# Evaluate predictions
pred = model.predict_distribution(x_test, n_samples=20)
print("Shifted Test Means:\n", pred.mean.squeeze(-1))
print("Shifted Test Total Stds:\n", pred.std.squeeze(-1))
```

---

## When to use this method

| Scenario / Goal | Recommended Choice | Rationale |
|:---|:---:|:---|
| **High Covariate Shift** (train and test features differ) | **Yes (Recommended)** | Standard BNNs are overconfident on shifted domains; VIDS adapts the prior to the test domain density. |
| **Out-of-Distribution Calibration** | **Yes** | Synthetic bootstrap environments train the prior to remain calibrated across a range of shifts. |
| **End-to-End deep networks** | **Warning** | VIDS fits a linear head on top of features. It is best used as a last-layer uncertainty head on top of pre-trained/frozen features. |

---

## Next steps

- [Effective Bayesian Laplace](heteroscedastic_laplace.md) — alternative post-hoc uncertainty decomposition with lower computational cost
- [IVON Optimizer](ivon.md) — variational training that maintains a weight posterior during stochastic optimisation
- [Conformal prediction](../conformal/index.md) — coverage-guaranteed intervals without distributional assumptions
- [Test-time adaptation](../test-time/ot-shift-conformal.md) — OT-inspired reweighting for score shift under covariate drift

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Slavutsky & Blei, "Quantifying Uncertainty in the Presence of Distribution Shifts" (arXiv:2506.18283 / NeurIPS 2025). |
