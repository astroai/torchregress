# Neural Ratio Estimation (NRE)

> ← [Normalizing Flows](nflows.md) | [Loss catalogue](index.md) →

**Neural Ratio Estimation** trains a classifier to estimate the likelihood-to-evidence ratio
$r(x,\theta) = p(x \mid \theta)/p(x)$ directly, without ever modelling a normalized density.
A classifier that separates true parameter–observation pairs from mismatched pairs estimates
$\log r(x,\theta)$ as its logit at its optimum, so ranking candidate parameters by that logit is
maximum-likelihood estimation. The training prior cancels in the ratio, which makes NRE attractive for
simulator-based parameter inference, likelihood-ratio tests, and amortized regression heads on
problems where a normalized likelihood is unavailable or expensive.

The implementation is in [`torchregress.losses.nre`](../api/losses.md): `NeuralRatioLoss` wraps any
`nn.Module` scorer, `build_ratio_classifier` constructs the usual MLP, and
`neural_ratio_logits` / `neural_ratio_loss` are the functional forms.

!!! abstract "The idea in one line"
    Train a classifier on *joint* pairs $(\theta, x) \sim p(\theta, x)$ versus *marginal* pairs
    $(\theta, x) \sim p(\theta)\,p(x)$; its logit is $\log p(x \mid \theta) - \log p(x)$, i.e. the
    likelihood ratio that identifies $\theta$ from $x$.

---

## When to use which objective

| Situation | Prefer | Why |
|:----------|:-------|:----|
| You can sample $(\theta, x)$ from a simulator and need an MLE/MAP estimate, a likelihood-ratio test, or hypothesis ranking | **`NeuralRatioLoss`** | Estimates $r(x,\theta)$ directly; no density model, Jacobian, or normalized likelihood required |
| You need a calibrated conditional density $p(y \mid x)$ for sampling or predictive intervals | [`GaussianNLLLoss`](gaussian.md) or [`NormalizingFlowLoss`](nflows.md) | NRE gives a ratio (likelihood up to an $x$-dependent constant), not a normalized density to sample from |
| You already train a conditional flow and want to rank alternate parameter contexts | [`ContrastiveFlowLoss`](nflows.md) | InfoNCE over flow log-densities; needs the `zuko` extra and a trained density model, and inherits its density-estimation quality |
| You want a full amortized posterior $p(\theta \mid x)$ with sampling | Normalizing flow density surrogate (`NormalizingFlowLoss` in SBI mode) | NPE-style surrogates model the posterior directly; NRE needs extra inference steps (grid, MCMC, or reweighting) to turn ratios into posterior samples |
| You want a regression head on data whose likelihood is intractable | **`NeuralRatioLoss`** (joint encoder + classifier) | The encoder can output $\hat\theta$ and be scored by the ratio against negative candidates instead of a squared error |

---

## Mathematical background

### The joint-to-marginal ratio

Let $\theta$ be a parameter/hypothesis (the *context* in the API) and $x$ an observation. NRE targets

$$
r(x,\theta) \;=\; \frac{p(x \mid \theta)}{p(x)} \;=\; \frac{p(\theta, x)}{p(\theta)\,p(x)} .
$$

Consider a classifier $s(\theta, x)$ trained with balanced labels to distinguish samples from the
joint $p(\theta, x)$ (label 1) from samples from the marginal $p(\theta)\,p(x)$ (label 0). Its
Bayes-optimal posterior is

$$
p\bigl(\text{joint} \mid \theta, x\bigr)
= \frac{p(\theta, x)}{p(\theta, x) + p(\theta)\,p(x)}
= \sigma\!\bigl(\log r(x,\theta)\bigr),
$$

so the logit of the optimal classifier is exactly the log likelihood ratio,

$$
s^\star(\theta, x) \;=\; \log r(x,\theta) \;=\; \log p(x \mid \theta) - \log p(x).
$$

This is the *likelihood-ratio trick* (Cranmer, Pavez & Louppe 2015): classification replaces
density estimation, and the classifier can be any differentiable scorer, e.g.
`build_ratio_classifier`.

### Losses

For one observation with a matched positive context $\theta^+$ and $K$ negative candidate
contexts $\{\theta^-_k\}_{k=1}^K$, write $s_+ = s(\theta^+, x)$ and $s^-_k = s(\theta^-_k, x)$.
The **balanced binary NRE** objective is

$$
\mathcal{L}_{\text{bce}}
= -\log \sigma(s_+)
  - \frac{1}{K}\sum_{k=1}^{K} \log \sigma(-s^-_k)
= \operatorname{softplus}(-s_+)
  + \frac{1}{K}\sum_{k=1}^{K} \operatorname{softplus}(s^-_k),
$$

where the $1/K$ factor balances the single positive against the $K$ negatives. The
**multiclass (categorical) NRE** objective treats the positive and the $K$ negatives as the
$K+1$ classes of a softmax classifier,

$$
\mathcal{L}_{\text{multiclass}}
= -\log \frac{\exp(s_+)}{\exp(s_+) + \sum_{k=1}^{K}\exp(s^-_k)}
= -s_+ + \log\sum_{k=0}^{K}\exp\bigl(s_k\bigr),
$$

with $s_0 = s_+$. This is the standard categorical cross-entropy, i.e. the **NRE-B** /
InfoNCE-style form: the softmax is normalized over the sampled candidate set, and the objective is
a contrastive bound rather than an exact finite-$K$ ratio estimator. As $K \to \infty$ its optimum
satisfies $\exp(s(\theta, x)) \propto r(x,\theta)$, i.e. the logit approaches $\log r$ up to an
additive constant; at finite $K$ NRE-B has an intrinsic bias term (Miller, Weniger & Forré 2022).
Use `criterion="cnre"` (below) or `criterion="bce"` (NRE-A) when an accurate absolute log-ratio
is needed. Use the multiclass form only for ranking with many candidates.

**Contrastive NRE (CNRE)** (Miller, Weniger & Forré 2022) removes that bias by adding a class
$y = 0$ in which none of the $K$ candidates generated $x$. With prior odds $\gamma$ between
"candidate $k$ generated $x$" ($y \geq 1$, uniform over $k$) and $y = 0$, the class posterior is

$$
q(y = k \mid \Theta, x) = \frac{(\gamma/K)\, e^{s(\theta_k, x)}}
{1 + (\gamma/K)\sum_{j=1}^{K} e^{s(\theta_j, x)}}, \qquad
q(y = 0 \mid \Theta, x) = \frac{1}{1 + (\gamma/K)\sum_{j=1}^{K} e^{s(\theta_j, x)}},
$$

and the loss mixes a dependent set (the true $\theta_+$ and $K-1$ negatives) with an independent
set (all $K$ negatives):

$$
\mathcal{L}_{\text{CNRE}} = -\frac{\gamma}{1+\gamma}\log q\bigl(y = 1 \mid \theta_+, \theta^-_{1:K-1}, x\bigr)
- \frac{1}{1+\gamma}\log q\bigl(y = 0 \mid \theta^-_{1:K}, x\bigr).
$$

The $y = 0$ class fixes the normalisation, so the optimum is $s = \log r(x, \theta)$ exactly for
every $K$. $K = 1$ recovers binary NRE with prior odds $\gamma$. The negatives must be drawn from
the reference prior independently of $x$, so `negative_target` is rejected for `criterion="cnre"`.

!!! tip "Choosing $\gamma$ and $K$"
    $\gamma = 1$ (the default) is a safe start. Larger $K$ makes each update more informative, but
    unlike NRE-B the estimate stays unbiased at any $K$. A typical range is $K = 2$–$32$.

In both objectives the classifier sees a positive pair and negative pairs. The reduction over the
batch is handled by [`BaseLoss`](base.md) (`'none' | 'mean' | 'sum'`, `mask`, `weights`).

### Why the training prior cancels for the MLE

Insert $p(\theta, x) = p(\theta)\,p(x \mid \theta)$ into the ratio:

$$
r(x,\theta) = \frac{p(\theta)\,p(x \mid \theta)}{p(\theta)\,p(x)} = \frac{p(x \mid \theta)}{p(x)} .
$$

The parameter prior $p(\theta)$ cancels identically. Consequently

$$
\arg\max_{\theta} \log r(x,\theta)
= \arg\max_{\theta} \bigl[\log p(x \mid \theta) - \log p(x)\bigr]
= \arg\max_{\theta} \log p(x \mid \theta),
$$

which is the maximum-likelihood estimate; $\log p(x)$ does not depend on $\theta$. The
requirement is **not** that the prior be uniform: it is that the negative contexts are drawn from
the *same* reference prior $p(\theta)$ (or reweighted to it), so that the marginal class really is
$p(\theta)\,p(x)$. Under that condition the training prior is irrelevant for ranking $\theta$.

If the training negatives instead come from a proposal $q(\theta)$, the optimal logit becomes

$$
s(\theta, x) = \log r(x,\theta) + \log\frac{p(\theta)}{q(\theta)},
$$

so the proposal can be corrected by subtracting $\log p(\theta) - \log q(\theta)$ when the two
densities are known. A MAP estimate adds the target prior back explicitly:
$\log p(\theta \mid x) = \log r(x,\theta) + \log p(\theta) + \text{const}$.

This argument is exact for the balanced binary objective. The finite-$K$ multiclass objective
superimposes the contrastive bias discussed above on the same prior-free target.

---

## API

```python
import torch

from torchregress.losses.nre import (
    NeuralRatioLoss,
    build_ratio_classifier,
    neural_ratio_logits,
    neural_ratio_loss,
)

# Class form: the loss owns (and optimizes) the classifier parameters.
classifier = build_ratio_classifier(context_dim=3, observable_dim=8, hidden_dims=(64, 64))
loss_fn = NeuralRatioLoss(classifier, criterion="bce", reduction="mean")
loss = loss_fn(
    positive_context,            # [N, 3] candidate parameters (e.g. labels or model estimates)
    observations,                # [N, 8] data
    negative_context=negative,   # [N, K, 3], [1, K, 3] or [K, 3] (shared bank)
    negative_target=None,        # optional [N, K, 8] mismatched observations
    mask=sample_mask,            # optional [N] boolean
    weights=sample_weights,      # optional [N] non-negative
)

# Functional form (same math, classifier passed explicitly).
loss = neural_ratio_loss(classifier, positive_context, observations, negative)

# Log-ratio estimation and plain scoring.
ratios = loss_fn.log_likelihood_ratio(positive_context, observations, negative)  # [N, K]
logits = neural_ratio_logits(classifier, candidate_context, observations)        # [N]
```

`criterion="bce"` is the balanced binary loss; `criterion="multiclass"` is the categorical /
InfoNCE-style loss; `criterion="cnre"` is the bias-free contrastive loss (tune `gamma`, default
`1.0`). The classifier is a plain `nn.Module` mapping
`concat([context, observable])` to one logit; `build_ratio_classifier` tags it with
`context_dim` / `observable_dim`, and `NeuralRatioLoss` infers the dimensions lazily for custom
modules.

### Negative-sampling conventions

`negative_context` is required; its shape determines how negatives are shared across the batch:

| Shape | Meaning |
|:------|:--------|
| `[N, K, C]` | Per-sample negative hypotheses |
| `[1, K, C]` | One shared bank of $K$ hypotheses broadcast across the batch |
| `[K, C]` | Shared bank as a 2-D tensor; rejected as **ambiguous** when `K == N` |

For a 2-D `[N, C]` input whose leading dimension equals the batch size, set
`shared_negative_context=True` (shared bank) or `False` (per-sample, one negative per sample).
The same rules apply to `negative_target` via `shared_negative_target`.

`negative_target` selects the NRE variant:

- **`None` (paired negatives)** — the negative hypotheses are scored against the *same*
  observation $x$: $s^-_k = s(\theta^-_k, x)$. The classifier learns the ratio, and
  `log_likelihood_ratio` then returns $\log p(x \mid \theta^+) - \log p(x \mid \theta^-_k)$,
  a likelihood ratio *between hypotheses*.
- **Given (mismatched pairs)** — each negative is a full pair $s^-_k = s(\theta^-_k, x^-_k)$
  with $x^-_k$ sampled independently of $\theta^-_k$ (e.g. shuffled observations). This is the
  canonical joint-versus-marginal NRE training setup.

!!! tip "How many negatives?"
    More negatives reduce the variance of the ratio estimate per step. Typical values are
    $K = 4$–$64$ for the binary loss and $K = 8$–$128$ for the multiclass loss; a shared bank
    of prior draws is cheap because the classifier evaluates the same candidates for every
    sample. Watch that the ratio estimate does not saturate (very large logits) when candidates
    are far apart.

---

## Runnable example: Gaussian MLE by grid argmax

The simulator is $c \sim \mathcal{N}(0, 1)$ and $x \mid c \sim \mathcal{N}(c, \sigma^2)$ with
$\sigma = 0.5$. We train the classifier on paired negatives drawn from the reference prior, then
invert a new observation by scoring a dense candidate grid and taking the argmax (the MLE).

```python
import torch

from torchregress.losses.nre import (
    NeuralRatioLoss,
    build_ratio_classifier,
    neural_ratio_logits,
)

torch.manual_seed(0)
sigma = 0.5


def simulate(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    c = torch.randn(n, 1)                       # latent parameter (NRE context)
    x = c + sigma * torch.randn(n, 1)           # observation
    return c, x


# 1. Train the ratio classifier. Negative contexts are alternative hypotheses
#    c' for the same observation, drawn from the same prior as c.
classifier = build_ratio_classifier(context_dim=1, observable_dim=1, hidden_dims=(64, 64))
loss_fn = NeuralRatioLoss(classifier, criterion="bce")
optimiser = torch.optim.Adam(classifier.parameters(), lr=3e-3)

for _ in range(600):
    c, x = simulate(256)
    negative_c = torch.randn(256, 8, 1)         # reference-prior alternatives
    loss = loss_fn(c, x, negative_context=negative_c)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
print(f"final training loss: {float(loss.detach()):.3f}")

# 2. Maximum-likelihood inversion on a grid: score every candidate for new data.
grid = torch.linspace(-3.0, 3.0, 121).reshape(-1, 1)
torch.manual_seed(123)
c_true, x_obs = simulate(5)
n_obs, n_candidates = x_obs.shape[0], grid.shape[0]
candidate_context = grid.unsqueeze(0).expand(n_obs, n_candidates, 1).reshape(-1, 1)
observations = x_obs.unsqueeze(1).expand(n_obs, n_candidates, 1).reshape(-1, 1)
with torch.no_grad():
    scores = neural_ratio_logits(classifier, candidate_context, observations)
scores = scores.reshape(n_obs, n_candidates)
c_hat = grid[scores.argmax(dim=1)].squeeze(-1)

# The flat-prior MLE of a Gaussian location model is x_obs (nearest grid point).
print("true c:     ", c_true.squeeze(-1))
print("grid MLE c: ", c_hat)
print(f"mean |c_hat - x_obs| = {float((c_hat - x_obs.squeeze(-1)).abs().mean()):.3f}")
print(f"mean |c_hat - c_true| = {float((c_hat - c_true.squeeze(-1)).abs().mean()):.3f}")

# 3. Pairwise log likelihood ratio between two hypotheses for one observation.
hypothesis_a = torch.tensor([[0.0]])
hypothesis_b = torch.tensor([[1.0]])
log_ratio = loss_fn.log_likelihood_ratio(
    hypothesis_a, x_obs[:1], hypothesis_b.unsqueeze(0)
)
print(f"estimated log-ratio at c=0 vs c=1: {float(log_ratio.detach()):.3f}")
```

On a typical run the grid-MLE estimate tracks the analytic MLE $x_{\text{obs}}$ within about one
grid step (a mean absolute deviation of about $0.03$ on a grid with spacing $0.05$) and the
generating $c$ within the observation noise ($\approx 0.2$ for $\sigma = 0.5$).

!!! note "Using NRE as a training loss"
    The example above trains the classifier on **known** parameter–observation pairs and performs
    inference at test time. To use NRE as an amortized regression loss instead, let `y_pred` be a
    model's predicted context $\hat\theta = g_\phi(x)$ and pass the negative hypotheses at training
    time: the classifier and the encoder are optimized jointly, and minimizing
    $-\log\sigma(s(\hat\theta, x))$ pushes $\hat\theta$ toward higher ratio than the alternatives.
    Joint encoder–classifier training is co-adaptive; evaluate the recovered estimates on held-out
    data, and consider pretraining or periodically refreshing the classifier.

---

## Frequentist uncertainties: wifi ensembles

A trained ratio estimator has no error bar. *wifi ensembles* (Benevedes & Thaler, PRD 112,
056024, 2025) add one. Train $M$ ratio estimators $f_i(x) \approx \log r(x)$ independently
(bootstrap resamples), freeze them, and model

$$
\log r_w(x) = w_0 + \sum_{i=1}^{M} w_i f_i(x),
$$

fitting $w$ on held-out samples $x \sim p$ (numerator) and $x \sim q$ (denominator) by the
symmetrised maximum-likelihood-classifier loss

$$
L(w) = \mathbb{E}_p\bigl[-s_w + e^{-s_w}\bigr] + \mathbb{E}_q\bigl[s_w + e^{s_w}\bigr],
\qquad s_w = \log r_w,
$$

which is convex in $w$ and minimised pointwise by $s = \log p/q$. Because $\hat w$ is an
M-estimator, its covariance is the sandwich $\Sigma_w = H^{-1} V H^{-1}$ ($H$ the Hessian of
$L$, $V$ the covariance of the per-sample scores), and

$$
\operatorname{Var}\bigl[\log \hat r(x)\bigr] = F(x)^\top \Sigma_w F(x), \qquad F = (1, f_1, \ldots, f_M).
$$

For a mixture $x \sim \kappa p + (1-\kappa) q$, the pseudo-MLE of $\kappa$ maximises
$\sum_j \log\bigl(1 + \kappa(\hat r_j - 1)\bigr)$ and its variance adds the ratio uncertainty
(Gong & Samaniego 1981):

$$
\operatorname{Var}(\hat\kappa) = \frac{J + c^\top \Sigma_w c}{I^2},\quad
I = \sum_j \frac{(r_j-1)^2}{D_j^2},\;
J = \sum_j \Bigl(\frac{r_j-1}{D_j}\Bigr)^2,\;
c = \sum_j \frac{r_j F_j}{D_j^2},\;
D_j = 1 + \kappa(r_j - 1).
$$

[`WifiRatioEnsemble`](../api/inference.md#wifi-ensembles) works on stacked member outputs, so the
members can be `NeuralRatioLoss` classifiers, plain binary classifiers, or flow log-density
differences.

```python
import torch
import torch.nn.functional as F
from torch import nn

from torchregress.inference import WifiRatioEnsemble

torch.manual_seed(0)
# Numerator p = N([1, 0], I), denominator q = N([0, 0], I) in 2-D.
def draw_p(n): return torch.randn(n, 2) + torch.tensor([1.0, 0.0])
def draw_q(n): return torch.randn(n, 2)

# 1) Train M classifiers on bootstrap resamples; logit ~ log p(x)/q(x).
x_train = torch.cat([draw_p(4000), draw_q(4000)])
label = torch.cat([torch.ones(4000), torch.zeros(4000)])
members = []
for _ in range(8):
    index = torch.randint(len(x_train), (len(x_train),))
    net = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1))
    optimiser = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(300):
        loss = F.binary_cross_entropy_with_logits(net(x_train[index]).squeeze(-1), label[index])
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    members.append(net)

# 2) Freeze the members and fit the wifi weights on fresh held-out samples.
features = lambda x: WifiRatioEnsemble.member_features(members, x)  # [N, M]
wifi = WifiRatioEnsemble().fit(features(draw_p(5000)), features(draw_q(5000)))

# 3) Log ratio with a frequentist standard error.
x_test = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]])
log_r, log_r_std = wifi.log_ratio(features(x_test)), wifi.log_ratio_std(features(x_test))
true_log_r = x_test[:, 0] - 0.5

# 4) Mixture fraction kappa in x ~ kappa p + (1 - kappa) q, with the ratio
#    uncertainty propagated (Gong-Samaniego).
from_p = torch.rand(10_000) < 0.2
x_mix = torch.where(from_p.unsqueeze(1), draw_p(10_000), draw_q(10_000))
estimate = wifi.mixture_fraction(features(x_mix))
print(f"kappa = {estimate.fraction:.3f} +/- {estimate.std:.3f} "
      f"(stat {estimate.std_statistical:.3f}, ratio {estimate.std_ratio:.3f})")
```

!!! warning "Validity conditions"
    The intervals are asymptotically valid when the span of $\{1, f_i\}$ contains the true log
    ratio, which is why the paper uses large ensembles ($M = 16$–$32$) of diverse, well-trained
    members. Fit $w$ on data **not** used to train the members. Nearly collinear members make
    individual weights unstable (large, alternating signs), but $\log \hat r(x)$ and its error
    stay well defined. In tests with a well-specified basis the 1σ intervals cover 64–71% (nominal
    68%), and dropping the ratio term from $\operatorname{Var}(\hat\kappa)$ falls to 25% coverage.

The equations are a reconstruction from the paper's text, validated by coverage tests on
problems with a known ratio rather than against the authors' code.

## Comparison with related objectives

| Objective | What it learns | Needs a density model? | Negatives required? | Main limitation |
|:----------|:---------------|:----------------------|:--------------------|:----------------|
| **`NeuralRatioLoss`** (this page) | $s(\theta, x) \approx \log p(x \mid \theta) - \log p(x)$ via classification | No | Yes, from the reference prior (or reweighted proposal) | Ratio only; negatives/prior mismatch biases the estimate |
| [`ContrastiveFlowLoss`](nflows.md) | InfoNCE over flow log-densities | Yes (`zuko` flow) | Yes, alternate contexts | Flow must be trained/fit; needs `zuko`; density calibration matters |
| [`NormalizingFlowLoss`](nflows.md) / [`GaussianNLLLoss`](gaussian.md) | Normalized conditional density $p(y \mid x)$ | Yes | No | Requires density parameterization; may be hard for intractable likelihoods |
| NPE-style flow surrogate (SBI) | Amortized posterior $p(\theta \mid x)$ | Yes (flow) | No (simulate $(\theta, x)$ pairs) | Posterior can be miscalibrated; needs SBC/TARP checks |

In short: NRE is the lightest of these when you have a simulator and care about **ratios and
rankings**, but it is not a replacement for a calibrated density or posterior surrogate. For
finite-sample coverage of prediction intervals, pair any inference with conformal methods and
calibration diagnostics (see [calibration metrics](../metrics/calibration.md)).

---

## Caveats and failure modes

!!! warning "What NRE does not give you"
    - **Ratio, not density.** The logit estimates $\log p(x \mid \theta) - \log p(x)$: differences
      between $\theta$ values are meaningful, but absolute values across *different observations*
      $x$ are only comparable up to $\log p(x)$. Do not treat the logit as a log-likelihood, a
      posterior probability, or a calibrated score out of the box.
    - **Negative sampling defines the question.** If negatives are not drawn from the reference
      prior $p(\theta)$ (or from a proposal $q(\theta)$ you can correct with
      $\log p(\theta) - \log q(\theta)$), the classifier learns
      $\log r(x,\theta) + \log[p(\theta)/q(\theta)]$ and rankings are tilted by the sampling
      density. Mismatched proposers are the most common silent failure.
    - **Paired-only negatives estimate ratios between hypotheses.** With `negative_target=None`
      the training signal is the ranking of candidate hypotheses for a fixed observation. That is
      exactly what MLE ranking and likelihood-ratio tests need, but it is not a normalized
      posterior; obtaining posterior weights requires the prior and, for posterior sampling,
      further inference steps.
    - **Finite candidate sets and calibration.** The balanced binary objective targets the
      log-ratio exactly at its optimum (when negatives follow the reference prior) but assumes
      balanced classes in expectation; the multiclass variant's optimum reaches the same target
      only as $K \to \infty$. With very few negatives or strongly imbalanced negative sampling,
      the logits are biased estimates of the log-ratio; choose $K$ generously and check that
      scores are not saturated. Sigmoid probabilities are calibrated only if the classifier is
      Bayes-optimal and the class balance matches training.
    - **Coverage and extrapolation.** Candidates outside the training prior's support are
      extrapolation: the network has never seen such contexts and the ratio is unreliable. When
      importance-reweighting from a broader proposal, monitor the effective sample size of the
      weights.
    - **Classifier capacity matters.** The classifier must learn the ratio from the raw
      concatenation `concat([context, observable])`; structure such as $|x - \theta|$ or a ratio
      peaked at a particular parameter value can require more capacity than a shallow MLP on a
      small training set. Increase width/depth or supply informative features when the ranking is
      poor. Ranking only needs a monotone transform of the true ratio, so exact calibration is not
      required for the argmax.
    - **Not a UQ method by itself.** NRE provides likelihood ratios, not predictive intervals or
      epistemic/aleatoric decompositions. Combine with [conformal prediction](../methods/conformal/index.md)
      and the calibration diagnostics above when downstream uncertainty is needed.

---

## References

| # | Reference |
|:-:|:----------|
| 1 | K. Cranmer, J. Pavez, G. Louppe. ["Approximating Likelihood Ratios with Calibrated Probabilistic Classifiers."](https://arxiv.org/abs/1506.02169) *arXiv:1506.02169*, **2015**. |
| 2 | C. Durkan, I. Murray, G. Papamakarios. ["On Contrastive Learning for Likelihood-free Inference."](https://arxiv.org/abs/2002.03712) *ICML*, **2020**. |
| 3 | B. K. Miller, C. Weniger, P. Forré. ["Contrastive Neural Ratio Estimation for Simulation-based Inference."](https://arxiv.org/abs/2210.06170) *NeurIPS*, **2022**. |
| 4 | B. K. Miller, A. Cole, P. Forré, G. Louppe, C. Weniger. ["Truncated Marginal Neural Ratio Estimation."](https://arxiv.org/abs/2107.01214) *NeurIPS*, **2021**. |
| 5 | A. van den Oord, Y. Li, O. Vinyals. ["Representation Learning with Contrastive Predictive Coding."](https://arxiv.org/abs/1807.03748) *arXiv:1807.03748*, **2018**. |
| 6 | I. Elsharkawy, Y. Kahn. ["Contrastive Normalizing Flows for Uncertainty-Aware Parameter Estimation."](https://arxiv.org/abs/2505.08709) *arXiv:2505.08709*, **2025** — related classifier-based parameter estimation with learned flow embeddings, not a ratio-classifier loss. |
| 7 | S. Benevedes, J. Thaler. ["Frequentist Uncertainties on Neural Density Ratios with wifi Ensembles."](https://arxiv.org/abs/2506.00113) *Phys. Rev. D* 112, 056024, **2025** — `WifiRatioEnsemble`. |
| 8 | B. Gong, F. J. Samaniego. "Pseudo Maximum Likelihood Estimation: Theory and Applications." *Ann. Statist.* 9(4):861–869, **1981**. (Variance of two-step estimators.) |

---

## Next steps

- [Normalizing flow losses](nflows.md) — the flow-based alternative, including `ContrastiveFlowLoss`
- [Loss catalogue](index.md) — where NRE sits among regression objectives
- [Gaussian losses](gaussian.md) — plain NLL when a normalized Gaussian likelihood is acceptable
- [Likelihood-ratio posterior utilities](../api/inference.md) — turn log-ratios into posterior weights, effective sample sizes and MLE/MAP estimates
- [SBC and TARP](../metrics/calibration.md#simulation-based-calibration-sbc-and-tarp) — validate posterior calibration before reporting intervals
- [Calibration metrics](../metrics/calibration.md) — diagnose ratio/logit calibration downstream
- [Conformal prediction](../methods/conformal/index.md) — distribution-free prediction intervals
