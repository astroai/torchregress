# torchregress

<div style="text-align: center; margin: 2em 0;" markdown>

**A PyTorch library for regression with built-in uncertainty estimation**

[:material-book-open-outline: Getting Started](../usage/quickstart.md){ .md-button .md-button--primary }
[:material-format-list-bulleted: Loss Functions](../losses/index.md){ .md-button }
[:material-github: GitHub](https://github.com/sfabbro/torchregress){ .md-button }

</div>

---

## What is torchregress?

torchregress extends PyTorch with **60+ loss functions**, **ensemble methods**, **conformal prediction**, and **evaluation metrics** purpose-built for regression tasks.  It addresses the challenges that practitioners face daily:

- **How uncertain is my prediction?** — Gaussian NLL, ensembles, evidential regression, MDN, normalizing flows
- **How do I handle outliers?** — Huber, Cauchy, Tukey biweight, CVaR losses
- **How do I get prediction intervals?** — Quantile regression, conformal prediction with coverage guarantees
- **My targets are noisy / censored / ordinal / imbalanced** — dedicated loss families for each
- **My inputs have measurement error** — EIV losses, RC, SIMEX algorithms

---

## Learning Paths

=== ":fontawesome-solid-graduation-cap: New to regression UQ"

    1. [Quick Start](../usage/quickstart.md) — fit your first model
    2. [Core Concepts](../guides/concepts.md) — uncertainty types, method overview
    3. [User Guide](user_guide.md) — task-oriented walkthrough
    4. [Basic Usage Example](../examples/basic_usage.md)

=== ":fontawesome-solid-flask: Experienced practitioner"

    1. [Method Selection Matrix](../guides/method_selection_matrix.md) — find the right loss
    2. [Losses Overview](../losses/index.md) — browse the full catalogue
    3. [Ensemble & UQ](../ensemble/index.md) — uncertainty decomposition
    4. [Conformal Prediction](../conformal/index.md) — coverage guarantees

=== ":fontawesome-solid-microscope: Statistician / researcher"

    1. [Mathematical Foundations](../math/index.md) — notation, NLL duality, proofs
    2. [Conformal Theory](../conformal/index.md#mathematical-deep-dive) — exchangeability, coverage proof
    3. [Comparative Evidence Matrix](../guides/comparative_evidence_matrix.md) — empirical benchmarks
    4. [API Reference](../api/index.md) — complete function signatures

---

## Feature Highlights

| Category | What's included |
|:---------|:---------------|
| **60+ loss functions** | Gaussian, robust, quantile, ordinal, censored, Poisson, MDN, flows, EIV, imbalanced |
| **Ensemble UQ** | Deep Ensemble, BatchEnsemble, SWAG, MC-Dropout, BNN — with aleatoric/epistemic decomposition |
| **Conformal prediction** | SplitConformal, CQR, CTI, DistributionalConformal — finite-sample coverage guarantees |
| **Algorithms** | IRLS, Regression Calibration, SIMEX — measurement error correction |
| **Post-hoc calibration** | Variance temperature scaling, isotonic calibration, PIT calibration |
| **Causal inference** | Doubly-robust ATE/CATE estimators, overlap diagnostics |
| **50+ metrics** | Point, distribution, interval, calibration, OOD, ordinal, censored, ensemble metrics |
| **Constraints** | Bounded heads, monotonicity, simplex, spectral norm |
