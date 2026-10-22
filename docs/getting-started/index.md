# Start Here

Welcome to **torchregress** — a PyTorch library for regression with built-in uncertainty estimation.

The library answers one question well: *given inputs $x$, what can be said about the conditional distribution $p(y \mid x)$ rather than only its mean?* Every feature is a way to represent, train, calibrate, or evaluate some part of that distribution:

| You want | Representation | Where to start |
|:---------|:---------------|:---------------|
| A point prediction | $\hat{y}(x)$ | [Standard losses](../losses/index.md#standard-losses) |
| A mean **and** a variance | $\mathcal{N}(\hat{y}(x), \hat{\sigma}^2(x))$ | [Gaussian losses](../losses/gaussian.md) |
| A full density | $p_\theta(y \mid x)$ (mixture, flow, basis) | [MDN](../losses/mdn.md), [flows](../losses/nflows.md) |
| An interval with a guarantee | $[L(x), U(x)]$ with $P(Y \in [L,U]) \ge 1-\alpha$ | [Conformal prediction](../methods/conformal/index.md) |
| A calibrated uncertainty score | recalibrated $\hat{\sigma}^2(x)$ | [Post-hoc calibration](../methods/calibration.md) |

Point forecasts are a special case (a Dirac distribution), which is why the same NLL reduction and scoring-rule contracts run through the whole library.

---

## Choose Your Path

=== ":material-school: New to regression uncertainty"

    1. [Installation](installation.md) — get set up in 30 seconds
    2. [Quick Start](quickstart.md) — train your first uncertainty-aware model
    3. [Core Concepts](concepts.md) — learn the vocabulary (aleatoric vs epistemic, proper scoring rules, conformal coverage)
    4. [User Guide](../guide/index.md) — task-oriented walkthrough

=== ":material-flask: Experienced practitioner"

    1. [Method Selection Matrix](../guide/method-selection.md) — find the right loss for your problem
    2. [Losses Catalogue](../losses/index.md) — browse the implemented loss families
    3. [Examples](../examples/index.md) — runnable comparison scripts
    4. [API Reference](../api/index.md) — complete signatures

=== ":material-microscope: Statistician / researcher"

    1. [Mathematical Foundations](../guide/math/index.md) — notation, NLL duality, proper scoring rules, proofs
    2. [Methods Reference](../methods/index.md) — conformal theory, ensemble decomposition, EIV algorithms
    3. [Reports](../reports/index.md) — benchmark evidence, adoption audits
    4. [API Reference](../api/index.md) — full function signatures with source
