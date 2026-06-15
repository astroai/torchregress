# Conformal Prediction

!!! note "Section moved"
    Conformal prediction is a **methodology**, not a loss function. It has been moved to its own section. See:

    - **[Conformal Prediction Overview](../methods/conformal/index.md)** — Methodology intro, comparison table, decision tree
    - **[Predictors](../methods/conformal/predictors.md)** — SplitConformal, CQR, DensityConformal, MonteCarloConformal, etc.
    - **[Distributional Conformal](../methods/conformal/distributional.md)** — DistributionalConformal, CTI, R2CConformal

The `ConformalLoss` wrapper is retained for backward compatibility. See the [ConformalLoss API](../api/losses.md#conformalloss), [`SplitConformal`](../api/losses.md#splitconformal), [`CQR`](../api/losses.md#cqr), and [`UACQR`](../api/losses.md#uacqr) reference sections, and the new [Predictors page](../methods/conformal/predictors.md) for the recommended standalone API.
