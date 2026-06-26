# Conformal Prediction

!!! info "Conformal prediction is a methodology, not a loss function"
    The content has moved to its own section. Start here:

    - **[Conformal Prediction Overview](../methods/conformal/index.md)** — methodology intro, comparison table, decision tree
    - **[Predictors](../methods/conformal/predictors.md)** — `SplitConformal`, `CQR`, `DensityConformal`, `MonteCarloConformal`, etc.
    - **[Distributional Conformal](../methods/conformal/distributional.md)** — `DistributionalConformal`, `CTI`

The `ConformalLoss` wrapper and individual predictors (`SplitConformal`, `CQR`, `UACQR`, etc.) are still exported from `torchregress.losses` and documented in the [Losses API](../api/losses.md). See the [Predictors page](../methods/conformal/predictors.md) for the recommended standalone API.
