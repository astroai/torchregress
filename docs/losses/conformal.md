# Conformal Prediction

Conformal prediction produces **prediction intervals with finite-sample coverage guarantees** — no distributional assumptions required. Unlike parametric uncertainty methods (Gaussian NLL, ensembles) which estimate density, conformal prediction directly controls the *frequency* of coverage errors via a held-out calibration set.

!!! info "Conformal prediction is a methodology, not a loss function"
    The `ConformalLoss` wrapper and individual predictors (`SplitConformal`, `CQR`, `UACQR`, etc.) are exported from `torchregress.losses`, but the methodology spans calibration, predictors, and distributional variants. Full documentation is in the methods section:

    - **[Conformal Prediction Overview](../methods/conformal/index.md)** — methodology intro, comparison table, decision tree
    - **[Predictors](../methods/conformal/predictors.md)** — `SplitConformal`, `CQR`, `DensityConformal`, `MonteCarloConformal`, etc.
    - **[Distributional Conformal](../methods/conformal/distributional.md)** — `DistributionalConformal`, `CTI`

See the [Predictors page](../methods/conformal/predictors.md) for the recommended standalone API. For a quickstart, see [§4 of the Quickstart](../getting-started/quickstart.md#4-conformal-prediction-for-guaranteed-coverage).

## References

| # | Reference |
|:-:|:----------|
| 1 | Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. |
| 2 | Romano, Y., Patterson, E. & Candès, E. (2019). Conformalized Quantile Regression. *NeurIPS*. |
| 3 | Angelopoulos, A. N. & Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. *arXiv:2107.07511*. |
