# API Reference

Complete public API surface for **torchregress**. Every page in this section is a
**standalone reference** — you can read it without any auto-generation tools.
For conceptual guides, derivations, and worked examples, see
[User Guide](../guide/index.md), [Methods & Algorithms](../methods/index.md), and
[Examples](../examples/index.md).

## Modules

| Module | What's in it | Reference |
|:-------|:-------------|:----------|
| `torchregress.losses` | Regression losses: point, robust, Gaussian, quantile, ordinal, censored, Poisson/Tweedie, EIV, conformal, … | [Losses API](losses.md) |
| `torchregress.ensemble` | Deep ensembles, heteroscedastic ensembles, batch ensembles, SWAG, MC-dropout, BNN | [Ensemble API](ensemble.md) |
| `torchregress.losses.conformal` | Conformal predictors & `ConformalLoss` | [Conformal API](conformal.md) |
| `torchregress.algorithms` | IRLS, SIMEX, RC, LatentNN, TicTac, heteroscedastic Laplace, IVON | [Algorithms API](algorithms.md) |
| `torchregress.metrics` | Point, distribution, interval, calibration, OOD, ensemble, multivariate, ordinal, censored, decision | [Metrics API](metrics.md) |
| `torchregress.calibration` | Variance temperature, isotonic mean, PIT, semi-conformal, label shift | [Calibration API](calibration.md) |
| `torchregress.constraints` | Output-head constraints (non-negative, bounded, non-crossing, simplex, spectral norm) | [Constraints API](constraints.md) |
| `torchregress.inference` | Prediction-powered inference (PPI) | [Inference API](inference.md) |
| `torchregress.causal` | Doubly-robust ATE/CATE, overlap diagnostics, policy value | [Causal API](causal.md) |
| `torchregress.viz` | Diagnostic, monitoring, results, and utility plotting | [Visualization API](viz.md) |
| `torchregress.test_time` | Bayesian linear heads, OT shift conformal, BLR predictive adapters | [Test-time API](test_time.md) |
| `torchregress.semi_supervised` | Teacher–student semi-supervised trainer, consensus/agreement, pseudo-loss, trust weighting | [Semi-supervised API](semi_supervised.md) |
| `torchregress.comparison` | Reproducible comparison-example helpers (seeds, timing, metrics, JSON summaries, fairness notes) | [Comparison API](comparison.md) |
| `torchregress.prediction` | Predictive batch containers (`PredictiveBatch`) | [Inference API](inference.md) |
| `torchregress.utils` | Tensor ops, validation, augment, labels, propensity, transforms | [Utilities API](utils.md) |

## Top-level exports

| Symbol | Description |
|:-------|:------------|
| `losses` | Lazy submodule of regression losses (point, robust, Gaussian, quantile, ordinal, censored, count, EIV, conformal). |
| `metrics` | Lazy submodule of evaluation metrics (point, distribution, interval, calibration, OOD, ensemble, decision). |
| `algorithms` | Lazy submodule of error-in-variables and robust algorithms (IRLS, SIMEX, RC, LatentNN, TicTac, IVON). |
| `ensemble` | Lazy submodule of deep ensembles, MC-dropout, SWAG, snapshot and Bayesian model building blocks. |
| `semi_supervised` | Lazy submodule of teacher–student semi-supervised training. |
| `test_time` | Lazy submodule of test-time adaptation (Bayesian heads, label shift, OT conformal reweighting, alignment). |
| `method_catalog` | Lazy submodule exposing scriptable method metadata. |
| `inference` | Lazy submodule of prediction-powered inference (PPI) estimators. |
| `constraints` | Lazy submodule of output-head constraints (non-negative, bounded, non-crossing, simplex, spectral norm). |
| `calibration` | Lazy submodule of post-hoc calibrators and shift estimators. |
| `causal` | Lazy submodule of doubly-robust causal estimation and overlap diagnostics. |
| `prediction` | Lazy submodule of predictive batch containers (`PredictiveBatch`). |
| `viz` | Lazy submodule of diagnostic, monitoring, results, and utility plotting. |
| `comparison` | Lazy submodule of reproducible comparison-example helpers (seeds, timing, metrics, JSON summaries). |
| `utils` | Lazy submodule of tensor ops, validation, augmentations, labels, propensity, and transforms. |
| `__version__` | Installed **torchregress** version string. |

## Package structure

```text
torchregress/
├── losses/                # Regression and distributional losses
│   ├── base.py            # BaseLoss, RegressionLoss, DistributionLoss, weighted wrappers
│   ├── gaussian.py        # GaussianNLL, FaithfulGaussian, MultivariateGaussian, LowRankGaussian
│   ├── beta_nll.py        # BetaNLL
│   ├── gaussian_wasserstein.py
│   ├── faithful_gaussian.py
│   ├── robust.py          # Huber, Cauchy, Tukey, Barron, Adaptive, CVaR, LogCosh, Charbonnier, PseudoHuber
│   ├── quantile.py        # Quantile, MultiQuantile, QuantileCrossover
│   ├── expectile.py       # Expectile, MultiExpectile, AsymmetricLeastSquares
│   ├── conformal.py       # SplitConformal, CQR, UACQR, Density, MonteCarlo, Local, CTI, SLS, ConformalLoss
│   ├── censored.py        # CensoredGaussianNLL, CensoredQuantile, AFT
│   ├── poisson.py         # PoissonDeviance, NegativeBinomial, ZeroInflatedPoisson
│   ├── poisson_gaussian.py
│   ├── tweedie.py         # Tweedie, Gamma, InverseGaussian, CompoundPoisson
│   ├── ordinal.py         # CORAL, CumulativeLink, OrdinalCrossEntropy
│   ├── mdn.py              # MixtureDensityLoss
│   ├── nflows.py           # NormalizingFlowLoss, ContrastiveFlowLoss
│   ├── imbalanced.py      # Density, Focal, LDS, Propensity, BMC, BalancedMSE
│   ├── uncertain_gt.py    # NoisyTargetGaussianNLL, PseudoLabel, Consistency
│   ├── eiv.py             # FunctionalEIV, StructuralEIV, ODR, EnsembleEIV
│   ├── transforms.py      # Log, BoxCox, Sqrt, YeoJohnson, TransformedTarget
│   ├── balanced_mse.py
│   ├── evidential.py      # EvidentialRegression
│   └── loss_registry.py
├── metrics/               # Evaluation metrics
│   ├── point.py           # mse, rmse, mae, r2_score, huber_loss, mape, msle, ev_score
│   ├── distribution.py    # crps_gaussian, gaussian_nll, energy_score, PIT
│   ├── interval.py        # interval_score, PICP, MPIW
│   ├── calibration.py     # ECE, MCE, PIT-based calibration
│   ├── ensemble.py        # GaussianNLLEnsemble, uncertainty_decomposition
│   ├── ood.py             # Mahalanobis, Typicality, Entropy, KDE
│   ├── multivariate.py    # MultivariateRMSE, MultivariateMAE
│   ├── ordinal.py         # ordinal_accuracy, QWK
│   ├── censored.py        # concordance_index, censoring_rate
│   ├── decision.py        # RiskCoverageCurve, RejectionPolicy
│   ├── tac.py             # Task-agnostic correlations
│   └── uncertain.py       # noisy_target_gaussian_nll, consistency_error
├── calibration/           # Post-hoc transforms and shift estimators
├── ensemble/              # Ensemble models and building blocks
├── algorithms/            # IRLS, SIMEX, RC, LatentNN, TicTac, IVON
├── test_time/             # Bayesian linear heads, OT shift, BLR adapters
├── inference/             # Prediction-powered inference
├── constraints/           # Output-head constraints
├── comparison.py          # Comparison-example helpers (seeds, JSON summary)
├── prediction.py          # Predictive batch containers
├── viz/                   # Diagnostic, monitoring, results, utils plots
├── semi_supervised.py     # TeacherStudentTrainer
└── utils/                 # tensor_ops, validation, augment, labels, propensity, transforms
```

## Core imports

The library uses **lazy submodule imports** — accessing a submodule via
attribute (`tr.losses`) is the only cost; it does not load the others.

```python
import torchregress as tr

# Top-level, eagerly imported
from torchregress import BaseLoss, RegressionLoss, DistributionLoss
from torchregress.algorithms import iteratively_reweighted_least_squares

# Lazy-loaded submodules
losses = tr.losses            # tr.losses.WeightedHuberLoss(...)
metrics = tr.metrics          # tr.metrics.mse(y_pred, y)
ensemble = tr.ensemble        # tr.ensemble.BaseEnsembleModel(...)
calibration = tr.calibration  # tr.calibration.VarianceTemperatureScaler()
viz = tr.viz                  # tr.viz.set_style(); tr.viz.plot_residuals(...)
constraints = tr.constraints
causal = tr.causal
inference = tr.inference
algorithms = tr.algorithms
test_time = tr.test_time
utils = tr.utils
comparison = tr.comparison
prediction = tr.prediction
semi_supervised = tr.semi_supervised
method_catalog = tr.method_catalog  # Scriptable method metadata
```

## Versioning & stability

- The library is on `0.x` — minor versions may include API changes.
- Mature/stable APIs are tagged `Core` in the [Method Catalog](../reports/method_catalog_generated.md).
- `Advanced` methods are powerful but require more careful validation before adoption.
