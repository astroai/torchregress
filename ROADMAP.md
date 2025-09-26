# TorchRegression Roadmap

This document outlines the development roadmap for the TorchRegression library.

## Version 0.1.0 (Current Release)

Primary focus: **API consistency, documentation, and testing**

### Core Functionality
- ✅ Standardized loss function interfaces
- ✅ Complete documentation for existing losses
- ✅ Comprehensive test coverage
- ✅ Basic examples for each loss type

### Included Losses
- ✅ Gaussian losses (MSE, Gaussian NLL variants)
- ✅ Robust losses (L1, Huber, etc.)
- ✅ Quantile losses
- ✅ Expectile losses
- ✅ Poisson losses
- ✅ Tweedie losses
- ✅ Modern regression losses (SQR, Barron, DeepAR)
- ✅ Conformal prediction methods
- ✅ Target transformation losses
- ✅ Ensemble uncertainty estimation

## Version 0.2.0 (Planned)

Primary focus: **Optimization and ecosystem integration**

### Planned Features
- Better masked operations and tensor utilities
- PyTorch ecosystem integration (TorchMetrics, Lightning)
- Batched implementation optimizations
- Registration system for custom losses
- Configurable loss builder system

## Version 0.3.0 (Future)

Primary focus: **Advanced features and specialized losses**

### Planned Features
- More Error-in-Variables (EIV) losses
- Heteroscedastic regression abstractions
- Censored regression and survival analysis
- Multivariate regression metrics
- Time series specific losses

## Completed Enhancements

### Conformal Prediction Methods
- ✅ Basic conformalized quantile regression
- ✅ Adaptive conformal prediction
- ✅ Conformalized quantile regression with multiple quantiles
- ✅ Multi-dimensional conformal prediction

### Calibration Methods
- ✅ Isotonic regression calibration
- ✅ Temperature scaling calibration
- ✅ Multi-dimensional calibration

### Ensemble Methods
- ✅ Bayesian model averaging
- ✅ Stacking ensembles
- ✅ Dynamic ensemble weighting
- ✅ Ensemble calibration

## Current Status

The library is currently in a stable state with comprehensive coverage of modern regression techniques, particularly in the areas of uncertainty quantification and robust regression. All planned features for version 0.1.0 have been implemented.