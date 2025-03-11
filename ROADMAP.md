# TorchRegression Roadmap

This document outlines the high-level development roadmap for the TorchRegression library.

## Version 0.1.0 (Initial Release)

Primary focus: **API consistency, documentation, and testing**

### Core Functionality
- Standardized loss function interfaces
- Complete documentation for existing losses
- Comprehensive test coverage
- Basic examples for each loss type

### Included Losses
- Gaussian losses (MSE, Gaussian NLL variants)
- Robust losses (L1, Huber, etc.)
- Quantile losses
- Expectile losses
- Poisson losses
- Tweedie losses

## Version 0.2.0

Primary focus: **Optimization and ecosystem integration**

- Better masked operations and tensor utilities
- PyTorch ecosystem integration (TorchMetrics, Lightning)
- Batched implementation optimizations
- Registration system for custom losses

## Version 0.3.0

Primary focus: **Advanced features and specialized losses**

- More Error-in-Variables (EIV) losses
- Configurable loss builder system
- Heteroscedastic regression abstractions
- Barron loss implementation
- Conformal prediction methods
- Distribution-free uncertainty losses
- Ensemble uncertainty estimation
- Target transformation losses

## Future Releases

- Censored regression and survival analysis
- Multivariate regression metrics
- Time series specific losses