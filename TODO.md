# TorchRegression TODO List

This document outlines planned improvements and ongoing work for the TorchRegression library.

## High Priority Items

### Ecosystem Integration
- [ ] **TorchMetrics compatibility**: Better align with torchmetrics API for consistent user experience
- [ ] **PyTorch Lightning integration**: Add LightningModule implementations for common regression tasks
- [ ] **Registration system for custom losses**: Add decorator-based registration for user extensions

### Advanced Features
- [ ] **Censored regression support**: Add losses for interval-censored and right-censored data
- [ ] **Time series specific losses**: Implement autoregressive and temporal consistency losses
- [ ] **Multivariate regression testing metrics**: Add metrics specific to multivariate outputs

## Medium Priority Items

### API Improvements
- [ ] **Configurable loss builder system**: Create a fluent API for loss construction
- [ ] **Unified uncertainty representation**: Standardize how uncertainty is represented across the library
- [ ] **Common masked operations module**: Consolidate masked operations into a single utility module

### Documentation & Examples
- [ ] **Comprehensive mkdocs documentation**: Set up mkdocs with API documentation and examples
- [ ] **Advanced examples**: Create examples for conformal prediction, calibration, and ensemble methods
- [ ] **Tutorial notebooks**: Add Jupyter notebooks demonstrating key features

## Low Priority Items

### Utility Enhancements
- [ ] **Label handling consolidation**: Integrate label combination methods with metrics and evaluation
- [ ] **Augmentation framework expansion**: Add specialized regression augmentations
- [ ] **IRLS integration**: Better integrate IRLS with standard loss functions
- [ ] **Ensemble model integration**: Add utilities for combining and calibrating ensemble predictions

## Completed Items

### API Consistency & Usability
- ✅ **Standardize loss function interfaces**: Unify parameter ordering across all losses
- ✅ **Simplified base classes hierarchy**: Review and simplify inheritance structure
- ✅ **Batched implementations**: Optimize Monte Carlo sampling through proper vectorization

### Modern Regression Features
- ✅ **Heteroscedastic regression abstractions**: High-level API for variance prediction
- ✅ **Modern regression losses**: SQR, Barron, conformal prediction, DeepAR
- ✅ **Target transformation losses**: Built-in support for log/box-cox transformations
- ✅ **Ensemble uncertainty estimation**: Built-in support for deep ensembles

### Codebase Improvements
- ✅ **Torch.compile compatibility**: Ensure all operations are compatible with torch.compile()
- ✅ **File organization**: Remove redundant files and improve module structure
- ✅ **Comprehensive testing**: Extensive test coverage for all loss functions

## Implementation Status

### Core Modules
- ✅ **Base loss framework**: Standardized parameter order and inheritance
- ✅ **Loss registration system**: Basic implementation complete
- ✅ **Validation utilities**: Comprehensive validation functions
- ✅ **Tensor operations**: Optimized masked operations

### Loss Functions
- ✅ **Gaussian losses**: MSE, Gaussian NLL variants
- ✅ **Robust losses**: Huber, Log-cosh, Tukey, and others
- ✅ **Quantile & Expectile losses**: Complete implementations
- ✅ **Poisson & Tweedie losses**: Specialized count data models
- ✅ **Error-in-Variables losses**: Advanced uncertainty handling
- ✅ **Modern regression losses**: SQR, Barron, conformal, DeepAR

### Metrics & Evaluation
- ✅ **Point metrics**: Standard regression metrics
- ✅ **Distribution metrics**: Uncertainty-aware evaluation
- ✅ **Interval metrics**: Prediction interval evaluation
- ✅ **Calibration metrics**: Reliability assessment
- ✅ **Out-of-distribution metrics**: Uncertainty quality measures

### Utilities
- ✅ **Validation utilities**: Input checking and error handling
- ✅ **Tensor operations**: Optimized core functions
- ✅ **PyTorch compatibility**: Compile and ecosystem support
- ✅ **Ensemble utilities**: Prediction combination methods

## Future Directions

The library is now in a stable state with comprehensive coverage of modern regression techniques. Future work will focus on ecosystem integration, advanced features for specialized domains, and performance optimizations.
