# TorchRegression TODO List

This document outlines planned improvements and ongoing work for the TorchRegression library.

## Recently Completed (2025-10-08)

See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for details.

- [x] **SWAG/MultiSWAG implementation**: Bayesian uncertainty estimation via weight averaging
- [x] **TorchCP integration**: Modern conformal prediction methods (Split, CQR, ACI)
- [x] **Uncertainty decomposition documentation**: Clarified which methods support epistemic/aleatoric split
- [x] **Fixed conditional imports**: All imports now direct per CLAUDE.md policy
- [x] **Updated dependencies**: torchcp and zuko now required dependencies

## High Priority Items

### Ecosystem Integration
- [ ] **TorchMetrics compatibility**: Better align with torchmetrics API for consistent user experience
- [ ] **PyTorch Lightning integration**: Add LightningModule implementations for common regression tasks
- [ ] **Registration system for custom losses**: Add decorator-based registration for user extensions

### Advanced Features
- [ ] **Noisy labels for regression**: Implement NoiseAdaptiveLoss, CoTeachingLoss, RENTLoss
- [ ] **Imbalanced regression**: DensityWeightedLoss, LDSLoss with calibration warnings
- [ ] **Evidential regression**: Normal-Inverse-Gamma loss for uncertainty
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

