# TorchRegression: TODO List

This document outlines planned improvements for the TorchRegression library to enhance its API, maintainability, and functionality.

## API Consistency & Usability

- [ ] **Standardize loss function interfaces**:
  - [ ] Unify parameter ordering across all losses (`forward(y_pred, y_true, ...)` to match PyTorch convention)
  - [ ] Standardize parameter naming (e.g., consistent use of "target" vs "y_true")
  - [ ] Ensure consistent docstring format and examples

- [ ] **Simplified base classes hierarchy**:
  - [ ] Review current `MaskedLoss`, `RegressionLoss`, `DistributionLoss` inheritance
  - [ ] Create clearer specialization with flatter hierarchy
  - [ ] Document class relationships and intended use cases

- [ ] **Configurable loss builder system**:
  - [ ] Create a fluent API for loss construction
  - [ ] Example: `loss = RegressionLossBuilder().with_quantile(0.9).with_eiv(sigma_x=0.1).build()`
  - [ ] Support chaining of multiple loss components

- [ ] **Registration system for custom losses**:
  - [ ] Add decorator-based registration for user extensions
  - [ ] Example: `@register_regression_loss("my_custom_loss")`
  - [ ] Create discovery mechanism for registered losses

## Simplifications & Optimizations

- [ ] **Unified uncertainty representation**:
  - [ ] Standardize how uncertainty is represented across the library (scalar, vector, matrix)
  - [ ] Single API for handling heteroscedastic uncertainty
  - [ ] Consistent utilities for covariance preparation

- [ ] **Common masked operations module**:
  - [ ] Consolidate masked operations into a single utility module
  - [ ] Reduce code duplication across loss functions
  - [ ] Optimize masked operations for performance

- [ ] **Batched implementations**:
  - [ ] Optimize Monte Carlo sampling through proper vectorization
  - [ ] Avoid unnecessary for-loops in EIV implementation
  - [ ] Add support for automatic batch size determination

## PyTorch Ecosystem Integration

- [ ] **TorchMetrics compatibility**:
  - [ ] Better align with torchmetrics API for consistent user experience
  - [ ] Implement metrics as TorchMetrics subclasses where appropriate
  - [ ] Support for metric composition and aggregation

- [ ] **PyTorch Lightning integration**:
  - [ ] Add LightningModule implementations for common regression tasks
  - [ ] Implement validation/test step methods with appropriate metrics
  - [ ] Create regression-specific callbacks

- [ ] **Torch.compile compatibility**:
  - [ ] Ensure all operations are compatible with torch.compile()
  - [ ] Add dynamo support for faster compilation
  - [ ] Create benchmarks comparing compiled vs non-compiled performance

## Missing Features & Use Cases

- [ ] **Heteroscedastic regression abstractions**:
  - [ ] High-level API for variance prediction alongside means
  - [ ] Support for joint optimization of mean and variance networks
  - [ ] Variance network utilities and implementations

- [ ] **Modern regression losses**:
  - [ ] Add distribution-free uncertainty losses (e.g., SQR, simultaneous quantile regression)
  - [ ] Implement Barron loss (generalization of L1/L2)
  - [ ] Support for conformal prediction methods
  - [ ] Add DeepAR-style autoregressive losses

- [ ] **Target transformation losses**:
  - [ ] Built-in support for log/box-cox transformations within losses
  - [ ] Variance stabilizing transformations for heteroscedastic data
  - [ ] Inverse transformation handling for prediction

- [ ] **Censored regression support**:
  - [ ] Add losses for interval-censored and right-censored data
  - [ ] Survival regression building blocks
  - [ ] Time-to-event prediction metrics

- [ ] **Ensemble uncertainty estimation**:
  - [ ] Built-in support for deep ensembles with uncertainty propagation
  - [ ] Simplified API for combining multiple models' predictions
  - [ ] Ensemble-specific metrics and aggregation methods

- [ ] **Multivariate regression testing metrics**:
  - [ ] Add metrics specific to multivariate outputs (beyond treating dimensions independently)
  - [ ] Support for evaluating joint distributions
  - [ ] Multi-output calibration metrics

## Utility Module Consolidation

- [ ] **Label handling consolidation**:
  - [ ] Integrate label combination methods with metrics and evaluation
  - [ ] Standardize API for multi-annotator data handling
  - [ ] Add documentation and examples for annotation consensus methods
  - [ ] Support for handling noisy labels in loss functions

- [ ] **Augmentation framework expansion**:
  - [ ] Add specialized regression augmentations (non-linear transformations)
  - [ ] Create pipeline for sequential augmentations
  - [ ] Implement time series specific augmentations
  - [ ] Add API for adaptive/learnable augmentation strategies

- [ ] **IRLS integration**:
  - [ ] Better integrate IRLS with standard loss functions
  - [ ] Add PyTorch Lightning callback for IRLS training
  - [ ] Create simplified API for common robust regression use cases
  - [ ] Add documentation and examples for efficient IRLS usage

- [ ] **Ensemble model integration**:
  - [ ] Add utilities for combining and calibrating ensemble predictions
  - [ ] Implement gradient boosting regression ensemble
  - [ ] Create ensemble metrics and visualization tools
  - [ ] Add efficient inference tools for large ensembles

## Implementation Plan

This section outlines a practical file-by-file approach to implementing the tasks above, addressing dependencies in logical order.

### 1. Core Utilities and Validation

1. **Update Validation Utilities** (`torchregression/utils/validation.py`)
   - [ ] Complete existing validation functions
   - [ ] Add validation for uncertainty representations (scalar/vector/matrix)
   - [ ] Create consistent error messages
   - [ ] Add comprehensive docstrings

2. **Enhance Tensor Operations** (`torchregression/utils/tensor_ops.py`) 
   - [ ] Implement optimized masked operations
   - [ ] Add batching support for computationally intensive operations
   - [ ] Create utilities for broadcasting uncertainty representations
   - [ ] Add tests comparing performance of different implementations

3. **Add PyTorch Compatibility Utilities** (`torchregression/utils/pytorch_compat.py`)
   - [ ] Implement torch.compile compatibility checks
   - [ ] Add dynamo support functions
   - [ ] Create performance benchmarking tools

### 2. Base Loss Framework

4. **Refactor Loss Base Classes** (`torchregression/losses/base.py`)
   - [ ] Standardize parameter order (y_pred, y_true) in all forward() methods
   - [ ] Simplify inheritance hierarchy
   - [ ] Add comprehensive docstrings with examples
   - [ ] Create clear separation of responsibilities between classes

5. **Implement Loss Registration System** (`torchregression/losses/loss_registry.py`)
   - [ ] Complete register_loss() implementation
   - [ ] Add decorator API for registration
   - [ ] Create discovery mechanism
   - [ ] Add robust error handling

### 3. Standard Loss Functions

6. **Update Basic Losses** (`torchregression/losses/torch_extensions.py`, `torchregression/losses/gaussian.py`)
   - [ ] Standardize interfaces
   - [ ] Use common validation utilities
   - [ ] Register with loss registry
   - [ ] Add tests for edge cases

7. **Enhance Advanced Losses**
   - [ ] `torchregression/losses/robust.py`: Standardize interfaces, add missing robust losses
   - [ ] `torchregression/losses/quantile.py`: Update parameter naming, optimize implementations
   - [ ] `torchregression/losses/expectile.py`: Standardize to match quantile losses
   - [ ] `torchregression/losses/tweedie.py`, `torchregression/losses/poisson.py`: Update interfaces

8. **Improve Distribution Losses**
   - [ ] `torchregression/losses/mdn.py`: Standardize interface, add better docstrings
   - [ ] `torchregression/losses/nflows.py`: Update parameter order, add examples
   - [ ] `torchregression/losses/categorical.py`: Standardize interface

### 4. Error-in-Variables System

9. **Optimize EIV Utilities** (`torchregression/losses/eiv/eiv_utils.py`)
   - [ ] Implement batched Monte Carlo sampling
   - [ ] Standardize uncertainty representation
   - [ ] Add automatic batch size determination
   - [ ] Optimize utility functions

10. **Update EIV Implementations**
    - [ ] `torchregression/losses/eiv/eiv_standard.py`: Standardize interfaces
    - [ ] `torchregression/losses/eiv/eiv_rfit.py`: Optimize and standardize
    - [ ] `torchregression/losses/eiv/eiv_quantile.py`: Update to match base quantile loss
    - [ ] `torchregression/losses/eiv/eiv_mdn.py`: Standardize interface
    - [ ] `torchregression/losses/eiv/eiv_chamfer.py`: Optimize and standardize

### 5. Metrics System

11. **Update Metrics Utilities** (`torchregression/metrics/utils.py`)
    - [ ] Add TorchMetrics compatibility layer
    - [ ] Implement metric composition utilities 
    - [ ] Standardize reduction handling

12. **Enhance Regression Metrics**
    - [ ] `torchregression/metrics/point.py`: Convert to TorchMetrics subclasses
    - [ ] `torchregression/metrics/interval.py`: Standardize interfaces
    - [ ] `torchregression/metrics/distribution.py`: Add missing metrics
    - [ ] `torchregression/metrics/calibration.py`: Improve visualization options
    - [ ] `torchregression/metrics/ood.py`: Standardize interfaces

### 6. Utility Module Integration

13. **Consolidate Label Handling** (`torchregression/utils/labels.py`)
    - [ ] Create standard API for multi-annotator data
    - [ ] Add integration with metrics system
    - [ ] Improve documentation and examples

14. **Enhance Augmentation Framework** (`torchregression/utils/augment.py`)
    - [ ] Add regression-specific augmentations
    - [ ] Create augmentation pipeline class
    - [ ] Add time series augmentations

15. **Improve IRLS Integration** (`torchregression/algorithms/irls.py`)
    - [ ] Create PyTorch Lightning callback
    - [ ] Add interface for common robust regression tasks
    - [ ] Optimize implementation for large datasets

16. **Enhance Ensemble Integration** (`torchregression/ensemble/ensemble.py`)
    - [ ] Add ensemble calibration utilities
    - [ ] Implement prediction combination strategies
    - [ ] Create visualization tools for ensemble predictions

### 7. New Features Implementation

17. **Implement Modern Loss Functions**
    - [ ] Create `torchregression/losses/barron.py` for generalized L1/L2 loss
    - [ ] Create `torchregression/losses/conformal.py` for conformal prediction
    - [ ] Create `torchregression/losses/autoregressive.py` for time series losses

18. **Add Target Transformation Support**
    - [ ] Create `torchregression/transforms/target.py` for Box-Cox and other transformations
    - [ ] Implement loss wrappers that handle transformations automatically

19. **Implement Censored Regression**
    - [ ] Create `torchregression/losses/censored.py` for interval-censored and right-censored losses
    - [ ] Add survival regression building blocks
    - [ ] Implement time-to-event metrics

20. **Add PyTorch Lightning Support**
    - [ ] Create `torchregression/lightning/regression.py` with LightningModule implementations
    - [ ] Add regression-specific callbacks
    - [ ] Create example notebooks

21. **Implement Uncertainty Estimation Tools**
    - [ ] Create `torchregression/models/heteroscedastic.py` for variance prediction
    - [ ] Create `torchregression/models/ensemble.py` for deep ensembles
    - [ ] Add multivariate regression metrics

### 8. Builder System (Last)

22. **Implement Loss Builder System** (`torchregression/losses/builder.py`)
    - [ ] Create fluent API for loss construction
    - [ ] Implement method chaining
    - [ ] Add comprehensive examples and tests
    - [ ] Document all available builder options