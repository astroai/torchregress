# torchregress: TODO List

This document outlines planned improvements for the torchregress library to enhance its API, maintainability, and functionality.

## API Consistency & Usability

- [x] **Standardize loss function interfaces**:
  - [x] Unify parameter ordering across all losses (`forward(y_pred, y_true, ...)` to match PyTorch convention)
  - [x] Standardize parameter naming (e.g., consistent use of "target" vs "y_true")
  - [x] Ensure consistent docstring format and examples

- [x] **Simplified base classes hierarchy**:
  - [x] Review current `MaskedLoss`, `RegressionLoss`, `DistributionLoss` inheritance
  - [x] Create clearer specialization with flatter hierarchy
  - [x] Document class relationships and intended use cases

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

- [x] **Batched implementations**:
  - [x] Optimize Monte Carlo sampling through proper vectorization
  - [x] Avoid unnecessary for-loops in EIV implementation
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

- [x] **Torch.compile compatibility**:
  - [x] Ensure all operations are compatible with torch.compile()
  - [x] Add dynamo support for faster compilation
  - [x] Create benchmarks comparing compiled vs non-compiled performance

## Missing Features & Use Cases

- [x] **Heteroscedastic regression abstractions**:
  - [x] High-level API for variance prediction alongside means
  - [x] Support for joint optimization of mean and variance networks
  - [x] Variance network utilities and implementations

- [x] **Modern regression losses**:
  - [x] Add distribution-free uncertainty losses (e.g., SQR, simultaneous quantile regression)
  - [x] Implement Barron loss (generalization of L1/L2)
  - [x] Support for conformal prediction methods
  - [x] Add DeepAR-style autoregressive losses

- [x] **Target transformation losses**:
  - [x] Built-in support for log/box-cox transformations within losses
  - [x] Variance stabilizing transformations for heteroscedastic data
  - [x] Inverse transformation handling for prediction

- [ ] **Censored regression support**:
  - [ ] Add losses for interval-censored and right-censored data
  - [ ] Survival regression building blocks
  - [ ] Time-to-event prediction metrics

- [x] **Ensemble uncertainty estimation**:
  - [x] Built-in support for deep ensembles with uncertainty propagation
  - [x] Simplified API for combining multiple models' predictions
  - [x] Ensemble-specific metrics and aggregation methods

- [x] **Multivariate regression testing metrics**:
  - [x] Add metrics specific to multivariate outputs (beyond treating dimensions independently)
  - [ ] Support for evaluating joint distributions
  - [x] Multi-output calibration metrics

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

1. **Update Validation Utilities** (`torchregress/utils/validation.py`)
   - [ ] Complete existing validation functions
   - [ ] Add validation for uncertainty representations (scalar/vector/matrix)
   - [ ] Create consistent error messages
   - [ ] Add comprehensive docstrings

2. **Enhance Tensor Operations** (`torchregress/utils/tensor_ops.py`) 
   - [ ] Implement optimized masked operations
   - [ ] Add batching support for computationally intensive operations
   - [ ] Create utilities for broadcasting uncertainty representations
   - [ ] Add tests comparing performance of different implementations

3. **Add PyTorch Compatibility Utilities** (`torchregress/utils/pytorch_compat.py`)
   - [x] Implement torch.compile compatibility checks
   - [x] Add dynamo support functions
   - [x] Create performance benchmarking tools

### 2. Base Loss Framework

4. **Refactor Loss Base Classes** (`torchregress/losses/base.py`)
   - [ ] Standardize parameter order (y_pred, y_true) in all forward() methods
   - [ ] Simplify inheritance hierarchy
   - [ ] Add comprehensive docstrings with examples
   - [ ] Create clear separation of responsibilities between classes

5. **Implement Loss Registration System** (`torchregress/losses/loss_registry.py`)
   - [ ] Complete register_loss() implementation
   - [ ] Add decorator API for registration
   - [ ] Create discovery mechanism
   - [ ] Add robust error handling

### 3. Standard Loss Functions

6. **Update Basic Losses** (`torchregress/losses/torch_extensions.py`, `torchregress/losses/gaussian.py`)
   - [ ] Standardize interfaces
   - [ ] Use common validation utilities
   - [ ] Register with loss registry
   - [ ] Add tests for edge cases

7. **Enhance Advanced Losses**
   - [ ] `torchregress/losses/robust.py`: Standardize interfaces, add missing robust losses
   - [ ] `torchregress/losses/quantile.py`: Update parameter naming, optimize implementations
   - [ ] `torchregress/losses/expectile.py`: Standardize to match quantile losses
   - [ ] `torchregress/losses/tweedie.py`, `torchregress/losses/poisson.py`: Update interfaces

8. **Improve Distribution Losses**
   - [ ] `torchregress/losses/mdn.py`: Standardize interface, add better docstrings
   - [ ] `torchregress/losses/nflows.py`: Update parameter order, add examples
   - [ ] `torchregress/losses/categorical.py`: Standardize interface

### 4. Error-in-Variables System

9. **Optimize EIV Utilities** (`torchregress/losses/eiv/eiv_utils.py`)
   - [x] Implement batched Monte Carlo sampling
   - [ ] Standardize uncertainty representation
   - [ ] Add automatic batch size determination
   - [ ] Optimize utility functions

10. **Update EIV Implementations**
    - [ ] `torchregress/losses/eiv/eiv_standard.py`: Standardize interfaces
    - [ ] `torchregress/losses/eiv/eiv_rfit.py`: Optimize and standardize
    - [ ] `torchregress/losses/eiv/eiv_quantile.py`: Update to match base quantile loss
    - [ ] `torchregress/losses/eiv/eiv_mdn.py`: Standardize interface
    - [ ] `torchregress/losses/eiv/eiv_chamfer.py`: Optimize and standardize

### 5. Metrics System

11. **Update Metrics Utilities** (`torchregress/metrics/utils.py`)
    - [x] Add TorchMetrics compatibility layer
    - [x] Implement metric composition utilities 
    - [x] Standardize reduction handling

12. **Enhance Regression Metrics**
    - [ ] `torchregress/metrics/point.py`: Convert to TorchMetrics subclasses
    - [ ] `torchregress/metrics/interval.py`: Standardize interfaces
    - [ ] `torchregress/metrics/distribution.py`: Add missing metrics
    - [ ] `torchregress/metrics/calibration.py`: Improve visualization options
    - [ ] `torchregress/metrics/ood.py`: Standardize interfaces

### 6. Utility Module Integration

13. **Consolidate Label Handling** (`torchregress/utils/labels.py`)
    - [ ] Create standard API for multi-annotator data
    - [ ] Add integration with metrics system
    - [ ] Improve documentation and examples

14. **Enhance Augmentation Framework** (`torchregress/utils/augment.py`)
    - [ ] Add regression-specific augmentations
    - [ ] Create augmentation pipeline class
    - [ ] Add time series augmentations

15. **Improve IRLS Integration** (`torchregress/algorithms/irls.py`)
    - [ ] Create PyTorch Lightning callback
    - [ ] Add interface for common robust regression tasks
    - [ ] Optimize implementation for large datasets

16. **Enhance Ensemble Integration** (`torchregress/ensemble/ensemble.py`)
    - [ ] Add ensemble calibration utilities
    - [ ] Implement prediction combination strategies
    - [ ] Create visualization tools for ensemble predictions
    - [x] Rename `*_utils.py` modules to base names (transform.py, histogram.py, quantile.py, irls.py) and adapt all imports

### 7. New Features Implementation

17. **Implement Modern Loss Functions**
    - [x] Create `torchregress/losses/barron.py` for generalized L1/L2 loss
    - [x] Create `torchregress/losses/conformal.py` for conformal prediction
    - [x] Create `torchregress/losses/autoregressive.py` for time series losses

18. **Add Target Transformation Support**
    - [x] Create `torchregress/transforms/target.py` for Box-Cox and other transformations
    - [ ] Implement loss wrappers that handle transformations automatically

19. **Implement Censored Regression**
    - [ ] Create `torchregress/losses/censored.py` for interval-censored and right-censored losses
    - [ ] Add survival regression building blocks
    - [ ] Implement time-to-event metrics

20. **Add PyTorch Lightning Support**
    - [ ] Create `torchregress/lightning/regression.py` with LightningModule implementations
    - [ ] Add regression-specific callbacks
    - [ ] Create example notebooks

21. **Implement Uncertainty Estimation Tools**
    - [x] Create `torchregress/models/heteroscedastic.py` for variance prediction
    - [x] Create `torchregress/models/ensemble.py` for deep ensembles
    - [ ] Add multivariate regression metrics

### 8. Builder System (Last)

22. **Implement Loss Builder System** (`torchregress/losses/builder.py`)
    - [ ] Create fluent API for loss construction
    - [ ] Implement method chaining
    - [ ] Add comprehensive examples and tests
    - [ ] Document all available builder options