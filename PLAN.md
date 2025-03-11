# TorchRegression Implementation Plan

This document outlines the detailed implementation plan for TorchRegression 0.1.0.

## Version 0.1.0 Implementation Tasks

### 1. Core Utilities (Foundation)

1. **Update Validation Utilities** (`utils/validation.py`)
   - [X] Complete existing validation functions
   - [X] Add consistent error messages
   - [X] Add comprehensive docstrings with examples
   - Files to change: `validation.py`
   - Priority: **HIGH**

2. **Enhance Tensor Operations** (`utils/tensor_ops.py`)
   - [X] Improve masked operations
   - [X] Add proper broadcasting support
   - [X] Add comprehensive docstrings
   - Files to change: `tensor_ops.py`
   - Priority: **HIGH**

### 2. Documentation

3. **Generate API Documentation**
   - [ ] Add proper docstrings to all existing loss functions
   - [ ] Include mathematical formulations (using LaTeX)
   - [ ] Document parameter ranges and defaults
   - [ ] Add "See Also" cross-references
   - Files to change: All loss function files
   - Priority: **HIGH**

4. **Create Basic Examples**
   - [ ] Example showing basic usage of each loss
   - [ ] Example demonstrating missing data handling
   - [ ] Example comparing different losses
   - Files to create: `examples/basic_usage.py`, `examples/missing_data.py`, `examples/loss_comparison.py`
   - Priority: **MEDIUM**

### 3. API Consistency

5. **Standardize Base Classes** (`losses/base.py`)
   - [X] Review and simplify class hierarchy
   - [X] Standardize parameter ordering (y_pred first)
   - [X] Implement consistent initialization patterns
   - Files to change: `losses/base.py`
   - Priority: **CRITICAL**

6. **Standardize Parameter Names**
   - [X] Change `y_true`/`target`/`targets` to consistent naming
   - [X] Ensure consistent ordering in all forward methods
   - [X] Add type annotations
   - Files to change: All loss function files
   - Priority: **CRITICAL**

7. **Update Gaussian Losses** (`losses/gaussian.py`)
   - [X] Update parameter ordering
   - [X] Add comprehensive docstrings
   - [X] Fix numerical stability issues
   - Files to change: `losses/gaussian.py`
   - Priority: **HIGH**

8. **Update Robust Losses** (`losses/robust.py`)
   - [X] Update parameter ordering
   - [X] Add comprehensive docstrings
   - [X] Fix numerical stability issues
   - Files to change: `losses/robust.py`
   - Priority: **HIGH**

9. **Update Other Losses**
   - [X] Update `quantile.py`, `expectile.py`, `poisson.py`, `poisson_gaussian.py`, `tweedie.py`, `rag.py`, `mdn.py`, `nflows.py`
   - [X] Standardize interfaces and parameter naming
   - [X] Add comprehensive docstrings
   - [X] Fix numerical stability issues
   - Files to change: Various loss function files
   - Priority: **MEDIUM**

10. **Update EIV Losses**

   - [X] Update `eiv.py` and clean `eiv_utils.py`
   - [X] Update parameter ordering
   - [X] Add comprehensive docstrings
   - [X] Fix numerical stability issues
   - Files to change: `losses/eiv.py`, `losses/eiv_utils.py`, 
   - Priority: **MEDIUM**

11. **Update Ensemble**
   - [X] Update, debug and split `ensemble.py` into different styles of esembling
   - [X] Standardize interfaces and parameter naming
   - [X] Add comprehensive docstrings
   - [X] Fix numerical stability issues
   - Files to change: Various loss function files
   - Priority: **MEDIUM**

### 4. Testing

12. **Complete Test Coverage**
    - [X] Ensure tests for basic cases for all losses
      - [X] `gaussian.py`: MSE, MAE, Gaussian NLL
      - [X] `robust.py`: Huber, Pseudo-Huber, Log-cosh, Cauchy
      - [X] `quantile.py` & `expectile.py`: Quantile and Expectile losses
      - [X] `poisson.py` & `tweedie.py`: Poisson and Tweedie losses
      - [X] `rag.py`, `mdn.py`, `nflows.py`: Advanced mixture/uncertainty losses
      - [X] `eiv.py`: Error-in-variables losses
      - [X] `ensemble/*`: All ensemble methods
    - [X] Add tests for edge cases (NaN, inf, extreme values)
      - [X] Test behavior with zero-valued inputs
      - [X] Test behavior with empty tensors
      - [X] Test with extremely large/small values
      - [X] Test with NaN/Inf values and masks
    - [X] Test numerical stability and gradient flow
      - [X] Test gradient flow through all losses
      - [X] Test numerical stability at extreme values
      - [X] Test backward pass with various reduction modes
    - Files to change: Test files for each loss module (`tests/test_*.py`)
    - Priority: **HIGH**

13. **Add Test for API Consistency**
    - [X] Create test that checks parameter ordering
    - [X] Ensure proper inheritance from base classes
    - [X] Test for consistent reduction behavior
    - Files to create: `tests/test_api_consistency.py`
    - Priority: **MEDIUM**

### 5. Package Infrastructure

14. **Update Package Metadata**
    - [ ] Complete package version information
    - [ ] Update README with completed features
    - [ ] Ensure proper imports in `__init__.py` files
    - Files to change: `setup.py`, `README.md`, `__init__.py` files
    - Priority: **MEDIUM**

15. **Continuous Integration Setup**
    - [ ] Set up CI workflow for tests
    - [ ] Add code coverage reporting
    - [ ] Configure automatic documentation building
    - Files to create/change: CI configuration files
    - Priority: **LOW**

### 6. Deferred to Version 0.2.0

1. **High-Level Wrappers**
   - [ ] Refactor the experimental wrappers to follow the standardized API
   - [ ] Properly document all wrapper functions with examples
   - [ ] Create integration tests for all wrapper functions
   - Files to change: `experimental/wrappers.py` → `wrappers.py`
   - Priority: **MEDIUM**

2. **Loss builder system**
3. **Registration system for custom losses**
4. **Sophisticated Error-in-Variables (EIV) losses**
5. **PyTorch Lightning integration** 
6. **Advanced loss functions (Barron, censored regression, etc.)**
7. **More uncertainty estimation tools**

## File Change Order (Implementation Sequence)

1. `utils/validation.py` - Foundation for all validation
2. `utils/tensor_ops.py` - Core tensor operations
3. `losses/base.py` - Base classes and common functionality
4. `losses/gaussian.py` - Most commonly used losses
5. `losses/robust.py` - Important robust alternatives
6. `losses/quantile.py` & `losses/expectile.py` - Distribution-free alternatives
7. `losses/poisson.py` & `losses/tweedie.py` - Special cases
8. `losses/rag.py`, `losses/mdn.py`, `losses/nflows.py`, `losses/eiv.py`: Advanced losses with uncertainty handling
9. `losses/eiv.py`, standard error-in-variables models
10. `ensemble/ensemble.py`, ensemble algorithms
11. Test files for each modified loss
12. Example files
13. Package metadata files

## Deferring to Later Releases

The following items are explicitly deferred to post-0.1.0 releases:

1. Loss builder system
2. Registration system for custom losses
3. Sophisticated Error-in-Variables (EIV) losses
4. PyTorch Lightning integration
5. Advanced loss functions (Barron, censored regression, etc.)
6. More uncertainty estimation tools