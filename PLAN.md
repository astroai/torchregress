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
   - [ ] Optimize `eiv_standard.py` to be standalone like other non-eiv losses.
   - [ ] Keep only `eiv.py` which will contain the `eiv_standard.py` logic and make an eiv branch of the other eiv losses
   - [ ] Update parameter ordering
   - [ ] Add comprehensive docstrings
   - [ ] Fix numerical stability issues
   - Files to change: `losses/eiv.py`, `losses/eiv_*.py`, 
   - Priority: **MEDIUM**

### 4. Testing

11. **Complete Test Coverage**
    - [ ] Ensure tests for basic cases for all losses
    - [ ] Add tests for edge cases (NaN, inf, extreme values)
    - [ ] Test numerical stability and gradient flow
    - Files to change: Test files for each loss
    - Priority: **HIGH**

12. **Add Test for API Consistency**
    - [ ] Create test that checks parameter ordering
    - [ ] Ensure proper inheritance from base classes
    - [ ] Test for consistent reduction behavior
    - Files to create: `tests/test_api_consistency.py`
    - Priority: **MEDIUM**

### 5. Package Infrastructure

13. **Update Package Metadata**
    - [ ] Complete package version information
    - [ ] Update README with completed features
    - [ ] Ensure proper imports in `__init__.py` files
    - Files to change: `setup.py`, `README.md`, `__init__.py` files
    - Priority: **MEDIUM**

14. **Continuous Integration Setup**
    - [ ] Set up CI workflow for tests
    - [ ] Add code coverage reporting
    - [ ] Configure automatic documentation building
    - Files to create/change: CI configuration files
    - Priority: **LOW**

## File Change Order (Implementation Sequence)

1. `utils/validation.py` - Foundation for all validation
2. `utils/tensor_ops.py` - Core tensor operations
3. `losses/base.py` - Base classes and common functionality
4. `losses/gaussian.py` - Most commonly used losses
5. `losses/robust.py` - Important robust alternatives
6. `losses/quantile.py` & `losses/expectile.py` - Distribution-free alternatives
7. `losses/poisson.py` & `losses/tweedie.py` - Special cases
8. `losses/rag.py`, `losses/mdn.py`, `losses/nflows.py`, `losses/eiv.py`: Advanced losses with uncertainty handling
9. Test files for each modified loss
10. Example files
11. Package metadata files

## Deferring to Later Releases

The following items are explicitly deferred to post-0.1.0 releases:

1. Loss builder system
2. Registration system for custom losses
3. Sophisticated Error-in-Variables (EIV) losses
4. PyTorch Lightning integration
5. Advanced loss functions (Barron, censored regression, etc.)
6. Ensemble and uncertainty estimation tools