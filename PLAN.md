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
   - [ ] Example comparing different losses
   - [ ] Example on photometric redshift benchmark
   - Files to create: `examples/basic_usage.py`, `examples/loss_comparison.py`, `examples/photoz.py`
   - Priority: **HIGH**

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
    - [X] Complete package version information in pyproject.toml
    - [ ] Update README with completed features
    - [ ] Ensure proper imports in `__init__.py` files
    - Files to change: `pyproject.toml`, `README.md`, `__init__.py` files
    - Priority: **MEDIUM**

15. **Modern Packaging Setup**
    - [X] Update to modern Python packaging using PEP 621 (pyproject.toml)
    - [X] Configure optional dependencies for different use cases
    - [X] Remove redundant files (setup.py, requirements.txt, pytest.ini)
    - Files to change: `pyproject.toml`
    - Priority: **MEDIUM**

### 6. Metrics

16. **Update Core Metrics**
    - [ ] Audit existing metrics code for API consistency
    - [ ] Standardize parameter naming (y_pred first, consistent with losses)
    - [ ] Ensure proper error handling and validation
    - [ ] Add missing type hints and docstrings
    - [ ] Update tests to match revised API
    - Files to change: `metrics/point.py`, `metrics/distribution.py`, `metrics/interval.py`, `metrics/calibration.py`, `metrics/ood.py`, `metrics/utils.py`, `tests/test_metrics.py`
    - Priority: **HIGH**

### 7. Visualization & Documentation

17. **Implement Visualization Tools**
    - [ ] Create diagnostic plotting utilities
      - [ ] Residual plots (scatter, histogram, QQ plots)
      - [ ] Calibration plots (reliability diagrams)
      - [ ] Uncertainty visualization (prediction intervals, ensemble variation)
      - [ ] Distribution comparison plots (predicted vs. actual)
    - [ ] Create training monitoring plots
      - [ ] Learning curve visualization
      - [ ] Validation metric tracking
      - [ ] Early stopping visualization
    - [ ] Create results visualization tools
      - [ ] Performance comparison plots
      - [ ] Parameter sensitivity analysis
      - [ ] Feature importance plots
    - Files to create: `viz/diagnostic.py`, `viz/monitoring.py`, `viz/results.py`, `viz/utils.py`
    - Priority: **MEDIUM**

18. **Create mkdocs Documentation**
    - [ ] Set up mkdocs configuration and structure
      - [ ] Create `mkdocs.yml` configuration file
      - [ ] Configure navigation, theme, and plugins
      - [ ] Set up automatic API documentation generation
    - [ ] Develop comprehensive mathematical documentation
      - [ ] Core loss functions with equations and derivations
      - [ ] Uncertainty estimation techniques
      - [ ] Metrics with proper mathematical notation
      - [ ] Statistical interpretations of different losses and metrics
    - [ ] Create usage documentation
      - [ ] Getting started guide with installation instructions
      - [ ] Basic and advanced usage examples
      - [ ] When to use each loss function
      - [ ] How to interpret different metrics
      - [ ] Recommended combinations for different problems
    - Files to create: `mkdocs.yml`, `docs/index.md`, `docs/losses/index.md`, `docs/metrics/index.md`, `docs/examples/index.md`, `docs/math/formulations.md`, `docs/usage/practical_usage.md`
    - Priority: **HIGH**

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
11. `metrics/*.py` - Review and update all metrics modules
12. `viz/*.py` - Implement visualization utilities
13. Test files for each modified loss and metrics
14. Example files 
15. `mkdocs.yml` - MkDocs configuration
16. `docs/` - Documentation files for mkdocs
17. Package metadata files

## Deferred to Version 0.2.0

The following items are explicitly deferred to version 0.2.0 or later:

1. **High-Level Wrappers**
   - Refactor the experimental wrappers to follow the standardized API
   - Properly document all wrapper functions with examples
   - Create integration tests for all wrapper functions
   - Files to change: `experimental/wrappers.py` → `wrappers.py`

2. **Loss Builder System**
   - Create a flexible system to build custom loss functions
   - Implement composable loss components
   - Add comprehensive documentation and examples

3. **Registration System for Custom Losses**
   - Develop a registry for custom loss functions
   - Create hooks for easy extension
   - Add documentation on how to create and register custom losses

4. **Sophisticated Error-in-Variables (EIV) Losses**
   - Implement advanced EIV models
   - Add specialized handling for different error distributions
   - Create comprehensive tests and examples

5. **PyTorch Lightning Integration**
   - Create Lightning-compatible modules
   - Add examples showing integration with Lightning workflows
   - Ensure compatibility with latest Lightning features

6. **Advanced Loss Functions**
   - Implement Barron loss
   - Add censored regression losses
   - Add other specialized loss functions

7. **More Uncertainty Estimation Tools**
   - Implement additional uncertainty quantification methods
   - Add calibration utilities
   - Create visualization tools for uncertainty

8. **Continuous Integration Setup**
   - Set up CI workflow for tests
   - Add code coverage reporting
   - Configure automatic documentation building
   - Files to create/change: CI configuration files