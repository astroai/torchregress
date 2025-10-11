# torchregress TODO List

This document outlines planned improvements and ongoing work for the library.

## High Priority Items

### Documentation & Examples (Highest User Impact)

- [ ] **Tutorial notebooks**: Add Jupyter notebooks demonstrating key features (uncertainty estimation, robust regression, conformal prediction)
- [ ] **Advanced examples**: Create examples for conformal prediction, calibration, and ensemble methods
- [ ] **Comprehensive API documentation**: Complete mkdocs setup with clear API docs, getting started guide, and migration guide

**Rationale:** Users can't effectively use features they don't understand. Good documentation has the highest ROI for adoption.

### Core Functionality Gaps

- [ ] **Censored regression support**: Add losses for interval-censored and right-censored data (common in survival analysis, astronomy)
- [ ] **Unified uncertainty representation**: Standardize how epistemic/aleatoric uncertainty is represented and decomposed across methods
- [ ] **Better ensemble utilities**: Add tools for combining predictions, uncertainty decomposition, and calibration across ensemble members

**Rationale:** These fill important gaps for scientific/industrial users (censored data is very common, uncertainty decomposition is critical for decision-making).

### API Quality of Life

- [ ] **Registration system for custom losses**: Add decorator-based registration (`@register_loss`) for user extensions
- [ ] **Configurable loss builder system**: Fluent API for loss construction (e.g., `Loss.mse().with_mask().with_weights()`)
- [ ] **Better error messages**: Add input validation with clear, actionable error messages

**Rationale:** Makes the library easier to extend and debug, reducing friction for advanced users.

## Medium Priority Items

### Ecosystem Integration

- [ ] **PyTorch Lightning integration**: Provide LightningModule templates for common regression workflows
- [ ] **TorchMetrics compatibility**: Align metric APIs with torchmetrics conventions for consistency

**Rationale:** Nice to have, but not critical - users can already integrate manually.

### Performance & Testing

- [ ] **Benchmark suite**: Create performance benchmarks for all loss functions and models
- [ ] **Numerical stability tests**: Add comprehensive tests for edge cases (NaN, Inf, very small/large values)
- [ ] **GPU optimization**: Profile and optimize critical paths for GPU performance

**Rationale:** Important for production use but not blocking for research/prototyping.

## Lower Priority Items

### Nice-to-Have Features

- [ ] **Augmentation framework expansion**: Add specialized regression augmentations (mixup for regression, etc.)
- [ ] **Label handling consolidation**: Integrate label combination methods with metrics and evaluation

**Rationale:** Useful but niche - most users can implement themselves or use external libraries.

