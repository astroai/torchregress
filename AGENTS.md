# AGENTS.md

This file provides merged guidance to coding agents (Claude, Gemini, Codex) when working in this repository.

## Project Overview

**torchregress** (lowercase) is a PyTorch library providing regression losses, metrics, and utilities with a focus on uncertainty estimation, robust regression, and missing data support.

**Naming Convention:** The library name is "torchregress" (all lowercase).

## Development Commands

This project uses [uv](https://github.com/astral-sh/uv) as the package manager.

### Setup
```bash
uv pip install -e .[all]
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=torchregress --cov-report=html

# Run single test file
uv run pytest tests/losses/test_gaussian.py

# Run specific test
uv run pytest tests/losses/test_gaussian.py::TestGaussianLosses::test_gaussian_nll_loss
```

If `uv` is not available, use the project venv directly:
```bash
.venv/bin/python -m pytest
```

### Code Quality
```bash
# Format code
uv run black .

# Lint code
uv run ruff check .

# Type checking
uv run mypy torchregress
```

### CI parity before push (recommended)

GitHub Actions on `main` runs **ruff + black**, **pytest with coverage**, and **CPU benchmark threshold jobs** (see `.github/workflows/ci.yml`). Match that locally so pushes do not fail CI unexpectedly:

```bash
./scripts/ci_local.sh
```

This installs `test` + `flows` + `dev` extras, then runs **ruff** / **black** on **`src/torchregress`**, **tests**, and **tools**, then **pytest --cov=…** and both **benchmark_smoke** threshold passes (CPU smoke + sweep).

### Pre-commit / pre-push hooks

Fast checks on **commit** (ruff + black + basic file hygiene) and full **CI parity on push**:

```bash
uvx pre-commit install
uvx pre-commit install --hook-type pre-push
```

After this, `git push` runs `./scripts/ci_local.sh` via the pre-push hook (requires `uv` on your `PATH`).

### Documentation
```bash
# Build docs (strict mode catches broken refs)
uv run mkdocs build --strict

# Serve docs locally
uv run mkdocs serve
```

## Documentation Quality Standards

All documentation must stay **synchronized with the codebase**. When modifying code, update the corresponding docs.

### Code-Documentation Sync

- Every **exported class/function** in `__init__.py` must appear in the relevant docs page.
- Never document classes or features that don't exist in code ("phantom classes").
- API usage in examples must match actual call signatures — verify with source.
- When adding/removing exports, update both the specific loss/method page **and** the overview `index.md`.

### Content Requirements

- **Mathematical rigor**: include LaTeX formulas for every loss/metric with the optimisation objective or scoring rule definition.
- **Reference tables**: use `| # | Reference |` / `|:-:|:----------|` header format. Include at least the seminal paper for each method.
- **Comparison tables**: every category page should include a "when to use which" comparison table.
- **Complete examples**: each page should have a self-contained, runnable code example (not just API snippets).
- **Cross-links**: link to related pages (→ See [page](path)), API reference (`[`Class`](../api/module.md#anchor)`), and examples.
- **Decision aids**: use mermaid flowcharts for method selection where appropriate.
- **Admonitions**: use `!!! tip`, `!!! warning`, `!!! info` for practical advice, gotchas, and context.

### Formatting Rules

- Reference tables must have a proper markdown header row (not `...` or bare rows).
- LaTeX: use `$$...$$` for display math, `$...$` for inline math. Verify formulas render correctly.
- Build must pass `mkdocs build --strict` with zero errors before commit.
- Target audience: both ML practitioners and statisticians. Be rigorous but accessible.

### Build & Publish
```bash
# Build distribution
uv build

# Publish to PyPI
uv publish
```

## Architecture

### Core Design Pattern

All losses inherit from a three-tier base class hierarchy in `torchregress/losses/base.py`:

1. **BaseLoss**: Root class providing reduction strategies (`mean`, `sum`, `none`) and mask/weight support
2. **RegressionLoss**: For point prediction losses (MSE, Huber, etc.)
3. **DistributionLoss**: For probabilistic losses that output distribution parameters (Gaussian NLL, MDN, etc.)

### Loss Function Convention

All loss functions follow PyTorch conventions:
- Parameter order: `forward(y_pred, target, mask=None, weights=None, **kwargs)`
- Support for missing data via boolean `mask` parameter (False = missing)
- Support for sample weighting via `weights` parameter
- Reductions handled by `_reduce()` method from BaseLoss

### Module Organization

```
torchregress/
├── losses/          # Loss functions (gaussian, robust, quantile, conformal, etc.)
├── ensemble/        # Ensemble models (DeepEnsemble, BatchEnsemble, etc.)
├── algorithms/      # Training algorithms (IRLS)
├── metrics/         # Evaluation metrics (point, interval, calibration, etc.)
├── utils/           # Utilities (masking, validation, transformations)
└── wrappers.py      # Convenience functions (wrap_pytorch_loss)
```

### Key Abstractions

**`wrap_pytorch_loss()`** (`torchregress/wrappers.py`): Wraps any standard PyTorch loss with torchregress's masking and weighting capabilities.

**WeightedLossWrapper** (`torchregress/losses/base.py`): Wraps any PyTorch loss to add mask and weight support. Convenience subclasses include `WeightedMSELoss`, `WeightedL1Loss`, `WeightedHuberLoss`, etc.

**Ensemble Models** (`torchregress/ensemble/`):
- `DeepEnsemble`: Multiple independently trained models
- `HeteroscedasticEnsembleModel`: Ensembles with aleatoric uncertainty
- `BatchEnsembleLinear`: Efficient batch ensemble layers

**IRLS Algorithm** (`torchregress/algorithms/irls.py`): Iteratively Reweighted Least Squares for robust regression.

### Distribution Parameters

Models that output distributions typically return tuples or concatenated tensors:
- **Gaussian (diagonal)**: `(mean, log_variance)` or concatenated `[mean, log_var]`
- **Gaussian (full covariance)**: `mean` plus `covariance_matrices` passed separately via `MultivariateGaussianLoss`
- **Gaussian (low-rank)**: `mean` plus `cov_factor` and `cov_diag` passed to `LowRankGaussianLoss`
- **MDN**: Raw output containing mixture weights, means, and log-variances
- **Quantile**: Multiple quantile predictions concatenated `[q1, q2, ..., qn]`

Use `create_gaussian_nll()` to pick the appropriate Gaussian loss based on covariance type.
For low-rank heads, `low_rank_output_dim()` and `split_low_rank_gaussian_output()` describe the output layout.

### Error-in-Variables (EIV) Losses

EIV losses treat `y_pred` as noisy inputs (`x_obs`) and require a model reference inside the loss:
- `FunctionalEIVLoss`, `StructuralEIVLoss`, `OrthogonalDistanceRegressionLoss`, `EnsembleEIVLoss`
- Call pattern: `loss_fn(x_obs, y_obs, mask=...)` (not `loss_fn(model(x), y)`).

### Uncertainty Decomposition

**Critical Distinction:** Not all uncertainty methods support epistemic/aleatoric decomposition.

| Method | Epistemic | Aleatoric | Use Case |
|--------|-----------|-----------|----------|
| Heteroscedastic Ensemble | ✅ | ✅ | Decomposed uncertainty with variance prediction |
| MDN (Mixture Density Network) | ✅ | ✅ | Multimodal distributions with decomposition |
| Normalizing Flows (ensemble) | ✅ | ✅ | Flexible distributions via flow ensemble |
| Deep Ensemble | ✅ | ❌ | Epistemic only (unless combined with variance prediction) |
| Quantile Regression | ❌ | ❌ | Distribution-free intervals, no decomposition |
| Conformal Prediction | ❌ | ❌ | Distribution-free coverage guarantees, NOT uncertainty decomposition |
| SWAG/MultiSWAG | ✅ | ⚠️ | Epistemic via weight posterior (aleatoric requires additional modeling) |

**Key Point:** Conformal prediction provides **coverage guarantees**, not uncertainty decomposition. Use it for calibrated intervals, not for separating epistemic/aleatoric uncertainty.

## Configuration

**pyproject.toml settings**:
- Python >= 3.10 required
- Black line length: 100
- Ruff: enforces E (pycodestyle), F (pyflakes), I (isort)
- MyPy: strict typing enabled with `disallow_untyped_defs`

**Test configuration**:
- Tests in `tests/` directory
- Pattern: `test_*.py` files with `test_*` functions
- Warnings for deprecation and user warnings are ignored

## Working with Loss Functions

When adding new loss functions:
1. Inherit from `RegressionLoss` (point predictions) or `DistributionLoss` (probabilistic)
2. Implement `forward(y_pred, target, mask=None, weights=None)`
3. Use `self._validate_inputs()` to check shapes
4. Use `self._reduce_with_mask()` or `self._reduce()` for reduction
5. Add to `torchregress/losses/__init__.py` exports
6. Add tests following patterns in `tests/`

## Dependencies

Core dependencies:
- torch >= 2.0.0
- numpy >= 1.21.0
- torchmetrics >= 1.0.0
- matplotlib, pandas (for visualization/data handling)
- scikit-learn (density weighting utilities)

Optional (feature-specific) dependencies:
- **zuko >= 1.4.0** (normalizing flows, install via `pip install torchregress[flows]`)

### Import Policy

All imports must be direct/unconditional. NO conditional imports like:
```python
# ❌ WRONG - Do not use try/except for optional dependencies
try:
    import some_module
    AVAILABLE = True
except ImportError:
    AVAILABLE = False
```

If a module is not installed, the import will fail immediately - this is the desired behavior. Users must install required dependencies for the features they use.

**Exception:** `zuko` (normalizing flows) is an optional dependency. The `nflows` module uses a guarded `try/except ImportError` in `losses/__init__.py` so that `import torchregress` works without zuko installed.

## Adoption and Audit Standards (Required)

The repository now follows a capability-first adoption-readiness standard.

### Method Framing Policy

- Use **task-first** method framing in docs/examples.
- Treat `SWAG`, `BNN`, and `MDN` as peer methods in selection matrices.
- Do not label a method family as experimental by default.
- Assign maturity only from evidence: API stability, tests, docs/examples, and runtime behavior.
- Keep Bayesian methods available without making Bayesian concepts mandatory for onboarding.

### Claim Discipline

Any capability claim should map to at least one method that is:

- implemented
- tested
- documented
- represented in examples/comparison artifacts

If one is missing, downgrade/remove the claim until evidence exists.

## Hard-Problem Coverage Expectations

When auditing or extending methods/examples, include:

- noisy labels
- noisy features / EIV
- uncertainty quantification (epistemic, aleatoric, total)
- non-Gaussian targets
- multi-target regression
- multimodal outputs
- imbalance / rare-target regression
- calibration quality
- OOD robustness / detection support
- robust outlier handling

## Comparative Example Standards

Comparison examples are decision artifacts, not API demos.

- Use fixed seeds and shared splits.
- Keep model capacity and training budgets comparable across methods.
- Report explicit tradeoffs: error, calibration/coverage, robustness, and runtime.
- Include caveats/failure modes in example notes.
- Emit machine-readable artifacts through `comparison_utils.write_comparison_summary_json(...)`.

### Summary Governance

Keep profile governance active:

- `smoke`, `audit`, `full` example summaries
- profile comparison: `audit -> full`
- thresholds:
  - `ci_conservative` (blocking CI profile)
  - `review_strict` (review/release profile)

## Native PyTorch Leverage Policy

Default to wrapping/using native primitives unless custom logic adds clear value.

- Prefer `torch.nn`, `torchmetrics`, `torch.distributions`, `torch.linalg`, AMP/compile APIs.
- Keep custom code where required for:
  - masks/weights/reduction semantics
  - EIV objectives
  - multimodal/task-specific outputs
  - uncertainty decomposition/reporting contracts
- Maintain parity and divergence evidence in:
  - `reports/native_pytorch_leverage_matrix_2026-02-26.json`
  - `tests/test_native_leverage_matrix_contracts.py`
  - `tests/test_native_parity.py`

Each matrix row must carry `coverage_evidence` with:

- `parity_tests`
- `known_divergences`

## Photo-z / RAIL / NNC-CRPS Workflow

Do not require manual data staging for photo-z baseline comparison.

Primary tools:

- `tools/photoz_rail_assets.py` (materialize dataset/baseline assets + checksum updates)
- `tools/photoz_rail_pipeline.py` (end-to-end collect + torchregress summary + RAIL merge)
- `tools/photoz_rail_compare.py` (merge-only adapter)

Presets:

- `rail`
- `nnc_crps`

Tracked manifest templates:

- `configs/photoz/rail_photoz_manifest.template.json`
- `configs/photoz/nnc_crps_photoz_manifest.template.json`

The runtime manifest path defaults to `data/rail/rail_photoz_manifest.json`. Since `data/` is ignored,
prefer CLI overrides and/or template bootstrap for reproducible automation.

Useful override flags (both assets and pipeline tools):

- `--dataset-url KEY=URL`
- `--baseline-url METHOD=URL`
- `--dataset-path KEY=PATH`
- `--baseline-path METHOD=PATH`

## Plan-driven features and method catalog

When implementing items under `docs/research/plans/`, prefer **batched** method-catalog updates: add `MethodMetadata` / evidence rows and run `tools/render_method_catalog.py` (plus `render_realdata_recommendation_guide`) **once per tranche** when the slice is feature-complete, not on every intermediate PR. See `docs/research/plans/README.md`.

## Regeneration and Gate Checklist

Before handoff for audit/governance changes, run:

```bash
uv run python tools/render_example_summaries.py --profile smoke
uv run python tools/render_example_summaries.py --profile audit
uv run python tools/render_example_summaries.py --profile full
uv run python tools/compare_example_summary_profiles.py \
  --base-dir reports/example_summaries \
  --source-profile audit \
  --target-profile full \
  --output reports/example_summaries/profile_comparison_audit_vs_full.json
uv run python tools/example_summary_thresholds.py \
  --base-dir reports/example_summaries \
  --profile full \
  --threshold-profile ci_conservative \
  --write-thresholds reports/example_summaries/thresholds_full.json \
  --thresholds reports/example_summaries/thresholds_full.json \
  --output-verdict reports/example_summaries/threshold_check_full_latest.json
uv run python tools/example_summary_thresholds.py \
  --base-dir reports/example_summaries \
  --profile full \
  --threshold-profile review_strict \
  --write-thresholds reports/example_summaries/thresholds_full_review_strict.json \
  --thresholds reports/example_summaries/thresholds_full_review_strict.json \
  --output-verdict reports/example_summaries/threshold_check_full_review_strict_latest.json \
  --runtime-multiplier 6.0 \
  --runtime-floor 0.35 \
  --metric-multiplier 3.0 \
  --metric-floor 0.2 \
  --prob-delta 0.25 \
  --r2-delta 1.0
uv run python tools/render_method_catalog.py \
  --markdown-out docs/reports/method_catalog_generated.md \
  --json-out reports/method_catalog_latest.json \
  --update-method-matrix docs/guide/method-selection.md \
  --comparative-evidence-md-out docs/reports/comparative_evidence_matrix.md \
  --comparative-evidence-json-out reports/comparative_evidence_matrix_latest.json
uv run python tools/adoption_audit.py --json reports/adoption_readiness_2026-02-25.json --print-summary
uv run python tools/render_review_packet.py
uv run pytest -q
# Lint Python packages only (not markdown/docs files).
uv run ruff check torchregress tests tools
uv run mypy torchregress
uv run mkdocs build
```

## Scheduled Governance Automation

Heavy governance refresh runs should stay outside default PR/push CI.

- Scheduled/manual workflow: `.github/workflows/governance-refresh.yml`
- Trigger policy:
  - `schedule` for recurring artifact refresh
  - `workflow_dispatch` for manual refresh
  - **no** `push`/`pull_request` triggers for this workflow

Manual local equivalent (full governance refresh):

```bash
uv run python tools/render_example_summaries.py --profile smoke
uv run python tools/render_example_summaries.py --profile audit
uv run python tools/render_example_summaries.py --profile full
uv run python tools/compare_example_summary_profiles.py \
  --base-dir reports/example_summaries \
  --source-profile audit \
  --target-profile full \
  --output reports/example_summaries/profile_comparison_audit_vs_full.json
uv run python tools/example_summary_thresholds.py \
  --base-dir reports/example_summaries \
  --profile full \
  --threshold-profile ci_conservative \
  --write-thresholds reports/example_summaries/thresholds_full.json \
  --thresholds reports/example_summaries/thresholds_full.json \
  --output-verdict reports/example_summaries/threshold_check_full_latest.json
uv run python tools/example_summary_thresholds.py \
  --base-dir reports/example_summaries \
  --profile full \
  --threshold-profile review_strict \
  --write-thresholds reports/example_summaries/thresholds_full_review_strict.json \
  --thresholds reports/example_summaries/thresholds_full_review_strict.json \
  --output-verdict reports/example_summaries/threshold_check_full_review_strict_latest.json \
  --runtime-multiplier 6.0 \
  --runtime-floor 0.35 \
  --metric-multiplier 3.0 \
  --metric-floor 0.2 \
  --prob-delta 0.25 \
  --r2-delta 1.0
uv run python -m tools.benchmark_smoke \
  --mode smoke \
  --iterations 2 \
  --warmup 1 \
  --device cpu \
  --thresholds reports/benchmark_thresholds/cpu/smoke.json \
  --fail-on-thresholds
uv run python -m tools.benchmark_smoke \
  --mode sweep \
  --iterations 2 \
  --warmup 1 \
  --device cpu \
  --thresholds reports/benchmark_thresholds/cpu/sweep.json \
  --fail-on-thresholds
uv run python -m tools.benchmark_report_summary reports/benchmark_smoke_latest.json --output reports/benchmark_smoke_latest.md
uv run python -m tools.benchmark_report_summary reports/benchmark_sweep_latest.json --group-by-name --output reports/benchmark_sweep_latest.md
uv run python tools/render_method_catalog.py \
  --markdown-out docs/reports/method_catalog_generated.md \
  --json-out reports/method_catalog_latest.json \
  --update-method-matrix docs/guide/method-selection.md \
  --comparative-evidence-md-out docs/reports/comparative_evidence_matrix.md \
  --comparative-evidence-json-out reports/comparative_evidence_matrix_latest.json
uv run python -m tools.render_realdata_recommendation_guide \
  --doc docs/reports/real_data_recommendation_guide.md \
  --comparative-json reports/comparative_evidence_matrix_latest.json
uv run python tools/adoption_audit.py --json reports/adoption_readiness_2026-02-25.json --print-summary
uv run python tools/render_review_packet.py
```

## Agent PR hygiene and lint gates

- Do **not** open PRs that only fix a single unused import or whitespace in isolation. Use **one** repo-wide Ruff/Black pass or attach cleanup to a substantive change.
- Optional: `pre-commit install` and `pre-commit install --hook-type pre-push` when configured in this repo.

### Mandatory pre-push gate (save GitHub Actions minutes)

**Do not `git push`** until local checks pass (automated agents: **blocking**). CI is expensive here; avoid burning minutes on **F401**, **Black** drift, or **SyntaxError**.

**Preferred (matches GitHub Actions):** `./scripts/ci_local.sh`

**Minimal** when you need a faster loop (small, localized edits only — widen if anything fails in CI):

1. `python -m compileall -q torchregress tests tools`
2. `uv run ruff check torchregress tests tools`
3. `uv run black --check torchregress tests tools`
4. `uv run pytest` (or a **narrow** file/`::test` path that covers your change)

If you have the pre-push hook installed, `git push` already runs `./scripts/ci_local.sh` — keep it that way for routine work.
