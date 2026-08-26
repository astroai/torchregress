# AGENTS.md

This file provides merged guidance to coding agents (Claude, Gemini, Codex) when working in this repository.

## Project Overview

**torchregress** (lowercase) is a PyTorch library providing regression losses, metrics, and utilities with a focus on uncertainty estimation, robust regression, and missing data support.

**Naming Convention:** The library name is "torchregress" (all lowercase).

## Sibling repositories

| Repo | Path (typical) | Scope |
|------|----------------|-------|
| **torchregress** (this repo) | `$HOME/src/torchregress` | Library, docs, general examples, CI |
| **torchregress-research** | `$HOME/src/torchregress-research` | NeurIPS manuscripts, SAGE/SPT benchmarks |
| **torchregress-harness** | `$HOME/src/torchregress-harness` | External-software parity harness |

Do not add paper manuscripts or NeurIPS benchmark scripts to this repo. Do not add SAGE/SPT
parity suites to **torchregress-harness** (they belong in **torchregress-research**).

## Development Commands

This project uses [pixi](https://pixi.sh) as the package manager. `pyproject.toml` is package metadata for PyPI/`pip`; local and CI work goes through pixi.

### Setup

Uses Python **3.13** by default (see `.python-version`). Supported: 3.12–3.14.

```bash
pixi install
```

**Extras & features reference:**

| Tag | Kind | Contents |
|---|---|---|
| *(none)* | core | torch, numpy, torchmetrics, scipy |
| `test` | extra / pixi feature | pytest, pytest-cov, polars, pyarrow, PyYAML, scikit-learn, pandas |
| `docs` | extra / pixi feature | zensical |
| `viz` | extra | matplotlib |
| `flows` | extra | zuko |
| `all` | extra | test + docs + viz + flows |
| `dev` | pixi feature | ruff, ty |

Default pixi environment includes `dev` + `test` + `docs` (and pulls matplotlib/zuko for CI).

### Testing
```bash
pixi run test
pixi run pytest tests/losses/test_gaussian.py
pixi run pytest tests/losses/test_gaussian.py::TestGaussianLosses::test_gaussian_nll_loss
```

### Code Quality
```bash
pixi run format
pixi run lint
pixi run typecheck
```

### CI parity before push (recommended)

GitHub Actions on `main` is **two-stage**: **`pre-commit run --all-files`**, then **`lint-test`** via pixi (`lint`, `typecheck`, `test`, `docs`, benchmark smoke). See `.github/workflows/ci.yml`. Match the full gate locally:

```bash
./scripts/ci_local.sh
# or
pixi run ci
```

### Pre-commit / pre-push hooks

```bash
pixi run pre-commit install
pixi run pre-commit install --hook-type pre-push
```

Or with a system/pre-commit install: `pre-commit install` / `pre-commit install --hook-type pre-push`.

### Documentation
```bash
pixi run docs
pixi run zensical serve
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
- Build must pass `zensical build` with zero errors before commit.
- Run `pixi run python tools/audit_docs_quality.py` before docs changes land (see
  `reports/docs_quality_audit.json` and CONTRIBUTING.md “Math and document structure”).
- Target audience: both ML practitioners and statisticians. Be rigorous but accessible.

### Build & Publish

PyPI releases are automated from annotated git tags via GitHub Actions. See
[`docs/RELEASING.md`](docs/RELEASING.md) for the full maintainer runbook.

```bash
# Prepare a release locally (bump, CI checks, build, commit, tag)
./scripts/release/prepare_release.sh patch

# Push to trigger publish
git push origin main
git push origin vX.Y.Z
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
- Reductions handled by `BaseLoss._reduce()` (unified zero-fill mask/weight policy)

### Module Organization

```
torchregress/
├── losses/          # Loss functions (gaussian, robust, quantile, conformal, etc.)
├── metrics/         # Evaluation metrics (point, interval, distribution, OOD, etc.)
├── calibration/     # Post-hoc transforms, calibration metrics, shift calibrator
├── ensemble/        # Ensemble models (DeepEnsemble, BatchEnsemble, etc.)
├── algorithms/      # Training algorithms (IRLS, SIMEX, RC, etc.)
├── test_time/       # Test-time adaptation (label shift, transport, subspace)
├── comparison.py    # Reproducible comparison-example helpers and summary JSON
├── prediction.py    # Predictive batch containers
├── utils/           # Shared helpers (gaussian_output, validation, tensor_ops, reduction)
└── losses/base.py   # WeightedLossWrapper and weighted point/Gaussian wrappers
```

### Key Abstractions

**Weighted loss wrappers** (`torchregress.losses.base`): `WeightedLossWrapper`, `WeightedMSELoss`, `GaussianNLLLoss`, and related subclasses add mask and sample-weight support around native PyTorch losses.

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

Detailed semantics belong in docs, not in agent instructions. Keep
`method_catalog.py` conservative and aligned with
`docs/guide/uncertainty-decomposition.md`: conformal coverage, predictive
spread, ensemble disagreement, and epistemic/aleatoric decomposition are
different contracts. Single-model quantile/MDN/flow losses are not epistemic
decomposition APIs; ensembles of those heads can expose epistemic disagreement,
and full variance decomposition requires per-member variance or distributional
spread plus member disagreement.

## Configuration

**pyproject.toml settings**:
- Python >= 3.12 and < 3.16 required (3.13 recommended; CI tests 3.12, 3.13, and 3.14)
- Ruff: enforces E (pycodestyle), F (pyflakes), I (isort); also used for formatting
- MyPy: strict typing for `torchregress.*`; examples/tools/docs are excluded or ignored

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
- torch >= 2.4.0
- numpy >= 2.0.0
- matplotlib >= 3.8.0
- torchmetrics >= 1.4.0
- scipy >= 1.11.0
- tqdm >= 4.66.0

Optional (feature-specific) dependencies:
- **zuko >= 1.6.0** (normalizing flows, install via `pip install 'torchregress[flows]'` or pixi default env)
- **pandas, scikit-learn, polars, pyarrow** (data handling, in the `test` extra)
- **zensical** (docs tooling, in the `docs` extra)

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

## Adoption Standards

The repository follows a capability-first documentation standard.

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
- Emit machine-readable artifacts through `torchregress.comparison.write_comparison_summary_json(...)`.

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

## Method catalog and docs refresh

When adding or changing exported methods, update `src/torchregress/method_catalog.py` and regenerate docs artifacts:

```bash
pixi run python tools/render_method_catalog.py \
  --markdown-out docs/reports/method_catalog_generated.md \
  --json-out reports/method_catalog_latest.json \
  --update-method-matrix docs/guide/method-selection.md \
  --comparative-evidence-md-out docs/reports/comparative_evidence_matrix.md \
  --comparative-evidence-json-out reports/comparative_evidence_matrix_latest.json
pixi run python -m tools.render_realdata_recommendation_guide \
  --doc docs/reports/real_data_recommendation_guide.md \
  --comparative-json reports/comparative_evidence_matrix_latest.json
```

Optional scheduled refresh: `.github/workflows/docs-refresh.yml` (manual or weekly).

## Agent PR hygiene and lint gates

- Do **not** open PRs that only fix a single unused import or whitespace in isolation. Use **one** repo-wide Ruff pass or attach cleanup to a substantive change.
- Optional: `pre-commit install` and `pre-commit install --hook-type pre-push` when configured in this repo.

### Mandatory pre-push gate (save GitHub Actions minutes)

**Do not `git push`** until local checks pass (automated agents: **blocking**). CI is expensive here; avoid burning minutes on **F401**, **Black** drift, or **SyntaxError**.

**Preferred (matches GitHub Actions):** `./scripts/ci_local.sh`

**Minimal** when you need a faster loop (small, localized edits only — widen if anything fails in CI):

1. `python -m compileall -q src/torchregress tests tools`
2. `pixi run lint`
3. `pixi run typecheck`
4. `pixi run test` (or a **narrow** file/`::test` path that covers your change)

If you have the pre-push hook installed, `git push` already runs `./scripts/ci_local.sh` — keep it that way for routine work.
