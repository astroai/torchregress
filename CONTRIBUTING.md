# Contributing to TorchRegress

## Development Setup

We use **Pixi** for environment management. The repo pins **Python 3.13** in `.python-version` (recommended default); **3.12** and **3.14** are also supported.

### Installation

```bash
pixi install
```

### Running Quality Checks & Tests Locally

Before committing or pushing, make sure the local gate check passes. This matches the two-stage GitHub Actions CI workflow.

Individual checks:

```bash
# Formatter (Ruff)
pixi run ruff format src/torchregress tests tools

# Linter (Ruff)
pixi run ruff check .

# Type checking (MyPy)
pixi run mypy src/torchregress

# Run tests
pixi run pytest
```

To run a specific test file:

```bash
pixi run pytest tests/losses/test_eiv.py
```

Alternatively, run the full CI parity script:

```bash
./scripts/ci_local.sh
```

### Pre-commit Hooks

We use `pre-commit` for fast checks on commit and full CI parity on push:

```bash
pre-commit install
pre-commit install --hook-type pre-push
pre-commit run --all-files
```

## Documentation Conventions

### Static Site Generator

Docs are built with [Zensical](https://zensical.org/) (`zensical build --strict`).
The `--strict` flag fails the build on any unresolved link reference — run locally
before pushing:

```bash
pixi run zensical build --strict
```

### Inline Citation Markers

Zensical is stricter than MkDocs about bracket notation: `[1]` in body text is
interpreted as a Markdown link reference target. To render a citation marker as
literal text (e.g., "… as shown in [1]."), **escape the brackets with backslashes**:

```markdown
<!-- ❌ Wrong — Zensical treats [1] as an unresolved link reference -->
… the Huber Loss [1] is the default choice.

<!-- ✅ Correct — \[1\] renders as literal [1] -->
… the Huber Loss \[1\] is the default choice.
```

This applies to all inline citations: `\[2\]`, `\[4, 5\]`, `\[1, 2, 3\]`, etc.
The References table at the bottom of each page uses `| # | Reference |` format
and does **not** need escaping.

> **Note:** LaTeX math inside `$...$` or `$$...$$` is **not** affected —
> Zensical correctly ignores brackets inside math delimiters.

### Math and document structure

Keep headings, display math, and lists structured so Zensical + MathJax render
predictably:

- **Headings** carry the title only — never inline body text or equations on the
  same `##` line.
- **Display math** uses `$$` on dedicated lines (multi-line blocks: opening and
  closing `$$` each on their own line). Put prose punctuation *outside* display
  math when it is not part of the formula.
- **Lists after equations** need a one-line bridge (`The components are:`) before
  bullets; prefer `-` over `*`.
- **Stop-gradient** in LaTeX: use `\operatorname{sg}(\mu)`, not
  `\text{stop\_gradient}`.
- **Soft line wraps** in guide prose are fine — consecutive non-blank lines render
  as one paragraph; do not reflow them unnecessarily.

Before docs PRs, run the quality audit:

```bash
pixi run python tools/audit_docs_quality.py
pixi run zensical build --strict
```

Tracker state lives in `reports/docs_quality_audit.json`; regenerate the
human-readable table with:

```bash
pixi run python tools/audit_docs_quality.py --markdown-out docs/reports/docs_quality_audit.md
```

## Releasing

PyPI releases are tag-driven and published from GitHub Actions using PyPI Trusted Publishing.
See [`docs/RELEASING.md`](docs/RELEASING.md) for the maintainer checklist and one-time setup.

## SLS Loss Architecture Guidelines

The `SLSLoss` (Super-Level-Set Regression) has three key architecture parameters
that control the expressiveness of its internal volume-preserving flow and frontier
networks.  The library defaults are the **recommended minimums**:

| Parameter | Default | Minimum | Effect of reducing below minimum |
|---|---|---|---|
| `hidden_dim` | 64 | 64 | Under-expresses coupling layers, frontier context net, and quantile net. JointCoverage drops ~4.5% (0.92 → 0.88); intervals widen. |
| `n_transforms` | 4 | 4 | Fewer coupling transforms → less expressive flow → broader densities → wider intervals. Compounds with undersized `hidden_dim`. |
| `warmup_steps` | 500 | 50* | Frontier may not converge before quantile training begins, especially on large/ complex datasets. |

\* On small/medium tabular datasets (few thousand samples), 50 warmup steps often
suffices — sweep data shows no gain above 50 on `synthetic_multivariate`.  500 is
the safe library default that ensures convergence across all dataset sizes.

These values were empirically validated via architecture sweeps in the
`torchregress-harness` multivariate intervals benchmarks.

## Normalizing Flow Architecture Guidelines

The `NormalizingFlowLoss` and `create_flow_model` use zuko normalizing flows
under the hood.  The library defaults (inherited from zuko) are the
**recommended minimums** for tabular regression:

| Parameter | Default | Minimum | Effect of reducing below minimum |
|---|---|---|---|
| `hidden_features` | `[64, 64]` | `[64, 64]` | Under-expresses coupling networks → overconfident (too-peaked) densities → under-coverage. On diabetes at 60ep: `[32,32]` coverage=0.59 vs `[64,64]` coverage=0.66 (+12%). |
| `n_transforms` | 5 | 3* | More transforms increase expressivity but overfit small tabular data. Sweep: T5→coverage 0.62, T3→coverage 0.75 on diabetes. 3 is sufficient for tables <10k samples; 5 is the safe default for general use. |
| `flow_type` | `nsf` | `nsf` | Neural Spline Flow provides best expressivity-per-parameter. MAF and RealNVP are alternatives but less expressive per transform. |

\* On small/medium tabular datasets (< 10k samples), 3 transforms outperforms 5 —
more transforms cause the flow to overfit, producing overconfident densities.
5 is the safe library default for larger datasets and higher-dimensional targets.

The harness uses `n_transforms=3, hidden_features=[64, 64]` for both scalar
(distributional_bins) and multivariate (multivariate_intervals) benchmarks.
These values were empirically validated via architecture sweeps in
`torchregress-harness`.

## Gaussian Negative Log-Likelihood Architecture Guidelines

The `GaussianNLLLoss` is a *pure loss function* — it has no internal neural
network and no tunable architecture parameters.  It computes the diagonal
Gaussian NLL:

$$\text{NLL}(y| \mu, \sigma^2) = \frac{1}{2}\left(\log(2\pi) + \log\sigma^2 + \frac{(y - \mu)^2}{\sigma^2}\right)$$

Because the loss itself is stateless, the "architecture" is entirely in your
model's output head.  The practical recommendations are:

| Parameter | Recommended | Why |
|---|---|---|
| Output dimensionality | `d * 2` (mean + log_var per target) | Standard diagonal Gaussian parameterisation. |
| `log_variance` | `True` (default) | Learning log σ² is numerically more stable than raw variance; the loss exponentiates and clamps internally. |
| `fixed_variance` | `None` (default) | Learn heteroscedastic variance per sample.  Set to a positive float (e.g. `1.0`) only when you want a fixed-variance baseline or your model outputs mean-only. |
| `min_variance` | `1e-6` (default) | Floor on σ² to prevent division-by-zero; rarely needs tuning. |

**When to choose GaussianNLL:**

- **Default starting point** for any regression task with uncertainty — it is the
  best all-rounder across all benchmark suites.
- Use when you need **per-sample aleatoric uncertainty** (heteroscedastic variance).
- Use as the **fine-tuning loss** after Wasserstein pre-training (see
  `gaussian_wasserstein_bound_demo.py`).
- For **multivariate targets with full covariance**, use
  ``MultivariateGaussianLoss`` instead.
- For **low-rank covariance** (e.g. high-dimensional outputs), use
  ``LowRankGaussianLoss``.
- For **proper scoring rules** instead of NLL, use
  ``GaussianCRPSLoss``.

The GaussianNLL family is the most heavily validated loss in the harness:
every benchmark suite includes it as a reference method, and the
`validate_gaussiannll_coverage()` CI smoketest guards against regressions
in output dimensionality, z-score computation, and loss correctness.

## EIV Loss Implementation Notes

- Always use `torch.double` when performing `gradcheck` on EIV losses.
- Analytical EIV losses use second-order derivatives; ensure your model is twice-differentiable if you need gradients of the EIV loss.
- For stochastic EIV losses (Monte Carlo, Ensemble), use finite-gradient checks instead of `gradcheck`.
