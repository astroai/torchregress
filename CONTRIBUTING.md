# Contributing to TorchRegress

## Development Setup

We use `uv` for environment management.

### Installation

```bash
uv sync
```

### Running Tests

To run all tests:
```bash
uv run pytest
```

To run a specific test file:
```bash
uv run pytest tests/losses/test_eiv.py
```

### Code Quality

We use `pre-commit` to ensure code quality.

```bash
pre-commit install
pre-commit run --all-files
```

## Documentation Conventions

### Static Site Generator

Docs are built with [Zensical](https://zensical.org/) (`zensical build --strict`).
The `--strict` flag fails the build on any unresolved link reference — run locally
before pushing:

```bash
uv run zensical build --strict
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
uv run python tools/audit_docs_quality.py
uv run zensical build --strict
```

Tracker state lives in `reports/docs_quality_audit.json`; regenerate the
human-readable table with:

```bash
uv run python tools/audit_docs_quality.py --markdown-out docs/reports/docs_quality_audit.md
```

## Releasing

PyPI releases are tag-driven and published from GitHub Actions using PyPI Trusted Publishing.
See [`docs/RELEASING.md`](docs/RELEASING.md) for the maintainer checklist and one-time setup.

## EIV Loss Implementation Notes

- Always use `torch.double` when performing `gradcheck` on EIV losses.
- Analytical EIV losses use second-order derivatives; ensure your model is twice-differentiable if you need gradients of the EIV loss.
- For stochastic EIV losses (Monte Carlo, Ensemble), use finite-gradient checks instead of `gradcheck`.
