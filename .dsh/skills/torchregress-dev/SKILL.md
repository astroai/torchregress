---
name: torchregress-dev
description: Develop torchregress (regression losses, UQ, calibration) — distribution heads, conformal prediction, robust losses, proper metrics
metadata:
  project: torchregress
  stack: python, pytorch, pixi
---

# torchregress development

PyTorch library of regression losses, metrics, and calibration tools for
problems that need more than a point prediction: uncertainty, robustness, and
messy real-world data. Library name is lowercase "torchregress".

## Capabilities

- Distribution prediction: means, spreads, quantiles, or full predictive
  distributions (`flows` extra for multi-target normalizing flows).
- Uncertainty separation: irreducible data noise vs model uncertainty.
- Robustness: missing values, sample weights, outliers, noisy inputs/labels,
  censored targets, imbalanced/rare outcomes — no ad hoc fixes.
- Calibration and coverage: conformal prediction sets (CQR) with stated
  coverage properties; calibration metrics.
- Distribution shift: lightweight test-time tools.
- Honest metrics: point error, interval quality, distributional accuracy,
  calibration — not just MSE.

## Boundaries (AGENTS.md)

- Do NOT add paper manuscripts or NeurIPS benchmark scripts here — they belong
  in `torchregress-research` (SAGE/SPT benchmarks).
- Do NOT add SAGE/SPT parity suites to `torchregress-harness`; that repo is for
  external-software parity.

## Gates

- `pixi run format`, `pixi run lint`, `pixi run typecheck`
- Full local parity: `./scripts/ci_local.sh` or `pixi run ci`
  (pre-commit → lint, typecheck, test, docs, benchmark smoke)
- Narrow: `pixi run pytest tests/losses/test_gaussian.py`

## Statistics principles

Validate calibration empirically (coverage matches stated), use proper scoring
rules (CRPS, interval score, NLL), and document assumptions about noise,
missing data, and censoring in every loss.
