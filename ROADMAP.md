# Roadmap

0.3.0b1 is the API cut. 0.3.0 is the release that makes that API correct,
fast, and documented. One public name per behavior. Compatibility aliases
are not added back.

| Release | Theme | Contents | Done when |
|:--|:--|:--|:--|
| **0.3.0b1** | Clean API | One name per behavior. Removed duplicate exports, no-op arguments, and agent clutter. Dependency floors match the lockfile. Pre-release tags publish to TestPyPI only. | `pyproject.toml` says `0.3.0b1` and tag `v0.3.0b1` exists |
| **0.3.0** | Correct, fast, documented | Bug fixes and performance work on the code that already exists. Docs that match the code. Harness parity. No new methods. No compatibility shims. Example pairs that differ only by dataset share one script (`--dataset`). | `pip install torchregress` works on Linux and macOS, Python 3.12–3.14, and the harness release run is green |
| 0.4 | GPU equality | Self-hosted CUDA runner, CPU–CUDA equality for every loss and metric, OOM guard, MPS decision, bf16/fp16 where it is safe | Every loss and metric passes CPU–CUDA equality |
| 0.5 | Estimators | Bundled conditional-flow estimator (`fit` / `sample` / `log_prob` / `interval`) and scikit-learn-style wrappers for the most-used families | The harness sbi and pzflow one-liners work |
| 0.6 | Structured outputs | Joint intervals beyond Gaussian. Censored and errors-in-variables tutorials for survey data | Harness rows for each |
| 1.0 | API freeze | Two minor releases with no unplanned breaking change. Every method family benchmarked. Software paper with the harness paper | No unplanned breaking change from 0.5 to 0.6 |

## 0.3.0 docs and harness work

These stay out of the beta diff and land while 0.3.0 is stabilized:

- Merge example pairs that share a harness and differ only by data source: causal, censored, EIV, multimodal, noisy-label, OOD selective prediction, ordinal, and uncertain-ground-truth comparisons.
- Merge the overlapping loss, conformal, contrastive-flow, count, Bayesian-linear-head, and Wasserstein-bound demos the same way.
- Move external-library comparators and the SAGE-Reg Higgs script to torchregress-harness and torchregress-research.

## After 0.3.0

Not scheduled inside 0.3.0: a shift-adaptation product surface (covariate shift, label shift, selection bias, calibration drift) beyond the test-time primitives that already exist; a model-card report from `health.py`; a hosted leaderboard.

1.0 is a promise about API stability, not about features.
