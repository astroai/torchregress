# Roadmap

> **Algorithm freeze for 0.3.0 (since 2026-10-05).** Until 0.3.0 is on PyPI,
> only fixes, tests, docs and packaging land. Which methods to add or remove
> is decided for 0.4.

1.0 is a promise about API stability, not about features. It comes only after
two minor releases with no unplanned breaking change.

| Release | Theme | Contents | Done when |
|:--|:--|:--|:--|
| **0.3.0** | First PyPI release | 0.3.0 audit of losses, conformal, metrics, calibration, inference, causal and ensembles ([report](docs/reports/audit_0.3.0.md)); metrics parity tests against scoringrules/properscoring/scipy/scikit-learn; harness release run on CANFAR; local CUDA test run (`pixi run test-cuda`) on the release candidate; trusted publishing from `astroai/torchregress` | `pip install torchregress` works on Linux and macOS, Python 3.12–3.14 |
| 0.3.x | Stabilise | Patch fixes from early users; audit batches 4–5 (`algorithms`, `test_time`, `utils`, `viz`); weekly latest-comparators harness job; conda-forge feedstock | No open High issues; feedstock merged |
| 0.4 | GPU and hardening | Self-hosted CUDA runner (weekly and per release candidate), CPU–CUDA equality for every loss and metric, OOM guard, MPS decision; bf16/fp16 where it is safe; API trim (private helpers, one spelling per idea) with a deprecation policy | Every loss and metric passes CPU–CUDA equality |
| 0.5 | Estimators, not just losses | Bundled `ConditionalFlow` estimator (fit / sample / log_prob / interval); scikit-learn-style wrappers for the most-used method families; amortised-inference helpers building on the sbi comparison | The one-line usage in the harness sbi and pzflow suites works |
| 0.6 | Structured and multi-target | Joint intervals beyond Gaussian; censored and errors-in-variables workflows for survey data (photo-z, galaxy properties) as tutorials | Harness rows for each, at parity or better |
| 1.0 | API freeze | Deprecation policy enforced for two minors; every method family benchmarked; complete docs; software paper together with the harness paper | No unplanned breaking change from 0.5 to 0.6 |

Not yet scheduled: a model-card style report built from `health.py` and the
metrics suite; a hosted leaderboard from the latest-comparators job.

## Deprecation policy (from 0.4)

A public name or argument that is going away is deprecated in minor release
N with a `DeprecationWarning` that names the replacement, and removed in N+2.
The CHANGELOG lists every deprecation and removal.

## Earlier roadmap items

The 0.1.0 and 0.2.0 milestones of the previous roadmap shipped as described in
`CHANGELOG.md`, except for these, which move to the releases above:

- CUDA runner, device fixture, `@pytest.mark.cuda`, CPU–CUDA equality, OOM
  guard and MPS decision → 0.4.
- Coverage target of 92% on the risky paths, and property-based tests beyond
  the metrics → 0.3.x.
- Notebook examples for every method category → after the examples are sorted
  into tutorial, comparison and real-data scripts (0.3.x).
