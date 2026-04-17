# Draft plans

Long-form design notes that are **not** committed API or documentation contracts.
Promote content into `docs/guides/`, ADRs, or issues when a direction is adopted.

## Method catalog and governance refresh (batched)

Work that originates in these plans should land in **small, reviewable PRs** with **code, tests, and user-facing docs** as needed.

**Do not** update the task-first **method catalog** on every step:

- Avoid adding rows to `torchregress/method_catalog.py` (`MethodMetadata`, `TaskRecommendation`, `ComparativeEvidenceRow`) or running the full catalog/evidence/real-data guide refresh (`tools/render_method_catalog.py`, `tools/render_realdata_recommendation_guide.py`) until a defined **implementation tranche** is complete.
- When the tranche is done, run **one closing batch**: edit `method_catalog.py` for all net-new public symbols from that tranche, then run the refresh commands documented in [`AGENTS.md`](../../../AGENTS.md) (method catalog markdown/JSON, method matrix markers, comparative evidence artifacts, real-data recommendation guide) and `uv run pytest` so snapshot tests stay green.

This keeps governance diffs predictable and groups discovery metadata with the features it describes.

## Implementation status (rolling)

High-level snapshot of plan intent vs. current code (not a formal contract):

| Area | Plan file(s) | Status |
|:-----|:-------------|:-------|
| OT shift conformal | `shift_aware_ot_conformal.md` | Core adapters, diagnostics, PredictiveBatch helper, tests/benchmarks landed; Workstream B (VIDS-style) still research-only. |
| Wasserstein supervision | `Wasserstein_Supervision.md` | Hybrid demo + behaviour tests + catalog rows; optional extra docs (e.g. dedicated pseudo-label page) remain optional. |
| Bayesian LR / PredictiveBatch | `bayesian_learning_rule_abstractions.md` | Delivered including a thin `SupportsPredictiveBatch` adapter demo for `BayesianLinearHead`. |
| β-NLL | `beta_nll.md` | Treated complete; spot-check doc checkboxes only. |
| Impact roadmap (`impact.md`) | β-NLL, balanced MSE, faithful Gaussian, packed ensemble, UACQR thin wrapper | Implemented in library; larger tracks (e.g. full `SplitConformalRegressor` *trainer* façade) remain optional sugar on existing predictors. |
| Conformal extensions | `impact.md` §7 | `UACQR` added as thin `CQR` wrapper; further paper-specific variants out of scope unless specified. |
| Roadmap E1–E4 follow-ons | `impact.md` | E2–E4 and faithful loss delivered; **E1** naming in impact doc refers to trainer-style APIs—**split/CQR/UACQR predictors** exist under `torchregress.losses`. |

For items marked **optional** or **research-grade**, prefer a scoped issue or new plan file before large refactors.
