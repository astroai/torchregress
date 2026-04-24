# Research

Library-level research directions and implementation plans for torchregress.

!!! info "Paper-specific research"
    Paper manuscripts, experiment results, and operational infrastructure live in [`papers/`](https://github.com/sfabbro/torchregress/tree/main/papers) at the repo root and are not part of the public documentation.

---

## Implementation Plans

Draft design notes for features under development. These are **not** committed API contracts — promote content into guides, issues, or ADRs when a direction is adopted.

| Plan | Status | Summary |
|:-----|:-------|:--------|
| [Self-Training](plans/self_training.md) | Active | Semi-supervised regression via pseudo-labels and consistency |
| [Shift-Aware OT Conformal](plans/shift_aware_ot_conformal.md) | Core landed | Transport-based conformal under distribution shift |
| [Wasserstein Supervision](plans/Wasserstein_Supervision.md) | Complete | Wasserstein-bound pretraining for covariance heads |
| [Bayesian Learning Rule](plans/bayesian_learning_rule_abstractions.md) | Delivered | BLR abstractions and PredictiveBatch adapter |
| [β-NLL](plans/beta_nll.md) | Complete | Detached-variance rescaled Gaussian NLL |
| [Impact Roadmap](plans/impact.md) | Partial | Multi-track feature impact plan |

→ See [plans/README.md](plans/README.md) for governance and batching policy.
