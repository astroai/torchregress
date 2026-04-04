## Test-Time Formulations for Tabular + Distribution Shift Robustness

### 🔵 Mainstream / Strong Baselines
- **TabPFN v2** (Hollmann et al., 2025) — Prior-fitted transformer; test-time ensembling scales well; strong OOD on small-to-medium tables via Bayesian marginalisation at inference.
- **TENT-style entropy minimisation** — Adapted from vision; minimises prediction entropy at test time via batch norm / feature stats updates. Mixed results on tabular without careful feature engineering.

### 🟣 In-Context / Retrieval at Test Time
- **TabICL / TabDPT** — Treat test rows as in-context examples; robustness comes from retrieval of similar training points at inference. Distribution shift handled implicitly via nearest-neighbour context selection.
- **CARTE** (INRIA, 2024) — Graph-based retrieval-augmented; specifically designed for covariate shift over heterogeneous tables.

### 🟠 Test-Time Training (TTT) Variants
- **Masked autoencoder TTT for tabular** (e.g., TabMAE-style) — Self-supervised auxiliary loss on test batch features before prediction head fires. Underexplored but well-founded for feature shift.
- **LoRA-based TTT** — Fine-tunes low-rank adapters at test time on self-supervised tabular objectives; rare in tabular but gaining traction from LLM adaptation work.

### 🟡 Uncommon but Well-Founded
- **Prediction-Powered Inference (PPI++)** — Uses unlabelled test data to debias predictions; formally handles covariate + label shift with coverage guarantees.
- **Online learning / FTRL at test time** — Treats deployment as an online game; strong distribution shift guarantees via regret bounds. Rarely applied to tabular ML pipelines directly.
- **Conformal Risk Control + TTA** — Combines test-time adaptation with distribution-free coverage; robust to shift without assuming shift type.

---

**Current best overall** (robustness + performance): **TabPFN v2** for small tables, **CARTE or TabDPT** for larger heterogeneous ones, with conformal wrappers for formal shift guarantees.
