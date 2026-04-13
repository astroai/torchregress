# SAGE-Reg — related work and experimental comparison map

This note ties **literature families**, **manuscript claims**, and **machine-readable benchmark rows** (CSV / `summary.json` from `examples/benchmarks/self_agreement_realdata_year.py`, Higgs track, multiseed scripts). It backs `main.tex` Table `tab:sage_baseline_map`.

## 0. Post-2023 deep semi-supervised regression (external SOTA)

These are **not** reimplemented as default Year rows today; cite them whenever the manuscript claims proximity to **current** regression SSL. They anchor the expanded Related Work in `main.tex`.

| Work | Venue / year | Core idea | Relation to SAGE-Reg |
|------|----------------|-----------|----------------------|
| Dai et al., contrastive + spectral seriation | NeurIPS 2023 | Ordinal / ranking structure from unlabeled data for continuous targets | Different signal: unlabeled **ordering** vs. our multi-view **distributional** agreement. |
| Huang et al., RankUp | NeurIPS 2024 | Auxiliary ranking classifier + regression distribution alignment | Strong SSR SOTA line; orthogonal loss template. |
| Sun et al., heteroscedastic pseudo-labels | NeurIPS 2025 (arXiv:2510.15266) | Bilevel reweighting of heteroscedastic pseudo-labels | Closest recent **pseudo-label uncertainty** line; we compare against a simpler scalar `ConfidenceWeightedPseudoLabel` row. |

## 1. Literature ↔ what we implement today

| Family | Representative references | In-repo status |
|--------|---------------------------|----------------|
| SSL survey / graph methods | Chapelle et al. | Context; no standalone graph baseline row. |
| Realistic SSL evaluation | Oliver et al. (2018) | Protocol discipline. |
| Consistency / VAT-style | Miyato et al. | Not a dedicated row; Mean Teacher covers teacher–student regression. |
| EMA teacher | Tarvainen & Valpola | **Implemented:** `MeanTeacher`. |
| Ladder / latent coupling | Rasmus et al. | Not implemented as a row. |
| Deep ensembles | Lakshminarayanan et al. | **Partial:** backbone comparison track. |
| Aleatoric / epistemic heads | Kendall & Gal | **Proxy:** scalar confidence in `ConfidenceWeightedPseudoLabel`. |
| Proposed method | (this work) | **Implemented:** `SAGE-Reg`. |

## 2. Primary Year / Higgs rows ↔ controlled contrast

| `Method` (JSON) | What is held fixed | What varies vs. SAGE-Reg | Closest prior-art bucket |
|-----------------|-------------------|---------------------------|---------------------------|
| `SupervisedOnly` | Labeled subset, backbone, epochs | Unlabeled data discarded | Supervised ERM |
| `MeanTeacher` | Same backbone & budgets | Consistency on predictive **mean**; EMA teacher | Mean Teacher (regression instantiation) |
| `ConfidenceWeightedPseudoLabel` | Same unlabeled pool | Trust from **scalar** weights | Conf.-weighted SSL; see also Sun et al.\ 2025 for modern heteroscedastic SSR |
| `SAGE-Reg` | — | Distributional agreement + detached trust | (proposed) |

## 3. Secondary tracks (same repository)

| Artifact | Role |
|----------|------|
| `self_agreement_synthetic.py` | Stress tests / confidence traps. |
| `self_agreement_backbone_comparison.py` | Gaussian vs. quantile vs. binned head **without** changing SSL method. |
| `self_agreement_supervised_gap_multiseed.py` | Multi-seed aggregation on tuned CSV rows. |
| `run_neurips_sage_reg_full.py` → `aggregate_sage_paper_report.py` | Dated bundle + `METRICS.md` digest. |

## 4. Deliberate non-goals (honest comparison boundary)

- **RankUp / spectral SSR / bilevel heteroscedastic pseudo-labels**: cited as SOTA; not yet benchmark rows in `torchregress`.
- **VAT / Pi-model / FixMatch-style** classification SSL: not ported to regression rows here.
- **Deep GP / full BNN**: not baseline rows in the current Year script.
- **Domain adaptation / shift**: sibling **SPT-Reg** track.
