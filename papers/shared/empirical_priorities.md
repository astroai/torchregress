# Joint empirical priorities (SAGE-Reg + SPT-Reg)

Single memo for strengthening **both** NeurIPS-scale tracks. The methods differ; the **reviewer bar** is similar: credible gains on real tabular data without hiding variance, shortcuts, or synthetic-only “Year” tables.

**Long-run experiment plan** (external baselines, extra datasets, execution order): [paper_strong_experiment_suite.md](paper_strong_experiment_suite.md).

## SAGE-Reg — semi-supervised vs supervised-only

**Aim:** Show that **self-agreement training** improves (or matches) **SupervisedOnly** on a **clean IID benchmark** (Year-class), while keeping the **stress** story (Higgs OOD NLL, confidence-weighted failure).

**Current pain points**

- **Year:** Mean NLL across three seeds still **slightly favors supervised**; gains are seed-noisy.
- **Higgs (large):** **SAGE OOD NLL is stable**; **supervised OOD NLL is high-variance** — strong for a *robustness-under-shift* narrative, weaker if the only claim is “always beats sup on NLL.”
- **Protocol:** Higgs uses **binary `labels` with a Gaussian head**; referees may push back unless we acknowledge or extend the target law.

**Concrete next moves (priority order)**

1. **More seeds + robust summaries** on Year (e.g. 5–10 seeds): report **mean, std, median** for `SAGEMinusSupervised` on NLL.
2. **Re-tune hyperparameters at the evaluation budget** (epochs + split sizes) used in the paper — not only the smaller sweep that picked the row.
3. **Ablate library knobs** already implemented: **batch-relative** disagreement scaling, **top‑k** trust gating — wire into `self_agreement_supervised_gap_confirm.py` / multiseed when an ablation CSV exists.
4. **Second IID tabular dataset** (same backbone budget) so Year is not a single point.
5. **Optimization hygiene at large n:** LR schedule, weight decay, optional early stopping on a small labeled val slice — reduces “supervised got lucky / unlucky” noise.
6. **Optional Higgs extension row:** Bernoulli NLL/Brier or a continuous column — only if time; otherwise **one honest limitations sentence** in the paper.

**Primary scripts:** `examples/benchmarks/self_agreement_supervised_gap_multiseed.py`, `self_agreement_realdata_year.py`, `self_agreement_higgs_ood.py`.

---

## SPT-Reg — adaptation / transport vs source

**Aim:** Show that **shift-factored predictive transport** (plus conformal where used) improves **target** probabilistic quality **without** only widening intervals, on **real** data splits.

**Current pain points**

- **Renderer “Year” JSONs** from `render_spt_reg_paper_artifacts.py` use **`year_local_dataset_<profile>.csv` (synthetic)** — good for CI and plumbing, **not** a substitute for **OpenML Year** in the manuscript.
- **Real-data tracks:** Often **validity/coverage** improves while **NLL / width** look like **conservative wrapping** — easy for reviewers to dismiss as “just conformal widening.”

**Concrete next moves (priority order)**

1. **One authoritative OpenML Year run** with paper budgets (`full`-scale `n_*` or `spt_reg_year_comparison.py --scale-split-factor`): write `*_summary.json` under `reports/` or `docs/research/` and **cite that path** in `main.tex`.
2. **Stage A selectivity sweep** on real tabular (Gaussian row): `prior_transport_*`, `prior_ratio_clip`, evidence thresholds — goal is **sharper** adapted laws when transport is trustworthy.
3. **Decomposed table rows** already in code (`SPTTransportGaussian`, `RawSplitConformalGaussian`, …): **force the paper text** to say what moves (transport vs conformal vs inflation).
4. **Diabetes / small real track:** same decomposition + one tuned config — keeps “two real tracks” honest.
5. **Renderer flags (implemented):** `render_spt_reg_paper_artifacts.py` supports `--year-cache-path`, `--year-allow-download`, and `--year-dataset-path`; see `papers/neurips_spt_reg/reproducibility.md` for copy-paste commands.

**Primary scripts:** `examples/spt_reg_year_comparison.py`, `spt_reg_realdata_comparison.py`, `tools/render_spt_reg_paper_artifacts.py`.

---

## Shared writing discipline

- Report **variance across seeds** wherever training is stochastic.
- Separate **IID** (Year-class) from **shift-stress** (Higgs, SPT covariate shift) claims.
- **Never** label synthetic `year_local_dataset_*.csv` results as “OpenML Year” in prose.

## Pointers

- SAGE status: [papers/neurips_sage_reg/status.md](../../papers/neurips_sage_reg/status.md)
- SPT status: [papers/neurips_spt_reg/status.md](../../papers/neurips_spt_reg/status.md)
- SAGE reproducibility: [papers/neurips_sage_reg/reproducibility.md](../../papers/neurips_sage_reg/reproducibility.md)
- SPT reproducibility: [papers/neurips_spt_reg/reproducibility.md](../../papers/neurips_spt_reg/reproducibility.md)
