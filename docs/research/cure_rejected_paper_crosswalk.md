# Crosswalk: CURE rejected manuscript → torchregress paper experiments

Source: Overleaf project **`~/src/overleaf/cure`** (`neurips_2025.tex`, `appendix.tex`, `references.bib`).  
CURE = **Classification-guided self-training** with **Regression-as-Classification (RaC)**, **soft pseudo-labels**, **curriculum** thresholding, and **denoising-style self-supervision** on tabular SSR.

This note is **ideas only** — no code from CURE was copied. Use it to align **datasets**, **metrics**, and **competitor lists** with SAGE-Reg / SPT-Reg.

---

## Datasets (high priority for “second IID + scale”)

CURE evaluates on **three TabReD** regression tasks (large \(10^5\)–\(10^6\) rows):

| TabReD name (paper) | Role |
|---------------------|------|
| **Cooking Time** | Large industrial regression |
| **Delivery ETA** | Logistics / temporal structure |
| **Maps Routing** | Geospatial-style tabular |

- **Reference:** Rubachev et al., *TabReD: A Benchmark of Tabular Machine Learning in-the-Wild*, arXiv:[2406.19380](https://arxiv.org/abs/2406.19380) (`references.bib`: `Rubachev2024`).
- **Excluded in CURE:** TabReD *Weather* (discrete-ish target), *Sberbank Housing* (too small for SSL story).

**For torchregress:** these are strong candidates for a **second / third real tabular track** beside OpenML Year — same narrative as `paper_strong_experiment_suite.md` Tier A “second IID OpenML” but **TabReD is the CURE-proven choice**. Prefer **time-based splits** as in TabReD rather than purely random splits when claiming “in-the-wild”.

**Related bib (OOD tabular):** *Wild-Tab* arXiv:[2312.01792](https://arxiv.org/abs/2312.01792) (`Wild-tab` in `references.bib`) — useful for **SPT-Reg** OOD generalization language and extra datasets.

---

## Semi-supervised split + shift protocol (reusable idea)

From `neurips_2025.tex` §Data:

- Keep TabReD **validation / test** as defined in the benchmark.
- Split **training** into **labeled** \(\mathcal{D}_L\) vs **unlabeled** \(\mathcal{D}_U\) (several **label ratios**, e.g. 0.05 in main table).
- **Shift control:** divide \(\mathcal{D}_L\) into two halves — one is the “manipulable” labeled pool for **induced shift**, the other merges into training with \(\mathcal{D}_U\); **unlabeled distribution stays fixed** to isolate **labeled-pool shift**.

**For torchregress:** SAGE could report **IID** (no shift) vs **labeled-shift** as two rows; SPT already speaks covariate shift — this is a **different knob** (who is labeled), easy to describe honestly.

---

## Metrics (extend beyond NLL-only)

CURE’s §Metrics includes:

- **Point:** MSE, mean bias, **NMAD**, **outlier rate** (NMAD-based 3σ rule).
- **Probabilistic:** **CRPS**, **90% interval coverage & width**, **ECE**, **calibration curves**, **PIT** histograms.

**For torchregress:** you already report CRPS / Cov90 / CalibMAE in several benchmarks — adding **NMAD**, **outlier rate**, and **PIT** (where a density is defined) would **match reviewer expectations** set by tabular UQ papers and make comparison to a future **CatBoost(RaC)** row fairer.

---

## Baselines named in CURE (competitiveness checklist)

Main table (`tab:combined_metrics`) compares:

| Method | Notes |
|--------|--------|
| **CatBoost(Reg)** | Strong point MSE; no full prob row in table |
| **CatBoost(RaC)** | Better calibration / CRPS / coverage; worse MSE than Reg — **same tension** SPT sees (validity vs sharpness) |
| **RealMLP** | Strong MLP tabular baseline |
| **meanTeacher**, **ICTReg**, **piModel** | SSL-style; paper notes underperformance vs CURE in their setting (label ratio 0.05, large \(n\)) |

**For torchregress long runs:** CatBoost baselines are wired through `scripts/run_neurips_sage_reg_full.py` → `tools/sage_catboost_baselines.py`. Next step for parity with CURE: **CatBoost with explicit RaC / binned target** (or use **ordered-bin / BinnedPDF** head in SAGE as the internal RaC analogue) and **RealMLP** supervised ceiling on the **same splits**.

---

## Methodological overlap (positioning, not implementation)

| CURE ingredient | torchregress analogue |
|-----------------|------------------------|
| RaC + PMF | `BinnedPDF` / quantile / MDN heads; conformal on bins (SPT) |
| Soft pseudo-labels | SAGE distributional agreement + weights (different construction) |
| Self-supervised recon | Not central in current SAGE examples — optional extension |
| Curriculum PL | `unlabeled_fractions`, batch-relative / top‑k gating |

Useful **wording**: CURE argues **RaC is necessary** for calibration; your Gaussian Year + Higgs story should **explicitly** say whether the claim is **Gaussian NLL** vs **binned / full-law** calibration (avoid apples-to-oranges vs CatBoost(RaC)).

---

## Morning launcher (torchregress)

The morning script **fetches** TabReD by default into **`data/tabred/`** (clone + patch upstream `DATA_DIR` + run `preprocessing/*.py` with Kaggle). Needs `~/.kaggle/kaggle.json` and `uv pip install polars kaggle loguru scikit-learn`.

```bash
./scripts/morning_tabred_bundle.sh              # fetch (skip-if-present) + SSL probe
TABRED_FETCH_ONLY=1 ./scripts/morning_tabred_bundle.sh   # download/preprocess only
SKIP_TABRED_FETCH=1 ./scripts/morning_tabred_bundle.sh   # probe only (data already there)
```

Manual fetch: `uv pip install -e '.[tabred]'` then `uv run python tools/fetch_tabred_data.py --out-dir data/tabred --skip-if-present`.

The TabReD SSL probe always uses **Polars** to assemble wide feature matrices (same stack as upstream preprocessing); install `[tabred]` before running it.

Benchmark: `examples/benchmarks/tabred_sage_ssl_probe.py` → `bundle_summary.json`, `results_long.csv`, per-dataset `rows.csv`.

## Practical next steps (ordered)

1. **Materialize TabReD** (or one dataset first) with official **time split**; use `tabred_sage_ssl_probe.py` / `morning_tabred_bundle.sh` for structured outputs (extend metrics there as needed).
2. Add **NMAD + outlier rate** (and optionally PIT) to the collate / report tools for that runner.
3. **CatBoost(RaC)** baseline: binned target + multiclass / ordinal CatBoost, matched bin count to SAGE binned head if you run that variant.
4. Cite **TabReD** + **Wild-Tab** in `papers/neurips_*_reg` only when numbers exist from (1).

---

## File pointers (local)

- Manuscript: `~/src/overleaf/cure/neurips_2025.tex` — §Experiments, §Metrics, §Data, Table `tab:combined_metrics`.
- Bib: `~/src/overleaf/cure/references.bib` — `Rubachev2024`, Wild-Tab entry.
