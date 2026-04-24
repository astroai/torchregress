# Strong paper experiment suite (SAGE-Reg + SPT-Reg)

This memo defines a **serious, long-run** empirical program aimed at **reviewer-grade competitiveness**: fixed protocols, **external baselines**, multiple real tabular datasets, and **pre-declared** primary metrics so results are not retrofitted.

**Companion:** [joint_empirical_priorities.md](joint_empirical_priorities.md) (shorter priority list).

---

## 0. Principles (non-negotiable)

1. **Primary metrics declared before runs** (e.g. target NLL + CRPS + Cov90 on a fixed test slice; for shift tracks, separate **target-domain** metrics).
2. **Seeds:** minimum **5**; report **mean, std, median** for any gap vs baseline.
3. **Baselines must include methods readers know** — not only ablations inside our trainers.
4. **Same compute budget** where comparability is claimed (epochs, model class, optimizer family); document when a baseline is intentionally stronger (e.g. tuned GBM).
5. **IID vs shift:** never merge claims — Year-class IID semi-sup is a different proposition from Higgs OOD or SPT covariate shift.

---

## 1. SAGE-Reg — competitiveness package

### 1.1 Current evidence to preserve (do not drop)

- **Regime curve (Year, single seed, default hypers):** `n_labeled` sweep shows SAGE **wins NLL** at **2048–4096** labeled (fixed `n_unlabeled=131072`), **loses** at **8192+**. Collated: `tools/collate_sage_year_labeled_sweep.py` on `year_direct_nl*_summary.json`.
- **Tuned-hyper multiseed:** sign of Year NLL gap **depends** on protocol (tuning row vs `YearRealDataConfig` defaults) — paper must **one-hot** which protocol each table uses.
- **Higgs large:** stable SAGE OOD NLL vs high-variance supervised — **robustness** story with **binary-label + Gaussian-head** caveat.

### 1.2 Tier A — long runs (core paper)

| ID | Experiment | Goal |
|----|------------|------|
| S-A1 | **Year multiseed × `n_labeled` grid** | For each `nl ∈ {2048, 4096, 8192}`, run **7–10 seeds**, same tuned row + same epoch budget. Primary: **NLL gap** (SAGE − sup); secondary: CRPS, Cov90. |
| S-A2 | **Re-tune at paper budget** | New sweep with **`year_n_*` and epochs matching the paper** (not the old 16-ep grid); then confirm + multiseed on **best row**. |
| S-A3 | **Second large tabular regression** | Prefer **TabReD** (*Cooking Time*, *Delivery ETA*, *Maps Routing*; arXiv:2406.19380) with **time-based splits**; alternatively OpenML (e.g. protein 503, Mercedes). Same MLP backbone and semi-sup protocol; **≥2 seeds**. Crosswalk from CURE draft: [cure_rejected_paper_crosswalk.md](cure_rejected_paper_crosswalk.md). |
| S-A4 | **External SSL baselines (same backbone budget)** | Implement or wrap **Π-model / Mean Teacher** (same Gaussian head + RMSE/NLL) on Year with **identical** labeled/unlabeled split. Goal: show SAGE is not “only beating pseudo-label.” |
| S-A5 | **Strong supervised ceiling (same labels only)** | **CatBoost** (and optionally XGBoost/LightGBM) trained **only on labeled** rows (no unlabeled). **`scripts/run_neurips_sage_reg_full.py`** runs the **Year** labeled-only CatBoost phase via `tools/sage_catboost_baselines.py` (`RMSEWithUncertainty`, NLL on normalized test target) unless `--skip-catboost`. |

**Deliverables:** one CSV per (dataset, nl, seed); `multiseed_summary.json` pattern; figure: **x = n_labeled**, **y = NLL gap**, error bars over seeds.

### 1.3 Tier B — depth (appendix / rebuttal)

| ID | Experiment | Goal |
|----|------------|------|
| S-B1 | **Batch-relative + top-k gating** ablation grid | Already in library; grid over modes at fixed `nl` where single-seed SAGE wins. |
| S-B2 | **Weight decay / LR schedule** | Reduce supervised “variance” artifact at large `nl`. |
| S-B3 | **Higgs:** +5 seeds, report **median** supervised OOD NLL; optional **Brier** on `labels` alongside Gaussian NLL (honest dual row). **CatBoost** Higgs tracks (classifier log-loss ID/OOD + `RMSEWithUncertainty` regression aligned with the Gaussian-head protocol) run from the same full script / `tools/sage_catboost_baselines.py`; paste official scores via `--external-scores-json` (template: `docs/research/higgs_external_scores.template.json`). |
| S-B4 | **Extra-large OpenML / TabReD**: default **diamonds 42225** (~54k); optional heavier cache (TabReD or verified parquet) with same MLP protocol — only after Tier A is stable. |

### 1.4 OpenML / infra note

The default **extra-large** OpenML regression track in the SPT full runner is **42225 (diamonds)**, ~54k rows — **stable** under sklearn and large enough for a serious second tabular stress test without Yolanda’s MD5 issues. **Mitigations** for other ids: ``torchregress.utils.openml_relaxed`` (ARFF without MD5) or ``tools/materialize_openml_large_tabular.py --data-id …``. For an even heavier custom source (e.g. TabReD), keep it as a **separate cited cache**, not OpenML roulette.

---

## 2. SPT-Reg — competitiveness package

### 2.1 Current evidence

- **Full-profile render on real OpenML Year cache** (not `year_local_dataset_*.csv`) is the authoritative large-tabular track; see `scripts/run_tabular_paper_bundle.sh` output under `docs/research/sage_reg_results/.../tabular_paper_bundle/spt/full/`.
- Narrative is often **validity vs sharpness**: compare **RawSplitConformalGaussian** vs **SPTRegGaussian** explicitly in text.

### 2.2 Tier A — long runs

| ID | Experiment | Goal |
|----|------------|------|
| P-A1 | **Stage A selectivity sweep** | Grid on `prior_transport_strength`, `prior_ratio_clip`, selection thresholds; **real Year + real diabetes**; primary: **target CRPS + Cov90** with NLL secondary. |
| P-A2 | **Domain adaptation baselines** | Add **CORAL** (feature alignment), **importance-weighted** source risk, or a lightweight **DANN**-style linear head — same Gaussian source model class where possible. Goal: SPT must beat or match **known DA** on at least one track. |
| P-A3 | **Second shift dataset** | e.g. **Bike sharing** with artificial covariate shift on time/season feature, or OpenML dataset with explicit train/test covariate shift — same SPT pipeline. |
| P-A4 | **Multiseed (≥5)** for the best configuration from P-A1 | Report variance on target NLL/CRPS/Cov90. |

### 2.3 Tier B

| ID | Experiment | Goal |
|----|------------|------|
| P-B1 | **MDN track:** family-matched **RawConformalMDN** vs **SPTRegMDN** only if synthetic shows lift first. |
| P-B2 | **Diagnostics artifact:** per-run JSON with transport on/off, shrink, conformal width delta (already partially in rows; extend). |

---

## 3. Automation (repo)

- **NeurIPS one-shot (SPT then SAGE, same flags):** `scripts/run_neurips_paper_bundle.sh` — forwards e.g. `--quick` to `scripts/run_neurips_spt_reg_full.py` and `scripts/run_neurips_sage_reg_full.py` (do not pass `--run-root` here).
- **Paper bundle only:** `scripts/run_tabular_paper_bundle.sh`.
- **Labeled sweep merge:** `tools/collate_sage_year_labeled_sweep.py`.
- **Next coding tasks for competitiveness:** thin wrappers or new `examples/benchmarks/` scripts for **GBM labeled-only**, **Mean Teacher** baseline, and **SPT + CORAL** — each with CLI matching existing summary JSON schema where possible.

### Image / vision regression (optional future data)

torchregress examples are **tabular-first**; vision regression is optional glue work (CNN backbone + Gaussian head). Candidate **ID-friendly** baselines if you add data: **UTKFace** or **IMDB-WIKI** (age from crops), **NYUv2 depth** (scale-normalized regression), or small **dSprites**-style synthetic \(y\) from latents. Treat as a **separate track** from Year/Higgs/Shifts so claims stay protocol-specific.

---

## 4. Suggested execution order (wall-clock aware)

1. **S-A5** (GBM ceiling) — cheap, anchors reader expectations.  
2. **S-A1** (multiseed × nl) — expensive but highest scientific value for SAGE.  
3. **S-A2** (re-tune) — only if S-A1 mean still crosses zero at `nl=4096`.  
4. **P-A1** + **P-A4** — SPT.  
5. **S-A3 / P-A3** — second dataset each track.  
6. **S-A4 / P-A2** — external method parity (largest engineering lift).

---

## 5. What “strong paper” means here

Not “every metric wins,” but:

- **One** IID semi-sup story: **where** SAGE helps (label-scarce regime) + **variance** across seeds + **honest** ceiling vs GBM on labels-only.  
- **One** shift story: SPT or SAGE-on-Higgs with **decomposed** transport vs conformal vs baselines.  
- **Zero** reliance on synthetic Year CSV posing as OpenML.

Update `papers/neurips_*_reg/status.md` after each tier completes.
