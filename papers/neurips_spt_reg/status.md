# SPT-Reg Status

**Manuscript:** `papers/neurips_spt_reg/main.tex`. Build: `./papers/compile_tex.sh neurips_spt_reg` (uses vendored `papers/neurips_tex/neurips_2026.sty`).

**Shared roadmap (SPT + SAGE empirical priorities):** [docs/research/joint_empirical_priorities.md](../../docs/research/joint_empirical_priorities.md)

## Scope

This note tracks the current state of the `torchregress` NeurIPS paper path for
Shift-Factored Predictive Transport (SPT-Reg).

Current portfolio policy for this cycle:

- SPT-Reg is **research-first / submission-gated** while SAGE-Reg is the primary submission track.
- Do not promote SPT-Reg to a NeurIPS main-track submission unless a real-data matched-validity efficiency win is shown against `RawSplitConformal`, `WeightedSplitConformal`, and `TargetRefitSmall`.
- The explicit lock memo for this NeurIPS cycle lives in [docs/research/neurips_2026_submission_portfolio.md](../../docs/research/neurips_2026_submission_portfolio.md).

Working rule:

- stay on `main`
- keep paper assets in `papers/neurips_spt_reg/`
- keep generated artifacts in `reports/neurips_spt_reg/`
- keep shared method code in `torchregress/`
- avoid expanding the paper scope unless the empirical evidence improves

## Current Paper Position

The credible `torchregress` paper path is:

- synthetic shift benchmark
- small real tabular benchmark
- larger Year-style real tabular benchmark

The current paper should emphasize:

- predictive-law adaptation, not retraining
- calibration, coverage, selective prediction, and PPI
- competing methods, not just internal ablations
- honest separation between validity gains and efficiency gains

The current paper should **not** claim yet:

- that SPT-Reg already beats strong source baselines across all families
- that MDN is a solved in-repo flagship success path

## What Is Implemented

### Core method

Implemented in [transport.py](../../src/torchregress/test_time/transport.py):

- `ShiftFactoredTransportConfig`
- `ShiftFactoredTransportState`
- `ShiftFactoredPredictiveTransport`

Implemented method behavior:

- Stage A: output-prior transport on a shared support grid
- Stage B: significance-weighted alignment only when rerun is explicitly allowed
- Stage C: representation-aware uncertainty inflation
- Stage D: conformal calibration with family-specific routing
- Stage E: PPI wrappers over adapted predictive summaries

Guardrails already added:

- prior estimation shrinkage toward source prior
- prior-ratio clipping
- skip prior transport by default when EM prior estimation does not converge
- Gaussian family preservation through conformal
- no accidental interval shrinking from negative conformal scores

### Predictive-family support

Implemented in [prediction.py](../../src/torchregress/prediction.py):

- `PredictiveBatch`
- quantile-to-density conversion
- bar-to-density conversion
- sample-to-density conversion for sampled/MDN predictive laws

Current supported SPT-facing predictive families:

- Gaussian
- quantile
- bar / histogram
- ordered-bin / BinnedPDF
- sampled predictive laws such as MDN outputs

### Benchmarks and artifacts

Implemented examples:

- [spt_reg_synthetic_comparison.py](examples/spt_reg_synthetic_comparison.py)
- [spt_reg_realdata_comparison.py](examples/spt_reg_realdata_comparison.py)
- [spt_reg_year_comparison.py](examples/spt_reg_year_comparison.py)

Artifact renderer:

- [render_spt_reg_paper_artifacts.py](tools/render_spt_reg_paper_artifacts.py)

One-shot orchestration (renderer + bundled comparisons + optional large-tabular/shifts hooks):

- [run_neurips_spt_reg_full.py](scripts/run_neurips_spt_reg_full.py)

Current default artifact surface:

- [synthetic_competing_methods_smoke.json](reports/neurips_spt_reg/synthetic_competing_methods_smoke.json)
- [tabular_competing_methods_smoke.json](reports/neurips_spt_reg/tabular_competing_methods_smoke.json)
- [year_competing_methods_smoke.json](reports/neurips_spt_reg/year_competing_methods_smoke.json)
- [artifact_manifest_latest.json](reports/neurips_spt_reg/artifact_manifest_latest.json)

### Manuscript

Current paper source:

- [main.tex](papers/neurips_spt_reg/main.tex)
- [refs.bib](papers/neurips_spt_reg/refs.bib)
- [reproducibility.md](papers/neurips_spt_reg/reproducibility.md)
- [README.md](papers/neurips_spt_reg/README.md)

The manuscript is now single-file and already reflects:

- real-data-first default path in `torchregress`
- MDN as a supported extension, not a headline claim

## Current Empirical Status

Default checked-in JSON under `reports/neurips_spt_reg/` is still **smoke**-profile unless you regenerate.

**Authoritative full-profile + real OpenML Year (local run, 2026-04-11):**
`scripts/run_tabular_paper_bundle.sh` (non-smoke) produced:

- `docs/research/sage_reg_results/2026-04-11/tabular_paper_bundle/spt/full/year_competing_methods_full.json`
- `docs/research/sage_reg_results/2026-04-11/tabular_paper_bundle/spt/full/artifact_manifest.json`
- combined digest: `.../tabular_paper_bundle/tabular_paper_bundle_report.json` and `METRICS.md`

Cite these paths in the manuscript when stating **large-tabular** SPT numbers — **not** synthetic `year_local_dataset_full.csv`.

**Renderer-only command (alternative):**

```bash
uv run python tools/render_spt_reg_paper_artifacts.py --profile full \
  --year-cache-path docs/research/sage_reg_results/2026-04-10/openml_year.csv \
  --output-dir reports/neurips_spt_reg
```

**Long-run competitiveness plan** (Stage A sweeps, DA baselines, second shift dataset, multiseed): [docs/research/paper_strong_experiment_suite.md](../../docs/research/paper_strong_experiment_suite.md).

These are smoke-profile results below unless noted otherwise.

### Synthetic

Current state from [synthetic_competing_methods_smoke.json](reports/neurips_spt_reg/synthetic_competing_methods_smoke.json):

- `SourceGaussian` is still competitive and hard to beat cleanly.
- `SPTRegGaussian` is now stable and valid, but not a decisive win.
- `SourceMDN` is a strong backbone and improves over Gaussian on point and CRPS-style behavior in the current smoke run.
- `SPTRegMDN` is not yet a win over `SourceMDN`.
- `SourceBinnedPDF` and `SPTRegBinnedPDF` are currently not the main synthetic success story.

Interpretation:

- MDN is worth keeping as a competing family.
- MDN is not yet evidence that SPT-Reg improves multimodal predictive laws in-repo.
- the synthetic benchmark is useful, but it still does not give the paper a clean flagship win.

### Small real tabular

Current state from [tabular_competing_methods_smoke.json](reports/neurips_spt_reg/tabular_competing_methods_smoke.json):

- `SourceGaussian` remains strong.
- `SPTRegGaussian` is usually best interpreted as a more conservative valid interval method, not a clear efficiency win.
- `SourceBinnedPDF` remains a strong probabilistic baseline.
- `SPTRegBinnedPDF` currently widens substantially and is not yet persuasive.

Interpretation:

- this track is still useful for the paper because it is real data
- but the current SPT gain is modest and reviewer-fragile

### Year-style real tabular

Current state from [year_competing_methods_smoke.json](reports/neurips_spt_reg/year_competing_methods_smoke.json):

- `SourceGaussian` is very strong.
- `SPTRegGaussian` is now valid and no longer broken, but largely behaves like a conservative wrapper near raw conformal.
- `SPTRegBinnedPDF` has some interesting movement but is not yet a clear headline result.

Interpretation:

- this track supports the claim that the implementation is stable
- it does not yet support a strong claim that transport improves sharpness or NLL at fixed validity

## Main Blockers

### Method blocker

The main issue is no longer correctness. The main issue is **efficiency**:

- transport often restores or preserves validity
- transport does not yet reliably improve sharpness, NLL, or CRPS on the real-data tracks

### Paper blocker

The paper still lacks one clean in-repo flagship empirical claim of the form:

> under realistic shift, transported predictive laws outperform raw source predictions and raw conformal baselines on at least one real-data track in a way that is hard to dismiss as mere interval widening

### Scope blocker

The code now supports more predictive families than the current evidence supports.
The paper must stay narrower than the implementation surface.

## What To Implement Next

Priority order:

1. Improve Stage A selectivity on real-data tracks.
   Focus on when prior transport should be weakened or skipped, especially when target-prior evidence is weak.

2. Add a family-matched raw conformal comparator for MDN.
   This is needed before claiming anything about SPT on sampled predictive laws.

3. Improve transport diagnostics.
   Add explicit per-run metadata/plots for:
   - whether transport was applied
   - shrink weight
   - target-prior distance from source prior
   - conformal method chosen
   - interval-width change versus source

4. Separate transport and conformal more clearly in benchmark rows.
   Add explicit `SPTTransport...` rows where helpful so the tables do not hide whether the gain comes from transport or just conformal widening.

5. Do not add more backbone families until the current real-data story improves.

## What To Experiment Next

### Immediate experiments

1. Run focused sweeps on Stage A parameters for the two real-data tracks:
   - `prior_transport_strength`
   - `prior_ratio_clip`
   - `top_fraction`
   - `min_selection_count`
   - convergence-required versus softened transport

2. Compare these rows explicitly on real tabular:
   - `SourceGaussian`
   - `RawSplitConformalGaussian`
   - `PriorTransportGaussian`
   - `SPTTransportGaussian`
   - `SPTRegGaussian`

3. Add the same decomposition for BinnedPDF on the real-data tracks.

4. Add `SourceMDN` versus raw conformal MDN versus `SPTRegMDN` on synthetic first.
   Only move MDN into the paper’s main experimental story if SPT beats the raw MDN baselines on at least one meaningful criterion.

### Near-term experiments

1. Re-run the default artifact pipeline after each method change:
   - synthetic
   - small tabular
   - Year-style tabular

2. Once a promising configuration exists, run at least an `audit`-style pass for SPT artifacts instead of relying only on `smoke`.

## What To Write Next

### Immediate writing tasks

1. Tighten the introduction and method claims in [main.tex](papers/neurips_spt_reg/main.tex) around what is already defensible:
   - predictive-law adaptation
   - family-agnostic transport interface
   - validity versus efficiency distinction

2. Add a short paragraph in the experiments section explicitly stating:
   - real-data tabular is the default in-repo evidence path
   - MDN is currently an extension family, not the primary real-data claim path

3. Draft a results paragraph template that can be filled from the JSON artifacts:
   - where SPT helps
   - where it only improves validity
   - where it does not yet win

### After the next method pass

1. Write one main table around real-data competing methods.
2. Write one synthetic ablation table.
3. Write one short limitations paragraph that explicitly names non-transportable shifts and interval-widening failure modes.
4. Keep the theory section compact:
   - one theorem-level statement
   - two or three short propositions
   - proof sketches in appendix

## Recommended Short-Term Plan

### Step 1

Stabilize and improve the real-data Gaussian path by sweeping Stage A selectivity and adding explicit transport-versus-conformal rows.

### Step 2

Only after Step 1, decide whether MDN deserves a main-paper row set or should remain a supporting extension.

### Step 3

Update `main.tex` to match the evidence exactly, then produce a cleaner artifact-backed results draft.

## Bottom Line

SPT-Reg is now a real method path in `torchregress`, not just a blueprint.
The code, examples, artifacts, and manuscript scaffold are all present.

What is still missing is not infrastructure. It is a **competitive** empirical story on **real** data: **explicit** comparison to **raw conformal** and **domain-adaptation** baselines, **multiseed** variance, and (when possible) **efficiency** gains rather than only **validity** / interval widening — see [paper_strong_experiment_suite.md](../../docs/research/paper_strong_experiment_suite.md).
