# Papers (`papers/`)

All NeurIPS manuscript workspaces, research memos, experiment results, and operational
infrastructure for the torchregress paper tracks.

## Paper Tracks

| Track | Folder | Topic |
|-------|--------|-------|
| **SAGE-Reg** | [neurips_sage_reg/](neurips_sage_reg/) | Semi-supervised regression via distributional self-agreement across views |
| **SPT-Reg** | [neurips_spt_reg/](neurips_spt_reg/) | Test-time shift-factored transport of predictive laws + conformal / PPI |
| **CURE** (rejected) | [neurips_cure/](neurips_cure/) | Datasets / metrics / baselines crosswalk from the rejected CURE paper |

## Shared Resources

Cross-paper planning, CANFAR operations, and dataset registry live in [shared/](shared/):

| File | Purpose |
|------|---------|
| `submission_portfolio.md` | NeurIPS 2026 allocation and submit gates |
| `empirical_priorities.md` | Cross-track empirical roadmap |
| `experiment_suite.md` | Long-run competitiveness suite (baselines, multi-dataset, tiers) |
| `dataset_registry.md` | Shared dataset registry across tracks |
| `canfar_batch_sweeps.md` | CANFAR batch: Year label-fraction sweep (40 shards) |
| `canfar_neurips_batch.md` | CANFAR batch: full NeurIPS SAGE-Reg + overnight scope |

## Per-Paper Layout Convention

Each paper folder follows:

- **`main.tex`** — single-file manuscript (NeurIPS 2026 style)
- **`refs.bib`** — bibliography
- **`reproducibility.md`** — map figures/tables to scripts and artifact paths
- **`status.md`** — claim boundary, experiment stage, stop/go (living document)
- **`related_work_notes.md`** — literature notes
- **`README.md`** — layout + working rules

## LaTeX Build

Vendored style: [neurips_tex/README.md](neurips_tex/README.md). From repo root:

```bash
./papers/compile_tex.sh neurips_sage_reg
./papers/compile_tex.sh neurips_spt_reg
```

Build artifacts (`main.pdf`, `.aux`, `.log`, …) are gitignored.
