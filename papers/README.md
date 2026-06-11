# Papers (`papers/`)

NeurIPS manuscript workspaces, literature notes, and **committed experiment artifacts**
(JSON/CSV/figures) for the SAGE-Reg and SPT-Reg paper tracks.

!!! note "Library vs papers"
    The installable **torchregress** library lives under `src/torchregress/`. Paper-scale
    batch runners and one-off aggregation scripts were removed from the repo root to keep
    contributor workflows focused on the library, tests, and docs. Reproduce paper numbers
    from the committed artifacts under each track's `results/` folder and `reports/neurips_*`.

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
