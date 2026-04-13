# NeurIPS-track papers (`papers/`)

Two parallel manuscript workspaces in **torchregress**, each with the same layout convention.

| Track | Folder | One-line topic |
|-------|--------|----------------|
| **SPT-Reg** | [neurips_spt_reg/](neurips_spt_reg/) | Test-time shift-factored transport of predictive laws + conformal / PPI |
| **SAGE-Reg** | [neurips_sage_reg/](neurips_sage_reg/) | Semi-supervised regression via distributional self-agreement across views |

Shared conventions:

- **`main.tex`** — single-file manuscript (NeurIPS 2026 style when `neurips_2026.sty` is on `TEXINPUTS`).
- **`refs.bib`** — bibliography for that paper only (standalone `pdflatex` + `bibtex` per folder).
- **`reproducibility.md`** — map figures/tables to scripts and artifact paths.
- **`status.md`** — claim boundary, experiment stage, stop/go (living document).
- **`README.md`** — layout + working rules + link to the sibling track.

Cross-cutting registry: [neurips_dataset_registry.md](neurips_dataset_registry.md).

Research memos and dated result trees often live under [docs/research/](../docs/research/README.md).

## LaTeX build

Vendored style: [neurips_tex/README.md](neurips_tex/README.md). From repo root:

```bash
./papers/compile_tex.sh neurips_sage_reg
./papers/compile_tex.sh neurips_spt_reg
```

Build artifacts (`main.pdf`, `.aux`, `.log`, …) are gitignored under each paper folder.
