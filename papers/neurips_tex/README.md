# Vendored NeurIPS LaTeX style

This directory holds a **vendored** copy of the CTAN `neurips` package style so manuscripts build without relying on a matching TeX Live year.

| File | Purpose |
|------|---------|
| `neurips_2024.sty` | Upstream-compatible NeurIPS 2024 style (downloaded for reference; see header for source). |
| `neurips_2026.sty` | **Use this** with `\usepackage[preprint]{neurips_2026}` in `papers/neurips_*_reg/main.tex`. Derived from `neurips_2024.sty` with year/ordinal/location updated for 2026 drafts and non-track boilerplate in the footer. |

For the **authoritative** camera-ready file before submission, replace this tree with the ZIP from [neurips.cc](https://neurips.cc) for the target year.

## Build (from repo root)

```bash
export TEXINPUTS="$(pwd)/papers/neurips_tex//:${TEXINPUTS}"
cd papers/neurips_sage_reg
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Same sequence under `papers/neurips_spt_reg/`.

## License

The NeurIPS style is distributed on CTAN under the LaTeX Project Public License (LPPL). Keep this README when redistributing.
