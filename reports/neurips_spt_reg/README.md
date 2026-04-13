# SPT-Reg Reports

This directory is reserved for SPT-Reg paper artifacts on the current branch.

Expected contents:

- benchmark summary JSON files
- rendered tables and figures
- run metadata for the NeurIPS manuscript

Primary entrypoint:

```bash
uv run python tools/render_spt_reg_paper_artifacts.py --profile smoke
```

The renderer writes namespaced summary JSON files and an artifact manifest to
this directory.

By default, the renderer writes the core `torchregress` SPT-Reg submission path:
synthetic + small-tabular + larger YearPredictionMSD-style real-data
benchmarks. The photo-z benchmark is optional here and can be included
explicitly with:

```bash
uv run python tools/render_spt_reg_paper_artifacts.py --profile smoke --include-photoz
```

That keeps the paper path in this repo aligned with the shift of heavier
photo-z effort into `torchz`.
