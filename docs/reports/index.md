# Reports

Generated evidence artifacts that back the method-selection guide and comparative examples.

| Report | Description |
|:-------|:------------|
| [Method Catalog](method_catalog_generated.md) | Auto-generated catalog of library methods with metadata |
| [Comparative Evidence Matrix](comparative_evidence_matrix.md) | Empirical evidence across hard-problem comparison examples |
| [Real-Data Recommendation Guide](real_data_recommendation_guide.md) | Data-driven method recommendations |
| [Docs Quality Audit](docs_quality_audit.md) | Per-file LaTeX/structure review status (generated) |

## Machine-readable artifacts

JSON snapshots live under `reports/` at the repo root:

- `reports/method_catalog_latest.json`
- `reports/comparative_evidence_matrix_latest.json`
- `reports/docs_quality_audit.json` — docs LaTeX/structure audit tracker
- `reports/native_pytorch_leverage_matrix_2026-02-26.json` — native-vs-custom policy matrix

Regenerate the docs-facing reports with:

```bash
uv run python tools/render_method_catalog.py \
  --markdown-out docs/reports/method_catalog_generated.md \
  --json-out reports/method_catalog_latest.json \
  --update-method-matrix docs/guide/method-selection.md \
  --comparative-evidence-md-out docs/reports/comparative_evidence_matrix.md \
  --comparative-evidence-json-out reports/comparative_evidence_matrix_latest.json
uv run python -m tools.render_realdata_recommendation_guide \
  --doc docs/reports/real_data_recommendation_guide.md \
  --comparative-json reports/comparative_evidence_matrix_latest.json
```
