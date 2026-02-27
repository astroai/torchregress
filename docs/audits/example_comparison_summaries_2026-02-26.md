# Example Comparison Summaries Audit (2026-02-26)

This audit records profile-comparison and threshold-governance artifacts used by CI/review.

## Artifacts

- Profile comparison:
  - `/Users/fabbros/src/torchregress/reports/example_summaries/profile_comparison_audit_vs_full.json`
- Conservative CI thresholds:
  - `/Users/fabbros/src/torchregress/reports/example_summaries/thresholds_full.json`
  - `/Users/fabbros/src/torchregress/reports/example_summaries/threshold_check_full_latest.json`
- Strict review thresholds:
  - `/Users/fabbros/src/torchregress/reports/example_summaries/thresholds_full_review_strict.json`
  - `/Users/fabbros/src/torchregress/reports/example_summaries/threshold_check_full_review_strict_latest.json`

## Threshold Profiles

- `ci_conservative`: default blocking policy for CI stability
- `review_strict`: tighter policy for human review and release gating

## Regeneration Commands

```bash
uv run python tools/render_example_summaries.py --profile smoke
uv run python tools/render_example_summaries.py --profile audit
uv run python tools/render_example_summaries.py --profile full
uv run python tools/compare_example_summary_profiles.py --base-dir reports/example_summaries --source-profile audit --target-profile full --output reports/example_summaries/profile_comparison_audit_vs_full.json
uv run python tools/example_summary_thresholds.py --base-dir reports/example_summaries --profile full --threshold-profile ci_conservative --write-thresholds reports/example_summaries/thresholds_full.json --output-verdict reports/example_summaries/threshold_check_full_latest.json
uv run python tools/example_summary_thresholds.py --base-dir reports/example_summaries --profile full --threshold-profile review_strict --write-thresholds reports/example_summaries/thresholds_full_review_strict.json --output-verdict reports/example_summaries/threshold_check_full_review_strict_latest.json --runtime-multiplier 6.0 --runtime-floor 0.35 --metric-multiplier 3.0 --metric-floor 0.2 --prob-delta 0.25 --r2-delta 1.0
```
