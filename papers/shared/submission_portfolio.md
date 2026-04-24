# NeurIPS 2026 submission portfolio lock

## Locked allocation

- **Primary submission:** `SAGE-Reg`.
- **SPT-Reg:** research-only by default for this cycle.
- **Benchmark paper:** reserve path only if SAGE misses submit gates; do not run in parallel with SAGE.

## Why this lock

- Current in-repo evidence is strongest for SAGE safety and label-scarce IID signal.
- SPT remains limited by real-data efficiency competitiveness versus conformal/refit comparators.
- Running SAGE + SPT + benchmark in one cycle risks experimental and writing dilution.

## SAGE submit gates

1. Multi-seed (`>=5`) low-label Year evidence is stable enough for bounded claims.
2. External SSL comparators are included (`MeanTeacher`, `PiModelConsistency`, scalar confidence, supervised).
3. Second real tabular dataset direction is coherent.
4. IID and OOD stories remain explicitly separated in text and tables.

## SPT promotion gates (required before submission)

1. At least one real benchmark with matched-validity efficiency win versus:
   - `RawSplitConformal...`
   - `WeightedSplitConformal...`
   - `TargetRefitSmall...`
2. Multi-seed stability for that win.
3. Diagnostics that explain when transport helps vs when conformal widening dominates.

## Image track policy

- Keep image regression as an optional rebuttal/appendix pack only.
- Use it to support cross-modality sanity, not to replace tabular core evidence.
