# SAGE-Reg Experiment Plan

## Objective

Validate the narrow claim:

> predictive laws that remain stable across perturbations are safer unlabeled supervisory signals than confidence-only predictions.

## Stage Ladder

## Current Status

- Stage 1: done
- Stage 2: done for the paper-v1 prototype surface
- Stage 3: done on synthetic confidence-trap stress tests
- Stage 4a: done as a first real-data sanity check on `year`
- Stage 4b: in progress via the FAIR Universe Higgs public-data benchmark wrapper

Paper-v1 backbone coverage is now explicit:

- Gaussian pseudo-loss: analytic Gaussian cross-entropy / NLL-to-consensus Gaussian
- quantile pseudo-loss: density cross-entropy on a shared support grid
- bar / binned PDF pseudo-loss: PMF cross-entropy on student bins

Current narrative:

- SAGE-Reg is clearly safer than confidence-weighted pseudo-labeling.
- Gaussian remains the strongest backbone.
- quantile and bar are supported end-to-end, but remain more representation-sensitive.

### Stage 1

Single synthetic benchmark with a Gaussian head.

Required baselines:

- supervised only
- point pseudo-label self-training
- confidence-weighted pseudo-labeling
- SAGE-Reg

Success condition:

- SAGE-Reg improves calibration, CRPS, or interval quality even if RMSE gains are small.

### Stage 2

Same synthetic task, add:

- quantile head
- bar / binned PDF head

Success condition:

- the same method logic works across predictive families without bespoke rewrites.

### Stage 3

Stress tests:

- higher aleatoric noise
- epistemic holes
- multimodal targets
- cases where confidence and true pseudo-label quality are mismatched

Success condition:

- agreement weighting beats confidence gating when confidence is misleading.

### Stage 4

One real tabular benchmark.

Success condition:

- at least modest gains on calibration/coverage with a consistent story.

Current recommended order:

- Stage 4a: OpenML/UCI `year` (YearPredictionMSD) as the first large IID real-data SSL benchmark.
- Stage 4b: FAIR Universe Higgs Uncertainty Challenge as the first dedicated uncertainty/OOD benchmark.

### Stage 5

Close the real-data gap to `SupervisedOnly`.

Primary question:

- can SAGE-Reg beat or at least match `SupervisedOnly` on one real-data setting while preserving the safety advantage over confidence-weighted pseudo-labeling?

Required tuning axes:

- `tau`
- perturbation scale / type
- pseudo weight
- agreement weight
- EMA vs no EMA if needed

Current implementation path:

- `examples/benchmarks/self_agreement_supervised_gap_tuning.py`

Current targeted refinement:

- stronger tabular perturbations via feature masking/dropout
- tempered disagreement weights via `weight_power`
- optional hard trust gate via `hard_weight_threshold`

Success condition:

- at least one real-data setting where SAGE-Reg improves NLL, CRPS, or coverage relative to `SupervisedOnly`
- while still clearly outperforming confidence-weighted pseudo-labeling

## Must-Have Ablations

1. confidence gating vs self-agreement gating
2. point pseudo-label vs distributional pseudo-label
3. single-view teacher vs multi-view consensus
4. no weighting vs agreement weighting
5. EMA teacher vs no EMA

## Metrics

- RMSE
- MAE
- Gaussian NLL or pseudo-NLL where applicable
- CRPS when available
- coverage
- interval width
- calibration summary
- disagreement-weight histograms

## Figure Plan

1. Main performance vs unlabeled fraction.
2. Calibration or coverage figure.
3. Agreement-weight histogram split by downstream error.
4. Ablation bar chart.

## Failure Criteria

- gains appear only for one backbone
- improvement is only RMSE noise with no calibration story
- agreement behaves like confidence ranking under all tested regimes
