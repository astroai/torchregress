# SAGE-Reg Claims And Risks

## Claims We Can Defend After the Current Prototype

- The code supports a narrow SAGE-Reg prototype for Gaussian, quantile, and bar predictors.
- The unlabeled signal is distributional self-agreement across perturbations.
- The public API is intentionally small.

## Claims We Should Not Make Yet

- broad state-of-the-art superiority
- robustness across domains
- theoretical guarantees
- benefits for multimodal density families beyond the current supported heads
- strong real-data claims without dedicated benchmarks

## Technical Risks

- shared-grid density approximation may blur sharp predictive laws
- pairwise agreement can be expensive for large `K`
- perturbation choice may dominate outcomes if not standardized
- EMA teacher may help or hurt depending on stochasticity source

## Paper Risks

- reviewers may say this is “just confidence weighting with extra steps”
- reviewers may say predictive variance already captures the needed signal
- reviewers may ask for clearer distinction from consistency regularization
- reviewers may compare directly against RaC/CURE-style methods

## Rebuttal Notes

- confidence and stability are different objects; broad but stable laws can still be useful
- SAGE-Reg separates trust estimation from target-law construction
- the method is not tied to one predictive family and does not require reconstruction branches

## Minimum Missing Evidence Before Submission

- one clean synthetic benchmark with strong calibration evidence
- at least one stress test where confidence is misleading and agreement helps
- one cross-backbone comparison
- one real-data sanity check
