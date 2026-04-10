# SAGE-Reg Related Work Notes

## Positioning Sentence

Prior semi-supervised regression work often centers scalar confidence, regression-as-classification
targets, or reconstruction-enhanced iterative self-training. SAGE-Reg is instead defined directly on
predictive distributions and uses cross-perturbation self-agreement as the trust signal.

## Separation From CURE-Style Framing

State this early:

- SAGE-Reg is **not** RaC-centric.
- SAGE-Reg uses **no** reconstruction or autoencoder branch.
- SAGE-Reg does **not** depend on iterative curriculum relabeling or full retraining cycles.
- SAGE-Reg is backbone-agnostic across predictive heads that can be mapped into a common law.

## Closest Adjacent Buckets

### Pseudo-labeling for regression

- Usually compresses an unlabeled prediction into a scalar pseudo target.
- Often weights that target by confidence, variance, or heuristic uncertainty.
- Main contrast: SAGE-Reg trusts samples based on distributional stability, not scalar confidence.

### Consistency regularization / teacher-student SSL

- Uses perturbations and agreement between model outputs.
- Main contrast: SAGE-Reg makes the object of agreement the **predictive law**, not just a point output.

### Uncertainty-aware regression

- Predictive variance can be broad but still reliable.
- Main contrast: SAGE-Reg does not equate low variance with high trust.

## Draft Related-Work Structure

1. Semi-supervised learning and pseudo-labeling.
2. Regression-specific SSL and uncertainty-aware pseudo-supervision.
3. Consistency methods and teacher-student learning.
4. Predictive-distribution methods for regression.
5. Explicit paragraph separating SAGE-Reg from CURE-style approaches.

## Reviewer-Facing Comparison Table Shell

| Axis | Confidence pseudo-labeling | CURE-style methods | SAGE-Reg |
|:-----|:---------------------------|:-------------------|:---------|
| Trusted object | scalar prediction | PMF / RaC target | predictive law |
| Trust signal | confidence / variance | confidence + auxiliary machinery | self-agreement across perturbations |
| Reconstruction branch | optional / no | central in some variants | no |
| Iterative curriculum | common | common | not required |
| Backbone dependence | often head-specific | often RaC-specific | backbone-agnostic over predictive families |
