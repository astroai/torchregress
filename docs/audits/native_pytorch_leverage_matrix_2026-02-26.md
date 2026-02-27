# Native PyTorch Leverage Matrix (2026-02-26)

This matrix records whether each overlap surface in `torchregress` should remain custom,
wrap native primitives, replace with native, or use a hybrid model.

## Decision Matrix

| Area | Surface | Native Candidate | Decision | Rationale | Action |
|---|---|---|---|---|---|
| standard_point_losses | `WeightedMSELoss`, `WeightedL1Loss`, `WeightedHuberLoss` | `torch.nn.MSELoss`, `torch.nn.L1Loss`, `torch.nn.HuberLoss` | `Hybrid` | Keep wrappers for masks/weights while relying on native math semantics. | Maintain wrapper APIs and parity checks. |
| gaussian_nll_diagonal | `WeightedGaussianNLLLoss`, `gaussian_nll` | `torch.nn.GaussianNLLLoss` | `Hybrid` | Native kernels are strong, wrappers add task ergonomics. | Keep wrappers + native agreement tests. |
| classification_weighted_wrappers | `WeightedCrossEntropyLoss`, `WeightedNLLLoss` | `torch.nn.CrossEntropyLoss`, `torch.nn.NLLLoss` | `Hybrid` | Supports regression-as-classification and consistent mask handling. | Keep wrappers with explicit scope docs. |
| point_metrics_baseline | `mean_squared_error`, `mean_absolute_error`, `rmse`, `r2_score` | `torchmetrics.functional` | `Wrap native` | Avoid metric reinvention and align behavior with community standards. | Prefer thin wrappers over bespoke implementations. |
| calibration_metrics_regression | `quantile_calibration_error`, `pit_uniformity_score` | torchmetrics calibration primitives | `Keep custom` | Regression calibration/PIT needs domain-specific semantics. | Keep custom definitions and tests. |
| ood_metrics | `ood_detection_metrics`, `typicality_score` | AUROC/AUPRC primitives | `Hybrid` | Composite OOD reporting is task-specific but can reuse native primitives. | Hybrid composition strategy. |
| ensemble_decomposition | uncertainty decomposition utilities | tensor reductions | `Keep custom` | Decomposition contracts and outputs are part of library UX value. | Retain API, use native ops internally. |
| conformal_prediction | conformal losses/utilities | none | `Keep custom` | No direct native equivalent for regression conformal pipelines. | Keep custom conformal stack. |
| mdn_and_flows | `MixtureDensityLoss`, `NormalizingFlowLoss` | `torch.distributions`, `zuko` | `Hybrid` | Keep task APIs; use native distribution math/backends. | Avoid duplicating primitive distribution code. |
| eiv_losses | EIV losses | none | `Keep custom` | EIV is specialized and absent from core native modules. | Keep custom implementations. |
| scaling_helpers | `compile_model`, AMP helpers | `torch.compile`, `torch.autocast`, `GradScaler` | `Wrap native` | Should remain lightweight compatibility wrappers. | Keep wrappers minimal and native-first. |

## Coverage Evidence Policy

Every row in `/Users/fabbros/src/torchregress/reports/native_pytorch_leverage_matrix_2026-02-26.json`
contains `coverage_evidence` with:

- `parity_tests`: concrete test IDs backing the decision
- `known_divergences`: intentional behavior differences from native APIs

This includes explicit coverage for classification wrappers (`WeightedCrossEntropyLoss`, `WeightedNLLLoss`).
