# Ranked SOTA SSL / probabilistic-regression roadmap for a general `torchregress` library

**Date:** 2026-06-13
**Goal:** design a general PyTorch library for probabilistic regression, semi-supervised regression, noisy labels, label/covariate shift, OOD routing, calibration, and test-time adaptation.

## 0. Executive summary

`torchregress` should not be a narrow reimplementation of image SSL papers. The valuable general library would be a **probabilistic semi-supervised regression toolkit** with modular components:

```text
DistributionHead
→ ProperScoringLoss
→ TeacherStudentSSLTrainer
→ PseudoLabelPolicy
→ ShiftEstimator
→ OODScorer
→ CalibrationLayer
→ TestTimeAdapter
→ EvaluationSuite
```

The highest-impact contribution would be making semi-supervised learning work for **continuous targets**, not just classification. Most SOTA SSL methods are classification-first. A general regression library should therefore implement the reusable primitives behind them: pseudo-label confidence, uncertainty weighting, distribution shift correction, conformal calibration, OOD routing, and scalable teacher-student distillation.

## 1. Ranking by library impact

| Rank | Component / method family | Impact | Why it belongs in `torchregress` |
|---:|---|---:|---|
| **1** | **Probabilistic regression heads + proper scoring** | Very high | Core of any calibrated regression library. |
| **2** | **Heteroscedastic pseudo-label semi-supervised regression** | Very high | The most direct SOTA SSL component for continuous targets. |
| **3** | **Conformal / SemiCP calibration** | Very high | Gives coverage and pseudo-label acceptance rules. |
| **4** | **Label-shift / target-prior correction over bins** | Very high | Essential whenever labeled and unlabeled targets differ. |
| **5** | **Teacher-student SSL engine** | Very high | General execution framework: EMA teacher, consistency, pseudo-labeling, distillation. |
| **6** | **Pseudo-label selection policies: FreeMatch, SoftMatch, SST, CSA, DiCaP, DIPS** | High | Modular confidence/quality rules reusable across tasks. |
| **7** | **OOD-safe SSL policies: PAF, SkipAlign, CaliMatch, USE** | High | Prevents unlabeled data from damaging training. |
| **8** | **Backbone/teacher adapters: TabPFN, TabICL, CatBoost, DINOv2, MAE** | High | Lets the library exploit best external models without owning them. |
| **9** | **Noisy-label modeling** | High | General regression labels are often measured, derived, simulated, or weak. |
| **10** | **Test-time adaptation: Tent, CoTTA, EATA/SAR-style filtering** | Medium-high | Useful for deployment shift; must be carefully constrained for regression. |
| **11** | **Generative tabular models: TabDDPM / TabDiff** | Medium | Useful for imputation/synthetic data, not core prediction. |
| **12** | **Image SSL method zoo** | Medium | Include wrappers/policies, not dozens of hard-coded paper replicas. |

## 2. Core abstractions

### 2.1 Distribution heads

Every model should output a distribution, not just a point. First-class heads:

| Head | Use |
|---|---|
| `GaussianHead` | Basic heteroscedastic regression. |
| `StudentTHead` | Heavy-tailed noisy labels. |
| `LaplaceHead` | Robust median-like regression. |
| `MixtureDensityHead` | Multimodal conditional targets. |
| `QuantileHead` | Quantile regression and conformal intervals. |
| `OrdinalCDFHead` | Ordered target bins; useful for redshift-like targets. |
| `SplineCDFHead` | Flexible continuous distribution with calibrated CDF. |
| `FlowHead` | High-capacity conditional density. |
| `DirichletCategoricalHead` | Discrete probabilistic classification/binning. |

The library should standardize:

```python
pred = model(x)
loss = pred.nll(y)
crps = pred.crps(y)
interval = pred.interval(level=0.9)
coverage = metrics.coverage(pred, y, level=0.9)
```

### 2.2 Proper scoring losses

Implement:

- NLL;
- CRPS;
- pinball / quantile loss;
- Brier score for discrete bins;
- energy score for multivariate targets;
- calibration losses for CDF/PIT;
- robust losses: Huber, Student-t NLL, trimmed NLL.

This is the base layer. Without proper scoring, SSL can improve point metrics while making probabilities worse.

## 3. Semi-supervised regression engine

### 3.1 Generic teacher-student trainer

A central `SSLRegressor` should support:

- supervised labeled loss;
- unlabeled consistency loss;
- pseudo-label density matching;
- EMA teacher;
- multi-teacher ensemble;
- weak/strong augmentations;
- pseudo-label acceptance/downweighting/routing;
- distributed shard training.

Example API:

```python
trainer = SSLRegressor(
    student=model,
    teacher="ema",
    distribution_head="mixture",
    supervised_loss="crps+nll",
    pseudo_label_policy="heteroscedastic_conformal",
    shift_estimator="bbse_em",
    ood_policy="perturbation_instability",
    calibration="split_conformal_cdf",
)
```

### 3.2 Heteroscedastic pseudo-labeling

This should be a flagship feature. The NeurIPS 2025 heteroscedastic pseudo-label work addresses continuous-output SSL by dynamically adjusting pseudo-label influence using uncertainty and bi-level optimization.

Implementation idea:

```text
L = L_labeled(y, p_s(y|x))
  + λ * w_u(x) * D[p_t(y|x_weak), p_s(y|x_strong)]
  + regularization on uncertainty
```

Where `w_u(x)` can depend on:

- teacher variance / entropy;
- ensemble disagreement;
- conformal set width;
- local-neighbor consistency;
- OOD score;
- sample difficulty history.

Provide default policies:

- `UncertaintyThresholdPolicy`;
- `SoftWeightPolicy`;
- `HeteroscedasticPolicy`;
- `ConformalWidthPolicy`;
- `EnsembleAgreementPolicy`;
- `PriorCorrectedPolicy`.

## 4. Calibration module

### 4.1 Calibration methods

Implement:

- temperature scaling for classification/binned targets;
- vector/matrix scaling for bins;
- isotonic calibration;
- CDF calibration;
- conformalized quantile regression;
- split conformal prediction;
- SemiCP-like semi-supervised conformal prediction;
- PIT recalibration;
- grouped/conditional calibration.

### 4.2 Metrics

Implement:

- NLL;
- CRPS;
- ECE / adaptive ECE;
- regression calibration error;
- PIT histogram diagnostics;
- marginal and conditional coverage;
- interval width;
- sharpness;
- subgroup calibration.

API sketch:

```python
calibrator = ConformalCalibrator(method="cdf", strata=["domain", "target_bin"])
calibrator.fit(pred_calib, y_calib, unlabeled_pred=pred_unlabeled)
pred_cal = calibrator(pred_test)
```

## 5. Shift correction module

### 5.1 Why it belongs in a regression library

Many regression problems have target distribution mismatch: labeled training data overrepresent easy/cheap samples; unlabeled/test data are broader. The general trick is to discretize or embed the continuous target and estimate target priors.

### 5.2 Methods to implement

| Method | Role |
|---|---|
| BBSE | Estimate target label/bin proportions from black-box predictions. |
| EM prior adjustment | Adjust predicted class/bin probabilities on target data. |
| LaSCal-style calibration | Estimate calibration error under label shift without target labels. |
| Doubly robust prior estimation | Estimate unlabeled class distribution before SSL. |
| Logit adjustment | Correct logits/probabilities using estimated priors. |
| Stratified prior estimation | Estimate priors by domain, batch, source, metadata cell. |

For regression:

```text
continuous y → adaptive bins → estimate p_target(bin) → correct PDF/bin logits → map back to continuous PDF/CDF
```

## 6. Pseudo-label policies

`torchregress` should implement paper-inspired policies rather than hard-coded image methods.

### 6.1 FreeMatch / SoftMatch / SST style

| Policy | Core idea | Regression adaptation |
|---|---|---|
| FreeMatch | self-adaptive confidence threshold | adaptive threshold on PDF entropy / interval width. |
| SoftMatch | soft confidence weighting | continuous weights from uncertainty instead of accept/reject. |
| SST | class-specific adaptive thresholds | bin-specific or region-specific thresholds. |

### 6.2 SimMatch / graph consistency style

Use approximate-nearest-neighbor graph consistency in embedding space. For regression:

- neighbors should have compatible predicted distributions;
- local CDF smoothness regularization;
- target-space graph consistency;
- avoid full all-pairs graph.

### 6.3 CSA / Sinkhorn allocation

Add a pseudo-label allocator:

```python
policy = SinkhornAllocationPolicy(target_prior="estimated", cost="nll_or_crps")
```

Useful when you have an estimated target distribution and want accepted pseudo-labels to match it.

### 6.4 DiCaP / correctness-aware weighting

Implement pseudo-label correctness models:

- empirical precision by confidence bin;
- historical reliability by sample region;
- dual-threshold policy: confident pseudo-labels get supervised loss, ambiguous samples get contrastive/consistency loss.

## 7. OOD-safe SSL module

### 7.1 PAF-style perturbation instability

Generalize PAF to regression:

```python
ood_score = distance(embedding(x), embedding(augment(x)))
pdf_instability = D[p(y|x), p(y|augment(x))]
```

Actions:

- accept pseudo-label;
- downweight;
- use only unsupervised representation loss;
- abstain;
- route to active learning.

### 7.2 SkipAlign-style non-alignment

Do not force uncertain/unseen samples into known clusters. For regression:

- do not force ambiguous samples into narrow target PDFs;
- use repulsion/diversity/representation loss without target pseudo-label loss;
- keep ambiguous samples useful for representation learning.

### 7.3 CaliMatch-style calibrated safe SSL

For classification or binned regression, calibrate both:

- the predictive distribution;
- the OOD/rejection detector.

This is important because overconfident OOD rejection can be as damaging as overconfident pseudo-labeling.

### 7.4 USE-style pre-filtering

Implement unlabeled-pool triage by entropy/uncertainty structure. Do not expose only a binary filter; expose routing policies.

```python
router = UnlabeledRouter(
    scores=["entropy", "ensemble_disagreement", "perturbation_instability"],
    actions=["pseudo_label", "unsupervised_only", "active_learning", "discard"]
)
```

## 8. Backbones and teacher adapters

### 8.1 Tabular teachers

Adapters should support:

- TabPFN / TabPFN-2.5;
- TabICL / TabICLv2;
- CatBoost;
- LightGBM;
- XGBoost;
- scikit-learn ensembles;
- AutoGluon if available.

These external models can be used for:

- pseudo-label generation;
- uncertainty estimates;
- teacher ensembles;
- distillation into PyTorch students.

### 8.2 Image encoders

Support:

- DINOv2;
- MAE/ViT;
- ConvNeXt;
- domain-specific encoders;
- frozen-feature extraction;
- fine-tuning and LoRA/adapters.

### 8.3 Multimodal adapters

For image+tabular tasks, implement:

- late fusion;
- cross-attention fusion;
- product-of-experts PDF fusion;
- mixture-of-experts fusion;
- modality dropout;
- missing-modality handling.

## 9. Noisy-label modeling

This should be first-class because many regression labels are measurements.

Implement:

- per-sample label variance;
- per-source label reliability weights;
- heavy-tailed Student-t likelihood;
- mixture clean/noisy likelihood;
- duplicated-label consistency diagnostics;
- robust pseudo-label filtering;
- label-noise-aware validation splits.

API sketch:

```python
loss = StudentTNLL(df="learned_or_fixed")
label_model = LabelNoiseModel(source_col="label_source", quality_col="quality_flag")
```

## 10. Test-time adaptation module

Test-time adaptation is useful for deployment shift but dangerous if unconstrained. The library should include it as an optional module with strong guardrails.

### 10.1 Methods

- **Tent:** entropy minimization and batch-norm/affine updates at test time.
- **CoTTA:** continual TTA with weight-averaged/augmentation-averaged predictions and stochastic restoration to reduce forgetting.
- **EATA/SAR-style filtering:** adapt only on reliable samples to reduce error accumulation.
- **Open-set TTA filtering:** reject samples whose confidence degrades after adaptation.

### 10.2 Regression adaptation

For regression, do not blindly minimize entropy because a sharp wrong PDF is bad. Instead adapt using:

- prediction consistency under augmentations;
- batch normalization/statistics update only;
- entropy minimization constrained by calibration/coverage;
- OOD-filtered adaptation batches;
- domain-specific nuisance adaptation layers.

API sketch:

```python
tta = TestTimeAdapter(
    method="consistency_bn",
    update_params="norm_and_adapters_only",
    sample_filter="low_ood_high_consistency",
    objective="cdf_consistency_not_entropy_only",
)
```

## 11. Generative tabular models

Include optional adapters for:

- TabDDPM;
- TabDiff;
- VAE/GAN tabular generators.

Use cases:

- imputation;
- synthetic minority samples;
- privacy-preserving data generation;
- density/OOD diagnostics;
- missing-column conditional generation.

Do not make them the core prediction path. They are usually too expensive and not consistently better for discriminative regression.

## 12. Benchmark suite

A general `torchregress` library should ship with shift/noise/SSL benchmarks.

### 12.1 Datasets/tasks

- low-label tabular regression;
- image regression;
- multimodal image+tabular regression;
- long-tailed target distribution;
- covariate shift;
- label shift;
- noisy labels;
- missing features;
- OOD target/domain.

### 12.2 Required baselines

- supervised MLP;
- supervised probabilistic MLP;
- CatBoost/LightGBM;
- TabPFN/TabICL adapters;
- EMA teacher SSL;
- heteroscedastic pseudo-label SSL;
- SSL + shift correction;
- SSL + conformal calibration;
- SSL + OOD routing.

### 12.3 Metrics

| Metric group | Metrics |
|---|---|
| Point | MAE, RMSE, bias, rank correlation |
| Distribution | NLL, CRPS, energy score |
| Calibration | PIT, coverage, interval width, ECE/adaptive ECE |
| Robustness | subgroup metrics, target-shift metrics, covariate-shift metrics |
| SSL behavior | pseudo-label acceptance rate, pseudo-label error, confidence-error curve |
| OOD | OOD AUROC/AUPRC, abstention utility, selective risk |

## 13. Minimal viable `torchregress` roadmap

### v0.1 — Core probabilistic regression

- Distribution heads: Gaussian, Student-t, quantile, mixture, ordinal CDF.
- Losses: NLL, CRPS, pinball, Brier, robust losses.
- Metrics: NLL, CRPS, PIT, coverage, interval width.
- Calibration: split conformal, quantile conformal, temperature scaling for bins.

### v0.2 — Semi-supervised regression

- EMA teacher/student trainer.
- Uncertainty-weighted pseudo-labeling.
- Heteroscedastic pseudo-label policy.
- Multi-teacher pseudo-labeling.
- Sharded unlabeled dataloaders.

### v0.3 — Shift and OOD

- BBSE/EM prior correction over bins.
- SemiCP-like calibration.
- PAF-style perturbation instability.
- CaliMatch-style calibrated OOD classifier wrapper.
- Open-set routing policies.

### v0.4 — External teachers and multimodal

- TabPFN/TabICL adapters.
- CatBoost/LightGBM adapters.
- DINOv2/MAE image encoders.
- Multimodal fusion modules.
- Distillation utilities.

### v0.5 — Test-time adaptation and active learning

- BN/statistics-only adaptation.
- Tent/CoTTA-inspired adapters with regression-safe losses.
- Active-learning hooks based on conformal width, OOD score, tail rarity, and disagreement.

## 14. Recommended design principle

The library should separate **what produces a pseudo-label** from **whether that pseudo-label should be trusted**.

A clean design:

```text
Teacher predicts distribution
→ Calibrator fixes distribution
→ ShiftEstimator adjusts target priors
→ OODScorer evaluates support mismatch
→ PseudoLabelPolicy computes weight/action
→ Student trains with weighted distributional loss
```

This compositional design makes `torchregress` more useful than a collection of paper-specific training loops.

## 15. References

1. Semi-Supervised Regression with Heteroscedastic Pseudo-Labels. https://arxiv.org/abs/2510.15266
2. Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score. https://arxiv.org/abs/2505.21147
3. Detecting and Correcting for Label Shift with Black Box Predictors. https://arxiv.org/abs/1802.03916
4. LaSCal: Label-Shift Calibration without target labels. https://openreview.net/forum?id=TALJtWX7w4
5. Improving realistic semi-supervised learning with doubly robust estimation. https://arxiv.org/abs/2502.00279
6. FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning. https://arxiv.org/abs/2205.07246
7. SoftMatch: Addressing the Quantity-Quality Trade-off in Semi-Supervised Learning. https://arxiv.org/abs/2301.10921
8. SimMatchV2: Semi-Supervised Learning with Graph Consistency. https://arxiv.org/abs/2308.06692
9. SST: Self-training with Self-adaptive Thresholding for Semi-supervised Learning. https://arxiv.org/abs/2506.00467
10. PAF: Perturbation-Aware Filtering for Open-Set Semi-Supervised Learning. https://openaccess.thecvf.com/content/CVPR2026/html/Han_PAF_Perturbation-Aware_Filtering_for_Open-Set_Semi-Supervised_Learning_CVPR_2026_paper.html
11. SkipAlign / Let the Void Be Void. https://ojs.aaai.org/index.php/AAAI/article/view/39194
12. CaliMatch. https://arxiv.org/abs/2508.00922
13. CalibrateMix. https://arxiv.org/abs/2511.12964
14. DiCaP. https://arxiv.org/abs/2511.20225
15. TabPFN-2.5. https://arxiv.org/abs/2511.08667
16. TabICLv2. https://arxiv.org/abs/2602.11139
17. DINOv2. https://arxiv.org/abs/2304.07193
18. TabDDPM. https://arxiv.org/abs/2209.15421
19. TabDiff. https://arxiv.org/abs/2410.20626
20. Tent: Fully Test-Time Adaptation by Entropy Minimization. https://arxiv.org/abs/2006.10726
21. CoTTA: Continual Test-Time Domain Adaptation. https://arxiv.org/abs/2203.13591
22. SafeECGMatch. https://arxiv.org/abs/2606.08037
