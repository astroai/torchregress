# SOTA semi-supervised learning methods with high potential for the torchz library (photometric-redshift inference)

**Date:** 2026-06-13
**Context:** billion-scale galaxy catalogues/images, tens-of-thousands of spectroscopic redshifts, strong label imbalance, label shift, covariate shift, noisy redshift labels, and a goal of beating SED fitting across diverse source populations. Primarily targeted at downstream tasks in the `torchz` library.

## 0. Executive takeaways

The most relevant SOTA SSL methods for redshifts are **not** the astronomy photo-z papers themselves. The right starting point is the modern ML SSL stack: large-scale representation learning, safe pseudo-labeling, label-shift correction, calibration, and probabilistic regression. Existing astronomy SSL papers are useful as baselines and domain inspiration, but they are generally behind current SSL practice.

For redshifts, the central mismatch is:

- **Unlabeled set:** ~10^9 photometric objects, heterogeneous in depth, morphology, blending, source type, field, and selection function.
- **Labeled set:** ~10^4 to 10^5 spectroscopic redshifts, strongly non-representative, often biased toward bright/low-z/target-selected objects, with quality flags and occasional catastrophic labels.
- **Target:** not a class label but a calibrated redshift PDF `p(z | image, catalog, metadata)`, often multimodal and tail-heavy.

The highest-impact path is therefore a **stack**, not a single paper:

```text
astro-native image/table representation pretraining
→ strong supervised teacher on spec-z labels
→ label-shift / redshift-prior estimation on the billion-object target pool
→ safe pseudo-labeling / uncertainty-weighted semi-supervised regression
→ OOD routing and abstention, not blind filtering
→ CRPS / PIT / conformal recalibration of the redshift PDF
→ evaluation against SED fitting on stratified redshift/source-type/depth splits
```

The best first benchmark suite should include:

1. **TabICLv2 / TabPFN-2.5 teacher + calibrated pseudo-labeling** for catalog-level redshifts.
2. **Heteroscedastic pseudo-label semi-supervised regression** for continuous/multimodal redshift targets.
3. **PAF / SkipAlign / CaliMatch safe SSL** for image SSL where unlabeled data contain OOD/rare sources.
4. **FreeMatch / SoftMatch / SimMatchV2 / SST** as the scalable pseudo-labeling baseline family.
5. **Label-shift correction**: BBSE/EM/LaSCal/doubly robust class-prior estimation over redshift bins or cells.
6. **SemiCP / conformal calibration** for coverage and pseudo-label acceptance.
7. **DINOv2/MAE-style representation learning, but trained or adapted on astronomical images**, not used naively from natural images.

## 1. Evaluation rubric for redshift relevance

I rank each method by expected impact on the actual redshift problem, not by generic image-classification leaderboard rank.

| Criterion | Why it matters for photo-z |
|---|---|
| **Scalability** | Billion-object photometric tables/images require streaming, sharding, approximate nearest neighbors, teacher distillation, and no all-pairs graph over the full catalogue. |
| **Label-shift robustness** | Spec-z training samples are not representative of the photometric target distribution. This is probably the dominant failure mode. |
| **OOD / open-set robustness** | Unlabeled objects include stars, blends, artifacts, AGN, rare high-z galaxies, low-surface-brightness objects, and sources absent from the labeled support. |
| **Probabilistic quality** | Redshift science needs calibrated PDFs: CRPS, PIT, NLL, coverage, tomographic `n(z)` bias, and catastrophic-tail behavior. |
| **Beating supervised / SED fitting** | The method must beat both supervised ML and template fitting across source populations, not only improve aggregate RMSE. |
| **Noisy-label tolerance** | Spec-z quality flags, blended spectra, grism labels, and proxy labels are not all equally reliable. |
| **Adaptability to continuous targets** | Most SSL methods are classification algorithms; redshift requires ordinal, continuous, or density modeling. |

## 2. Overall ranking by potential redshift impact

| Rank | Method family | Primary role | Redshift impact | Main reason |
|---:|---|---|---:|---|
| **1** | **TabICLv2 / TabPFN-2.5 teacher + pseudo-label filtering** | Catalog/table teacher | Very high | Strongest modern tabular foundation-model direction; can act as a high-quality teacher for photometric catalog features. |
| **2** | **Heteroscedastic pseudo-label semi-supervised regression** | Continuous redshift SSL | Very high | Directly addresses continuous pseudo-label uncertainty and noisy labels. |
| **3** | **Label-shift correction: BBSE / EM / LaSCal / doubly robust prior estimation** | Redshift-prior correction | Very high | Attacks the spec-z vs photometric-population mismatch directly. |
| **4** | **SemiCP / conformal pseudo-label filtering** | Coverage and pseudo-label safety | Very high | Gives calibrated prediction sets / acceptance rules under limited labels. |
| **5** | **PAF / SkipAlign / CaliMatch safe open-set SSL** | OOD-safe image SSL | High | Prevents unlabeled OOD/rare sources from poisoning pseudo-label training. |
| **6** | **FreeMatch / SoftMatch / SimMatchV2 / SST** | Scalable image SSL baseline | High | Strong, scalable pseudo-label/consistency family; must be adapted for redshift bins and shift. |
| **7** | **Astro-native MAE/DINOv2-style representation pretraining** | Image representation | High | Required for billion-scale images; natural-image DINOv2 is an inspiration, not a final model. |
| **8** | **CalibrateMix / DiCaP / correctness-aware pseudo-label weighting** | Calibration add-on | Medium-high | Useful for avoiding overconfident pseudo-labels; needs redshift-PDF adaptation. |
| **9** | **CSA / Sinkhorn pseudo-label allocation + DIPS-style selection** | Pseudo-label assignment | Medium-high | Useful when redshift-bin priors are estimated and pseudo-label budgets must respect target distribution. |
| **10** | **SimPro / LoFT / long-tailed SSL** | Tail handling | Medium-high | Relevant for rare redshift/source populations; classification-centric. |
| **11** | **USE unlabeled-pool triage** | Pre-filtering / routing | Medium | Useful, but high entropy can mean rare high-z, not garbage. Route rather than delete. |
| **12** | **S5 / scalable semi-supervised segmentation** | Dense image pretraining | Medium | Good analogy for survey-scale imaging and segmentation/deblending, less direct for object redshifts. |
| **13** | **SeBA** | Tabular few-shot representation | Medium | Interesting augmentation-free tabular SSL; not yet a billion-scale workhorse. |
| **14** | **TabDDPM / TabDiff** | Tabular generation/imputation | Medium-low | Useful for missing bands and density modeling; not first-line redshift predictor. |
| **15** | **Diffusion classifiers / DPT-style generative SSL** | Generative SSL | Low-medium | Theoretically attractive but too expensive for billion-object discriminative redshift inference. |
| **16** | **SafeECGMatch / ACE-Net / UCL-FP / AdaptiveSSL** | Domain-specific ideas | Low-medium | Useful ideas but not core redshift methods. |
| **Reject** | **DiffClass ICLR-2026 Spotlight, LabelProp-DINO CVPR-2026 Oral, STabMAE AISTATS-2026, EC-SSL, EviSSL** | Claimed methods | Reject unless sourced | I found no credible evidence for these specific claims as stated. |

## 3. Detailed ranked evaluation

### 1. TabICLv2 / TabPFN-2.5 teacher + calibrated pseudo-labeling

**Status:** real and highly competitive tabular foundation-model direction.
**Best redshift use:** catalog-based redshift teacher using magnitudes, colors, errors, morphology, WISE/NIR, flags, depths, PSF/seeing, extinction, field metadata, and optionally image embeddings.
**Potential impact:** very high.

TabPFN-2.5 reports scaling to datasets up to 50,000 samples and 2,000 features, with strong TabArena performance versus tuned tree baselines and a distillation engine for compact deployment. TabICLv2 claims stronger scaling, including million-scale generalization under 50GB GPU memory, and strong results on TabArena/TALENT.

**Why this matters for billion-scale redshifts:**

The model itself will not ingest a billion rows in-context. Instead, use it as a **high-quality teacher**:

1. Train/evaluate on spec-z labels and high-quality validation splits.
2. Predict calibrated redshift PDFs or redshift-bin probabilities for large unlabeled shards.
3. Select pseudo-labels with conformal/uncertainty criteria.
4. Distill into a scalable PyTorch/GBDT/MLP model for billion-object inference.
5. Re-estimate target redshift priors per survey region/depth/SOM cell.

**Strengths:**

- Excellent fit to catalog-level features.
- No need for hand-crafted tabular augmentations.
- Strong supervised baseline/teacher for low-label regimes.
- Good candidate to beat SED fitting on empirical patterns SED templates miss: blends, morphology, survey-specific color systematics, and calibration residuals.

**Weaknesses:**

- Not a pure SSL algorithm.
- Context/inference scaling is still far below 10^9 rows.
- Probabilities must be recalibrated; synthetic-prior Bayesian framing is not a guarantee of scientific calibration.
- Needs explicit label-shift correction because spec-z priors dominate otherwise.

**Redshift adaptation:**

- Convert `z` to ordinal/adaptive bins or train a density head around TabPFN/TabICL embeddings.
- Calibrate with CRPS, PIT, and conditional coverage by magnitude/color/redshift/source type.
- Use as one teacher in an ensemble with CatBoost/LightGBM/MLP/image models.
- Distill pseudo-labeled data into a billion-scale model.

**Key sources:** TabPFN-2.5 model report, TabICLv2.

---

### 2. Heteroscedastic pseudo-label semi-supervised regression

**Status:** real NeurIPS 2025 semi-supervised regression line.
**Best redshift use:** continuous redshift regression with uncertainty-weighted pseudo-labels.
**Potential impact:** very high.

Most SSL algorithms are classification methods. Redshift is better treated as continuous, ordinal, or density-valued. Heteroscedastic pseudo-label SSL is therefore central: it explicitly models that pseudo-label reliability varies by sample.

**Why this matters:**

A pseudo-label for a bright low-z elliptical is not equivalent to one for a faint high-z blue compact object or a blended AGN/galaxy. Naive pseudo-label regression will overfit easy populations and poison the tails. Heteroscedastic SSL lets the pseudo-label influence depend on predicted uncertainty and validation behavior.

**Strengths:**

- Directly aligned with continuous redshift.
- Can handle noisy labels and pseudo-label uncertainty.
- Natural fit to mixture-density, quantile, or histogram PDF heads.
- Better than forcing all redshifts into hard classes.

**Weaknesses:**

- A unimodal Gaussian head is not enough for photo-z aliasing.
- Needs a multimodal extension: MDN, spline CDF, discrete PDF, or normalizing flow.
- Must avoid uncertainty collapse, where the model inflates uncertainty to evade pseudo-label loss.

**Redshift adaptation:**

Use a teacher distribution `q_t(z | x)` and train a student with:

```text
L = L_spec_NLL_or_CRPS
  + w_u(x) * D(q_student(z|x_aug), stopgrad(q_teacher(z|x_weak)))
  + regularizers for PIT, smoothness, and prior consistency
```

Where `w_u(x)` is a function of teacher entropy, ensemble disagreement, conformal set width, OOD score, and redshift-bin rarity.

---

### 3. Label-shift correction: BBSE / EM / LaSCal / doubly robust class-prior estimation

**Status:** real, methodologically essential.
**Best redshift use:** estimate and correct the target redshift distribution of the unlabeled photometric population.
**Potential impact:** very high.

This is probably the most underrated component. Spec-z labels are heavily selected. If an SSL method uses pseudo-labels without correcting the target redshift/source distribution, it will reproduce spectroscopic selection bias.

**Core idea:**

Discretize redshift into bins/cells and estimate the target bin prior `p_target(z_bin)` using unlabeled data and model predictions. Then correct pseudo-label assignment, losses, or calibration using that estimated target prior.

**Relevant methods:**

- **BBSE:** estimates target label proportions from black-box predictions under label shift.
- **EM prior adjustment:** iteratively adjusts predicted class priors on target data.
- **LaSCal:** estimates calibration error under label shift without target labels.
- **Doubly robust realistic SSL:** estimates unlabeled class distribution before pseudo-label training and integrates it to reduce bias.

**Strengths:**

- Directly addresses label shift.
- Scales well: operates on binned predictions and confusion matrices, not raw images.
- Can be applied per field, depth, SOM cell, magnitude bin, source class, or HEALPix region.

**Weaknesses:**

- Pure label shift assumes `p(x|y)` stable across source and target; astronomy often also has covariate shift due to depth, photometric noise, blending, and selection.
- Confusion matrices must be invertible and estimated reliably.
- Rare high-z bins may be unstable with tens of thousands of spec-z labels.

**Redshift adaptation:**

- Use adaptive redshift bins with enough validation labels per bin.
- Estimate priors conditionally: `p(z_bin | SOM cell, magnitude, field, source type)`.
- Apply logit adjustment before pseudo-label selection.
- Monitor tomographic `n(z)` bias, not just point metrics.

---

### 4. SemiCP / conformal pseudo-label filtering and calibration

**Status:** real; CVPR 2026 version available.
**Best redshift use:** coverage control, pseudo-label acceptance, and calibrated redshift prediction intervals.
**Potential impact:** very high.

Conformal prediction is not a base learner; it is a calibration and decision layer. For redshifts, that is exactly what is needed: accept pseudo-labels only when prediction sets are reliable, and produce intervals/PDF sets with empirical coverage.

**Strengths:**

- Gives coverage discipline under limited calibration labels.
- Naturally maps to pseudo-label acceptance: accept if conformal set width is small, or if HPD mass is concentrated.
- Can be stratified by magnitude, color, field, redshift bin, source type, or SOM cell.

**Weaknesses:**

- Exchangeability assumptions are broken by spec-z selection bias unless stratified/reweighted.
- Marginal coverage can hide failures in high-z tails.
- Prediction sets can become too large in rare/OOD regimes.

**Redshift adaptation:**

- Use conformalized quantile regression or conformalized CDF/PDF bins.
- Use conformal width as pseudo-label weight or active-learning priority.
- Calibrate separately for target-population strata.

---

### 5. PAF / SkipAlign / CaliMatch safe open-set SSL

**Status:** real and highly relevant to image SSL under OOD contamination.
**Best redshift use:** safe image-based SSL and pseudo-label gating for unlabeled survey images.
**Potential impact:** high.

**PAF** dynamically identifies OOD samples by measuring representation instability under semantic-preserving perturbations. **SkipAlign** uses selective non-alignment: low-confidence samples are not pulled into ID clusters, avoiding collapsed ID/OOD boundaries. **CaliMatch** calibrates both the classifier and OOD detector for safe SSL under label-distribution mismatch.

**Why this matters for redshifts:**

A billion-object photometric catalogue will contain sources outside the training support: stars, artifacts, strong blends, AGN, rare high-z galaxies, odd SEDs, and data-quality failures. Standard pseudo-labeling will confidently force these into known redshift/source bins. Safe SSL methods reduce that failure.

**Strengths:**

- Directly targets the open-set unlabeled-pool problem.
- Computationally feasible compared with generative diffusion classifiers.
- Compatible with image encoders and redshift-bin heads.
- CaliMatch adds calibration of both classifier and OOD detector.

**Weaknesses:**

- Mostly classification methods; redshift needs ordinal/continuous adaptation.
- “OOD” must not mean “discard rare science.” High-z rare objects may look OOD.
- Requires perturbations that preserve astrophysical semantics.

**Redshift adaptation:**

Use perturbations such as:

- noise injection based on inverse variance;
- PSF and seeing perturbations;
- band dropout;
- background perturbation;
- small WCS jitter;
- mask/deblend perturbation;
- aperture/scale perturbation.

Actions should be **route/downweight/query**, not simply delete.

---

### 6. FreeMatch / SoftMatch / SimMatchV2 / SST

**Status:** real, strong general image SSL baseline family.
**Best redshift use:** scalable pseudo-label/consistency baseline for redshift-bin classification.
**Potential impact:** high.

These methods are still important because they are the best understood scalable baseline family:

- **FreeMatch:** self-adaptive confidence thresholding.
- **SoftMatch:** confidence-based soft weighting to address the pseudo-label quantity/quality tradeoff.
- **SimMatchV2:** graph consistency between labeled/unlabeled augmentations; strong ImageNet 1%/10% label results.
- **SST:** self-adaptive thresholding with strong ImageNet results using large ViTs.

**Strengths:**

- Scalable and practical.
- Good baselines for image classification SSL.
- Can beat fully supervised baselines in low-label regimes when unlabeled data are compatible.

**Weaknesses for redshift:**

- Assumes classification; redshift needs ordinal/density heads.
- Sensitive to label shift and unlabeled distribution mismatch.
- Usually optimized for accuracy, not CRPS/PIT/coverage.
- Without safe/OOD filtering, can amplify spec-z bias.

**Redshift adaptation:**

- Use redshift bins with ordinal/CRPS loss rather than cross-entropy alone.
- Apply label-shift correction to pseudo-label priors.
- Combine with PAF/CaliMatch and CalibrateMix.
- Use pseudo-labels as soft PDFs, not hard bins.

---

### 7. Astro-native MAE/DINOv2-style representation learning

**Status:** DINOv2 and MAE-style foundation-model pretraining are real and central, but natural-image weights are not enough.
**Best redshift use:** pretraining image encoders on unlabeled survey cutouts before SSL.
**Potential impact:** high.

DINOv2 showed that self-supervised vision models trained at scale on curated data can produce robust features. For astronomy, the analogue should be trained on astronomical cutouts/tiles with multi-band structure, weights, masks, PSFs, WCS, and metadata.

**Strengths:**

- Uses the billion-object unlabeled pool naturally.
- Reduces dependence on tens of thousands of spec-z labels.
- Can feed both image heads and catalog/tabular models via embeddings.

**Weaknesses:**

- Natural-image DINOv2 does not encode SED/PSF/noise physics by default.
- Self-supervised representation alone is not semi-supervised redshift learning.
- Requires careful pretext tasks: not all augmentations preserve redshift information.

**Redshift adaptation:**

- Masked band reconstruction: predict missing filters from observed filters.
- Cross-survey alignment: HSC/Euclid/LSST/UNIONS bands and seeing/depth metadata.
- Object-centric and context cutouts.
- Contrastive positives from same source in different surveys/epochs/augmentations.
- Negative sampling by sky proximity and color similarity, not random only.

---

### 8. CalibrateMix / DiCaP / correctness-aware pseudo-label weighting

**Status:** real; mostly classification-focused.
**Best redshift use:** calibration and pseudo-label quality weighting.
**Potential impact:** medium-high.

CalibrateMix targets poor calibration in image SSL using guided mixup between easy/hard labeled and unlabeled samples. DiCaP is a distribution-calibrated pseudo-labeling method for semi-supervised multi-label learning that estimates pseudo-label correctness likelihood and uses dual-thresholding.

**Strengths:**

- Directly targets overconfidence.
- Easy to combine with existing SSL methods.
- Correctness-aware pseudo-label weighting is highly relevant to noisy redshift pseudo-labels.

**Weaknesses:**

- Classification/multi-label framing, not continuous redshift PDF.
- ECE improvements do not automatically imply calibrated redshift PDFs.
- Mixup in image/catalog feature space may violate astrophysical semantics if naive.

**Redshift adaptation:**

- Mix or interpolate in latent representation, not raw flux unless physically justified.
- Use redshift-PDF correctness likelihood: agreement with ensemble, conformal width, local neighbor consistency, source-type prior.
- Evaluate with CRPS/PIT, not only ECE.

---

### 9. CSA / Sinkhorn allocation + DIPS-style pseudo-label selection

**Status:** real family of pseudo-label selection/allocation ideas.
**Best redshift use:** constrain pseudo-labels to plausible target redshift priors.
**Potential impact:** medium-high.

Naive thresholding accepts easy low-z samples and ignores rare high-z samples. Allocation methods can impose a target distribution over accepted pseudo-labels.

**Strengths:**

- Good fit to estimated redshift-bin priors.
- Prevents pseudo-label collapse to common classes/bins.
- Useful with label-shift correction.

**Weaknesses:**

- If the estimated prior is wrong, allocation can suppress real rare populations.
- Requires binning, which can be awkward for continuous/multimodal redshift.

**Redshift adaptation:**

- Use adaptive redshift bins plus source-type/magnitude cells.
- Allocate accepted pseudo-label mass according to estimated `p_target(z_bin | cell)`.
- Combine with soft PDFs rather than hard bin assignments.

---

### 10. SimPro / LoFT / long-tailed SSL

**Status:** real and relevant for class imbalance.
**Best redshift use:** rare redshift/source populations and unlabeled-prior mismatch.
**Potential impact:** medium-high.

Long-tailed SSL methods estimate or correct class distributions when labeled and unlabeled distributions differ. That maps naturally to redshift bins, source classes, and high-z tails.

**Strengths:**

- Tail-aware pseudo-labeling is important for high-z galaxies, quasars, dusty galaxies, LSB sources.
- Compatible with prior-estimation and pseudo-label allocation.

**Weaknesses:**

- Classification-centric.
- Rare high-z bins may be underrepresented even in unlabeled data at usable S/N.
- Needs scientific priors to avoid hallucinating tail populations.

---

### 11. USE: Uncertainty Structure Estimation

**Status:** real 2026 preprint line.
**Best redshift use:** unlabeled-pool triage before SSL.
**Potential impact:** medium.

USE computes uncertainty/entropy structure on unlabeled data and filters harmful samples before SSL.

**Strengths:**

- Simple pre-processing idea.
- Scales if applied with a cheap proxy model.
- Useful for identifying obvious artifacts/OOD regions.

**Weaknesses:**

- High uncertainty can identify the most scientifically valuable sources, not just harmful noise.
- Should route samples to active learning or special handling, not discard wholesale.

---

### 12. S5 / scalable semi-supervised semantic segmentation

**Status:** real AAAI 2026 remote-sensing method.
**Best redshift use:** survey-image foundation pretraining, segmentation/deblending/context extraction.
**Potential impact:** medium.

S5 curates a large remote-sensing pretraining set with entropy filtering and diversity expansion, then scales semi-supervised segmentation and MoE fine-tuning. The analogy to astronomy is strong: large images, expensive labels, dense scenes, domain-specific foundation models.

**Strengths:**

- Good model for scaling semi-supervised dense prediction.
- Relevant to deblending, segmentation, morphology, artifact masks.

**Weaknesses:**

- Not an object-level redshift estimator.
- Remote-sensing semantics differ from astronomical flux/color/SED inference.

---

### 13. SeBA

**Status:** real 2026 tabular few-shot SSL method.
**Best redshift use:** augmentation-free tabular representation learning in low-label settings.
**Potential impact:** medium.

SeBA avoids tabular augmentations by splitting features into complementary views and aligning nearest-neighbor relationships.

**Strengths:**

- Addresses a real weakness of tabular SSL: bad augmentations.
- Could be useful for feature representation before redshift models.

**Weaknesses:**

- Few-shot tabular SSL is not the same as billion-row catalog learning.
- Evidence base is much smaller than TabPFN/TabICL or tree ensembles.

---

### 14. TabDDPM / TabDiff

**Status:** real tabular generative modeling methods.
**Best redshift use:** missing-feature imputation, simulation, density/OOD diagnostics.
**Potential impact:** medium-low for primary redshift inference.

**Strengths:**

- Can model heterogeneous tabular features.
- Useful for missing bands, synthetic data, privacy, and density checks.

**Weaknesses:**

- Generative likelihood is not automatically a reliable OOD score.
- Diffusion inference/sampling is expensive.
- Not SOTA for discriminative photo-z prediction.

---

### 15. Diffusion classifiers / DPT-style generative SSL

**Status:** real broader method family, but not practical first-pass for billion-scale redshift.
**Best redshift use:** specialized experiments on density/likelihood/OOD.
**Potential impact:** low-medium.

**Strengths:**

- Principled generative view.
- Could model conditional data likelihoods and missing bands.

**Weaknesses:**

- Multi-pass inference cost is a poor fit for 10^9 objects.
- Density likelihoods can be misleading for OOD.
- Discriminative models with calibration are likely more effective.

---

### 16. SafeECGMatch / ACE-Net / UCL-FP / AdaptiveSSL

**Status:** real or plausible, but domain-specific or lower-priority.
**Best redshift use:** borrow ideas only.
**Potential impact:** low-medium.

- **SafeECGMatch:** useful calibration-aware safe SSL design, but ECG-specific.
- **UCL-FP:** complementary labels and feature perturbation; not a major SSL venue result.
- **AdaptiveSSL:** interesting combination of Wasserstein uncertainty calibration, diversity sampling, and dynamic pseudo-labeling; evidence base weaker than top-tier SSL methods.
- **ACE-Net:** semi-supervised medical image captioning/evidential uncertainty; not directly relevant.

---

## 4. Methods/claims to exclude unless independently sourced

These were claimed in the supplied reports but I would not treat them as verified SOTA methods:

| Claimed method | Claimed venue/status | Evaluation |
|---|---|---|
| **DiffClass – Diffusion Generative Classifier for SSL** | ICLR 2026 Spotlight | I did not find credible evidence for this method as stated. There are diffusion classifiers and diffusion SSL papers, but this exact claim appears unreliable. |
| **LabelProp-DINO** | CVPR 2026 Oral | Frozen-DINO label propagation is plausible, but I found no evidence for this exact CVPR 2026 Oral method. |
| **STabMAE** | AISTATS 2026 | I found no credible match for the claimed tabular method. |
| **EC-SSL** | NeurIPS 2025 / arXiv Jan 2026 | I found no credible match as stated. |
| **EviSSL** | ICLR 2026 | Evidential SSL exists in niche domains, but not this claimed consensus method. |

## 5. Redshift-specific architecture proposal

### 5.1 Data representation

Use three branches:

1. **Catalog branch**: fluxes/magnitudes, colors, errors, missingness, morphology, WISE/NIR, flags, extinction, local depth, PSF, seeing, sky region.
2. **Image branch**: multi-band cutouts, weight/mask maps, PSF summaries, WCS-aware coordinates, context cutouts.
3. **Metadata branch**: survey, field, depth, Galactic extinction, calibration metadata, observation epoch if relevant.

The image branch should be pretrained on the full unlabeled survey sample with an astronomy-native SSL objective. The catalog branch should use TabPFN/TabICL/CatBoost/GBDT teachers and distillation.

### 5.2 Redshift target representation

Avoid a single scalar regression head. Use at least one of:

- adaptive redshift-bin PDF with CRPS loss;
- ordinal cumulative-probability head;
- mixture density network;
- quantile/CDF head;
- normalizing-flow density head conditioned on features;
- hybrid bin + local continuous residual model.

For billion-scale inference, a discrete/adaptive-bin PDF or spline CDF is likely the best first choice.

### 5.3 Semi-supervised training loop

```text
1. Train supervised teacher ensemble on spec-z labels.
2. Calibrate teacher with CRPS/PIT/conformal methods on held-out spec-z.
3. Predict soft PDFs for unlabeled shards.
4. Estimate target redshift priors by bins/cells from unlabeled predictions using BBSE/EM/DR correction.
5. Compute pseudo-label reliability:
   - entropy / PDF width
   - ensemble disagreement
   - conformal set width
   - representation instability under perturbations
   - OOD detector score
   - local-neighbor consistency
6. Accept, downweight, or route pseudo-labels.
7. Train student on labeled + weighted pseudo-labeled data.
8. Distill into scalable inference model.
9. Recalibrate on validation and external fields.
10. Iterate with active-learning requests for high-value failures.
```

### 5.4 Do not discard rare/OOD objects blindly

For redshifts, “OOD” has two meanings:

- **bad OOD:** artifacts, bad deblends, stars when training galaxies only, corrupted photometry;
- **science OOD:** rare high-z galaxies, AGN, dusty galaxies, strong lenses, LSBs, odd SEDs.

Safe SSL should route objects into:

- accept pseudo-label;
- downweight pseudo-label;
- abstain but keep for representation learning;
- send to SED fitting/template prior;
- request spectra/active learning;
- train separate specialist model.

## 6. Billion-scale engineering constraints

### What scales

- Streaming pseudo-labeling by shards.
- EMA teacher/student models.
- TabPFN/TabICL used on sampled contexts + distillation.
- Approximate nearest-neighbor search on compressed embeddings.
- Label-shift correction over bins/cells.
- Conformal calibration per stratum if calibration sets are not too fragmented.
- DINO/MAE-style pretraining on tiled/cutout WebDataset shards.

### What does not scale directly

- Full graph label propagation over 10^9 objects.
- Diffusion classifier inference per source.
- TabPFN-style full-context inference over all rows.
- Dense all-pairs contrastive objectives.
- Hand-inspected pseudo-label selection.

### Recommended scalable design

- Precompute embeddings for all objects.
- Store embeddings/catalog features in parquet/Zarr plus ANN index.
- Use sample-based teacher inference and distillation.
- Run pseudo-label selection in distributed shards.
- Maintain per-stratum calibration/shift summaries, not all raw predictions in memory.
- Keep a small, high-quality validation and calibration set separate from pseudo-label training.

## 7. Evaluation protocol against SED fitting

A method only “beats SED fitting” if it wins beyond aggregate RMSE.

### Required baselines

- EAZY / LePhare / BPZ or whichever SED-fitting baseline is standard for the survey.
- CatBoost/LightGBM supervised catalog model.
- Supervised image CNN/ViT.
- Supervised hybrid image+catalog model.
- FreeMatch/SoftMatch redshift-bin SSL baseline.
- PAF/CaliMatch safe SSL variant.
- TabICL/TabPFN teacher-distilled model.

### Required metrics

| Metric group | Metrics |
|---|---|
| Point redshift | bias, NMAD, MAE/RMSE, catastrophic outlier fraction |
| PDF quality | CRPS, NLL, PIT histogram, HPD coverage, interval width |
| Tomography | mean `z` bias per bin, `n(z)` error, high-z tail recovery |
| Robustness | field transfer, depth transfer, seeing/PSF transfer, survey transfer |
| Tail performance | high-z, quasars/AGN, dusty galaxies, compact sources, blends, LSBs |
| Label-noise resilience | performance by spec-z quality flag, duplicate disagreement, proxy-label source |
| Calibration | conditional coverage by magnitude/color/source type/redshift bin |

### Stress tests

- Train on bright spectroscopic labels, test on faint photometric target.
- Train on one field, test on another field with different depth/seeing.
- Remove high-z labels and test whether SSL recovers or hallucinates high-z mass.
- Add controlled label noise to spec-z and measure degradation.
- Evaluate on rare populations: AGN, stars, compact galaxies, strong blends.

## 8. Recommended first experiment matrix

| Experiment | Model stack | Purpose |
|---|---|---|
| **Catalog teacher** | TabICLv2/TabPFN/CatBoost ensemble + CRPS/PDF head | Strong tabular baseline and pseudo-label teacher. |
| **Shift-corrected pseudo-labeling** | Teacher + BBSE/EM/DR prior correction + CSA allocation | Test whether target redshift priors improve high-z/tail behavior. |
| **Heteroscedastic SSR** | Teacher/student with uncertainty-weighted continuous pseudo-labels | Test continuous redshift SSL without crude hard bins. |
| **Image SSL baseline** | astro-MAE/DINO encoder + SoftMatch/FreeMatch redshift bins | Scalable image SSL baseline. |
| **Safe image SSL** | same encoder + PAF/SkipAlign/CaliMatch gating | Test OOD/rare-source robustness. |
| **Hybrid distillation** | image embeddings + catalog features + PDF student | Best practical billion-scale candidate. |
| **Calibration layer** | SemiCP / conformalized CDF / PIT recalibration | Test scientific usability of PDFs. |

## 9. Final recommendation

For redshifts at billion-object scale, prioritize:

1. **Catalog foundation-model teacher + scalable distillation** rather than direct TabPFN/TabICL inference over all objects.
2. **Heteroscedastic semi-supervised regression** rather than hard pseudo-label classification alone.
3. **Label-shift correction over redshift/source/magnitude cells** as a first-class component.
4. **Safe SSL/OOD routing** for images and embeddings: PAF, SkipAlign, CaliMatch-style ideas.
5. **Redshift-PDF calibration** with CRPS/PIT/conformal methods before claiming success.

The core experiment should ask:

> Can a shift-corrected, safe, semi-supervised image+catalog model improve CRPS, PIT, tomographic `n(z)` bias, and catastrophic outlier rate over SED fitting across redshift/source-type/depth strata?

If the answer is yes, then the method is scientifically meaningful. If it only improves aggregate RMSE, it is not enough.

## References

1. PAF: Perturbation-Aware Filtering for Open-Set Semi-Supervised Learning, CVPR 2026. https://openaccess.thecvf.com/content/CVPR2026/html/Han_PAF_Perturbation-Aware_Filtering_for_Open-Set_Semi-Supervised_Learning_CVPR_2026_paper.html
2. SkipAlign / Let the Void Be Void: Robust Open-Set Semi-Supervised Learning via Selective Non-Alignment, AAAI 2026. https://ojs.aaai.org/index.php/AAAI/article/view/39194
3. CaliMatch: Adaptive Calibration for Improving Safe Semi-supervised Learning, ICCV 2025 / arXiv 2025. https://arxiv.org/abs/2508.00922
4. CalibrateMix: Guided-Mixup Calibration of Image Semi-Supervised Models, arXiv 2025. https://arxiv.org/abs/2511.12964
5. FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning. https://arxiv.org/abs/2205.07246
6. SoftMatch: Addressing the Quantity-Quality Trade-off in Semi-Supervised Learning. https://arxiv.org/abs/2301.10921
7. SimMatchV2: Semi-Supervised Learning with Graph Consistency. https://arxiv.org/abs/2308.06692
8. SST: Self-training with Self-adaptive Thresholding for Semi-supervised Learning. https://arxiv.org/abs/2506.00467
9. TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models. https://arxiv.org/abs/2511.08667
10. TabICLv2: A better, faster, scalable, and open tabular foundation model. https://arxiv.org/abs/2602.11139
11. SeBA: Semi-supervised few-shot learning via Separated-at-Birth Alignment. https://arxiv.org/abs/2605.08519
12. Semi-Supervised Regression with Heteroscedastic Pseudo-Labels. https://arxiv.org/abs/2510.15266
13. Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score. https://arxiv.org/abs/2505.21147
14. Detecting and Correcting for Label Shift with Black Box Predictors. https://arxiv.org/abs/1802.03916
15. LaSCal: Label-Shift Calibration without target labels. https://openreview.net/forum?id=TALJtWX7w4
16. Improving realistic semi-supervised learning with doubly robust estimation. https://arxiv.org/abs/2502.00279
17. DINOv2: Learning Robust Visual Features without Supervision. https://arxiv.org/abs/2304.07193
18. S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing. https://ojs.aaai.org/index.php/AAAI/article/view/37715
19. TabDDPM: Modelling Tabular Data with Diffusion Models. https://arxiv.org/abs/2209.15421
20. TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation. https://arxiv.org/abs/2410.20626
21. DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning. https://arxiv.org/abs/2511.20225
22. USE: Uncertainty Structure Estimation for SSL. https://arxiv.org/abs/2603.00404
23. SafeECGMatch: Calibration-Aware Joint Frequency and Time Space Semi-Supervised Learning for Open-Set ECG Classification. https://arxiv.org/abs/2606.08037
24. Tent: Fully Test-Time Adaptation by Entropy Minimization. https://arxiv.org/abs/2006.10726
25. CoTTA: Continual Test-Time Domain Adaptation. https://arxiv.org/abs/2203.13591
