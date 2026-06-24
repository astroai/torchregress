`torchregress` is already a strong regression-UQ toolbox, but it is not yet ready to make a credible “SOTA ensembling” claim.** Its method breadth is good; the main gaps are statistical correctness, faithful implementation/naming of efficient ensembles, shift-targeted diversity methods, scalable execution, and public benchmark evidence.

My assessment of the code I could inspect:

| Area                                     |                                     Readiness |
| ---------------------------------------- | --------------------------------------------: |
| Independent probabilistic deep ensembles |                                        Strong |
| Non-Gaussian regression ensembles        |                                        Strong |
| General uncertainty API                  |                                          Good |
| Efficient ensembles                      |                                       Partial |
| Distribution-shift robustness            |                            Early/experimental |
| Statistical/API correctness              |                                   Needs fixes |
| SOTA benchmark evidence                  | Not verifiable from the accessible repository |

The repository already includes mean-only and heteroscedastic deep ensembles, distributional ensembles for bins, ordinal outputs and MDNs, BatchEnsemble, MC dropout, BNNs, SWAG/MultiSWAG, and learned combiners.

The README says the larger reproduction benchmarks live in private sibling repositories. Those were not exposed through the current GitHub connection, so I could not assess their results or reproducibility.

## Fix these before adding more methods

### 1. Model-instance ensembles may start identically

When the user passes a model **instance**, `BaseEnsembleModel` deep-copies it without resetting its parameters. The tests verify only that the parameters are different Python objects, not that their values differ. With the same minibatches and deterministic training, members can remain identical or nearly identical.

This is the most important issue because it can silently eliminate epistemic uncertainty.

Add an explicit member-construction contract:

```python
DeepEnsemble(
    member_factory=lambda member_id, seed: MyModel(...),
    ensemble_size=5,
    base_seed=1234,
    initialization="independent",
)
```

Also support:

* `member_seed_fn`
* `reset_parameters=True`
* independent `DataLoader` generators
* bootstrap/subsample data strategies
* per-member augmentation
* a warning when initial member predictions or parameters are identical

### 2. Predictive variance uses the wrong denominator for a mixture

The exact variance of the finite ensemble predictive mixture uses (1/M). The implementation uses `torch.var(..., unbiased=True)`, corresponding to (1/(M-1)). The heteroscedastic helper does the same whenever (M>1).

For an ensemble of five members, that inflates the disagreement component by 25%. That may be appropriate when estimating the variance of an assumed population of models, but it is not the variance of the predictive mixture actually being returned.

Use:

```python
torch.var(member_means, dim=member_dim, correction=0)
```

Expose `correction` only for users who explicitly want sample-statistical semantics.

### 3. Generic full covariance has an incorrect contraction

`BaseEnsembleModel.predict_full_covariance` documents a `[B, D, D]` output but currently contracts to `[B, M, M]`:

```python
torch.einsum("bmd,bnd->bmn", ...)
```

The heteroscedastic implementation has the correct output-space contraction:

```python
torch.einsum("bmd,bme->bde", ...)
```

Add exact-value tests using hand-constructed multivariate member predictions. The current visible ensemble tests do not exercise the generic full-covariance path.

### 4. `PackedEnsembleRegressor` is BatchEnsemble, not Packed-Ensembles

The implementation explicitly wraps `HeteroscedasticBatchEnsembleModel` and cites the BatchEnsemble paper. It scales the BatchEnsemble rank-one factors; it does not implement the grouped-width architecture of Packed-Ensembles.

That naming will be challenged in a paper or library comparison. True Packed-Ensembles partition model width using grouped linear, convolution, normalization and attention operations. The established implementation also distinguishes its capacity parameters such as `alpha` and `gamma`. ([OpenReview][1])

Choose one:

* Rename it `BatchEnsembleRegressor`.
* Implement genuine `PackedLinear`, `PackedConv*`, `PackedLayerNorm`, and eventually packed attention.

This matters because recent evidence suggests rank-one BatchEnsemble members can collapse toward functionally near-identical predictors, behaving more like one model on calibration and OOD detection. That is currently a 2026 preprint rather than a final consensus result, but it makes member-diversity diagnostics essential. ([arXiv][2])

### 5. `MultiSWAG` discards within-mode posterior variance

`predict_with_uncertainty` samples each SWAG posterior, averages those samples inside each SWAG, and then computes variance only across the resulting SWAG means. The within-SWAG variability is discarded, while the returned aleatoric variance is always zero.

The appropriate decomposition is:

[
\operatorname{Var}(Y)
=====================

\underbrace{\operatorname{Var}*{k}!\left(
\mathbb E*{w\mid k}[\mu_w]
\right)}*{\text{between-mode epistemic}}
+
\underbrace{\mathbb E_k!\left[
\operatorname{Var}*{w\mid k}(\mu_w)
\right]}*{\text{within-mode epistemic}}
+
\underbrace{\mathbb E*{k,w}[\sigma_w^2]}_{\text{aleatoric}}.
]

Within-SWAG weight sampling is still epistemic. It becomes aleatoric only when the sampled network explicitly predicts an observation distribution and you average its conditional variances.

### 6. `BayesianModelAveraging` is not Bayesian model averaging

The class learns unconstrained logits over point predictions and applies a softmax. There is no marginal likelihood, posterior model probability, prior over models, or posterior integration.

Rename it to something like:

* `LearnedConvexEnsemble`
* `ValidationWeightedEnsemble`
* `SoftmaxModelCombiner`

True BMA could remain a separate optional interface.

### 7. The OOD documentation is too confident

The documentation says high ensemble disagreement is a strong OOD signal and that ensembles “excel” at OOD detection.

Real-world regression-shift evaluations show that common UQ methods, including ensembles, can become substantially overconfident under shift. ([arXiv][3])

The current visible OOD tutorial is also mostly an interpolation/extrapolation toy: it defines an OOD sample count but evaluates a single interval from ([-3.5,3.5]) after training on ([-3,3]), rather than reporting separate ID and OOD populations.

Change the documentation language to:

> Disagreement is a useful candidate score, but it must be validated for the anticipated shifts. Deep ensembles can remain confidently wrong when members extrapolate similarly.

### 8. Correct the VIDS citation

The VIDS documentation cites arXiv `2506.03942`, which is a different paper. “Quantifying Uncertainty in the Presence of Distribution Shifts” is arXiv `2506.18283`.  ([DBLP][4])

---

# What you should add

Do **not** add another ten loosely integrated ensemble classes. Add three or four methods that address distinct weaknesses and can share a common ensemble infrastructure.

## Priority 1: shift-robust diversity

### A. DARE-style ensembles

This is the most directly relevant addition for regression under support shift. DARE was designed around the failure of ordinary deep ensembles to express sufficient uncertainty outside the training domain. It increases off-domain functional diversity while controlling in-domain degradation and does not require an OOD training dataset. ([arXiv][5])

Suggested API:

```python
model = DAREEnsemble(
    member_factory=...,
    ensemble_size=5,
    id_loss_tolerance=0.02,
    anti_regularization_schedule="adaptive",
)
```

Treat it as experimental until reproduced on several shift families.

### B. Function-space repulsive ensembles

Implement a generic diversity regularizer rather than hard-coding only DARE:

```python
DeepEnsemble(
    ...,
    diversity_regularizer=FunctionSpaceRepulsion(
        probe_sampler=...,
        kernel="rbf",
        weight=...
    ),
)
```

Repulsive deep ensembles explicitly prevent member functions from collapsing. A particularly attractive variant is a repulsive **last-layer or multi-head ensemble**, which provides low-cost diversity on top of a shared pretrained representation. ([OpenReview][6])

Support several probe distributions:

* training inputs plus perturbations
* feature-space interpolations
* bounding-box samples
* unlabeled deployment inputs
* domain-specific physically plausible samples

The probe distribution is part of the statistical assumption and should be recorded in model metadata.

### C. Anchored ensembles

Anchored ensembles are regression-native and give each member a different regularization anchor, providing a more explicit prior than random initialization alone. ([Proceedings of Machine Learning Research][7])

They are especially valuable for:

* low-data regression
* extrapolation
* active learning
* scientific surrogate models

This could be implemented mostly through your existing `optimizer_factory` and fit infrastructure once per-member priors are first-class objects.

## Priority 2: robustness through ensemble composition

### D. Hyper-deep ensemble selection

Your current ensemble members generally share one architecture and training configuration. Hyper-deep ensembles and Neural Ensemble Search obtain diversity from architectures and hyperparameters, not only seeds, and have reported improved calibration and robustness to shift. ([NeurIPS Proceedings][8])

You do not need full NAS initially. Implement a pool selector:

```python
selector = EnsembleSubsetSelector(
    objective="nll",
    diversity_metric="prediction_correlation",
    budget=5,
)
selected = selector.select(candidate_models, calibration_loader)
```

Useful objectives:

[
\text{score}(S)
===============

\text{proper_score}(S)
+
\lambda,\text{redundancy}(S)
+
\gamma,\text{cost}(S).
]

Allow selection on:

* ordinary validation data
* synthetic shift validation environments
* held-out domains
* temporal validation data

This likely gives more practical value than adding another approximate BNN.

### E. Standard-plus-robust model mixtures

Support heterogeneous ensembles containing, for example:

* an ERM member
* an adversarially trained member
* a GroupDRO/domain-robust member
* a heavily regularized member
* a shift-adapted last-layer model

Calibrated mixtures of standard and robust models can mitigate ID/OOD accuracy tradeoffs. ([Proceedings of Machine Learning Research][9])

Your existing combiners are a starting point, but selection and calibration must use proper probabilistic scores, not just point MSE.

## Priority 3: efficient ensembles

### F. True Packed-Ensembles

Implement actual Packed-Ensembles instead of extending the current BatchEnsemble facade. Packed-Ensembles were designed to preserve deep-ensemble diversity while fitting within a single-model-style memory budget. ([arXiv][10])

For `torchregress`, start with:

* `PackedLinear`
* `PackedMLP`
* packed normalization
* heteroscedastic and quantile output heads

Conv/attention variants can follow.

### G. MIMO

MIMO produces multiple predictions in one model pass and is an important efficient-ensemble comparator. ([OpenReview][11])

For regression, the implementation needs careful handling of:

* repeated-input probability during training
* independent targets per channel
* heteroscedastic outputs per channel
* shared-input inference
* output correlation diagnostics

It should be marked experimental until its effective diversity is compared against a full deep ensemble.

### Lower priority

Snapshot/checkpoint ensembles and Masksembles are useful inexpensive baselines, but they are less important than DARE, repulsion, true Packed-Ensembles, MIMO and hyperparameter-diverse selection. TorchUncertainty already provides these efficient families, so matching its method count alone would not differentiate `torchregress`. ([Torch Uncertainty][12])

---

# The most important non-ensemble addition

## Weighted conformal regression under covariate shift

Your shift-aware conformal implementation is currently explicitly experimental and classification-style.

For regression deployment, implement:

1. Density-ratio estimation from source and unlabeled target covariates.
2. Weighted split conformal regression.
3. Weighted CQR.
4. Weighted conformal predictive distributions.
5. Effective-sample-size, maximum-weight and clipping diagnostics.
6. Cross-fitting for the ratio estimator.
7. Coverage warnings when support overlap is weak.

Weighted conformal prediction has a direct covariate-shift justification when (p_{\text{target}}(x)/p_{\text{source}}(x)) is known or adequately estimated. ([NeurIPS Proceedings][13])

Recent work also emphasizes that estimated or unbounded importance weights can produce severe undercoverage; clipping and explicit diagnostics therefore belong in the public API. ([arXiv][14])

For users who care about distribution shift, this will probably deliver more operational value than a seventh epistemic-uncertainty approximation.

---

# Recommended architecture refactor

The ensemble classes should be compositions of five independent policies:

```python
Ensemble(
    member_factory=...,
    data_strategy=...,
    diversity_strategy=...,
    aggregation_strategy=...,
    execution_strategy=...,
)
```

### `MemberFactory`

Controls architecture, seed, prior, hyperparameters and initialization.

### `MemberDataStrategy`

```python
SameData()
Bootstrap()
Subsample(fraction=0.8)
DomainBootstrap()
AugmentationEnsemble([...])
```

Bootstrapped deep ensembles are particularly relevant for regression because they attempt to capture finite-sample uncertainty beyond optimizer randomness. ([arXiv][15])

### `DiversityStrategy`

```python
Independent()
AnchoredPrior()
DARE()
FunctionSpaceRepulsion()
RandomizedPriorFunction()
```

### `AggregationStrategy`

```python
UniformMixture()
ProperScoreWeights()
Stacking()
ShiftConditionedWeights()
```

For probabilistic members, aggregation should happen in **distribution space**, not by averaging parameters.

### `ExecutionStrategy`

```python
Sequential()
Vectorized()
Distributed()
CPUOffload()
```

The current full-ensemble forward and fit paths execute members sequentially. The learned combiners have a `vmap` inference path, but the main base ensemble does not.

SOTA scalability requires at least:

* AMP
* `torch.compile`
* batched/vmapped inference
* distributed member training
* checkpoint/offload support
* deterministic member-level seeding
* peak-memory and latency benchmarks

---

# Use a richer uncertainty decomposition

A uniform output contract would materially improve the package:

```python
@dataclass
class EnsemblePrediction:
    mean: Tensor
    member_means: Tensor

    aleatoric_variance: Tensor | None
    within_member_epistemic_variance: Tensor | None
    between_member_epistemic_variance: Tensor
    predictive_variance: Tensor

    predictive_distribution: Distribution | None
    samples: Tensor | None
```

This handles:

* ordinary deep ensembles
* heteroscedastic ensembles
* MultiSWAG
* BNN ensembles
* MDN/flow ensembles
* multivariate regression

It also avoids forcing all uncertainty into the ambiguous two-bin aleatoric/epistemic decomposition.

---

# What “SOTA-ready” benchmark evidence should look like

Do not use only random train/test splits or toy extrapolation.

## Shift families

At minimum:

* support expansion and extrapolation
* feature mean/variance shift
* feature corruption and missingness
* temporal shift
* domain/geographic shift
* label noise shift
* conditional or concept shift
* new subpopulation shift

The real-world regression-shift benchmark with eight image regression datasets is valuable because it showed that methods that appeared calibrated ID often became overconfident under realistic shifts. ([arXiv][3])

WILDS-based evaluation provides another established comparison point for calibration and generalization under domain shift, including regression tasks. ([arXiv][16])

For tabular or scientific regression, temporal QSAR shift is a useful realistic stress test and demonstrates how uncertainty degradation tracks changes in both covariate and target spaces. ([arXiv][17])

## Baselines

Include:

* single deterministic model
* single heteroscedastic model
* deep ensembles with (M=3,5,10)
* bootstrapped deep ensemble
* BatchEnsemble
* true Packed-Ensemble
* MIMO
* SWAG and corrected MultiSWAG
* anchored ensemble
* DARE
* repulsive last-layer ensemble
* VIDS
* conformalized versions of the best predictors

## Metrics

Report all four categories:

**Prediction**

* RMSE, MAE
* subgroup and tail RMSE

**Probabilistic quality**

* NLL
* CRPS
* energy score for multivariate outputs

**Calibration and decision quality**

* coverage at multiple nominal levels
* mean and conditional coverage gap
* interval width
* PIT diagnostics
* risk–coverage/AURC
* error–uncertainty rank correlation

**Ensemble quality and cost**

* pairwise predictive correlation
* effective rank of the member-prediction matrix
* disagreement on shifted inputs
* parameters and checkpoint size
* training FLOPs/time
* peak memory
* inference throughput and latency

Report Pareto fronts rather than declaring one universal winner. Large-scale studies find that multi-mode ensembling generally helps under shift, but the best method depends on architecture, fine-tuning regime, accuracy and calibration target. ([NeurIPS Proceedings][18])

# Concrete development order

1. **Correctness release**

   * independent member initialization
   * `correction=0` predictive variance
   * full-covariance fix
   * corrected MultiSWAG decomposition
   * rename fake Packed-Ensemble and BMA classes
   * exact numerical tests

2. **Shift robustness release**

   * DARE
   * function-space repulsion and repulsive last-layer heads
   * anchored ensembles
   * weighted conformal regression
   * realistic shift benchmark suite

3. **Efficient ensemble release**

   * true Packed-Ensembles
   * MIMO
   * vectorized and distributed execution
   * diversity/cost diagnostics

4. **Ensemble selection release**

   * candidate pools
   * hyper-deep subset selection
   * heterogeneous robust/standard mixtures
   * shift-aware validation objectives

After those changes, `torchregress` would have a defensible differentiator: **a regression-first ensemble framework covering arbitrary predictive distributions, explicit uncertainty decomposition, realistic shift evaluation, and conformal deployment guarantees**. Adding more method names before correcting the present semantics would make the catalog broader, but not more SOTA-ready.

[1]: https://openreview.net/forum?id=XXTyv1zD9zD&utm_source=chatgpt.com "Packed Ensembles for efficient uncertainty estimation | OpenReview"
[2]: https://arxiv.org/abs/2601.16936?utm_source=chatgpt.com "Is BatchEnsemble a Single Model? On Calibration and Diversity of Efficient Ensembles"
[3]: https://arxiv.org/abs/2302.03679?utm_source=chatgpt.com "How Reliable is Your Regression Model's Uncertainty Under Real-World Distribution Shifts?"
[4]: https://dblp.org/rec/journals/corr/abs-2506-03942.html?utm_source=chatgpt.com "dblp: Average Calibration Losses for Reliable Uncertainty in Medical Image Segmentation."
[5]: https://arxiv.org/abs/2304.04042?utm_source=chatgpt.com "Deep Anti-Regularized Ensembles provide reliable out-of-distribution uncertainty quantification"
[6]: https://openreview.net/forum?id=LAKplpLMbP8&utm_source=chatgpt.com "Repulsive Deep Ensembles are Bayesian | OpenReview"
[7]: https://proceedings.mlr.press/v108/pearce20a.html?utm_source=chatgpt.com "Uncertainty in Neural Networks: Approximately Bayesian Ensembling"
[8]: https://proceedings.neurips.cc/paper/2020/hash/481fbfa59da2581098e841b7afc122f1-Abstract.html?utm_source=chatgpt.com "Hyperparameter Ensembles for Robustness and Uncertainty Quantification"
[9]: https://proceedings.mlr.press/v180/kumar22a.html?utm_source=chatgpt.com "Calibrated ensembles can mitigate accuracy tradeoffs under distribution shift"
[10]: https://arxiv.org/abs/2210.09184?utm_source=chatgpt.com "Packed-Ensembles for Efficient Uncertainty Estimation"
[11]: https://openreview.net/forum?id=OGg9XnKxFAH&utm_source=chatgpt.com "Training independent subnetworks for robust prediction | OpenReview"
[12]: https://torch-uncertainty.github.io/quickstart.html?utm_source=chatgpt.com "Quickstart — TorchUncertainty Docs"
[13]: https://proceedings.neurips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html?utm_source=chatgpt.com "Conformal Prediction Under Covariate Shift"
[14]: https://arxiv.org/abs/2605.02072?utm_source=chatgpt.com "Weight Clipping for Robust Conformal Inference under Unbounded Covariate Shifts"
[15]: https://arxiv.org/abs/2202.10903?utm_source=chatgpt.com "Confident Neural Network Regression with Bootstrapped Deep Ensembles"
[16]: https://arxiv.org/abs/2306.12306?utm_source=chatgpt.com "Beyond Deep Ensembles: A Large-Scale Evaluation of Bayesian Deep Learning under Distribution Shift"
[17]: https://arxiv.org/abs/2502.03982?utm_source=chatgpt.com "Temporal Distribution Shift in Real-World Pharmaceutical Data: Implications for Uncertainty Quantification in QSAR Models"
[18]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/5d97b7e62022c859347397f6c1e8d0f9-Abstract-Conference.html?utm_source=chatgpt.com "Beyond Deep Ensembles: A Large-Scale Evaluation of Bayesian Deep Learning under Distribution Shift"
