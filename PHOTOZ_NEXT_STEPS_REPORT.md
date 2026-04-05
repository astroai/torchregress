# Torchregress Next Steps For Photo-z

This report summarizes what the `torchz` + `torchregress` work suggests we should
do next in `torchregress`, based on:

- real tabular benchmarks on `CANDELS` and `GalaxiesML`
- noisy-label, noisy-feature, imbalanced-tail, and shift stress scenarios
- work on deep ensembles, CRPS, MDN, evidential, regression-as-classification,
  conformal wrappers, and EIV/noisy-input methods
- the recent foundation-model benchmark work inspired by
  [Izbicki & Rodrigues 2026](https://arxiv.org/html/2603.26611v1)

The goal is to make `torchregress` stronger, more reliable, and more honest for
probabilistic photo-z workflows.

## Executive Summary

The clearest message from the current evidence is:

1. `CRPS`, `MDN`, calibrated distributional methods, and stable ensembles are worth
   promoting.
2. the current `EIV` and imbalance story is not yet strong enough to recommend as a
   default for photo-z.
3. `conformal` and regression-as-classification methods help calibration and
   sometimes point accuracy, but they are not automatic winners and should not be
   oversold.
4. `torchregress` needs a more ergonomic and more competitive noisy-input story.
5. the next serious comparison surface is tabular foundation models, using the same
   density-aware metric suite.

If we want the biggest near-term impact, the next `torchregress` work should be:

1. stabilize and simplify the probabilistic APIs we already have
2. redesign `EIV` around input-noise marginalization and multimodal heads
3. make density-oriented metrics first-class and universal
4. update method recommendations to match empirical results
5. benchmark against `TabPFN` and `TabICL` before adding more exotic losses

## What The Benchmarks Suggest

### What is working well

- `GaussianCRPS` is a real improvement over plain Gaussian NLL on several photo-z
  surfaces. It improved both point metrics and probabilistic quality in multiple
  `GalaxiesML` runs.
- `MDN` is one of the strongest current probabilistic families in practice.
  It is competitive on point metrics and often the best learned method on `NLL`.
- `binned_pdf` and ordinal/classification-style methods are competitive, especially
  when combined with temperature scaling or conformal wrappers.
- `deep ensembles` are useful when implemented carefully with stable variance
  handling. They improved `RMSE`, catastrophic fraction, and some calibration
  metrics on `CANDELS`.
- `CRPS` should remain first-class. It is one of the most useful reality checks
  when a model looks good on `NLL` but behaves poorly in other ways.

### What is not working well enough yet

- `FunctionalEIVLoss` is not competitive enough on real photo-z tables and is too
  awkward to use. The API is hard to integrate, and the empirical results were weak.
- `DensityWeightedLoss` and `LDSLoss` were often worse than simpler baselines in the
  real photo-z stress tests.
- the current noisy-target story is mixed. `NoisyTargetGaussianNLL` helps in some
  probabilistic settings but is not a dominant winner on classical photo-z metrics.
- feature-perturbation / uncertainty-aware input augmentation did not shine on clean
  data, and only helped selectively in noisy-feature scenarios.
- conformal wrappers improved coverage and calibration, but did not consistently
  improve the classical photo-z metrics that users care most about.

### Important caution from the results

Several intuitively appealing “stack everything” pipelines did **not** dominate:

- `EIV + density weighting + conformal + ensemble + multimodal head`
- `feature-perturbed Gaussian`
- older `FunctionalEIV` routes

The evidence so far says these compositions can help, but only selectively. The
extra complexity is not automatically buying better `RMSE`, `NMAD`, or catastrophic
fraction.

## Main Problems To Solve In Torchregress

## 1. EIV is not yet a strong recommended method

### Problem

The current `EIV` methods look more like a research surface than a mature default.
The older Jacobian / orthogonal-distance style API is unintuitive, and the empirical
results did not justify recommending it for photo-z.

### What to do next

Prioritize a new `EIV` track based on **input-noise marginalization** rather than
the old loss-first formulation.

Concrete next work:

- keep `InputNoiseMarginalizationLoss` as the mainline noisy-input route
- add stronger multimodal variants:
  - `InputNoiseMDN`
  - `InputNoiseBinnedPDF`
  - later `InputNoiseCumulativeLink`
- add ensemble variants:
  - `DeepEnsembleInputNoiseGaussian`
  - `DeepEnsembleInputNoiseMDN`
- benchmark them specifically on:
  - explicit noisy-feature corruption
  - cross-dataset / shift scenarios

### Recommended API direction

The primary noisy-input interface should look like:

```python
loss = InputNoiseMarginalizationLoss(
    base_loss=...,
    n_samples=...,
    antithetic=True,
)
loss(y_pred, y_true, x_obs=x_obs, sigma_x=sigma_x)
```

or with a helper wrapper around the model:

```python
predictive = NoisyInputPredictor(model, sigma_x=sigma_x, n_samples=...)
```

That is much easier to explain and integrate than the current “loss takes observed
inputs as `y_pred`” pattern.

### Recommendation

- downgrade `FunctionalEIVLoss` as a recommended starting point for photo-z
- keep it as an advanced or legacy method until it wins a benchmark slice

## 2. Imbalance handling needs to be demoted and rethought

### Problem

The imbalance losses did not earn default status in the photo-z benchmarks.
`DensityWeightedLoss` in particular often hurt.

### What to do next

- stop treating `DensityWeightedLoss` and `LDSLoss` as broadly recommended defaults
- add better benchmark-backed guidance about when they help
- consider less intrusive imbalance strategies:
  - tail-aware evaluation and selection, not only tail-aware training
  - density-aware calibration layers
  - post-hoc interval widening in under-covered tails

### Recommendation

Until the evidence changes, `torchregress` should not present density weighting as
one of the first things to try for photo-z.

## 3. Distribution metrics should be universal and first-class

### Problem

The photo-z work repeatedly showed that relying only on point metrics or only on
`NLL` is misleading. We needed `CRPS`, `PIT`, `coverage`, `HPD`, and now `CDE loss`
to understand what was actually happening.

### What to do next

Build one universal distribution-metric report that works consistently for:

- Gaussian outputs
- quantile outputs
- samples
- binned / discrete PDFs
- mixture distributions

Required metrics:

- `NLL` or log-likelihood when available
- `CRPS`
- `CDE loss`
- `PIT chi-square`
- `PIT KS`
- `coverage_90`
- interval width / interval score
- optional `HPD` coverage and calibration

### Recommendation

Make this density-report helper central in docs and examples. Users should not have
to compose five separate metric calls manually.

## 4. Ensembles should expand beyond Gaussian

### Problem

The current ensemble story became much better once Gaussian variance handling was
fixed, but Gaussian-only ensembling is too narrow for modern photo-z work.

### What to do next

Keep building the non-Gaussian ensemble family:

- `BinnedPDFEnsembleModel`
- `CumulativeLinkEnsembleModel`
- `MDNEnsembleModel`

The key principle is correct:

- average predictive **distributions**, not raw parameters, unless the parameter
  space is stable enough
- for `MDN`, keep the mixture-of-mixtures approach, not naive parameter averaging

### Recommendation

Push these models into the benchmark and docs as first-class uncertainty methods,
not just experimental extras.

## 5. Conformal methods need more precise messaging

### Problem

Conformal methods improved coverage and sometimes helped robustness, but they were
not automatic winners on the main photo-z metrics.

### What to do next

- document conformal methods as **coverage tools first**
- make it explicit that they may trade some sharpness or point accuracy for better
  coverage guarantees
- separate “density estimation quality” from “coverage guarantee quality” in method
  recommendations and benchmark summaries

### Recommendation

Keep conformal methods in the recommended set, but as a different category:

- best when coverage guarantees matter
- not necessarily best when optimizing `NMAD` or `RMSE`

## 6. Method recommendations should reflect evidence, not intuition

### Problem

Some methods looked like they should win on paper but did not.
The method catalog should not oversell them.

### What to do next

Update `method_catalog.py` and docs so the “recommended start” path for photo-z is
closer to:

- `GaussianCRPS`
- `MDN`
- `binned_pdf` with calibration / conformalization
- calibrated ensembles

and less centered on:

- `DensityWeightedLoss`
- `FunctionalEIVLoss`
- generic noisy-target methods as universal defaults

## Foundation Models: Why They Matter For Torchregress

The recent tabular foundation-model benchmark paper is a serious warning that our
next competition is not only hand-designed tabular neural losses, but also strong
pretrained CDE systems such as `TabPFN` and `TabICL`.

The paper reports strong performance for TFMs on:

- `CDE loss`
- log-likelihood
- `CRPS`

and notes that calibration can still lag at larger scales, which is exactly the kind
of gap `torchregress` should try to exploit.

### What this means for torchregress

`torchregress` should aim to be:

1. the best toolkit for training custom probabilistic regressors
2. the best toolkit for evaluating, calibrating, and comparing them against
   foundation models

That means:

- density metrics need to be excellent
- calibration tooling needs to be excellent
- the docs should include honest comparisons to tabular foundation models

## Concrete Priority Plan

## Phase 1: Reliability and API cleanup

1. finish the noisy-input API redesign around input-noise marginalization
2. keep ensemble variance handling numerically safe everywhere
3. expose one universal density-report helper
4. improve docs and examples for:
   - `CRPS`
   - `MDN`
   - calibrated ensembles
   - conformal wrappers

## Phase 2: Competitive EIV research

1. add:
   - `InputNoiseMDN`
   - `InputNoiseBinnedPDF`
2. benchmark them against:
   - `GaussianCRPS`
   - `MDN`
   - `binned_pdf + conformal`
3. test them on:
   - noisy features
   - real shift scenarios
   - later `CLAUDS`

## Phase 3: Foundation-model comparison

1. benchmark `torchregress` probabilistic pipelines against:
   - `TabPFN`
   - `TabICL`
2. focus on:
   - `CDE loss`
   - `CRPS`
   - `PIT`
   - `coverage`
   - photo-z classical metrics
3. see whether `torchregress` wins by:
   - custom loss design
   - calibration
   - noisy-input handling
   - domain-specific robustness

## Phase 4: Recommendation update

After the TFM comparisons:

- rewrite the photo-z recommendation section in the docs
- explicitly downgrade methods that still fail to win
- promote the methods that consistently survive across datasets

## Proposed “Recommended Start” Today

If we had to recommend a small set **today**, before the next benchmark round, it
would be:

1. `GaussianCRPS`
2. `MDN`
3. `binned_pdf` with calibration or conformalization
4. calibrated deep ensembles

And for things to treat as **research / advanced**, not default:

1. `FunctionalEIVLoss`
2. `DensityWeightedLoss`
3. `LDSLoss`
4. large stacked “everything at once” pipelines

## Open Questions

These are the questions the next benchmark round should answer:

1. Can `InputNoiseMDN` or `InputNoiseBinnedPDF` make EIV genuinely competitive?
2. Can `torchregress` beat `TabPFN` or `TabICL` on calibration and robustness even
   if they win on generic density metrics?
3. Which methods actually survive across:
   - `CANDELS`
   - `GalaxiesML`
   - later `CLAUDS`
4. Are there specific photo-z regimes where:
   - multimodality matters most
   - conformal methods matter most
   - noisy-input methods matter most

## Bottom Line

The next best move for `torchregress` is **not** adding more exotic methods at the
margin.

It is:

- clean up and strengthen the probabilistic methods that already look good
- redesign `EIV` around a more robust noisy-input interface
- make density and calibration metrics universal and easy to use
- benchmark honestly against tabular foundation models

That is the shortest path to making `torchregress` more competitive, more useful,
and easier to trust in real photo-z work.

