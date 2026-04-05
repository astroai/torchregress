# EIV Round Conclusion

This note freezes the current conclusion of the photo-z EIV work so the next round can focus on test-time methods.

## What held up

- The strongest practical EIV-style family for photo-z is still the `InputNoise*` family.
- Among those, the most promising methods are:
  - `InputNoiseMDN`
  - `InputNoiseBinnedPDF`
  - `InputNoiseGaussianCRPS` as a simpler Gaussian reference
- These methods are best understood as robustness and uncertainty-propagation methods, not as direct attenuation-bias correction.

## What did not become winners

- Classical EIV corrections are no longer obviously broken, but they are not strong photo-z defaults:
  - `RegressionCalibration`
  - `SIMEX`
  - `FunctionalEIV`
  - `StructuralEIV`
  - `OrthogonalDistanceRegression`
  - `EnsembleEIV`
- `LatentNN` is a useful attenuation benchmark, but it does not dominate the stronger probabilistic photo-z methods in our current synthetic or real-data comparisons.

## Important conceptual conclusion

There are two different problem classes:

- Attenuation/deattenuation methods:
  - `RegressionCalibration`
  - `SIMEX`
  - `LatentNN`
- Robust predictive uncertainty methods:
  - `InputNoiseMDN`
  - `InputNoiseBinnedPDF`
  - `InputNoiseGaussianCRPS`

The current evidence says the second class is more useful for real photo-z, while the first class is mainly useful as a benchmark or bias-analysis reference.

## Hybrid error-aware work

- Error-aware encoders were added and tested.
- Hybrid `NoiseAware + InputNoise` variants are stable and principled.
- They did not beat the best plain `InputNoise*` methods in the current noisy-feature photo-z runs.

So they should remain experimental, not the default torchregress recommendation.

## Torchregress recommendation after this round

Keep and support:

- `InputNoiseMDN`
- `InputNoiseBinnedPDF`
- `InputNoiseGaussianCRPS`
- `LatentNN`
- `RegressionCalibration`
- `SIMEX`

Promote for photo-z:

- `InputNoiseMDN`
- `InputNoiseBinnedPDF`

Keep as reference baselines, not default recommendations:

- `LatentNN`
- `RegressionCalibration`
- `SIMEX`
- classical loss-based EIV methods

## What is complete enough to stop here

- Explicit EIV adapter surface
- Full-covariance input-noise sampling
- Test-time marginalization for `InputNoise*`
- Error-aware encoder/regressor
- Synthetic attenuation benchmark
- Real noisy-feature comparisons in `torchz`

## What we are intentionally not pushing further in this round

- More classical EIV variants
- More posterior-centered input-noise variants
- More hybrid error-aware + input-noise tuning

Those can come back later if test-time methods reveal a specific gap.

## Next round

Focus on test-time methods on top of the best current probabilistic stacks:

- `MDN`
- `BinnedPDF`
- `InputNoiseMDN`
- `InputNoiseBinnedPDF`

Priority test-time themes:

- test-time adaptation
- test-time calibration
- test-time marginalization / refinement
- low-SNR slice handling at inference
