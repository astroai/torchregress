# torchregress Handoff (2026-04-03)

## Current package boundary

`torchregress` should stay at the method/tooling layer.

It now owns:

- regression losses and uncertainty methods
- EIV and error-aware methods
- calibration and conformal tooling
- predictive containers and density-conversion helpers
- test-time utilities

It should not own:

- opinionated neural architectures
- tabular foundation-model wrappers
- application-specific benchmark runners

Those belong in application repos such as `torchz`.

## Main completed areas

### EIV / noisy-feature work

- `RegressionCalibration`
- `SIMEX`
- `LatentNN`
- `InputNoise*`
- error-aware encoders

Curated notes:

- `EIV_ROUND_CONCLUSION.md`
- `SIMEX_SIMFEX_NOTE.md`

Main recommendation from this round:

- keep `InputNoiseMDN` and `InputNoiseBinnedPDF` as the strongest practical photo-z EIV-style tools
- keep `LatentNN`, `RegressionCalibration`, and `SIMEX` as benchmark/reference methods

### Test-time tooling

New reusable modules:

- `torchregress.prediction.PredictiveBatch`
- `torchregress.test_time.label_shift`
- `torchregress.test_time.selection`
- `torchregress.test_time.subspace`
- `torchregress.test_time.calibration`
- `torchregress.test_time.dynamic`

These are intended as substrates for:

- DistPFN-T style posterior correction
- AdapTable-style output correction
- FTAT / PFT3A style confidence filtering and local consistency
- SSA-style regression adaptation

## Validation run for this handoff

Executed:

```bash
PYTHONPATH=/Users/seb/src/torchregress /Users/seb/src/torchz/.venv/bin/python -m pytest \
  tests/algorithms/test_rc.py \
  tests/algorithms/test_simex.py \
  tests/algorithms/test_error_aware.py \
  tests/algorithms/test_latentnn.py \
  tests/losses/test_eiv.py \
  tests/test_metrics.py \
  tests/test_prediction_and_test_time.py \
  tests/test_public_api_contracts.py
```

## Next recommended steps

1. Build higher-level test-time recipes on top of `torchregress.test_time`.
2. Keep model implementations in `torchz` or model-specific repos.
3. Compare test-time pipelines on `torchz` photo-z benchmarks rather than adding more architecture ownership here.
