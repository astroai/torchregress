# BLR predictive adapter (SupportsPredictiveBatch)

Small adapter example wrapping [`BayesianLinearHead`](../methods/test-time/bayesian-linear-regression.md) with a
`predict_distribution(...)` method so it satisfies the
`SupportsPredictiveBatch` protocol from `torchregress.test_time.base`.

This is useful when adaptation utilities expect a model-like object returning
`PredictiveBatch`.

## Run

```bash
uv run python examples/test_time_blr_predictive_adapter_demo.py --n-train 96 --n-test 24 --dim 4 --seed 0
```

## Code

[`examples/test_time_blr_predictive_adapter_demo.py`](https://github.com/sfabbro/torchregress/blob/main/examples/test_time_blr_predictive_adapter_demo.py)
