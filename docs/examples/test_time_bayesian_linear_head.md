# Bayesian linear head (test-time) demo

This script fits a [`BayesianLinearHead`](../methods/test-time/bayesian-linear-regression.md) on
synthetic linear data, checks agreement with a two-step
[`RecursiveBayesianHead`](../methods/test-time/bayesian-linear-regression.md) at `forgetting_factor=1`,
and prints a rough held-out coverage statistic for Gaussian predictive intervals.

## Run

```bash
uv run python examples/test_time_bayesian_linear_head_demo.py --n-train 200 --n-test 500 --dim 5 --seed 0
```

## Code

[`examples/test_time_bayesian_linear_head_demo.py`](https://github.com/sfabbro/torchregress/blob/main/examples/test_time_bayesian_linear_head_demo.py)

## See also

- [Bayesian linear regression (test-time)](../methods/test-time/bayesian-linear-regression.md) (includes **benchmark scripts** under `examples/benchmarks/`)
- [OT shift conformal demo](ot_shift_conformal_demo.md)
