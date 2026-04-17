# Beta-NLL heteroscedastic demo

This example trains a small Gaussian-head MLP on **synthetic heteroscedastic** data (noise scale grows with $|x|$), starting from the **same initial weights** for two runs:

1. [`GaussianNLLLoss`](../losses/gaussian.md) (plain NLL)
2. [`BetaNLLLoss`](../losses/beta_nll.md) with $\beta = 0.5$

Both runs are scored on the same validation split using **Gaussian NLL** and **RMSE**, so you can compare calibration of the predictive Gaussian (NLL) and point error (RMSE) under a shared metric.

## Run

```bash
uv run python examples/heteroscedastic_beta_nll_demo.py --epochs 80 --seed 0
```

Tune `--epochs`, `--lr`, and `--seed` for your environment.

## Code

The script lives at [`examples/heteroscedastic_beta_nll_demo.py`](https://github.com/sfabbro/torchregress/blob/main/examples/heteroscedastic_beta_nll_demo.py).

## See also

- [Beta-NLL reference](../losses/beta_nll.md)
- [Gaussian losses](../losses/gaussian.md)
