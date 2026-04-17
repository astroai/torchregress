# Balanced MSE demo

Short synthetic run comparing plain [`MSELoss`](../losses/base.md) with [`BalancedMSELoss`](../losses/imbalanced.md) and [`BMCLoss`](../losses/imbalanced.md) on a skewed target distribution.

## Script

[`examples/balanced_mse_demo.py`](https://github.com/sfabbro/torchregress/blob/main/examples/balanced_mse_demo.py)

## Run

```bash
uv run python examples/balanced_mse_demo.py --steps 80 --n 256 --seed 0
```

Tune `--lr` if the reported training MSE diverges on your machine.
