# Gaussian Wasserstein bound demo

This script shows a single forward–backward step with
[`GaussianWassersteinBoundLoss`](../losses/gaussian_wasserstein.md) in **full covariance**
mode on batched SPD matrices.

## Run

```bash
uv run python examples/gaussian_wasserstein_bound_demo.py --batch 16 --dim 3 --seed 0
```

## Code

[`examples/gaussian_wasserstein_bound_demo.py`](https://github.com/sfabbro/torchregress/blob/main/examples/gaussian_wasserstein_bound_demo.py)

## See also

- [Gaussian Wasserstein bound reference](../losses/gaussian_wasserstein.md)
- [Multivariate Gaussian NLL](../losses/gaussian.md)
