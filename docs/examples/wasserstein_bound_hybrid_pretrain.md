# Wasserstein-bound hybrid pretrain demo

Short [`GaussianWassersteinBoundLoss`](../losses/gaussian_wasserstein.md) phase on **diagonal**
pseudo-covariance targets from [`NeighborhoodCovariancePseudoLabeler`](https://github.com/sfabbro/torchregress/blob/main/torchregress/algorithms/covariance_pseudo_labels.py), followed by [`GaussianNLLLoss`](../losses/gaussian.md) fine-tuning on a tiny linear head.

## Run

```bash
uv run python examples/wasserstein_bound_hybrid_pretrain_demo.py --pretrain-steps 40 --finetune-steps 60 --seed 0
```

## Code

[`examples/wasserstein_bound_hybrid_pretrain_demo.py`](https://github.com/sfabbro/torchregress/blob/main/examples/wasserstein_bound_hybrid_pretrain_demo.py)

## See also

- [Gaussian Wasserstein bound reference](../losses/gaussian_wasserstein.md)
- [Gaussian Wasserstein bound demo](gaussian_wasserstein_bound.md)
