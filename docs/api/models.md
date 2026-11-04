# Models API

Complete reference for `torchregress.models`: a strong-default MLP for tabular
regression, its input preprocessing and a training helper. For background,
equations and when to use it, see [Tabular MLP](../methods/tabular_mlp.md).

```python
from torchregress.models import (
    TabularMLP, PeriodicEmbedding, TabularPreprocessor,
    TabularFit, TabularEnsembleFit, fit_tabular, fit_tabular_ensemble,
    FTTransformer, FeatureTokenizer,
    FlowMatchingMLP, cfm_loss, sample_cfm,
    NormalizingFlowMap,
)
```

---

## TabularMLP

`TabularMLP(in_features, out_features, hidden=(256, 256, 256), embedding="periodic", n_frequencies=16, embedding_dim=8, frequency_scale=0.5, dropout=0.1, activation="silu")`

`nn.Module` mapping preprocessed features `(batch, in_features)` to raw head
outputs `(batch, out_features)`.

| Argument | Default | Description |
|:---------|:-------:|:------------|
| `in_features` | — | Input columns (`TabularPreprocessor.n_features_out_`) |
| `out_features` | — | Head width; match the loss (`D`, `2 * D` for `GaussianNLLLoss`, `len(quantiles)` for `MultiQuantileLoss`) |
| `hidden` | `(256, 256, 256)` | Hidden widths |
| `embedding` | `"periodic"` | `"periodic"` or `"none"` |
| `n_frequencies` | 16 | Frequencies per feature |
| `embedding_dim` | 8 | Per-feature embedding width |
| `frequency_scale` | 0.5 | Std of the frequency initialisation |
| `dropout` | 0.1 | Dropout after each hidden activation |
| `activation` | `"silu"` | `"silu"`, `"selu"`, `"relu"`, `"gelu"`, `"mish"` or a zero-argument module factory |

## PeriodicEmbedding

`PeriodicEmbedding(n_features, n_frequencies=16, embedding_dim=8, frequency_scale=0.5)`

Per-feature $\operatorname{ReLU}(W_j[\sin 2\pi c_j x_j \| \cos 2\pi c_j x_j] + b_j)$
concatenated with the raw value (Gorishniy et al., 2022). `out_features` is
`n_features * (embedding_dim + 1)`.

## TabularPreprocessor

`TabularPreprocessor(clip="smooth", clip_value=3.0, missing_indicator="auto", dtype=None)`

| Member | Description |
|:-------|:------------|
| `fit(X)` / `transform(X)` / `fit_transform(X)` | NumPy in → NumPy out, tensor in → tensor out; median/IQR scaling, clipping, median imputation of NaN |
| `inverse_transform(Z)` | Exact for `clip` in `{"smooth", None}`; restores NaN where an indicator is set |
| `n_features_in_`, `n_features_out_` | Widths; out = in + number of missing-indicator columns |
| `center_`, `scale_`, `indicator_columns_` | Fitted statistics |
| `is_fitted` | Whether `fit` was called |

## fit_tabular

`fit_tabular(model, loss_fn, X_train, y_train, *, X_val=None, y_val=None, val_fraction=0.15, epochs=100, batch_size=256, lr=2e-3, weight_decay=1e-2, patience=20, seed=0, device=None, scheduler="onecycle", grad_clip=1.0, standardize_target=True, output_layout="auto", preprocessor=None) -> TabularFit`

Trains with AdamW and early stopping; calls `loss_fn(model(x), y)` with `y` of
shape `(batch, D)` in standardised units. Restores the best-validation weights.
Leaves the global torch RNG untouched.

## TabularFit

| Attribute / method | Description |
|:-------------------|:------------|
| `model`, `preprocessor` | Trained module (eval mode) and fitted preprocessor |
| `history` | `train_loss`, `val_loss`, `lr` per epoch (standardised target units) |
| `best_epoch`, `best_val_loss`, `stopped_early` | Early-stopping outcome |
| `target_mean`, `target_std`, `output_layout` | Target standardisation and output mapping |
| `predict(X, batch_size=8192)` | Head outputs in the original target scale, ready for the training loss |

## fit_tabular_ensemble

`fit_tabular_ensemble(make_model, loss_fn, X_train, y_train, *, n_members=5, seed=0, **fit_kwargs) -> TabularEnsembleFit`

Trains `n_members` models from seeds `seed, seed + 1, …`; `make_model(in_features)`
builds each one. `TabularEnsembleFit.predict_members(X)` returns
`(n_members, n, out_features)`; `predict(X)` returns their mean.


## FlowMatchingMLP / cfm_loss / sample_cfm

`FlowMatchingMLP(n_features, context_dim, *, hidden=(128, 128), time_embedding=16)`

Time-conditioned MLP velocity field for conditional flow matching.

`cfm_loss(velocity, context, target, *, generator=None, eps=1e-3)`

OT-CFM velocity-regression loss; returns a scalar mean loss.

`sample_cfm(velocity, context, *, n_samples=128, steps=32, generator=None)`

Euler sampler returning `[B, n_samples, n_features]`.  CFM has no exact
likelihood; use sample-based CRPS/energy/coverage/PIT or an ODE likelihood.


## NormalizingFlowMap

`NormalizingFlowMap(n_features, context_dim, *, n_layers=4, hidden=(128, 128), time_features=16, map_weight=1.0, nll_weight=0.1, arch="coupling", tarflow_layers=3, tarflow_heads=4)`

Time-conditioned invertible affine-coupling flow map.  It provides
`forward(x, s, t, context)`/`inverse(z, s, t, context)`, exact
`log_prob(context, target)`, `sample(context, n_samples)`,
`flow_map_loss`, and `training_loss` (flow-map matching + endpoint NLL).
See [Normalizing Flow Maps](../methods/normalizing_flow_maps.md).

`GaussianCopulaSplineHead` is a conditional, full-vector spline flow built from a full-covariance Gaussian base and coordinatewise monotone rational-quadratic splines. Intrinsic likelihoods for any observed subset marginalize missing coordinates analytically. Gaussian label-noise convolution uses caller-supplied finite normal quadrature nodes and requires convergence checks. The copula is Gaussian: the model cannot express arbitrary nonlinear or tail dependence. It requires the optional `flows` extra.
