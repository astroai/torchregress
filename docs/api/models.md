# Models API

Complete reference for `torchregress.models`: a strong-default MLP for tabular
regression, its input preprocessing and a training helper. For background,
equations and when to use it, see [Tabular MLP](../methods/tabular_mlp.md).

```python
from torchregress.models import (
    TabularMLP, PeriodicEmbedding, TabularPreprocessor,
    TabularFit, TabularEnsembleFit, fit_tabular, fit_tabular_ensemble,
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
