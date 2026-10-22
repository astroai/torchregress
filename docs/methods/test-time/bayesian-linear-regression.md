# Bayesian linear regression heads

> ← [OT Shift Conformal](ot-shift-conformal.md) | [Shift Roadmap](regression-shift-adaptation-roadmap.md) →

`torchregress.test_time` exposes conjugate Gaussian linear regression on **fixed features** (a design matrix $\Phi$), useful for lightweight test-time heads on top of frozen representations.

## Model

The head assumes a linear-Gaussian readout with a Gaussian weight prior,

$$
y = \Phi w + \varepsilon,\qquad \varepsilon \sim \mathcal{N}(0, \sigma^2 I),\qquad w \sim \mathcal{N}(m_0, \Lambda_0^{-1}),
$$

so the posterior is available in closed form. After seeing rows $\Phi$ with targets $y$ and optional sample weights $W$, the precision and canonical parameter are

$$
\Lambda = \Lambda_0 + \sigma^{-2}\Phi^\top W \Phi, \qquad h = \Lambda_0 m_0 + \sigma^{-2}\Phi^\top W y,
$$

and the posterior is $w \mid \mathcal{D} \sim \mathcal{N}(m, S)$ with $m = \Lambda^{-1}h$ and $S = \Lambda^{-1}$. For a test row $\phi(x)$ the predictive distribution is Gaussian with

$$
\mathbb{E}[y \mid x] = \phi(x)^\top m, \qquad
\operatorname{Var}[y \mid x] = \underbrace{\sigma^2}_{\text{aleatoric}} + \underbrace{\phi(x)^\top S\,\phi(x)}_{\text{epistemic}},
$$

which is exactly the `include_noise` split exposed by `predict(..., return_std=True)`: epistemic variance shrinks where the design matrix carries information, while the noise floor $\sigma^2$ does not. A `RecursiveBayesianHead` applies the same update batch by batch, optionally discounting old precision with $\Lambda \leftarrow \lambda\Lambda$ before each step.

```python
import torch
from torchregress.test_time import BayesianLinearHead

head = BayesianLinearHead(in_features=16, out_features=1,
                          prior_precision=1.0, noise_variance=0.1)
head.fit(Phi_train, y_train)                      # closed-form posterior update

pred = head.predict(Phi_test, return_std=True)    # {"mean", "variance", "std"}
epistemic = head.predict(Phi_test, return_std=True,
                         include_noise=False)["variance"]
```

## API

| Class | Role |
| --- | --- |
| `BayesianLinearHead` | One-shot `fit`; maintains posterior precision $\Lambda$ and canonical $h = \Lambda m$. |
| `RecursiveBayesianHead` | `partial_fit` for streaming data; optional `forgetting_factor` in $(0,1]$ scales $\Lambda$ before each mini-batch. |

Both support:

- optional intercept column (`fit_intercept`);
- diagonal isotropic prior on weights (`prior_precision` on $\Lambda_0$);
- homoscedastic Gaussian noise (`noise_variance`);
- **multi-output**: `out_features > 1` uses the **same** $\Lambda$ and independent $h$ rows per output;
- `predict(..., return_std, include_noise)`;
- `predictive_batch` returning `PredictiveBatch` with `extra` diagnostics (`epistemic_variance`, `aleatoric_variance`, `posterior_trace`, `n_observations_seen`);
- `sample_weights(n_samples)` for draws $w \sim \mathcal{N}(m, S)$ with posterior covariance $S = \Lambda^{-1}$ (with jitter).

## Notes

- Call `fit` before `predict` / `sample_weights`. `fit` resets the posterior to the prior then accumulates the batch.
- With `forgetting_factor=1`, disjoint `partial_fit` batches commute and match a single `fit` on the concatenation.
- Forgetting shrinks past precision mass; it is **not** equivalent to refitting on finite windows unless you also reset $h$ (this implementation only scales $\Lambda$ as specified in the research plan).

## Synthetic benchmarks (Milestone 3)

Runnable scripts (CPU-only, no dataset downloads):

| Script | Purpose |
| --- | --- |
| [`examples/benchmarks/bayesian_linear_head_lowshot_adaptation.py`](https://github.com/astroai/torchregress/blob/main/examples/benchmarks/bayesian_linear_head_lowshot_adaptation.py) | Varying training set sizes: RMSE vs ridge MAP (matched prior) and Gaussian NLL with vs without predictive variance. |
| [`examples/benchmarks/bayesian_linear_head_online_drift.py`](https://github.com/astroai/torchregress/blob/main/examples/benchmarks/bayesian_linear_head_online_drift.py) | Label generator switches between two weight vectors; streaming `RecursiveBayesianHead` under several forgetting factors vs a phase-B-only batch oracle. |

Example:

```bash
pixi run python examples/benchmarks/bayesian_linear_head_lowshot_adaptation.py --shots 4,8,16,32 --seed 0
pixi run python examples/benchmarks/bayesian_linear_head_online_drift.py --seed 1
```

## Catalogue maturity (recommendation level)

Both heads are registered in `torchregress.method_catalog` with maturity **Available**: they are maintained, documented, and backed by the synthetic benchmarks above, but they are **not** positioned as decision-grade alternatives to ensembles, `SWAG`, or BNNs for nonlinear epistemic uncertainty. Use them when the modelling assumption—linear Gaussian readout on **fixed** features—is acceptable, and validate on domain metrics before deployment.

The [comparative evidence matrix](../../reports/comparative_evidence_matrix.md) includes a **Strong** (synthetic-scope) row for *low-shot linear adaptation on fixed features (last layer)* pointing at the benchmark scripts above; gaps there call out the lack of real frozen-backbone protocols.

See also: module `torchregress.test_time.bayes`.
