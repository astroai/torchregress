# Normalizing Flows: Design Notes and Recent Alternatives

> ← [Normalizing Flows](nflows.md) | [Noise-Aware Densities](noise_aware_density.md) →

This note records the design decisions behind the flow support in torchregress.
It is a **research/engineering note**, not a benchmark ranking.

## What is already covered

| Family | torchregress entry point | Role |
|:--|:--|:--|
| Conditional NSF (cNSF) | `create_flow_model(..., flow_type="nsf")` | Default scalar/multi-target density; best calibration per parameter in the harness |
| Conditional RealNVP / cINN-style affine coupling | `flow_type="realnvp"` / `"cinn"` | Fast, lean coupling flow; susceptible to rare catastrophic point tails |
| Conditional MAF | `flow_type="maf"` | Unbounded autoregressive density; slower sampling |
| Bernstein polynomial flow (BPF) | `flow_type="bpf"` | Smooth scalar/joint density with a polynomial basis; no spline tail bound |
| Contrastive likelihood-ratio flow | `ContrastiveFlowLoss` | Hypothesis ranking / NRE, **not** calibrated density estimation |
| Bias-free contrastive ratio estimation | `NeuralRatioLoss(criterion="cnre")` | Exact $\log r$ at any number of negatives (Miller et al. 2022) |
| Ensemble of flows | `DensityEnsemble` | Mixture density plus entropy decomposition (aleatoric / epistemic); fitted weights with frequentist error bars on $\log p$ |
| Frequentist ratio uncertainty | `inference.WifiRatioEnsemble` | wifi ensembles: error bars on $\log r(x)$ and on mixture fractions |
| Joint (multi-target) calibration | `sample_hpd_level`, `sample_hpd_coverage`, `model_hpd_level` | Highest-density-region coverage from draws; tests the joint shape |
| EIV / simulator-aware flow | `InputNoiseAugmentationLoss`, `LatentMarginalizationLoss`, `flow_eiv` harness suites | Observation-model-aware alternatives |

The main recommendation for scarce/noisy/imbalanced regression is the
statistical composition — a conditional density (NSF), a measurement-error
model, a sampling-selection model, and (optionally) a representation
regularizer — rather than a larger flow.

## Recent flow-based and flow-map alternatives

| Method | Why it is interesting | Why it is not the default here |
|:--|:--|:--|
| **Conditional Flow Matching** (Tong et al., TMLR 2023; Lipman et al., 2022) | Simulation-free, stable vector-field regression; excellent for high-dimensional conditional generation | Density evaluation requires integrating an ODE / estimating the Jacobian; expensive and less interpretable for calibrated low-dimensional NLL, PIT, and CRPS |
| **Flow Map Matching / consistency models** (Boffi et al., 2024) | One/few-step generation; useful when sampling cost dominates | Learned two-time maps do not by themselves give a normalized conditional likelihood; calibration and model comparison remain harder than exact flow NLL |
| **Diffusion / score-based models** | Flexible multimodal generation, strong high-dimensional sample quality | No exact cheap density; likelihood evaluation and calibration for scalar regression are more complex and slower than NSF/MDN |
| **DRIFT / Hybrid Bernstein Normalizing Flows** (UAI 2024/2025) | Flexible distributional regression; smooth polynomial bases; multivariate marginal-plus-flow construction | Newer and less battle-tested; BPF is supported in zuko and available as `flow_type="bpf"` for a lean first test |
| **Contrastive Normalizing Flows for uncertainty-aware parameter estimation** (Elsharkawy & Kahn, 2025) | Density shaping / classification under systematics and distribution shifts | Different objective from density calibration; useful for SBI systematics, not a replacement for NLL |

## Contrastive and ratio-based flow literature

Survey of contrastive/ratio methods proposed around flows, judged against the goal of the best
multi-target conditional density $p(y \mid x)$:

| Method | Objective | Gives a calibrated $p(y \mid x)$? | Status in torchregress |
|:--|:--|:--|:--|
| CNRE (Miller, Weniger & Forré, 2022) | $K{+}1$-class contrastive classifier with an "independent" class | No: a likelihood ratio $p(x \mid \theta)/p(x)$ | `NeuralRatioLoss(criterion="cnre")` |
| Contrastive Normalizing Flows (Elsharkawy & Kahn, 2025; IML 2025) | Flows trained to separate signal/background densities, then a classifier and Neyman intervals for a signal fraction under systematics | No: densities are deliberately shaped for discrimination | Related work only; a signal-fraction workflow belongs in a research repo |
| *wifi* ensembles (Benevedes & Thaler, PRD 112, 056024, 2025) | $\log r(x) = \sum_i w_i f_i(x)$ over frozen, independently trained (bootstrap) ratio networks plus a constant; $w$ fit on held-out data by a convex symmetrised maximum-likelihood-classifier loss; sandwich covariance on $w$ gives $\sigma[\log r(x)]$; Gong–Samaniego propagation to downstream parameters (e.g. a mixture fraction) | No: a ratio with frequentist error bars | `inference.WifiRatioEnsemble`; the same recipe for density mixtures is `DensityEnsemble.fit_weights` / `log_prob_std` |
| Factorizable Normalizing Flows for profiled SBI (Valsecchi, Donegà & Wallny, 2026) | Flows that model nuisance-parameter deformations of a multivariate distribution of interest, amortised over nuisances, with a Poisson-bootstrap ensemble | Partly: a conditional density in nuisance parameters for unbinned likelihood fits | Not implemented; HEP-analysis scope |
| MCD, Marginal Contrastive Discrimination (Meziani, Ndiaye & Riu, 2022) | $p(y \mid x) = p(y)\, r(x, y)$ with the ratio learned by joint-vs-marginal classification | Yes, if the marginal $p(y)$ and the ratio are both accurate | Not implemented; low-dimensional baseline to compare against flows |

**Takeaway:** contrastive objectives estimate *ratios* or shape densities for discrimination.
For calibrated multi-target conditional densities, exact-likelihood flows (NSF/MAF), ensembles of
them, and joint HPD calibration checks remain the primary tools.

**Practical verdict:** for low-dimensional tabular regression with noisy or
scarce labels, the exact conditional likelihood of NSF/MAF/RealNVP/BPF plus a
noise model is a better default than a flow-matching or diffusion objective.
Use flow matching when the target is high-dimensional and sampling/generation is
the product; use flow-map matching only after profiling sampling cost.

## Robustness recipe

1. Standardize targets and choose the spline bound from the training range
   (`recommended_tail_bound`).
2. Use validation-NLL early stopping; a fixed long budget overfits small data.
3. Combine a density head with a measurement-error model
   (`NoiseAwareDensityLoss`) and, if labels are contaminated, a small
   contamination component.
4. For imbalance, distinguish:
   - **population density estimation** — fit the unweighted likelihood plus a
     sampling-selection model;
   - **tail-balanced prediction** — use LDS/tempered weighting, and report the
     reweighted calibration explicitly.
5. Add representation regularization (`ConR`, `RankSim`, FDS) to the encoder;
   do not expect the density head alone to recover unsupported target regions.
6. For small labels, compare against an MDN and a smoothed binned-CDF head.  In
   one dimension these can beat a more expressive flow.

## References

| # | Reference |
|:-:|:----------|
| 1 | Y. Lipman et al. ["Flow Matching for Generative Modeling."](https://arxiv.org/abs/2210.02747) 2022. |
| 2 | A. Tong et al. ["Improving and generalizing flow-based generative models with minibatch optimal transport."](https://arxiv.org/abs/2302.00482) TMLR 2023. |
| 3 | N. M. Boffi et al. ["Flow map matching with stochastic interpolants."](https://arxiv.org/abs/2406.07507) 2024. |
| 4 | M. Keramati et al. ["ConR: Contrastive Regularizer for Deep Imbalanced Regression."](https://arxiv.org/abs/2309.06651) ICLR 2024. |
| 5 | I. Elsharkawy & Y. Kahn. ["Contrastive Normalizing Flows for Uncertainty-Aware Parameter Estimation."](https://arxiv.org/abs/2505.08709) 2025. |
| 6 | D. Kook et al. "DRIFT: Distributional Regression with Flexible Tails." UAI 2024. |
| 7 | J. Arpogaus et al. "Hybrid Bernstein Normalizing Flows." UAI 2025. |
| 8 | C. Durkan et al. ["Neural Spline Flows."](https://arxiv.org/abs/1906.04032) NeurIPS 2019. |
| 9 | L. Ardizzone et al. ["Guided Image Generation with Conditional Invertible Neural Networks."](https://arxiv.org/abs/1907.02392) 2019. |
| 10 | B. K. Miller, C. Weniger, P. Forré. ["Contrastive Neural Ratio Estimation."](https://arxiv.org/abs/2210.06170) NeurIPS 2022. |
| 11 | S. Benevedes, J. Thaler. ["Frequentist Uncertainties on Neural Density Ratios with wifi Ensembles."](https://arxiv.org/abs/2506.00113) Phys. Rev. D 112, 056024, 2025. |
| 12 | D. Valsecchi, M. Donegà, R. Wallny. ["Profiling systematic uncertainties in Simulation-Based Inference with Factorizable Normalizing Flows."](https://arxiv.org/abs/2602.13184) 2026. |
| 13 | K. Meziani, A. Ndiaye, B. Riu. ["MCD: Marginal Contrastive Discrimination for conditional density estimation."](https://arxiv.org/abs/2206.01592) 2022. |
