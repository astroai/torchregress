Based on the search results, here is the synthesised SOTA table for errors-in-variables (EIV) methods that scale to deep regression as of 2026.

---

## EIV Methods for Deep Regression — SOTA 2026

| # | Method | Core EIV Mechanism | Scalability | Input Noise Type | Key Limitation | Primary Reference |
|---|---|---|---|---|---|---|
| 1 | **Bayesian Deep-EIV + Diffusion Prior** | Scalable variational inference scheme for Bayesian EIV, using a diffusion model as a prior over true inputs; first EIV framework to improve prediction performance in deep nets | ✅ Mini-batch VI | Gaussian, learned | Requires pre-trained diffusion model for input prior | Faller, Martin & Elster, *Mach. Learn.* 2025 |
| 2 | **MC Dropout EIV BNN** | Combines variational inference for Bayesian NNs with uncertain inputs via MC Dropout; produces a decomposition of predictive uncertainty into aleatoric and epistemic components that is more complete under EIV | ✅ Dropout-based, scalable | Gaussian homoscedastic | Dropout approximation; requires known noise magnitude | Martin & Elster, *NPL* 2022 |
| 3 | **LatentNN** | Jointly optimises network parameters and latent true input values by maximising the joint likelihood of observing both inputs and outputs; corrects attenuation bias in the low-SNR regime | ⚠️ Scales to moderate-dim; test-time optimisation per sample | Gaussian, heteroscedastic | Per-sample latent optimisation at inference; cost scales with dataset | Ting, *arXiv* 2512.23138, 2025 |
| 4 | **Neural SIMEX** | Model-agnostic adaptation of the SIMEX simulation-extrapolation approach to deep nets: train on data with progressively amplified input noise, fit a trend over noise-scale λ, extrapolate to λ = −1 (zero-noise) | ✅ Embarrassingly parallel; any architecture | Gaussian (classical), some non-Gaussian extensions | Only approximately corrects nonlinear models; requires known noise variance | Carroll et al., *Measurement Error in Nonlinear Models*, 2006 + NN adaptation |
| 5 | **Regression Calibration for DNNs** | Estimate E[x_true \| x_obs] via an auxiliary model (e.g. MLP/GP), substitute calibrated inputs into the main network; corrects first-order attenuation bias | ✅ Standard backprop | Gaussian, non-differential | First-order only; breaks for highly nonlinear regression; requires replicate or validation data | Carroll et al. (classical) + DNN adaptation |
| 6 | **Reparameterised Stochastic Input Integration** | At each forward pass, sample *x_true ~ p(x_true \| x_obs, σ)* via reparameterisation trick and average gradients; equivalent to Monte Carlo EIV marginalisation | ✅ Mini-batch; any architecture | Gaussian, known per-sample σ | Adds MC variance to gradients; slow convergence at high noise | Kendall & Gal, NeurIPS 2017 (general framework) |
| 7 | **Heteroscedastic Noise-Augmented Training** | During training, add per-sample Gaussian noise scaled to known σ_x; the loss is computed under the noisy-input distribution, implicitly marginalising EIV | ✅ Zero overhead; pure data augmentation | Gaussian, known σ | Does not correct attenuation bias; neural networks suffer the same systematic compression as linear regression unless combined with likelihood correction | Standard DL + Tikhonov regularisation interpretation |
| 8 | **Assumed Density Filtering / Moment Propagation** | Analytically propagate mean and variance of noisy inputs layer-by-layer through the network using moment matching (linearisation or Gaussian quadrature) | ✅ Analytically tractable; no MC sampling | Gaussian | Approximation degrades with deep/highly nonlinear activations; no analytic solution for transformers | Solin et al. 2018; Wu et al. 2019 |
| 9 | **Conditional Normalising Flow Latent Input Model** | Learn p(x_true \| x_obs) with a conditional NF; marginalise the regression likelihood over the learned latent input distribution during training using importance sampling or the reparameterisation trick | ✅ Amortised inference, scalable | Arbitrary (non-Gaussian, heteroscedastic) | Model must effectively differentiate between distributional regimes; expensive to train jointly | Papamakarios et al. 2021 + EIV integration |
| 10 | **VAE Input Denoiser + Regression Head** | Jointly train a VAE encoder p(z \| x_obs) with a regression decoder f(z); the latent z represents cleaned input; ELBO includes both reconstruction and regression terms | ✅ Standard backprop; end-to-end | Arbitrary, learned from data | Posterior collapse risk; latent space may not align with regression-relevant features | Kingma & Welling (VAE) + regression objective |
| 11 | **Score-Based Input Prior + Posterior Input Inference** | Use a pre-trained score model (DDPM-style) as prior p(x_true); at inference, use Langevin/DDIM sampling to find posterior x_true \| x_obs, then regress from denoised input | ⚠️ Expensive at inference (iterative sampling) | Arbitrary; powerful for images | Inference cost is proportional to diffusion steps × batch size | Song et al., ICLR 2021; Chung et al. 2022 |
| 12 | **Neural Error Veracity Scoring (Conformal EIV Detection)** | Model-agnostic approach using veracity scores derived from any regressor to identify and down-weight or correct erroneous input values; accounts for aleatoric and epistemic uncertainty jointly | ✅ Any regressor; conformal guarantees | Any (model-agnostic) | Detection-then-correction pipeline; does not marginalise over EIV distribution | Northcutt et al. / Kuan & Mueller 2023 |

---

### Summary Notes

**Best for production/scalability:** Methods 2, 6, 7, 8 — minimal overhead, standard backprop.

**Best bias correction:** Methods 1, 3, 9 — explicitly model the EIV likelihood; method 3 is directly motivated by the attenuation bias problem that directly affects surveys like APOGEE, DESI, and Gaia XP, making it highly relevant for spectroscopic regression.

**Best for unknown/non-Gaussian noise:** Methods 9, 10, 11 — data-driven noise models, no Gaussianity assumption.

**Classical baseline that still works:** Method 4 (Neural SIMEX) and 5 (Regression Calibration) — well-understood, easy to audit.