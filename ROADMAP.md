# ROADMAP.md

## Recent fixes (June 2026)

### EIV: Removed `.detach()` on propagated variance

**Files:** `src/torchregress/losses/eiv.py`

Three `.detach()` calls were removed from variance propagation computations in
`FunctionalEIVLoss` (analytical + MC paths) and `StructuralEIVLoss`.  See
lines ~794, ~931, ~1066.

**Why:** The `.detach()` blocked gradient flow through the model's Jacobian,
preventing the model from learning to reduce attenuation bias.  With gradients
flowing through the variance term, the model can now adjust its sensitivity to
noisy inputs — the `log(var)` NLL term naturally balances Jacobian shrinkage
against residual accuracy.

**Before:** `FunctionalEIV` achieved identical `RMSE_clean` to `MSE` on
linear EIV datasets — the loss had no mechanism to correct attenuation.

**After:** Gradients flow through Jacobian → variance → loss, enabling
attenuation-bias correction.  Verify with the `eiv` suite in
`torchregress-harness`.

**Risk:** Fixed: default `eps` raised from `1e-8` to `1e-3` (June 2026).  The
larger jitter prevents gradient explosions via `1/var`.  All 19 EIV methods
pass benchmark re-run without NaN/crashes.

### CORAL architecture implemented

**Files:** `src/torchregress/utils/ordinal.py`, `losses/ordinal.py`, `utils/__init__.py`

Added `CORALHead` — shared-weight ordinal output layer with monotonic bias
constraints (`b₁ ≥ b₂ ≥ ... ≥ b_{K-1}`) via `-cumsum(softplus(δ_k))`.
`CORALLoss` docstring updated to document that CORAL's distinctiveness comes
from architecture, not loss formula.  Both are verified by the `ordinal` suite
in `torchregress-harness`.

---

## Known issues / to-do

### High priority

1. **SIMEX is unreliable with `n_simulations=1`** ✅ — Fixed: increased to
   `n_simulations=5` in `torchregress-harness/suites/tabular/eiv.py` (June 2026).
   **Benchmark re-run:** SIMEX with n_simulations=5 is still not competitive.
   RMSE_clean at σx=1.0: 1.112 vs MSE 0.512 (2.2× worse). At σx=2.0: 0.977 vs
   MSE 0.813 (20% worse). On nonlinear data: 0.864 vs MSE 0.828 (4% worse).
   The fix improves statistical stability of the extrapolation but does not
   make SIMEX competitive for linear EIV problems.

2. **`InputNoiseMC` worsens attenuation bias on linear data**  — The
   β²σ² penalty in MC marginalization amplifies rather than corrects
   attenuation.  Avoid plain `InputNoiseMC` on linear problems; prefer
   `RC+FunctionalEIV` or `RC+GaussianNLL`.

3. **`RC+FunctionalEIV` is the current best composability method**  —
   Regression Calibration pre-processing + FunctionalEIV provides ~54%
   RMSE_clean improvement over MSE at σx=2.0 (60-epoch benchmark, June 2026).
   RC de-biases inputs first, then FunctionalEIV provides an additional ~19%
   improvement over RC alone at σx=2.0.

4. **GaussianNLL+InputNoiseMC produces well-calibrated distributions** —
   ECE < 0.003 at σx=2.0.  Recommended for probabilistic EIV prediction
   with calibrated uncertainty.

### Medium priority

5. **`eps=1e-8` in BaseEIVLoss may be too small** ✅ — Fixed: raised default
   `eps` from `1e-8` to `1e-3` in all EIV loss classes (`BaseEIVLoss`,
   `FunctionalEIVLoss`, `StructuralEIVLoss`, `OrthogonalDistanceRegressionLoss`,
   `EnsembleEIVLoss`) to prevent gradient explosions via `1/var` (June 2026).
   **Benchmark re-run:** All 19 EIV methods ran without NaN/crashes across 5
   datasets (σx ∈ {0.2, 0.5, 1.0, 2.0}, linear+nonlinear). No numerical
   instability observed. FunctionalEIV results nearly identical to pre-fix
   (e.g., RMSE_clean 0.529 vs expected ~0.53 at σx=1.0), confirming eps=1e-3
   is a safe default.

6. **Conformal prediction for EIV works out-of-the-box** ✅ — Investigated (June 2026).
   `SplitCP` methods achieve near-nominal coverage (~88% at α=0.1 target 90%)
   on nonlinear EIV data at σx ∈ {0.5, 1.0, 2.0}. **No differential undercoverage
   for SplitCP+InputNoiseMC** — it produces slightly narrower intervals with equal
   or better coverage versus SplitCP+FunctionalEIV (0.889 vs 0.874 at σx=2.0). The
   slight ~2pp gap below nominal 90% is consistent across both methods and
   attributable to finite-sample calibration effects (~375 cal points), not MC
   variance inflation. See diagnostic at
   `torchregress-harness/tools/diagnose_conformal_eiv_coverage.py`.

   **SplitCP+MSE_WarmupMC added (June 2026):** On linear data, warmupMC conformal
   achieves best coverage calibration at σx=2.0 (0.905 vs 0.882/0.877) and best
   RMSE_clean, but widest intervals (3.56 vs 3.00/3.08).  FunctionalEIV gives
   tighter intervals with adequate coverage.

### Low priority

7. **`TLS_Init+FunctionalEIV` initialization path** ✅ — Investigated (June 2026).
   **Finding: TLS β is NOT preserved.** The linear model initialised to TLS β
   then fine-tuned with FunctionalEIVLoss drifts completely to the OLS/MSE
   solution within ~50 epochs. The log(var) NLL term exerts persistent downward
   pressure on β that the residual fit term cannot counteract for linear models.
   FunctionalEIVLoss from scratch also converges to the same attenuated solution.
   **Recommendation:** Do not use TLS_Init+FunctionalEIV. Prefer RC preprocessing
   (RegressionCalibration) before any downstream training. See diagnostic at
   `torchregress-harness/tools/diagnose_tls_eiv_drift.py`.

8. **RC (RegressionCalibration) overfits on linear data at high noise**  —
   Investigated (June 2026).  At σx=2.0, RC shrinks inputs to ~4% of original
   variance (reliability ≈ 0.19).  The resulting low-variance input space
   causes the downstream MLP to overfit: train loss keeps dropping but
   RMSE_clean rises +77% from epoch 30→200 (0.477→0.845, 5-run mean).
   The best epoch is ~29 (mean RMSE_clean 0.408 across 5 seeds).  The
   60-epoch benchmark result (0.452) is already past the optimum;
   performance at epoch ~30 varies by seed (mean 0.477, range 0.356–0.658).
   **Recommendation:** Early-stop RC-trained
   models at ~30 epochs at high noise (σx≥1.0), or use RC+FunctionalEIV which
   is less prone to overfitting (RMSE_clean 0.367 at σx=2.0, 60 epochs, vs
   RC's 0.452).  See diagnostic at
   `torchregress-harness/tools/diagnose_rc_overfitting.py`.

---

## Benchmark re-run summary (June 2026)

Full EIV suite re-run at **60 epochs** on 5 datasets after eps (1e-8→1e-3) and
SIMEX (n_simulations 1→5) fixes.  Apples-to-apples with the original ROADMAP
benchmarks.  JSON artifact: `torchregress-harness/results/latest/eiv_benchmark_60epoch_2026-06-18.json`.

Key findings by RMSE_clean (lower=better):

### Linear EIV data

| Method | σx=0.2 | σx=1.0 | σx=2.0 |
|--------|--------|--------|--------|
| MSE (baseline) | 0.141 | 0.518 | 0.795 |
| FunctionalEIV | 0.143 | 0.519 | 0.810 |
| FunctionalEIV_MC | 0.142 | 0.449 | 0.804 |
| InputNoiseMC | 0.154 | 0.693 | 0.937 |
| **RC** | **0.132** | **0.266** | **0.452** |
| **RC+FunctionalEIV** | **0.132** | **0.263** | **0.367** |
| MSE_WarmupMC | 0.135 | 0.458 | 0.789 |
| TLS_Init+FunctionalEIV | 0.112 | 0.526 | 0.850 |
| SIMEX | 0.616 | 1.067 | 0.950 |

**Takeaways:**
- RC+FunctionalEIV dominates: **~54% improvement over MSE at σx=2.0**
  (was ~41% in original benchmarks; the `.detach()` fix unlocked these gains)
- RC alone overfits at high noise with 60 epochs (0.452 at σx=2.0 vs best
  ~0.408 at epoch ~29). RC shrinks inputs to ~4% of original variance,
  creating a low-signal regime where the MLP memorises noise.
  RC+FunctionalEIV is more robust (0.367 vs RC's 0.452).
  See `tools/diagnose_rc_overfitting.py`.
- Plain FunctionalEIV is slightly *worse* than MSE on linear data — the Jacobian
  log(var) term still attenuates β despite the `.detach()` fix
- FunctionalEIV_MC outperforms analytical at σx=1.0 (0.449 vs 0.519) but
  loses its advantage at σx=2.0 with 60 epochs (0.804 vs 0.810); at 30
  epochs it was notably better (0.732)
- StructuralEIV gives identical numbers to FunctionalEIV (σxy=0 makes them
  algebraically equivalent)
- TLS_Init+FunctionalEIV only helps at low noise (σx=0.2); hurts at σx≥1.0
- SIMEX is not competitive even at n_simulations=5
- InputNoiseMC worsens attenuation at high noise (confirms issue 2)
- MSE_WarmupMC (20-epoch MSE → MC) modestly outperforms pure MC at σx=0.2
  (0.135 vs 0.142) and σx=2.0 (0.789 vs 0.804), and is close at σx=1.0
  (0.458 vs 0.449).  Warmup helps where MC is weakest but doesn't change
  the overall picture — RC+FunctionalEIV still dominates.  Added to harness
  as `eiv/MSE_WarmupMC` (June 2026).

  **Warmup sweep on linear data (June 2026, 60-epoch diagnostic, 3 seeds):**
  At σx=1.0 and σx=2.0, warmup degrades MC monotonically — more MSE
  pre-training = worse final RMSE_clean.

  | Warmup epoch | σx=1.0 RMSE_clean | σx=2.0 RMSE_clean |
  |-------------|-------------------|--------------------|
  | MSE baseline | 0.729 | 1.201 |
  | 0 (pure MC) | **0.561** | **0.945** |
  | 5 | 0.588 | 1.075 |
  | 10 | 0.590 | 1.094 |
  | 20 | 0.595 | 1.108 |
  | 50 | 0.617 | 1.086 |

  **Finding:** Warmup=0 (pure MC, no MSE phase) is best at both noise levels.
  The MSE warmup's gradient direction pulls the model away from the MC
  optimum — on linear data, even a short warmup degrades final performance.
  This contrasts with the harness benchmark where MSE_WarmupMC (20ep)
  showed modest improvement at σx=2.0 — the harness benefits from the
  `min(warmup_epochs, epochs // 3)` guard and different training dynamics.
  **Recommendation:** Use `warmup_epochs=0` (pure MC) for linear EIV with
  `WarmupMCTrainer`.  On nonlinear data, warmup is neutral (doesn't help
  or hurt).  See `tools/diagnose_mc_warmup.py` (now supports `--linear`).

Conformal methods were also benchmarked on linear data (previously only tested
on nonlinear in item 6).  Results at 60 epochs, June 2026:

| Method | σx=0.2 (Cov/Wid/RMSEₖ) | σx=1.0 (Cov/Wid/RMSEₖ) | σx=2.0 (Cov/Wid/RMSEₖ) |
|--------|--------------------------|--------------------------|--------------------------|
| SplitCP+FunctionalEIV | 0.905 / 0.843 / 0.145 | 0.898 / 2.437 / 0.523 | 0.882 / 2.999 / 0.846 |
| SplitCP+InputNoiseMC | 0.900 / 0.855 / 0.154 | 0.885 / 2.506 / 0.703 | 0.877 / 3.083 / 0.947 |
| SplitCP+MSE_WarmupMC | 0.917 / 0.853 / 0.141 | 0.897 / 2.529 / 0.445 | 0.905 / 3.563 / 0.771 |

**Takeaways:**
- SplitCP+MSE_WarmupMC achieves the closest-to-nominal coverage at σx=2.0
  (0.905 vs 0.882/0.877) and the best RMSE_clean across all noise levels,
  but at the cost of wider intervals at σx=1.0–2.0 (2.529 vs 2.437 at σx=1.0,
  3.563 vs 2.999 at σx=2.0).
- SplitCP+FunctionalEIV produces the tightest intervals and maintains adequate
  coverage (~88–90%).  Best efficiency (Width/Coverage ratio) at σx≥1.0.
- SplitCP+InputNoiseMC underperforms both at σx≥1.0 (wider intervals, worse
  RMSE_clean) — MC noise marginalization inflates calibration residuals.
- Choice depends on priority: coverage calibration → WarmupMC; interval
  efficiency → FunctionalEIV.  Added to harness as
  `eiv/SplitCP+MSE_WarmupMC` (June 2026).

### Nonlinear EIV data

| Method | σx=0.5 | σx=1.0 | σx=2.0 |
|--------|--------|--------|--------|
| MSE (baseline) | 0.837 | 0.835 | 0.839 |
| FunctionalEIV | 0.832 | 0.831 | 0.834 |
| FunctionalEIV_MC | 1.597 | 0.978 | 0.865 |
| InputNoiseMC | 0.829 | 0.830 | 0.839 |
| RC | 0.849 | 0.854 | 0.888 |
| RC+FunctionalEIV | 0.846 | 0.848 | 0.867 |
| MSE_WarmupMC | 1.405 | — | — |
| TLS_Init+FunctionalEIV | 0.865 | 0.824 | 0.835 |
| SIMEX | 0.881 | 0.886 | 0.900 |

**Takeaways:**
- No method significantly beats MSE on nonlinear data (all within ~2%)
- RC makes things *worse* at all noise levels — linear reliability model is
  misspecified for nonlinear ground truth; degradation worsens with noise
  (0.849→0.854→0.888)
- FunctionalEIV_MC failure is **worst at moderate noise** (σx=0.5: +91% vs
  MSE, σx=1.0: +17%, σx=2.0: +3%).  Hypothesis: at higher noise, the
  noise itself washes out the pathological MC variance landscape, preventing
  the model from latching onto the bad basin; at moderate noise, there's
  enough clean signal to drive the model into explosive predictions.  Root
  cause diagnosed at σx=0.5 in
  `torchregress-harness/tools/diagnose_mc_eiv_nonlinear.py` — multi-noise
  diagnosis not yet run.
- TLS_Init+FunctionalEIV beats MSE at σx=1.0 (0.824 vs 0.835) — the TLS
  initialization provides a useful starting point at this noise level, but
  doesn't help at σx=0.5 (0.865) or σx=2.0 (0.835)
- FunctionalEIV_MC catastrophically fails at σx=0.5 (1.597 vs 0.837 — **91%
  degradation**).  Longer training (60 vs 30 epochs) makes it *worse* (1.597 vs
  1.406) — the pathological loss landscape deepens with more epochs.  **Root
  cause (June 2026):** Noisy MC variance estimates from randomly-initialised
  nonlinear networks create a bad basin within 10 epochs.  The model learns
  explosive wrong predictions (pred_range ~10 vs ~2 for MSE), not collapsed
  ones.  Increasing n_samples (200) helps partially (1.271) but doesn't fix it.
  GELU doesn't help.

  **Warmup strategy tested — does NOT work (June 2026).**  MSE pre-training
  for 5–50 epochs gets the model to RMSE_clean ≈ 0.82–0.84 (close to MSE
  baseline), but switching to MC immediately destroys it (+60–70% at σx=0.5,
  +25–30% at σx=1.0).  The MC loss landscape itself is the problem — it
  actively pulls a well-trained model into the bad basin regardless of
  starting point.  This is NOT an initialization issue; the MC variance
  estimator is fundamentally broken for nonlinear models.  See diagnostics at
  `tools/diagnose_mc_eiv_nonlinear.py` and `tools/diagnose_mc_warmup.py`.

  **Root cause mechanism diagnosed (June 2026): gradient noise overwhelms
  signal.**  At the post-MSE optimum, the MC gradient is nearly orthogonal
  to the clean truth gradient (cos=0.166) and 35.7× larger in norm (16.5 vs
  0.46).  The MC empirical variance estimator injects enormous gradient
  variance that causes a random walk away from the good solution — not a
  directed push but a cumulative drift over many steps.  A single MC step
  has negligible effect (ΔRMSE=+0.0002); degradation accumulates over epochs.
  By contrast, analytical FEIV gradient is 4.5× smaller (norm 4.5) and
  moderately aligned with truth (cos=0.452 vs clean MSE, cos=0.834 vs noisy
  MSE).  See `tools/diagnose_mc_gradient_direction.py`.

  **Hybrid Jacobian-variance + MC-mean approach tested — WORKS on nonlinear,
  HURTS on linear (June 2026).**  Using analytical Jacobian-based variance
  (stable) with MC perturbation-based mean (bias-corrected) resolves the MC
  path's instability on nonlinear data:
  - σx=0.5 nonlinear: hybrid 0.960 vs pure MC 1.410 (−32%), still +12% vs MSE
  - σx=1.0 nonlinear: hybrid 0.840 vs pure MC 1.044 (−20%), **beats MSE**
    (0.855) and analytical FunctionalEIV (0.843)
  The hybrid shows an "overshoot then recover" trajectory — early MC phase
  degrades mildly, then Jacobian regularization pulls the model back.  This
  confirms the problem is the empirical MC variance estimator, not the MC
  mean.

  **However, the hybrid does NOT preserve MC's advantage on linear data**
  (tested June 2026, 100-epoch diagnostic with separate data generator;
  absolute values differ from 60-epoch harness benchmarks, but relative
  comparisons are valid):
  - σx=0.2 linear: hybrid 0.149 ≈ pure MC 0.149 (neutral — both beat MSE 0.159)
  - σx=1.0 linear: hybrid 0.754 vs pure MC 0.550 (**+37% — destroys MC advantage**)
  - σx=2.0 linear: hybrid 1.161 vs pure MC 1.002 (**+16% — degrades MC**)
  On linear data, the Jacobian is nearly constant (≈ β), so the Jacobian
  variance term provides almost no useful gradient signal — it adds noise
  without the stabilizing regularization that helps on nonlinear data.
  The hybrid is NOT a universal MC drop-in: it rescues MC on nonlinear but
  degrades it on linear.  **Recommendation:** Use pure MC (or MSE_WarmupMC)
  on linear data; use hybrid on nonlinear.  See diagnostic at
  `tools/diagnose_hybrid_mc.py` (now supports `--linear` flag).

See `torchregress-harness/suites/tabular/eiv.py` for the comprehensive
EIV benchmark suite (20 methods across 6 datasets at 3 noise levels).
Key composability methods to watch:

| Method | Status | Note |
|--------|--------|------|
| `RC+FunctionalEIV` | 🏆 Best | ~54% RMSE_clean improvement over MSE at σx=2.0 (60-epoch, June 2026) |
| `GaussianNLL+InputNoiseMC` | ✅ | Well-calibrated distributions |
| `RC+GaussianNLL` | ✅ | Pipeline approach |
| `TLS_Init+FunctionalEIV` | ❌ | TLS β drifts to MSE — not recommended (June 2026) |
| `SplitCP+FunctionalEIV` | ✅ | Conformal on EIV |
| `SplitCP+InputNoiseMC` | ✅ | No differential undercoverage found (June 2026) |
| `SplitCP+MSE_WarmupMC` | ✅ | Best coverage cal at high noise, widest intervals; FunctionalEIV tighter (June 2026) |
| `SIMEX` | ⚠️ | Fixed n_simulations=5, still not competitive (June 2026) |
| `RC` | ⚠️ | Overfits on linear data at σx≥1.0 beyond ~30 epochs; prefer RC+FunctionalEIV (June 2026) |
| `FunctionalEIV_MC` | ❌ | MC broken for nonlinear at σx≤1.0; warmup doesn't rescue. Hybrid J+MC rescues nonlinear (−32% at σx=0.5) but degrades linear (+37% at σx=1.0) — use-case dependent. Both available in torchregress: hybrid via ``FunctionalEIVLoss(mode="hybrid")``, warmup via ``WarmupMCTrainer`` (June 2026) |
| `MSE_WarmupMC` | ✅ | MSE pre-training (20ep) → MC modestly improves MC at low/high noise, doesn't hurt at moderate noise. Now available in torchregress as `WarmupMCTrainer` in ``torchregress.algorithms`` (June 2026) |

### R2C conformal prediction

See `torchregress-harness/suites/tabular/r2c_conformal.py` for the
regression-as-classification conformal benchmark suite.  Compares:

- `R2C+Softmax` (torchregress R2CConformal + binned CE)
- `R2C+CumulativeLink` (torchregress R2CConformal + ordinal cumulative logits)
- `R2C+CORAL` (torchregress R2CConformal + CORAL architecture)
- `torchcp_R2C` (torchcp R2CCP implementation)
- `Split+MLP` and `CQR+MLP` baselines

**Status:** Working.  CORAL achieves near-nominal coverage (0.917 at α=0.1).
R2C methods tend to over-cover on small datasets (inherent to APS scoring with
many bins).  torchcp_R2C requires verifying output format.
