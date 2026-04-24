"""Compose semisupervised *inference* (PPI) with *predictive* split conformal intervals.

This script answers a common design question: PPI targets uncertainty for summaries
such as the population mean E[Y]; conformal prediction targets finite-sample coverage
for individual outcomes Y at new draws (marginal or weighted split CP, depending on
setup). They solve different problems; used together with a **labeled split** you can
pursue sharper predictions *and* defensible inference without reusing the same points
for every nuisance fit.

See ``docs/methods/inference.md`` → *Inference vs prediction*.

Run:
    uv run python examples/ppi_mean_plus_split_conformal.py
"""

from __future__ import annotations

import torch

from torchregress.inference import PPIConfig, ppi_calibrated_mean_ci, ppi_mean_ci
from torchregress.losses import SplitConformal


def _affine_fit(m: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (slope, intercept) for OLS y ~ slope * m + intercept (1-D)."""
    mf = m.reshape(-1).float()
    yf = y.reshape(-1).float()
    m_c = mf - mf.mean()
    denom = (m_c * m_c).sum()
    if float(denom.item()) < 1e-20:
        z = torch.zeros((), dtype=mf.dtype, device=mf.device)
        return z, yf.mean()
    slope = ((m_c * (yf - yf.mean())).sum() / denom).detach()
    intercept = (yf.mean() - mf.mean() * slope).detach()
    return slope, intercept


def _affine_apply(m: torch.Tensor, slope: torch.Tensor, intercept: torch.Tensor) -> torch.Tensor:
    return (intercept + slope * m.reshape(-1).float()).reshape(m.shape)


def main() -> None:
    torch.manual_seed(0)
    n_cal, n_ppi, n_u = 140, 60, 4_000
    n_l = n_cal + n_ppi

    y = torch.randn(n_l)
    pred = 0.25 * y - 0.6 + 0.12 * torch.randn(n_l)
    y_u = torch.randn(n_u)
    pred_u = 0.25 * y_u - 0.6 + 0.12 * torch.randn(n_u)

    y_cal, pred_cal_fold = y[:n_cal], pred[:n_cal]
    y_ppi, pred_ppi_fold = y[n_cal:], pred[n_cal:]

    slope, intercept = _affine_fit(pred_cal_fold, y_cal)
    pred_cal = _affine_apply(pred_cal_fold, slope, intercept)
    pred_ppi = _affine_apply(pred_ppi_fold, slope, intercept)
    pred_unlab = _affine_apply(pred_u, slope, intercept)

    alpha = 0.1
    cfg = PPIConfig(alpha=alpha, n_boot=500, seed=1)

    # --- 1) Population mean: PPI on held-out labeled fold + unlabeled (honest affine) ---
    ppi_split = ppi_mean_ci(y_ppi, pred_ppi, pred_unlab, config=cfg)

    # Same pipeline but affine refit inside bootstrap (uses all labeled; often tighter) ---
    ppi_all_cal = ppi_calibrated_mean_ci(
        torch.cat([y_cal, y_ppi]),
        torch.cat([pred_cal_fold, pred_ppi_fold]),
        pred_u,
        config=cfg,
    )

    # --- 2) Individual Y: split conformal on calibration fold, evaluate on PPI fold ---
    cp = SplitConformal(alpha=alpha)
    cp.calibrate(pred_cal, y_cal)
    lo, hi = cp.predict_interval(pred_ppi)
    lo1, hi1 = lo.reshape(-1), hi.reshape(-1)
    yv = y_ppi.reshape(-1)
    covered = ((yv >= lo1) & (yv <= hi1)).float().mean()
    width = (hi - lo).mean()

    print("=== Different estimands ===")
    print("PPI (split affine): CI for E[Y] using labeled fold 2 + unlabeled")
    print(
        f"  estimate={ppi_split['estimate']:+.4f}  "
        f"{100 * (1 - alpha):.0f}% CI=[{ppi_split['ci_lower']:+.4f}, {ppi_split['ci_upper']:+.4f}]"
    )
    print("ppi_calibrated_mean_ci (all labeled): same estimand, refits affine each bootstrap")
    print(
        f"  estimate={ppi_all_cal['estimate']:+.4f}  "
        f"{100 * (1 - alpha):.0f}% CI=[{ppi_all_cal['ci_lower']:+.4f}, {ppi_all_cal['ci_upper']:+.4f}]"
    )
    print()
    print("=== Predictive intervals (not a CI for E[Y]) ===")
    print(
        f"SplitConformal on fold 1, evaluated on fold 2: mean width={width.item():.4f}  "
        f"empirical coverage≈{covered.item():.2f} (target {1 - alpha:.2f} on this toy draw)"
    )


if __name__ == "__main__":
    main()
