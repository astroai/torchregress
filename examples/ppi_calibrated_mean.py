"""Compare standard PPI vs affine post-hoc calibrated PPI for a population mean.

Synthetic setting: outcomes are Gaussian; model scores are an affine miscalibration of
the outcome (wrong slope and intercept). Calibrated PPI (Chen et al., arXiv:2604.21260,
Section 3.3) fits a linear map on the labeled set before the usual rectified estimator,
which often shortens bootstrap intervals.

Run:
    uv run python examples/ppi_calibrated_mean.py
"""

from __future__ import annotations

import torch

from torchregress.inference import PPIConfig, ppi_calibrated_mean_ci, ppi_mean_ci


def main() -> None:
    torch.manual_seed(7)
    true_mean = 0.0
    n_labeled, n_unlabeled = 120, 4000
    y_l = true_mean + torch.randn(n_labeled)
    # Deliberately miscalibrated monotone scores (informative but mis-scaled)
    pred_l = 0.22 * y_l - 0.9 + 0.12 * torch.randn(n_labeled)
    y_u = true_mean + torch.randn(n_unlabeled)
    pred_u = 0.22 * y_u - 0.9 + 0.12 * torch.randn(n_unlabeled)

    cfg = PPIConfig(alpha=0.1, n_boot=600, seed=11)
    raw = ppi_mean_ci(y_l, pred_l, pred_u, config=cfg)
    cal = ppi_calibrated_mean_ci(y_l, pred_l, pred_u, config=cfg)

    def _summarize(name: str, out: dict[str, object]) -> None:
        lo = float(out["ci_lower"])
        hi = float(out["ci_upper"])
        est = float(out["estimate"])
        print(f"{name:22s}  est={est:+.4f}  90% CI=[{lo:+.4f}, {hi:+.4f}]  width={hi - lo:.4f}")

    print(f"True mean (simulation): {true_mean:.4f}\n")
    _summarize("PPI (raw score)", raw)
    _summarize("Calibrated PPI (affine)", cal)


if __name__ == "__main__":
    main()
