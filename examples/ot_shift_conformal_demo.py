"""
Minimal demo: score CDF gap, OT-style reweighter, and weighted split-conformal sets.

Uses synthetic 1-D nonconformity scores for calibration and a shifted target pool.
"""

from __future__ import annotations

import argparse

import torch

import torchregress as tr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    cal = torch.rand(60)
    tgt = torch.rand(50) * 0.4 + 0.55

    gap = tr.test_time.OptimalTransportCoverageGap().estimate(
        calibration_scores=cal,
        target_score_summary=tgt,
    )
    rw = tr.test_time.OTShiftReweighter(
        entropy_penalty=5e-2,
        n_steps=150,
        learning_rate=0.08,
    ).fit(cal, tgt)
    ad = tr.test_time.WeightedSplitConformalAdapter(alpha=0.1)
    ad.calibrate(cal, rw.weights_)

    cand = torch.rand(8, 5) * 0.8
    sets = ad.predict_from_test_scores(cand)

    print("l2_cdf_gap:", round(gap["l2_cdf_gap"], 6))
    print("threshold:", round(float(ad.threshold_.item()), 6))
    print("mean set size:", float(sets.float().mean().item()))


if __name__ == "__main__":
    main()
