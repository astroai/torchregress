"""
Toy benchmark: score CDF gap, OT-style reweights, weighted split conformal, diagnostics.

Prints marginal diagnostics from ``OptimalTransportCoverageGap``, ``ScoreCDFReweighter``,
``WeightedSplitConformalAdapter.coverage_diagnostics``, and mean prediction-set size on
random candidate score matrices (classification-style).
"""

from __future__ import annotations

import argparse

import torch

import torchregress as tr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    cal = torch.rand(80)
    tgt = torch.rand(60) * 0.35 + 0.55
    gap = tr.test_time.OptimalTransportCoverageGap().estimate(
        calibration_scores=cal,
        target_score_summary=tgt,
    )
    rw = tr.test_time.ScoreCDFReweighter(
        entropy_penalty=5e-2,
        n_steps=120,
        learning_rate=0.08,
    ).fit(cal, tgt)
    ad = tr.test_time.WeightedSplitConformalAdapter(alpha=0.1)
    ad.calibrate(cal, rw.weights_)
    diag = ad.coverage_diagnostics(cal, rw.weights_)
    cand = torch.rand(6, 4) * 0.9
    sets = ad.predict_from_test_scores(cand)
    pb = tr.test_time.weighted_split_classification_predictive_batch(
        ad,
        cand,
        gap_diagnostics=gap,
        calibration_ess_inv_square=diag["calibration_ess_inv_square"],
    )

    print("l2_cdf_gap:", round(gap["l2_cdf_gap"], 6))
    print("weighted_empirical_coverage:", round(diag["weighted_empirical_coverage"], 4))
    print("nominal_coverage:", round(diag["nominal_coverage"], 4))
    print("mean set size:", float(sets.float().mean().item()))
    print("predictive_batch mean[0]:", float(pb.mean[0, 0].item()))


if __name__ == "__main__":
    main()
