"""
``PredictiveBatch`` helpers for classification-style weighted split conformal outputs.

Pairs with :class:`~torchregress.test_time.ot_conformal.WeightedSplitConformalAdapter` after
``calibrate``; packs prediction sets and optional shift diagnostics into ``extra``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from torchregress.prediction import PredictiveBatch

from .ot_conformal import WeightedSplitConformalAdapter


@torch.no_grad()
def weighted_split_classification_predictive_batch(
    adapter: WeightedSplitConformalAdapter,
    candidate_scores: torch.Tensor,
    *,
    gap_diagnostics: Optional[dict[str, Any]] = None,
    calibration_ess_inv_square: Optional[float] = None,
) -> PredictiveBatch:
    """
    Build a :class:`~torchregress.prediction.PredictiveBatch` from per-class nonconformity scores.

    Args:
        adapter: Fitted adapter with ``threshold_`` set.
        candidate_scores: ``[batch, n_classes]`` scores (smaller is more conformant).
        gap_diagnostics: Optional dict from :class:`~torchregress.test_time.ot_conformal.OptimalTransportCoverageGap` (or similar) for ``extra``.
        calibration_ess_inv_square: Optional scalar ``1 / sum w_i^2`` from the reweighter for ``extra``.

    Returns:
        ``PredictiveBatch`` with ``point`` / ``mean`` equal to **set size** per row (float),
        ``std`` zeros, and ``extra`` containing ``label_inclusion_mask``, ``alpha``, ``threshold``.
    """
    if candidate_scores.dim() != 2:
        raise ValueError("candidate_scores must be 2-D [batch, n_classes]")
    mask = adapter.predict_from_test_scores(candidate_scores)
    set_sizes = mask.float().sum(dim=-1, keepdim=True)
    thr = adapter.threshold_
    thr_val = float(thr.item()) if thr is not None and torch.is_tensor(thr) else None
    extra: dict[str, Any] = {
        "label_inclusion_mask": mask,
        "alpha": float(adapter.alpha),
        "threshold": thr_val,
    }
    if gap_diagnostics is not None:
        extra["shift_gap_diagnostics"] = gap_diagnostics
    if calibration_ess_inv_square is not None:
        extra["calibration_ess_inv_square"] = float(calibration_ess_inv_square)
    z = torch.zeros_like(set_sizes)
    return PredictiveBatch(
        point=set_sizes,
        mean=set_sizes,
        std=z,
        extra=extra,
    )
