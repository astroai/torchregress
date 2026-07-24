"""Shift calibration methods for test-time adaptation and prior correction."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
from torch import Tensor


def _ensure_tensor(x: Union[Tensor, np.ndarray, list[float]]) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return torch.as_tensor(x)


class RepresentationShiftInflator:
    def __init__(
        self,
        *,
        base_temperature: float = 1.0,
        slope: float = 1.0,
        max_temperature: float = 5.0,
        source_sample_size: int | None = None,
        random_state: int | None = 0,
        clip_quantile: float | None = None,
        eps: float = 1.0e-6,
    ) -> None:
        self.base_temperature = float(base_temperature)
        self.slope = float(slope)
        self.max_temperature = float(max_temperature)
        self.source_sample_size = source_sample_size
        self.random_state = random_state
        self.clip_quantile = clip_quantile
        self.eps = float(eps)
        self.source_mean_: Tensor | np.ndarray | None = None
        self.source_var_: Tensor | np.ndarray | None = None
        self.reference_scale_: float | None = None

    def fit(self, source_representations: Tensor | np.ndarray) -> "RepresentationShiftInflator":
        reps = _ensure_tensor(source_representations).double()
        if reps.ndim == 1:
            reps = reps.unsqueeze(0)
        if self.source_sample_size is not None and self.source_sample_size < reps.shape[0]:
            idx = torch.randperm(reps.shape[0], device=reps.device)[: self.source_sample_size]
            reps = reps[idx]
        if self.clip_quantile is not None:
            lo = reps.quantile(self.clip_quantile, dim=0)
            hi = reps.quantile(1.0 - self.clip_quantile, dim=0)
            reps = reps.clamp(lo, hi)
        self.source_mean_ = reps.mean(dim=0).numpy()
        self.source_var_ = reps.var(dim=0, unbiased=False).clamp(self.eps, None).numpy()
        d2 = self._squared_mahalanobis(reps)
        self.reference_scale_ = float(d2.clamp(0.0, None).sqrt().median().item())
        return self

    def _squared_mahalanobis(self, reps: Tensor) -> Tensor:
        if self.source_mean_ is None or self.source_var_ is None:
            raise RuntimeError("call fit() before computing shift scores")
        mean = _ensure_tensor(self.source_mean_)
        var = _ensure_tensor(self.source_var_)
        centered = reps - mean.unsqueeze(0)
        return (centered**2 / var.unsqueeze(0)).sum(dim=1)

    def shift_scores(self, target_representations: Tensor | np.ndarray) -> Tensor | np.ndarray:
        is_numpy = isinstance(target_representations, np.ndarray)
        result = (
            self._squared_mahalanobis(_ensure_tensor(target_representations).double())
            .clamp(0.0, None)
            .sqrt()
        )
        if is_numpy:
            return result.numpy()
        return result

    def temperatures(self, target_representations: Tensor | np.ndarray) -> Tensor | np.ndarray:
        is_numpy = isinstance(target_representations, np.ndarray)
        scores = (
            self._squared_mahalanobis(_ensure_tensor(target_representations).double())
            .clamp(0.0, None)
            .sqrt()
        )
        ref = max(float(self.reference_scale_ or 1.0), self.eps)
        temps = self.base_temperature * (1.0 + self.slope * scores / ref)
        result = temps.clamp(self.base_temperature, self.max_temperature)
        if is_numpy:
            return result.numpy()
        return result

    def calibrate_probabilities(
        self, probabilities: Tensor | np.ndarray, target_representations: Tensor | np.ndarray
    ) -> Tensor | np.ndarray:
        is_numpy = isinstance(probabilities, np.ndarray)
        reps = _ensure_tensor(target_representations)
        temps = _ensure_tensor(self.temperatures(reps)).unsqueeze(-1)
        logits = _ensure_tensor(probabilities).double().clamp(self.eps, None).log()
        scaled = logits / temps
        scaled = scaled - scaled.max(dim=1, keepdim=True).values
        out = scaled.exp()
        result = out / out.sum(dim=1, keepdim=True).clamp(self.eps, None)
        if is_numpy:
            return result.numpy()
        return result

    def calibrate_std(
        self, std: Tensor | np.ndarray, target_representations: Tensor | np.ndarray
    ) -> Tensor | np.ndarray:
        is_numpy = isinstance(std, np.ndarray)
        reps = _ensure_tensor(target_representations)
        temps = _ensure_tensor(self.temperatures(reps))
        result = (_ensure_tensor(std).double() * temps).clamp(self.eps, None)
        if is_numpy:
            return result.numpy()
        return result


class BinnedLabelShiftEstimator:
    def __init__(
        self,
        *,
        n_bins: int = 10,
        binning_strategy: str = "adaptive",
        method: str = "em",
        max_iter: int = 100,
        tol: float = 1e-6,
        eps: float = 1e-8,
    ) -> None:
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        if binning_strategy not in {"adaptive", "uniform"}:
            raise ValueError("binning_strategy must be 'adaptive' or 'uniform'")
        if method not in {"bbse", "em"}:
            raise ValueError("method must be 'bbse' or 'em'")

        self.n_bins = n_bins
        self.binning_strategy = binning_strategy
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps

        self.bin_edges_: Tensor | np.ndarray | None = None
        self.source_prior_: Tensor | np.ndarray | None = None
        self.target_prior_: Tensor | np.ndarray | None = None
        self.confusion_matrix_: Tensor | np.ndarray | None = None

    def _to_tensor(self, array: Union[Tensor, np.ndarray, list[float]]) -> Tensor:
        if isinstance(array, Tensor):
            return array
        return torch.as_tensor(array)

    def _bin_values(self, y: Tensor) -> Tensor:
        if self.bin_edges_ is None:
            raise RuntimeError("bin_edges_ not fitted")
        edges = _ensure_tensor(self.bin_edges_)
        y_t = _ensure_tensor(y)
        indices = torch.bucketize(y_t, edges) - 1
        return indices.clamp(0, self.n_bins - 1)

    def _prepare_predictions(self, pred: Tensor) -> Tensor:
        pred_t = _ensure_tensor(pred)
        if pred_t.ndim == 1:
            bin_idx = self._bin_values(pred_t)
            one_hot = torch.zeros(
                pred_t.shape[0], self.n_bins, dtype=pred_t.dtype, device=pred_t.device
            )
            one_hot[torch.arange(pred_t.shape[0]), bin_idx] = 1.0
            return one_hot
        elif pred_t.ndim == 2:
            if pred_t.shape[1] != self.n_bins:
                raise ValueError(
                    f"Predicted probability dimension {pred_t.shape[1]} "
                    f"must match n_bins={self.n_bins}"
                )
            if (pred_t < 0.0).any():
                raise ValueError("Predicted probabilities must be non-negative")
            row_sum = pred_t.sum(dim=1, keepdim=True)
            return pred_t / row_sum.clamp(self.eps, None)
        else:
            raise ValueError("Predictions must be 1D point predictions or 2D probability vectors")

    def fit(
        self,
        y_source: Union[Tensor, np.ndarray, list[float]],
        pred_source: Union[Tensor, np.ndarray, list[float]],
        pred_target: Union[Tensor, np.ndarray, list[float]],
    ) -> "BinnedLabelShiftEstimator":
        y_src = self._to_tensor(y_source).reshape(-1).double()
        p_src = self._to_tensor(pred_source).double()
        p_tgt = self._to_tensor(pred_target).double()

        if self.binning_strategy == "adaptive":
            quantiles = torch.linspace(0, 1, self.n_bins + 1, dtype=y_src.dtype)
            self.bin_edges_ = torch.quantile(y_src, quantiles)
            for idx in range(1, len(self.bin_edges_)):
                if self.bin_edges_[idx] <= self.bin_edges_[idx - 1]:
                    self.bin_edges_[idx] = self.bin_edges_[idx - 1] + 1e-5
        else:
            self.bin_edges_ = torch.linspace(y_src.min(), y_src.max(), self.n_bins + 1)

        edges = self.bin_edges_
        edges[0] = -torch.inf if y_src.is_cuda else float("-inf")
        edges[-1] = torch.inf if y_src.is_cuda else float("inf")

        y_src_bins = self._bin_values(y_src)
        counts = torch.zeros(self.n_bins, dtype=torch.float64)
        counts.scatter_add_(0, y_src_bins.long(), torch.ones_like(y_src_bins, dtype=torch.float64))
        self.source_prior_ = counts / max(y_src.shape[0], 1)

        p_src_probs = self._prepare_predictions(p_src)
        p_tgt_probs = self._prepare_predictions(p_tgt)

        if self.method == "bbse":
            self._fit_bbse(y_src_bins, p_src_probs, p_tgt_probs)
        else:
            self._fit_em(p_tgt_probs)

        return self

    def _fit_bbse(self, y_src_bins: Tensor, p_src_probs: Tensor, p_tgt_probs: Tensor) -> None:
        conf_mat = torch.zeros(self.n_bins, self.n_bins, dtype=torch.float64)
        for j in range(self.n_bins):
            mask = y_src_bins == j
            sum_mask = mask.sum()
            if sum_mask > 0:
                conf_mat[:, j] = p_src_probs[mask].sum(dim=0) / sum_mask
            else:
                conf_mat[:, j] = 1.0 / self.n_bins

        self.confusion_matrix_ = conf_mat

        mu_target = p_tgt_probs.mean(dim=0)

        try:
            p_tgt_est = torch.linalg.solve(conf_mat, mu_target)
        except RuntimeError:
            p_tgt_est = torch.linalg.lstsq(conf_mat, mu_target).solution

        p_tgt_est = p_tgt_est.clamp(0.0, None)
        total = p_tgt_est.sum()
        self.target_prior_ = p_tgt_est / max(total.item(), self.eps)

    def _fit_em(self, p_tgt_probs: Tensor) -> None:
        if self.source_prior_ is None:
            raise RuntimeError("source_prior_ not fitted")
        src_prior = _ensure_tensor(self.source_prior_).clamp(self.eps, None)

        curr_prior = src_prior.clone()

        for _ in range(self.max_iter):
            weights = curr_prior / src_prior
            q = p_tgt_probs * weights.unsqueeze(0)
            q = q / q.sum(dim=1, keepdim=True).clamp(self.eps, None)

            next_prior = q.mean(dim=0)

            if torch.norm(next_prior - curr_prior) < self.tol:
                curr_prior = next_prior
                break
            curr_prior = next_prior

        self.target_prior_ = curr_prior

    def get_bin_weights(self) -> Tensor:
        if self.source_prior_ is None or self.target_prior_ is None:
            raise RuntimeError("call fit() before requesting weights")
        src = _ensure_tensor(self.source_prior_).clamp(self.eps, None)
        return _ensure_tensor(self.target_prior_) / src

    def sample_weights(self, y: Union[Tensor, np.ndarray, list[float]]) -> Tensor:
        y_arr = self._to_tensor(y)

        if y_arr.dtype in (torch.int32, torch.int64, torch.uint8, torch.int16):
            bin_idx = y_arr.clamp(0, self.n_bins - 1)
        else:
            bin_idx = self._bin_values(y_arr)

        bin_w = self.get_bin_weights()
        return bin_w[bin_idx.long()].to(device=y_arr.device, dtype=torch.float32)


__all__ = [
    "RepresentationShiftInflator",
    "BinnedLabelShiftEstimator",
]
