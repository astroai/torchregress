"""Shift calibration methods for test-time adaptation and prior correction."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

from torchregress.utils.numpy_stats import subsample_rows, winsorize


class RepresentationShiftCalibrator:
    """Map representation shift magnitude to a conservative temperature factor.

    References
    ----------
    .. [1] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration
       of Modern Neural Networks. In *ICML 2017*. https://arxiv.org/abs/1706.04599
    """

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
        self.source_mean_: np.ndarray | None = None
        self.source_var_: np.ndarray | None = None
        self.reference_scale_: float | None = None

    def fit(self, source_representations: np.ndarray) -> "RepresentationShiftCalibrator":
        reps = np.asarray(source_representations, dtype=float)
        reps_stats = subsample_rows(reps, self.source_sample_size, random_state=self.random_state)
        reps_stats = winsorize(reps_stats, self.clip_quantile)
        self.source_mean_ = reps_stats.mean(axis=0)
        self.source_var_ = np.clip(reps_stats.var(axis=0), self.eps, None)
        d2 = self._squared_mahalanobis(reps_stats)
        self.reference_scale_ = float(np.median(np.sqrt(np.clip(d2, 0.0, None))))
        return self

    def _squared_mahalanobis(self, reps: np.ndarray) -> np.ndarray:
        if self.source_mean_ is None or self.source_var_ is None:
            raise RuntimeError("call fit() before computing shift scores")
        centered = reps - self.source_mean_[None, :]
        return np.sum(centered**2 / self.source_var_[None, :], axis=1)

    def shift_scores(self, target_representations: np.ndarray) -> np.ndarray:
        reps = np.asarray(target_representations, dtype=float)
        return np.sqrt(np.clip(self._squared_mahalanobis(reps), 0.0, None))

    def temperatures(self, target_representations: np.ndarray) -> np.ndarray:
        scores = self.shift_scores(target_representations)
        ref = max(float(self.reference_scale_ or 1.0), self.eps)
        temps = self.base_temperature * (1.0 + self.slope * scores / ref)
        return np.clip(temps, self.base_temperature, self.max_temperature)

    def calibrate_probabilities(
        self, probabilities: np.ndarray, target_representations: np.ndarray
    ) -> np.ndarray:
        probs = np.asarray(probabilities, dtype=float)
        temps = self.temperatures(target_representations)[:, None]
        logits = np.log(np.clip(probs, self.eps, None))
        scaled = logits / temps
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        out = np.exp(scaled)
        return out / np.clip(out.sum(axis=1, keepdims=True), self.eps, None)

    def calibrate_std(self, std: np.ndarray, target_representations: np.ndarray) -> np.ndarray:
        sigma = np.asarray(std, dtype=float)
        temps = self.temperatures(target_representations)
        return np.clip(sigma * temps, self.eps, None)


class BinnedLabelShiftEstimator:
    """Estimates and corrects target prior distributions under label shift.

    Supports Black Box Shift Estimation (BBSE) and EM prior adjustment.
    Continuous targets are binned either uniformly or adaptively using quantiles.

    References
    ----------
    .. [1] Lipton, Z., Wang, Y. X., & Smola, A. (2018). Detecting and Correcting for
       Label Shift with Black Box Predictors. In *ICML 2018*. https://arxiv.org/abs/1802.03916
    .. [2] Saerens, M., Latinne, P., & Decaestecker, C. (2002). Adjusting the outputs
       of a classifier to new a priori probabilities. *Neural Computation*, 14(1).
    """

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

        # Fitted parameters
        self.bin_edges_: np.ndarray | None = None
        self.source_prior_: np.ndarray | None = None
        self.target_prior_: np.ndarray | None = None
        self.confusion_matrix_: np.ndarray | None = None

    def _to_numpy(self, array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy()
        return np.asarray(array)

    def _bin_values(self, y: np.ndarray) -> np.ndarray:
        """Assign values to bin indices based on fitted edges."""
        if self.bin_edges_ is None:
            raise RuntimeError("bin_edges_ not fitted")
        # digitize returns 1-based indices, map to 0-based
        indices = np.digitize(y, self.bin_edges_) - 1
        return np.clip(indices, 0, self.n_bins - 1)

    def _prepare_predictions(self, pred: np.ndarray) -> np.ndarray:
        """Ensure predictions are formatted as probability distributions over bins."""
        if pred.ndim == 1:
            # Point predictions, assign to bins and convert to one-hot
            bin_idx = self._bin_values(pred)
            one_hot = np.zeros((pred.shape[0], self.n_bins))
            one_hot[np.arange(pred.shape[0]), bin_idx] = 1.0
            return one_hot
        elif pred.ndim == 2:
            if pred.shape[1] != self.n_bins:
                raise ValueError(
                    f"Predicted probability dimension {pred.shape[1]} "
                    f"must match n_bins={self.n_bins}"
                )
            # Ensure proper normalized probabilities
            row_sum = pred.sum(axis=1, keepdims=True)
            return pred / np.clip(row_sum, self.eps, None)
        else:
            raise ValueError("Predictions must be 1D point predictions or 2D probability vectors")

    def fit(
        self,
        y_source: Union[np.ndarray, torch.Tensor],
        pred_source: Union[np.ndarray, torch.Tensor],
        pred_target: Union[np.ndarray, torch.Tensor],
    ) -> "BinnedLabelShiftEstimator":
        """Fit the estimator and estimate the target prior distribution.

        Parameters
        ----------
        y_source : Union[np.ndarray, torch.Tensor]
            True source labels, shape (N_source,).
        pred_source : Union[np.ndarray, torch.Tensor]
            Predictions on the source set, shape (N_source,) or (N_source, n_bins).
        pred_target : Union[np.ndarray, torch.Tensor]
            Predictions on the target set, shape (N_target,) or (N_target, n_bins).
        """
        y_src = self._to_numpy(y_source).reshape(-1)
        p_src = self._to_numpy(pred_source)
        p_tgt = self._to_numpy(pred_target)

        # 1. Fit bin edges on source labels
        if self.binning_strategy == "adaptive":
            # Quantile spacing to ensure equal representation
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            self.bin_edges_ = np.percentile(y_src, quantiles)
            # De-duplicate edges to ensure strict monotonicity
            for idx in range(1, len(self.bin_edges_)):
                if self.bin_edges_[idx] <= self.bin_edges_[idx - 1]:
                    self.bin_edges_[idx] = self.bin_edges_[idx - 1] + 1e-5
        else:
            # Uniform spacing
            self.bin_edges_ = np.linspace(y_src.min(), y_src.max(), self.n_bins + 1)

        # Extend boundaries to cover out-of-range values during target evaluation
        self.bin_edges_[0] = -np.inf
        self.bin_edges_[-1] = np.inf

        # 2. Bin true labels and compute source prior
        y_src_bins = self._bin_values(y_src)
        counts = np.bincount(y_src_bins, minlength=self.n_bins)
        self.source_prior_ = counts / max(y_src.shape[0], 1)

        # 3. Format soft/hard predictions
        p_src_probs = self._prepare_predictions(p_src)
        p_tgt_probs = self._prepare_predictions(p_tgt)

        # 4. Estimate target prior based on chosen method
        if self.method == "bbse":
            self._fit_bbse(y_src_bins, p_src_probs, p_tgt_probs)
        else:
            self._fit_em(p_tgt_probs)

        return self

    def _fit_bbse(
        self, y_src_bins: np.ndarray, p_src_probs: np.ndarray, p_tgt_probs: np.ndarray
    ) -> None:
        """Black Box Shift Estimation prior correction."""
        # Compute confusion matrix C_{i, j} = P(pred_bin = i | true_bin = j)
        conf_mat = np.zeros((self.n_bins, self.n_bins))
        for j in range(self.n_bins):
            mask = y_src_bins == j
            sum_mask = mask.sum()
            if sum_mask > 0:
                conf_mat[:, j] = p_src_probs[mask].sum(axis=0) / sum_mask
            else:
                conf_mat[:, j] = 1.0 / self.n_bins  # uniform fallback

        self.confusion_matrix_ = conf_mat

        # Average prediction vector on target
        mu_target = p_tgt_probs.mean(axis=0)

        # Solve system C * p_target = mu_target
        try:
            p_tgt_est = np.linalg.solve(conf_mat, mu_target)
        except np.linalg.LinAlgError:
            # fallback to least squares if singular
            p_tgt_est, _, _, _ = np.linalg.lstsq(conf_mat, mu_target, rcond=None)

        # Project estimated target prior onto the probability simplex
        p_tgt_est = np.clip(p_tgt_est, 0.0, None)
        total = p_tgt_est.sum()
        self.target_prior_ = p_tgt_est / max(total, self.eps)

    def _fit_em(self, p_tgt_probs: np.ndarray) -> None:
        """EM-based target prior estimation (Saerens algorithm)."""
        if self.source_prior_ is None:
            raise RuntimeError("source_prior_ not fitted")
        src_prior = np.clip(self.source_prior_, self.eps, None)

        # Initialize target prior to source prior
        curr_prior = src_prior.copy()

        for _ in range(self.max_iter):
            # Compute adjusted target posterior for each sample:
            # q_{n, j} \propto p(y_bin = j | x_n) * (p_target(j) / p_source(j))
            weights = curr_prior / src_prior
            q = p_tgt_probs * weights[None, :]
            q = q / np.clip(q.sum(axis=1, keepdims=True), self.eps, None)

            # Update prior to mean target posterior
            next_prior = q.mean(axis=0)

            # Check convergence
            if np.linalg.norm(next_prior - curr_prior) < self.tol:
                curr_prior = next_prior
                break
            curr_prior = next_prior

        self.target_prior_ = curr_prior

    def get_bin_weights(self) -> np.ndarray:
        """Compute importance weights w_j = p_target(j) / p_source(j) per bin."""
        if self.source_prior_ is None or self.target_prior_ is None:
            raise RuntimeError("call fit() before requesting weights")
        src = np.clip(self.source_prior_, self.eps, None)
        return self.target_prior_ / src

    def sample_weights(self, y: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Compute sample-level importance weights for given targets.

        Parameters
        ----------
        y : Union[np.ndarray, torch.Tensor]
            Target continuous values or class indices. If float, binned internally.
        """
        is_tensor = isinstance(y, torch.Tensor)
        y_arr = self._to_numpy(y)

        # Determine if targets are continuous values or already bin indices
        if np.issubdtype(y_arr.dtype, np.integer):
            bin_idx = np.clip(y_arr, 0, self.n_bins - 1)
        else:
            bin_idx = self._bin_values(y_arr)

        bin_w = self.get_bin_weights()
        weights = bin_w[bin_idx]

        if is_tensor:
            return torch.as_tensor(weights, dtype=torch.float32, device=y.device)
        return weights


__all__ = [
    "RepresentationShiftCalibrator",
    "BinnedLabelShiftEstimator",
]
