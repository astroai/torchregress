"""Shift-aware calibration adapters for test-time correction."""

from __future__ import annotations

import numpy as np


class RepresentationShiftCalibrator:
    """Map representation shift magnitude to a conservative temperature factor."""

    def __init__(
        self,
        *,
        base_temperature: float = 1.0,
        slope: float = 1.0,
        max_temperature: float = 5.0,
        eps: float = 1.0e-6,
    ) -> None:
        self.base_temperature = float(base_temperature)
        self.slope = float(slope)
        self.max_temperature = float(max_temperature)
        self.eps = float(eps)
        self.source_mean_: np.ndarray | None = None
        self.source_var_: np.ndarray | None = None
        self.reference_scale_: float | None = None

    def fit(self, source_representations: np.ndarray) -> "RepresentationShiftCalibrator":
        reps = np.asarray(source_representations, dtype=float)
        self.source_mean_ = reps.mean(axis=0)
        self.source_var_ = np.clip(reps.var(axis=0), self.eps, None)
        d2 = self._squared_mahalanobis(reps)
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

    def calibrate_probabilities(self, probabilities: np.ndarray, target_representations: np.ndarray) -> np.ndarray:
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


__all__ = ["RepresentationShiftCalibrator"]
