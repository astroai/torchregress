"""Input preprocessing for tabular neural networks."""

from __future__ import annotations

import warnings
from typing import Any, Literal, Optional, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


class TabularPreprocessor:
    """Robust scaling, outlier clipping and missing-value handling for numeric features.

    Follows the RealMLP recipe (Holzmueller et al., 2024): each column is centred at
    its median and divided by its interquartile range (IQR), then passed through a
    smooth clip. Columns with zero IQR fall back to the half range, then to 1.

    Per column, with median :math:`m_j` and scale :math:`s_j`:

    .. math::

        z = (x - m_j) / s_j, \\qquad
        \\tilde z = \\frac{z}{\\sqrt{1 + (z / c)^2}} \\quad (\\text{smooth clip}),

    so :math:`\\tilde z \\in (-c, c)` and the map is strictly monotone and invertible.

    Missing values (NaN) are handled in two ways: they are replaced by the training
    median (``z = 0``) and, for every column that contains NaN *at fit time*, a binary
    indicator column (1 = missing) is appended after the scaled features. The output
    width is therefore ``n_features_out_ = n_features_in_ + n_missing_indicators_``;
    build the model with ``in_features=preprocessor.n_features_out_``. Set
    ``missing_indicator="never"`` for plain median imputation, or ``"always"`` to add an
    indicator for every column.

    Parameters
    ----------
    clip : {"smooth", "hard", None}, default "smooth"
        ``"smooth"`` is the RealMLP soft clip, ``"hard"`` clamps to ``[-c, c]``
        (not invertible outside that range) and ``None`` disables clipping.
    clip_value : float, default 3.0
        The clip scale :math:`c` (in robust-scaled units).
    missing_indicator : {"auto", "always", "never"}, default "auto"
        Which columns receive a missing-value indicator feature.
    dtype : torch.dtype or None, default None
        Output dtype for tensors. ``None`` keeps the input's floating dtype
        (``float32`` for integer input).

    Attributes
    ----------
    center_, scale_ : numpy.ndarray
        Fitted per-column median and scale (``float64``).
    indicator_columns_ : numpy.ndarray
        Indices of the columns that get a missing indicator.
    n_features_in_, n_features_out_ : int
        Input and output widths.

    Examples
    --------
    >>> import numpy as np
    >>> from torchregress.models import TabularPreprocessor
    >>> X = np.array([[1.0, 10.0], [2.0, np.nan], [3.0, 30.0], [4.0, 40.0]])
    >>> prep = TabularPreprocessor().fit(X)
    >>> prep.transform(X).shape
    (4, 3)
    """

    def __init__(
        self,
        clip: Optional[Literal["smooth", "hard"]] = "smooth",
        clip_value: float = 3.0,
        missing_indicator: Literal["auto", "always", "never"] = "auto",
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if clip not in ("smooth", "hard", None):
            raise ValueError(f"clip must be 'smooth', 'hard' or None, got {clip!r}")
        if clip_value <= 0:
            raise ValueError("clip_value must be positive")
        if missing_indicator not in ("auto", "always", "never"):
            raise ValueError("missing_indicator must be 'auto', 'always' or 'never'")
        self.clip = clip
        self.clip_value = float(clip_value)
        self.missing_indicator = missing_indicator
        self.dtype = dtype
        self._fitted = False

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _to_numpy(X: ArrayLike) -> np.ndarray:
        if isinstance(X, torch.Tensor):
            arr = X.detach().cpu().numpy()
        else:
            arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_features), got shape {arr.shape}")
        return arr.astype(np.float64, copy=False)

    def _out_dtype(self, X: ArrayLike) -> Any:
        if self.dtype is not None:
            return self.dtype
        if isinstance(X, torch.Tensor):
            return X.dtype if X.is_floating_point() else torch.float32
        arr = np.asarray(X)
        return arr.dtype if np.issubdtype(arr.dtype, np.floating) else np.float32

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._fitted

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("TabularPreprocessor is not fitted; call fit() first")

    # ---------------------------------------------------------------------- API
    def fit(self, X: ArrayLike) -> "TabularPreprocessor":
        """Estimate per-column median and scale from ``X`` (NaN ignored).

        Parameters
        ----------
        X : numpy.ndarray or torch.Tensor
            Training features of shape ``(n_samples, n_features)``.

        Returns
        -------
        TabularPreprocessor
            ``self``.
        """
        arr = self._to_numpy(X)
        if np.isinf(arr).any():
            raise ValueError("X contains infinite values")
        n, d = arr.shape
        all_nan = np.isnan(arr).all(axis=0)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            q25, med, q75 = np.nanpercentile(arr, [25, 50, 75], axis=0)
            lo, hi = np.nanmin(arr, axis=0), np.nanmax(arr, axis=0)
        med = np.where(all_nan, 0.0, med)
        scale = q75 - q25
        half_range = 0.5 * (hi - lo)
        scale = np.where(scale > 0, scale, half_range)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        has_nan = np.isnan(arr).any(axis=0)
        if self.missing_indicator == "always":
            ind = np.arange(d)
        elif self.missing_indicator == "auto":
            ind = np.flatnonzero(has_nan)
        else:
            ind = np.empty(0, dtype=np.int64)
        self.center_ = med
        self.scale_ = scale
        self.indicator_columns_ = ind.astype(np.int64)
        self.n_features_in_ = d
        self.n_features_out_ = d + len(ind)
        self._fitted = True
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Scale, clip and impute ``X``; returns the same container type as the input.

        Parameters
        ----------
        X : numpy.ndarray or torch.Tensor
            Features with ``n_features_in_`` columns.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Array of shape ``(n_samples, n_features_out_)``; scaled features first,
            then the missing indicators.
        """
        self._check_fitted()
        arr = self._to_numpy(X)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {arr.shape[1]}")
        missing = np.isnan(arr)
        z = (arr - self.center_) / self.scale_
        z = np.where(missing, 0.0, z)
        c = self.clip_value
        if self.clip == "smooth":
            # hypot avoids overflowing (z / c) ** 2 for |z| > ~1e154; infinities
            # saturate at +-c instead of becoming NaN.
            with np.errstate(over="ignore", invalid="ignore"):
                z = np.where(np.isinf(z), np.sign(z) * c, z / np.hypot(1.0, z / c))
        elif self.clip == "hard":
            z = np.clip(z, -c, c)
        if len(self.indicator_columns_):
            z = np.concatenate([z, missing[:, self.indicator_columns_].astype(np.float64)], 1)
        dtype = self._out_dtype(X)
        if isinstance(X, torch.Tensor):
            return torch.as_tensor(z, dtype=dtype, device=X.device)
        return z.astype(dtype)

    def fit_transform(self, X: ArrayLike) -> ArrayLike:
        """Fit on ``X`` and return ``transform(X)``."""
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: ArrayLike) -> ArrayLike:
        """Invert :meth:`transform` (exact for ``clip`` in ``{"smooth", None}``).

        Entries flagged by a missing indicator are restored to NaN. Without
        indicators, imputed entries come back as the training median. With
        ``clip="hard"`` values beyond ``clip_value`` are not recoverable.

        Parameters
        ----------
        Z : numpy.ndarray or torch.Tensor
            Output of :meth:`transform`, shape ``(n_samples, n_features_out_)``.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Features of shape ``(n_samples, n_features_in_)``.
        """
        self._check_fitted()
        arr = self._to_numpy(Z)
        if arr.shape[1] != self.n_features_out_:
            raise ValueError(f"expected {self.n_features_out_} columns, got {arr.shape[1]}")
        d = self.n_features_in_
        z = arr[:, :d].copy()
        c = self.clip_value
        if self.clip == "smooth":
            z = z / np.sqrt(np.clip(1.0 - (z / c) ** 2, 1e-12, None))
        x = z * self.scale_ + self.center_
        if len(self.indicator_columns_):
            miss = arr[:, d:] > 0.5
            sub = x[:, self.indicator_columns_]
            x[:, self.indicator_columns_] = np.where(miss, np.nan, sub)
        if isinstance(Z, torch.Tensor):
            return torch.as_tensor(x, dtype=self._out_dtype(Z), device=Z.device)
        return x.astype(self._out_dtype(Z))
