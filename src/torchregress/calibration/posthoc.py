"""Post-hoc calibration transforms for regression outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from torchregress.utils.distributions import normal_cdf


@dataclass
class VarianceTemperatureScaler:
    """Scalar variance-temperature calibration for Gaussian predictive variance.

    Fits ``T`` (and optionally an additive variance floor ``f``) by Gaussian
    negative log-likelihood on a held-out calibration split, so that the
    calibrated predictive variance is ``T * pred_var + f``.

    Two optional extensions of the scalar temperature, both off by default:

    * ``target_var``: per-sample variance of a *noisy* target (labels with their
      own uncertainty). The likelihood then uses ``T * pred_var + f + target_var``
      (as in :func:`torchregress.metrics.uncertain.noisy_target_gaussian_nll`),
      so label noise is not attributed to the model. :meth:`transform` returns
      the model's variance only.
    * ``fit_floor``: also fit ``f >= 0``. A temperature alone cannot represent an
      error budget with a floor (e.g. template or systematics error added to
      a photon-noise error), which shows up as under-coverage for the
      best-measured samples and over-coverage for the worst.

    ``clip`` drops calibration samples whose residual lies more than ``clip``
    robust standard deviations (1.4826 MAD) from the median before fitting, so
    a few catastrophic failures do not set the scale. This is a trade-off: the
    variance is then fitted *without* the outliers, so outliers met at test time
    are scored against a narrower predictive distribution (better CRPS and
    coverage of the bulk, worse NLL on the outliers).

    Without ``target_var``, ``fit_floor`` or a preset ``variance_floor`` the
    Gaussian NLL optimum is available in closed form,
    ``T = mean((y - mu)^2 / pred_var)``, and is returned exactly (no
    iterations). Otherwise the closed-form value seeds an Adam optimisation
    over ``log T`` (and ``log f``).

    ``T`` is not bounded by default. ``temperature_bounds=(t_min, t_max)``
    constrains it (the constrained optimum is the projection of the
    unconstrained one in the closed-form case, and Adam iterates are projected
    onto the box otherwise). Versions before 0.3 silently clamped ``T`` to
    ``[0.05, 20]``; pass ``temperature_bounds=(0.05, 20.0)`` to reproduce that.

    References
    ----------
    .. [1] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration
       of Modern Neural Networks. In *ICML 2017*. https://arxiv.org/abs/1706.04599
    .. [2] Kuleshov, V., Fenner, N., & Ermon, S. (2018). Accurate Uncertainties for
       Deep Learning Using Calibrated Regression. In *ICML 2018*.
       https://arxiv.org/abs/1807.00263
    """

    temperature: float = 1.0
    eps: float = 1e-8
    variance_floor: float = 0.0

    def fit(
        self,
        pred_mean: Tensor,
        pred_var: Tensor,
        target: Tensor,
        *,
        max_iter: int = 200,
        lr: float = 0.05,
        target_var: Tensor | None = None,
        fit_floor: bool = False,
        clip: float | None = None,
        temperature_bounds: tuple[float, float] | None = None,
    ) -> "VarianceTemperatureScaler":
        """Fit the temperature (and optionally the floor) on a calibration split.

        Parameters
        ----------
        pred_mean, pred_var, target : Tensor
            Predicted mean, predicted variance and targets (same shape).
        max_iter : int
            Adam iterations when no closed form applies.
        lr : float
            Adam learning rate (on ``log T`` / ``log f``).
        target_var : Tensor, optional
            Per-sample variance of noisy targets, added to the likelihood only.
        fit_floor : bool
            Also fit an additive variance floor ``f >= 0``.
        clip : float, optional
            Drop samples with residuals beyond ``clip`` robust sigmas first.
        temperature_bounds : tuple of float, optional
            ``(t_min, t_max)`` box for ``T``; ``None`` (default) leaves it free.

        Returns
        -------
        VarianceTemperatureScaler
            ``self``, fitted.
        """
        if pred_mean.shape != pred_var.shape or pred_mean.shape != target.shape:
            raise ValueError("pred_mean, pred_var, and target must share shape")
        if target_var is not None and target_var.shape != pred_var.shape:
            raise ValueError("target_var must share shape with pred_var")
        if clip is not None and not clip > 0:
            raise ValueError("clip must be positive")
        if temperature_bounds is not None:
            t_lo, t_hi = (float(b) for b in temperature_bounds)
            if not 0.0 < t_lo <= t_hi:
                raise ValueError("temperature_bounds must satisfy 0 < t_min <= t_max")
            log_bounds: tuple[float, float] | None = (math.log(t_lo), math.log(t_hi))
        else:
            log_bounds = None

        mean = pred_mean.detach().double()
        # ``eps`` is applied after rescaling to the typical variance below, so a
        # target measured in tiny units (variances ~1e-10) is not clamped.
        var = pred_var.detach().double().clamp_min(torch.finfo(torch.float64).tiny)
        y = target.detach().double()
        noise = (
            torch.zeros_like(var)
            if target_var is None
            else target_var.detach().double().clamp_min(0.0)
        )
        keep = torch.isfinite(mean) & torch.isfinite(var) & torch.isfinite(y)
        keep &= torch.isfinite(noise)
        if clip is not None:
            residual = (y - mean)[keep]
            centre = residual.median()
            robust = 1.4826 * (residual - centre).abs().median()
            if float(robust) > 0:
                keep &= (y - mean - centre).abs() <= clip * robust
        if int(keep.sum()) < 2:
            raise ValueError("need at least two finite calibration samples")
        mean, var, y, noise = mean[keep], var[keep], y[keep], noise[keep]

        # Work in units of the typical predicted variance, so the optimiser sees
        # O(1) numbers whatever the physical scale of the target.
        unit = var.median().clamp_min(torch.finfo(torch.float64).tiny)
        var_u, noise_u = (var / unit).clamp_min(self.eps), noise / unit
        residual_sq = (y - mean) ** 2 / unit
        floor_u = self.variance_floor / float(unit)

        def project(log_t_value: float) -> float:
            if log_bounds is None:
                return log_t_value
            return min(max(log_t_value, log_bounds[0]), log_bounds[1])

        # M-CAL-002: closed-form NLL optimum of T * var (moment-corrected for
        # label noise and a preset floor); exact when neither is present.
        excess = (residual_sq - noise_u - floor_u).clamp_min(0.0) / var_u
        t_closed = float(excess.mean())
        exact = target_var is None and not fit_floor and floor_u == 0.0
        if exact and t_closed > 0.0:
            # The NLL is convex in log T, so projecting the optimum onto the
            # bounds gives the constrained optimum.
            self.temperature = math.exp(project(math.log(t_closed)))
            return self
        log_t0 = math.log(t_closed) if t_closed > 0.0 else math.log(self.temperature)

        log_t = torch.nn.Parameter(torch.tensor(project(log_t0), dtype=torch.float64))
        parameters = [log_t]
        log_f = None
        if fit_floor:
            start = max(floor_u, 1e-2)
            log_f = torch.nn.Parameter(torch.tensor(math.log(start), dtype=torch.float64))
            parameters.append(log_f)
        optimizer = torch.optim.Adam(parameters, lr=lr)

        def total_variance() -> Tensor:
            t = torch.exp(log_t)
            floor = (
                torch.exp(log_f)
                if log_f is not None
                else torch.tensor(floor_u, dtype=torch.float64)
            )
            return (var_u * t + floor + noise_u).clamp_min(self.eps)

        for _ in range(max_iter):
            optimizer.zero_grad(set_to_none=True)
            total = total_variance()
            nll = 0.5 * (torch.log(total) + residual_sq / total + math.log(2.0 * math.pi))
            loss = nll.mean()
            loss.backward()
            optimizer.step()
            if log_bounds is not None:
                with torch.no_grad():
                    log_t.clamp_(log_bounds[0], log_bounds[1])

        self.temperature = float(torch.exp(log_t).item())
        if log_f is not None:
            self.variance_floor = float(torch.exp(log_f).item() * float(unit))
        return self

    def transform(self, pred_var: Tensor) -> Tensor:
        """Calibrated variance ``T * pred_var + f``, clamped below at ``eps``.

        ``eps`` is in the caller's variance units: set it below the smallest
        meaningful variance when targets live on tiny scales.
        """
        return (pred_var * self.temperature + self.variance_floor).clamp_min(self.eps)


@dataclass
class IsotonicMeanCalibrator:
    """Isotonic regression calibrator for point predictions.

    Fits the least-squares non-decreasing map ``g`` from predictions to targets
    with the Pool Adjacent Violators Algorithm (PAVA), directly on PyTorch
    tensors so no scikit-learn dependency is required. The fit reproduces
    ``sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")``:

    1. targets sharing the same prediction are averaged (ties are pooled, with
       the tie counts as weights);
    2. weighted PAVA merges adjacent blocks that violate monotonicity;
    3. each block is stored by its first and last prediction (the
       ``X_thresholds_`` of scikit-learn), so :meth:`transform` returns the
       block value on the whole block and interpolates linearly *between*
       blocks.

    ``out_of_bounds="clip"`` (default) holds the end values outside the
    training range; any other value extrapolates the end segments linearly.

    References
    ----------
    .. [1] Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate
       multiclass probability estimates. In *KDD 2002*. https://doi.org/10.1145/775047.775151
    .. [2] Barlow, R. E., Bartholomew, D. J., Bremner, J. M., & Brunk, H. D. (1972).
       *Statistical Inference under Order Restrictions*. Wiley.
    """

    out_of_bounds: str = "clip"

    def __post_init__(self) -> None:
        self._x: Tensor | None = None
        self._y: Tensor | None = None

    @staticmethod
    def _pava(x: Tensor | np.ndarray, y: Tensor | np.ndarray) -> tuple[Any, Any]:
        """Isotonic fit; returns block thresholds ``(x_thresholds, y_thresholds)``.

        Each PAVA block contributes its first and last (unique) ``x`` with the
        block mean as value; single-point blocks contribute one point.
        """
        is_numpy = isinstance(x, np.ndarray)
        x_t = torch.as_tensor(x).detach().reshape(-1).double()
        y_t = torch.as_tensor(y).detach().reshape(-1).double()

        if x_t.numel() == 0:
            if is_numpy:
                return np.array([], dtype=float), np.array([], dtype=float)
            return x_t.clone(), y_t.clone()

        # 1. Pool ties: mean target per unique prediction, counts as weights.
        ux, inverse = torch.unique(x_t, sorted=True, return_inverse=True)
        counts = torch.zeros_like(ux).index_add_(0, inverse, torch.ones_like(y_t))
        sums = torch.zeros_like(ux).index_add_(0, inverse, y_t)
        means = (sums / counts).tolist()
        weights = counts.tolist()

        # 2. Weighted PAVA over the unique predictions (stack of blocks).
        values: list[float] = []
        block_w: list[float] = []
        first: list[int] = []
        last: list[int] = []
        for i, (m, w) in enumerate(zip(means, weights)):
            values.append(m)
            block_w.append(w)
            first.append(i)
            last.append(i)
            while len(values) >= 2 and values[-2] > values[-1]:
                v2, w2, l2 = values.pop(), block_w.pop(), last.pop()
                first.pop()
                w1 = block_w[-1]
                values[-1] = (values[-1] * w1 + v2 * w2) / (w1 + w2)
                block_w[-1] = w1 + w2
                last[-1] = l2

        # 3. Block end points (scikit-learn X_thresholds_ / y_thresholds_).
        idx: list[int] = []
        vals: list[float] = []
        for v, a, b in zip(values, first, last):
            idx.append(a)
            vals.append(v)
            if b != a:
                idx.append(b)
                vals.append(v)
        result_x = ux[torch.tensor(idx, dtype=torch.long)]
        result_y = torch.tensor(vals, dtype=torch.float64)

        if is_numpy:
            return result_x.numpy(), result_y.numpy()
        return result_x, result_y

    def _interpolate(self, x_query: Tensor) -> Tensor:
        xq = torch.as_tensor(x_query)
        if self._x is None or self._y is None:
            raise ValueError("IsotonicMeanCalibrator must be fitted before transform")

        if self._x.numel() == 0:
            return torch.zeros_like(xq)

        if self._x.numel() == 1:
            val = float(self._y[0].item())
            return torch.full_like(xq, val)

        idx = torch.searchsorted(self._x, xq.contiguous()).clamp(1, len(self._x) - 1)
        x_left = self._x[idx - 1]
        x_right = self._x[idx]
        y_left = self._y[idx - 1]
        y_right = self._y[idx]

        # Thresholds are strictly increasing unique predictions, so denom > 0;
        # no absolute tolerance (it would break predictions on tiny scales).
        denom = x_right - x_left
        t = torch.where(denom > 0, (xq - x_left) / denom, torch.zeros_like(xq))

        if self.out_of_bounds == "clip":
            t = t.clamp(0.0, 1.0)
            y_out = y_left + t * (y_right - y_left)
            y_out = torch.where(xq <= self._x[0], self._y[0], y_out)
            y_out = torch.where(xq >= self._x[-1], self._y[-1], y_out)
            return y_out

        return y_left + t * (y_right - y_left)

    def fit(self, pred_mean: Tensor, target: Tensor) -> "IsotonicMeanCalibrator":
        x = pred_mean.detach().reshape(-1).double()
        y = target.detach().reshape(-1).double()
        if x.shape[0] != y.shape[0]:
            raise ValueError("pred_mean and target must share sample dimension")

        self._x, self._y = self._pava(x, y)
        return self

    def transform(self, pred_mean: Tensor) -> Tensor:
        x = pred_mean.detach().reshape(-1).double()
        y_hat = self._interpolate(x)
        return y_hat.reshape(pred_mean.shape).to(pred_mean.dtype)


@dataclass
class PITCalibrator:
    """Monotonic PIT-value calibrator using empirical CDF mapping."""

    eps: float = 1e-6

    def __post_init__(self) -> None:
        self._x: Tensor | None = None
        self._y: Tensor | None = None

    @staticmethod
    def pit_from_gaussian(pred_mean: Tensor, pred_std: Tensor, target: Tensor) -> Tensor:
        std = pred_std.clamp_min(1e-8)
        z = (target - pred_mean) / std
        return normal_cdf(z).clamp(min=1e-6, max=1.0 - 1e-6)

    def fit(self, pit_values: Tensor) -> "PITCalibrator":
        pit = pit_values.detach().reshape(-1).clamp(self.eps, 1.0 - self.eps)
        pit_sorted = pit.sort().values
        n = pit_sorted.shape[0]
        targets = (torch.arange(n, dtype=torch.float64) + 0.5) / n
        self._x = pit_sorted
        self._y = targets
        return self

    def transform(self, pit_values: Tensor) -> Tensor:
        if self._x is None or self._y is None:
            raise ValueError("PITCalibrator must be fitted before transform")
        pit = pit_values.detach().reshape(-1)
        idx = torch.searchsorted(self._x, pit).clamp(1, len(self._x) - 1)
        x_left = self._x[idx - 1]
        x_right = self._x[idx]
        y_left = self._y[idx - 1]
        y_right = self._y[idx]
        denom = x_right - x_left
        t = torch.where(denom.abs() < 1e-12, 0.0, (pit - x_left) / denom)
        mapped = y_left + t.clamp(0.0, 1.0) * (y_right - y_left)
        mapped = torch.where(pit <= self._x[0], self._y[0], mapped)
        mapped = torch.where(pit >= self._x[-1], self._y[-1], mapped)
        return mapped.clamp(self.eps, 1.0 - self.eps).reshape(pit_values.shape).to(pit_values.dtype)


@dataclass
class VarianceRestoringCalibrator:
    """Restore the marginal variance and tails of conditional-mean predictions.

    A predictor trained with mean-squared error estimates the conditional mean
    ``E[y | x]``. The law of total variance,

        Var(y) = Var(E[y | x]) + E[Var(y | x)],

    implies that such predictions are generically *under-dispersed*: the target
    noise term is not represented, the marginal variance of the predictions is
    smaller than that of the targets, and extreme targets (e.g. stars with
    ``[Fe/H] < -2.5``) are compressed toward the bulk. This calibrator re-shapes
    the point predictions themselves so that their marginal distribution
    matches the calibration-target distribution. It provides two maps:

    * ``method="affine"`` (moment matching): with prior mean ``mu0`` (default
      ``mean(target)``) and ``s = sqrt(Var(target) / Var(pred))``,

          T(z) = mu0 + s * (z - mu0).

      Fitted attributes are ``prior_mean_`` (``mu0``) and ``scale_`` (``s``).
      The variance ratio is matched exactly by construction; the mean moves by
      the same stretch, so it equals the target mean only when the fit
      predictions are already centred on ``mu0``.
    * ``method="quantile"`` (1-D optimal transport): sort the calibration
      predictions and targets as ``x_(1) <= ... <= x_(n)`` and
      ``y_(1) <= ... <= y_(n)``; between the pairs ``(x_(i), y_(i))`` the map is
      the piecewise-linear monotone interpolant, continued beyond the
      calibration range according to ``out_of_bounds``. In one dimension the
      monotone map is the W2-optimal transport map between the empirical
      marginals, and it reproduces the calibration marginal exactly. Fitted
      attributes are the sorted grids ``_x`` and ``_y``.

    .. warning::
       Marginal matching is *not* conditional-mean calibration. For a perfectly
       calibrated MSE-optimal predictor (``E[y | z] = z``), the missing spread
       is irreducible target noise: stretching restores the marginal variance
       and tails but increases the MSE (by about ``(s - 1)^2 Var(pred)`` for the
       affine map). For an attenuated / under-dispersed predictor
       (``pred = rho * y + noise`` with ``rho < 1``) it can reduce both the
       Wasserstein-1 distance and the MSE. There is no unconditional
       improvement guarantee, so :meth:`describe` reports ``mse_before`` and
       ``mse_after`` next to the marginal diagnostics. Use this calibrator when
       the *population* distribution of predictions is the quantity of interest
       (e.g. population-level metallicity studies), not as a free fix for
       conditional bias.

    A per-sample heteroscedastic rescaling, ``sigma * sqrt(1 + sigma**2 /
    Var(mu))``, was considered for the original audit but is intentionally not
    implemented: it is an ad hoc rule rather than the solution of a transport or
    moment-matching objective and it diverges where the predicted-mean variance
    is small. Global moment matching and the 1-D optimal-transport map are the
    principled versions of the same intuition.

    Parameters
    ----------
    method : {"affine", "quantile"}
        Moment matching (two parameters) or 1-D optimal transport (fully
        non-parametric) marginal matching.
    out_of_bounds : {"clip", "linear"}
        ``"clip"`` (default) holds the end values outside the calibration
        prediction range; ``"linear"`` extrapolates the end segments linearly.
        Only used by ``method="quantile"``.
    eps : float
        Small positive floor in target-variance (squared-target) units, used as
        a denominator guard for the affine scale and for the ratios reported by
        :meth:`describe`. Lower it when targets live on tiny scales.

    References
    ----------
    .. [1] Villani, C. (2009). *Optimal Transport: Old and New*. Grundlehren der
       mathematischen Wissenschaften 338. Springer.
    .. [2] Peyre, G., & Cuturi, M. (2019). Computational Optimal Transport.
       *Foundations and Trends in Machine Learning*, 11(5-6), 355-607.
    .. [3] Kuleshov, V., Fenner, N., & Ermon, S. (2018). Accurate Uncertainties
       for Deep Learning Using Calibrated Regression. In *ICML 2018*.
       https://arxiv.org/abs/1807.00263
    .. [4] Casella, G., & Berger, R. L. (2002). *Statistical Inference*
       (2nd ed.). Duxbury. (Law of total variance.)
    """

    method: str = "affine"
    out_of_bounds: str = "clip"
    eps: float = 1e-8

    def __post_init__(self) -> None:
        self.prior_mean_: float | None = None
        self.scale_: float | None = None
        self._x: Tensor | None = None
        self._y: Tensor | None = None

    def _validate_hyperparameters(self) -> None:
        if self.method not in ("affine", "quantile"):
            raise ValueError(f"unknown method {self.method!r}; expected 'affine' or 'quantile'")
        if self.method == "quantile" and self.out_of_bounds not in ("clip", "linear"):
            raise ValueError(
                f"unknown out_of_bounds {self.out_of_bounds!r}; expected 'clip' or 'linear'"
            )
        if not self.eps > 0:
            raise ValueError("eps must be positive")

    def fit(
        self,
        pred_mean: Tensor,
        target: Tensor,
        *,
        prior_mean: float | Tensor | None = None,
    ) -> "VarianceRestoringCalibrator":
        """Fit the marginal map on a held-out calibration split.

        Parameters
        ----------
        pred_mean : Tensor
            Point predictions on the calibration set (any shape; flattened).
        target : Tensor
            Targets on the calibration set (same number of elements).
        prior_mean : float or Tensor, optional
            Centre ``mu0`` of the affine map. Defaults to ``mean(target)``.
            Must be a finite scalar and is only accepted by
            ``method="affine"``.

        Returns
        -------
        VarianceRestoringCalibrator
            ``self``, fitted.

        Notes
        -----
        Non-finite pairs are dropped before fitting. At least two finite
        samples and a non-zero prediction variance are required.
        """
        self._validate_hyperparameters()
        if pred_mean.numel() != target.numel():
            raise ValueError("pred_mean and target must contain the same number of elements")
        if prior_mean is not None:
            if self.method != "affine":
                raise ValueError("prior_mean is only used by method='affine'")
            prior = torch.as_tensor(prior_mean, dtype=torch.float64).detach().reshape(-1)
            if prior.numel() != 1:
                raise ValueError("prior_mean must be a scalar")
            if not bool(torch.isfinite(prior).all()):
                raise ValueError("prior_mean must be finite")

        x = pred_mean.detach().reshape(-1).double()
        y = target.detach().reshape(-1).double()
        keep = torch.isfinite(x) & torch.isfinite(y)
        if int(keep.sum()) < 2:
            raise ValueError("need at least two finite calibration samples")
        x, y = x[keep], y[keep]
        var_x = float(x.var(unbiased=False))
        if not var_x > 0.0:
            raise ValueError(
                "pred_mean must have non-zero variance to fit a variance-restoring map"
            )

        self.prior_mean_ = None
        self.scale_ = None
        self._x = None
        self._y = None
        if self.method == "affine":
            var_y = float(y.var(unbiased=False))
            self.scale_ = math.sqrt(var_y / max(var_x, self.eps))
            if not math.isfinite(self.scale_):
                raise ValueError("degenerate calibration data: affine scale is not finite")
            if prior_mean is None:
                self.prior_mean_ = float(y.mean())
            else:
                self.prior_mean_ = float(torch.as_tensor(prior_mean, dtype=torch.float64))
        else:
            self._x = x.sort().values.contiguous()
            self._y = y.sort().values.contiguous()
        return self

    def _affine_map(self, x: Tensor) -> Tensor:
        prior = self.prior_mean_
        scale = self.scale_
        if prior is None or scale is None:
            raise ValueError("VarianceRestoringCalibrator must be fitted before transform")
        return prior + scale * (x - prior)

    def _quantile_map(self, x: Tensor) -> Tensor:
        grid_x = self._x
        grid_y = self._y
        if grid_x is None or grid_y is None:
            raise ValueError("VarianceRestoringCalibrator must be fitted before transform")
        n = grid_x.numel()
        if n < 2:  # defensive; fit() requires at least two finite samples
            return torch.full_like(x, float(grid_y[0]))

        # torch.searchsorted returns n for queries beyond the last grid point.
        idx = torch.searchsorted(grid_x, x.contiguous()).clamp(1, n - 1)
        x_left = grid_x[idx - 1]
        x_right = grid_x[idx]
        y_left = grid_y[idx - 1]
        y_right = grid_y[idx]

        # Duplicate calibration predictions would give a zero width; map them to
        # the left end of the tie block (the map is not single-valued there).
        denom = x_right - x_left
        safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        t = torch.where(denom > 0, (x - x_left) / safe_denom, torch.zeros_like(x))

        if self.out_of_bounds == "clip":
            t = t.clamp(0.0, 1.0)
            mapped = y_left + t * (y_right - y_left)
            mapped = torch.where(x <= grid_x[0], grid_y[0], mapped)
            return torch.where(x >= grid_x[-1], grid_y[-1], mapped)
        return y_left + t * (y_right - y_left)

    def transform(self, pred_mean: Tensor) -> Tensor:
        """Apply the fitted marginal map, preserving shape and dtype.

        ``transform`` raises a ``ValueError`` if the calibrator has not been
        fitted yet.
        """
        self._validate_hyperparameters()
        x = pred_mean.detach().reshape(-1).double()
        if self.method == "affine":
            mapped = self._affine_map(x)
        else:
            mapped = self._quantile_map(x)
        return mapped.reshape(pred_mean.shape).to(pred_mean.dtype)

    def describe(self, pred_mean: Tensor, target: Tensor) -> dict[str, float]:
        """Diagnostics of the fitted map on the data supplied by the caller.

        This is an evaluation helper, not a fitting step. It reports both the
        point-error trade-off and the marginal-distribution match, so that the
        cost of restoring the marginal is visible:

        ``mse_before`` / ``mse_after`` / ``mse_ratio``
            Mean squared error of the raw and transformed predictions.
        ``wasserstein1_before`` / ``wasserstein1_after``
            Empirical 1-Wasserstein distance between the predictions and the
            targets (equal sample sizes; sorted-pair average absolute gap).
        ``var_ratio_before`` / ``var_ratio_after``
            ``Var(pred) / Var(target)`` before and after the transform; both
            methods match it to about one after fitting.

        Non-finite pairs are dropped. The ``eps`` floor is used for ratios
        whose denominator is zero, so every returned value is finite.
        """
        if pred_mean.numel() != target.numel():
            raise ValueError("pred_mean and target must contain the same number of elements")
        raw = pred_mean.detach().reshape(-1)
        y = target.detach().reshape(-1)
        keep = torch.isfinite(raw) & torch.isfinite(y)
        if int(keep.sum()) < 2:
            raise ValueError("need at least two finite samples to describe")
        raw = raw[keep].double()
        y = y[keep].double()
        after = self.transform(raw).double()

        mse_before = float(torch.mean((raw - y) ** 2))
        mse_after = float(torch.mean((after - y) ** 2))
        w1_before = float(torch.mean((raw.sort().values - y.sort().values).abs()))
        w1_after = float(torch.mean((after.sort().values - y.sort().values).abs()))
        var_y = float(y.var(unbiased=False))
        var_before = float(raw.var(unbiased=False))
        var_after = float(after.var(unbiased=False))
        denom = max(var_y, self.eps)
        return {
            "mse_before": mse_before,
            "mse_after": mse_after,
            "mse_ratio": mse_after / max(mse_before, self.eps),
            "wasserstein1_before": w1_before,
            "wasserstein1_after": w1_after,
            "var_ratio_before": var_before / denom,
            "var_ratio_after": var_after / denom,
        }


__all__ = [
    "VarianceTemperatureScaler",
    "IsotonicMeanCalibrator",
    "PITCalibrator",
    "VarianceRestoringCalibrator",
]
