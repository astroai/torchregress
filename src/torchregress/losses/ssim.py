"""
Structural similarity (SSIM) for 1-D sequences: spectra, light curves, profiles.

SSIM (Wang et al. 2004) compares two signals window by window through their
local means, variances and covariance. This module evaluates it along the last
dimension of ``[..., N]`` tensors, batched over the leading dimensions, with the
two things image SSIM implementations do not offer and sequence data needs:

* masks -- local moments are computed over the valid samples only (normalised
  convolution, ``conv(m x) / conv(m)``), windows with too little valid support
  are excluded, and non-finite values at masked samples never enter the result;
* per-sample weights (e.g. inverse variances) that enter the local moments the
  same way, ``conv(w m x) / conv(w m)``; a zero weight is a masked sample.

Without mask or weights the local SSIM map equals ``torchmetrics``' 2-D SSIM on
``[R, 1, N, 1]`` inputs with a ``(k, 1)`` kernel; ``padding="reflect"`` also
reproduces its edge handling and score exactly.

SSIM is a similarity index, not a likelihood: it has no noise model, depends
on ``data_range``, and its structure term is a windowed correlation, so it is
blind to local gain and offset errors. For a likelihood that is *exactly*
invariant to a chosen set of smooth gain/offset/baseline distortions, see
:func:`torchregress.losses.marginalize_linear_nuisance`.

References:
    Wang, Bovik, Sheikh & Simoncelli (2004): "Image quality assessment: from
    error visibility to structural similarity." IEEE TIP 13(4): 600-612.
    Knutsson & Westin (1993): "Normalized and differential convolution."
    Proc. CVPR 1993: 515-523.
"""

from __future__ import annotations

from typing import Any, Literal, NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..utils.validation import validate_reduction
from .base import RegressionLoss
from .loss_registry import register_regression_loss

SSIMPadding = Literal["valid", "same", "reflect"]
_PADDINGS = ("valid", "same", "reflect")


class _SSIMRows(NamedTuple):
    score: Tensor  # [R] mean local SSIM over valid windows (0 where no window)
    has_window: Tensor  # [R] bool
    ssim_map: Tensor  # [R, N_out], 0 at excluded windows
    window_valid: Tensor  # [R, N_out] bool


def _window(
    kernel_size: int, sigma: Optional[float], dtype: torch.dtype, device: torch.device
) -> Tensor:
    """Normalised window: Gaussian (same construction as torchmetrics) or uniform."""
    if sigma is None:
        return torch.full((kernel_size,), 1.0 / kernel_size, dtype=dtype, device=device)
    half = (kernel_size - 1) / 2.0
    dist = torch.arange(-half, half + 0.5, 1.0, dtype=dtype, device=device)
    gauss = torch.exp(-((dist / sigma) ** 2) / 2.0)
    return gauss / gauss.sum()


def _validate_options(
    kernel_size: int,
    sigma: Optional[float],
    k1: float,
    k2: float,
    padding: str,
    min_coverage: float,
) -> None:
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
        raise TypeError(f"kernel_size must be an int, got {type(kernel_size).__name__}")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    if sigma is not None and not float(sigma) > 0.0:
        raise ValueError(f"sigma must be > 0 (or None for a uniform window), got {sigma}")
    if not (float(k1) > 0.0 and float(k2) > 0.0):
        raise ValueError(f"k1 and k2 must be > 0, got k1={k1}, k2={k2}")
    if padding not in _PADDINGS:
        raise ValueError(f"padding must be one of {_PADDINGS}, got {padding!r}")
    if not 0.0 < float(min_coverage) <= 1.0:
        raise ValueError(f"min_coverage must be in (0, 1], got {min_coverage}")


def _ssim_rows(
    y_pred: Tensor,
    target: Tensor,
    *,
    kernel_size: int,
    sigma: Optional[float],
    data_range: float | Tensor | None,
    k1: float,
    k2: float,
    padding: str,
    min_coverage: float,
    mask: Optional[Tensor],
    weights: Optional[Tensor],
) -> tuple[_SSIMRows, tuple[int, ...]]:
    """Row-wise SSIM on ``[R, N]`` views; returns the rows and the leading shape."""
    _validate_options(kernel_size, sigma, k1, k2, padding, min_coverage)
    if y_pred.shape != target.shape:
        raise ValueError(
            f"target shape {tuple(target.shape)} must match y_pred shape {tuple(y_pred.shape)}"
        )
    if y_pred.dim() < 1:
        raise ValueError("y_pred must have at least one (sequence) dimension")
    if not (y_pred.is_floating_point() and target.is_floating_point()):
        raise TypeError("y_pred and target must be floating-point tensors")
    dtype = torch.promote_types(y_pred.dtype, target.dtype)
    device = y_pred.device
    shape = y_pred.shape
    lead, n = shape[:-1], shape[-1]
    pad = (kernel_size - 1) // 2
    if padding == "valid" and n < kernel_size:
        raise ValueError(
            f"padding='valid' needs sequence length >= kernel_size ({kernel_size}), got {n}"
        )
    if padding == "reflect" and n <= pad:
        raise ValueError(f"padding='reflect' needs sequence length > {pad}, got {n}")

    x = y_pred.to(dtype).reshape(-1, n)
    y = target.to(device=device, dtype=dtype).reshape(-1, n)
    rows = x.shape[0]

    valid: Optional[Tensor] = None
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("mask must have dtype torch.bool")
        valid = torch.broadcast_to(mask.to(device), shape).reshape(rows, n)
    amp: Optional[Tensor] = None  # per-sample moment weight w * m
    if weights is not None:
        w = torch.broadcast_to(weights.to(device=device, dtype=dtype), shape).reshape(rows, n)
        if valid is not None:
            w = torch.where(valid, w, torch.zeros_like(w))
        if not bool(torch.all(torch.isfinite(w) & (w >= 0))):
            raise ValueError("weights must be finite and >= 0 on unmasked samples")
        # A zero weight (e.g. zero inverse variance at a bad pixel) is a masked
        # sample: coverage, data range, offset and non-finite masking all use
        # this one validity (w is already 0 where the mask is False).
        valid = w > 0
        amp = w
    elif valid is not None:
        amp = valid.to(dtype)

    if valid is not None:
        zero = torch.zeros((), dtype=dtype, device=device)
        x = torch.where(valid, x, zero)
        y = torch.where(valid, y, zero)

    # Per-row data range c1/c2. Default: dynamic range of the target's valid samples.
    if data_range is None:
        y_det = y.detach()
        if valid is None:
            span = y_det.amax(dim=-1) - y_det.amin(dim=-1)
        else:
            inf = torch.full((), float("inf"), dtype=dtype, device=device)
            hi = torch.where(valid, y_det, -inf).amax(dim=-1)
            lo = torch.where(valid, y_det, inf).amin(dim=-1)
            span = torch.where(valid.any(dim=-1), hi - lo, torch.ones_like(hi))
        dr = span.unsqueeze(-1)
    elif isinstance(data_range, Tensor):
        dr_t = data_range.to(device=device, dtype=dtype)
        if not bool(torch.all(torch.isfinite(dr_t) & (dr_t > 0))):
            raise ValueError("data_range must be finite and > 0")
        dr = torch.broadcast_to(dr_t, lead).reshape(rows, 1)
    else:
        if not float(data_range) > 0.0:
            raise ValueError(f"data_range must be > 0, got {data_range}")
        dr = torch.full((rows, 1), float(data_range), dtype=dtype, device=device)
    c1 = (k1 * dr) ** 2
    c2 = (k2 * dr) ** 2

    # Shift both signals by one per-row constant (the target's valid mean). The
    # covariance terms are shift invariant, the luminance term is restored below;
    # this only removes cancellation in E[x^2] - E[x]^2.
    if valid is None:
        offset = y.detach().mean(dim=-1, keepdim=True)
    else:
        count = valid.sum(dim=-1, keepdim=True).to(dtype)
        offset = y.detach().sum(dim=-1, keepdim=True) / count.clamp_min(1.0)
    xs = x - offset
    ys = y - offset
    if valid is not None:
        zero = torch.zeros((), dtype=dtype, device=device)
        xs = torch.where(valid, xs, zero)
        ys = torch.where(valid, ys, zero)

    # Edge handling: "same" treats the sequence as masked beyond its ends.
    support = valid.to(dtype) if valid is not None else None
    if padding == "same":
        if amp is None:
            amp = torch.ones_like(xs)
        if support is None:
            support = torch.ones_like(xs)
    if padding != "valid":
        mode = "reflect" if padding == "reflect" else "constant"

        def _pad(t: Tensor) -> Tensor:
            return F.pad(t.unsqueeze(1), (pad, pad), mode=mode).squeeze(1)

        xs, ys = _pad(xs), _pad(ys)
        amp = _pad(amp) if amp is not None else None
        support = _pad(support) if support is not None else None

    if amp is None:
        channels = [xs, ys, xs * xs, ys * ys, xs * ys]
    else:
        ax, ay = amp * xs, amp * ys
        channels = [ax, ay, ax * xs, ay * ys, ax * ys, amp]
    if support is not None and weights is not None:
        channels.append(support)
    # One depthwise (grouped) convolution over all moment channels of all rows.
    # A [C*R, 1, L] batch with a single input channel would take PyTorch's CPU
    # im2col path, which copies the input kernel_size times.
    kernel = _window(kernel_size, sigma, dtype, device).view(1, 1, kernel_size)
    stacked = torch.stack(channels, dim=0).reshape(1, len(channels) * rows, -1)
    n_conv = stacked.shape[1]
    moments = F.conv1d(stacked, kernel.expand(n_conv, 1, kernel_size), groups=n_conv)
    moments = moments.reshape(len(channels), rows, -1)

    if amp is None:
        mass = None
        mu_x, mu_y, e_xx, e_yy, e_xy = moments.unbind(0)
        window_valid = torch.ones_like(mu_x, dtype=torch.bool)
    else:
        mass = moments[5]
        # Normalised convolution: windows need enough valid kernel mass (the
        # coverage; 1 when nothing is masked) and a positive weighted mass.
        # Excluded windows use a safe denominator.
        window_valid = mass > 0
        if support is not None:
            cover = moments[6] if weights is not None else mass
            window_valid = window_valid & (cover >= min_coverage)
        safe = torch.where(window_valid, mass, torch.ones_like(mass))
        mu_x, mu_y, e_xx, e_yy, e_xy = (moments[i] / safe for i in range(5))

    var_x = (e_xx - mu_x * mu_x).clamp_min(0.0)
    var_y = (e_yy - mu_y * mu_y).clamp_min(0.0)
    cov_xy = e_xy - mu_x * mu_y
    lum_x = mu_x + offset
    lum_y = mu_y + offset
    ssim_map = ((2.0 * lum_x * lum_y + c1) * (2.0 * cov_xy + c2)) / (
        (lum_x * lum_x + lum_y * lum_y + c1) * (var_x + var_y + c2)
    )
    if mass is not None:
        ssim_map = torch.where(window_valid, ssim_map, torch.zeros_like(ssim_map))
    n_windows = window_valid.sum(dim=-1)
    has_window = n_windows > 0
    score = ssim_map.sum(dim=-1) / n_windows.clamp_min(1).to(dtype)
    return _SSIMRows(score, has_window, ssim_map, window_valid), lead


def structural_similarity_1d(
    y_pred: Tensor,
    target: Tensor,
    *,
    kernel_size: int = 11,
    sigma: Optional[float] = 1.5,
    data_range: float | Tensor | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
    padding: SSIMPadding = "valid",
    min_coverage: float = 0.5,
    mask: Optional[Tensor] = None,
    weights: Optional[Tensor] = None,
    reduction: str = "mean",
    return_map: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Structural similarity of 1-D sequences along the last dimension.

    For each window position :math:`j` with normalised window :math:`g` and
    per-sample moment weights :math:`a_i = w_i m_i` (mask :math:`m`, weights
    :math:`w`, both 1 by default), the local moments are normalised convolutions

    .. math::

        \mu_x = \frac{\sum_i g_i a_{j+i} x_{j+i}}{\sum_i g_i a_{j+i}},\quad
        \sigma_x^2 = \frac{\sum_i g_i a_{j+i} x_{j+i}^2}{\sum_i g_i a_{j+i}} - \mu_x^2,\quad
        \sigma_{xy} = \frac{\sum_i g_i a_{j+i} x_{j+i} y_{j+i}}{\sum_i g_i a_{j+i}} - \mu_x\mu_y,

    and

    .. math::

        \mathrm{SSIM}_j = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}
        {(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)},\qquad
        C_1 = (k_1 L)^2,\; C_2 = (k_2 L)^2.

    A row's score is the mean of :math:`\mathrm{SSIM}_j` over its valid windows.
    Moments are population (biased) moments, as in ``torchmetrics``.

    Args:
        y_pred: Predicted sequences ``[..., N]``.
        target: Reference sequences, same shape.
        kernel_size: Odd window length.
        sigma: Gaussian window standard deviation in samples; ``None`` for a
            uniform window. The Gaussian is truncated to ``kernel_size``.
        data_range: Dynamic range :math:`L` in the stabilisers :math:`C_1, C_2`:
            a positive float, a positive tensor broadcastable to the leading
            shape ``[...]`` (one value per row), or ``None`` for the per-row
            range ``max - min`` of the target's valid samples. A constant row
            with ``data_range=None`` has :math:`L = 0` and gives ``NaN``.
        k1, k2: SSIM stabiliser constants.
        padding: Edge handling. ``"valid"`` (default): only windows entirely
            inside the sequence (``N - kernel_size + 1`` positions). ``"same"``:
            ``N`` positions; the sequence is treated as masked beyond its ends,
            so edge windows are truncated and renormalised like masked windows.
            ``"reflect"``: ``N`` positions on a reflect-padded sequence, the
            convention of ``torchmetrics`` 1.9.
        min_coverage: A window is used only if its valid samples carry at least
            this fraction of the window mass,
            :math:`\sum_i g_i \mathbb 1[w_{j+i} m_{j+i} > 0]`, and their
            weighted mass is positive. Irrelevant without mask or zero weights
            (except for ``padding="same"`` at the edges).
        mask: Optional boolean mask (``True`` = valid), broadcastable to
            ``y_pred``. Masked samples never enter a moment; non-finite values
            there are ignored.
        weights: Optional per-sample weights ``>= 0`` (e.g. inverse variances),
            broadcastable to ``y_pred``, entering every local moment as above.
            A sample with zero weight is treated exactly as a masked sample
            (no moment, no coverage, no data range; non-finite values there
            are ignored). A weight that is constant over a window cancels;
            weights only reweight samples *within* windows. They are not a
            noise model.
        reduction: ``"mean"`` (over rows with at least one valid window),
            ``"sum"`` (same rows), or ``"none"`` (one score per row, shape
            ``[...]``; ``NaN`` for rows without a valid window).
        return_map: Also return the local SSIM map ``[..., N_out]`` with ``NaN``
            at excluded windows.

    Returns:
        The SSIM score (higher is more similar; 1 for identical sequences), and
        the local map if ``return_map``.

    Raises:
        ValueError: On invalid options, shapes, non-positive ``data_range`` or
            negative / non-finite weights on valid samples.
    """
    reduction = validate_reduction(reduction)
    rows, lead = _ssim_rows(
        y_pred,
        target,
        kernel_size=kernel_size,
        sigma=sigma,
        data_range=data_range,
        k1=k1,
        k2=k2,
        padding=padding,
        min_coverage=min_coverage,
        mask=mask,
        weights=weights,
    )
    nan = torch.full((), float("nan"), dtype=rows.score.dtype, device=rows.score.device)
    if reduction == "none":
        out = torch.where(rows.has_window, rows.score, nan).reshape(lead)
    else:
        kept = torch.where(rows.has_window, rows.score, torch.zeros_like(rows.score))
        total = kept.sum()
        if reduction == "sum":
            out = total
        else:
            count = rows.has_window.sum()
            out = torch.where(count > 0, total / count.clamp_min(1).to(total.dtype), nan)
    if not return_map:
        return out
    ssim_map = torch.where(rows.window_valid, rows.ssim_map, nan)
    return out, ssim_map.reshape(*lead, ssim_map.shape[-1])


@register_regression_loss("structural_similarity_1d")
class StructuralSimilarity1dLoss(RegressionLoss):
    r"""``1 - SSIM`` of 1-D sequences, one value per row of ``[..., N]`` inputs.

    See :func:`structural_similarity_1d` for the definition and arguments. The
    per-row loss is :math:`1 - \overline{\mathrm{SSIM}}` over the row's valid
    windows; rows without a valid window (fully masked) are dropped from
    ``"mean"``/``"sum"`` and are 0 under ``reduction="none"``.

    ``weights`` follows :func:`structural_similarity_1d`: per-sample weights
    inside the local moments, not per-row weights of the reduction. For
    per-row weights use ``reduction="none"`` and weight the result.

    SSIM has no noise model and is insensitive to local gain and offset errors
    (its structure term is a windowed correlation); see
    :class:`MarginalizedNuisanceGaussianNLLLoss` for the likelihood counterpart.
    """

    def __init__(
        self,
        kernel_size: int = 11,
        sigma: Optional[float] = 1.5,
        data_range: Optional[float] = None,
        k1: float = 0.01,
        k2: float = 0.03,
        padding: SSIMPadding = "valid",
        min_coverage: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        _validate_options(kernel_size, sigma, k1, k2, padding, min_coverage)
        if data_range is not None and not float(data_range) > 0.0:
            raise ValueError(f"data_range must be > 0, got {data_range}")
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.padding: SSIMPadding = padding
        self.min_coverage = min_coverage

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Return the reduced ``1 - SSIM`` loss.

        Args:
            y_pred: Predicted sequences ``[..., N]``.
            target: Reference sequences, same shape.
            mask: Optional boolean sample mask broadcastable to ``y_pred``.
            weights: Optional per-sample moment weights broadcastable to ``y_pred``.
        """
        rows, lead = _ssim_rows(
            y_pred,
            target,
            kernel_size=self.kernel_size,
            sigma=self.sigma,
            data_range=self.data_range,
            k1=self.k1,
            k2=self.k2,
            padding=self.padding,
            min_coverage=self.min_coverage,
            mask=mask,
            weights=weights,
        )
        loss = (1.0 - rows.score).reshape(lead)
        return self._reduce(loss, rows.has_window.reshape(lead))
