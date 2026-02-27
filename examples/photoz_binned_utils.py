"""Example-local helpers for binned photo-z PDF training/evaluation.

These utilities intentionally live in `examples/` to keep specialized
ordered-bin experimentation out of the core public torchregress API.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def make_bins_from_train_targets(
    targets: torch.Tensor,
    *,
    n_bins: int,
    strategy: str = "quantile",
    min_width: float = 1e-4,
) -> torch.Tensor:
    """Build monotonic bin edges from training targets."""
    y = targets.reshape(-1).float()
    if y.numel() == 0:
        raise ValueError("targets must contain at least one value")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    if strategy == "quantile":
        q = torch.linspace(0.0, 1.0, n_bins + 1, device=y.device)
        edges = torch.quantile(y, q)
    elif strategy == "uniform":
        lo, hi = torch.min(y), torch.max(y)
        if torch.isclose(lo, hi):
            hi = lo + 1.0
        edges = torch.linspace(lo, hi, n_bins + 1, device=y.device)
    else:
        raise ValueError(f"Unknown strategy={strategy!r}. Expected 'quantile' or 'uniform'.")

    # Enforce strictly increasing edges.
    for i in range(1, edges.numel()):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + min_width
    return edges


def bin_targets(targets: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    """Map continuous targets to class indices [0, n_bins-1]."""
    y = targets.reshape(-1).float()
    edges = bin_edges.reshape(-1).float()
    if edges.numel() < 3:
        raise ValueError("bin_edges must have shape [n_bins+1] with n_bins >= 2")
    bins = torch.searchsorted(edges[1:], y)
    return bins.clamp(min=0, max=edges.numel() - 2).long()


def logits_to_pdf(logits: torch.Tensor, temperature: float | torch.Tensor = 1.0) -> torch.Tensor:
    """Convert logits [B, K] to normalized PDF probabilities."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be [B, K], got shape={list(logits.shape)}")
    t = torch.as_tensor(temperature, dtype=logits.dtype, device=logits.device).clamp_min(1e-6)
    return torch.softmax(logits / t, dim=-1)


def pdf_to_point_estimate(pdf: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    """Compute expected value from discrete PDF over bins."""
    if pdf.dim() != 2:
        raise ValueError(f"pdf must be [B, K], got shape={list(pdf.shape)}")
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return (pdf * centers.unsqueeze(0)).sum(dim=-1, keepdim=True)


def pdf_quantiles(
    pdf: torch.Tensor,
    bin_edges: torch.Tensor,
    quantiles: Iterable[float],
) -> dict[float, torch.Tensor]:
    """Approximate quantiles from binned PDF using first-crossing CDF bins."""
    if pdf.dim() != 2:
        raise ValueError(f"pdf must be [B, K], got shape={list(pdf.shape)}")
    cdf = torch.cumsum(pdf, dim=-1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    out: dict[float, torch.Tensor] = {}
    for q in quantiles:
        qf = float(q)
        qv = torch.tensor(qf, dtype=pdf.dtype, device=pdf.device)
        idx = torch.argmax((cdf >= qv).long(), dim=-1)
        out[qf] = centers[idx].unsqueeze(1)
    return out


def ordered_bin_crps_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bin_edges: torch.Tensor,
    *,
    reduction: str = "mean",
    mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    temperature: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Discrete ordered-bin CRPS based on CDF mismatch."""
    pdf = logits_to_pdf(logits, temperature=temperature)
    cdf_pred = torch.cumsum(pdf, dim=-1)
    target_bins = bin_targets(targets, bin_edges)

    k = cdf_pred.shape[-1]
    grid = torch.arange(k, device=logits.device).unsqueeze(0)
    cdf_true = (grid >= target_bins.unsqueeze(1)).to(cdf_pred.dtype)

    widths = (bin_edges[1:] - bin_edges[:-1]).to(cdf_pred.device).clamp_min(1e-8)
    loss = ((cdf_pred - cdf_true) ** 2) * widths.unsqueeze(0)
    per_sample = loss.sum(dim=-1)

    if mask is not None:
        m = mask.reshape(-1)
        per_sample = per_sample[m]
        if weights is not None:
            weights = weights.reshape(-1)[m]
    if weights is not None:
        w = weights.reshape(-1).to(per_sample.device)
        if reduction == "none":
            return per_sample * w
        if reduction == "sum":
            return torch.sum(per_sample * w)
        return torch.sum(per_sample * w) / torch.sum(w).clamp_min(1.0)

    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return torch.sum(per_sample)
    if reduction == "mean":
        return torch.mean(per_sample)
    raise ValueError(f"Unknown reduction={reduction!r}")


def apply_temperature(logits: torch.Tensor, temperature: float | torch.Tensor) -> torch.Tensor:
    """Apply scalar temperature scaling to logits."""
    t = torch.as_tensor(temperature, dtype=logits.dtype, device=logits.device).clamp_min(1e-6)
    return logits / t


def fit_temperature_scaler(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bin_edges: torch.Tensor,
    *,
    max_iter: int = 200,
    lr: float = 0.05,
) -> float:
    """Fit a scalar temperature on calibration logits/targets using NLL objective."""
    if logits.dim() != 2:
        raise ValueError("logits must have shape [B, K]")
    y_bin = bin_targets(targets, bin_edges).to(logits.device)
    log_t = torch.nn.Parameter(torch.zeros((), device=logits.device, dtype=logits.dtype))
    opt = torch.optim.Adam([log_t], lr=lr)

    for _ in range(max_iter):
        opt.zero_grad()
        temp = torch.exp(log_t).clamp_min(1e-6)
        loss = F.cross_entropy(logits / temp, y_bin, reduction="mean")
        loss.backward()
        opt.step()

    return float(torch.exp(log_t).detach().cpu().item())
