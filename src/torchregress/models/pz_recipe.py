"""Training helper for the lean two-stage PZRecipe model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from ..losses.recipe import PZRecipeLoss
from .ft_transformer import PZRecipe


@dataclass
class PZRecipeFit:
    """Fitted PZRecipe model and its quantile bin centers."""

    model: PZRecipe
    bin_centers: Tensor
    history: dict[str, list[float]]

    @torch.no_grad()
    def predict_proba(self, x: Tensor | np.ndarray, *, stage: int = 1) -> Tensor:
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.bin_centers.device)
        return self.model.eval().predict_proba(xt, stage=stage)

    @torch.no_grad()
    def predict(self, x: Tensor | np.ndarray, *, stage: int = 1) -> Tensor:
        proba = self.predict_proba(x, stage=stage)
        centers = self.bin_centers.to(proba.device)
        return proba @ centers


def _centers_from_targets(y: Tensor, n_bins: int) -> Tensor:
    q = torch.linspace(0.0, 1.0, n_bins + 2)[1:-1]
    centers = torch.quantile(y.float(), q)
    # Quantile bins can be duplicated for discrete targets; fall back to the
    # observed support/range instead of failing.
    if int(centers.unique().numel()) < n_bins:
        lo, hi = float(y.min()), float(y.max())
        if hi <= lo:
            hi = lo + 1.0
        centers = torch.linspace(lo, hi, n_bins)
    return centers


def fit_pz_recipe(
    model: PZRecipe,
    X: Tensor | np.ndarray,
    y: Tensor | np.ndarray,
    *,
    stage1_epochs: int = 100,
    stage2_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    n_stage2_bins: Optional[int] = None,
    seed: int = 0,
    device: Optional[torch.device | str] = None,
) -> PZRecipeFit:
    """Fit stage 1 (CE+CDF+RnC-N) and stage 2 (frozen-feature classifier).

    This is a compact reference implementation of the paper's two-stage
    recipe; it does not reproduce the historical TabM training schedule.  Pass
    a custom ``model.stage2_head`` or tune the stage-2 schedule for closer
    parity.
    """
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev).reshape(-1)
    n = xt.shape[0]
    if n == 0:
        raise ValueError("empty training set")
    centers = _centers_from_targets(yt.detach().cpu(), model.n_bins).to(dev)
    model.to(dev)
    history: dict[str, list[float]] = {"stage1": [], "stage2": []}
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(n, generator=generator).to(dev)
    bs = max(1, min(int(batch_size), n))
    stage1_loss = PZRecipeLoss(
        centers.detach().cpu(),
        ce_weight=0.5,
        cdf_weight=0.5,
        rnc_weight=0.1,
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(int(stage1_epochs)):
        total = 0.0
        for start in range(0, n, bs):
            idx = order[start : start + bs]
            opt.zero_grad(set_to_none=True)
            logits = model(xt[idx], stage=1)
            features = model.encode(xt[idx])
            loss = stage1_loss(logits, yt[idx], features=features)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        history["stage1"].append(total / n)

    # Stage 2: freeze the encoder; train only the stage-2 head.
    n_stage2 = int(n_stage2_bins or model.stage2_bins)
    stage2_centers = _centers_from_targets(yt.detach().cpu(), n_stage2).to(dev)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.stage1_head.parameters():
        p.requires_grad_(False)
    opt2 = torch.optim.AdamW(model.stage2_head.parameters(), lr=lr, weight_decay=weight_decay)
    model.eval()
    model.stage2_head.train()
    for _ in range(int(stage2_epochs)):
        total = 0.0
        for start in range(0, n, bs):
            idx = order[start : start + bs]
            with torch.no_grad():
                features = model.encode(xt[idx])
                labels = (
                    torch.searchsorted(stage2_centers, yt[idx]).clamp(1, stage2_centers.numel()) - 1
                )
            opt2.zero_grad(set_to_none=True)
            logits = model.stage2_head(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            opt2.step()
            total += float(loss.detach()) * len(idx)
        history["stage2"].append(total / n)
    model.eval()
    return PZRecipeFit(model=model, bin_centers=centers.detach().cpu(), history=history)
