"""Training helpers for :class:`~torchregress.models.TabularMLP`."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import torch
from torch import nn

from ..losses.gaussian import GaussianNLLLoss
from .preprocessing import ArrayLike, TabularPreprocessor

OutputLayout = Literal["auto", "location", "gaussian", "variance", "none"]


def _as_2d_tensor(a: ArrayLike, dtype: torch.dtype, name: str) -> torch.Tensor:
    t = torch.as_tensor(a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a)
    t = t.to(dtype)
    if t.ndim == 1:
        t = t[:, None]
    if t.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {tuple(t.shape)}")
    return t


@dataclass
class TabularFit:
    """Result of :func:`fit_tabular`: trained model, preprocessing and history.

    Attributes
    ----------
    model : torch.nn.Module
        The trained model, with the best-validation weights restored, in eval mode.
    preprocessor : TabularPreprocessor or None
        Fitted input preprocessor (``None`` when ``preprocessor=False``).
    target_mean, target_std : torch.Tensor
        Per-target statistics used for target standardisation (0 and 1 when
        standardisation is off).
    output_layout : str
        How :meth:`predict` maps head outputs back to the original target scale.
    history : dict
        ``train_loss``, ``val_loss`` and ``lr`` per epoch. Losses are in the
        standardised target space.
    best_epoch : int
        Zero-based epoch of the restored weights.
    best_val_loss : float
        Validation loss at ``best_epoch`` (training loss if no validation data).
    stopped_early : bool
        Whether patience ran out before ``epochs``.
    """

    model: nn.Module
    preprocessor: Optional[TabularPreprocessor]
    target_mean: torch.Tensor
    target_std: torch.Tensor
    output_layout: str
    history: dict[str, list[float]] = field(default_factory=dict)
    best_epoch: int = 0
    best_val_loss: float = float("nan")
    stopped_early: bool = False

    @torch.no_grad()
    def predict(self, X: ArrayLike, batch_size: int = 8192) -> torch.Tensor:
        """Raw head outputs for ``X`` in the original target scale.

        Location outputs (point predictions, quantiles) are mapped to
        ``mu + sigma * out``. With the ``"gaussian"`` layout ``[mean, log_variance]``
        the mean is mapped likewise and ``log_variance`` is shifted by
        ``2 log sigma``. The result feeds the same loss function used for training.
        Flat outputs with more than one target are read as consecutive blocks of ``D``
        columns. The ``"variance"`` layout (``GaussianNLLLoss(log_variance=False)``)
        scales the variance block by ``sigma ** 2``.

        Parameters
        ----------
        X : numpy.ndarray or torch.Tensor
            Raw (unpreprocessed) features.
        batch_size : int
            Inference chunk size.

        Returns
        -------
        torch.Tensor
            Shape ``(n_samples, out_features)``.
        """
        param = next(self.model.parameters())
        Xt = self.preprocessor.transform(X) if self.preprocessor is not None else X
        Xt = _as_2d_tensor(Xt, param.dtype, "X").to(param.device)
        was_training = self.model.training
        self.model.eval()
        try:
            out = torch.cat(
                [self.model(Xt[i : i + batch_size]) for i in range(0, len(Xt), batch_size)]
            )
        finally:
            self.model.train(was_training)
        mu = self.target_mean.to(out)
        sd = self.target_std.to(out)
        d = mu.numel()
        if self.output_layout == "none":
            return out
        n, k = out.shape
        if k % d != 0:
            raise ValueError(f"out_features={k} is not a multiple of the {d} targets")
        out = out.reshape(n, k // d, d)
        if self.output_layout == "gaussian":
            out = torch.cat([out[:, :1] * sd + mu, out[:, 1:] + 2.0 * torch.log(sd)], dim=1)
        elif self.output_layout == "variance":
            out = torch.cat([out[:, :1] * sd + mu, out[:, 1:] * sd**2], dim=1)
        else:
            out = out * sd + mu
        return out.reshape(n, k)


@dataclass
class TabularEnsembleFit:
    """A list of :class:`TabularFit` members trained from different seeds.

    Attributes
    ----------
    members : list of TabularFit
        The individual fits.
    """

    members: list[TabularFit]

    def predict_members(self, X: ArrayLike, batch_size: int = 8192) -> torch.Tensor:
        """Member outputs stacked as ``(n_members, n_samples, out_features)``."""
        return torch.stack([m.predict(X, batch_size) for m in self.members])

    def predict(self, X: ArrayLike, batch_size: int = 8192) -> torch.Tensor:
        """Average of the member head outputs, shape ``(n_samples, out_features)``.

        Averaging is exact for point and quantile heads. For Gaussian heads it
        averages the mean and the log-variance (not the mixture variance); use
        :meth:`predict_members` and a mixture when the epistemic spread matters.
        """
        return self.predict_members(X, batch_size).mean(dim=0)


def _resolve_layout(loss_fn: Callable[..., torch.Tensor], layout: OutputLayout) -> str:
    if layout != "auto":
        return layout
    if isinstance(loss_fn, GaussianNLLLoss):
        return "gaussian" if loss_fn.log_variance else "variance"
    return "location"


def fit_tabular(
    model: nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    X_train: ArrayLike,
    y_train: ArrayLike,
    *,
    X_val: Optional[ArrayLike] = None,
    y_val: Optional[ArrayLike] = None,
    val_fraction: float = 0.15,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 2e-3,
    weight_decay: float = 1e-2,
    patience: int = 20,
    seed: int = 0,
    device: Union[str, torch.device, None] = None,
    scheduler: Literal["onecycle", "cosine", "none"] = "onecycle",
    grad_clip: Optional[float] = 1.0,
    standardize_target: bool = True,
    output_layout: OutputLayout = "auto",
    preprocessor: Union[TabularPreprocessor, None, bool] = None,
) -> TabularFit:
    """Train ``model`` with AdamW, a one-cycle schedule and early stopping.

    Recipe (after RealMLP, Holzmueller et al. 2024): inputs are robust-scaled and
    smooth-clipped by a :class:`TabularPreprocessor` fitted on the training rows;
    targets are standardised; AdamW with decoupled weight decay and a one-cycle
    (warm-up then cosine) learning-rate schedule over ``epochs``; gradient-norm
    clipping; validation loss each epoch with the best weights restored at the end.

    ``loss_fn`` is called as ``loss_fn(model(x), y)`` with ``y`` of shape
    ``(batch, D)`` in standardised units, so any torchregress loss works as long as
    the model's ``out_features`` matches the loss (``2 * D`` for
    :class:`~torchregress.losses.GaussianNLLLoss`, ``len(quantiles)`` for
    :class:`~torchregress.losses.MultiQuantileLoss` with ``D = 1``).

    The global torch RNG is left untouched: dropout, shuffling and the validation
    split are driven by ``seed`` in a forked RNG state. The model's initial weights
    are whatever it was constructed with, so call ``torch.manual_seed`` before
    building it for fully reproducible runs.

    Parameters
    ----------
    model : torch.nn.Module
        Maps ``(batch, in_features)`` to ``(batch, out_features)``; trained in place.
    loss_fn : callable
        ``loss_fn(y_pred, target) -> scalar tensor``.
    X_train, y_train : array-like
        Raw features ``(n, F)`` and targets ``(n,)`` or ``(n, D)``.
    X_val, y_val : array-like, optional
        Explicit validation set. If omitted, ``val_fraction`` of the training rows is
        held out (random, seeded).
    val_fraction : float, default 0.15
        Held-out share when no explicit validation set is given. ``0`` disables
        validation: the model trains for all ``epochs`` and the final weights are kept.
    epochs, batch_size, lr, weight_decay : float or int
        AdamW budget. ``lr`` is the peak learning rate.
    patience : int, default 20
        Epochs without validation improvement before stopping. With a one-cycle
        schedule a stop leaves the learning rate high; the best weights are restored.
    seed : int, default 0
        Seed for the split, shuffling and dropout.
    device : str or torch.device, optional
        Device to train on; defaults to the model's current device.
    scheduler : {"onecycle", "cosine", "none"}
        Learning-rate schedule.
    grad_clip : float or None
        Max gradient norm.
    standardize_target : bool, default True
        Standardise targets with the training mean and std.
    output_layout : {"auto", "location", "gaussian", "variance", "none"}
        How outputs map back to the original scale in :meth:`TabularFit.predict`.
        ``"auto"`` picks ``"gaussian"`` for :class:`~torchregress.losses.GaussianNLLLoss`
        (``"variance"`` with ``log_variance=False``) and ``"location"`` otherwise
        (point and quantile heads). Use
        ``standardize_target=False`` for heads that are neither (for example
        evidential or mixture heads).
    preprocessor : TabularPreprocessor, None or False
        ``None`` builds and fits a default one on the training rows; a fitted
        instance is used as is; an unfitted one is fitted on the training rows;
        ``False`` skips preprocessing (inputs must already be numeric and finite).

    Returns
    -------
    TabularFit
        Trained model (best weights restored), preprocessing, history and a
        :meth:`TabularFit.predict` method.

    Examples
    --------
    >>> import numpy as np
    >>> from torchregress.losses import WeightedMSELoss
    >>> from torchregress.models import TabularMLP, fit_tabular
    >>> X = np.random.default_rng(0).normal(size=(200, 4)); y = X[:, 0] ** 2
    >>> fit = fit_tabular(TabularMLP(4, 1, hidden=(16,)), WeightedMSELoss(), X, y, epochs=2)
    >>> fit.predict(X).shape
    torch.Size([200, 1])
    """
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be >= 1")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if (X_val is None) != (y_val is None):
        raise ValueError("pass both X_val and y_val, or neither")

    param = next(model.parameters())
    dev = torch.device(device) if device is not None else param.device
    model.to(dev)
    dtype = param.dtype

    # Split first so the preprocessor and target statistics never see validation rows.
    n = len(X_train)
    if len(y_train) != n:
        raise ValueError("X_train and y_train have different lengths")
    gen = torch.Generator().manual_seed(seed)
    if X_val is None and val_fraction > 0:
        n_val = max(1, int(round(n * val_fraction)))
        if n_val >= n:
            raise ValueError("val_fraction leaves no training rows")
        perm = torch.randperm(n, generator=gen).numpy()
        va_idx, tr_idx = perm[:n_val], perm[n_val:]
        Xtr_raw, ytr_raw = _take(X_train, tr_idx), _take(y_train, tr_idx)
        Xva_raw, yva_raw = _take(X_train, va_idx), _take(y_train, va_idx)
    else:
        Xtr_raw, ytr_raw, Xva_raw, yva_raw = X_train, y_train, X_val, y_val

    prep: Optional[TabularPreprocessor]
    if preprocessor is False:
        prep = None
    else:
        prep = preprocessor if isinstance(preprocessor, TabularPreprocessor) else None
        prep = prep if prep is not None else TabularPreprocessor()
        if not prep.is_fitted:
            prep.fit(Xtr_raw)
    Xtr = _prepare_x(prep, Xtr_raw, dtype, dev)
    ytr = _as_2d_tensor(ytr_raw, dtype, "y_train").to(dev)
    in_features = getattr(model, "in_features", None)
    if in_features is not None and in_features != Xtr.shape[1]:
        raise ValueError(
            f"model expects {in_features} input features but the preprocessed data has "
            f"{Xtr.shape[1]} (use TabularPreprocessor.n_features_out_)"
        )
    if not torch.isfinite(ytr).all():
        raise ValueError("y_train contains NaN or infinite values")

    d = ytr.shape[1]
    if standardize_target:
        t_mean = ytr.mean(0)
        t_std = (
            ytr.std(0, unbiased=False).clamp_min(1e-12)
            if len(ytr) > 1
            else torch.ones(d, dtype=dtype, device=dev)
        )
    else:
        t_mean, t_std = (
            torch.zeros(d, dtype=dtype, device=dev),
            torch.ones(d, dtype=dtype, device=dev),
        )
    ytr = (ytr - t_mean) / t_std
    has_val = Xva_raw is not None and yva_raw is not None
    if Xva_raw is not None and yva_raw is not None:
        Xva = _prepare_x(prep, Xva_raw, dtype, dev)
        yva = (_as_2d_tensor(yva_raw, dtype, "y_val").to(dev) - t_mean) / t_std
        if Xva.shape[1] != Xtr.shape[1]:
            raise ValueError("validation features have a different width than training")

    layout = _resolve_layout(loss_fn, output_layout)
    if not standardize_target and output_layout == "auto" and layout == "location":
        layout = "none"  # identity mapping; keep a Gaussian layout so wrappers can use it
    steps_per_epoch = math.ceil(len(Xtr) / batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched: Optional[torch.optim.lr_scheduler.LRScheduler]
    total_steps = epochs * steps_per_epoch
    if scheduler == "onecycle" and total_steps < 4:
        sched = None  # too few steps for a warm-up and an annealing phase
    elif scheduler == "onecycle":
        # OneCycleLR needs at least one full step in its warm-up phase
        # (pct_start * total_steps > 1), else it divides by zero.
        pct_start = min(0.5, max(0.1, 2.0 / total_steps))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=total_steps, pct_start=pct_start
        )
    elif scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    elif scheduler == "none":
        sched = None
    else:
        raise ValueError(f"unknown scheduler {scheduler!r}")

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
    best = float("inf")
    best_epoch = 0
    best_state: Optional[dict[str, torch.Tensor]] = None
    bad = 0
    stopped = False
    fork_devices = [dev] if dev.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        for epoch in range(epochs):
            model.train()
            order = torch.randperm(len(Xtr), generator=gen).to(dev)
            total = 0.0
            for i in range(0, len(Xtr), batch_size):
                idx = order[i : i + batch_size]
                loss = loss_fn(model(Xtr[idx]), ytr[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                if sched is not None:
                    sched.step()
                total += float(loss.detach()) * len(idx)
            train_loss = total / len(Xtr)
            history["train_loss"].append(train_loss)
            history["lr"].append(opt.param_groups[0]["lr"])
            if has_val:
                model.eval()
                with torch.no_grad():
                    val_loss = _batched_loss(model, loss_fn, Xva, yva, 8192)
            else:
                val_loss = train_loss
            history["val_loss"].append(val_loss)
            if not math.isfinite(val_loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            if val_loss < best:
                best, best_epoch, bad = val_loss, epoch, 0
                if has_val:
                    best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if has_val and bad >= patience:
                    stopped = True
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = len(history["val_loss"]) - 1
        best = history["val_loss"][-1]
    model.eval()
    return TabularFit(
        model=model,
        preprocessor=prep,
        target_mean=t_mean.detach().cpu(),
        target_std=t_std.detach().cpu(),
        output_layout=layout,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best,
        stopped_early=stopped,
    )


def fit_tabular_ensemble(
    make_model: Callable[[int], nn.Module],
    loss_fn: Callable[..., torch.Tensor],
    X_train: ArrayLike,
    y_train: ArrayLike,
    *,
    n_members: int = 5,
    seed: int = 0,
    **fit_kwargs: Any,
) -> TabularEnsembleFit:
    """Train ``n_members`` independent models (seeds ``seed, seed + 1, ...``).

    Each member gets its own initialisation, validation split, shuffling and
    dropout stream, so the ensemble also averages over the split. The preprocessor
    is fitted once on all of ``X_train`` and shared.

    Parameters
    ----------
    make_model : callable
        ``make_model(in_features) -> nn.Module``, for example
        ``lambda d: TabularMLP(d, 1)``.
    loss_fn : callable
        See :func:`fit_tabular`.
    X_train, y_train : array-like
        Training data.
    n_members : int, default 5
        Number of members.
    seed : int, default 0
        Seed of the first member.
    **fit_kwargs
        Forwarded to :func:`fit_tabular` (not ``seed``, ``preprocessor``).

    Returns
    -------
    TabularEnsembleFit
        Members with :meth:`TabularEnsembleFit.predict`.
    """
    if n_members < 1:
        raise ValueError("n_members must be >= 1")
    prep = TabularPreprocessor().fit(X_train)
    members = []
    for m in range(n_members):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + m)
            model = make_model(prep.n_features_out_)
        members.append(
            fit_tabular(
                model,
                loss_fn,
                X_train,
                y_train,
                seed=seed + m,
                preprocessor=prep,
                **fit_kwargs,
            )
        )
    return TabularEnsembleFit(members)


def _take(a: ArrayLike, idx: np.ndarray) -> ArrayLike:
    return a[torch.as_tensor(idx)] if isinstance(a, torch.Tensor) else np.asarray(a)[idx]


def _prepare_x(
    prep: Optional[TabularPreprocessor], X: ArrayLike, dtype: torch.dtype, dev: torch.device
) -> torch.Tensor:
    Z = prep.transform(X) if prep is not None else X
    t = _as_2d_tensor(Z, dtype, "X").to(dev)
    if not torch.isfinite(t).all():
        raise ValueError("features contain NaN or inf; use the default preprocessor to impute")
    return t


def _batched_loss(
    model: nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    X: torch.Tensor,
    y: torch.Tensor,
    batch: int,
) -> float:
    total = 0.0
    for i in range(0, len(X), batch):
        total += float(loss_fn(model(X[i : i + batch]), y[i : i + batch])) * len(X[i : i + batch])
    return total / len(X)
