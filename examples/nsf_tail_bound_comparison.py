"""
Comparison: NSF spline tail bound on a heavy-tailed target (needs ``zuko``).

A rational-quadratic spline flow is the identity outside ``[-B, B]``; zuko fixes
``B = 5``. After standardising a heavy-tailed target, a few training rows still
lie beyond 5 and get no learning signal from the spline. This script trains the
same conditional NSF head twice on a shared split, changing only the bound:

* **default bound** - ``create_flow_model(...)`` (``B = 5``).
* **recommended bound** - ``create_flow_model(..., bound=recommended_tail_bound(y_std))``
  (``1.25 * max |y_std|`` clipped to ``[5, 25]``).

Target: ``y = sin(2 x0) + 0.5 x1 + 0.4 * t_3`` (Student-t noise, 3 degrees of
freedom). Both rows use the training recipe of ``docs/losses/nflows.md``:
standardised target, 16 bins, 3 transforms, ``[64, 64]`` coupling networks,
Adam, early stopping on a 15% validation carve with the best weights restored.

Run::

    python examples/nsf_tail_bound_comparison.py \\
        --summary-json-path reports/nsf_tail_bound_comparison_latest.json

Notes / caveats
---------------
* One seed and one split; the NLL gap between the rows is of the order of a few
  hundredths of a nat, comparable to the seed-to-seed spread of a flow this small.
  ``docs/losses/nflows.md`` reports the 5-seed numbers (1.935 -> 1.852 for this
  kind of target, with the fixed-epoch recipe as the baseline).
* NLL is in target units (the standardisation Jacobian ``log sd(y_train)`` is
  added back). Intervals are read off 512 flow samples per test row, so coverage
  and width carry Monte Carlo noise as well.
* The default-bound row triggers ``NormalizingFlowLoss``'s one-time out-of-range
  ``UserWarning``; it is captured and recorded in the ``Notes`` column instead of
  being printed.
* Requires the optional ``zuko`` dependency (``pip install 'torchregress[flows]'``).
"""

from __future__ import annotations

import argparse
import copy
import math
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from torchregress.comparison import (
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from torchregress.losses import NormalizingFlowLoss, create_flow_model, recommended_tail_bound
from torchregress.metrics import interval_score


@dataclass(frozen=True)
class TailBoundConfig:
    seed: int = 20261014
    n_train: int = 2000
    n_test: int = 1000
    noise_scale: float = 0.4
    df: float = 3.0
    context_dim: int = 16
    hidden: int = 64
    n_transforms: int = 3
    bins: int = 16
    epochs: int = 80
    patience: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    val_fraction: float = 0.15
    n_samples: int = 512
    alpha: float = 0.1


def _simulate(cfg: TailBoundConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_train + cfg.n_test
    X = rng.uniform(-2.0, 2.0, size=(n, 2)).astype(np.float32)
    noise = cfg.noise_scale * rng.standard_t(cfg.df, size=n)
    y = (np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] + noise).astype(np.float32)
    return {
        "X_train": X[: cfg.n_train],
        "y_train": y[: cfg.n_train],
        "X_test": X[cfg.n_train :],
        "y_test": y[cfg.n_train :],
    }


def _backbone(cfg: TailBoundConfig) -> nn.Module:
    return nn.Sequential(
        nn.Linear(2, cfg.hidden),
        nn.SiLU(),
        nn.Linear(cfg.hidden, cfg.hidden),
        nn.SiLU(),
        nn.Linear(cfg.hidden, cfg.context_dim),
    )


def _train(
    model: nn.Module,
    loss_fn: NormalizingFlowLoss,
    X: torch.Tensor,
    y: torch.Tensor,
    cfg: TailBoundConfig,
) -> int:
    """Adam with early stopping on a seeded validation carve; returns the best epoch."""
    gen = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(len(X), generator=gen)
    n_val = int(cfg.val_fraction * len(X))
    va, tr = perm[:n_val], perm[n_val:]
    params = list(model.parameters()) + list(loss_fn.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)
    best, best_epoch, bad = float("inf"), 0, 0
    best_state = copy.deepcopy((model.state_dict(), loss_fn.state_dict()))
    for epoch in range(cfg.epochs):
        model.train()
        for idx in tr[torch.randperm(len(tr), generator=gen)].split(cfg.batch_size):
            opt.zero_grad(set_to_none=True)
            loss_fn(model(X[idx]), y[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(X[va]), y[va]))
        if val < best - 1e-4:
            best, best_epoch, bad = val, epoch, 0
            best_state = copy.deepcopy((model.state_dict(), loss_fn.state_dict()))
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state[0])
    loss_fn.load_state_dict(best_state[1])
    return best_epoch


def _run(
    name: str,
    bound: float | None,
    data: dict[str, torch.Tensor],
    y_mean: float,
    y_scale: float,
    cfg: TailBoundConfig,
) -> dict[str, object]:
    set_comparison_seed(cfg.seed + 1)  # same initial weights for both rows
    model = _backbone(cfg)
    flow = create_flow_model(
        n_features=1,
        context_dim=cfg.context_dim,
        flow_type="nsf",
        n_transforms=cfg.n_transforms,
        hidden_features=[64, 64],
        bins=cfg.bins,
        bound=bound,
    )
    loss_fn = NormalizingFlowLoss(flow=flow)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        best_epoch, train_s = timed_call(
            _train, model, loss_fn, data["X_train"], data["y_train_std"], cfg
        )
    range_warning = any("tail bound" in str(w.message) for w in caught)

    def _evaluate() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        with torch.no_grad():
            ctx = model(data["X_test"])
            y_std = (data["y_test"] - y_mean) / y_scale
            log_p = loss_fn.log_prob(ctx, y_std.unsqueeze(-1)).reshape(-1)
            torch.manual_seed(cfg.seed + 2)
            q = loss_fn.quantile(ctx, [cfg.alpha / 2, 1 - cfg.alpha / 2], n_samples=cfg.n_samples)
        return (
            log_p.numpy(),
            (q[:, 0] * y_scale + y_mean).numpy(),
            (q[:, 1] * y_scale + y_mean).numpy(),
        )

    (log_p, lo, hi), eval_s = timed_call(_evaluate)
    y = data["y_test"].numpy().astype(np.float64)
    used_bound = 5.0 if bound is None else bound
    return {
        "Method": name,
        "TailBound": used_bound,
        "TargetCoverage": 1.0 - cfg.alpha,
        "NLL": float(np.mean(-log_p) + math.log(y_scale)),
        "Coverage": float(np.mean((y >= lo) & (y <= hi))),
        "Width": float(np.mean(hi - lo)),
        "IntervalScore": float(interval_score(lo, hi, y, alpha=cfg.alpha).mean().item()),
        "best_epoch": best_epoch,
        "train_s": train_s,
        "eval_s": eval_s,
        "Notes": (
            "out-of-range warning raised" if range_warning else "all training targets in range"
        ),
    }


def main(
    cfg: TailBoundConfig | None = None,
    summary_json_path: str | None = None,
) -> list[dict[str, object]]:
    cfg = cfg or TailBoundConfig()
    set_comparison_seed(cfg.seed)
    raw = _simulate(cfg)
    y_mean = float(raw["y_train"].mean())
    y_scale = float(raw["y_train"].std())
    data = {k: torch.from_numpy(v) for k, v in raw.items()}
    data["y_train_std"] = ((data["y_train"] - y_mean) / y_scale).unsqueeze(-1)
    bound = recommended_tail_bound(data["y_train_std"])
    frac_out = float((data["y_train_std"].abs() > 5.0).float().mean())
    print(
        f"Standardised training target: max |y| = {float(data['y_train_std'].abs().max()):.2f}, "
        f"{frac_out:.2%} of rows beyond the default bound 5; recommended_tail_bound = {bound:.2f}"
    )

    rows = [
        _run("NSF default bound", None, data, y_mean, y_scale, cfg),
        _run("NSF recommended_tail_bound", bound, data, y_mean, y_scale, cfg),
    ]

    print_fairness_notes(
        title="NSF tail bound comparison",
        seed_policy=f"fixed seed {cfg.seed}; shared split; identical initial weights",
        train_budget=(
            f"MLP context {cfg.context_dim}, NSF {cfg.n_transforms} transforms, {cfg.bins} bins, "
            f"[64, 64]; Adam lr {cfg.lr}, batch {cfg.batch_size}, up to {cfg.epochs} epochs, "
            f"early stopping (patience {cfg.patience}) on a {cfg.val_fraction:.0%} carve"
        ),
        metric_policy=(
            "test NLL in target units, 90% interval coverage / width / interval score "
            f"from {cfg.n_samples} samples, runtime"
        ),
    )
    print_comparison_summary(
        f"NSF tail bound on Student-t({cfg.df:g}) noise",
        rows,
        metric_order=[
            "TailBound",
            "TargetCoverage",
            "NLL",
            "Coverage",
            "Width",
            "IntervalScore",
            "best_epoch",
            "train_s",
            "eval_s",
        ],
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/nsf_tail_bound_comparison.py",
            task="NSF spline tail bound on a heavy-tailed (Student-t) target",
            config=cfg,
            rows=rows,
            notes=[
                f"recommended_tail_bound = {bound:.3f}; "
                f"{frac_out:.4f} of standardised training targets beyond 5",
                "NLL in target units (log sd(y_train) added back).",
                "Single seed; see docs/losses/nflows.md for 5-seed numbers.",
            ],
        )
        print(f"\nWrote summary JSON: {out}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSF tail bound comparison (needs zuko)")
    parser.add_argument("--epochs", type=int, default=TailBoundConfig.epochs)
    parser.add_argument("--seed", type=int, default=TailBoundConfig.seed)
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(
        TailBoundConfig(seed=args.seed, epochs=args.epochs),
        summary_json_path=args.summary_json_path,
    )
