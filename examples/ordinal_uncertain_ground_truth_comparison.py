"""Comparison for uncertain ground truth in regression-as-classification."""

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from torchregress.comparison import (
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from torchregress.losses import CumulativeLinkLoss, OrdinalCrossEntropyLoss
from torchregress.metrics import (
    mean_absolute_class_error,
    ordinal_accuracy,
    quadratic_weighted_kappa,
)
from torchregress.utils import cumulative_logits_to_pmf, ordinal_predict


@dataclass(frozen=True)
class OrdinalUGTComparisonConfig:
    seed: int = 260305
    n_train: int = 512
    n_test: int = 256
    n_features: int = 6
    num_classes: int = 5
    hidden: int = 32
    epochs: int = 30
    teacher_epochs: int = 24
    batch_size: int = 64
    lr: float = 1e-2
    labeled_fraction: float = 0.4
    pseudo_threshold: float = 0.45
    pseudo_weight: float = 0.7
    teacher_temperature: float = 1.25


class _MLP(torch.nn.Module):
    def __init__(self, n_features: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _make_soft_targets(
    latent: Tensor, y_true: Tensor, cutpoints: Tensor, num_classes: int
) -> Tensor:
    probs = F.one_hot(y_true, num_classes=num_classes).to(dtype=torch.float32)

    left_pad = torch.full((1,), cutpoints[0] - 1.0, dtype=latent.dtype)
    right_pad = torch.full((1,), cutpoints[-1] + 1.0, dtype=latent.dtype)
    boundaries = torch.cat([left_pad, cutpoints, right_pad])
    lower = boundaries[y_true]
    upper = boundaries[y_true + 1]
    boundary_distance = torch.minimum(latent - lower, upper - latent).clamp_min(0.0)
    ambiguity = torch.exp(-2.5 * boundary_distance).clamp(0.0, 1.0)

    center_mass = 1.0 - 0.55 * ambiguity
    neighbor_mass = 0.55 * ambiguity
    probs = probs * center_mass.unsqueeze(1)

    left_mask = y_true > 0
    right_mask = y_true < (num_classes - 1)
    two_sided = left_mask & right_mask

    probs[left_mask, y_true[left_mask] - 1] += torch.where(
        two_sided[left_mask],
        0.5 * neighbor_mass[left_mask],
        neighbor_mass[left_mask],
    )
    probs[right_mask, y_true[right_mask] + 1] += torch.where(
        two_sided[right_mask],
        0.5 * neighbor_mass[right_mask],
        neighbor_mass[right_mask],
    )
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _make_data(
    cfg: OrdinalUGTComparisonConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    set_comparison_seed(cfg.seed)
    x = torch.randn(cfg.n_train + cfg.n_test, cfg.n_features)

    w = torch.tensor([0.9, -0.7, 0.4, 0.2, -0.3, 0.1], dtype=torch.float32)[: cfg.n_features]
    latent = x @ w
    latent = latent + 0.35 * x[:, 0] * x[:, 1] - 0.2 * x[:, 2] ** 2
    latent = latent + 0.4 * torch.randn_like(latent)

    cutpoints = torch.tensor([-1.0, -0.25, 0.4, 1.1], dtype=latent.dtype)[: cfg.num_classes - 1]
    y_true = torch.bucketize(latent, cutpoints).long()
    soft_targets = _make_soft_targets(latent, y_true, cutpoints, cfg.num_classes)
    y_observed = torch.multinomial(soft_targets, num_samples=1).squeeze(1)

    x_train = x[: cfg.n_train]
    hard_train = y_observed[: cfg.n_train]
    soft_train = soft_targets[: cfg.n_train]
    x_test = x[cfg.n_train :]
    y_test = y_true[cfg.n_train :]
    soft_test = soft_targets[cfg.n_train :]
    return x_train, hard_train, soft_train, x_test, y_test, soft_test


def _make_labeled_mask(cfg: OrdinalUGTComparisonConfig, n_train: int) -> Tensor:
    n_labeled = max(24, int(round(cfg.labeled_fraction * n_train)))
    n_labeled = min(max(n_labeled, 1), n_train - 1)
    perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(cfg.seed + 7))
    mask = torch.zeros(n_train, dtype=torch.bool)
    mask[perm[:n_labeled]] = True
    return mask


def _train_soft(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    x_train: Tensor,
    target: Tensor,
    cfg: OrdinalUGTComparisonConfig,
    *,
    epochs: int | None = None,
    sample_weights: Tensor | None = None,
) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loader = DataLoader(
        TensorDataset(
            x_train,
            target,
            sample_weights if sample_weights is not None else torch.ones(x_train.shape[0]),
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed + 13),
    )
    total_epochs = cfg.epochs if epochs is None else epochs
    for _ in range(total_epochs):
        for xb, yb, wb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            weights = wb if sample_weights is not None else None
            loss = loss_fn(logits, yb, weights=weights)
            loss.backward()
            optimizer.step()


def _pmf_from_model(model: torch.nn.Module, x: Tensor, *, encoding: str) -> Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(x)
    if encoding == "class_logits":
        return torch.softmax(logits, dim=1)
    if encoding == "cumulative_logits":
        return cumulative_logits_to_pmf(logits)
    raise ValueError(f"Unknown encoding: {encoding}")


def _evaluate(
    model: torch.nn.Module,
    x_test: Tensor,
    y_test: Tensor,
    soft_test: Tensor,
    *,
    encoding: str,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x_test)
    pred, pmf = ordinal_predict(logits, encoding=encoding, return_pmf=True)
    assert isinstance(pred, Tensor)
    assert isinstance(pmf, Tensor)
    true_nll = -torch.log(pmf.gather(1, y_test.unsqueeze(1)).clamp_min(1e-8)).mean().item()
    plaus_ce = -(soft_test * torch.log(pmf.clamp_min(1e-8))).sum(dim=1).mean().item()
    return {
        "Accuracy": float(ordinal_accuracy(pred, y_test, encoding="labels").item()),
        "OrdinalMAE": float(mean_absolute_class_error(pred, y_test, encoding="labels").item()),
        "QWK": float(quadratic_weighted_kappa(pred, y_test, encoding="labels").item()),
        "TrueNLL": float(true_nll),
        "PlausibilityCE": float(plaus_ce),
    }


def _build_pseudo_targets(
    teacher: torch.nn.Module,
    x_train: Tensor,
    soft_train: Tensor,
    label_mask: Tensor,
    cfg: OrdinalUGTComparisonConfig,
) -> tuple[Tensor, Tensor, float]:
    teacher.eval()
    with torch.no_grad():
        teacher_logits = teacher(x_train) / cfg.teacher_temperature
        teacher_probs = torch.softmax(teacher_logits, dim=1)
    confidence = teacher_probs.max(dim=1).values
    accepted = (~label_mask) & (confidence >= cfg.pseudo_threshold)
    full_target = soft_train.clone()
    full_target[~label_mask] = teacher_probs[~label_mask]
    sample_weights = label_mask.float() + cfg.pseudo_weight * confidence * accepted.float()
    accept_rate = float(accepted.float().mean().item())
    return full_target, sample_weights, accept_rate


def run_comparison(cfg: OrdinalUGTComparisonConfig) -> list[dict[str, object]]:
    set_comparison_seed(cfg.seed)
    x_train, hard_train, soft_train, x_test, y_test, soft_test = _make_data(cfg)
    label_mask = _make_labeled_mask(cfg, x_train.shape[0])

    x_labeled = x_train[label_mask]
    hard_labeled = hard_train[label_mask]
    soft_labeled = soft_train[label_mask]

    teacher = _MLP(cfg.n_features, cfg.hidden, cfg.num_classes)
    _train_soft(
        teacher,
        OrdinalCrossEntropyLoss(),
        x_labeled,
        soft_labeled,
        cfg,
        epochs=cfg.teacher_epochs,
    )
    pseudo_target, pseudo_weights, pseudo_accept_rate = _build_pseudo_targets(
        teacher, x_train, soft_train, label_mask, cfg
    )

    rows: list[dict[str, object]] = []
    methods = [
        (
            "HardOrdinalCE",
            _MLP(cfg.n_features, cfg.hidden, cfg.num_classes),
            OrdinalCrossEntropyLoss(),
            x_labeled,
            hard_labeled,
            None,
            "class_logits",
            "hard sampled labels only",
        ),
        (
            "SoftOrdinalCE",
            _MLP(cfg.n_features, cfg.hidden, cfg.num_classes),
            OrdinalCrossEntropyLoss(),
            x_labeled,
            soft_labeled,
            None,
            "class_logits",
            "soft plausibility targets on labeled subset",
        ),
        (
            "SoftOrdinalCE+Pseudo",
            _MLP(cfg.n_features, cfg.hidden, cfg.num_classes),
            OrdinalCrossEntropyLoss(),
            x_train,
            pseudo_target,
            pseudo_weights,
            "class_logits",
            "soft plausibility targets plus confidence-gated soft pseudo labels",
        ),
        (
            "SoftCumulativeLink",
            _MLP(cfg.n_features, cfg.hidden, cfg.num_classes - 1),
            CumulativeLinkLoss(),
            x_labeled,
            soft_labeled,
            None,
            "cumulative_logits",
            "cumulative-link objective trained on soft plausibility targets",
        ),
    ]

    for idx, (name, model, loss_fn, train_x, target, weights, encoding, notes) in enumerate(
        methods
    ):
        set_comparison_seed(cfg.seed + idx)
        _, train_s = timed_call(
            _train_soft, model, loss_fn, train_x, target, cfg, sample_weights=weights
        )
        metrics, eval_s = timed_call(_evaluate, model, x_test, y_test, soft_test, encoding=encoding)
        rows.append(
            {
                "Method": name,
                **metrics,
                "PseudoAcceptRate": pseudo_accept_rate if name == "SoftOrdinalCE+Pseudo" else None,
                "LabeledFraction": float(label_mask.float().mean().item()),
                "train_s": float(train_s),
                "eval_s": float(eval_s),
                "Notes": notes,
            }
        )
    return rows


def main(
    cfg: OrdinalUGTComparisonConfig | None = None,
    summary_json_path: str | None = None,
) -> None:
    cfg = cfg or OrdinalUGTComparisonConfig()
    rows = run_comparison(cfg)

    print_fairness_notes(
        title="Ordinal Uncertain-GT Comparison",
        seed_policy="fixed seed; shared ordered-bin split and labeled/unlabeled mask",
        train_budget="same MLP depth/width, epochs, and pseudo-label threshold across methods",
        metric_policy="accuracy, ordinal class-MAE, QWK, true-class NLL, plausibility CE, runtime",
    )
    print_comparison_summary(
        "Ordinal uncertain-ground-truth summary",
        rows,
        metric_order=[
            "Accuracy",
            "OrdinalMAE",
            "QWK",
            "TrueNLL",
            "PlausibilityCE",
            "PseudoAcceptRate",
            "train_s",
            "eval_s",
        ],
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/ordinal_uncertain_ground_truth_comparison.py",
            task="Ordinal regression / uncertain ground truth",
            config=cfg,
            rows=rows,
            notes=[
                "Ordered-bin regression-as-classification comparison with plausibility-style soft targets.",
                "Soft pseudo labels are generated by a teacher trained on the labeled soft-target subset.",
                "Hard baseline sees sampled labels only; soft methods use ambiguous target distributions directly.",
            ],
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ordinal uncertain-ground-truth comparison")
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path)
