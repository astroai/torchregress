from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from torchregress.prediction import PredictiveBatch
from torchregress.semi_supervised import (
    TeacherStudentTrainer,
    build_consensus_predictive_batch,
    disagreement_to_weight,
    distributional_pseudo_loss,
    predictive_agreement_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _load_example_module(stem: str) -> ModuleType:
    path = EXAMPLES_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"_example_{stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_build_consensus_predictive_batch_for_gaussian_views() -> None:
    mean = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)
    std = torch.full_like(mean, 0.2)
    consensus = build_consensus_predictive_batch(
        [
            PredictiveBatch(mean=mean - 0.1, std=std),
            PredictiveBatch(mean=mean, std=std),
            PredictiveBatch(mean=mean + 0.1, std=std),
        ],
        n_support=96,
    )
    assert consensus.support is not None
    assert consensus.density is not None
    assert consensus.mean is not None
    integral = torch.trapezoid(consensus.density, consensus.support, dim=-1)
    assert torch.allclose(integral, torch.ones_like(integral), atol=1e-3)
    assert torch.allclose(consensus.mean, mean, atol=0.08)


def test_build_consensus_predictive_batch_for_quantile_and_bar_views() -> None:
    quantile_consensus = build_consensus_predictive_batch(
        [
            PredictiveBatch(
                quantiles=torch.tensor([[0.0, 0.4, 0.8], [0.2, 0.5, 1.0]], dtype=torch.float32),
                quantile_levels=[0.1, 0.5, 0.9],
            ),
            PredictiveBatch(
                quantiles=torch.tensor(
                    [[0.05, 0.45, 0.85], [0.15, 0.55, 0.95]], dtype=torch.float32
                ),
                quantile_levels=[0.1, 0.5, 0.9],
            ),
        ],
        n_support=80,
    )
    assert quantile_consensus.density is not None
    q_integral = torch.trapezoid(quantile_consensus.density, quantile_consensus.support, dim=-1)
    assert torch.allclose(q_integral, torch.ones_like(q_integral), atol=1e-3)

    bin_edges = torch.tensor([-1.0, -0.2, 0.4, 1.1], dtype=torch.float32)
    bar_consensus = build_consensus_predictive_batch(
        [
            PredictiveBatch(
                bar_logits=torch.tensor([[2.5, 0.2, -1.0], [0.5, 1.0, -0.2]], dtype=torch.float32),
                bin_edges=bin_edges,
            ),
            PredictiveBatch(
                bar_logits=torch.tensor([[2.1, 0.3, -0.6], [0.2, 1.2, -0.1]], dtype=torch.float32),
                bin_edges=bin_edges,
            ),
        ],
        n_support=80,
    )
    assert bar_consensus.density is not None
    b_integral = torch.trapezoid(bar_consensus.density, bar_consensus.support, dim=-1)
    assert torch.allclose(b_integral, torch.ones_like(b_integral), atol=1e-3)


def test_build_consensus_predictive_batch_for_support_density_views() -> None:
    support_a = torch.linspace(-1.0, 1.0, 32)
    density_a = torch.exp(-0.5 * support_a.square())
    density_a = density_a / density_a.sum()
    support_b = torch.linspace(-0.8, 1.2, 40)
    density_b = torch.exp(-0.5 * (support_b - 0.1).square())
    density_b = density_b / density_b.sum()

    consensus = build_consensus_predictive_batch(
        [
            PredictiveBatch(
                support=support_a.unsqueeze(0).expand(2, -1),
                density=density_a.unsqueeze(0).expand(2, -1),
            ),
            PredictiveBatch(
                support=support_b.unsqueeze(0).expand(2, -1),
                density=density_b.unsqueeze(0).expand(2, -1),
            ),
        ],
        n_support=80,
    )
    assert consensus.support is not None
    assert consensus.density is not None
    integral = torch.trapezoid(consensus.density, consensus.support, dim=-1)
    assert torch.allclose(integral, torch.ones_like(integral), atol=1e-3)


def test_predictive_agreement_score_tracks_mismatch_and_weights() -> None:
    mean = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)
    std = torch.full_like(mean, 0.2)
    identical = predictive_agreement_score(
        [
            PredictiveBatch(mean=mean, std=std),
            PredictiveBatch(mean=mean, std=std),
            PredictiveBatch(mean=mean, std=std),
        ],
        n_support=96,
    )
    shifted = predictive_agreement_score(
        [
            PredictiveBatch(mean=mean - 0.5, std=std),
            PredictiveBatch(mean=mean, std=std),
            PredictiveBatch(mean=mean + 0.5, std=std),
        ],
        n_support=96,
    )
    assert torch.allclose(identical, torch.zeros_like(identical), atol=1e-4)
    assert float(shifted.mean().item()) > float(identical.mean().item())

    stable_weight = torch.exp(-identical / 0.15).mean()
    unstable_weight = torch.exp(-shifted / 0.15).mean()
    assert float(stable_weight.item()) > float(unstable_weight.item())


def test_disagreement_to_weight_supports_tempered_and_hard_gates() -> None:
    disagreement = torch.tensor([0.0, 0.05, 0.20, 0.50], dtype=torch.float32)
    base = disagreement_to_weight(disagreement, 0.2)
    tempered = disagreement_to_weight(disagreement, 0.2, power=2.0)
    assert torch.all(tempered <= base)


def test_distributional_pseudo_loss_is_small_for_matching_predictions() -> None:
    consensus = build_consensus_predictive_batch(
        [
            PredictiveBatch(
                mean=torch.tensor([[0.1], [0.8]], dtype=torch.float32), std=torch.full((2, 1), 0.2)
            ),
            PredictiveBatch(
                mean=torch.tensor([[0.1], [0.8]], dtype=torch.float32), std=torch.full((2, 1), 0.2)
            ),
        ],
        n_support=96,
    )
    matching = PredictiveBatch(
        mean=torch.tensor([[0.1], [0.8]], dtype=torch.float32),
        std=torch.full((2, 1), 0.2),
    )
    mismatched = PredictiveBatch(
        mean=torch.tensor([[0.8], [0.1]], dtype=torch.float32),
        std=torch.full((2, 1), 0.2),
    )
    match_loss = distributional_pseudo_loss(matching, consensus, n_support=96)
    mismatch_loss = distributional_pseudo_loss(mismatched, consensus, n_support=96)
    assert float(match_loss.item()) < float(mismatch_loss.item())


def test_distributional_pseudo_loss_supports_quantile_density_cross_entropy() -> None:
    consensus = build_consensus_predictive_batch(
        [
            PredictiveBatch(
                quantiles=torch.tensor([[0.0, 0.5, 1.0], [0.2, 0.6, 1.1]], dtype=torch.float32),
                quantile_levels=[0.1, 0.5, 0.9],
            ),
            PredictiveBatch(
                quantiles=torch.tensor(
                    [[0.05, 0.45, 0.95], [0.25, 0.65, 1.15]], dtype=torch.float32
                ),
                quantile_levels=[0.1, 0.5, 0.9],
            ),
        ],
        n_support=96,
    )
    matching = PredictiveBatch(
        quantiles=torch.tensor([[0.0, 0.5, 1.0], [0.2, 0.6, 1.1]], dtype=torch.float32),
        quantile_levels=[0.1, 0.5, 0.9],
    )
    mismatched = PredictiveBatch(
        quantiles=torch.tensor([[0.9, 1.4, 1.9], [-0.4, 0.0, 0.4]], dtype=torch.float32),
        quantile_levels=[0.1, 0.5, 0.9],
    )
    assert float(distributional_pseudo_loss(matching, consensus, n_support=96).item()) < float(
        distributional_pseudo_loss(mismatched, consensus, n_support=96).item()
    )


def test_distributional_pseudo_loss_supports_bar_pmf_cross_entropy() -> None:
    edges = torch.tensor([-1.0, -0.2, 0.4, 1.2], dtype=torch.float32)
    consensus = build_consensus_predictive_batch(
        [
            PredictiveBatch(
                bar_logits=torch.tensor([[2.8, 0.1, -1.0], [0.4, 1.2, -0.5]], dtype=torch.float32),
                bin_edges=edges,
            ),
            PredictiveBatch(
                bar_logits=torch.tensor([[2.4, 0.2, -0.8], [0.2, 1.3, -0.4]], dtype=torch.float32),
                bin_edges=edges,
            ),
        ],
        n_support=96,
    )
    matching = PredictiveBatch(
        bar_logits=torch.tensor([[2.6, 0.2, -0.9], [0.3, 1.1, -0.4]], dtype=torch.float32),
        bin_edges=edges,
    )
    mismatched = PredictiveBatch(
        bar_logits=torch.tensor([[-1.0, 0.1, 2.6], [1.4, -0.5, 0.0]], dtype=torch.float32),
        bin_edges=edges,
    )
    assert float(distributional_pseudo_loss(matching, consensus, n_support=96).item()) < float(
        distributional_pseudo_loss(mismatched, consensus, n_support=96).item()
    )


class _TinyBackbone(torch.nn.Module):
    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TinyGaussian(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _TinyBackbone()
        self.mean_head = torch.nn.Linear(8, 1)
        self.log_var_head = torch.nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.mean_head(h), self.log_var_head(h).clamp(min=-4.0, max=2.0)


class _TinyQuantile(torch.nn.Module):
    levels = [0.1, 0.5, 0.9]

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _TinyBackbone()
        self.head = torch.nn.Linear(8, len(self.levels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sort(self.head(self.backbone(x)), dim=-1).values


class _TinyBar(torch.nn.Module):
    def __init__(self, bin_edges: torch.Tensor) -> None:
        super().__init__()
        self.backbone = _TinyBackbone()
        self.head = torch.nn.Linear(8, bin_edges.numel() - 1)
        self.register_buffer("bin_edges", bin_edges)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


@pytest.mark.parametrize("backbone", ["gaussian", "quantile", "bar"])
def test_self_agreement_trainer_runs_all_target_backbones(backbone: str) -> None:
    torch.manual_seed(0)
    x_labeled = torch.linspace(-1.0, 1.0, 18).unsqueeze(-1)
    y_labeled = torch.sin(1.3 * x_labeled) + 0.1 * x_labeled
    x_unlabeled = torch.linspace(-1.2, 1.2, 24).unsqueeze(-1)

    labeled_loader = DataLoader(TensorDataset(x_labeled, y_labeled), batch_size=6, shuffle=False)
    unlabeled_loader = DataLoader(TensorDataset(x_unlabeled), batch_size=6, shuffle=False)

    if backbone == "gaussian":
        model = _TinyGaussian()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

        def supervised_loss_fn(
            model_: torch.nn.Module, x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            mean, log_var = model_(x)
            var = torch.exp(log_var).clamp_min(1e-5)
            return torch.nn.functional.gaussian_nll_loss(mean, y, var)

        def predictive_batch_fn(model_: torch.nn.Module, x: torch.Tensor) -> PredictiveBatch:
            mean, log_var = model_(x)
            return PredictiveBatch(mean=mean, std=torch.exp(0.5 * log_var))

    elif backbone == "quantile":
        model = _TinyQuantile()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

        def supervised_loss_fn(
            model_: torch.nn.Module, x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            quantiles = model_(x)
            diff = y - quantiles
            levels = torch.tensor(
                _TinyQuantile.levels, dtype=quantiles.dtype, device=quantiles.device
            )
            return torch.maximum(levels * diff, (levels - 1.0) * diff).mean()

        def predictive_batch_fn(model_: torch.nn.Module, x: torch.Tensor) -> PredictiveBatch:
            return PredictiveBatch(
                quantiles=model_(x),
                quantile_levels=list(_TinyQuantile.levels),
            )

    else:
        bin_edges = torch.linspace(-1.5, 1.5, 9)
        model = _TinyBar(bin_edges)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

        def supervised_loss_fn(
            model_: torch.nn.Module, x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            logits = model_(x)
            targets = torch.bucketize(y.view(-1), bin_edges[1:-1]).long()
            return F.cross_entropy(logits, targets)

        def predictive_batch_fn(model_: torch.nn.Module, x: torch.Tensor) -> PredictiveBatch:
            return PredictiveBatch(bar_logits=model_(x), bin_edges=bin_edges)

    trainer = TeacherStudentTrainer(
        optimizer=optimizer,
        supervised_loss_fn=supervised_loss_fn,
        predictive_batch_fn=predictive_batch_fn,
        augment_fn=lambda x: x + 0.03 * torch.randn_like(x),
        n_views=3,
        agreement_weight=0.5,
        ema_decay=0.95,
        n_support=64,
    )
    history = trainer.fit(model, labeled_loader, unlabeled_loader, epochs=2)
    assert history["total_loss"]
    assert len(history["total_loss"]) == len(history["mean_weight"])
    assert all(math.isfinite(v) for v in history["total_loss"])
    assert all(math.isfinite(v) for v in history["unsupervised_loss"])
    assert all(0.0 < v <= 1.0 for v in history["mean_weight"])


def test_self_agreement_trainer_runs_tiny_fit_loop() -> None:
    torch.manual_seed(0)
    x_labeled = torch.linspace(-1.0, 1.0, 16).unsqueeze(-1)
    y_labeled = (0.7 * x_labeled + 0.2).sin()
    x_unlabeled = torch.linspace(-1.2, 1.2, 24).unsqueeze(-1)

    labeled_loader = DataLoader(TensorDataset(x_labeled, y_labeled), batch_size=8, shuffle=False)
    unlabeled_loader = DataLoader(TensorDataset(x_unlabeled), batch_size=8, shuffle=False)

    model = torch.nn.Sequential(
        torch.nn.Linear(1, 12),
        torch.nn.Tanh(),
        torch.nn.Linear(12, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    def supervised_loss_fn(
        model_: torch.nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        raw = model_(x)
        mean = raw[:, :1]
        std = torch.nn.functional.softplus(raw[:, 1:2]) + 1e-3
        var = std.square()
        return torch.nn.functional.gaussian_nll_loss(mean, y, var)

    def predictive_batch_fn(model_: torch.nn.Module, x: torch.Tensor) -> PredictiveBatch:
        raw = model_(x)
        return PredictiveBatch(
            mean=raw[:, :1],
            std=torch.nn.functional.softplus(raw[:, 1:2]) + 1e-3,
        )

    trainer = TeacherStudentTrainer(
        optimizer=optimizer,
        supervised_loss_fn=supervised_loss_fn,
        predictive_batch_fn=predictive_batch_fn,
        augment_fn=lambda x: x + 0.03 * torch.randn_like(x),
        n_views=3,
        agreement_weight=0.5,
        ema_decay=0.95,
        n_support=64,
    )
    history = trainer.fit(model, labeled_loader, unlabeled_loader, epochs=2)
    assert history["total_loss"]
    assert len(history["total_loss"]) == len(history["mean_weight"])
    assert all(math.isfinite(v) for v in history["total_loss"])
    assert all(0.0 < v <= 1.0 for v in history["mean_weight"])


def test_self_agreement_trainer_cosine_lr_schedule_runs() -> None:
    torch.manual_seed(0)
    x_labeled = torch.linspace(-1.0, 1.0, 16).unsqueeze(-1)
    y_labeled = (0.7 * x_labeled + 0.2).sin()
    x_unlabeled = torch.linspace(-1.2, 1.2, 24).unsqueeze(-1)

    labeled_loader = DataLoader(TensorDataset(x_labeled, y_labeled), batch_size=8, shuffle=False)
    unlabeled_loader = DataLoader(TensorDataset(x_unlabeled), batch_size=8, shuffle=False)

    model = torch.nn.Sequential(
        torch.nn.Linear(1, 12),
        torch.nn.Tanh(),
        torch.nn.Linear(12, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    def supervised_loss_fn(
        model_: torch.nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        raw = model_(x)
        mean = raw[:, :1]
        std = torch.nn.functional.softplus(raw[:, 1:2]) + 1e-3
        var = std.square()
        return torch.nn.functional.gaussian_nll_loss(mean, y, var)

    def predictive_batch_fn(model_: torch.nn.Module, x: torch.Tensor) -> PredictiveBatch:
        raw = model_(x)
        return PredictiveBatch(
            mean=raw[:, :1],
            std=torch.nn.functional.softplus(raw[:, 1:2]) + 1e-3,
        )

    trainer = TeacherStudentTrainer(
        optimizer=optimizer,
        supervised_loss_fn=supervised_loss_fn,
        predictive_batch_fn=predictive_batch_fn,
        augment_fn=lambda x: x + 0.03 * torch.randn_like(x),
        n_views=3,
        agreement_weight=0.5,
        ema_decay=0.95,
        n_support=64,
    )
    epochs = 3
    history = trainer.fit(
        model,
        labeled_loader,
        unlabeled_loader,
        epochs=epochs,
        lr_schedule="cosine",
        lr_min=1e-5,
    )
    assert len(history["total_loss"]) == epochs * len(labeled_loader)
    assert all(math.isfinite(v) for v in history["total_loss"])


def test_predictive_agreement_score_requires_multiple_views() -> None:
    with pytest.raises(ValueError):
        predictive_agreement_score([PredictiveBatch(mean=torch.zeros(4, 1), std=torch.ones(4, 1))])


def test_teacher_student_trainer_modular_runs() -> None:
    from torchregress.semi_supervised import TeacherStudentTrainer, uncertainty_to_weight

    torch.manual_seed(0)
    x_labeled = torch.linspace(-1.0, 1.0, 8).unsqueeze(-1)
    y_labeled = 0.5 * x_labeled
    x_unlabeled = torch.linspace(-1.0, 1.0, 12).unsqueeze(-1)

    labeled_loader = DataLoader(TensorDataset(x_labeled, y_labeled), batch_size=4, shuffle=False)
    unlabeled_loader = DataLoader(TensorDataset(x_unlabeled), batch_size=4, shuffle=False)

    model = torch.nn.Sequential(
        torch.nn.Linear(1, 4),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def supervised_loss_fn(model_, x, y):
        pred = model_(x)[:, :1]
        return F.mse_loss(pred, y)

    def predictive_batch_fn(model_, x):
        raw = model_(x)
        return PredictiveBatch(
            mean=raw[:, :1],
            std=F.softplus(raw[:, 1:2]) + 1e-3,
        )

    # Use a custom weight function that weights based on uncertainty
    def custom_weight_fn(views, consensus):
        return uncertainty_to_weight(consensus, tau=0.5)

    trainer = TeacherStudentTrainer(
        optimizer=optimizer,
        supervised_loss_fn=supervised_loss_fn,
        predictive_batch_fn=predictive_batch_fn,
        sample_weight_fn=custom_weight_fn,
        n_views=2,
        agreement_weight=0.1,
        n_support=32,
    )

    history = trainer.fit(model, labeled_loader, unlabeled_loader, epochs=1)
    assert "total_loss" in history
    assert "mean_weight" in history
    assert len(history["total_loss"]) == len(labeled_loader)


def test_uncertainty_to_weight_values() -> None:
    from torchregress.semi_supervised import uncertainty_to_weight

    std = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float32)
    batch = PredictiveBatch(mean=torch.zeros_like(std), std=std)
    weights = uncertainty_to_weight(batch, tau=1.0)
    # weights should be exp(-std / tau)
    expected = torch.exp(-std / 1.0)
    assert torch.allclose(weights, expected)


def test_conformal_width_to_weight_values() -> None:
    from torchregress.semi_supervised import conformal_width_to_weight

    lower = torch.tensor([1.0, 2.0, 3.0])
    upper = torch.tensor([2.0, 4.0, 4.0])  # widths = [1.0, 2.0, 1.0]

    # test soft weights
    weights = conformal_width_to_weight(lower, upper, tau=2.0)
    assert torch.allclose(weights, torch.exp(torch.tensor([-0.5, -1.0, -0.5])))

    # test threshold mask
    mask = conformal_width_to_weight(lower, upper, threshold=1.5)
    assert torch.allclose(mask, torch.tensor([1.0, 0.0, 1.0]))


def test_predictive_agreement_score_values() -> None:
    mean = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    std = torch.full_like(mean, 0.2)
    views = [
        PredictiveBatch(mean=mean - 0.02, std=std),
        PredictiveBatch(mean=mean + 0.02, std=std),
    ]
    score = predictive_agreement_score(views, n_support=64)
    assert score.shape == (2,)
    assert torch.all(score >= 0.0)
