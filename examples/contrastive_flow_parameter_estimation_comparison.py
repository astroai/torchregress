"""Shared-budget comparison for parameter-estimation methods on synthetic pseudoexperiments."""

from dataclasses import dataclass
from typing import Optional

import torch
from contrastive_flow_benchmark_utils import (
    MLP,
    ParameterGrid,
    estimate_parameters_from_scores,
    gaussian_summary_loss,
    score_flow_summary_model,
    score_gaussian_summary_model,
    train_contrastive_density_model,
    train_flow_density_model,
    train_supervised_density_model,
    try_make_flow_losses,
)
from contrastive_flow_parameter_estimation import (
    MU_RANGE,
    NUISANCE_RANGE,
    generate_parameterized_pseudoexperiments,
)
from torch.utils.data import DataLoader, TensorDataset

from torchregress.comparison import (
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)


@dataclass(frozen=True)
class ContrastiveFlowComparisonConfig:
    n_train: int = 512
    n_test: int = 96
    events_per_experiment: int = 128
    batch_size: int = 64
    epochs: int = 30
    lr: float = 2e-3
    seed: int = 17
    hidden: int = 64
    flow_context_dim: int = 32
    flow_transforms: int = 4
    n_negatives: int = 4
    mu_grid_size: int = 31
    nuisance_grid_size: int = 21


def _sample_negative_params(cfg: ContrastiveFlowComparisonConfig, batch_size: int) -> torch.Tensor:
    mu = torch.empty(batch_size, cfg.n_negatives).uniform_(MU_RANGE[0], MU_RANGE[1])
    nuisance = torch.empty(batch_size, cfg.n_negatives).uniform_(
        NUISANCE_RANGE[0], NUISANCE_RANGE[1]
    )
    return torch.stack([mu, nuisance], dim=-1)


def main(
    cfg: Optional[ContrastiveFlowComparisonConfig] = None,
    summary_json_path: str | None = None,
) -> None:
    cfg = cfg or ContrastiveFlowComparisonConfig()
    set_comparison_seed(cfg.seed)

    params_train, summaries_train = generate_parameterized_pseudoexperiments(
        cfg.n_train,
        n_events=cfg.events_per_experiment,
        seed=cfg.seed,
    )
    params_test, summaries_test = generate_parameterized_pseudoexperiments(
        cfg.n_test,
        n_events=cfg.events_per_experiment,
        seed=cfg.seed + 1,
    )
    loader = DataLoader(
        TensorDataset(params_train, summaries_train),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    grid = ParameterGrid(
        lows=(MU_RANGE[0], NUISANCE_RANGE[0]),
        highs=(MU_RANGE[1], NUISANCE_RANGE[1]),
        steps=(cfg.mu_grid_size, cfg.nuisance_grid_size),
    )

    summary_rows: list[dict[str, object]] = []

    gaussian_model = MLP(2, summaries_train.shape[1] * 2, hidden=cfg.hidden)
    gaussian_loss = gaussian_summary_loss(summaries_train.shape[1])
    _, train_s = timed_call(
        train_supervised_density_model,
        gaussian_model,
        gaussian_loss,
        loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
    )
    (gaussian_metrics, _), eval_s = timed_call(
        estimate_parameters_from_scores,
        lambda summary, grid_tensor: score_gaussian_summary_model(
            gaussian_model, summary, grid_tensor
        ),
        summaries_test,
        params_test,
        grid,
    )
    summary_rows.append(
        {
            "Method": "GaussianSummary",
            **gaussian_metrics,
            "train_s": train_s,
            "eval_s": eval_s,
            "Notes": "diagonal Gaussian density over pseudoexperiment summaries",
        }
    )

    flow_context_model, flow_loss, contrastive_context_model, contrastive_loss, flow_err = (
        try_make_flow_losses(
            param_dim=2,
            summary_dim=summaries_train.shape[1],
            hidden=cfg.hidden,
            context_dim=cfg.flow_context_dim,
            n_transforms=cfg.flow_transforms,
        )
    )

    if flow_context_model is not None and flow_loss is not None:
        _, train_s = timed_call(
            train_flow_density_model,
            flow_context_model,
            getattr(flow_loss, "flow"),
            flow_loss,
            loader,
            epochs=cfg.epochs,
            lr=cfg.lr,
        )
        (flow_metrics, _), eval_s = timed_call(
            estimate_parameters_from_scores,
            lambda summary, grid_tensor: score_flow_summary_model(
                flow_context_model, flow_loss, summary, grid_tensor
            ),
            summaries_test,
            params_test,
            grid,
        )
        summary_rows.append(
            {
                "Method": "NormalizingFlow",
                **flow_metrics,
                "train_s": train_s,
                "eval_s": eval_s,
                "Notes": "conditional flow NLL over pseudoexperiment summaries",
            }
        )
    else:
        summary_rows.append(
            {
                "Method": "NormalizingFlow",
                "ParamMAE": None,
                "Dim0_MAE": None,
                "Dim1_MAE": None,
                "train_s": None,
                "eval_s": None,
                "Notes": f"skipped (optional dependency unavailable: {flow_err})",
            }
        )

    if contrastive_context_model is not None and contrastive_loss is not None:
        _, train_s = timed_call(
            train_contrastive_density_model,
            contrastive_context_model,
            getattr(contrastive_loss, "flow"),
            contrastive_loss,
            loader,
            epochs=cfg.epochs,
            lr=cfg.lr,
            negative_sampler=lambda batch_size: _sample_negative_params(cfg, batch_size),
        )
        (contrastive_metrics, _), eval_s = timed_call(
            estimate_parameters_from_scores,
            lambda summary, grid_tensor: score_flow_summary_model(
                contrastive_context_model, contrastive_loss, summary, grid_tensor
            ),
            summaries_test,
            params_test,
            grid,
        )
        summary_rows.append(
            {
                "Method": "ContrastiveFlow",
                **contrastive_metrics,
                "train_s": train_s,
                "eval_s": eval_s,
                "Notes": "contrastive likelihood-ratio flow over positive vs alternate hypotheses",
            }
        )
    else:
        summary_rows.append(
            {
                "Method": "ContrastiveFlow",
                "ParamMAE": None,
                "Dim0_MAE": None,
                "Dim1_MAE": None,
                "train_s": None,
                "eval_s": None,
                "Notes": f"skipped (optional dependency unavailable: {flow_err})",
            }
        )

    print_fairness_notes(
        title="Contrastive Flow Parameter Estimation Comparison",
        seed_policy="fixed synthetic generator and shared train/test pseudoexperiments",
        train_budget=f"{cfg.epochs} epochs, batch_size={cfg.batch_size}, lr={cfg.lr}",
        metric_policy="parameter MAE on shared scan grid + runtime",
    )
    print_comparison_summary(
        "Synthetic Parameter-Estimation Comparison Summary",
        summary_rows,
        metric_order=["MSE", "MAE", "R2", "ParamMAE", "Dim0_MAE", "Dim1_MAE", "train_s", "eval_s"],
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/contrastive_flow_parameter_estimation_comparison.py",
            task="Synthetic parameter estimation under nuisance shift",
            config=cfg,
            rows=summary_rows,
            notes=[
                "Shared pseudoexperiment generator and shared parameter scan grid",
                "Flow rows are optional when zuko is unavailable",
            ],
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    main()
