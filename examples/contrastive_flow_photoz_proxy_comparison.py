"""Photo-z proxy benchmark for contrastive parameter-estimation methods."""

from dataclasses import dataclass
from typing import Optional

import torch
from comparison_utils import (
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
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
from photoz_benchmark_comparison import _data_source_name, _make_splits
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class ContrastivePhotoZProxyConfig:
    seed: int = 240410
    n_train: int = 128
    n_cal: int = 64
    n_test: int = 64
    batch_size: int = 32
    epochs: int = 20
    lr: float = 2e-3
    hidden: int = 64
    flow_context_dim: int = 24
    flow_transforms: int = 4
    n_negatives: int = 4
    n_train_experiments: int = 256
    n_test_experiments: int = 80
    catalog_size: int = 48
    delta_z_range: tuple[float, float] = (-0.35, 0.35)
    noise_scale_range: tuple[float, float] = (0.6, 1.8)
    dataset_path: str | None = None
    train_dataset_path: str | None = None
    cal_dataset_path: str | None = None
    test_dataset_path: str | None = None
    force_simulated: bool = False
    require_real_data: bool = False
    allow_download: bool = False
    sample_size_if_generate: int = 600


def _sample_catalog_params(
    n: int,
    *,
    delta_z_range: tuple[float, float],
    noise_scale_range: tuple[float, float],
    generator: torch.Generator,
) -> torch.Tensor:
    delta = delta_z_range[0] + (delta_z_range[1] - delta_z_range[0]) * torch.rand(
        n, generator=generator
    )
    scale = noise_scale_range[0] + (noise_scale_range[1] - noise_scale_range[0]) * torch.rand(
        n, generator=generator
    )
    return torch.stack([delta, scale], dim=1)


def _summarize_photoz_catalog(
    x: torch.Tensor,
    xerr: torch.Tensor,
    y: torch.Tensor,
    *,
    delta_z: float,
    noise_scale: float,
    generator: torch.Generator,
) -> torch.Tensor:
    noise = torch.randn(y.shape[0], 1, generator=generator) * (0.04 + 0.03 * noise_scale)
    z_obs = y + delta_z + noise
    color0 = x[:, :1] + torch.randn(x.shape[0], 1, generator=generator) * (
        xerr[:, :1] * noise_scale
    )
    color1 = x[:, 1:2] + torch.randn(x.shape[0], 1, generator=generator) * (
        xerr[:, 1:2] * noise_scale
    )
    z_center = z_obs - z_obs.mean()
    c_center = color0 - color0.mean()
    corr = (z_center * c_center).mean() / (
        z_center.std(unbiased=False).clamp_min(1e-4) * c_center.std(unbiased=False).clamp_min(1e-4)
    )
    high_z = (z_obs[:, 0] > 0.75).float().mean()
    err_mean = (xerr.mean(dim=1).mean() * noise_scale).reshape(())
    quantiles = torch.quantile(z_obs[:, 0], torch.tensor([0.1, 0.5, 0.9], dtype=z_obs.dtype))
    return torch.stack(
        [
            z_obs.mean(),
            z_obs.std(unbiased=False),
            quantiles[0],
            quantiles[1],
            quantiles[2],
            high_z,
            color0.mean(),
            color1.mean(),
            corr.clamp(-5.0, 5.0),
            err_mean,
        ]
    )


def generate_photoz_proxy_experiments(
    x_pool: torch.Tensor,
    xerr_pool: torch.Tensor,
    y_pool: torch.Tensor,
    *,
    n_experiments: int,
    catalog_size: int,
    delta_z_range: tuple[float, float],
    noise_scale_range: tuple[float, float],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    params = _sample_catalog_params(
        n_experiments,
        delta_z_range=delta_z_range,
        noise_scale_range=noise_scale_range,
        generator=generator,
    )
    summaries: list[torch.Tensor] = []
    n_pool = x_pool.shape[0]
    for delta_z, noise_scale in params.tolist():
        idx = torch.randint(0, n_pool, (catalog_size,), generator=generator)
        summaries.append(
            _summarize_photoz_catalog(
                x_pool[idx],
                xerr_pool[idx],
                y_pool[idx],
                delta_z=delta_z,
                noise_scale=noise_scale,
                generator=generator,
            )
        )
    return params.float(), torch.stack(summaries).float()


def main(
    cfg: Optional[ContrastivePhotoZProxyConfig] = None,
    summary_json_path: str | None = None,
) -> None:
    cfg = cfg or ContrastivePhotoZProxyConfig()
    set_comparison_seed(cfg.seed)

    splits = _make_splits(cfg)
    x_train_pool = torch.cat([splits["x_train"], splits["x_cal"]], dim=0)
    xerr_train_pool = torch.cat([splits["xerr_train"], splits["xerr_cal"]], dim=0)
    y_train_pool = torch.cat([splits["y_train"], splits["y_cal"]], dim=0)

    params_train, summaries_train = generate_photoz_proxy_experiments(
        x_train_pool,
        xerr_train_pool,
        y_train_pool,
        n_experiments=cfg.n_train_experiments,
        catalog_size=cfg.catalog_size,
        delta_z_range=cfg.delta_z_range,
        noise_scale_range=cfg.noise_scale_range,
        seed=cfg.seed,
    )
    params_test, summaries_test = generate_photoz_proxy_experiments(
        splits["x_test"],
        splits["xerr_test"],
        splits["y_test"],
        n_experiments=cfg.n_test_experiments,
        catalog_size=cfg.catalog_size,
        delta_z_range=cfg.delta_z_range,
        noise_scale_range=cfg.noise_scale_range,
        seed=cfg.seed + 1,
    )

    loader = DataLoader(
        TensorDataset(params_train, summaries_train),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    grid = ParameterGrid(
        lows=(cfg.delta_z_range[0], cfg.noise_scale_range[0]),
        highs=(cfg.delta_z_range[1], cfg.noise_scale_range[1]),
        steps=(25, 19),
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
            "Notes": f"photo-z proxy summaries from {_data_source_name(cfg)}",
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
    neg_generator = torch.Generator().manual_seed(cfg.seed + 23)

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
                "Notes": f"conditional flow on photo-z proxy summaries from {_data_source_name(cfg)}",
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
            negative_sampler=lambda batch_size: _sample_catalog_params(
                batch_size * cfg.n_negatives,
                delta_z_range=cfg.delta_z_range,
                noise_scale_range=cfg.noise_scale_range,
                generator=neg_generator,
            ).reshape(batch_size, cfg.n_negatives, 2),
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
                "Notes": f"contrastive flow on photo-z proxy summaries from {_data_source_name(cfg)}",
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
        title="Contrastive Flow Photo-z Proxy Comparison",
        seed_policy="fixed catalog resampling and shared parameter scan grid",
        train_budget=f"{cfg.epochs} epochs, batch_size={cfg.batch_size}, lr={cfg.lr}",
        metric_policy="parameter MAE on shared photo-z proxy pseudoexperiments + runtime",
    )
    print_comparison_summary(
        "Photo-z Proxy Parameter-Estimation Summary",
        summary_rows,
        metric_order=["MSE", "MAE", "R2", "ParamMAE", "Dim0_MAE", "Dim1_MAE", "train_s", "eval_s"],
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/contrastive_flow_photoz_proxy_comparison.py",
            task="Photo-z proxy parameter estimation under catalog shift",
            config=cfg,
            rows=summary_rows,
            notes=[
                f"Data source: {_data_source_name(cfg)}",
                "Catalog summaries are generated from real-or-simulated photo-z covariates with synthetic global shifts",
                "Flow rows are optional when zuko is unavailable",
            ],
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    main()
