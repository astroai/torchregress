import argparse
import csv
import os
from typing import Dict, List

import torch

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

from tail_extremes_benchmark import (
    build_methods,
    compute_metrics,
    predict,
    split_data,
    train_model,
)


def parse_float_list(value: str) -> List[float]:
    if not value:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def run_sweep(
    feature_noise_levels: List[float],
    label_noise_levels: List[float],
    train_size: int,
    test_size: int,
    epochs: int,
    batch_size: int,
    noise_scale: float,
    tail_quantile: float,
    seed: int,
    device: torch.device,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []

    for feature_noise in feature_noise_levels:
        for label_noise in label_noise_levels:
            (
                (x_train, y_obs_train, _),
                (x_test, _, y_true_test, tail_mask),
                y_sigma_mean,
            ) = split_data(
                train_size,
                test_size,
                noise_scale,
                feature_noise,
                label_noise,
                tail_quantile,
                seed,
            )

            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    x_train,
                    y_obs_train,
                    torch.zeros_like(y_obs_train),
                    torch.arange(x_train.shape[0]),
                ),
                batch_size=batch_size,
                shuffle=True,
            )

            methods = build_methods(y_obs_train, feature_noise, float(y_sigma_mean))
            for method in methods:
                model, loss_fn, is_gaussian, needs_indices = method.build()
                if hasattr(loss_fn, "model"):
                    model = loss_fn.model
                train_model(model, loss_fn, train_loader, epochs, device, use_indices=needs_indices)
                model.eval()
                with torch.no_grad():
                    preds = predict(model.to(device), x_test.to(device), is_gaussian)
                metrics = compute_metrics(preds, y_true_test.to(device), tail_mask.to(device))
                metrics["method"] = method.name
                metrics["feature_noise"] = feature_noise
                metrics["label_noise"] = label_noise
                results.append(metrics)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep tail performance over noise settings.")
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-scale", type=float, default=0.1)
    parser.add_argument("--feature-noise-list", type=str, default="0.0,0.05,0.1")
    parser.add_argument("--label-noise-list", type=str, default="0.0,0.2,0.4")
    parser.add_argument("--tail-quantile", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--only-clean-inputs", action="store_true")
    parser.add_argument("--output-csv", type=str, default="")
    args = parser.parse_args()

    feature_noise_levels = parse_float_list(args.feature_noise_list)
    label_noise_levels = parse_float_list(args.label_noise_list)

    if args.only_clean_inputs:
        feature_noise_levels = [0.0]

    device = torch.device(args.device)
    results = run_sweep(
        feature_noise_levels,
        label_noise_levels,
        args.train_size,
        args.test_size,
        args.epochs,
        args.batch_size,
        args.noise_scale,
        args.tail_quantile,
        args.seed,
        device,
    )

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        fieldnames = [
            "feature_noise",
            "label_noise",
            "method",
            "rmse",
            "mae",
            "tail_rmse",
            "tail_mae",
            "tail_rmse_ratio",
        ]
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    # Print best method by tail_rmse for each scenario
    scenarios: Dict[tuple, Dict[str, float]] = {}
    for row in results:
        key = (row["feature_noise"], row["label_noise"])
        best = scenarios.get(key)
        if best is None or row["tail_rmse"] < best["tail_rmse"]:
            scenarios[key] = row

    for (feature_noise, label_noise), best in sorted(scenarios.items()):
        print(
            f"feature_noise={feature_noise:.3f} label_noise={label_noise:.3f} "
            f"best={best['method']} tail_rmse={best['tail_rmse']:.4f}"
        )


if __name__ == "__main__":
    main()
