"""CatBoost RMSEWithUncertainty baselines for SAGE paper ceilings (Year + optional Higgs)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "examples" / "benchmarks"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import self_agreement_higgs_ood as higgs_mod  # noqa: E402
import self_agreement_realdata_year as year_mod  # noqa: E402


def _require_catboost() -> tuple[Any, Any]:
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Install catboost: uv pip install catboost") from e
    return CatBoostClassifier, CatBoostRegressor


def _gaussian_nll(y: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    var = np.maximum(var, 1e-8)
    return float(0.5 * np.mean(np.log(2.0 * np.pi * var) + (y - mu) ** 2 / var))


def _train_val_idx(n: int, seed: int, val_frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    val = perm[:n_val]
    train = perm[n_val:]
    if train.size == 0:
        train, val = perm[1:], perm[:1]
    return train, val


def run_year_catboost(
    *,
    cache_path: Path,
    out_dir: Path,
    n_labeled_list: list[int],
    seeds: list[int],
    iterations: int,
) -> None:
    _, CatBoostRegressor = _require_catboost()
    rows: list[dict[str, Any]] = []
    nu, nt = 131_072, 32_768
    for seed in seeds:
        for nl in n_labeled_list:
            cfg = year_mod.YearRealDataConfig(
                seed=seed,
                cache_path=str(cache_path),
                allow_download=False,
                n_labeled=nl,
                n_unlabeled=nu,
                n_test=nt,
            )
            split = year_mod._make_split(cfg)
            x = split.x_labeled.cpu().numpy()
            y = split.y_labeled.cpu().numpy().reshape(-1)
            xt = split.x_test.cpu().numpy()
            yt = split.y_test.cpu().numpy().reshape(-1)

            tr_idx, va_idx = _train_val_idx(x.shape[0], seed + nl)
            x_tr, y_tr = x[tr_idx], y[tr_idx]
            x_va, y_va = x[va_idx], y[va_idx]

            reg = CatBoostRegressor(
                loss_function="RMSEWithUncertainty",
                iterations=iterations,
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
                early_stopping_rounds=100,
            )
            reg.fit(x_tr, y_tr, eval_set=(x_va, y_va), use_best_model=True)
            pred = reg.predict(xt, prediction_type="RMSEWithUncertainty")
            mu = pred[:, 0].astype(np.float64)
            sig2 = pred[:, 1].astype(np.float64)
            nll = _gaussian_nll(yt.astype(np.float64), mu, sig2)
            rmse = float(np.sqrt(np.mean((yt - mu) ** 2)))
            rows.append(
                {
                    "Track": "year_catboost_labeled_only",
                    "Seed": seed,
                    "n_labeled": nl,
                    "RMSE_test": rmse,
                    "NLL_test_gaussian": nll,
                    "CatBoost_best_iteration": int(reg.get_best_iteration() or iterations),
                }
            )
            print(f"[Year CatBoost] seed={seed} nl={nl} RMSE={rmse:.4f} NLL={nll:.4f}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "year_catboost_labeled_only.json"
    p.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"Wrote {p}", flush=True)


def run_higgs_catboost(
    *,
    parquet_path: Path,
    out_dir: Path,
    seed: int,
    scale_factor: int,
    parquet_max_sample_rows: int,
    iterations: int,
    external_scores: dict[str, Any] | None,
) -> None:
    CatBoostClassifier, CatBoostRegressor = _require_catboost()
    base = higgs_mod.HiggsOODConfig(
        seed=seed,
        dataset_path=str(parquet_path),
        target_column="labels",
        ood_score_column="PRI_met",
        drop_columns=("weights", "detailed_labels"),
        parquet_max_sample_rows=parquet_max_sample_rows,
    )
    cfg = higgs_mod.higgs_scale_split_sizes(base, scale_factor)
    split = higgs_mod.make_split(cfg)

    def to_xy(x_t: Any, y_t: Any) -> tuple[np.ndarray, np.ndarray]:
        return x_t.cpu().numpy(), y_t.cpu().numpy().reshape(-1)

    x_tr, y_tr = to_xy(split.x_train, split.y_train)
    x_id, y_id = to_xy(split.x_id_test, split.y_id_test)
    x_ood, y_ood = to_xy(split.x_ood_test, split.y_ood_test)
    y_tr_c = (y_tr > 0.5).astype(np.int32)
    y_id_c = (y_id > 0.5).astype(np.int32)
    y_ood_c = (y_ood > 0.5).astype(np.int32)
    tr_i, va_i = _train_val_idx(x_tr.shape[0], seed, 0.05)

    clf = CatBoostClassifier(
        iterations=iterations,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        early_stopping_rounds=100,
    )
    clf.fit(
        x_tr[tr_i],
        y_tr_c[tr_i],
        eval_set=(x_tr[va_i], y_tr_c[va_i]),
        use_best_model=True,
    )
    proba_id = clf.predict_proba(x_id)[:, 1]
    proba_ood = clf.predict_proba(x_ood)[:, 1]
    try:
        from sklearn.metrics import log_loss  # noqa: PLC0415

        logloss_id = float(
            log_loss(y_id_c, np.column_stack([1.0 - proba_id, proba_id]), labels=[0, 1])
        )
        logloss_ood = float(
            log_loss(y_ood_c, np.column_stack([1.0 - proba_ood, proba_ood]), labels=[0, 1])
        )
    except Exception:
        logloss_id = float(
            -np.mean(y_id_c * np.log(np.clip(proba_id, 1e-8, 1 - 1e-8)))
            - np.mean((1 - y_id_c) * np.log(np.clip(1 - proba_id, 1e-8, 1 - 1e-8)))
        )
        logloss_ood = float(
            -np.mean(y_ood_c * np.log(np.clip(proba_ood, 1e-8, 1 - 1e-8)))
            - np.mean((1 - y_ood_c) * np.log(np.clip(1 - proba_ood, 1e-8, 1 - 1e-8)))
        )

    reg = CatBoostRegressor(
        loss_function="RMSEWithUncertainty",
        iterations=iterations,
        random_seed=seed + 17,
        verbose=False,
        allow_writing_files=False,
        early_stopping_rounds=100,
    )
    reg.fit(
        x_tr[tr_i],
        y_tr[tr_i].astype(np.float64),
        eval_set=(x_tr[va_i], y_tr[va_i].astype(np.float64)),
        use_best_model=True,
    )
    p_id = reg.predict(x_id, prediction_type="RMSEWithUncertainty")
    p_ood = reg.predict(x_ood, prediction_type="RMSEWithUncertainty")
    summary = {
        "Track": "higgs_catboost_baselines",
        "seed": seed,
        "scale_factor": scale_factor,
        "classifier": {"logloss_id": logloss_id, "logloss_ood": logloss_ood},
        "regressor_rmse_uncertainty": {
            "rmse_id": float(np.sqrt(np.mean((y_id - p_id[:, 0]) ** 2))),
            "rmse_ood": float(np.sqrt(np.mean((y_ood - p_ood[:, 0]) ** 2))),
            "nll_gaussian_id": _gaussian_nll(y_id.astype(np.float64), p_id[:, 0], p_id[:, 1]),
            "nll_gaussian_ood": _gaussian_nll(y_ood.astype(np.float64), p_ood[:, 0], p_ood[:, 1]),
        },
        "winning_solution_reference": external_scores,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "higgs_catboost_baselines.json"
    p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Higgs CatBoost] {p}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--year-cache", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-labeled", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768])
    p.add_argument("--seeds", type=int, nargs="+", default=[260410, 260411, 260412])
    p.add_argument("--iterations", type=int, default=4000)
    p.add_argument("--higgs-parquet", type=Path, default=None)
    p.add_argument("--higgs-seed", type=int, default=260410)
    p.add_argument("--higgs-scale-factor", type=int, default=10)
    p.add_argument("--higgs-parquet-max-rows", type=int, default=10_000_000)
    args = p.parse_args()
    run_year_catboost(
        cache_path=args.year_cache,
        out_dir=args.out_dir,
        n_labeled_list=list(args.n_labeled),
        seeds=list(args.seeds),
        iterations=args.iterations,
    )
    if args.higgs_parquet is not None and args.higgs_parquet.is_file():
        run_higgs_catboost(
            parquet_path=args.higgs_parquet,
            out_dir=args.out_dir,
            seed=args.higgs_seed,
            scale_factor=args.higgs_scale_factor,
            parquet_max_sample_rows=args.higgs_parquet_max_rows,
            iterations=args.iterations,
            external_scores=None,
        )


if __name__ == "__main__":
    main()
