from __future__ import annotations

import json
from pathlib import Path

from tools import photoz_benchmark_suite


def test_photoz_benchmark_suite_runs_core_examples(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "reports/example_summaries"

    def _fake_run_example(
        module_name: str,
        *,
        profile: str,
        output_dir: Path,
        real_data_only: bool,
        dataset_path: Path | None,
        train_dataset_path: Path | None,
        cal_dataset_path: Path | None,
        test_dataset_path: Path | None,
    ) -> Path:
        del real_data_only
        del dataset_path
        del train_dataset_path
        del cal_dataset_path
        del test_dataset_path
        path = output_dir / f"{module_name}_{profile}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "artifact": "comparison_example_summary",
                    "version": 1,
                    "example": f"examples/{module_name}.py",
                    "rows": [{"Method": "stub"}],
                }
            ),
            encoding="utf-8",
        )
        return path

    monkeypatch.setattr(photoz_benchmark_suite, "_run_example", _fake_run_example)

    report = photoz_benchmark_suite.run_suite(profile="smoke", output_dir=output_dir)

    assert report["artifact"] == "photoz_benchmark_suite_report"
    summary_paths = report["summary_paths"]
    assert isinstance(summary_paths, dict)
    assert set(summary_paths) == {
        "photoz_benchmark_comparison",
        "photoz_nnc_crps_rail_comparison",
        "ppi_photoz_inference_comparison",
    }
    assert Path(report["markdown_report_path"]).exists()
    for path_str in summary_paths.values():
        assert Path(path_str).exists()
    assert report["rail_merge"] is None


def test_photoz_benchmark_suite_can_include_rail_merge(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "reports/example_summaries"
    manifest_path = tmp_path / "rail_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    def _fake_run_example(
        module_name: str,
        *,
        profile: str,
        output_dir: Path,
        real_data_only: bool,
        dataset_path: Path | None,
        train_dataset_path: Path | None,
        cal_dataset_path: Path | None,
        test_dataset_path: Path | None,
    ) -> Path:
        del real_data_only
        del dataset_path
        del train_dataset_path
        del cal_dataset_path
        del test_dataset_path
        path = output_dir / f"{module_name}_{profile}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "artifact": "comparison_example_summary",
                    "version": 1,
                    "rows": [{"Method": module_name}],
                }
            ),
            encoding="utf-8",
        )
        return path

    def _fake_run_pipeline(**kwargs):
        merged = kwargs["output_dir"] / "photoz_rail_baseline_comparison_smoke.json"
        merged.write_text(
            json.dumps(
                {
                    "artifact": "comparison_example_summary",
                    "rows": [{"Method": "flexzboost"}],
                }
            ),
            encoding="utf-8",
        )
        return {
            "artifact": "photoz_rail_pipeline_report",
            "merged_output_path": str(merged),
        }

    monkeypatch.setattr(photoz_benchmark_suite, "_run_example", _fake_run_example)
    monkeypatch.setattr(
        photoz_benchmark_suite.photoz_rail_pipeline,
        "run_pipeline",
        _fake_run_pipeline,
    )

    report = photoz_benchmark_suite.run_suite(
        profile="smoke",
        output_dir=output_dir,
        include_rail_merge=True,
        manifest_path=manifest_path,
    )

    rail_merge = report["rail_merge"]
    assert isinstance(rail_merge, dict)
    assert Path(rail_merge["report_path"]).exists()
    assert Path(rail_merge["merged_output_path"]).exists()


def test_photoz_benchmark_suite_real_data_only_skips_ppi(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "reports/example_summaries"
    seen: list[str] = []

    def _fake_run_example(
        module_name: str,
        *,
        profile: str,
        output_dir: Path,
        real_data_only: bool,
        dataset_path: Path | None,
        train_dataset_path: Path | None,
        cal_dataset_path: Path | None,
        test_dataset_path: Path | None,
    ) -> Path:
        assert real_data_only is True
        assert dataset_path is None
        assert train_dataset_path is None
        assert cal_dataset_path is None
        assert test_dataset_path is None
        seen.append(module_name)
        path = output_dir / f"{module_name}_{profile}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "artifact": "comparison_example_summary",
                    "version": 1,
                    "rows": [{"Method": module_name}],
                }
            ),
            encoding="utf-8",
        )
        return path

    monkeypatch.setattr(photoz_benchmark_suite, "_run_example", _fake_run_example)

    report = photoz_benchmark_suite.run_suite(
        profile="smoke",
        output_dir=output_dir,
        real_data_only=True,
    )

    assert seen == [
        "photoz_benchmark_comparison",
        "photoz_nnc_crps_rail_comparison",
    ]
    assert report["skipped_examples"] == ["ppi_photoz_inference_comparison"]
    assert set(report["summary_paths"]) == set(seen)


def test_photoz_benchmark_suite_passes_dataset_path(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "reports/example_summaries"
    dataset_path = tmp_path / "nnc_photoz_real.csv"
    dataset_path.write_text(
        "spec_z,spec_z_err,u,g,r,i,z_mag,u_err,g_err,r_err,i_err,z_mag_err,u_g,g_r,r_i,i_z,u_g_err,g_r_err,r_i_err,i_z_err\n",
        encoding="utf-8",
    )
    seen: list[tuple[str, Path | None]] = []

    def _fake_run_example(
        module_name: str,
        *,
        profile: str,
        output_dir: Path,
        real_data_only: bool,
        dataset_path: Path | None,
        train_dataset_path: Path | None,
        cal_dataset_path: Path | None,
        test_dataset_path: Path | None,
    ) -> Path:
        assert real_data_only is True
        assert train_dataset_path is None
        assert cal_dataset_path is None
        assert test_dataset_path is None
        seen.append((module_name, dataset_path))
        path = output_dir / f"{module_name}_{profile}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "artifact": "comparison_example_summary",
                    "version": 1,
                    "rows": [{"Method": module_name}],
                }
            ),
            encoding="utf-8",
        )
        return path

    monkeypatch.setattr(photoz_benchmark_suite, "_run_example", _fake_run_example)

    report = photoz_benchmark_suite.run_suite(
        profile="smoke",
        output_dir=output_dir,
        real_data_only=True,
        dataset_path=dataset_path,
    )

    assert report["dataset_path"] == str(dataset_path)
    assert seen == [
        ("photoz_benchmark_comparison", dataset_path),
        ("photoz_nnc_crps_rail_comparison", dataset_path),
    ]


def test_photoz_benchmark_suite_passes_explicit_split_paths(
    monkeypatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "reports/example_summaries"
    train_path = tmp_path / "transferz_train.csv"
    cal_path = tmp_path / "transferz_cal.csv"
    test_path = tmp_path / "transferz_test.csv"
    header = "spec_z,spec_z_err,g_r,r_i,i_z,z_y,g_r_err,r_i_err,i_z_err,z_y_err\n"
    for path in (train_path, cal_path, test_path):
        path.write_text(header, encoding="utf-8")

    seen: list[tuple[str, Path | None, Path | None, Path | None]] = []

    def _fake_run_example(
        module_name: str,
        *,
        profile: str,
        output_dir: Path,
        real_data_only: bool,
        dataset_path: Path | None,
        train_dataset_path: Path | None,
        cal_dataset_path: Path | None,
        test_dataset_path: Path | None,
    ) -> Path:
        assert real_data_only is True
        assert dataset_path is None
        seen.append((module_name, train_dataset_path, cal_dataset_path, test_dataset_path))
        path = output_dir / f"{module_name}_{profile}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "artifact": "comparison_example_summary",
                    "version": 1,
                    "rows": [{"Method": module_name}],
                }
            ),
            encoding="utf-8",
        )
        return path

    monkeypatch.setattr(photoz_benchmark_suite, "_run_example", _fake_run_example)

    report = photoz_benchmark_suite.run_suite(
        profile="smoke",
        output_dir=output_dir,
        real_data_only=True,
        train_dataset_path=train_path,
        cal_dataset_path=cal_path,
        test_dataset_path=test_path,
    )

    assert report["train_dataset_path"] == str(train_path)
    assert report["cal_dataset_path"] == str(cal_path)
    assert report["test_dataset_path"] == str(test_path)
    assert seen == [
        ("photoz_benchmark_comparison", train_path, cal_path, test_path),
        ("photoz_nnc_crps_rail_comparison", train_path, cal_path, test_path),
    ]
