"""Materialize photo-z RAIL manifest assets and update checksum metadata.

The manifest can define two optional collections:

- `dataset_files`: dataset/split assets used for parity and provenance
- `baseline_payloads`: RAIL baseline payload JSON files used by merge tooling
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "photoz"
DEFAULT_TEMPLATE_BY_PRESET = {
    "rail": CONFIG_DIR / "rail_photoz_manifest.template.json",
    "nnc_crps": CONFIG_DIR / "nnc_crps_photoz_manifest.template.json",
}


def _default_manifest(preset: str) -> dict[str, Any]:
    dataset_id = (
        "rail_photoz_paper_reference_dataset"
        if preset == "rail"
        else "nnc_crps_photoz_paper_reference_dataset"
    )
    split_id = (
        "rail_photoz_paper_reference_split_v1"
        if preset == "rail"
        else "nnc_crps_photoz_paper_reference_split_v1"
    )
    return {
        "artifact": "rail_photoz_manifest",
        "version": 1,
        "dataset_id": dataset_id,
        "split_id": split_id,
        "core_baselines": ["flexzboost", "pzflow", "delight", "bpz"],
        "optional_baselines": ["lephare"],
        "checksum_policy": {
            "train_catalog_sha256": "",
            "test_catalog_sha256": "",
            "calibration_catalog_sha256": "",
        },
        "notes": [
            "Populate URLs via CLI overrides or by editing manifest fields.",
            "paper-parity mode enforces dataset/split/core-baseline constraints.",
        ],
    }


def load_manifest(
    path: Path,
    *,
    preset: str = "rail",
    template_path: Path | None = None,
) -> dict[str, Any]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    candidate_template = template_path or DEFAULT_TEMPLATE_BY_PRESET.get(preset)
    if candidate_template is not None and candidate_template.exists():
        data = json.loads(candidate_template.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    return _default_manifest(preset)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def ensure_manifest_defaults(manifest: dict[str, Any]) -> None:
    checksum_policy = manifest.setdefault("checksum_policy", {})
    if not isinstance(checksum_policy, dict):
        raise ValueError("manifest.checksum_policy must be an object when provided.")

    dataset_files = manifest.get("dataset_files")
    if dataset_files is None:
        manifest["dataset_files"] = [
            {
                "key": "train_catalog_sha256",
                "path": "data/rail/datasets/train_catalog.parquet",
                "url": "",
                "required": True,
            },
            {
                "key": "test_catalog_sha256",
                "path": "data/rail/datasets/test_catalog.parquet",
                "url": "",
                "required": True,
            },
            {
                "key": "calibration_catalog_sha256",
                "path": "data/rail/datasets/calibration_catalog.parquet",
                "url": "",
                "required": True,
            },
        ]
        checksum_policy.setdefault("train_catalog_sha256", "")
        checksum_policy.setdefault("test_catalog_sha256", "")
        checksum_policy.setdefault("calibration_catalog_sha256", "")

    baseline_payloads = manifest.get("baseline_payloads")
    if baseline_payloads is None:
        core = [str(x) for x in manifest.get("core_baselines", []) if isinstance(x, str)]
        optional = [str(x) for x in manifest.get("optional_baselines", []) if isinstance(x, str)]
        payloads: list[dict[str, Any]] = []
        for method in core:
            payloads.append(
                {
                    "method": method,
                    "path": f"data/rail/baselines/{method}.json",
                    "url": "",
                    "required": True,
                }
            )
        for method in optional:
            if method in core:
                continue
            payloads.append(
                {
                    "method": method,
                    "path": f"data/rail/baselines/{method}.json",
                    "url": "",
                    "required": False,
                }
            )
        manifest["baseline_payloads"] = payloads


def apply_manifest_overrides(
    manifest: dict[str, Any],
    *,
    dataset_urls: dict[str, str] | None = None,
    baseline_urls: dict[str, str] | None = None,
    dataset_paths: dict[str, str] | None = None,
    baseline_paths: dict[str, str] | None = None,
) -> None:
    ensure_manifest_defaults(manifest)
    dataset_map: dict[str, dict[str, Any]] = {}
    baseline_map: dict[str, dict[str, Any]] = {}
    dataset_entries = manifest.get("dataset_files", [])
    baseline_entries = manifest.get("baseline_payloads", [])
    if isinstance(dataset_entries, list):
        for entry in dataset_entries:
            if isinstance(entry, dict) and isinstance(entry.get("key"), str):
                dataset_map[entry["key"]] = entry
    if isinstance(baseline_entries, list):
        for entry in baseline_entries:
            if isinstance(entry, dict) and isinstance(entry.get("method"), str):
                baseline_map[entry["method"]] = entry

    for key, url in (dataset_urls or {}).items():
        if key not in dataset_map:
            raise ValueError(f"Unknown dataset key override: {key}")
        dataset_map[key]["url"] = url
    for method, url in (baseline_urls or {}).items():
        if method not in baseline_map:
            raise ValueError(f"Unknown baseline method override: {method}")
        baseline_map[method]["url"] = url
    for key, path in (dataset_paths or {}).items():
        if key not in dataset_map:
            raise ValueError(f"Unknown dataset key path override: {key}")
        dataset_map[key]["path"] = path
    for method, path in (baseline_paths or {}).items():
        if method not in baseline_map:
            raise ValueError(f"Unknown baseline method path override: {method}")
        baseline_map[method]["path"] = path


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolve_path(path_value: str, *, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def _download_to_path(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, target.open("wb") as out:  # nosec: manifest-driven URL
        out.write(response.read())


def _materialize_entry(
    entry: dict[str, Any],
    *,
    repo_root: Path,
    allow_download: bool,
    overwrite: bool,
) -> tuple[Path | None, str]:
    path_value = entry.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("Each manifest entry requires a non-empty string `path`.")
    target = _resolve_path(path_value, repo_root=repo_root)
    required = bool(entry.get("required", True))
    url = entry.get("url")
    url_text = str(url).strip() if isinstance(url, str) else ""

    if target.exists() and not overwrite:
        return target, "existing"
    if allow_download and url_text:
        _download_to_path(url_text, target)
        return target, "downloaded"
    if target.exists():
        return target, "existing"
    if required:
        return None, "missing_required"
    return None, "missing_optional"


def materialize_manifest_assets(
    manifest: dict[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    allow_download: bool = True,
    overwrite: bool = False,
    strict_checksums: bool = False,
    fail_on_missing: bool = True,
) -> dict[str, Any]:
    ensure_manifest_defaults(manifest)
    checksum_policy = manifest.setdefault("checksum_policy", {})
    dataset_entries = manifest.get("dataset_files", [])
    baseline_entries = manifest.get("baseline_payloads", [])

    if not isinstance(dataset_entries, list):
        raise ValueError("manifest.dataset_files must be a list when provided.")
    if not isinstance(baseline_entries, list):
        raise ValueError("manifest.baseline_payloads must be a list when provided.")

    dataset_results: list[dict[str, Any]] = []
    baseline_results: list[dict[str, Any]] = []
    missing_required: list[str] = []

    for entry in dataset_entries:
        if not isinstance(entry, dict):
            raise ValueError("dataset_files entries must be objects.")
        key = entry.get("key")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("dataset_files entries require a non-empty `key`.")
        expected = checksum_policy.get(key)
        path, status = _materialize_entry(
            entry,
            repo_root=repo_root,
            allow_download=allow_download,
            overwrite=overwrite,
        )
        sha = file_sha256(path) if path is not None else None
        if path is not None:
            entry["sha256"] = sha
            checksum_policy[key] = sha
        if strict_checksums and isinstance(expected, str) and expected and sha is not None:
            if expected != sha:
                raise ValueError(
                    f"Checksum mismatch for dataset key `{key}`: expected {expected}, got {sha}"
                )
        if status == "missing_required":
            missing_required.append(f"dataset:{key}")
        dataset_results.append(
            {
                "key": key,
                "path": entry.get("path"),
                "status": status,
                "sha256": sha,
            }
        )

    for entry in baseline_entries:
        if not isinstance(entry, dict):
            raise ValueError("baseline_payloads entries must be objects.")
        method = entry.get("method")
        if not isinstance(method, str) or not method.strip():
            raise ValueError("baseline_payloads entries require non-empty `method`.")
        path, status = _materialize_entry(
            entry,
            repo_root=repo_root,
            allow_download=allow_download,
            overwrite=overwrite,
        )
        sha = file_sha256(path) if path is not None else None
        if path is not None:
            entry["sha256"] = sha
        if status == "missing_required":
            missing_required.append(f"baseline:{method}")
        baseline_results.append(
            {
                "method": method,
                "path": entry.get("path"),
                "status": status,
                "sha256": sha,
            }
        )

    if missing_required and fail_on_missing:
        raise FileNotFoundError(
            "Required manifest assets are missing and could not be materialized: "
            + ", ".join(sorted(missing_required))
        )

    return {
        "artifact": "photoz_rail_materialization_report",
        "version": 1,
        "allow_download": allow_download,
        "overwrite": overwrite,
        "strict_checksums": strict_checksums,
        "missing_required": missing_required,
        "dataset_results": dataset_results,
        "baseline_results": baseline_results,
    }


def collect_baseline_input_paths(
    manifest: dict[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    require_required: bool = True,
) -> list[Path]:
    ensure_manifest_defaults(manifest)
    entries = manifest.get("baseline_payloads", [])
    if not isinstance(entries, list):
        raise ValueError("manifest.baseline_payloads must be a list when provided.")
    paths: list[Path] = []
    missing: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        path_value = entry.get("path")
        method = entry.get("method", "unknown")
        required = bool(entry.get("required", True))
        if not isinstance(path_value, str) or not path_value.strip():
            if required:
                missing.append(str(method))
            continue
        path = _resolve_path(path_value, repo_root=repo_root)
        if path.exists():
            paths.append(path)
        elif required and require_required:
            missing.append(str(method))

    if missing and require_required:
        raise FileNotFoundError(
            "Required baseline payloads are missing: " + ", ".join(sorted(missing))
        )
    return paths


def _parse_overrides(items: list[str], *, label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid {label} override `{item}`. Expected KEY=VALUE format.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid {label} override `{item}`. Empty key/value is not allowed.")
        parsed[key] = value
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize photo-z RAIL manifest assets.")
    parser.add_argument(
        "--preset",
        choices=["rail", "nnc_crps"],
        default="rail",
        help="Manifest preset used if --manifest does not exist.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/rail/rail_photoz_manifest.json"),
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Optional template path used when --manifest does not exist.",
    )
    parser.add_argument(
        "--write-manifest",
        type=Path,
        default=None,
        help="Path to write updated manifest. Defaults to --manifest path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/example_summaries/photoz_rail_materialization_latest.json"),
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download missing assets even if URLs are present.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files when URL is provided.",
    )
    parser.add_argument(
        "--strict-checksums",
        action="store_true",
        help="Fail if existing checksum_policy entries do not match computed hashes.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Do not fail when required assets remain missing.",
    )
    parser.add_argument(
        "--dataset-url",
        action="append",
        default=[],
        metavar="KEY=URL",
        help="Override a dataset_files URL without editing manifest (repeatable).",
    )
    parser.add_argument(
        "--baseline-url",
        action="append",
        default=[],
        metavar="METHOD=URL",
        help="Override a baseline_payloads URL without editing manifest (repeatable).",
    )
    parser.add_argument(
        "--dataset-path",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="Override a dataset_files path without editing manifest (repeatable).",
    )
    parser.add_argument(
        "--baseline-path",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Override a baseline_payloads path without editing manifest (repeatable).",
    )
    args = parser.parse_args()

    manifest = load_manifest(
        args.manifest,
        preset=args.preset,
        template_path=args.template,
    )
    apply_manifest_overrides(
        manifest,
        dataset_urls=_parse_overrides(args.dataset_url, label="dataset-url"),
        baseline_urls=_parse_overrides(args.baseline_url, label="baseline-url"),
        dataset_paths=_parse_overrides(args.dataset_path, label="dataset-path"),
        baseline_paths=_parse_overrides(args.baseline_path, label="baseline-path"),
    )
    report = materialize_manifest_assets(
        manifest,
        allow_download=not args.no_download,
        overwrite=args.overwrite,
        strict_checksums=args.strict_checksums,
        fail_on_missing=not args.allow_missing,
    )

    output_manifest = args.write_manifest or args.manifest
    write_manifest(output_manifest, manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote updated manifest: {output_manifest}")
    print(f"Wrote materialization report: {args.report}")
    print(
        f"Dataset entries: {len(report['dataset_results'])}, "
        f"baselines: {len(report['baseline_results'])}, "
        f"missing_required: {len(report['missing_required'])}"
    )


if __name__ == "__main__":
    main()
