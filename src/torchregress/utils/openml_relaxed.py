"""OpenML fetch fallback when scikit-learn's MD5 check fails (stale catalog metadata).

Some OpenML datasets (historically id **42705** / Yolanda; others) can report an MD5
that no longer matches the hosted ARFF. :func:`sklearn.datasets.fetch_openml` then raises
``ValueError: md5 checksum ...``. This module downloads the same URL and parses the
ARFF without verifying that checksum, using the public JSON API for column metadata.

See also: ``docs/research/paper_strong_experiment_suite.md`` (OpenML / infra note).
"""

from __future__ import annotations

import gzip
import io
import json
import warnings
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from scipy.io import arff as scipy_arff

_OPENML_DATA = "https://api.openml.org/api/v1/json/data/{}"
_OPENML_FEATURES = "https://api.openml.org/api/v1/json/data/features/{}"


def _http_json(url: str, *, timeout: float) -> dict[str, Any]:
    req = Request(url, headers={"User-Agent": "torchregress-openml-relaxed/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return cast(dict[str, Any], json.loads(raw.decode("utf-8")))


def _feature_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    feats = payload["data_features"]["feature"]
    return feats if isinstance(feats, list) else [feats]


def _truthy_openml_flag(value: object) -> bool:
    return str(value).lower() == "true"


def _openml_dataset_description(data_id: int, *, timeout: float = 60.0) -> dict[str, Any]:
    payload = _http_json(_OPENML_DATA.format(int(data_id)), timeout=timeout)
    return cast(dict[str, Any], payload["data_set_description"])


def _openml_feature_list(data_id: int, *, timeout: float = 60.0) -> list[dict[str, Any]]:
    payload = _http_json(_OPENML_FEATURES.format(int(data_id)), timeout=timeout)
    return _feature_rows(payload)


def _download_bytes(url: str, *, timeout: float) -> bytes:
    req = Request(url, headers={"Accept-encoding": "gzip", "User-Agent": "torchregress/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        out = resp.read()
    return bytes(out)


def _load_arff_frame(arff_bytes: bytes) -> pd.DataFrame:
    """Parse ARFF bytes; scipy expects a text stream."""
    if len(arff_bytes) >= 2 and arff_bytes[0] == 0x1F and arff_bytes[1] == 0x8B:
        raw = gzip.decompress(arff_bytes)
    else:
        raw = arff_bytes
    text_stream = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", newline="")
    data, _meta = scipy_arff.loadarff(text_stream)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(1)
            if not sample.empty and isinstance(sample.iloc[0], (bytes, bytearray)):
                df[col] = df[col].apply(
                    lambda x: (
                        x.decode("utf-8", errors="replace")
                        if isinstance(x, (bytes, bytearray))
                        else x
                    )
                )
    return df


def fetch_openml_regression_frame_skip_checksum(
    *,
    data_id: int,
    target_column: str = "target",
    download_timeout: float = 600.0,
    json_timeout: float = 120.0,
) -> tuple[pd.DataFrame, str]:
    """Load numeric regression frame for ``data_id`` without MD5 verification.

    Matches the post-processing used in ``examples/spt_reg_year_comparison.py`` /
    ``self_agreement_realdata_year.py``: numeric/bool features as float32 and a
    single numeric target stored under ``target_column``.
    """
    desc = _openml_dataset_description(data_id, timeout=json_timeout)
    fmt = str(desc.get("format", "")).lower()
    if "sparse" in fmt:
        raise ValueError(
            f"OpenML data_id={data_id} is {fmt!r}; sparse ARFF is not supported by "
            "torchregress.utils.openml_relaxed (use sklearn or materialize locally)."
        )

    features = _openml_feature_list(data_id, timeout=json_timeout)
    target_names = [f["name"] for f in features if _truthy_openml_flag(f.get("is_target"))]
    if len(target_names) != 1:
        raise ValueError(
            f"OpenML data_id={data_id}: expected exactly one target column, got {target_names!r}"
        )
    raw_target = target_names[0]

    ignore = {
        f["name"]
        for f in features
        if _truthy_openml_flag(f.get("is_ignore"))
        or _truthy_openml_flag(f.get("is_row_identifier"))
    }

    url = str(desc["url"])
    try:
        blob = _download_bytes(url, timeout=download_timeout)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Failed to download OpenML ARFF from {url!r}") from exc

    full = _load_arff_frame(blob)
    if raw_target not in full.columns:
        cols = list(full.columns)
        raise ValueError(f"Target column {raw_target!r} missing from ARFF columns {cols}")

    feature_cols = [c for c in full.columns if c not in ignore and c != raw_target]
    numeric = full[feature_cols].select_dtypes(include=["number", "bool"]).copy()
    if numeric.empty:
        raise ValueError("OpenML ARFF has no numeric/bool feature columns after selection")

    for col in numeric.columns:
        if numeric[col].dtype == bool:
            numeric[col] = numeric[col].astype("float32")
        else:
            numeric[col] = numeric[col].astype("float32")

    target_series = pd.to_numeric(full[raw_target], errors="coerce")
    out = numeric.copy()
    out[target_column] = target_series.to_numpy(dtype=np.float32)
    out = out.dropna()
    if out.empty:
        raise ValueError("OpenML frame is empty after dropping NaN target/features")

    tag = f"OpenML:id={data_id} (relaxed_arff_no_md5)"
    return out, tag


def fetch_openml_regression_with_sklearn_fallback(
    *,
    data_id: int | None,
    name: str | None,
    version: int,
    target_column: str,
) -> tuple[pd.DataFrame, str]:
    """Try ``fetch_openml``; on MD5 mismatch for ``data_id``, use relaxed ARFF parse."""
    from sklearn.datasets import fetch_openml

    if data_id is not None:
        try:
            bunch = fetch_openml(data_id=int(data_id), as_frame=True)
        except ValueError as exc:
            if "md5 checksum" not in str(exc).lower():
                raise
            warnings.warn(
                "sklearn.datasets.fetch_openml failed OpenML MD5 check; "
                "parsing ARFF without checksum verification (stale OpenML MD5 metadata). "
                "Prefer a verified local --cache-path when publishing numbers.",
                UserWarning,
                stacklevel=2,
            )
            return fetch_openml_regression_frame_skip_checksum(
                data_id=int(data_id),
                target_column=target_column,
            )
        tag = f"OpenML:id={data_id}"
    elif name is not None:
        try:
            bunch = fetch_openml(name=name, version=version, as_frame=True)
        except ValueError as exc:
            if "md5 checksum" not in str(exc).lower():
                raise
            warnings.warn(
                "sklearn.datasets.fetch_openml failed OpenML MD5 check; "
                "resolving dataset id via OpenML API and parsing ARFF without checksum.",
                UserWarning,
                stacklevel=2,
            )
            resolved = _resolve_data_id_by_name_version(
                name=name, version=int(version), timeout=120.0
            )
            return fetch_openml_regression_frame_skip_checksum(
                data_id=int(resolved),
                target_column=target_column,
            )
        tag = f"OpenML:{name}:v{version}"
    else:
        raise ValueError("openml regression fetch requires data_id or name")

    feats = cast(pd.DataFrame, bunch.data)
    numeric = feats.select_dtypes(include=["number", "bool"]).copy()
    if numeric.empty:
        raise ValueError("OpenML dataset has no numeric/bool feature columns after selection")
    for col in numeric.columns:
        if numeric[col].dtype == bool:
            numeric[col] = numeric[col].astype("float32")
        else:
            numeric[col] = numeric[col].astype("float32")

    target_series = pd.to_numeric(cast(pd.Series, bunch.target), errors="coerce")
    frame = numeric.copy()
    frame[target_column] = target_series.to_numpy(dtype=np.float32)
    frame = frame.dropna()
    if frame.empty:
        raise ValueError("OpenML frame is empty after dropping NaN target/features")
    return frame, tag


def _resolve_data_id_by_name_version(*, name: str, version: int, timeout: float) -> int:
    url = (
        "https://api.openml.org/api/v1/json/data/list/data_name/"
        f"{name}/limit/2/data_version/{version}"
    )
    payload = _http_json(url, timeout=timeout)
    data = payload.get("data", {})
    datasets = data.get("dataset", [])
    if isinstance(datasets, dict):
        datasets = [datasets]
    if not datasets:
        raise ValueError(f"No OpenML dataset found for name={name!r} version={version}")
    return int(datasets[0]["did"])
