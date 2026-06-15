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
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from scipy.io import arff as scipy_arff

from torchregress.utils.security import validate_url

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


def _load_arff_arrays(arff_bytes: bytes) -> tuple[np.ndarray, list[str]]:
    """Parse ARFF bytes and return (structured_array, column_names)."""
    if len(arff_bytes) >= 2 and arff_bytes[0] == 0x1F and arff_bytes[1] == 0x8B:
        raw = gzip.decompress(arff_bytes)
    else:
        raw = arff_bytes
    text_stream = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", newline="")
    data, _meta = scipy_arff.loadarff(text_stream)

    # Decode byte-string columns to ordinary Python strings
    col_names = list(data.dtype.names)
    for col in col_names:
        if data[col].dtype.kind in ("S", "a"):
            data[col] = np.array(
                [
                    v.decode("utf-8", errors="replace") if isinstance(v, (bytes, bytearray)) else v
                    for v in data[col]
                ]
            )
    return data, col_names


def _numeric_mask(ar: np.ndarray) -> np.ndarray:
    """Boolean mask for values that are numeric, boolean, or numeric strings."""
    if ar.dtype.kind in ("f", "i", "u", "b"):
        return np.ones(len(ar), dtype=bool)
    return np.array(
        [
            isinstance(v, (int, float, np.integer, np.floating, np.bool_))
            or (isinstance(v, str) and _is_numeric_string(v))
            for v in ar
        ]
    )


def _is_numeric_string(s: str) -> bool:
    """Check if a string can be converted to a float."""
    try:
        float(s.strip() or "NaN")
        return True
    except ValueError:
        return False


def _to_float32(ar: np.ndarray) -> np.ndarray:
    """Convert array to float32, coercing non-numeric to NaN.

    Handles boolean, numeric, and numeric-string values.
    """
    if ar.dtype.kind in ("f", "i", "u"):
        return ar.astype(np.float32)
    if ar.dtype.kind == "b":
        return ar.astype(np.float32)

    converted = np.full(ar.shape, np.nan, dtype=np.float32)
    mask = _numeric_mask(ar)
    converted[mask] = np.array([float(v) for v in ar[mask]], dtype=np.float32)
    return converted


def fetch_openml_regression_frame_skip_checksum(
    *,
    data_id: int,
    target_column: str = "target",
    download_timeout: float = 600.0,
    json_timeout: float = 120.0,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load numeric regression arrays for ``data_id`` without MD5 verification.

    Returns ``(X, y, tag)`` where ``X`` and ``y`` are float32 numpy arrays.
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

    url = validate_url(str(desc["url"]))
    try:
        blob = _download_bytes(url, timeout=download_timeout)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Failed to download OpenML ARFF from {url!r}") from exc

    full, cols = _load_arff_arrays(blob)
    if raw_target not in cols:
        raise ValueError(f"Target column {raw_target!r} missing from ARFF columns {cols}")

    feature_cols = [c for c in cols if c not in ignore and c != raw_target]

    # Select numeric feature columns and convert to float32
    X_parts = []
    for col in feature_cols:
        ar = full[col]
        if ar.dtype.kind in ("f", "i", "u", "b"):
            xf = ar.astype(np.float32)
        else:
            xf = _to_float32(ar)
        X_parts.append(xf.reshape(-1, 1))

    if not X_parts:
        raise ValueError("OpenML ARFF has no numeric/bool feature columns after selection")

    X = np.concatenate(X_parts, axis=1).astype(np.float32)
    y = _to_float32(full[raw_target]).astype(np.float32)

    # Drop rows where X or y contains NaN
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X_out = X[valid].copy()
    y_out = y[valid].copy()
    if X_out.size == 0:
        raise ValueError("OpenML frame is empty after dropping NaN target/features")

    tag = f"OpenML:id={data_id} (relaxed_arff_no_md5)"
    return X_out, y_out, tag


def fetch_openml_regression_with_sklearn_fallback(
    *,
    data_id: int | None,
    name: str | None,
    version: int,
    target_column: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load OpenML regression data as (X, y, tag) numpy arrays.

    Always uses the relaxed ARFF parser directly (no sklearn dependency).
    The function name is kept for backward compatibility.
    """
    if data_id is not None:
        return fetch_openml_regression_frame_skip_checksum(
            data_id=int(data_id),
            target_column=target_column,
        )
    if name is not None:
        resolved = _resolve_data_id_by_name_version(name=name, version=int(version), timeout=120.0)
        return fetch_openml_regression_frame_skip_checksum(
            data_id=int(resolved),
            target_column=target_column,
        )
    raise ValueError("openml regression fetch requires data_id or name")


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
