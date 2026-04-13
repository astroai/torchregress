"""Tests for OpenML ARFF fallback (stale MD5 metadata)."""

from __future__ import annotations

import pytest

from torchregress.utils import openml_relaxed as m


def test_load_arff_frame_plain() -> None:
    arff = b"""@relation t
@attribute a numeric
@attribute b numeric
@data
1.0,2.0
3.0,4.0
"""
    df = m._load_arff_frame(arff)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_fetch_openml_regression_frame_skip_checksum_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_desc(*args: object, **kwargs: object) -> dict[str, str]:
        return {"format": "ARFF", "url": "http://dummy.local/y.arff"}

    def fake_features(*args: object, **kwargs: object) -> list[dict[str, str]]:
        return [
            {
                "name": "a",
                "is_target": "false",
                "is_ignore": "false",
                "is_row_identifier": "false",
            },
            {
                "name": "b",
                "is_target": "true",
                "is_ignore": "false",
                "is_row_identifier": "false",
            },
        ]

    arff = b"""@relation t
@attribute a numeric
@attribute b numeric
@data
1.0,2.0
3.0,4.0
"""
    monkeypatch.setattr(m, "_openml_dataset_description", fake_desc)
    monkeypatch.setattr(m, "_openml_feature_list", fake_features)
    monkeypatch.setattr(m, "_download_bytes", lambda url, timeout=0.0: arff)

    frame, tag = m.fetch_openml_regression_frame_skip_checksum(data_id=1, target_column="target")
    assert len(frame) == 2
    assert "target" in frame.columns
    assert "relaxed_arff_no_md5" in tag


def test_fetch_openml_regression_with_sklearn_fallback_on_md5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd
    import sklearn.datasets

    calls: list[int] = []

    def boom(*args: object, **kwargs: object) -> None:
        calls.append(1)
        raise ValueError(
            "md5 checksum of local file for https://openml.org/x does not match description: "
            "expected: aaa but got bbb. Downloaded file could have been modified"
        )

    def relaxed(**kwargs: object) -> tuple[object, str]:
        calls.append(2)
        assert kwargs["data_id"] == 42225
        return pd.DataFrame({"f0": [1.0], "target": [2.0]}), "relaxed"

    monkeypatch.setattr(sklearn.datasets, "fetch_openml", boom)
    monkeypatch.setattr(m, "fetch_openml_regression_frame_skip_checksum", relaxed)

    frame, tag = m.fetch_openml_regression_with_sklearn_fallback(
        data_id=42225,
        name=None,
        version=1,
        target_column="target",
    )
    assert calls == [1, 2]
    assert len(frame) == 1
    assert tag == "relaxed"


def test_fetch_openml_regression_with_sklearn_fallback_passes_other_valueerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sklearn.datasets

    def boom(*args: object, **kwargs: object) -> None:
        raise ValueError("something else entirely")

    monkeypatch.setattr(sklearn.datasets, "fetch_openml", boom)

    with pytest.raises(ValueError, match="something else"):
        m.fetch_openml_regression_with_sklearn_fallback(
            data_id=1,
            name=None,
            version=1,
            target_column="target",
        )
