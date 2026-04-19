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


def test_fetch_openml_regression_frame_skip_checksum_sparse_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_desc(*args: object, **kwargs: object) -> dict[str, str]:
        return {"format": "sparse_ARFF", "url": "http://dummy.local/y.arff"}

    monkeypatch.setattr(m, "_openml_dataset_description", fake_desc)

    with pytest.raises(ValueError, match="sparse ARFF is not supported"):
        m.fetch_openml_regression_frame_skip_checksum(data_id=1)


def test_fetch_openml_regression_frame_skip_checksum_multiple_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_desc(*args: object, **kwargs: object) -> dict[str, str]:
        return {"format": "ARFF", "url": "http://dummy.local/y.arff"}

    def fake_features(*args: object, **kwargs: object) -> list[dict[str, str]]:
        return [{"name": "t1", "is_target": "true"}, {"name": "t2", "is_target": "true"}]

    monkeypatch.setattr(m, "_openml_dataset_description", fake_desc)
    monkeypatch.setattr(m, "_openml_feature_list", fake_features)

    with pytest.raises(ValueError, match="expected exactly one target column"):
        m.fetch_openml_regression_frame_skip_checksum(data_id=1)


def test_fetch_openml_regression_frame_skip_checksum_download_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from urllib.error import URLError

    def fake_desc(*args: object, **kwargs: object) -> dict[str, str]:
        return {"format": "ARFF", "url": "http://dummy.local/y.arff"}

    def fake_features(*args: object, **kwargs: object) -> list[dict[str, str]]:
        return [{"name": "target", "is_target": "true"}]

    def fake_download(*args: object, **kwargs: object) -> bytes:
        raise URLError("Not found")

    monkeypatch.setattr(m, "_openml_dataset_description", fake_desc)
    monkeypatch.setattr(m, "_openml_feature_list", fake_features)
    monkeypatch.setattr(m, "_download_bytes", fake_download)

    with pytest.raises(RuntimeError, match="Failed to download OpenML ARFF"):
        m.fetch_openml_regression_frame_skip_checksum(data_id=1)


def test_fetch_openml_regression_frame_skip_checksum_missing_target_col(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    def fake_desc(*args: object, **kwargs: object) -> dict[str, str]:
        return {"format": "ARFF", "url": "http://dummy.local/y.arff"}

    def fake_features(*args: object, **kwargs: object) -> list[dict[str, str]]:
        return [{"name": "target", "is_target": "true"}]

    def fake_download(*args: object, **kwargs: object) -> bytes:
        return b""

    def fake_load_arff(*args: object, **kwargs: object) -> pd.DataFrame:
        return pd.DataFrame({"a": [1.0], "b": [2.0]})

    monkeypatch.setattr(m, "_openml_dataset_description", fake_desc)
    monkeypatch.setattr(m, "_openml_feature_list", fake_features)
    monkeypatch.setattr(m, "_download_bytes", fake_download)
    monkeypatch.setattr(m, "_load_arff_frame", fake_load_arff)

    with pytest.raises(ValueError, match="Target column 'target' missing from ARFF columns"):
        m.fetch_openml_regression_frame_skip_checksum(data_id=1)


def test_fetch_openml_regression_frame_skip_checksum_no_numeric_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    def fake_desc(*args: object, **kwargs: object) -> dict[str, str]:
        return {"format": "ARFF", "url": "http://dummy.local/y.arff"}

    def fake_features(*args: object, **kwargs: object) -> list[dict[str, str]]:
        return [{"name": "target", "is_target": "true"}]

    def fake_download(*args: object, **kwargs: object) -> bytes:
        return b""

    def fake_load_arff(*args: object, **kwargs: object) -> pd.DataFrame:
        return pd.DataFrame({"a": ["str1", "str2"], "target": [1.0, 2.0]})

    monkeypatch.setattr(m, "_openml_dataset_description", fake_desc)
    monkeypatch.setattr(m, "_openml_feature_list", fake_features)
    monkeypatch.setattr(m, "_download_bytes", fake_download)
    monkeypatch.setattr(m, "_load_arff_frame", fake_load_arff)

    with pytest.raises(ValueError, match="no numeric/bool feature columns after selection"):
        m.fetch_openml_regression_frame_skip_checksum(data_id=1)


def test_fetch_openml_regression_frame_skip_checksum_empty_after_dropna(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    import pandas as pd

    def fake_desc(*args: object, **kwargs: object) -> dict[str, str]:
        return {"format": "ARFF", "url": "http://dummy.local/y.arff"}

    def fake_features(*args: object, **kwargs: object) -> list[dict[str, str]]:
        return [{"name": "target", "is_target": "true"}]

    def fake_download(*args: object, **kwargs: object) -> bytes:
        return b""

    def fake_load_arff(*args: object, **kwargs: object) -> pd.DataFrame:
        return pd.DataFrame({"a": [1.0, np.nan], "target": [np.nan, 2.0]})

    monkeypatch.setattr(m, "_openml_dataset_description", fake_desc)
    monkeypatch.setattr(m, "_openml_feature_list", fake_features)
    monkeypatch.setattr(m, "_download_bytes", fake_download)
    monkeypatch.setattr(m, "_load_arff_frame", fake_load_arff)

    with pytest.raises(
        ValueError, match="OpenML frame is empty after dropping NaN target/features"
    ):
        m.fetch_openml_regression_frame_skip_checksum(data_id=1)
