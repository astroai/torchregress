import importlib

import pytest

import torchregress as tr
from torchregress.losses import nflows


def test_top_level_import_remains_available_without_using_flow_api() -> None:
    # Guardrail for optional dependency behavior: importing torchregress should not require zuko.
    assert hasattr(tr, "losses")
    assert importlib.import_module("torchregress.losses") is not None


def test_flow_helpers_raise_clear_importerror_when_backend_marked_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(nflows, "HAS_ZUKO", False)

    with pytest.raises(ImportError, match="torchregress\\[flows\\]"):
        nflows.create_flow_model(n_features=2, context_dim=4)

    with pytest.raises(ImportError, match="torchregress\\[flows\\]"):
        nflows.create_flow_loss(n_features=2, context_dim=4)
