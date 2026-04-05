import builtins
from unittest import mock

import pytest

from torchregress.health import check_health


def test_check_health_success(capsys):
    """Test that check_health runs successfully without raising errors."""
    check_health()
    captured = capsys.readouterr()

    assert "torchregress version:" in captured.out
    assert "PyTorch version:" in captured.out
    assert "Python version:" in captured.out
    assert "Imports: OK" in captured.out
    assert "Tensor ops: OK" in captured.out
    assert "Training step: OK" in captured.out
    assert "Metric compute: OK" in captured.out
    assert "[SUCCESS] System appears healthy." in captured.out


def test_check_health_import_error(capsys, monkeypatch):
    """Test check_health handles import errors."""
    real_import = builtins.__import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torchregress" and fromlist and "algorithms" in fromlist:
            raise ImportError("Mocked import error")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(SystemExit) as excinfo:
        check_health()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Imports: FAILED (Mocked import error)" in captured.out


@mock.patch("torch.randn")
def test_check_health_compute_error(mock_randn, capsys):
    """Test check_health handles basic compute errors."""
    mock_randn.side_effect = Exception("Compute failed")

    with pytest.raises(SystemExit) as excinfo:
        check_health()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Tensor ops: FAILED (Compute failed)" in captured.out


@mock.patch("torchregress.health.WeightedLossWrapper")
def test_check_health_training_error(mock_wrapper, capsys):
    """Test check_health handles training loop errors."""
    mock_wrapper.side_effect = Exception("Training failed")

    with pytest.raises(SystemExit) as excinfo:
        check_health()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Training step: FAILED (Training failed)" in captured.out


@mock.patch("torchregress.health.MedianAbsoluteError", create=True)
def test_check_health_metric_error(mock_metric, capsys, monkeypatch):
    """Test check_health handles metrics compute errors."""

    # We need to mock the import of MedianAbsoluteError inside the function
    real_import = builtins.__import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torchregress.metrics" and fromlist and "MedianAbsoluteError" in fromlist:
            raise Exception("Metric failed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(SystemExit) as excinfo:
        check_health()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Metric compute: FAILED (Metric failed)" in captured.out
