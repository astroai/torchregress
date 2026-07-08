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


def test_check_health_main(capsys):
    """Test that check_health is called when run as main."""
    import runpy
    runpy.run_module("torchregress.health", run_name="__main__")
    captured = capsys.readouterr()
    assert "[SUCCESS] System appears healthy." in captured.out
