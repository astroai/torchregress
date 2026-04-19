import pytest
import torch
import torch.nn as nn

from torchregress.utils.scaling import (
    AMP,
    GradientAccumulation,
    StandardScaler,
    _is_autocast_supported,
    _is_runtime_device_available,
    compile_model,
)


def test_scaling_utils():
    # 1. Test Gradient Accumulation
    acc = GradientAccumulation(batch_size=2, effective_batch_size=8)
    # steps 0, 1, 2 should not sync; step 3 should sync (4 batches * 2 = 8 effective)

    syncs = []
    for step in range(8):
        with acc(step) as scale:
            assert scale == 0.25  # 1/4 accumulation steps
        if acc.sync_step:
            syncs.append(step)

    assert syncs == [3, 7], f"Gradient accumulation sync steps failed. Expected [3, 7], got {syncs}"

    # 2. Test AMP
    amp = AMP(enabled=False)  # Force disable to run on CPU if needed, or check availability

    model = nn.Linear(10, 1)
    x = torch.randn(5, 10)

    # Just checking it doesn't crash
    with amp():
        _ = model(x)

    # 3. Test compile_model
    # This might print warnings if torch.compile is not available/supported
    compiled = compile_model(model)
    _ = compiled(x)


def test_gradient_accumulation_round_up():
    acc = GradientAccumulation(batch_size=64, effective_batch_size=250)
    assert acc.accumulation_steps == 4


def test_gradient_accumulation_input_validation():
    with pytest.raises(ValueError):
        GradientAccumulation(batch_size=0, effective_batch_size=128)
    with pytest.raises(ValueError):
        GradientAccumulation(batch_size=32, effective_batch_size=0)


def test_amp_device_enablement():
    cuda_amp = AMP(device_type="cuda", enabled=True)
    assert cuda_amp.enabled == torch.cuda.is_available()

    cpu_amp = AMP(device_type="cpu", enabled=True)
    assert cpu_amp.enabled is True
    assert cpu_amp.dtype == torch.bfloat16


def test_amp_mps_enablement_matches_runtime():
    amp = AMP(device_type="mps", enabled=True)
    expected = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    assert amp.enabled == expected


def test_compile_model_returns_original_on_compile_failure(monkeypatch):
    model = nn.Linear(3, 1)

    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")

    def _boom(*args, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(torch, "compile", _boom)
    compiled = compile_model(model)
    assert compiled is model


def test_amp_explicit_dtype():
    amp = AMP(device_type="cpu", enabled=True, dtype=torch.float32)
    assert amp.dtype == torch.float32


def test_amp_cpu_float16_fallback(caplog):
    amp = AMP(device_type="cpu", enabled=True, dtype=torch.float16)
    assert amp.dtype == torch.bfloat16
    assert "AMP float16 on CPU is not recommended" in caplog.text


def test_amp_autocast_not_supported(monkeypatch, caplog):
    monkeypatch.setattr("torchregress.utils.scaling._is_runtime_device_available", lambda x: True)
    monkeypatch.setattr("torchregress.utils.scaling._is_autocast_supported", lambda x: False)
    amp = AMP(device_type="cuda", enabled=True)
    assert amp.enabled is False
    assert "autocast is not supported" in caplog.text


def test_compile_model_no_compile_attr(monkeypatch, caplog):
    model = nn.Linear(3, 1)
    # temporarily hide torch.compile
    monkeypatch.delattr(torch, "compile", raising=False)
    compiled = compile_model(model)
    assert compiled is model
    assert "torch.compile not found" in caplog.text


def test_compile_model_windows_warning(monkeypatch, caplog):
    model = nn.Linear(3, 1)
    monkeypatch.setattr("os.name", "nt")

    # Needs to not fail if compile is available but we still want to see the warning
    # If compile isn't available we skip the warning and return early, so we need to mock it if it's not there
    if not hasattr(torch, "compile"):
        monkeypatch.setattr(torch, "compile", lambda m, **kwargs: m, raising=False)

    compile_model(model, backend="inductor")
    assert "torch.compile with inductor backend is unstable on Windows" in caplog.text


def test_is_runtime_device_available_xpu(monkeypatch):
    class MockXPU:
        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(torch, "xpu", MockXPU(), raising=False)
    assert _is_runtime_device_available("xpu") is True


def test_is_autocast_supported_fallback(monkeypatch):
    import torchregress.utils.scaling as scaling

    # Simple way to test this without breaking builtin hasattr
    # Mock hasattr just for this function using monkeypatch on the module

    original_hasattr = getattr(scaling, "hasattr", hasattr)

    def mock_hasattr(obj, name):
        if name == "is_autocast_available" and obj is getattr(torch.amp, "autocast_mode", None):
            return False
        return original_hasattr(obj, name)

    # Since hasattr is a builtin, we need to mock it within the scaling module
    # We do this by adding it to the module globals

    monkeypatch.setattr("torchregress.utils.scaling.hasattr", mock_hasattr, raising=False)

    assert _is_autocast_supported("cuda") is True
    assert _is_autocast_supported("cpu") is True
    assert _is_autocast_supported("mps") is False


def test_standard_scaler():
    # Test basic fit and transform
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = StandardScaler()

    # Check shape
    X_scaled = scaler.fit_transform(X)
    assert X_scaled.shape == X.shape

    # Check mean is approx 0
    assert torch.allclose(X_scaled.mean(dim=0), torch.zeros(2), atol=1e-5)

    # Check var is approx 1
    assert torch.allclose(X_scaled.var(dim=0, unbiased=False), torch.ones(2), atol=1e-5)


def test_standard_scaler_no_mean():
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = StandardScaler(with_mean=False)

    _ = scaler.fit_transform(X)
    assert scaler.mean_ is None

    # Variance should still be 1 (after dividing by std)
    # But mean won't be 0


def test_standard_scaler_no_std():
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = StandardScaler(with_std=False)

    X_scaled = scaler.fit_transform(X)
    assert scaler.scale_ is None
    assert scaler.var_ is None

    # Mean should be 0
    assert torch.allclose(X_scaled.mean(dim=0), torch.zeros(2), atol=1e-5)


def test_standard_scaler_zero_variance():
    # Test with a column of constant values
    X = torch.tensor([[1.0, 5.0], [1.0, 10.0], [1.0, 15.0]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # The first column should be all zeros after mean subtraction,
    # and division by 1 (due to the zero-variance handling)
    assert torch.allclose(X_scaled[:, 0], torch.zeros(3))
