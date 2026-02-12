import pytest
import torch
import torch.nn as nn

from torchregress.utils import AMP, GradientAccumulation, compile_model


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
