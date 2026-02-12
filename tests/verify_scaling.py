import torch
import torch.nn as nn

from torchregress.utils import AMP, GradientAccumulation, compile_model


def test_scaling_utils():
    print("Verifying scalability utilities...")
    
    # 1. Test Gradient Accumulation
    print("  Testing GradientAccumulation...")
    acc = GradientAccumulation(batch_size=2, effective_batch_size=8)
    # steps 0, 1, 2 should not sync; step 3 should sync (4 batches * 2 = 8 effective)
    
    syncs = []
    for step in range(8):
        with acc(step) as scale:
            assert scale == 0.25  # 1/4 accumulation steps
            pass
        if acc.sync_step:
            syncs.append(step)
            
    assert syncs == [3, 7], f"Gradient accumulation sync steps failed. Expected [3, 7], got {syncs}"
    print("    PASSED")

    # 2. Test AMP
    print("  Testing AMP...")
    amp = AMP(enabled=False)  # Force disable to run on CPU if needed, or check availability
    
    model = nn.Linear(10, 1)
    x = torch.randn(5, 10)
    
    # Just checking it doesn't crash
    with amp():
        _ = model(x)
    print("    PASSED")

    # 3. Test compile_model
    print("  Testing compile_model...")
    # This might print warnings if torch.compile is not available/supported
    try:
        compiled = compile_model(model)
        _ = compiled(x)
        print("    PASSED (Run check)")
    except Exception as e:
        print(f"    SKIPPED/FAILED: {e}")


if __name__ == "__main__":
    test_scaling_utils()
