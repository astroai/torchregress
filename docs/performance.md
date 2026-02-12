# Performance Playbook

This guide details best practices for scaling regression models in `torchregress` using modern PyTorch features.

## 1. Automatic Mixed Precision (AMP)

Using `float16` or `bfloat16` can significantly reduce memory usage and speed up training on compatible hardware.

`torchregress` provides a simple wrapper:

```python
from torchregress.utils import AMP
from torch.cuda.amp import GradScaler

# 1. Initialize
amp = AMP(device_type="cuda", dtype=torch.float16)
scaler = GradScaler()

# 2. Training Loop
for x, y in dataloader:
    optimizer.zero_grad()
    
    # 3. Autocast context
    with amp():
        loss = model(x, y)
    
    # 4. Scale and step
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

> **Note**: For A100/H100 GPUs, prefer `torch.bfloat16` as it doesn't strictly require a GradScaler, though using one is still safe.

## 2. Gradient Accumulation

When your batch size is limited by GPU memory, use gradient accumulation to simulate a larger effective batch size.

```python
from torchregress.utils import GradientAccumulation

# Target effective batch size of 256, actual GPU batch size 64
accumulator = GradientAccumulation(batch_size=64, effective_batch_size=256)

for i, (x, y) in enumerate(dataloader):
    # Context manager handles synchronization logic and yields loss scale
    with accumulator(i) as scale:
        loss = model(x, y)
        (loss * scale).backward()
        
    # Only step optimizer on sync steps
    if accumulator.sync_step:
        optimizer.step()
        optimizer.zero_grad()
```

## 3. Torch Compile (PyTorch 2.0+)

`torch.compile` can provide significant speedups by fusing operations. `torchregress` provides a safe wrapper that falls back to the original model if compilation fails.

```python
from torchregress.utils import compile_model

model = MyModel()
model = compile_model(model, mode="default") # or "reduce-overhead"
```

### Supported Modes:
- `default`: Balances compile time and runtime performance.
- `reduce-overhead`: Optimizes for small batches (high memory overhead).
- `max-autotune`: Maximum performance (very long compile time).

## 4. Generic Tips

*   **Dataloading**: Always use `num_workers > 0` and `pin_memory=True` in your DataLoaders.
*   **Eval Mode**: Remember `model.eval()` and `torch.no_grad()` during inference to save memory.
*   **TF32**: On Ampere+ GPUs, enable TF32 for a free speedup:
    ```python
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ```
