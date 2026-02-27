# Performance Playbook

This guide details best practices for scaling regression models in `torchregress` using modern PyTorch features.

## 1. Automatic Mixed Precision (AMP)

Using `float16` or `bfloat16` can significantly reduce memory usage and speed up training on compatible hardware.

`torchregress` provides a simple wrapper:

```python
from torchregress.utils import AMP
from torch.amp import GradScaler

# 1. Initialize
amp = AMP(device_type="cuda", dtype=torch.float16)
scaler = GradScaler("cuda")

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

`AMP` automatically disables itself when the requested runtime device is unavailable (for example, `device_type="cuda"` on a CPU-only machine).

> **Note**: For A100/H100 GPUs, prefer `torch.bfloat16` as it doesn't strictly require a GradScaler, though using one is still safe. For CPU autocast, prefer `torch.bfloat16`.

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

## 5. Benchmark Regression Checks (CI)

`torchregress` includes a lightweight benchmark harness for performance smoke checks and
small scale sweeps:

- `tools/benchmark_smoke.py`

### Stable CI Invocation (CPU)

Use warmup and at least 2 iterations to reduce cold-start noise:

```bash
python -m tools.benchmark_smoke \
  --mode smoke \
  --iterations 2 \
  --warmup 1 \
  --device cpu \
  --thresholds reports/benchmark_thresholds/cpu/smoke.json \
  --fail-on-thresholds

python -m tools.benchmark_smoke \
  --mode sweep \
  --iterations 2 \
  --warmup 1 \
  --device cpu \
  --thresholds reports/benchmark_thresholds/cpu/sweep.json \
  --fail-on-thresholds
```

### Threshold Baseline Layout (Per Device)

- CPU smoke thresholds: `reports/benchmark_thresholds/cpu/smoke.json`
- CPU sweep thresholds: `reports/benchmark_thresholds/cpu/sweep.json`
- CUDA sweep thresholds (reserved path for future CI/hardware baselines): `reports/benchmark_thresholds/cuda/sweep.json`

### Benchmark Report Summaries

Convert JSON benchmark reports into Markdown tables for audit/docs updates:

```bash
python -m tools.benchmark_report_summary reports/benchmark_smoke_latest.json
python -m tools.benchmark_report_summary reports/benchmark_sweep_latest.json --group-by-name
```

CI also renders and uploads Markdown summaries as artifacts (`benchmark-cpu-summaries`):

- `reports/benchmark_smoke_latest.md`
- `reports/benchmark_sweep_latest.md`

Committed baseline snapshot (docs-visible):

- [Benchmark CPU Baselines (2026-02-26)](../audits/benchmark_cpu_baselines_2026-02-26.md)

### Refreshing Threshold Baselines

Re-generate a sweep report and derive thresholds (example for CPU):

```bash
python -m tools.benchmark_smoke \
  --mode sweep \
  --iterations 2 \
  --warmup 0 \
  --device cpu \
  --output reports/benchmark_sweep_cpu_YYYY-MM-DD.json \
  --write-thresholds reports/benchmark_thresholds/cpu/sweep.json \
  --threshold-multiplier 6.0 \
  --threshold-floor-ms 3.0
```

Re-generate a smoke report and derive smoke thresholds:

```bash
python -m tools.benchmark_smoke \
  --mode smoke \
  --iterations 2 \
  --warmup 1 \
  --device cpu \
  --output reports/benchmark_smoke_YYYY-MM-DD.json \
  --write-thresholds reports/benchmark_thresholds/cpu/smoke.json \
  --threshold-multiplier 6.0 \
  --threshold-floor-ms 3.0
```

### CUDA-Ready Benchmark Plan (Placeholder)

`torchregress` currently enforces CPU benchmark thresholds in CI. For future GPU CI:

- keep separate baselines per GPU class / runner image
- derive thresholds with higher iteration counts than CPU smoke checks
- version and document the runner hardware in the benchmark artifact
- only enable hard failures after baseline stability is confirmed

The reserved path for a future CUDA sweep baseline is `reports/benchmark_thresholds/cuda/sweep.json`.

Notes:

- The threshold multiplier/floor are intentionally conservative for CI stability on hosted runners
  (small sub-millisecond kernels can show occasional jitter spikes).
- For local profiling or true performance comparisons, use a dedicated benchmark workflow with
  more iterations, repeated runs, and controlled hardware.
