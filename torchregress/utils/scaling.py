"""
Scalability utilities for training large regression models.

This module provides helpers for:
- Gradient accumulation
- Automatic Mixed Precision (AMP)
- torch.compile integration
"""

import contextlib
import logging
import math
import os
from typing import Any, Generator, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GradientAccumulation:
    """
    Context manager for gradient accumulation.

    Usage:
        accumulator = GradientAccumulation(batch_size=64, effective_batch_size=256)

        for i, (x, y) in enumerate(dataloader):
            with accumulator(i):
                loss = model(x, y)
                loss.backward()

            if accumulator.sync_step:
                optimizer.step()
                optimizer.zero_grad()
    """

    def __init__(self, batch_size: int, effective_batch_size: int):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if effective_batch_size <= 0:
            raise ValueError(f"effective_batch_size must be positive, got {effective_batch_size}")

        # Round up so effective batch is at least the requested size.
        self.accumulation_steps = max(1, math.ceil(effective_batch_size / batch_size))
        if effective_batch_size % batch_size != 0:
            logger.info(
                "effective_batch_size=%s is not divisible by batch_size=%s; "
                "using accumulation_steps=%s (effective=%s).",
                effective_batch_size,
                batch_size,
                self.accumulation_steps,
                self.accumulation_steps * batch_size,
            )
        self.step = 0
        self.sync_step = False

    @contextlib.contextmanager
    def __call__(self, step: int) -> Generator[float, None, None]:
        """
        Context for a training step.

        Args:
            step: Current global step or batch index.
        """
        self.step = step
        is_sync_step = (step + 1) % self.accumulation_steps == 0
        self.sync_step = is_sync_step

        # Yield scale factor so user can normalize loss
        yield 1.0 / self.accumulation_steps


class AMP:
    """
    Helper for Automatic Mixed Precision.

    Usage:
        amp = AMP(enabled=True)
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        with amp():
            loss = model(x)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    """

    def __init__(
        self, device_type: str = "cuda", enabled: bool = True, dtype: Optional[torch.dtype] = None
    ):
        self.device_type = device_type
        self.enabled = enabled

        if enabled and not _is_runtime_device_available(device_type):
            logger.warning(
                "AMP requested for device_type='%s' but runtime device is not "
                "available. Disabling AMP.",
                device_type,
            )
            self.enabled = False
        elif enabled and not _is_autocast_supported(device_type):
            logger.warning(
                "AMP requested for device_type='%s' but autocast is not supported. Disabling AMP.",
                device_type,
            )
            self.enabled = False

        if dtype is not None:
            self.dtype = dtype
        elif device_type in {"cuda", "mps", "xpu"}:
            self.dtype = torch.float16
        else:
            self.dtype = torch.bfloat16

        if self.device_type == "cpu" and self.dtype == torch.float16:
            logger.warning("AMP float16 on CPU is not recommended; using bfloat16 instead.")
            self.dtype = torch.bfloat16

    def __call__(self) -> Any:
        # Use torch.amp.autocast for modern PyTorch.
        return torch.amp.autocast(
            device_type=self.device_type, enabled=self.enabled, dtype=self.dtype
        )


def compile_model(
    model: nn.Module,
    mode: str = "default",
    fullgraph: bool = False,
    dynamic: bool = False,
    backend: str = "inductor",
) -> nn.Module:
    """
    Safe wrapper around torch.compile.

    Args:
        model: PyTorch model to compile.
        mode: Compilation mode ("default", "reduce-overhead", "max-autotune").
        fullgraph: Whether to capture full graph (limits python dynamism).
        dynamic: Use dynamic shapes.
        backend: Compiler backend.

    Returns:
        Compiled model or original model if compilation fails/is unavailable.
    """
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile not found (requires PyTorch 2.0+). Returning original model.")
        return model

    # Check for OS-specific issues (e.g. Windows support is essentially non-existent for inductor)
    if os.name == "nt" and backend == "inductor":
        logger.warning(
            "torch.compile with inductor backend is unstable on Windows. Proceeding with caution."
        )

    try:
        logger.info(f"Compiling model with mode={mode}, backend={backend}...")
        compiled_model = torch.compile(
            model, mode=mode, fullgraph=fullgraph, dynamic=dynamic, backend=backend
        )
        return compiled_model
    except Exception as e:
        logger.warning(f"Model compilation failed: {e}. Returning original model.")
        return model


def _is_runtime_device_available(device_type: str) -> bool:
    if device_type == "cuda":
        return torch.cuda.is_available()
    if device_type == "mps":
        return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    if device_type == "xpu":
        return bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    # CPU (and other fallback backends) are treated as always available.
    return True


def _is_autocast_supported(device_type: str) -> bool:
    if hasattr(torch.amp.autocast_mode, "is_autocast_available"):
        return bool(torch.amp.autocast_mode.is_autocast_available(device_type))
    # Conservative fallback for older PyTorch.
    return device_type in {"cuda", "cpu"}
