"""
Scalability utilities for training large regression models.

This module provides helpers for:
- Gradient accumulation
- Automatic Mixed Precision (AMP)
- torch.compile integration
"""

import contextlib
import logging
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
        self.accumulation_steps = effective_batch_size // batch_size
        if self.accumulation_steps < 1:
            self.accumulation_steps = 1
        self.step = 0
        self.sync_step = False
        
    @contextlib.contextmanager
    def __call__(self, step: int) -> Generator[None, None, None]:
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
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        with amp():
            loss = model(x)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    """
    
    def __init__(
        self, 
        device_type: str = "cuda", 
        enabled: bool = True, 
        dtype: Optional[torch.dtype] = None
    ):
        self.device_type = device_type
        self.enabled = enabled and torch.cuda.is_available()
        self.dtype = dtype or torch.float16
        
    def __call__(self) -> Any:
        # Use torch.autocast for modern PyTorch (>=1.10)
        return torch.autocast(device_type=self.device_type, enabled=self.enabled, dtype=self.dtype)


def compile_model(
    model: nn.Module, 
    mode: str = "default", 
    fullgraph: bool = False,
    dynamic: bool = False,
    backend: str = "inductor"
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
            model,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend
        )
        return compiled_model
    except Exception as e:
        logger.warning(f"Model compilation failed: {e}. Returning original model.")
        return model
