"""
Utility functions for metrics calculations.
"""

import torch
import numpy as np
from typing import Union, Optional, Dict, Tuple, List, Callable

def convert_to_tensor(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Convert numpy arrays to torch tensors."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x

def apply_reduction(values: torch.Tensor, reduction: str) -> Union[torch.Tensor, float]:
    """Apply reduction to tensor values."""
    if reduction == "none":
        return values
    elif reduction == "mean":
        return torch.mean(values)
    elif reduction == "sum":
        return torch.sum(values)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

def ensure_batch_dim(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has a batch dimension."""
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x

def create_metric_result(
    result: Union[torch.Tensor, float, Dict],
    as_numpy: bool
) -> Union[torch.Tensor, float, np.ndarray, Dict]:
    """Convert result to appropriate type based on input."""
    if isinstance(result, Dict):
        # Convert dict values
        return {k: v.cpu().numpy() if isinstance(v, torch.Tensor) and as_numpy else v 
                for k, v in result.items()}
    elif isinstance(result, torch.Tensor):
        return result.cpu().numpy() if as_numpy else result
    return result
