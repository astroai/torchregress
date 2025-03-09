"""
PyTorch compatibility utilities.

This module provides functions for handling compatibility across
different PyTorch versions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import functools
import warnings
from torch import Tensor
from typing import Optional, Union, Dict, Any, List, Tuple, Callable

def convert_reduction_type(reduction: str) -> str:
    """
    Convert between different reduction naming conventions.
    
    Args:
        reduction: Reduction name ('none', 'mean', 'sum', 'batch_mean', etc.)
        
    Returns:
        PyTorch compatible reduction name
    """
    # Handle common aliases
    reduction = reduction.lower()
    
    if reduction in ['none', 'mean', 'sum']:
        return reduction
    elif reduction == 'batch_mean' or reduction == 'average':
        return 'mean'
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}. "
                       f"Expected one of 'none', 'mean', 'sum', 'batch_mean', 'average'")
    
def convert_to_pytorch_loss(loss_fn: Union[nn.Module, Callable], **kwargs) -> nn.Module:
    """
    Convert a loss function (callable or module) to a PyTorch loss module.
    
    Args:
        loss_fn: Loss function or module
        **kwargs: Additional arguments for the loss function
        
    Returns:
        A PyTorch nn.Module loss function
    """
    if isinstance(loss_fn, nn.Module):
        # Already a module
        return loss_fn
    
    # Create a wrapper module
    class LossFunctionWrapper(nn.Module):
        def __init__(self, loss_fn, **kwargs):
            super().__init__()
            self.loss_fn = loss_fn
            self.kwargs = kwargs
            
        def forward(self, output, target):
            return self.loss_fn(target, output, **self.kwargs)
    
    return LossFunctionWrapper(loss_fn, **kwargs)

def extract_output_size(model: nn.Module, input_size: Union[int, List[int], Tuple[int, ...]]) -> Tuple[int, ...]:
    """
    Determine the output size of a PyTorch model given input size.
    
    Args:
        model: PyTorch model
        input_size: Size of input (excluding batch dimension)
        
    Returns:
        Output size (excluding batch dimension)
    """
    # Convert int to tuple
    if isinstance(input_size, int):
        input_size = (input_size,)
    
    # Create dummy input
    dummy_input = torch.zeros(1, *input_size)  # Add batch dimension
    
    # Get output
    with torch.no_grad():
        output = model(dummy_input)
    
    # Return output size (excluding batch dimension)
    if isinstance(output, torch.Tensor):
        return tuple(output.shape[1:])
    elif isinstance(output, tuple) and all(isinstance(o, torch.Tensor) for o in output):
        return tuple(o.shape[1:] for o in output)
    else:
        raise ValueError(f"Model output is not a tensor or tuple of tensors, got {type(output)}")

def set_all_seeds(seed: int) -> None:
    """
    Set all seeds for reproducibility.
    
    Args:
        seed: Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Additional settings for full reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_cuda_available() -> bool:
    """
    Check if CUDA is available.
    
    Returns:
        Boolean indicating whether CUDA is available
    """
    return torch.cuda.is_available()
    
def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device based on availability and request.
    
    Args:
        device_str: Requested device ('cuda', 'cpu', or None for auto-detection)
        
    Returns:
        PyTorch device
    """
    if device_str is None:
        # Auto-detect
        return torch.device('cuda:0' if is_cuda_available() else 'cpu')
    
    if device_str.startswith('cuda') and not is_cuda_available():
        print("Warning: CUDA requested but not available. Using CPU instead.")
        return torch.device('cpu')
    
    return torch.device(device_str)

def get_torch_version() -> Tuple[int, int, int]:
    """
    Get the current PyTorch version as a tuple of integers.
    
    Returns:
        Tuple of (major, minor, patch) version numbers
    """
    version_str = torch.__version__.split('+')[0]  # Remove git commit hash if present
    version_parts = version_str.split('.')
    
    # Handle different version string formats
    try:
        if len(version_parts) >= 3:
            return (int(version_parts[0]), int(version_parts[1]), int(version_parts[2]))
        elif len(version_parts) == 2:
            return (int(version_parts[0]), int(version_parts[1]), 0)
        else:
            return (int(version_parts[0]), 0, 0)
    except (ValueError, IndexError):
        warnings.warn(f"Couldn't parse PyTorch version: {torch.__version__}. Using (0, 0, 0).")
        return (0, 0, 0)

def requires_torch_version(major: int, minor: int = 0, patch: int = 0) -> Callable:
    """
    Decorator to check if the current PyTorch version meets the minimum requirement.
    
    Args:
        major: Major version requirement
        minor: Minor version requirement
        patch: Patch version requirement
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            version = get_torch_version()
            if version < (major, minor, patch):
                warnings.warn(
                    f"Function {func.__name__} requires PyTorch version >= {major}.{minor}.{patch}, "
                    f"but you have {torch.__version__}. This might cause issues."
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator

def has_torch_function(tensors) -> bool:
    """
    Check if any tensor has a custom __torch_function__ implementation.
    
    This is used to ensure compatibility with custom tensor types.
    
    Args:
        tensors: A tensor or sequence of tensors
        
    Returns:
        True if any tensor has a custom __torch_function__ implementation
    """
    if hasattr(torch, '_has_torch_function'):
        # PyTorch 1.8+
        if not isinstance(tensors, (list, tuple)):
            tensors = [tensors]
        return torch._has_torch_function(tensors)
    else:
        # Fallback for older PyTorch versions
        if not isinstance(tensors, (list, tuple)):
            tensors = [tensors]
        for t in tensors:
            if hasattr(t, '__torch_function__'):
                return True
        return False


def get_reduction_enum(reduction: str) -> int:
    """
    Get the enum value for a reduction mode.
    
    This ensures compatibility across PyTorch versions.
    
    Args:
        reduction: Reduction mode ('none', 'mean', 'sum')
        
    Returns:
        Enum value for the reduction mode
    """
    if reduction == 'none':
        return 0
    elif reduction == 'mean':
        return 1
    elif reduction == 'sum':
        return 2
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def handle_torch_function(func: Callable, types, args=(), kwargs=None):
    """
    Handle __torch_function__ dispatch for custom tensor types.
    
    This ensures compatibility with custom tensor types by dispatching
    to their __torch_function__ implementations.
    
    Args:
        func: The function to dispatch
        types: Types of the arguments
        args: Function arguments
        kwargs: Function keyword arguments
        
    Returns:
        Result of the dispatched function
    """
    if kwargs is None:
        kwargs = {}
    
    # Get the highest priority implementation
    with_impl = [t for t in types if hasattr(t, '__torch_function__')]
    if not with_impl:
        return NotImplemented
    
    # Find the implementation with highest priority
    priorities = {t: i for i, t in enumerate(with_impl)}
    sorted_types = sorted(with_impl, key=lambda t: priorities.get(t, 0), reverse=True)
    
    # Dispatch to the highest priority implementation
    for t in sorted_types:
        result = t.__torch_function__(func, sorted_types, args, kwargs)
        if result is not NotImplemented:
            return result
    
    return NotImplemented
