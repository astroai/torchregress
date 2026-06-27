"""
Custom neural network layers for ensemble models.

This module provides specialized layer implementations that are used
in ensemble models, such as BatchEnsemble layers.
"""

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchEnsembleLinear(nn.Module):
    """
    BatchEnsemble linear layer implementation.

    This implements the BatchEnsemble technique from:
    "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning"

    Instead of maintaining M copies of a model, BatchEnsemble uses rank-1 perturbations
    to create M virtual models that share parameters.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        ensemble_size: Number of ensemble members
        bias: Whether to use bias
        device: Device to use
        dtype: Data type
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int = 4,
        bias: bool = True,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        # Main weight matrix - shared across ensemble
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        # Fast weight vectors for ensemble (rank-1 perturbation)
        self.r_vectors = nn.Parameter(torch.empty((ensemble_size, in_features), **factory_kwargs))
        self.s_vectors = nn.Parameter(torch.empty((ensemble_size, out_features), **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Initialize r and s vectors with random signs
        nn.init.normal_(self.r_vectors, mean=1.0, std=0.1)
        nn.init.normal_(self.s_vectors, mean=1.0, std=0.1)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 2:
            input = input.unsqueeze(1).expand(-1, self.ensemble_size, -1)
        elif input.dim() != 3:
            raise ValueError(f"Input must be 2D or 3D, got {input.dim()}D")
        if input.shape[1] != self.ensemble_size:
            raise ValueError(
                f"Input ensemble dimension size {input.shape[1]} doesn't match "
                f"expected ensemble size {self.ensemble_size}"
            )
        batch_size = input.shape[0]
        r_input = input * self.r_vectors.unsqueeze(0)
        r_input = r_input.reshape(-1, self.in_features)
        output = F.linear(r_input, self.weight, None)
        output = output.view(batch_size, self.ensemble_size, self.out_features)
        output = output * self.s_vectors.unsqueeze(0)
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        """String representation of the module."""
        return "in_features={}, out_features={}, ensemble_size={}, bias={}".format(
            self.in_features, self.out_features, self.ensemble_size, self.bias is not None
        )
