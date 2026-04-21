"""
Label handling utilities.

This module provides functions for encoding, decoding, and manipulating
label data with PyTorch tensors. While similar functionality exists in libraries
like scikit-learn, these implementations are tensor-native and optimized for
integration with PyTorch regression models.
"""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F


# Basic encoding/decoding functions
def encode_onehot(
    labels: Union[torch.Tensor, np.ndarray], num_classes: Optional[int] = None
) -> torch.Tensor:
    """
    Convert class indices to one-hot encodings.

    While similar to torch.nn.functional.one_hot, this implementation
    handles arbitrary input shapes and automatically determines the
    number of classes when not provided.

    Args:
        labels: Class indices of shape [...] with integer values in [0, num_classes-1]
        num_classes: Number of classes. If None, inferred from labels.

    Returns:
        One-hot encoded tensor of shape [..., num_classes]
    """
    labels = torch.as_tensor(labels)
    if num_classes is None:
        num_classes = int(torch.max(labels).item()) + 1

    return F.one_hot(labels.long(), num_classes=num_classes).float()


def decode_onehot(onehot: Union[torch.Tensor, np.ndarray], dim: int = -1) -> torch.Tensor:
    """
    Convert one-hot encodings to class indices.

    Args:
        onehot: One-hot encoded tensor
        dim: Dimension containing the one-hot encoding

    Returns:
        Class indices tensor
    """
    return torch.argmax(torch.as_tensor(onehot), dim=dim)


def label_smoothing(onehot: Union[torch.Tensor, np.ndarray], alpha: float = 0.1) -> torch.Tensor:
    """
    Apply label smoothing to one-hot encoded labels.

    While torch.nn.CrossEntropyLoss supports label_smoothing, this standalone
    function allows applying smoothing to any one-hot tensor for flexibility
    in custom loss functions.

    Args:
        onehot: One-hot encoded tensor
        alpha: Smoothing factor in [0, 1]

    Returns:
        Smoothed labels tensor
    """
    onehot = torch.as_tensor(onehot)
    num_classes = onehot.shape[-1]
    return (1.0 - alpha) * onehot + alpha / num_classes


def soft_to_hard_labels(
    soft_labels: Union[torch.Tensor, np.ndarray], dim: int = -1
) -> torch.Tensor:
    """
    Convert soft labels (probabilities) to hard labels (one-hot).

    Args:
        soft_labels: Soft labels tensor with probability distributions
        dim: Dimension containing the class probabilities

    Returns:
        One-hot encoded tensor
    """
    soft_labels = torch.as_tensor(soft_labels)
    indices = torch.argmax(soft_labels, dim=dim)
    return F.one_hot(indices, num_classes=soft_labels.shape[dim]).float()


def combine_binary_average(labels: Union[torch.Tensor, np.ndarray], dim: int = 0) -> torch.Tensor:
    """
    Simple averaging of binary labels from multiple annotators.

    Args:
        labels: Binary labels tensor [annotators, samples] or [samples, annotators]
        dim: Dimension along which to average (annotator dimension)

    Returns:
        Average labels [samples]
    """
    return torch.mean(torch.as_tensor(labels).float(), dim=dim)


def combine_binary_weighted_average(
    labels: Union[torch.Tensor, np.ndarray], weights: Union[torch.Tensor, np.ndarray], dim: int = 0
) -> torch.Tensor:
    """
    Weighted averaging of binary labels from multiple annotators.

    Args:
        labels: Binary labels tensor [annotators, samples] or [samples, annotators]
        weights: Weights for each annotator [annotators]
        dim: Dimension along which to average (annotator dimension)

    Returns:
        Weighted average labels [samples]
    """
    labels = torch.as_tensor(labels)
    weights = torch.as_tensor(weights)
    norm_weights = weights / torch.sum(weights)

    if dim == 0:
        weights_expanded = norm_weights.unsqueeze(1)
    else:
        weights_expanded = norm_weights.unsqueeze(0)

    return torch.sum(labels.float() * weights_expanded, dim=dim)
