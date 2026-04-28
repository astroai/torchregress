import pytest
import torch
from torchregress.utils.validation import (
    validate_reduction,
    validate_shape,
    validate_positive,
    validate_range,
    validate_integer,
    validate_quantile,
    validate_batch_consistency,
    validate_same_device,
    validate_weights,
    check_tensor
)

def test_validate_reduction():
    # Happy paths
    assert validate_reduction('mean') == 'mean'
    assert validate_reduction('median', ['mean', 'median', 'sum']) == 'median'

    # Error paths
    with pytest.raises(ValueError, match="reduction must be one of .* got unknown"):
        validate_reduction('unknown')

    with pytest.raises(ValueError, match="reduction must be one of .* got sum"):
        validate_reduction('sum', ['mean', 'median'])


def test_validate_shape():
    x = torch.randn(3, 4)
    # Happy paths
    assert validate_shape(x, (3, 4), "x") is x
    assert validate_shape(x, (3, None), "x") is x

    # allow_broadcast behavior
    # When expect shape is (3, 4), and actual shape is (3, 1), it passes if allow_broadcast=True
    y = torch.randn(3, 1)
    assert validate_shape(y, (3, 4), "y", allow_broadcast=True) is y

    # Error paths
    with pytest.raises(ValueError, match="x has 2 dimensions, expected 3"):
        validate_shape(x, (3, 4, 5), "x")

    with pytest.raises(ValueError, match="x has shape .* expected \\(2, 4\\)"):
        validate_shape(x, (2, 4), "x")

    with pytest.raises(ValueError, match="y has shape .* expected \\(3, 4\\)"):
        validate_shape(y, (3, 4), "y", allow_broadcast=False)


def test_validate_positive():
    # Floats
    assert validate_positive(5.0, "alpha") == 5.0
    assert validate_positive(0.0, "alpha", allow_zero=True) == 0.0

    with pytest.raises(ValueError, match="alpha must be positive, got -1.0"):
        validate_positive(-1.0, "alpha")

    with pytest.raises(ValueError, match="alpha must be non-negative, got -1.0"):
        validate_positive(-1.0, "alpha", allow_zero=True)

    with pytest.raises(ValueError, match="alpha must be positive, got 0.0"):
        validate_positive(0.0, "alpha")

    # Tensors
    t = torch.tensor([1.0, 2.0])
    assert validate_positive(t, "weights") is t

    t_zero = torch.tensor([0.0, 2.0])
    assert validate_positive(t_zero, "weights", allow_zero=True) is t_zero

    with pytest.raises(ValueError, match="weights must be positive, got tensor with minimum value 0.0"):
        validate_positive(t_zero, "weights")

    t_neg = torch.tensor([-1.0, 2.0])
    with pytest.raises(ValueError, match="weights must be positive, got tensor with minimum value -1.0"):
        validate_positive(t_neg, "weights")

    with pytest.raises(ValueError, match="weights must be non-negative, got tensor with minimum value -1.0"):
        validate_positive(t_neg, "weights", allow_zero=True)


def test_validate_range():
    # Floats
    assert validate_range(0.5, 0.0, 1.0, "prob") == 0.5

    with pytest.raises(ValueError, match="prob must be between 0.0 and 1.0, got 1.5"):
        validate_range(1.5, 0.0, 1.0, "prob")

    with pytest.raises(ValueError, match="prob must be between 0.0 and 1.0, got -0.1"):
        validate_range(-0.1, 0.0, 1.0, "prob")

    # Tensors
    t = torch.tensor([0.1, 0.9])
    assert validate_range(t, 0.0, 1.0, "probs") is t

    t_high = torch.tensor([0.1, 1.5])
    with pytest.raises(ValueError, match="probs must be between 0.0 and 1.0, got tensor with values outside range \\[0.1.*, 1.5.*\\]"):
        validate_range(t_high, 0.0, 1.0, "probs")

    t_low = torch.tensor([-0.1, 0.9])
    with pytest.raises(ValueError, match=r"probs must be between 0.0 and 1.0, got tensor with values outside range \[-0.1.*, 0.89.*\]"):
        validate_range(t_low, 0.0, 1.0, "probs")


def test_validate_integer():
    # Integer tensor
    t = torch.tensor([1, 2, 3], dtype=torch.int32)
    assert validate_integer(t) is t

    # Float tensor with integer values
    t_float = torch.tensor([1.0, 2.0, 3.0])
    res = validate_integer(t_float)
    assert res.dtype == torch.int64
    assert torch.equal(res, torch.tensor([1, 2, 3]))

    # Float tensor with non-integer values
    t_bad = torch.tensor([1.5, 2.0])
    with pytest.raises(ValueError, match="tensor must contain only integer values"):
        validate_integer(t_bad)


def test_validate_quantile():
    # Float
    res = validate_quantile(0.5)
    assert isinstance(res, torch.Tensor)
    assert res.item() == 0.5

    # Tensor
    t = torch.tensor([0.1, 0.9])
    res = validate_quantile(t)
    assert torch.equal(res, t)

    # Errors
    with pytest.raises(ValueError, match=r"Quantile\(s\) must be in range \[0, 1\], got -0.1.* to 0.5"):
        validate_quantile(torch.tensor([-0.1, 0.5]))

    with pytest.raises(ValueError, match=r"Quantile\(s\) must be in range \[0, 1\], got 0.5 to 1.5"):
        validate_quantile(torch.tensor([0.5, 1.5]))


def test_validate_batch_consistency():
    a = torch.randn(3, 4)
    b = torch.randn(3, 5)

    # Empty list
    validate_batch_consistency([])

    # Single tensor
    validate_batch_consistency([a])

    # Same batch size
    validate_batch_consistency([a, b])

    # Different batch sizes
    c = torch.randn(5, 4)
    with pytest.raises(ValueError, match="Batch size mismatch: tensor_0 has batch size 3, but tensor_1 has batch size 5"):
        validate_batch_consistency([a, c])

    with pytest.raises(ValueError, match="Batch size mismatch: a has batch size 3, but c has batch size 5"):
        validate_batch_consistency([a, c], names=["a", "c"])


def test_validate_same_device():
    a = torch.randn(3, 4)
    b = torch.randn(2, 5)

    # Empty list
    with pytest.raises(ValueError, match="No tensors provided"):
        validate_same_device([])

    # Same device (CPU)
    assert validate_same_device([a, b]) == torch.device('cpu')

    # Different devices (mock)
    if torch.cuda.is_available():
        c = torch.randn(3, 4, device='cuda:0')
        with pytest.raises(ValueError, match="Device mismatch: tensor_0 is on cpu, but tensor_1 is on cuda:0"):
            validate_same_device([a, c])

        with pytest.raises(ValueError, match="Device mismatch: a is on cpu, but c is on cuda:0"):
            validate_same_device([a, c], names=["a", "c"])


def test_validate_weights():
    # None allowed
    assert validate_weights(None, 5) is None

    # None not allowed
    with pytest.raises(ValueError, match="weights cannot be None"):
        validate_weights(None, 5, allow_none=False)

    # Valid weights
    weights = torch.ones(5)
    res = validate_weights(weights, 5)
    assert torch.equal(res, weights)

    # Invalid shape
    with pytest.raises(ValueError, match="weights must have same batch size as inputs, got 4, expected 5"):
        validate_weights(torch.ones(4), 5)

    # Invalid dimensions
    with pytest.raises(ValueError, match="weights must have 1 or 2 dimensions, got 3"):
        validate_weights(torch.ones(5, 1, 1), 5)

    # Invalid values
    with pytest.raises(ValueError, match="weights must be non-negative"):
        validate_weights(torch.tensor([-1.0, 1.0, 1.0, 1.0, 1.0]), 5)


def test_check_tensor():
    # Valid tensor
    x = torch.tensor([1.0, 2.0, 3.0])
    assert check_tensor(x) is x

    # Not a tensor
    with pytest.raises(TypeError, match="tensor must be a torch.Tensor, got <class 'list'>"):
        check_tensor([1.0, 2.0])

    # Max elements exceeded
    with pytest.raises(ValueError, match="tensor contains 3 elements, which exceeds the maximum allowed limit of 2."):
        check_tensor(x, max_elements=2)

    # NaN
    y = torch.tensor([1.0, float('nan'), 3.0])
    with pytest.raises(ValueError, match="tensor contains NaN values"):
        check_tensor(y)

    # Inf
    z = torch.tensor([1.0, float('inf'), 3.0])
    with pytest.raises(ValueError, match="tensor contains infinite values"):
        check_tensor(z)
