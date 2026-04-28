import pytest
import torch

from torchregress.utils.validation import (
    check_tensor,
    validate_batch_consistency,
    validate_integer,
    validate_positive,
    validate_quantile,
    validate_range,
    validate_reduction,
    validate_same_device,
    validate_shape,
    validate_weights,
)


def test_validate_reduction():
    assert validate_reduction("mean") == "mean"
    assert validate_reduction("median", ["mean", "median", "sum"]) == "median"
    with pytest.raises(ValueError, match="reduction must be one of"):
        validate_reduction("unknown")


def test_validate_shape():
    x = torch.randn(3, 4)
    assert torch.equal(validate_shape(x, (3, 4), "x"), x)
    assert torch.equal(validate_shape(x, (3, None), "x"), x)

    with pytest.raises(ValueError, match="x has 2 dimensions, expected 3"):
        validate_shape(x, (3, 4, 5), "x")

    with pytest.raises(ValueError, match=r"x has shape torch.Size\(\[3, 4\]\), expected \(2, 4\)"):
        validate_shape(x, (2, 4), "x")

    y = torch.randn(3, 1)
    assert torch.equal(validate_shape(y, (3, 4), "y", allow_broadcast=True), y)
    with pytest.raises(ValueError, match=r"y has shape torch.Size\(\[3, 1\]\), expected \(3, 4\)"):
        validate_shape(y, (3, 4), "y", allow_broadcast=False)


def test_validate_positive():
    assert validate_positive(5.0, "alpha") == 5.0
    assert validate_positive(0.0, "alpha", allow_zero=True) == 0.0

    with pytest.raises(ValueError, match="alpha must be positive, got -1.0"):
        validate_positive(-1.0, "alpha")

    with pytest.raises(ValueError, match="alpha must be positive, got 0.0"):
        validate_positive(0.0, "alpha", allow_zero=False)

    with pytest.raises(ValueError, match="alpha must be non-negative, got -1.0"):
        validate_positive(-1.0, "alpha", allow_zero=True)

    t = torch.tensor([1.0, 2.0])
    assert torch.equal(validate_positive(t, "weights"), t)

    t_zero = torch.tensor([0.0, 1.0])
    assert torch.equal(validate_positive(t_zero, "weights", allow_zero=True), t_zero)

    with pytest.raises(
        ValueError, match=r"weights must be positive, got tensor with minimum value 0.0"
    ):
        validate_positive(t_zero, "weights", allow_zero=False)

    t_neg = torch.tensor([-1.0, 1.0])
    with pytest.raises(
        ValueError, match=r"weights must be non-negative, got tensor with minimum value -1.0"
    ):
        validate_positive(t_neg, "weights", allow_zero=True)


def test_validate_range():
    assert validate_range(0.5, 0.0, 1.0, "probability") == 0.5
    with pytest.raises(ValueError, match="probability must be between 0.0 and 1.0, got 1.5"):
        validate_range(1.5, 0.0, 1.0, "probability")

    t = torch.tensor([0.1, 0.9])
    assert torch.equal(validate_range(t, 0.0, 1.0, "probabilities"), t)

    t_out = torch.tensor([-0.1, 0.5])
    with pytest.raises(
        ValueError,
        match=r"probabilities must be between 0.0 and 1.0, got tensor with values outside range \[-0.1.*, 0.5\]",
    ):
        validate_range(t_out, 0.0, 1.0, "probabilities")


def test_validate_integer():
    t_int = torch.tensor([1, 2, 3], dtype=torch.int64)
    assert torch.equal(validate_integer(t_int), t_int)

    t_float_int = torch.tensor([1.0, 2.0, 3.0])
    res = validate_integer(t_float_int)
    assert torch.equal(res, t_int)
    assert res.dtype == torch.int64

    t_float = torch.tensor([1.5, 2.0, 3.0])
    with pytest.raises(
        ValueError,
        match="tensor must contain only integer values, got tensor with non-integer values",
    ):
        validate_integer(t_float)


def test_validate_quantile():
    assert torch.equal(validate_quantile(0.5), torch.tensor(0.5))

    q_tensor = torch.tensor([0.1, 0.5, 0.9])
    assert torch.equal(validate_quantile(q_tensor), q_tensor)

    with pytest.raises(ValueError, match=r"Quantile\(s\) must be in range \[0, 1\], got -0.1.*"):
        validate_quantile(torch.tensor([-0.1, 0.5]))

    with pytest.raises(
        ValueError, match=r"Quantile\(s\) must be in range \[0, 1\], got 0.5 to 1.5"
    ):
        validate_quantile(torch.tensor([0.5, 1.5]))


def test_validate_batch_consistency():
    a = torch.randn(3, 4)
    b = torch.randn(3, 5)
    # Should not raise
    validate_batch_consistency([a, b])

    # Empty list should return None
    assert validate_batch_consistency([]) is None

    c = torch.randn(5, 4)
    with pytest.raises(
        ValueError,
        match="Batch size mismatch: tensor_0 has batch size 3, but tensor_1 has batch size 5",
    ):
        validate_batch_consistency([a, c])

    with pytest.raises(
        ValueError, match="Batch size mismatch: A has batch size 3, but C has batch size 5"
    ):
        validate_batch_consistency([a, c], names=["A", "C"])


def test_validate_same_device():
    a = torch.randn(3, 4)
    b = torch.randn(2, 5)
    assert validate_same_device([a, b]) == torch.device("cpu")

    with pytest.raises(ValueError, match="No tensors provided"):
        validate_same_device([])

    if torch.cuda.is_available():
        c = torch.randn(3, 4, device="cuda:0")
        with pytest.raises(
            ValueError, match="Device mismatch: tensor_0 is on cpu, but tensor_1 is on cuda:0"
        ):
            validate_same_device([a, c])
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            c = torch.randn(3, 4, device="mps")
            with pytest.raises(
                ValueError, match="Device mismatch: tensor_0 is on cpu, but tensor_1 is on mps:0"
            ):
                validate_same_device([a, c])
        except Exception:
            pass


def test_validate_weights():
    weights = torch.ones(5)
    assert torch.equal(validate_weights(weights, 5), weights)

    assert validate_weights(None, 5) is None

    with pytest.raises(ValueError, match="weights cannot be None"):
        validate_weights(None, 5, allow_none=False)

    bad_dim = torch.randn(5, 1, 1)
    with pytest.raises(ValueError, match="weights must have 1 or 2 dimensions, got 3"):
        validate_weights(bad_dim, 5)

    bad_batch = torch.ones(4)
    with pytest.raises(
        ValueError, match="weights must have same batch size as inputs, got 4, expected 5"
    ):
        validate_weights(bad_batch, 5)

    bad_weights = torch.tensor([-1.0, 1.0, 1.0, 1.0, 1.0])
    with pytest.raises(
        ValueError, match=r"weights must be non-negative, got tensor with minimum value -1.0"
    ):
        validate_weights(bad_weights, 5)


def test_check_tensor():
    x = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(check_tensor(x), x)

    with pytest.raises(TypeError, match="tensor must be a torch.Tensor, got <class 'list'>"):
        check_tensor([1.0, 2.0, 3.0])  # type: ignore[arg-type]

    large_tensor = torch.zeros(10)
    with pytest.raises(
        ValueError,
        match="tensor contains 10 elements, which exceeds the maximum allowed limit of 5.",
    ):
        check_tensor(large_tensor, max_elements=5)

    nan_tensor = torch.tensor([1.0, float("nan"), 3.0])
    with pytest.raises(ValueError, match="tensor contains NaN values"):
        check_tensor(nan_tensor)

    inf_tensor = torch.tensor([1.0, float("inf"), 3.0])
    with pytest.raises(ValueError, match="tensor contains infinite values"):
        check_tensor(inf_tensor)


def test_validate_same_device_mock():
    # To cover the device mismatch branch when cuda/mps is not available
    a = torch.randn(3, 4)

    class MockTensor:
        @property
        def device(self):
            return "mock_device:0"

    b = MockTensor()

    with pytest.raises(
        ValueError, match="Device mismatch: tensor_0 is on cpu, but tensor_1 is on mock_device:0"
    ):
        validate_same_device([a, b])  # type: ignore[list-item]
