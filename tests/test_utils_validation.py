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
    assert validate_reduction("sum") == "sum"
    assert validate_reduction("custom", ["mean", "custom"]) == "custom"

    with pytest.raises(ValueError, match="reduction must be one of"):
        validate_reduction("invalid")
    with pytest.raises(ValueError, match="reduction must be one of"):
        validate_reduction("mean", ["sum", "min"])


def test_validate_shape():
    tensor = torch.zeros(2, 3)
    assert validate_shape(tensor, (2, 3), "test_tensor") is tensor
    assert validate_shape(tensor, (2, 3), "test_tensor", allow_broadcast=False) is tensor

    # The shape function expects the same number of dimensions
    assert validate_shape(tensor, (2, None), "test_tensor") is tensor

    # Broadcast shapes - allow_broadcast allows size 1 in dimensions that expected size > 1
    tensor_broadcast = torch.zeros(2, 1)
    assert (
        validate_shape(tensor_broadcast, (2, 3), "test_tensor", allow_broadcast=True)
        is tensor_broadcast
    )

    with pytest.raises(ValueError, match="test_tensor has 1 dimensions, expected 2"):
        validate_shape(torch.zeros(3), (2, 3), "test_tensor")

    with pytest.raises(ValueError, match="test_tensor has shape"):
        validate_shape(tensor, (3, 2), "test_tensor")

    with pytest.raises(ValueError, match="test_tensor has shape"):
        validate_shape(tensor_broadcast, (2, 3), "test_tensor", allow_broadcast=False)


def test_validate_positive():
    assert validate_positive(5.0, "val") == 5.0
    assert validate_positive(0.0, "val", allow_zero=True) == 0.0

    t = torch.tensor([1.0, 2.0])
    assert validate_positive(t, "val") is t
    t_zero = torch.tensor([0.0, 2.0])
    assert validate_positive(t_zero, "val", allow_zero=True) is t_zero

    with pytest.raises(ValueError, match="val must be positive"):
        validate_positive(0.0, "val")
    with pytest.raises(ValueError, match="val must be non-negative"):
        validate_positive(-1.0, "val", allow_zero=True)

    with pytest.raises(ValueError, match="val must be positive"):
        validate_positive(t_zero, "val")
    with pytest.raises(ValueError, match="val must be non-negative"):
        validate_positive(torch.tensor([-1.0, 1.0]), "val", allow_zero=True)


def test_validate_range():
    assert validate_range(0.5, 0.0, 1.0, "val") == 0.5
    assert validate_range(0.0, 0.0, 1.0, "val") == 0.0
    assert validate_range(1.0, 0.0, 1.0, "val") == 1.0

    t = torch.tensor([0.1, 0.9])
    assert validate_range(t, 0.0, 1.0, "val") is t

    with pytest.raises(ValueError, match="val must be between"):
        validate_range(-0.1, 0.0, 1.0, "val")
    with pytest.raises(ValueError, match="val must be between"):
        validate_range(1.1, 0.0, 1.0, "val")

    with pytest.raises(ValueError, match="val must be between"):
        validate_range(torch.tensor([-0.1, 0.5]), 0.0, 1.0, "val")
    with pytest.raises(ValueError, match="val must be between"):
        validate_range(torch.tensor([0.5, 1.1]), 0.0, 1.0, "val")


def test_validate_integer():
    t_int = torch.tensor([1, 2, 3])
    assert validate_integer(t_int) is t_int

    t_float_int = torch.tensor([1.0, 2.0, 3.0])
    res = validate_integer(t_float_int)
    assert res.dtype == torch.int64
    assert torch.all(res == torch.tensor([1, 2, 3]))

    t_float = torch.tensor([1.5, 2.0])
    with pytest.raises(ValueError, match="tensor must contain only integer values"):
        validate_integer(t_float)


def test_validate_quantile():
    assert torch.allclose(validate_quantile(0.5), torch.tensor(0.5))
    assert torch.allclose(validate_quantile([0.1, 0.9]), torch.tensor([0.1, 0.9]))

    with pytest.raises(ValueError, match="Quantile\\(s\\) must be in range"):
        validate_quantile(-0.1)
    with pytest.raises(ValueError, match="Quantile\\(s\\) must be in range"):
        validate_quantile(1.1)
    with pytest.raises(ValueError, match="Quantile\\(s\\) must be in range"):
        validate_quantile([0.5, 1.5])


def test_validate_batch_consistency():
    t1 = torch.zeros(3, 4)
    t2 = torch.zeros(3, 5)
    t3 = torch.zeros(2, 4)

    validate_batch_consistency([])  # Should not raise
    validate_batch_consistency([t1])  # Should not raise
    validate_batch_consistency([t1, t2])  # Should not raise

    with pytest.raises(
        ValueError,
        match="Batch size mismatch: tensor_0 has batch size 3, but tensor_1 has batch size 2",
    ):
        validate_batch_consistency([t1, t3])
    with pytest.raises(
        ValueError, match="Batch size mismatch: t1 has batch size 3, but t3 has batch size 2"
    ):
        validate_batch_consistency([t1, t3], ["t1", "t3"])


def test_validate_same_device():
    t1 = torch.zeros(2)
    t2 = torch.zeros(3)

    with pytest.raises(ValueError, match="No tensors provided"):
        validate_same_device([])

    assert validate_same_device([t1]) == t1.device
    assert validate_same_device([t1, t2]) == t1.device

    if torch.cuda.is_available():
        t_cuda = torch.zeros(2, device="cuda:0")
        with pytest.raises(ValueError, match="Device mismatch"):
            validate_same_device([t1, t_cuda])

        with pytest.raises(
            ValueError, match="Device mismatch: cpu_t is on .*cpu.*, but cuda_t is on .*cuda.*"
        ):
            validate_same_device([t1, t_cuda], ["cpu_t", "cuda_t"])


def test_validate_weights():
    w = torch.ones(5)
    assert validate_weights(w, 5) is w
    assert validate_weights(w, 5, allow_none=False) is w

    assert validate_weights(None, 5) is None
    with pytest.raises(ValueError, match="weights cannot be None"):
        validate_weights(None, 5, allow_none=False)

    w_2d = torch.ones(5, 1)
    assert validate_weights(w_2d, 5) is w_2d

    w_3d = torch.ones(5, 1, 1)
    with pytest.raises(ValueError, match="weights must have 1 or 2 dimensions"):
        validate_weights(w_3d, 5)

    with pytest.raises(ValueError, match="weights must have same batch size"):
        validate_weights(w, 4)

    w_neg = torch.tensor([-1.0, 1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="weights must be non-negative"):
        validate_weights(w_neg, 5)


def test_check_tensor():
    t = torch.tensor([1.0, 2.0])
    assert check_tensor(t) is t

    with pytest.raises(TypeError, match="tensor must be a torch.Tensor"):
        check_tensor([1.0, 2.0])  # type: ignore[arg-type]

    t_large = torch.zeros(10)
    with pytest.raises(ValueError, match="which exceeds the maximum allowed limit"):
        check_tensor(t_large, max_elements=5)

    t_nan = torch.tensor([1.0, float("nan")])
    with pytest.raises(ValueError, match="tensor contains NaN values"):
        check_tensor(t_nan)

    t_inf = torch.tensor([1.0, float("inf")])
    with pytest.raises(ValueError, match="tensor contains infinite values"):
        check_tensor(t_inf)
