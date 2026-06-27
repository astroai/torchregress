import pytest
import torch

from torchregress.utils.validation import (
    check_tensor,
    validate_metric_inputs,
    validate_positive,
    validate_quantile,
    validate_range,
    validate_reduction,
    validate_sample_weight,
    validate_weights,
)


def test_validate_reduction():
    assert validate_reduction("mean") == "mean"
    assert validate_reduction("median", ["mean", "median", "sum"]) == "median"
    with pytest.raises(ValueError, match="reduction must be one of"):
        validate_reduction("unknown")


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


def test_validate_weights_flatten():
    weights = torch.ones(5, 1)
    flattened = validate_weights(weights, 5, flatten=True)
    assert flattened.shape == (5,)

    bad_shape = torch.ones(5, 2)
    with pytest.raises(ValueError, match="Sample weights should be 1D"):
        validate_weights(bad_shape, 5, flatten=True)


def test_validate_sample_weight_delegates_to_validate_weights():
    weights = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(validate_sample_weight(weights, 3), weights)


def test_validate_metric_inputs():
    y_pred = torch.randn(4, 2)
    y_true = torch.randn(4, 2)
    validate_metric_inputs(y_pred, y_true)

    with pytest.raises(ValueError, match="same batch size"):
        validate_metric_inputs(y_pred, torch.randn(3, 2))

    with pytest.raises(ValueError, match="cannot be scalars"):
        validate_metric_inputs(torch.tensor(1.0), torch.tensor(2.0))

    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_metric_inputs(torch.tensor([float("nan")]), torch.tensor([1.0]))


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
