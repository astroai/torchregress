import pytest
import numpy as np
import matplotlib.pyplot as plt
from torchregress.viz.diagnostic import _prepare_interval_data, _add_interval_elements

def test_prepare_interval_data():
    y_pred = np.array([3.0, 1.0, 2.0])
    y_lower = np.array([2.5, 0.5, 1.5])
    y_upper = np.array([3.5, 1.5, 2.5])

    # Test sorted=False
    x, pred, lower, upper, true = _prepare_interval_data(
        y_pred=y_pred.copy(),
        y_lower=y_lower.copy(),
        y_upper=y_upper.copy(),
        x=None,
        y_true=None,
        sorted_by_pred=False
    )
    np.testing.assert_array_equal(x, [0, 1, 2])
    np.testing.assert_array_equal(pred, [3.0, 1.0, 2.0])
    assert true is None

    # Test sorted=True
    x, pred, lower, upper, true = _prepare_interval_data(
        y_pred=y_pred.copy(),
        y_lower=y_lower.copy(),
        y_upper=y_upper.copy(),
        x=np.array([10, 20, 30]),
        y_true=np.array([3.1, 0.9, 2.1]),
        sorted_by_pred=True
    )
    # Expected order based on sorted y_pred ([1.0, 2.0, 3.0] -> indices [1, 2, 0])
    np.testing.assert_array_equal(pred, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(x, [20, 30, 10])
    np.testing.assert_array_equal(true, [0.9, 2.1, 3.1])

def test_add_interval_elements():
    fig, ax = plt.subplots()
    x = np.array([0, 1, 2])
    y_pred = np.array([1.0, 2.0, 3.0])
    y_lower = np.array([0.5, 1.5, 2.5])
    y_upper = np.array([1.5, 2.5, 3.5])

    _add_interval_elements(
        ax=ax,
        x=x,
        y_pred=y_pred,
        y_lower=y_lower,
        y_upper=y_upper,
        y_true=None,
        title="Test Plot",
        xlabel="X",
        ylabel="Y",
        color_pred="blue",
        color_interval="red",
        color_true="green",
        alpha=0.5
    )

    # Check title
    assert ax.get_title() == "Test Plot"
    assert ax.get_xlabel() == "X"
    assert ax.get_ylabel() == "Y"
    plt.close(fig)

    # Test with y_true (coverage calculation)
    fig, ax = plt.subplots()
    y_true = np.array([1.0, 2.0, 4.0]) # 2 out of 3 in interval -> 66.7% coverage
    _add_interval_elements(
        ax=ax,
        x=x,
        y_pred=y_pred,
        y_lower=y_lower,
        y_upper=y_upper,
        y_true=y_true,
        title="Test Plot",
        xlabel="X",
        ylabel="Y",
        color_pred="blue",
        color_interval="red",
        color_true="green",
        alpha=0.5
    )
    assert ax.get_title() == "Test Plot (Coverage: 66.7%)"
    plt.close(fig)

print("Tests prepared.")
