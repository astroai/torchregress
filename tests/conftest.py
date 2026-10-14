import os

import pytest
import torch


def pytest_configure(config):
    """Register the ``cuda`` marker for device-sensitive tests."""
    config.addinivalue_line(
        "markers",
        "cuda: device-sensitive test that needs a CUDA GPU (skipped when CUDA is unavailable "
        "unless TORCHREGRESS_PARITY_DEVICE is set, e.g. to 'cpu')",
    )


def pytest_collection_modifyitems(config, items):
    """Skip ``cuda``-marked tests when no GPU is present.

    Setting ``TORCHREGRESS_PARITY_DEVICE`` (e.g. ``cpu``) opts in to running the
    device-parity tests against that device, so they can be exercised without a GPU.
    """
    if torch.cuda.is_available() or os.environ.get("TORCHREGRESS_PARITY_DEVICE"):
        return
    skip_cuda = pytest.mark.skip(
        reason="CUDA is not available (set TORCHREGRESS_PARITY_DEVICE=cpu)"
    )
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)


@pytest.fixture(autouse=True)
def _disable_matplotlib_latex():
    """CI runners often lack a TeX install; avoid cross-test usetex leakage."""
    import matplotlib.pyplot as plt

    previous = plt.rcParams["text.usetex"]
    plt.rcParams["text.usetex"] = False
    yield
    plt.rcParams["text.usetex"] = previous


@pytest.fixture
def device():
    """Return the device to use for tensor operations.

    ``TORCHREGRESS_TEST_DEVICE`` overrides the default (CUDA when available, else CPU).
    """
    override = os.environ.get("TORCHREGRESS_TEST_DEVICE")
    if override:
        return override
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def batch_size():
    """Return a standard batch size for tests."""
    return 10


@pytest.fixture
def sample_data(batch_size, device):
    """Return sample prediction and target tensors."""
    y_pred = torch.randn(batch_size, 1, device=device)
    y_true = torch.randn(batch_size, 1, device=device)
    return y_pred, y_true


@pytest.fixture
def sample_mask(batch_size, device):
    """Return a sample boolean mask."""
    return torch.randint(0, 2, (batch_size, 1), device=device).bool()


@pytest.fixture
def sample_weights(batch_size, device):
    """Return sample weights for weighted loss tests."""
    return torch.rand(batch_size, 1, device=device)
