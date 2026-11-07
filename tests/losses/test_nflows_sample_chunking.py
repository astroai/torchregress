"""NormalizingFlowLoss.sample splits large requests into bounded blocks."""

from __future__ import annotations

import pytest
import torch
from scipy import stats

zuko = pytest.importorskip("zuko")

from torchregress.losses import nflows  # noqa: E402
from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model  # noqa: E402

CHUNK = 64


@pytest.fixture
def small_chunks(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Shrink the block size and record the rows of every block."""
    monkeypatch.setattr(nflows, "_SAMPLE_CHUNK_ROWS", CHUNK)
    rows: list[int] = []
    original = NormalizingFlowLoss._sample_block

    def spy(self, y_pred, n_samples, conditional, n_rows=None):  # type: ignore[no-untyped-def]
        batch = y_pred.shape[0] if n_rows is None else n_rows
        rows.append(batch * n_samples)
        return original(self, y_pred, n_samples, conditional, n_rows=n_rows)

    monkeypatch.setattr(NormalizingFlowLoss, "_sample_block", spy)
    return rows


@pytest.mark.parametrize(
    ("batch", "n_samples", "shape"),
    [
        (10, 5, (10, 5, 3)),  # one block
        (40, 10, (40, 10, 3)),  # several context blocks
        (3, 150, (3, 150, 3)),  # draws alone exceed a block
        (200, 1, (200, 3)),  # n_samples == 1 keeps the 2-D shape
    ],
)
def test_conditional_shapes_and_block_sizes(
    small_chunks: list[int], batch: int, n_samples: int, shape: tuple[int, ...]
) -> None:
    torch.manual_seed(0)
    loss = NormalizingFlowLoss(create_flow_model(3, 2, flow_type="nsf", n_transforms=2))
    samples = loss.sample(torch.randn(batch, 2), n_samples=n_samples)
    assert samples.shape == shape
    assert bool(torch.isfinite(samples).all())
    assert max(small_chunks) <= CHUNK
    assert sum(small_chunks) == batch * n_samples


@pytest.mark.parametrize(("batch", "n_samples", "shape"), [(200, 1, (200, 2)), (30, 4, (30, 4, 2))])
def test_unconditional_shapes_and_block_sizes(
    small_chunks: list[int], batch: int, n_samples: int, shape: tuple[int, ...]
) -> None:
    torch.manual_seed(1)
    loss = NormalizingFlowLoss(create_flow_model(2, 0, flow_type="maf", n_transforms=2))
    samples = loss.sample(torch.zeros(batch, 0), n_samples=n_samples)
    assert samples.shape == shape
    assert max(small_chunks) <= CHUNK
    assert sum(small_chunks) == batch * n_samples


def test_chunked_draws_follow_the_same_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(2)
    loss = NormalizingFlowLoss(create_flow_model(2, 2, flow_type="tarflow", hidden_features=16))
    with torch.no_grad():
        for block in loss.flow.transform.transforms:
            torch.nn.init.normal_(block.head.weight, std=0.3)
    context = torch.randn(4, 2)
    monkeypatch.setattr(nflows, "_SAMPLE_CHUNK_ROWS", 10**9)
    torch.manual_seed(3)
    whole = loss.sample(context, n_samples=4000)
    monkeypatch.setattr(nflows, "_SAMPLE_CHUNK_ROWS", 500)
    torch.manual_seed(4)
    chunked = loss.sample(context, n_samples=4000)
    # Same per-context law. The randomized flow is heavy-tailed (excess kurtosis
    # up to ~90), so moments are noisy; compare each marginal with a two-sample
    # Kolmogorov-Smirnov test instead (8 tests, Bonferroni-style threshold).
    for i in range(context.shape[0]):
        for j in range(2):
            result = stats.ks_2samp(whole[i, :, j].numpy(), chunked[i, :, j].numpy())
            assert result.pvalue > 1e-3, (i, j, result)


def test_small_requests_are_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    loss = NormalizingFlowLoss(create_flow_model(3, 2, flow_type="nsf", n_transforms=2))
    context = torch.randn(8, 2)
    torch.manual_seed(5)
    reference = loss.sample(context, n_samples=16)
    monkeypatch.setattr(nflows, "_SAMPLE_CHUNK_ROWS", 10**9)
    torch.manual_seed(5)
    torch.testing.assert_close(loss.sample(context, n_samples=16), reference)
