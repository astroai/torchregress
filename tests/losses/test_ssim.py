"""Tests for 1-D structural similarity (``structural_similarity_1d``)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchmetrics.functional.image import structural_similarity_index_measure as tm_ssim

from torchregress.losses import StructuralSimilarity1dLoss, structural_similarity_1d


def _pair(rows: int = 3, n: int = 64, dtype: torch.dtype = torch.float64, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, n, generator=gen, dtype=dtype).cumsum(-1)
    y = x + 0.4 * torch.randn(rows, n, generator=gen, dtype=dtype)
    return x, y


def _np_window(k: int, sigma: float | None) -> np.ndarray:
    if sigma is None:
        return np.full(k, 1.0 / k)
    d = np.arange(k) - (k - 1) / 2
    g = np.exp(-0.5 * (d / sigma) ** 2)
    return g / g.sum()


def _np_ssim_row(
    x,
    y,
    *,
    k,
    sigma,
    data_range,
    k1=0.01,
    k2=0.03,
    padding="valid",
    mask=None,
    weights=None,
    min_coverage=0.5,
):
    """Direct per-window reference (no convolution, no shift trick)."""
    n = len(x)
    p = (k - 1) // 2
    g = _np_window(k, sigma)
    m = np.ones(n, dtype=bool) if mask is None else mask.copy()
    w = np.ones(n) if weights is None else weights.copy()
    x = np.where(m, x, 0.0)
    y = np.where(m, y, 0.0)
    w = np.where(m, w, 0.0)
    m = m & (w > 0)  # a zero weight is a masked sample
    gated = mask is not None or weights is not None or padding == "same"
    if padding == "reflect":
        x, y = np.pad(x, p, mode="reflect"), np.pad(y, p, mode="reflect")
        m, w = np.pad(m, p, mode="reflect"), np.pad(w, p, mode="reflect")
    elif padding == "same":
        x, y = np.pad(x, p), np.pad(y, p)
        m, w = np.pad(m, p), np.pad(w, p)
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    out = []
    for j in range(len(x) - k + 1):
        sl = slice(j, j + k)
        a = g * w[sl] * m[sl]
        mass = a.sum()
        cover = (g * m[sl]).sum()
        if mass <= 0 or (gated and cover < min_coverage):
            out.append(np.nan)
            continue
        mx, my = (a * x[sl]).sum() / mass, (a * y[sl]).sum() / mass
        vx = max((a * (x[sl] - mx) ** 2).sum() / mass, 0.0)
        vy = max((a * (y[sl] - my) ** 2).sum() / mass, 0.0)
        cxy = (a * (x[sl] - mx) * (y[sl] - my)).sum() / mass
        out.append(((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2)))
    out = np.asarray(out)
    return out, np.nanmean(out) if np.isfinite(out).any() else np.nan


class TestTorchmetricsParity:
    @pytest.mark.parametrize("sigma,kernel", [(1.5, 11), (None, 7), (2.0, 15)])
    @pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-12), (torch.float32, 2e-5)])
    def test_reflect_padding_matches_torchmetrics_score(self, sigma, kernel, dtype, tol) -> None:
        x, y = _pair(dtype=dtype)
        if sigma is None:
            opts = {"gaussian_kernel": False, "kernel_size": (kernel, 1)}
        else:
            # torchmetrics derives the Gaussian size from sigma: int(3.5 s + 0.5) * 2 + 1.
            assert int(3.5 * sigma + 0.5) * 2 + 1 == kernel
            opts = {"sigma": (sigma, 0.1), "kernel_size": (kernel, 1)}
        ref = tm_ssim(
            x[:, None, :, None], y[:, None, :, None], data_range=5.0, reduction="none", **opts
        )
        ours = structural_similarity_1d(
            x,
            y,
            kernel_size=kernel,
            sigma=sigma,
            data_range=5.0,
            padding="reflect",
            reduction="none",
        )
        torch.testing.assert_close(ours, ref, atol=tol, rtol=0)
        mean = structural_similarity_1d(
            x, y, kernel_size=kernel, sigma=sigma, data_range=5.0, padding="reflect"
        )
        torch.testing.assert_close(mean, ref.mean(), atol=tol, rtol=0)

    def test_valid_map_equals_cropped_torchmetrics_map(self) -> None:
        x, y = _pair()
        _, full = tm_ssim(
            x[:, None, :, None],
            y[:, None, :, None],
            sigma=(1.5, 0.1),
            kernel_size=(11, 1),
            data_range=5.0,
            reduction="none",
            return_full_image=True,
        )
        score, ssim_map = structural_similarity_1d(
            x, y, data_range=5.0, reduction="none", return_map=True
        )
        cropped = full[:, 0, 5:-5, 0]
        assert ssim_map.shape == (3, 64 - 10)
        # torchmetrics forms E[x^2] - E[x]^2 without centring: agreement to its round-off.
        torch.testing.assert_close(ssim_map, cropped, atol=1e-11, rtol=0)
        torch.testing.assert_close(score, cropped.mean(-1), atol=1e-11, rtol=0)


class TestNumpyReference:
    @pytest.mark.parametrize("padding", ["valid", "same", "reflect"])
    @pytest.mark.parametrize("sigma,kernel", [(1.5, 11), (None, 5)])
    def test_masked_weighted_map_matches_direct_windows(self, padding, sigma, kernel) -> None:
        x, y = _pair(rows=4, n=48, seed=3)
        gen = torch.Generator().manual_seed(7)
        mask = torch.rand(4, 48, generator=gen) > 0.25
        mask[1, 10:30] = False  # a long gap: windows inside it must be excluded
        weights = torch.rand(4, 48, generator=gen, dtype=torch.float64) + 0.1
        weights[2, 5:40:3] = 0.0  # zero weights count as masked samples
        ssim_map_score, ssim_map = structural_similarity_1d(
            x,
            y,
            kernel_size=kernel,
            sigma=sigma,
            data_range=4.0,
            padding=padding,
            mask=mask,
            weights=weights,
            reduction="none",
            return_map=True,
        )
        for i in range(4):
            ref_map, ref_score = _np_ssim_row(
                x[i].numpy(),
                y[i].numpy(),
                k=kernel,
                sigma=sigma,
                data_range=4.0,
                padding=padding,
                mask=mask[i].numpy(),
                weights=weights[i].numpy(),
            )
            np.testing.assert_array_equal(np.isnan(ssim_map[i].numpy()), np.isnan(ref_map))
            np.testing.assert_allclose(ssim_map[i].numpy(), ref_map, atol=1e-12, equal_nan=True)
            np.testing.assert_allclose(float(ssim_map_score[i]), ref_score, atol=1e-12)
        assert torch.isnan(ssim_map[1, 15]).item()  # inside the gap

    def test_unmasked_valid_matches_direct_windows(self) -> None:
        x, y = _pair(rows=2, n=40, seed=5)
        _, ssim_map = structural_similarity_1d(
            x, y, data_range=3.0, reduction="none", return_map=True
        )
        for i in range(2):
            ref_map, _ = _np_ssim_row(x[i].numpy(), y[i].numpy(), k=11, sigma=1.5, data_range=3.0)
            np.testing.assert_allclose(ssim_map[i].numpy(), ref_map, atol=1e-12)


class TestMaskAndWeights:
    def test_all_true_mask_and_unit_or_constant_weights_change_nothing(self) -> None:
        x, y = _pair()
        base = structural_similarity_1d(x, y, reduction="none")
        ones = torch.ones_like(x)
        torch.testing.assert_close(
            structural_similarity_1d(x, y, mask=ones.bool(), reduction="none"), base
        )
        torch.testing.assert_close(
            structural_similarity_1d(x, y, weights=ones, reduction="none"), base
        )
        # A per-row constant weight cancels in every normalised moment.
        scale = torch.tensor([[0.5], [3.0], [10.0]], dtype=torch.float64)
        torch.testing.assert_close(
            structural_similarity_1d(x, y, weights=scale * ones, reduction="none"), base
        )

    def test_nan_at_masked_samples_does_not_leak_into_value_or_gradient(self) -> None:
        x, y = _pair()
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, 20:24] = False
        x_bad, y_bad = x.clone(), y.clone()
        x_bad[~mask] = float("nan")
        y_bad[~mask] = float("inf")
        w = torch.ones_like(x)
        w[~mask] = float("nan")
        pred = x_bad.clone().requires_grad_(True)
        score = structural_similarity_1d(pred, y_bad, mask=mask, weights=w)
        score.backward()
        assert torch.isfinite(score)
        assert pred.grad is not None and torch.isfinite(pred.grad).all()
        assert (pred.grad[~mask] == 0).all()
        clean = structural_similarity_1d(x, y, mask=mask)
        torch.testing.assert_close(score.detach(), clean)

    def test_fully_masked_row_is_nan_and_dropped_from_mean(self) -> None:
        x, y = _pair()
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[1] = False
        rows = structural_similarity_1d(x, y, data_range=5.0, mask=mask, reduction="none")
        assert torch.isnan(rows[1]) and torch.isfinite(rows[[0, 2]]).all()
        mean = structural_similarity_1d(x, y, data_range=5.0, mask=mask)
        torch.testing.assert_close(mean, rows[[0, 2]].mean())

    def test_low_support_windows_are_excluded_by_min_coverage(self) -> None:
        x, y = _pair(rows=1, n=40)
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[0, ::2] = False  # half of every window is missing
        _, strict = structural_similarity_1d(
            x,
            y,
            sigma=None,
            kernel_size=5,
            mask=mask,
            min_coverage=0.7,
            reduction="none",
            return_map=True,
        )
        _, loose = structural_similarity_1d(
            x,
            y,
            sigma=None,
            kernel_size=5,
            mask=mask,
            min_coverage=0.3,
            reduction="none",
            return_map=True,
        )
        assert torch.isnan(strict).all()
        assert torch.isfinite(loose).all()

    def test_identical_sequences_score_one(self) -> None:
        x, _ = _pair()
        mask = torch.rand(x.shape, generator=torch.Generator().manual_seed(1)) > 0.2
        weights = torch.rand(x.shape, dtype=torch.float64) + 0.5
        for padding in ("valid", "same", "reflect"):
            score = structural_similarity_1d(x, x, mask=mask, weights=weights, padding=padding)
            torch.testing.assert_close(score, torch.tensor(1.0, dtype=torch.float64))

    @pytest.mark.parametrize("padding", ["valid", "same", "reflect"])
    def test_zero_weight_is_a_masked_sample(self, padding) -> None:
        x, y = _pair(rows=2, n=48, seed=5)
        w = torch.rand(x.shape, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
        w = w + 0.5
        w[0, 10:16] = 0.0  # a bad stretch: one window keeps a single weighted sample
        w[0, 12] = 1.0
        w[1, ::3] = 0.0
        kw = {"kernel_size": 7, "padding": padding, "reduction": "none", "return_map": True}
        score, smap = structural_similarity_1d(x, y, weights=w, **kw)
        ref_score, ref_map = structural_similarity_1d(x, y, mask=w > 0, weights=w, **kw)
        torch.testing.assert_close(score, ref_score, equal_nan=True)
        torch.testing.assert_close(smap, ref_map, equal_nan=True)
        assert torch.isnan(smap).any()  # low-coverage windows are excluded

    def test_values_at_zero_weight_samples_are_ignored(self) -> None:
        x, y = _pair(rows=2, n=48, seed=7)
        w = torch.ones_like(x)
        w[:, 20:23] = 0.0
        base = structural_similarity_1d(x, y, weights=w, reduction="none")
        for bad in (float("nan"), float("inf"), 1e12):
            y_bad, x_bad = y.clone(), x.clone()
            y_bad[w == 0] = bad
            x_bad[w == 0] = -bad
            pred = x_bad.requires_grad_(True)
            score = structural_similarity_1d(pred, y_bad, weights=w, reduction="none")
            torch.testing.assert_close(score.detach(), base)
            score.sum().backward()
            assert pred.grad is not None and torch.isfinite(pred.grad).all()
            assert (pred.grad[w == 0] == 0).all()

    def test_weights_shift_the_score_towards_heavily_weighted_samples(self) -> None:
        x, y = _pair(rows=1, n=64, seed=2)
        y = y.clone()
        y[0, 32:] = x[0, 32:]  # second half reproduced exactly
        w = torch.ones_like(x)
        w[0, 32:] = 100.0
        plain = structural_similarity_1d(x, y)
        weighted = structural_similarity_1d(x, y, weights=w)
        assert weighted > plain


class TestShapesAndOptions:
    def test_leading_dims_are_batched(self) -> None:
        x, y = _pair(rows=6)
        flat = structural_similarity_1d(x, y, data_range=5.0, reduction="none")
        nested = structural_similarity_1d(
            x.reshape(2, 3, -1), y.reshape(2, 3, -1), data_range=5.0, reduction="none"
        )
        torch.testing.assert_close(nested, flat.reshape(2, 3))
        single = structural_similarity_1d(x[0], y[0], data_range=5.0, reduction="none")
        assert single.shape == ()
        torch.testing.assert_close(single, flat[0])

    def test_per_row_data_range(self) -> None:
        x, y = _pair()
        ranges = torch.tensor([2.0, 5.0, 9.0], dtype=torch.float64)
        rows = structural_similarity_1d(x, y, data_range=ranges, reduction="none")
        for i in range(3):
            torch.testing.assert_close(
                rows[i], structural_similarity_1d(x[i], y[i], data_range=float(ranges[i]))
            )
        default = structural_similarity_1d(x, y, reduction="none")
        target_span = y.amax(-1) - y.amin(-1)
        torch.testing.assert_close(
            default, structural_similarity_1d(x, y, data_range=target_span, reduction="none")
        )

    def test_reductions(self) -> None:
        x, y = _pair()
        rows = structural_similarity_1d(x, y, reduction="none")
        torch.testing.assert_close(structural_similarity_1d(x, y, reduction="sum"), rows.sum())
        torch.testing.assert_close(structural_similarity_1d(x, y), rows.mean())

    @pytest.mark.parametrize(
        "kwargs,error",
        [
            ({"kernel_size": 4}, ValueError),
            ({"kernel_size": 101}, ValueError),
            ({"sigma": 0.0}, ValueError),
            ({"padding": "circular"}, ValueError),
            ({"min_coverage": 0.0}, ValueError),
            ({"data_range": -1.0}, ValueError),
            ({"weights": -torch.ones(3, 64, dtype=torch.float64)}, ValueError),
            ({"mask": torch.ones(3, 64)}, TypeError),
        ],
    )
    def test_invalid_inputs_fail_closed(self, kwargs, error) -> None:
        x, y = _pair()
        with pytest.raises(error):
            structural_similarity_1d(x, y, **kwargs)

    def test_moments_use_one_depthwise_convolution(self, monkeypatch) -> None:
        # A [C*R, 1, L] single-channel batch takes PyTorch's CPU im2col path,
        # which copies the input kernel_size times (8x memory at k = 11). The
        # moments must go through one grouped (depthwise) convolution instead.
        import torch.nn.functional as F

        calls: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
        real = F.conv1d

        def spy(inp, weight, *args, **kwargs):
            groups = kwargs.get("groups", args[4] if len(args) > 4 else 1)
            calls.append((tuple(inp.shape), tuple(weight.shape), groups))
            return real(inp, weight, *args, **kwargs)

        monkeypatch.setattr(F, "conv1d", spy)
        x, y = _pair(rows=4, n=40)
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, 3] = False
        structural_similarity_1d(x, y, mask=mask, weights=torch.ones_like(x) + 1.0)
        assert len(calls) == 1
        (inp, weight, groups) = calls[0]
        assert inp[0] == 1 and groups == inp[1] == weight[0] and weight[1] == 1

    def test_constant_row_with_dynamic_range_is_nan(self) -> None:
        x, _ = _pair(rows=1)
        flat = torch.ones_like(x)
        assert torch.isnan(structural_similarity_1d(flat, flat))
        torch.testing.assert_close(
            structural_similarity_1d(flat, flat, data_range=1.0),
            torch.tensor(1.0, dtype=torch.float64),
        )


class TestGradients:
    @pytest.mark.parametrize("padding", ["valid", "same", "reflect"])
    def test_gradcheck_masked_weighted(self, padding) -> None:
        x, y = _pair(rows=2, n=20, seed=4)
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[0, 5:8] = False
        weights = (
            torch.rand(x.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
            + 0.5
        )
        pred = x.clone().requires_grad_(True)
        targ = y.clone().requires_grad_(True)

        def fn(p, t):
            return structural_similarity_1d(
                p,
                t,
                kernel_size=5,
                sigma=1.0,
                data_range=3.0,
                padding=padding,
                mask=mask,
                weights=weights,
                reduction="none",
            )

        assert torch.autograd.gradcheck(fn, (pred, targ))

    def test_gradcheck_unmasked(self) -> None:
        x, y = _pair(rows=2, n=16, seed=6)
        pred = x.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda p: structural_similarity_1d(p, y, kernel_size=7, sigma=None, data_range=2.0),
            (pred,),
        )

    def test_float32_matches_float64(self) -> None:
        x, y = _pair()
        s64 = structural_similarity_1d(x, y, reduction="none")
        s32 = structural_similarity_1d(x.float(), y.float(), reduction="none")
        assert s32.dtype == torch.float32
        torch.testing.assert_close(s32.double(), s64, atol=1e-5, rtol=0)


class TestLossClass:
    def test_loss_is_one_minus_ssim_with_reductions(self) -> None:
        x, y = _pair()
        rows = structural_similarity_1d(x, y, reduction="none")
        torch.testing.assert_close(StructuralSimilarity1dLoss(reduction="none")(x, y), 1 - rows)
        torch.testing.assert_close(StructuralSimilarity1dLoss()(x, y), (1 - rows).mean())
        torch.testing.assert_close(
            StructuralSimilarity1dLoss(reduction="sum")(x, y), (1 - rows).sum()
        )

    def test_fully_masked_row_contributes_zero(self) -> None:
        x, y = _pair()
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[2] = False
        loss_fn = StructuralSimilarity1dLoss(data_range=5.0, reduction="none")
        per_row = loss_fn(x, y, mask=mask)
        assert per_row[2] == 0
        rows = structural_similarity_1d(x, y, data_range=5.0, mask=mask, reduction="none")
        torch.testing.assert_close(per_row[:2], 1 - rows[:2])
        mean = StructuralSimilarity1dLoss(data_range=5.0)(x, y, mask=mask)
        torch.testing.assert_close(mean, (1 - rows[:2]).mean())

    def test_training_step_reduces_loss(self) -> None:
        x, y = _pair()
        pred = (y + 0.5 * torch.randn_like(y)).requires_grad_(True)
        loss_fn = StructuralSimilarity1dLoss(data_range=5.0)
        opt = torch.optim.SGD([pred], lr=5.0)
        start = loss_fn(pred, y).item()
        for _ in range(20):
            opt.zero_grad()
            loss_fn(pred, y).backward()
            opt.step()
        assert loss_fn(pred, y).item() < start

    def test_invalid_constructor_arguments(self) -> None:
        with pytest.raises(ValueError):
            StructuralSimilarity1dLoss(kernel_size=10)
        with pytest.raises(ValueError):
            StructuralSimilarity1dLoss(data_range=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu() -> None:
    x, y = _pair()
    mask = torch.rand(x.shape, generator=torch.Generator().manual_seed(0)) > 0.2
    cpu = structural_similarity_1d(x, y, mask=mask, padding="same", reduction="none")
    gpu = structural_similarity_1d(
        x.cuda(), y.cuda(), mask=mask.cuda(), padding="same", reduction="none"
    )
    torch.testing.assert_close(gpu.cpu(), cpu)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_float32_matches_cpu() -> None:
    x, y = _pair(dtype=torch.float32)
    cpu = structural_similarity_1d(x, y, reduction="none")
    mps = structural_similarity_1d(x.to("mps"), y.to("mps"), reduction="none")
    torch.testing.assert_close(mps.cpu(), cpu, atol=1e-5, rtol=0)
