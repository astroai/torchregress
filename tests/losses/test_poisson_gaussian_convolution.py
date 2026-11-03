"""Tests for PoissonGaussianLoss (Poisson shot noise convolved with Gaussian read noise)."""

from __future__ import annotations

import math

import mpmath
import pytest
import torch
from scipy.stats import poisson

import torchregress.losses._poisson_gaussian_kernel as kernel
from torchregress.losses import PoissonGaussianLoss, create_loss_from_config
from torchregress.losses._poisson_gaussian_kernel import poisson_gaussian_nll

F64 = torch.float64


def _oracle_nll(y: float, rate: float, var: float, gain: float, offset: float) -> float:
    """Untruncated convolution NLL in 50-digit arithmetic.

    The sum runs over every count whose term exceeds 1e-60 of the largest one,
    walking outward from the joint mode, so nothing material is cut.
    """
    mpmath.mp.dps = 50
    y, lam, s, g, b = (mpmath.mpf(x) for x in (y, rate, var, gain, offset))

    def term(k: int) -> mpmath.mpf:
        log_p = k * mpmath.log(lam) - lam - mpmath.loggamma(k + 1) if lam > 0 else 0
        if lam == 0 and k > 0:
            return mpmath.mpf(0)
        return mpmath.exp(log_p - (y - g * k - b) ** 2 / (2 * s)) / mpmath.sqrt(2 * mpmath.pi * s)

    centre = max(0, int(float((y - b) / g)))
    start = max(0, int(min(float(lam), centre)))
    stop = int(max(float(lam), centre)) + 1
    peak = max(term(k) for k in range(start, stop + 1, max(1, (stop - start) // 200)))
    total = sum(term(k) for k in range(start, stop))
    k = stop
    while True:
        t = term(k)
        total += t
        if t < peak * mpmath.mpf("1e-60") and k > lam:
            break
        k += 1
    k = start - 1
    while k >= 0:
        t = term(k)
        total += t
        if t < peak * mpmath.mpf("1e-60") and k < lam:
            break
        k -= 1
    return float(-mpmath.log(total))


ORACLE_CASES = [
    # (y, rate, read_var, gain, offset)
    (0.0, 0.0, 1.0, 1.0, 0.0),
    (1.3, 0.0, 4.0, 1.0, 0.0),
    (0.4, 0.05, 1.0, 1.0, 0.0),
    (2.5, 2.0, 9.0, 1.0, 0.0),
    (-4.0, 2.0, 9.0, 1.0, 0.0),
    (7.0, 5.0, 0.01, 1.0, 0.0),
    (104.5, 3.0, 9.0, 2.0, 100.0),
    (181.0, 40.0, 9.0, 2.0, 100.0),
    (1050.0, 1000.0, 25.0, 1.0, 0.0),
    (3.0e4, 2.5e4, 100.0, 1.2, 50.0),
    # Far outliers: posterior window disjoint from the prior window.
    (500.0, 2.0, 1.0, 1.0, 0.0),
    (-300.0, 50.0, 4.0, 1.0, 0.0),
    (2000.0, 10.0, 100.0, 0.5, -20.0),
]


@pytest.mark.parametrize(("y", "rate", "var", "gain", "offset"), ORACLE_CASES)
def test_matches_high_precision_oracle(y, rate, var, gain, offset) -> None:
    got = poisson_gaussian_nll(
        torch.tensor([y], dtype=F64), torch.tensor([rate], dtype=F64), var, gain, offset
    )
    want = _oracle_nll(y, rate, var, gain, offset)
    assert got.item() == pytest.approx(want, rel=1e-11, abs=1e-11)


def test_float32_tracks_float64_at_large_rates() -> None:
    y = torch.tensor([1.0e4, 1.02e4, 9.8e3, 1.5e4 + 3.0])
    rate = torch.tensor([1.0e4, 1.0e4, 1.0e4, 1.5e4])
    got = poisson_gaussian_nll(y, rate, 16.0)
    ref = poisson_gaussian_nll(y.double(), rate.double(), 16.0)
    torch.testing.assert_close(got.double(), ref, rtol=1e-5, atol=1e-4)


def _gaussian_gap(rate: float, gain: float, offset: float, var: float) -> tuple[float, float]:
    """Max |NLL - Gaussian NLL| and the max after the first Edgeworth term, |z| <= 2.5."""
    sd = math.sqrt(gain**2 * rate + var)
    z = torch.linspace(-2.5, 2.5, 21, dtype=F64)
    y = gain * rate + offset + sd * z
    got = poisson_gaussian_nll(y, torch.full_like(y, rate), var, gain, offset)
    gauss = -torch.distributions.Normal(gain * rate + offset, sd).log_prob(y)
    skew = gain**3 * rate / sd**3
    edgeworth = gauss - torch.log1p(skew / 6 * (z**3 - 3 * z))
    return (got - gauss).abs().max().item(), (got - edgeworth).abs().max().item()


def test_gaussian_limit_at_high_counts() -> None:
    # N(g rate + b, g^2 rate + var) is the high-count limit; the gap is the
    # skewness term skew/6 He3(z) with skew ~ 1/sqrt(rate).
    gap_hi, edge_hi = _gaussian_gap(2.0e4, 1.5, 30.0, 25.0)
    gap_lo, _ = _gaussian_gap(50.0, 1.5, 30.0, 25.0)
    assert gap_hi < 1.1e-2
    assert edge_hi < 2e-4
    assert gap_lo > 10 * gap_hi


@pytest.mark.parametrize("rate", [0.3, 2.0, 7.5, 60.0])
def test_poisson_limit_on_integer_data(rate: float) -> None:
    var = 1e-6
    k = torch.arange(0, int(rate + 6 * math.sqrt(rate)) + 3, dtype=F64)
    lam = torch.full_like(k, rate, requires_grad=True)
    nll = poisson_gaussian_nll(k, lam, var)
    # sigma -> 0: N(k; k, var) contributes 1 / sqrt(2 pi var); other counts vanish.
    want = -torch.from_numpy(poisson.logpmf(k.numpy(), rate)) + 0.5 * math.log(2 * math.pi * var)
    torch.testing.assert_close(nll.detach(), want, rtol=1e-10, atol=1e-9)
    nll.sum().backward()
    torch.testing.assert_close(lam.grad, 1.0 - k / rate, rtol=1e-8, atol=1e-8)


def _grad_inputs() -> tuple[torch.Tensor, ...]:
    y = torch.tensor([2.5, -1.0, 40.0, 104.0, 500.0, -80.0, 0.3], dtype=F64)
    rate = torch.tensor([2.0, 0.7, 35.0, 3.0, 2.0, 30.0, 0.02], dtype=F64)
    var = torch.tensor([4.0, 1.0, 9.0, 9.0, 1.0, 4.0, 1.0], dtype=F64)
    gain = torch.tensor([1.0, 1.0, 1.2, 2.0, 1.0, 0.8, 1.0], dtype=F64)
    offset = torch.tensor([0.0, 0.5, -3.0, 100.0, 0.0, 0.0, 0.0], dtype=F64)
    return tuple(t.requires_grad_() for t in (y, rate, var, gain, offset))


def test_gradcheck_float64_including_far_rows() -> None:
    inputs = _grad_inputs()
    lower, upper, far = kernel._support_window(*(t.detach() for t in inputs), tail_sigma=12.0)
    assert bool(far.any()) and bool((~far).any())
    assert torch.autograd.gradcheck(lambda *a: poisson_gaussian_nll(*a), inputs)


def test_gradgradcheck_float64() -> None:
    inputs = _grad_inputs()
    assert torch.autograd.gradgradcheck(lambda *a: poisson_gaussian_nll(*a), inputs)


def test_zero_rate_right_derivative() -> None:
    y = torch.tensor([0.0, 1.5, -2.0], dtype=F64)
    rate = torch.zeros(3, dtype=F64, requires_grad=True)
    poisson_gaussian_nll(y, rate, 2.0, 1.3, 0.1).sum().backward()
    h = 1e-7
    fd = (
        poisson_gaussian_nll(y, torch.full_like(y, h), 2.0, 1.3, 0.1)
        - poisson_gaussian_nll(y, torch.zeros_like(y), 2.0, 1.3, 0.1)
    ) / h
    torch.testing.assert_close(rate.grad, fd, rtol=1e-5, atol=1e-6)


def test_chunking_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = torch.Generator().manual_seed(0)
    rate = 50 * torch.rand(300, generator=gen, dtype=F64)
    y = torch.poisson(rate, generator=gen) + 2 * torch.randn(300, generator=gen, dtype=F64)
    y[::37] += 400.0  # far outliers in several chunks
    r1 = rate.clone().requires_grad_()
    one = poisson_gaussian_nll(y, r1, 4.0)
    one.sum().backward()
    monkeypatch.setattr(kernel, "_CHUNK_TERMS", 1000)
    assert kernel._chunk_size(rate, 12.0) < 300
    r2 = rate.clone().requires_grad_()
    many = poisson_gaussian_nll(y, r2, 4.0)
    many.sum().backward()
    torch.testing.assert_close(many, one, rtol=0, atol=0)
    torch.testing.assert_close(r2.grad, r1.grad, rtol=0, atol=0)


def test_streaming_and_autograd_paths_agree() -> None:
    inputs = _grad_inputs()
    first = torch.autograd.grad(poisson_gaussian_nll(*inputs).sum(), inputs)
    detached = tuple(t.detach().requires_grad_() for t in inputs)
    flat = tuple(t.reshape(-1) for t in detached)
    second = torch.autograd.grad(kernel._forward_autograd(flat, 12.0, 4096).sum(), detached)
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10)


class TestPoissonGaussianLoss:
    def test_registry_and_config(self) -> None:
        loss = create_loss_from_config({"type": "poisson_gaussian", "gain": 2.0})
        assert isinstance(loss, PoissonGaussianLoss)
        assert loss.detector_parameters(torch.zeros(1)) == (2.0, 0.0, 1.0)

    def test_matches_kernel_and_log_input(self) -> None:
        rate = torch.tensor([[3.0, 40.0], [0.5, 8.0]], dtype=F64)
        y = torch.tensor([[104.5, 181.0], [99.0, 117.0]], dtype=F64)
        fn = PoissonGaussianLoss(gain=2.0, offset=100.0, read_noise_variance=9.0, reduction="none")
        want = poisson_gaussian_nll(y, rate, 9.0, 2.0, 100.0)
        torch.testing.assert_close(fn(rate, y), want)
        fn_log = PoissonGaussianLoss(
            gain=2.0, offset=100.0, read_noise_variance=9.0, log_input=True, reduction="none"
        )
        torch.testing.assert_close(fn_log(rate.log(), y), want)

    def test_mask_ignores_nan_entries(self) -> None:
        rate = torch.tensor([2.0, float("nan"), 5.0], dtype=F64, requires_grad=True)
        y = torch.tensor([1.0, float("nan"), 6.0], dtype=F64)
        mask = torch.tensor([True, False, True])
        loss = PoissonGaussianLoss()(rate, y, mask=mask)
        want = poisson_gaussian_nll(y[mask], rate.detach()[mask], 1.0).mean()
        torch.testing.assert_close(loss.detach(), want)
        loss.backward()
        assert torch.isfinite(rate.grad).all() and rate.grad[1] == 0

    def test_weights_and_sum(self) -> None:
        rate = torch.tensor([2.0, 5.0], dtype=F64)
        y = torch.tensor([1.0, 6.0], dtype=F64)
        per = PoissonGaussianLoss(reduction="none")(rate, y)
        w = torch.tensor([1.0, 3.0], dtype=F64)
        got = PoissonGaussianLoss()(rate, y, weights=w)
        torch.testing.assert_close(got, (per * w).sum() / w.sum())
        torch.testing.assert_close(PoissonGaussianLoss(reduction="sum")(rate, y), per.sum())

    def test_per_call_calibration_maps(self) -> None:
        rate = torch.tensor([[2.0, 5.0]], dtype=F64)
        y = torch.tensor([[11.0, 18.0]], dtype=F64)
        var = torch.tensor([[1.0, 9.0]], dtype=F64)
        got = PoissonGaussianLoss(reduction="none")(
            rate,
            y,
            gain=torch.tensor(1.5, dtype=F64),
            offset=torch.tensor(8.0),
            read_noise_variance=var,
        )
        torch.testing.assert_close(got, poisson_gaussian_nll(y, rate, var, 1.5, 8.0))

    def test_learnable_parameters_receive_gradients(self) -> None:
        fn = PoissonGaussianLoss(
            gain=1.5,
            offset=2.0,
            read_noise_variance=4.0,
            learn_gain=True,
            learn_offset=True,
            learn_read_noise=True,
        )
        names = {n for n, _ in fn.named_parameters()}
        assert names == {"log_gain", "offset", "log_read_noise_variance"}
        fn(torch.tensor([3.0, 9.0]), torch.tensor([5.0, 17.0])).backward()
        for p in fn.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()

    def test_maximum_likelihood_recovers_rate_and_read_noise(self) -> None:
        gen = torch.Generator().manual_seed(1)
        true_rate, gain, offset, var = 3.0, 2.0, 10.0, 6.0
        counts = torch.poisson(torch.full((4000,), true_rate, dtype=F64), generator=gen)
        y = gain * counts + offset + math.sqrt(var) * torch.randn(4000, generator=gen, dtype=F64)
        log_rate = torch.zeros((), dtype=F64, requires_grad=True)
        fn = PoissonGaussianLoss(
            gain=gain, offset=offset, read_noise_variance=1.0, learn_read_noise=True, log_input=True
        ).double()
        opt = torch.optim.LBFGS([log_rate, *fn.parameters()], max_iter=100)

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = fn(log_rate.expand_as(y), y)
            loss.backward()
            return loss

        opt.step(closure)
        assert log_rate.exp().item() == pytest.approx(true_rate, rel=0.05)
        assert fn.log_read_noise_variance.exp().item() == pytest.approx(var, rel=0.15)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"gain": 0.0}, "gain"),
            ({"read_noise_variance": -1.0}, "read_noise_variance"),
            ({"tail_sigma": 0.0}, "tail_sigma"),
            ({"max_terms": 1}, "max_terms"),
        ],
    )
    def test_invalid_configuration_raises(self, kwargs, match) -> None:
        with pytest.raises(ValueError, match=match):
            PoissonGaussianLoss(**kwargs)

    def test_invalid_inputs_fail_closed(self) -> None:
        fn = PoissonGaussianLoss()
        with pytest.raises(ValueError, match="rate"):
            fn(torch.tensor([-1.0]), torch.tensor([0.0]))
        with pytest.raises(ValueError, match="target"):
            fn(torch.tensor([1.0]), torch.tensor([float("nan")]))
        with pytest.raises(TypeError):
            fn(torch.tensor([1]), torch.tensor([1]))
        with pytest.raises(ValueError, match="max_terms"):
            PoissonGaussianLoss(max_terms=64)(torch.tensor([1.0e4]), torch.tensor([1.0e4]))


@pytest.mark.parametrize("convert", ["double", "to"])
def test_learnable_parameters_convert_exactly_to_float64(convert: str) -> None:
    gain, offset, var = 1.7, 0.3, 2.5
    learn = PoissonGaussianLoss(
        gain=gain,
        offset=offset,
        read_noise_variance=var,
        learn_gain=True,
        learn_offset=True,
        learn_read_noise=True,
    )
    learn = learn.double() if convert == "double" else learn.to(F64)
    assert all(p.dtype == F64 for p in learn.parameters())
    assert learn.log_gain.item() == math.log(gain)
    assert learn.offset.item() == offset
    assert learn.log_read_noise_variance.item() == math.log(var)
    fixed = PoissonGaussianLoss(gain=gain, offset=offset, read_noise_variance=var)
    gen = torch.Generator().manual_seed(4)
    rate = torch.rand(64, dtype=F64, generator=gen) * 30.0
    y = gain * torch.poisson(rate, generator=gen) + offset
    y = y + math.sqrt(var) * torch.randn(64, dtype=F64, generator=gen)
    torch.testing.assert_close(
        learn(rate, y),
        fixed(rate, y),
        rtol=1e-15,
        atol=1e-15,
    )
    elementwise = PoissonGaussianLoss(
        gain=gain, offset=offset, read_noise_variance=var, learn_read_noise=True, reduction="none"
    ).double()
    fixed_none = PoissonGaussianLoss(
        gain=gain, offset=offset, read_noise_variance=var, reduction="none"
    )
    torch.testing.assert_close(elementwise(rate, y), fixed_none(rate, y), rtol=1e-15, atol=1e-15)


def test_learnable_parameters_float32_unchanged() -> None:
    learn = PoissonGaussianLoss(read_noise_variance=2.5, learn_read_noise=True).float()
    assert learn.log_read_noise_variance.dtype == torch.float32
    assert learn.log_read_noise_variance.item() == torch.tensor(math.log(2.5)).item()
