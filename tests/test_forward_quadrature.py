"""Analytic and independent numerical checks for scalar forward likelihoods."""

import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad

from torchregress.losses.forward_quadrature import GaussianLatentForwardObservationNLLLoss


def test_linear_map_matches_analytic_gaussian_and_gradients():
    mean = torch.tensor([-0.3, 0.5, 1.0], dtype=torch.double, requires_grad=True)
    std = torch.tensor([0.2, 0.4, 0.1], dtype=torch.double, requires_grad=True)
    y = torch.tensor([-1.0, 0.2, 1.8], dtype=torch.double)
    error = torch.tensor([0.7, 0.8, 0.5], dtype=torch.double, requires_grad=True)
    slope = torch.tensor(1.3, dtype=torch.double, requires_grad=True)
    bias = torch.tensor(-0.2, dtype=torch.double, requires_grad=True)
    actual = GaussianLatentForwardObservationNLLLoss(64, "none")(
        mean, std, y, error, lambda z: slope * z + bias
    )
    expected = -torch.distributions.Normal(
        slope * mean + bias, torch.sqrt(error.square() + (slope * std).square())
    ).log_prob(y)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    params = (mean, std, error, slope, bias)
    g_actual = torch.autograd.grad(actual.sum(), params, retain_graph=True)
    g_expected = torch.autograd.grad(expected.sum(), params)
    for a, e in zip(g_actual, g_expected, strict=True):
        torch.testing.assert_close(a, e, atol=2e-11, rtol=2e-11)


def test_nonlinear_map_against_independent_adaptive_integration():
    mean, std, y, error = 0.2, 0.35, -0.15, 0.7
    value, _ = quad(
        lambda t: (
            np.exp(-0.5 * t * t)
            / np.sqrt(2 * np.pi)
            * np.exp(-0.5 * ((y - np.exp(0.4 * (mean + std * t))) / error) ** 2)
            / (error * np.sqrt(2 * np.pi))
        ),
        -12,
        12,
        epsabs=1e-13,
        epsrel=1e-13,
    )
    args = [torch.tensor([v], dtype=torch.double) for v in (mean, std, y, error)]
    actual = GaussianLatentForwardObservationNLLLoss(64)(*args, lambda z: torch.exp(0.4 * z))
    assert actual.item() == pytest.approx(-math.log(value), abs=1e-12)


def test_zero_latent_scatter_is_normalized_observation_likelihood():
    args = [torch.tensor([v], dtype=torch.double) for v in (0.7, 0, -0.2, 0.4)]
    actual = GaussianLatentForwardObservationNLLLoss(32)(*args, lambda z: z.square())
    expected = -torch.distributions.Normal(0.49, 0.4).log_prob(
        torch.tensor(-0.2, dtype=torch.double)
    )
    torch.testing.assert_close(actual, expected)


def test_mask_and_weights_select_before_map_and_ignore_missing_nans():
    nan = float("nan")
    mean = torch.tensor([0.2, nan, nan, 0.5], dtype=torch.double, requires_grad=True)
    std = torch.tensor([0.1, nan, nan, 0.2], dtype=torch.double)
    y = torch.tensor([-0.1, nan, nan, 0.6], dtype=torch.double)
    error = torch.tensor([0.8, nan, nan, 0.9], dtype=torch.double)
    mask = torch.tensor([True, False, True, True])
    weights = torch.tensor([1.0, 5.0, 0.0, 3.0], dtype=torch.double)

    def mapping(z):
        assert z.shape == (2, 32) and torch.isfinite(z).all()
        return z

    per = GaussianLatentForwardObservationNLLLoss(32, "none")(
        mean, std, y, error, mapping, mask, weights
    )
    assert per[1] == per[2] == 0
    actual = GaussianLatentForwardObservationNLLLoss()(mean, std, y, error, mapping, mask, weights)
    torch.testing.assert_close(actual, per.sum() / 4)
    actual.backward()
    assert torch.isfinite(mean.grad).all() and mean.grad[1] == mean.grad[2] == 0


@pytest.mark.parametrize("size", [0, 3])
def test_all_empty_has_differentiable_zero_and_never_calls_map(size):
    mean = torch.full((size,), float("nan"), requires_grad=True)
    std = torch.full((size,), float("nan"), requires_grad=True)

    def forbidden(_):
        raise AssertionError("Map must not run for empty observations")

    loss = GaussianLatentForwardObservationNLLLoss()(
        mean, std, mean, std, forbidden, mask=torch.zeros(size, dtype=torch.bool)
    )
    assert loss == 0
    loss.backward()
    assert torch.isfinite(mean.grad).all() and torch.isfinite(std.grad).all()


def test_autocast_and_half_residual_promote_before_arithmetic():
    args = [torch.tensor([v], dtype=torch.float16) for v in (0, 0, 60000, 1)]
    fn = GaussianLatentForwardObservationNLLLoss()

    def mapping(z):
        return z * 0 - 60000

    expected = fn(*args, mapping)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = fn(*args, mapping)
    assert actual.dtype == torch.float32 and torch.isfinite(actual)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.item() == pytest.approx(0.5 * 120000**2, rel=1e-6)


def test_nonlinear_gradcheck():
    args = [
        torch.tensor([v], dtype=torch.double, requires_grad=True) for v in (0.2, 0.3, -0.1, 0.8)
    ]
    fn = GaussianLatentForwardObservationNLLLoss(32)
    assert torch.autograd.gradcheck(lambda *a: fn(*a, lambda z: torch.exp(0.2 * z)), tuple(args))


@pytest.mark.parametrize("index,value", [(1, -1), (3, 0), (3, -1), (2, float("inf"))])
def test_invalid_active_values_rejected(index, value):
    args = [torch.tensor([v]) for v in (0.0, 0.2, -0.4, 0.5)]
    args[index] = torch.tensor([float(value)])
    with pytest.raises(ValueError):
        GaussianLatentForwardObservationNLLLoss()(*args, lambda z: z)


def test_invalid_map_and_constructor_rejected():
    args = [torch.tensor([v]) for v in (0.0, 0.2, -0.4, 0.5)]
    for mapping in (lambda z: z[:, 0], lambda z: z * float("nan")):
        with pytest.raises(ValueError, match="observation_map"):
            GaussianLatentForwardObservationNLLLoss()(*args, mapping)
    for points in (0, 257, True, 2.5):
        with pytest.raises(ValueError):
            GaussianLatentForwardObservationNLLLoss(points)


def _normal_linear_nll(mean, std, y, error, slope, bias):
    scale = torch.sqrt(error.square() + (slope * std).square())
    return -torch.distributions.Normal(slope * mean + bias, scale).log_prob(y)


@pytest.mark.parametrize("points", [4, 32])
@pytest.mark.parametrize("ratio", [1e-3, 0.1, 1.0, 3.0, 14.0, 100.0, 1e3, 1e4])
def test_linear_map_exact_for_narrow_and_wide_likelihoods(points, ratio):
    # ratio = slope * latent_std / observation_std; the prior-centred rule fails for
    # ratio >> 1 (e.g. 14.2 is the harness repro: -0.30 nats at Q=64).
    slope, bias = 1.3, -0.2
    std = torch.tensor([0.548, 0.2, 1.5], dtype=torch.double)
    error = slope * std / ratio
    mean = torch.tensor([0.1, -2.0, 3.0], dtype=torch.double)
    # Observations at 0, 1.5 and -3 marginal standard deviations from the prediction.
    marginal = torch.sqrt(error.square() + (slope * std).square())
    y = slope * mean + bias + torch.tensor([0.0, 1.5, -3.0], dtype=torch.double) * marginal
    actual = GaussianLatentForwardObservationNLLLoss(points, "none")(
        mean, std, y, error, lambda z: slope * z + bias
    )
    expected = _normal_linear_nll(mean, std, y, error, slope, bias)
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=0)


def test_harness_repro_narrow_linear_likelihood():
    m, s, y, e = (torch.tensor([v], dtype=torch.double) for v in (0.1, 0.548, 0.3, 0.05))
    expected = _normal_linear_nll(m, s, y, e, 1.3, -0.2)
    for points in (16, 32, 64, 128, 256):
        actual = GaussianLatentForwardObservationNLLLoss(points, "none")(
            m, s, y, e, lambda z: 1.3 * z - 0.2
        )
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


def _quad_nll(mean, std, y, error, mapping):
    def integrand(z):
        log_value = (
            -0.5 * ((z - mean) / std) ** 2
            - 0.5 * ((y - mapping(z)) / error) ** 2
            - math.log(2 * math.pi * std * error)
        )
        return math.exp(log_value)

    # Break points at the prior mean, every real preimage of y and a few local peak
    # widths error/|map'| around it let the adaptive integrator resolve narrow peaks.
    lo, hi = mean - 14 * std, mean + 14 * std
    grid = np.linspace(lo, hi, 4001)
    with np.errstate(over="ignore", invalid="ignore"):
        values = mapping(grid) - y
    crossing = np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:]))
    points = {mean}
    for i in crossing:
        root = grid[i] - values[i] * (grid[i + 1] - grid[i]) / (values[i + 1] - values[i])
        width = error * (grid[i + 1] - grid[i]) / abs(values[i + 1] - values[i])
        points.update(root + k * width for k in (-30, -10, -3, -1, 0, 1, 3, 10, 30))
    points = sorted(p for p in points if lo < p < hi)
    value, _ = quad(integrand, lo, hi, points=points, limit=1000, epsabs=0, epsrel=1e-13)
    return -math.log(value)


def _softplus(z):
    return np.logaddexp(0.0, z)


NONLINEAR_CASES = [
    # (torch map, numpy map, mean, std, y, error)
    (torch.exp, np.exp, 0.1, 0.548, 1.3, 0.05),
    (torch.exp, np.exp, 0.0, 1.0, 2.0, 0.01),
    (torch.exp, np.exp, 1.5, 0.3, 4.0, 0.02),
    (torch.nn.functional.softplus, _softplus, 0.2, 0.8, 1.1, 0.02),
    (torch.nn.functional.softplus, _softplus, 0.3, 0.4, 0.9, 1.0),
    (lambda z: z**3, lambda z: z**3, 0.5, 0.6, 0.4, 0.01),
    (lambda z: z**3, lambda z: z**3, -0.2, 1.0, -1.5, 0.05),
    (lambda z: z**3, lambda z: z**3, 1.0, 0.2, 1.2, 0.5),
]


@pytest.mark.parametrize("case", range(len(NONLINEAR_CASES)))
def test_nonlinear_maps_match_adaptive_integration_for_narrow_likelihoods(case):
    torch_map, numpy_map, mean, std, y, error = NONLINEAR_CASES[case]
    expected = _quad_nll(mean, std, y, error, numpy_map)
    args = [torch.tensor([v], dtype=torch.double) for v in (mean, std, y, error)]
    actual = GaussianLatentForwardObservationNLLLoss(32)(*args, torch_map)
    assert actual.item() == pytest.approx(expected, abs=1e-6)


def test_likelihood_plateau_converges_with_order():
    # y is 5 error units above softplus's lower asymptote, so the likelihood keeps a
    # plateau of relative height exp(-12.5) across the broad prior's left tail. That
    # shoulder is not Gaussian around the peak, so convergence is algebraic rather
    # than exponential; the prior-centred rule is still 7e-3 nats off at Q=256.
    expected = _quad_nll(-1.0, 2.0, 0.05, 0.01, _softplus)
    args = [torch.tensor([v], dtype=torch.double) for v in (-1.0, 2.0, 0.05, 0.01)]
    softplus = torch.nn.functional.softplus
    errors = [
        abs(GaussianLatentForwardObservationNLLLoss(q)(*args, softplus).item() - expected)
        for q in (16, 64, 256)
    ]
    assert errors[0] < 1e-3 and errors[1] < 1e-5 and errors[2] < 1e-8
    assert errors[0] > errors[1] > errors[2]


def test_flux_regime_matches_adaptive_integration():
    # Harness-like fluxes y = exp(z) + noise at true SNR 7-50 with broad latent scatter;
    # the prior-centred rule had errors up to hundreds of nats here at Q=32.
    rng = np.random.default_rng(7)
    size = 40
    mean = rng.uniform(-1.0, 2.0, size)
    std = rng.uniform(0.1, 0.8, size)
    error = np.exp(mean) / rng.uniform(7.0, 50.0, size)
    y = np.exp(mean + std * rng.standard_normal(size)) + error * rng.standard_normal(size)
    expected = torch.tensor(
        [_quad_nll(*row, np.exp) for row in zip(mean, std, y, error, strict=True)],
        dtype=torch.double,
    )
    args = [torch.from_numpy(v) for v in (mean, std, y, error)]
    for points, tolerance in ((32, 1e-6), (64, 1e-9)):
        actual = GaussianLatentForwardObservationNLLLoss(points, "none")(*args, torch.exp)
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0)


@pytest.mark.parametrize(
    "mapping,values",
    [
        (lambda z: 1.3 * z - 0.2, (0.1, 0.548, 0.3, 0.05)),
        (torch.exp, (0.4, 0.6, 1.9, 0.04)),
        (lambda z: z**3, (0.5, 0.6, 0.4, 0.02)),
    ],
)
def test_narrow_likelihood_gradcheck(mapping, values):
    args = tuple(torch.tensor([v], dtype=torch.double, requires_grad=True) for v in values)
    fn = GaussianLatentForwardObservationNLLLoss(32)
    assert torch.autograd.gradcheck(lambda *a: fn(*a, mapping), args)


def test_narrow_linear_gradients_match_analytic():
    mean = torch.tensor([0.1, -0.4], dtype=torch.double, requires_grad=True)
    std = torch.tensor([0.548, 0.9], dtype=torch.double, requires_grad=True)
    y = torch.tensor([0.3, -0.5], dtype=torch.double, requires_grad=True)
    error = torch.tensor([0.05, 0.002], dtype=torch.double, requires_grad=True)
    slope = torch.tensor(1.3, dtype=torch.double, requires_grad=True)
    params = (mean, std, y, error, slope)
    actual = GaussianLatentForwardObservationNLLLoss(32, "sum")(
        mean, std, y, error, lambda z: slope * z - 0.2
    )
    expected = _normal_linear_nll(mean, std, y, error, slope, -0.2).sum()
    for a, e in zip(
        torch.autograd.grad(actual, params), torch.autograd.grad(expected, params), strict=True
    ):
        torch.testing.assert_close(a, e, atol=1e-9, rtol=1e-9)


def test_inference_mode_and_float32_narrow_likelihood():
    values = (0.1, 0.548, 0.3, 0.05)
    expected = _normal_linear_nll(
        *(torch.tensor([v], dtype=torch.double) for v in values), 1.3, -0.2
    )
    args = [torch.tensor([v], dtype=torch.float32) for v in values]
    with torch.inference_mode():
        actual = GaussianLatentForwardObservationNLLLoss(32)(*args, lambda z: 1.3 * z - 0.2)
    assert actual.dtype == torch.float32
    assert actual.item() == pytest.approx(expected.item(), abs=1e-5)


def test_constants_follow_input_dtype_and_device_not_module_casts():
    fn = GaussianLatentForwardObservationNLLLoss(32).to(torch.float16)
    assert fn.nodes.dtype == torch.float64 and not dict(fn.named_buffers())
    m, s, y, e = (torch.tensor([v], dtype=torch.double) for v in (0.1, 0.548, 0.3, 0.05))
    actual = fn(m, s, y, e, lambda z: 1.3 * z - 0.2)
    expected = _normal_linear_nll(m, s, y, e, 1.3, -0.2)
    torch.testing.assert_close(actual, expected.squeeze(), atol=1e-12, rtol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_module_moves_to_mps():
    fn = GaussianLatentForwardObservationNLLLoss(32, "none").to("mps")
    values = (0.1, 0.548, 0.3, 0.05)
    args = [torch.tensor([v], dtype=torch.float32, device="mps") for v in values]
    actual = fn(*args, torch.exp)
    cpu = fn(*(a.cpu() for a in args), torch.exp)
    assert actual.device.type == "mps"
    torch.testing.assert_close(actual.cpu(), cpu, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("points", [1, 2])
def test_one_and_two_node_rules_stay_prior_centred(points):
    m, s, y, e = (torch.tensor([v], dtype=torch.double) for v in (0.2, 0.3, -0.1, 0.8))
    x, w = np.polynomial.hermite.hermgauss(points)
    nodes = m + math.sqrt(2) * s * torch.from_numpy(x)
    log_terms = torch.distributions.Normal(torch.exp(nodes), e).log_prob(y) + torch.from_numpy(
        np.log(w) - 0.5 * math.log(math.pi)
    )
    actual = GaussianLatentForwardObservationNLLLoss(points, "none")(m, s, y, e, torch.exp)
    torch.testing.assert_close(actual, -torch.logsumexp(log_terms, 0)[None])
