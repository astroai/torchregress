"""Synthetic checks for a correlated spline flow with sparse targets."""

import math

import pytest
import torch

pytest.importorskip("zuko")

from torchregress.models.copula_spline import GaussianCopulaSplineHead


def test_identity_flow_exact_marginals_for_arbitrary_masks():
    torch.manual_seed(2)
    head = GaussianCopulaSplineHead(4, 6, width=8).double()
    x = torch.randn(5, 4, dtype=torch.double)
    y = torch.randn(5, 6, dtype=torch.double)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    y[~mask] = torch.nan
    got = head.intrinsic_log_prob(x, y, mask)
    safe = y.nan_to_num()
    _, lower, _ = head._distribution_parameters(x)
    sd = lower.diagonal(dim1=-2, dim2=-1)
    torch.testing.assert_close(sd, torch.ones_like(sd), atol=1e-7, rtol=0)
    expected = ((-0.5 * ((safe / sd).square() + math.log(2 * math.pi)) - sd.log()) * mask).sum(-1)
    torch.testing.assert_close(got, expected, atol=1e-12, rtol=1e-12)


def test_correlated_complete_and_partial_densities_match_gaussian_reference():
    head = GaussianCopulaSplineHead(2, 3, width=4).double()
    # Packed lower triangular order: (0,0),(1,0),(1,1),(2,0),(2,1),(2,2).
    with torch.no_grad():
        head.net[-1].bias[3 + 1] = 0.6
        head.net[-1].bias[3 + 3] = -0.25
    x = torch.zeros(3, 2, dtype=torch.double)
    y = torch.tensor([[0.4, -0.8, 0.2]] * 3, dtype=torch.double)
    mask = torch.tensor([[1, 1, 1], [1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    got = head.intrinsic_log_prob(x, y, mask)
    mean, lower, _ = head._distribution_parameters(x)
    covariance = lower @ lower.transpose(-1, -2)
    ref = []
    for i in range(3):
        cols = mask[i].nonzero().flatten()
        law = torch.distributions.MultivariateNormal(
            mean[i, cols], covariance_matrix=covariance[i][cols][:, cols]
        )
        ref.append(law.log_prob(y[i, cols]))
    torch.testing.assert_close(got, torch.stack(ref), atol=1e-12, rtol=1e-12)


def test_quoted_noise_matches_independent_gaussian_convolution():
    head = GaussianCopulaSplineHead(2, 3, width=4).double()
    x = torch.zeros(1, 2, dtype=torch.double)
    y = torch.tensor([[0.0, 0.3, 0.0]], dtype=torch.double)
    mask = torch.tensor([[False, True, False]])
    sigma = torch.tensor([[float("nan"), 0.25, float("nan")]], dtype=torch.double)
    nodes = torch.quasirandom.SobolEngine(3, scramble=True, seed=9).draw(4096, dtype=torch.double)
    nodes = torch.special.ndtri(nodes)
    got, ess = head.observed_log_prob(x, y, sigma, mask, nodes)
    expected = torch.distributions.Normal(0.0, math.sqrt(1 + 0.25**2)).log_prob(y[0, 1])
    torch.testing.assert_close(got[0], expected, atol=2e-4, rtol=0)
    assert 0 < ess.item() <= 1
    zero, zero_ess = head.observed_log_prob(x, y, torch.zeros_like(sigma), mask, nodes[:16])
    torch.testing.assert_close(zero, head.intrinsic_log_prob(x, y, mask), atol=1e-12, rtol=0)
    torch.testing.assert_close(zero_ess, torch.ones_like(zero_ess), atol=1e-12, rtol=0)


def test_native_samples_are_coupled_and_gradients_reach_flow_parameters():
    head = GaussianCopulaSplineHead(2, 3, width=8).double()
    with torch.no_grad():
        head.net[-1].bias[3 + 1] = 0.7
        head.net[-1].bias[3 + 6 + 0] = 0.3  # first spline width logit
    x = torch.zeros(1, 2, dtype=torch.double)
    draws = head.sample(x, 12000, generator=torch.Generator().manual_seed(4))[:, 0]
    assert torch.corrcoef(draws.T)[0, 1] > 0.4
    y = torch.tensor([[0.3, -0.5, 0.8]], dtype=torch.double)
    loss = -head.intrinsic_log_prob(x, y, torch.ones_like(y, dtype=torch.bool)).mean()
    loss.backward()
    assert torch.isfinite(head.net[-1].bias.grad).all()
    assert head.net[-1].bias.grad.abs().sum() > 0


def test_nonlinear_partial_density_matches_numerical_missing_integral():
    head = GaussianCopulaSplineHead(2, 2, width=4, bins=4).double()
    with torch.no_grad():
        head.net[-1].bias[2 + 1] = 0.6  # nonzero Gaussian base correlation
        head.net[-1].bias[2 + 3 + 0] = 0.8  # nonlinear first-coordinate spline
        head.net[-1].bias[2 + 3 + 4] = -0.5
    observed = torch.tensor([[0.4, 0.0]], dtype=torch.double)
    partial = head.intrinsic_log_prob(
        torch.zeros(1, 2, dtype=torch.double), observed, torch.tensor([[True, False]])
    ).exp()[0]
    grid = torch.linspace(-12, 12, 4001, dtype=torch.double)
    values = torch.stack((torch.full_like(grid, 0.4), grid), dim=-1)
    joint = head.intrinsic_log_prob(
        torch.zeros(len(grid), 2, dtype=torch.double),
        values,
        torch.ones_like(values, dtype=torch.bool),
    ).exp()
    numerical = torch.trapezoid(joint, grid)
    torch.testing.assert_close(partial, numerical, atol=1e-6, rtol=1e-6)


def test_output_only_first_step_then_context_gradient():
    torch.manual_seed(19)
    head = GaussianCopulaSplineHead(3, 2, width=8).double()
    context = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
    values = torch.randn(5, 2, dtype=torch.double)
    mask = torch.tensor([[1, 0], [0, 1], [1, 1], [1, 0], [0, 1]], dtype=torch.bool)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
    loss = -head.intrinsic_log_prob(context, values, mask).mean()
    loss.backward()
    assert torch.count_nonzero(context.grad) == 0
    assert head.net[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    context.grad = None
    (-head.intrinsic_log_prob(context, values, mask).mean()).backward()
    assert torch.isfinite(context.grad).all() and context.grad.abs().sum() > 0


def test_partial_nonlinear_density_gradcheck_and_empty_rows():
    torch.manual_seed(37)
    head = GaussianCopulaSplineHead(2, 3, width=4, bins=4).double()
    with torch.no_grad():
        head.net[-1].weight.normal_(0, 0.04)
        head.net[-1].bias[3 + 1] = 0.4
    context = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
    values = torch.tensor([[0.2, -0.3, 0.7], [float("nan")] * 3], dtype=torch.double)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    assert torch.autograd.gradcheck(
        lambda c: head.intrinsic_log_prob(c, values, mask), (context,), atol=2e-5, rtol=1e-4
    )
    assert head.intrinsic_log_prob(context, values, mask)[1] == 0
    noisy, ess = head.observed_log_prob(
        context,
        values,
        torch.full_like(values, float("nan")).masked_fill(mask, 0.0),
        mask,
        torch.zeros(8, 3, dtype=torch.double),
    )
    assert noisy[1] == 0 and ess[1] == 1


def test_invalid_masks_and_nonfinite_observations_fail_closed():
    head = GaussianCopulaSplineHead(2, 3, width=4).double()
    x = torch.zeros(1, 2, dtype=torch.double)
    y = torch.tensor([[float("nan"), 0.0, 0.0]], dtype=torch.double)
    with pytest.raises(ValueError, match="Observed values"):
        head.intrinsic_log_prob(x, y, torch.tensor([[True, False, False]]))
    with pytest.raises(ValueError, match="boolean"):
        head.intrinsic_log_prob(x, y.nan_to_num(), torch.ones_like(y))


def test_autocast_uses_at_least_float32_for_covariance_and_splines():
    head = GaussianCopulaSplineHead(2, 3, width=4)
    x = torch.zeros(2, 2, dtype=torch.bfloat16, requires_grad=True)
    y = torch.zeros(2, 3, dtype=torch.bfloat16)
    sigma = torch.full_like(y, 0.2)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    nodes = torch.zeros(8, 3, dtype=torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        score, ess = head.observed_log_prob(x, y, sigma, mask, nodes)
        draws = head.sample(x, 3)
    assert score.dtype == ess.dtype == draws.dtype == torch.float32
    assert torch.isfinite(score).all() and torch.isfinite(draws).all()
    (-score.mean()).backward()
    assert torch.isfinite(head.net[-1].weight.grad).all()


def _per_pattern_reference(head, context, value, mask):
    """The former implementation: one Cholesky of Sigma_oo per missingness pattern."""
    dtype = torch.promote_types(head._out.weight.dtype, value.dtype)
    mean, lower, transform = head._distribution_parameters(context, work_dtype=dtype)
    expanded = mask if value.ndim == 2 else mask.unsqueeze(0)
    safe = value.to(dtype).masked_fill(~expanded, 0)
    latent = transform.inv(safe)
    log_jacobian = transform.log_abs_det_jacobian(latent, safe)
    covariance = lower @ lower.transpose(-1, -2)
    result = safe[..., 0] * 0
    for pattern in torch.unique(mask, dim=0):
        rows = (mask == pattern).all(-1).nonzero().flatten()
        columns = pattern.nonzero().flatten()
        if columns.numel() == 0:
            continue
        chol = torch.linalg.cholesky(covariance[rows][:, columns][:, :, columns])
        z = latent.index_select(-2, rows).index_select(-1, columns)
        residual = z - mean[rows][:, columns]
        whitened = torch.linalg.solve_triangular(chol, residual.unsqueeze(-1), upper=False).squeeze(
            -1
        )
        normal = -0.5 * (columns.numel() * math.log(2 * math.pi) + whitened.square().sum(-1))
        normal = normal - torch.diagonal(chol, dim1=-2, dim2=-1).log().sum(-1)
        jac = log_jacobian.index_select(-2, rows).index_select(-1, columns).sum(-1)
        result = result.index_copy(-1, rows, normal - jac)
    return result


def _random_head(seed, target_dim):
    torch.manual_seed(seed)
    head = GaussianCopulaSplineHead(3, target_dim, width=16, bins=5).double()
    with torch.no_grad():
        # Nonzero correlations, nonlinear splines and context dependence.
        head._out.weight.normal_(0, 0.3)
        head._out.bias.normal_(0, 0.5)
    return head


def _ragged_mask(batch, target_dim, generator):
    mask = torch.rand(batch, target_dim, generator=generator) < 0.6
    mask[0] = True  # fully observed
    mask[1] = False  # empty row
    for row in range(2, 2 + target_dim):  # all missing but one, every coordinate
        mask[row] = False
        mask[row, row - 2] = True
    mask[2 + target_dim] = True
    mask[2 + target_dim, -1] = False  # all observed but the last
    return mask


def _value_and_grads(fn, head, context):
    out = fn()
    probe = torch.linspace(0.5, 1.5, out.numel(), dtype=out.dtype).reshape(out.shape)
    params = [context, *head.parameters()]
    grads = torch.autograd.grad((out * probe).sum(), params, allow_unused=True)
    return out.detach(), [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]


def _assert_relative(got, expected, rel=1e-10):
    # Relative to the largest reference magnitude; zeros (empty rows) stay exact.
    scale = expected.abs().max().clamp_min(1.0)
    assert (got - expected).abs().max() <= rel * scale


@pytest.mark.parametrize("target_dim", [1, 2, 4, 7])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batched_marginals_equal_per_pattern_reference(target_dim, seed):
    head = _random_head(seed, target_dim)
    generator = torch.Generator().manual_seed(100 + seed)
    batch = 3 + target_dim + 40
    context = torch.randn(batch, 3, generator=generator, dtype=torch.double, requires_grad=True)
    value = 1.5 * torch.randn(batch, target_dim, generator=generator, dtype=torch.double)
    mask = _ragged_mask(batch, target_dim, generator)
    value[~mask] = torch.nan
    got, got_grads = _value_and_grads(
        lambda: head.intrinsic_log_prob(context, value, mask), head, context
    )
    ref, ref_grads = _value_and_grads(
        lambda: _per_pattern_reference(head, context, value, mask), head, context
    )
    assert got[1] == 0 and ref[1] == 0
    _assert_relative(got, ref)
    for g, r in zip(got_grads, ref_grads, strict=True):
        _assert_relative(g, r)
    full = torch.ones_like(mask)
    complete = value.nan_to_num(0.3)
    got, got_grads = _value_and_grads(
        lambda: head.intrinsic_log_prob(context, complete, full), head, context
    )
    ref, ref_grads = _value_and_grads(
        lambda: _per_pattern_reference(head, context, complete, full), head, context
    )
    _assert_relative(got, ref)
    for g, r in zip(got_grads, ref_grads, strict=True):
        _assert_relative(g, r)


@pytest.mark.parametrize("complete", [False, True])
def test_batched_noise_integration_equals_per_pattern_reference(complete):
    target_dim = 5
    head = _random_head(7, target_dim)
    generator = torch.Generator().manual_seed(8)
    batch = 3 + target_dim + 20
    context = torch.randn(batch, 3, generator=generator, dtype=torch.double, requires_grad=True)
    value = torch.randn(batch, target_dim, generator=generator, dtype=torch.double)
    mask = torch.ones(batch, target_dim, dtype=torch.bool)
    if not complete:
        mask = _ragged_mask(batch, target_dim, generator)
    sigma = 0.3 * torch.rand(batch, target_dim, generator=generator, dtype=torch.double)
    value[~mask] = torch.nan
    sigma[~mask] = torch.nan
    nodes = torch.randn(6, target_dim, generator=generator, dtype=torch.double)

    def reference():
        perturbed = value.nan_to_num()[None] + nodes[:, None] * sigma.nan_to_num()[None]
        log_weights = _per_pattern_reference(head, context, perturbed, mask)
        return torch.logsumexp(log_weights, 0) - math.log(len(nodes))

    got, got_grads = _value_and_grads(
        lambda: head.observed_log_prob(context, value, sigma, mask, nodes)[0], head, context
    )
    ref, ref_grads = _value_and_grads(reference, head, context)
    _assert_relative(got, ref)
    for g, r in zip(got_grads, ref_grads, strict=True):
        _assert_relative(g, r)


def test_deferred_validation_keeps_error_messages():
    head = GaussianCopulaSplineHead(2, 3, width=4).double()
    x = torch.zeros(2, 2, dtype=torch.double)
    y = torch.zeros(2, 3, dtype=torch.double)
    mask = torch.tensor([[True, False, True], [True, True, True]])
    nodes = torch.zeros(4, 3, dtype=torch.double)
    sigma = torch.full_like(y, 0.1)
    with pytest.raises(ValueError, match="standard-normal nodes"):
        head.observed_log_prob(x, y, sigma, mask, nodes.fill_(torch.nan))
    nodes.zero_()
    with pytest.raises(ValueError, match="noise scales"):
        head.observed_log_prob(x, y, sigma.masked_fill(~mask, -1.0).fill_(-1.0), mask, nodes)
    # Invalid noise on a missing coordinate is ignored, as before.
    score, _ = head.observed_log_prob(x, y, sigma.masked_fill(~mask, torch.nan), mask, nodes)
    assert torch.isfinite(score).all()
    with pytest.raises(ValueError, match="Observed values"):
        head.observed_log_prob(x, y.masked_fill(mask, torch.inf), sigma, mask, nodes)
    with torch.no_grad():
        head._out.bias[0] = torch.inf
    with pytest.raises(FloatingPointError, match="base parameters"):
        head.intrinsic_log_prob(x, y, mask)
    with pytest.raises(FloatingPointError, match="base parameters"):
        head.observed_log_prob(x, y, sigma, mask, nodes)
