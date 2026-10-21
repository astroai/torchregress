from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from torchregress.algorithms import (
    IRLSBreakdownError,
    IRLSCallbacks,
    IRLSContinuationStage,
    IRLSFitConfig,
    iteratively_reweighted_fit,
    iteratively_reweighted_least_squares,
    reweight_fixed_predictions,
    robust_weights,
)


def _linear_problem(
    design: torch.Tensor, target: torch.Tensor, calls: list[torch.Tensor] | None = None
) -> IRLSCallbacks[torch.Tensor]:
    def residuals(beta: torch.Tensor) -> torch.Tensor:
        return target - design @ beta

    def weighted_fit(beta: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
        del beta
        if calls is not None:
            calls.append(precision.detach().clone())
        root = precision.sqrt()
        return torch.linalg.lstsq(design * root[:, None], target * root).solution

    return IRLSCallbacks(residuals, weighted_fit)


def test_analytic_dense_wls_and_refit_count() -> None:
    dtype = torch.float64
    x = torch.linspace(-2.0, 2.0, 31, dtype=dtype)
    design = torch.stack((torch.ones_like(x), x), dim=1)
    target = design @ torch.tensor([1.5, -0.75], dtype=dtype)
    base_precision = torch.linspace(0.5, 2.0, target.numel(), dtype=dtype)
    calls: list[torch.Tensor] = []
    problem = _linear_problem(design, target, calls)
    result = iteratively_reweighted_fit(
        problem,
        torch.zeros(2, dtype=dtype),
        IRLSFitConfig(
            max_iter=1,
            weight=lambda residuals: torch.ones_like(residuals),
            scale="none",
            base_precision=base_precision,
            objective_tol=None,
            weight_tol=None,
        ),
    )
    root = base_precision.sqrt()
    expected = torch.linalg.lstsq(design * root[:, None], target * root).solution
    torch.testing.assert_close(result.state, expected)
    torch.testing.assert_close(result.precision, base_precision)
    assert len(calls) == result.iterations == 1


def test_fixed_prediction_compatibility_name_and_deprecated_alias() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    x = torch.zeros(5, 1)
    target = torch.arange(5.0).unsqueeze(1)
    explicit = reweight_fixed_predictions(model, x, target, max_iter=1, variance_type="fixed")
    with pytest.deprecated_call(match="only reweights fixed predictions"):
        legacy = iteratively_reweighted_least_squares(
            model, x, target, max_iter=1, variance_type="fixed"
        )
    torch.testing.assert_close(legacy[0], explicit[0])
    torch.testing.assert_close(legacy[2], explicit[2])


def test_statsmodels_rlm_huber_parity() -> None:
    import statsmodels.api as sm

    dtype = torch.float64
    x = torch.linspace(-3.0, 3.0, 121, dtype=dtype)
    design = torch.stack((torch.ones_like(x), x), dim=1)
    target = 0.7 + 1.8 * x + 0.15 * torch.sin(4.0 * x)
    target = target.clone()
    target[::17] += torch.tensor([7.0, -6.0, 8.0, -7.0, 6.5, -8.0, 7.5, -6.5])
    initial = torch.linalg.lstsq(design, target).solution
    result = iteratively_reweighted_fit(
        _linear_problem(design, target),
        initial,
        IRLSFitConfig(
            max_iter=100,
            weight="huber",
            weight_params={"delta": 1.345},
            scale="mad",
            objective_tol=1e-12,
            weight_tol=1e-10,
        ),
    )
    reference = sm.RLM(target.numpy(), design.numpy(), M=sm.robust.norms.HuberT(t=1.345)).fit(
        start_params=initial.numpy(), scale_est="mad", conv="weights", tol=1e-10, maxiter=100
    )
    torch.testing.assert_close(
        result.state, torch.from_numpy(reference.params), rtol=2e-5, atol=2e-6
    )
    assert result.converged


def test_nonlinear_callback_refits_after_each_update() -> None:
    dtype = torch.float64
    x = torch.linspace(-1.0, 1.0, 80, dtype=dtype)
    target = 1.7 * torch.exp(0.6 * x)
    target = target.clone()
    target[::13] += 5.0
    calls = 0

    def residuals(theta: torch.Tensor) -> torch.Tensor:
        return target - theta[0] * torch.exp(theta[1] * x)

    def weighted_fit(theta: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        fitted = theta.clone()
        root = precision.sqrt()
        for _ in range(8):
            exponential = torch.exp(fitted[1] * x)
            prediction = fitted[0] * exponential
            jacobian = torch.stack((exponential, fitted[0] * x * exponential), dim=1)
            step = torch.linalg.lstsq(
                jacobian * root[:, None], (target - prediction) * root
            ).solution
            fitted = fitted + step
        return fitted

    result = iteratively_reweighted_fit(
        IRLSCallbacks(residuals, weighted_fit),
        torch.tensor([1.0, 0.0], dtype=dtype),
        IRLSFitConfig(max_iter=40, weight="cauchy", scale="mad", weight_tol=1e-9),
    )
    torch.testing.assert_close(
        result.state, torch.tensor([1.7, 0.6], dtype=dtype), atol=2e-3, rtol=2e-3
    )
    assert calls == result.iterations
    assert calls > 1


def test_full_covariance_block_mahalanobis_and_gls_refit() -> None:
    dtype = torch.float64
    generator = torch.Generator().manual_seed(9)
    n = 60
    truth = torch.tensor([1.0, -2.0], dtype=dtype)
    index = torch.arange(n, dtype=dtype)
    sigma1 = 0.15 + 0.005 * index
    sigma2 = 0.25 + 0.003 * index
    rho = 0.4
    covariance = torch.zeros(n, 2, 2, dtype=dtype)
    covariance[:, 0, 0] = sigma1.square()
    covariance[:, 1, 1] = sigma2.square()
    covariance[:, 0, 1] = covariance[:, 1, 0] = rho * sigma1 * sigma2
    noise = torch.linalg.cholesky(covariance) @ torch.randn(
        n, 2, 1, dtype=dtype, generator=generator
    )
    observations = truth + noise.squeeze(-1)
    observations[0] += torch.tensor([12.0, -10.0], dtype=dtype)

    def residuals(mean: torch.Tensor) -> torch.Tensor:
        return observations - mean

    def weighted_fit(mean: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
        del mean
        normal = precision.sum(dim=0)
        rhs = torch.einsum("nij,nj->i", precision, observations)
        return torch.linalg.solve(normal, rhs)

    problem = IRLSCallbacks(
        residuals,
        weighted_fit,
        covariance_fn=lambda state: covariance,
    )
    initial = torch.zeros(2, dtype=dtype)
    one_step = iteratively_reweighted_fit(
        problem,
        initial,
        IRLSFitConfig(
            max_iter=1,
            weight="huber",
            scale="none",
            objective_tol=None,
            weight_tol=None,
        ),
    )
    whitened = torch.linalg.solve_triangular(
        torch.linalg.cholesky(covariance), observations.unsqueeze(-1), upper=False
    ).squeeze(-1)
    expected_weights = robust_weights(torch.linalg.vector_norm(whitened, dim=1), "huber")
    torch.testing.assert_close(one_step.weights, expected_weights)
    assert one_step.precision.shape == covariance.shape
    result = iteratively_reweighted_fit(
        problem,
        initial,
        IRLSFitConfig(max_iter=50, weight="huber", scale="none", weight_tol=1e-9),
    )
    torch.testing.assert_close(result.state, truth, atol=0.12, rtol=0.08)
    assert result.weights[0] < 0.05


def test_mask_base_precision_and_effective_sample_size() -> None:
    values = torch.tensor([1.0, 2.0, 100.0, 4.0], dtype=torch.float64)
    base = torch.tensor([1.0, 2.0, 8.0, 3.0], dtype=torch.float64)
    mask = torch.tensor([True, True, False, True])

    def residuals(mean: torch.Tensor) -> torch.Tensor:
        return values - mean

    def weighted_fit(mean: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
        del mean
        return (values * precision).sum() / precision.sum()

    result = iteratively_reweighted_fit(
        IRLSCallbacks(residuals, weighted_fit),
        torch.tensor(0.0, dtype=torch.float64),
        IRLSFitConfig(
            max_iter=1,
            weight=lambda residual: torch.ones_like(residual),
            scale="none",
            mask=mask,
            base_precision=base,
            objective_tol=None,
            weight_tol=None,
        ),
    )
    torch.testing.assert_close(result.state, torch.tensor(17.0 / 6.0, dtype=torch.float64))
    torch.testing.assert_close(
        result.precision, torch.tensor([1.0, 2.0, 0.0, 3.0], dtype=torch.float64)
    )
    assert result.effective_sample_size == pytest.approx(36.0 / 14.0)


def test_element_mask_and_block_base_precision() -> None:
    residual_matrix = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=torch.float64)
    element_mask = torch.tensor([[True, False], [True, True], [False, True]])
    seen_precision: list[torch.Tensor] = []

    def diagonal_fit(state: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
        seen_precision.append(precision)
        return state

    iteratively_reweighted_fit(
        IRLSCallbacks(lambda state: residual_matrix, diagonal_fit),
        torch.zeros(2, dtype=torch.float64),
        IRLSFitConfig(
            max_iter=1,
            weight=lambda residual: torch.ones_like(residual),
            scale="none",
            mask=element_mask,
            objective_tol=None,
            weight_tol=None,
        ),
    )
    torch.testing.assert_close(seen_precision[0], element_mask.to(torch.float64))

    block_precision = torch.tensor(
        [
            [[2.0, -0.5], [-0.5, 1.0]],
            [[1.5, -0.2], [-0.2, 2.0]],
            [[3.0, -0.7], [-0.7, 1.5]],
        ],
        dtype=torch.float64,
    )
    block = iteratively_reweighted_fit(
        IRLSCallbacks(lambda state: residual_matrix, lambda state, precision: state),
        torch.zeros(2, dtype=torch.float64),
        IRLSFitConfig(
            max_iter=1,
            weight=lambda residual: torch.ones_like(residual),
            scale="none",
            base_precision=block_precision,
            objective_tol=None,
            weight_tol=None,
        ),
    )
    torch.testing.assert_close(block.precision, block_precision)


def test_continuation_schedule_runs_each_stage() -> None:
    values = torch.tensor([0.0, 1.0, 2.0, 20.0], dtype=torch.float64)
    calls = 0

    def residuals(mean: torch.Tensor) -> torch.Tensor:
        return values - mean

    def weighted_fit(mean: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        del mean
        calls += 1
        return (values * precision).sum() / precision.sum()

    config = IRLSFitConfig(
        continuation=(
            IRLSContinuationStage(2, "huber", {"delta": 2.0}),
            IRLSContinuationStage(3, "tukey", {"c": 4.685}),
        ),
        scale="mad",
        objective_tol=None,
        weight_tol=None,
    )
    result = iteratively_reweighted_fit(
        IRLSCallbacks(residuals, weighted_fit), torch.tensor(0.0, dtype=torch.float64), config
    )
    assert calls == result.iterations == 5
    assert [item.stage for item in result.history] == [0, 0, 1, 1, 1]
    assert [item.weight for item in result.history] == ["huber", "huber", "tukey", "tukey", "tukey"]


def test_empty_mask_and_rank_breakdown_are_diagnostic() -> None:
    fit_called = False

    def empty_fit(state: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
        nonlocal fit_called
        fit_called = True
        return state

    empty = iteratively_reweighted_fit(
        IRLSCallbacks(lambda state: torch.empty(0), empty_fit),
        torch.tensor(0.0),
    )
    assert empty.termination == "empty"
    assert empty.history == ()
    assert not fit_called

    def rank_deficient(state: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
        raise IRLSBreakdownError("weighted design is rank deficient")

    broken = iteratively_reweighted_fit(
        IRLSCallbacks(lambda state: torch.ones(4), rank_deficient),
        torch.tensor(0.0),
        IRLSFitConfig(scale="none"),
    )
    assert broken.termination == "breakdown"
    assert "rank deficient" in broken.message


def test_nonfinite_and_zero_weight_breakdown_diagnostics() -> None:
    initial_nonfinite = iteratively_reweighted_fit(
        IRLSCallbacks(lambda state: torch.tensor([float("nan")]), lambda state, precision: state),
        torch.tensor(0.0),
        IRLSFitConfig(scale="none"),
    )
    assert initial_nonfinite.termination == "non_finite"
    assert initial_nonfinite.iterations == 0

    nonfinite = iteratively_reweighted_fit(
        IRLSCallbacks(
            lambda state: torch.ones(5) - state,
            lambda state, precision: torch.tensor(float("nan")),
        ),
        torch.tensor(0.0),
        IRLSFitConfig(scale="none"),
    )
    assert nonfinite.termination == "non_finite"
    assert not nonfinite.history[-1].finite

    zero_weight = iteratively_reweighted_fit(
        IRLSCallbacks(lambda state: torch.full((5,), 10.0), lambda state, precision: state),
        torch.tensor(0.0),
        IRLSFitConfig(weight="tukey", weight_params={"c": 1.0}, scale="none"),
    )
    assert zero_weight.termination == "breakdown"
    assert zero_weight.effective_sample_size == 0.0

    nonfinite_step = iteratively_reweighted_fit(
        IRLSCallbacks(
            lambda state: torch.arange(4.0) - state,
            lambda state, precision: state,
            state_step_fn=lambda previous, current: float("nan"),
        ),
        torch.tensor(0.0),
        IRLSFitConfig(max_iter=1, scale="mad"),
    )
    assert nonfinite_step.termination == "non_finite"
    assert not nonfinite_step.history[-1].parameter_step_finite


def test_custom_whitening_and_parameter_step_convergence() -> None:
    values = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0], dtype=torch.float64)
    problem = IRLSCallbacks(
        residuals_fn=lambda state: values - state,
        weighted_fit_fn=lambda state, precision: (values * precision).sum() / precision.sum(),
        whiten_fn=lambda state, residuals: residuals / 2.0,
        state_step_fn=lambda previous, current: (current - previous).abs(),
    )
    result = iteratively_reweighted_fit(
        problem,
        torch.tensor(0.0, dtype=torch.float64),
        IRLSFitConfig(
            weight="pseudo_huber",
            scale="none",
            objective_tol=1e-12,
            weight_tol=1e-12,
            parameter_tol=1e-12,
        ),
    )
    torch.testing.assert_close(result.weights, robust_weights(values.abs() / 2.0, "pseudo_huber"))
    assert result.converged
    assert result.history[-1].parameter_step == pytest.approx(0.0)


@pytest.mark.parametrize("name", ["huber", "pseudo_huber", "tukey", "cauchy", "barron"])
def test_robust_weights_compile(name: str) -> None:
    values = torch.linspace(0.0, 5.0, 21)
    params = {"alpha": 0.0} if name == "barron" else {}

    def kernel(tensor: torch.Tensor) -> torch.Tensor:
        return robust_weights(tensor, name, params)

    compiled = torch.compile(kernel, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(values), kernel(values))


def _device_cases() -> list[tuple[str, torch.dtype]]:
    cases = [("cpu", torch.float32), ("cpu", torch.float64)]
    if torch.cuda.is_available():
        cases.extend([("cuda", torch.float32), ("cuda", torch.float64)])
    if torch.backends.mps.is_available():
        cases.append(("mps", torch.float32))
    return cases


@pytest.mark.parametrize(("device", "dtype"), _device_cases())
def test_device_dtype_and_determinism(device: str, dtype: torch.dtype) -> None:
    values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 20.0], device=device, dtype=dtype)

    def run():
        return iteratively_reweighted_fit(
            IRLSCallbacks(
                lambda state: values - state,
                lambda state, precision: (values * precision).sum() / precision.sum(),
            ),
            torch.tensor(0.0, device=device, dtype=dtype),
            IRLSFitConfig(max_iter=20, weight="cauchy", scale="mad"),
        )

    first = run()
    second = run()
    assert first.state.device.type == device
    assert first.state.dtype == dtype
    assert first.weights.device.type == device
    assert first.weights.dtype == dtype
    torch.testing.assert_close(first.state, second.state, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.weights, second.weights, rtol=0.0, atol=0.0)
    assert [(item.objective, item.weight_change) for item in first.history] == [
        (item.objective, item.weight_change) for item in second.history
    ]


def test_result_and_history_are_immutable() -> None:
    result = iteratively_reweighted_fit(
        IRLSCallbacks(lambda state: torch.arange(4.0) - state, lambda state, precision: state),
        torch.tensor(0.0),
        IRLSFitConfig(max_iter=2, scale="mad"),
    )
    assert isinstance(result.history, tuple)
    with pytest.raises(FrozenInstanceError):
        result.converged = True
    with pytest.raises(FrozenInstanceError):
        result.history[0].objective = 0.0


@pytest.mark.parametrize("factor", [1e-200, 1e-20, 1.0, 1e200])
def test_precision_rescaling_preserves_ess_and_objective(factor: float) -> None:
    residuals = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    base = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    problem = IRLSCallbacks(lambda state: residuals, lambda state, precision: state)

    def run(multiplier: float):
        return iteratively_reweighted_fit(
            problem,
            torch.tensor(0.0, dtype=torch.float64),
            IRLSFitConfig(max_iter=1, scale="none", base_precision=base * multiplier),
        )

    expected = run(1.0)
    result = run(factor)
    assert result.termination == expected.termination == "max_iterations"
    assert result.iterations == 1
    assert result.effective_sample_size == pytest.approx(expected.effective_sample_size)
    assert result.objective == pytest.approx(expected.objective)
    torch.testing.assert_close(result.precision / factor, expected.precision)


@pytest.mark.parametrize("missing", [10.0, float("nan"), float("inf")])
@pytest.mark.parametrize("scale", ["mad", "std", "rms", "none"])
def test_fully_masked_output_does_not_break_observed_fit(missing: float, scale: str) -> None:
    residuals = torch.tensor([[1.0, missing], [2.0, missing], [3.0, missing]])
    mask = torch.tensor([[True, False]]).expand_as(residuals)
    seen: list[torch.Tensor] = []

    def fit(state: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
        seen.append(precision)
        return state

    result = iteratively_reweighted_fit(
        IRLSCallbacks(lambda state: residuals, fit),
        torch.tensor(0.0),
        IRLSFitConfig(max_iter=1, scale=scale, mask=mask),
    )
    observed = iteratively_reweighted_fit(
        IRLSCallbacks(lambda state: residuals[:, 0], lambda state, precision: state),
        torch.tensor(0.0),
        IRLSFitConfig(max_iter=1, scale=scale),
    )
    assert result.termination == "max_iterations"
    assert result.history[0].residuals_finite
    assert len(seen) == 1
    torch.testing.assert_close(result.weights, observed.weights)
    torch.testing.assert_close(result.precision[:, 0], observed.precision)
    assert not result.precision[:, 1].any()
