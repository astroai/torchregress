from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from torchregress.algorithms import (
    BatchedIRLSFitConfig,
    BatchedIRLSProblem,
    BatchedIRLSStep,
    IRLSCallbacks,
    IRLSFitConfig,
    IRLSRowStatus,
    batched_iteratively_reweighted_fit,
    iteratively_reweighted_fit,
    robust_weights,
    solve_normal_equations,
)
from torchregress.losses import effective_precision, student_t_nll

DTYPE = torch.float64


def _take(tensor: torch.Tensor, rows: torch.Tensor | None) -> torch.Tensor:
    return tensor if rows is None else tensor[rows]


def _data(
    batch: int = 6, pixels: int = 120, n: int = 3, seed: int = 0, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.linspace(-1.0, 1.0, pixels, dtype=DTYPE)
    design = torch.stack([x**k for k in range(n)], dim=-1).expand(batch, pixels, n).clone()
    design = design * (1.0 + 0.1 * torch.rand(batch, 1, n, generator=generator, dtype=DTYPE))
    truth = torch.randn(batch, n, generator=generator, dtype=DTYPE) + 2.0
    precision = 50.0 + 50.0 * torch.rand(batch, pixels, generator=generator, dtype=DTYPE)
    noise = torch.randn(batch, pixels, generator=generator, dtype=DTYPE) * precision.rsqrt()
    target = (design @ truth.unsqueeze(-1)).squeeze(-1) + noise
    outliers = torch.rand(batch, pixels, generator=generator) < 0.05
    target = torch.where(outliers, target + 3.0, target)
    to = {"device": device}
    return design.to(**to), target.to(**to), precision.to(**to), truth.to(**to)


def _linear_problem(
    design: torch.Tensor,
    *,
    nonnegative: torch.Tensor | None = None,
    prior: torch.Tensor | None = None,
    calls: list[torch.Tensor | None] | None = None,
) -> BatchedIRLSProblem:
    def predict(theta: torch.Tensor, rows: torch.Tensor | None) -> torch.Tensor:
        return (_take(design, rows) @ theta.unsqueeze(-1)).squeeze(-1)

    def solve(
        theta: torch.Tensor, weights: torch.Tensor, target: torch.Tensor, rows: torch.Tensor | None
    ) -> BatchedIRLSStep:
        if calls is not None:
            calls.append(None if rows is None else rows.clone())
        x = _take(design, rows)
        xw = x * weights.unsqueeze(-1)
        normal = x.transpose(1, 2) @ xw
        if prior is not None:
            normal = normal + torch.diag_embed(prior).to(device=prior.device, dtype=prior.dtype)
        rhs = (xw * target.unsqueeze(-1)).sum(1)
        out = solve_normal_equations(normal, rhs, nonnegative=nonnegative, start=theta)
        return BatchedIRLSStep(
            out.solution, out.standard_errors, failed=out.singular, ridge_used=out.ridge_used
        )

    penalty = None
    if prior is not None:

        def penalty(theta: torch.Tensor, rows: torch.Tensor | None) -> torch.Tensor:
            return 0.5 * (prior * theta.square()).sum(-1)

    return BatchedIRLSProblem(predict, solve, penalty=penalty, exact_solve=True)


def _fit(design, target, precision, mask=None, config=None, **kwargs):  # type: ignore[no-untyped-def]
    theta0 = torch.zeros(
        design.shape[0], design.shape[-1], dtype=design.dtype, device=design.device
    )
    return batched_iteratively_reweighted_fit(
        _linear_problem(design, **kwargs), theta0, target, precision, mask, config
    )


def test_student_t_fit_is_stationary_point_of_exact_nll() -> None:
    design, target, precision, truth = _data()
    nu = 4.0
    config = BatchedIRLSFitConfig(weight_params={"nu": nu}, step_tol=1e-8, max_iter=200)
    result = _fit(design, target, precision, config=config)
    assert result.converged.all()
    assert (result.iterations >= 2).all()
    theta = result.theta.clone().requires_grad_(True)
    residual = target - (design @ theta.unsqueeze(-1)).squeeze(-1)
    nll = student_t_nll(residual, precision, nu)
    (grad,) = torch.autograd.grad(nll.sum(), theta)
    se = result.standard_errors
    assert (grad.abs() * se).max() < 1e-6  # gradient in standard-error units
    torch.testing.assert_close(result.objective, nll.detach(), rtol=1e-12, atol=1e-12)
    expected_w = precision * (nu + 1.0) / (nu + precision * residual.detach().square())
    torch.testing.assert_close(result.weights, expected_w, rtol=1e-12, atol=0.0)
    # Robust: the 5% +3 outliers barely move the fit (they would bias least squares).
    assert ((result.theta - truth).abs() / se).max() < 5.0


def test_batched_rows_equal_rows_fitted_alone() -> None:
    design, target, precision, _ = _data(batch=8, seed=1)
    mask = torch.rand(target.shape, generator=torch.Generator().manual_seed(2)) > 0.1
    config = BatchedIRLSFitConfig(weight_params={"nu": 3.0}, eta=0.02)
    batched = _fit(design, target, precision, mask, config)
    for row in range(design.shape[0]):
        sl = slice(row, row + 1)
        alone = _fit(design[sl], target[sl], precision[sl], mask[sl], config)
        torch.testing.assert_close(alone.theta[0], batched.theta[row], rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(alone.objective[0], batched.objective[row], rtol=1e-12, atol=0)
        assert alone.status[0] == batched.status[row]
        assert alone.iterations[0] == batched.iterations[row]


def test_finished_rows_leave_the_active_set() -> None:
    design, target, precision, _ = _data(batch=10, seed=3)
    target[0] = (design[0] @ torch.ones(3, dtype=DTYPE)).clone()  # noiseless: converges first
    calls: list[torch.Tensor | None] = []
    result = _fit(design, target, precision, calls=calls)
    assert calls[0] is None  # the whole batch, no gather
    sizes = [design.shape[0] if c is None else c.numel() for c in calls]
    assert sizes == sorted(sizes, reverse=True) and sizes[-1] < sizes[0]
    assert int(result.iterations.sum()) == sum(sizes)  # cost is sum of per-row iterations


def test_gaussian_exact_solve_takes_one_iteration_and_matches_wls() -> None:
    design, target, precision, _ = _data(seed=4)
    config = BatchedIRLSFitConfig(weight_params={"nu": math.inf})
    result = _fit(design, target, precision, config=config)
    assert result.converged.all() and (result.iterations == 1).all()
    root = precision.sqrt()
    expected = torch.linalg.lstsq(design * root.unsqueeze(-1), (target * root).unsqueeze(-1))
    torch.testing.assert_close(result.theta, expected.solution.squeeze(-1), rtol=1e-9, atol=1e-11)
    residual = target - (design @ result.theta.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(result.objective, student_t_nll(residual, precision, math.inf))
    torch.testing.assert_close(result.weights, precision)


def test_masked_empty_and_non_finite_rows_fail_closed() -> None:
    design, target, precision, _ = _data(batch=5, seed=5)
    mask = torch.ones_like(target, dtype=torch.bool)
    clean = _fit(design, target, precision, mask)
    # Row 1: masked entries hold garbage -> identical to the clean row.
    mask[1, :30] = False
    dirty_target = target.clone()
    dirty_target[1, :30] = math.nan
    dirty_precision = precision.clone()
    dirty_precision[1, :30] = -5.0
    # Row 2: everything masked -> EMPTY. Row 3: NaN on an observed entry -> FAILED.
    mask[2] = False
    dirty_target[3, 7] = math.nan
    # Row 4: only n observations with positive precision -> FAILED (too few).
    dirty_precision[4, 3:] = 0.0
    result = _fit(design, dirty_target, dirty_precision, mask)
    reference = _fit(design[1:2, 30:], target[1:2, 30:], precision[1:2, 30:])
    torch.testing.assert_close(result.theta[1], reference.theta[0], rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(result.objective[1], reference.objective[0], rtol=1e-12, atol=0)
    assert torch.equal(result.weights[1, :30], torch.zeros(30, dtype=DTYPE))
    expected_status = [
        IRLSRowStatus.CONVERGED,
        IRLSRowStatus.CONVERGED,
        IRLSRowStatus.EMPTY,
        IRLSRowStatus.FAILED,
        IRLSRowStatus.FAILED,
    ]
    assert result.status.tolist() == [int(s) for s in expected_status]
    assert result.n_observations.tolist() == [120, 90, 0, 119, 3]
    for row in (2, 3, 4):
        for values in (result.theta, result.model, result.weights, result.objective):
            assert torch.isnan(values[row]).all()
        assert result.iterations[row] == 0
    torch.testing.assert_close(result.theta[0], clean.theta[0])


def test_singular_solve_and_non_finite_start_fail_closed() -> None:
    design, target, precision, _ = _data(batch=3, seed=6)
    design = design.clone()
    design[1, :, 2] = 0.0
    design[1, :, 1] = 0.0
    design[1, :, 0] = 0.0  # all-empty design: no information at all

    def solve(theta, weights, target_, rows):  # type: ignore[no-untyped-def]
        step = _linear_problem(design).solve(theta, weights, target_, rows)
        failed = torch.zeros(theta.shape[0], dtype=torch.bool)
        failed[_take(torch.arange(3), rows) == 0] = True  # the caller flags row 0 as singular
        return replace(step, failed=failed)

    problem = replace(_linear_problem(design), solve=solve)
    theta0 = torch.zeros(3, 3, dtype=DTYPE)
    theta0[2, 0] = math.inf
    result = batched_iteratively_reweighted_fit(problem, theta0, target, precision)
    assert result.status[0] == IRLSRowStatus.FAILED
    assert result.status[2] == IRLSRowStatus.FAILED
    assert torch.isnan(result.theta[0]).all() and torch.isnan(result.theta[2]).all()
    # Row 1 has zero information: every coefficient is pinned at 0 with infinite error.
    assert torch.equal(result.theta[1], torch.zeros(3, dtype=DTYPE))
    assert torch.isinf(result.standard_errors[1]).all()


def _overshooting_problem(
    design: torch.Tensor, *, factor: float, damped_fraction: float | None, predicted: bool = False
) -> BatchedIRLSProblem:
    """A linear problem whose 'full' step overshoots the exact solution by ``factor``."""
    linear = _linear_problem(design)

    def exact(theta, weights, target, rows):  # type: ignore[no-untyped-def]
        return linear.solve(theta, weights, target, rows)

    def solve(theta, weights, target, rows):  # type: ignore[no-untyped-def]
        step = exact(theta, weights, target, rows)
        full = theta + factor * (step.theta - theta)
        q0 = (weights * (target - linear.predict(theta, rows)).square()).sum(-1)
        return BatchedIRLSStep(
            full, step.standard_errors, predicted_objective=q0 if predicted else None
        )

    def damped(theta, weights, target, rows, damping):  # type: ignore[no-untyped-def]
        step = exact(theta, weights, target, rows)
        if damped_fraction is None:  # every damped step is worse too
            return theta + factor * (step.theta - theta)
        return theta + damped_fraction * (step.theta - theta)

    return replace(linear, solve=solve, damped_solve=damped, exact_solve=False)


def test_damped_steps_never_count_as_converged() -> None:
    design, target, precision, _ = _data(batch=4, seed=7)
    theta0 = torch.zeros(4, 3, dtype=DTYPE)
    config = BatchedIRLSFitConfig(weight_params={"nu": math.inf}, max_iter=6)
    # The full step overshoots 3x (raises the objective); the damped step moves only
    # 1e-9 of the way, far below tol in standard errors, yet lowers the objective.
    problem = _overshooting_problem(design, factor=3.0, damped_fraction=1e-9)
    result = batched_iteratively_reweighted_fit(problem, theta0, target, precision, None, config)
    assert (result.status == IRLSRowStatus.MAX_ITERATIONS).all()
    assert (result.iterations == 6).all()
    # The same problem with an honest full step converges at the second iteration.
    honest = _overshooting_problem(design, factor=1.0, damped_fraction=1e-9)
    ok = batched_iteratively_reweighted_fit(honest, theta0, target, precision, None, config)
    assert ok.converged.all() and (ok.iterations == 2).all()


def test_rows_without_any_decrease_stall_unless_at_round_off() -> None:
    design, target, precision, _ = _data(batch=3, seed=8)
    exact = _fit(
        design, target, precision, config=BatchedIRLSFitConfig(weight_params={"nu": math.inf})
    )
    config = BatchedIRLSFitConfig(weight_params={"nu": math.inf}, max_iter=5, max_damping=3)
    # Start next to the optimum; the full step overshoots by 1e8 so it is far above tol.
    problem = _overshooting_problem(design, factor=-1e8, damped_fraction=None)
    start = exact.theta + 1e-9 * exact.standard_errors
    stalled = batched_iteratively_reweighted_fit(problem, start, target, precision, None, config)
    assert (stalled.status == IRLSRowStatus.STALLED).all() and (stalled.iterations == 1).all()
    torch.testing.assert_close(stalled.theta, start)  # rejected steps are not taken
    resolved_problem = _overshooting_problem(
        design, factor=-1e8, damped_fraction=None, predicted=True
    )
    resolved = batched_iteratively_reweighted_fit(
        resolved_problem, start, target, precision, None, config
    )
    assert resolved.converged.all()


def test_lm_damping_recovers_from_overshoot() -> None:
    design, target, precision, _ = _data(batch=4, seed=9)
    linear = _linear_problem(design)

    def solve(theta, weights, target_, rows):  # type: ignore[no-untyped-def]
        step = linear.solve(theta, weights, target_, rows)
        return replace(step, theta=theta + 2.5 * (step.theta - theta))  # always overshoots

    def damped(theta, weights, target_, rows, damping):  # type: ignore[no-untyped-def]
        step = linear.solve(theta, weights, target_, rows)
        fraction = (2.5 / (1.0 + 10.0 * damping)).unsqueeze(-1)
        return theta + fraction * (step.theta - theta)

    problem = replace(linear, solve=solve, damped_solve=damped, exact_solve=False)
    config = BatchedIRLSFitConfig(weight_params={"nu": 5.0}, max_iter=200, damping_start=0.1)
    result = batched_iteratively_reweighted_fit(
        problem, torch.zeros(4, 3, dtype=DTYPE), target, precision, None, config
    )
    # The overshooting full step is rejected until it falls below tol (then it is an
    # honest certificate: it overstates the remaining distance); damping does the work.
    assert result.converged.all()
    reference = _fit(design, target, precision, config=config)
    assert ((result.theta - reference.theta).abs() / reference.standard_errors).max() < 1e-2


def test_eta_variance_model_and_fixed_point() -> None:
    design, target, precision, _ = _data(seed=10)
    eta, nu = 0.05, 5.0
    config = BatchedIRLSFitConfig(weight_params={"nu": nu}, eta=eta, step_tol=1e-9, max_iter=300)
    result = _fit(design, target, precision, config=config)
    assert result.converged.all()
    model = (design @ result.theta.unsqueeze(-1)).squeeze(-1)
    prec_eff = effective_precision(precision, model, eta)
    torch.testing.assert_close(result.effective_precision, prec_eff, rtol=1e-12, atol=0)
    residual = target - model
    weights = prec_eff * (nu + 1.0) / (nu + prec_eff * residual.square())
    torch.testing.assert_close(result.weights, weights, rtol=1e-12, atol=0)
    # Fixed point: the weighted normal equations hold at the final weights.
    score = (design * (weights * residual).unsqueeze(-1)).sum(1)
    assert (score.abs() * result.standard_errors).max() < 1e-6
    torch.testing.assert_close(result.objective, student_t_nll(residual, prec_eff, nu))


def test_penalty_enters_objective_and_shrinks() -> None:
    design, target, precision, _ = _data(seed=11)
    prior = torch.tensor([0.0, 1e4, 1e4], dtype=DTYPE)
    plain = _fit(design, target, precision)
    shrunk = _fit(design, target, precision, prior=prior)
    assert (shrunk.theta[:, 1:].abs() < plain.theta[:, 1:].abs()).all()
    torch.testing.assert_close(shrunk.penalty, 0.5 * (prior * shrunk.theta.square()).sum(-1))
    model = (design @ shrunk.theta.unsqueeze(-1)).squeeze(-1)
    nll = student_t_nll(target - model, precision, 5.0)
    torch.testing.assert_close(shrunk.objective, nll + shrunk.penalty)
    assert torch.equal(plain.penalty, torch.zeros_like(plain.penalty))


def test_nonnegative_solves_through_the_controller() -> None:
    design, target, precision, _ = _data(seed=12)
    target = target - 3.0 * design[..., 2]  # pushes the last coefficient negative
    bounds = torch.tensor([False, False, True])
    gauss = BatchedIRLSFitConfig(weight_params={"nu": math.inf})
    result = _fit(design, target, precision, config=gauss, nonnegative=bounds)
    assert result.converged.all() and (result.iterations == 1).all()
    assert torch.equal(result.theta[:, 2], torch.zeros(6, dtype=DTYPE))
    # KKT: the Gaussian score is zero on the free coefficients, non-positive at the bound.
    residual = target - result.model
    score = (design * (precision * residual).unsqueeze(-1)).sum(1)
    assert (score[:, :2].abs() * result.standard_errors[:, :2]).max() < 1e-8
    assert (score[:, 2] < 0).all()
    robust = _fit(
        design, target, precision, config=BatchedIRLSFitConfig(max_iter=300), nonnegative=bounds
    )
    assert robust.converged.all() and (robust.theta[:, 2] >= 0).all()


def test_huber_family_objective_and_weights() -> None:
    design, target, precision, _ = _data(seed=13)
    config = BatchedIRLSFitConfig(weight="huber", weight_params={"delta": 1.345})
    result = _fit(design, target, precision, config=config)
    assert result.converged.all()
    model = (design @ result.theta.unsqueeze(-1)).squeeze(-1)
    u = (target - model) * precision.sqrt()
    weights = precision * robust_weights(u, "huber", {"delta": 1.345})
    torch.testing.assert_close(result.weights, weights, rtol=1e-12, atol=0)
    a = u.abs()
    rho = torch.where(a <= 1.345, 0.5 * a.square(), 1.345 * (a - 0.5 * 1.345))
    torch.testing.assert_close(result.objective, rho.sum(-1))


def test_warm_start_weights_reach_the_same_fixed_point() -> None:
    design, target, precision, _ = _data(seed=14)
    config = BatchedIRLSFitConfig(weight_params={"nu": 4.0}, step_tol=1e-6)
    cold = _fit(design, target, precision, config=config)
    initial = cold.weights.clone()
    initial[0, 0] = -1.0  # invalid row: falls back to the precision
    warm = batched_iteratively_reweighted_fit(
        _linear_problem(design),
        torch.zeros_like(cold.theta),
        target,
        precision,
        None,
        config,
        initial_weights=initial,
    )
    scaled = (warm.theta - cold.theta).abs() / cold.standard_errors
    assert scaled.max() < 1e-5
    assert (warm.iterations[1:] <= cold.iterations[1:]).all()
    assert warm.iterations[0] == cold.iterations[0]


def test_matches_existing_controller_row_by_row() -> None:
    design, target, precision, _ = _data(batch=3, seed=15)
    nu = 4.0
    config = BatchedIRLSFitConfig(weight_params={"nu": nu}, step_tol=1e-10, max_iter=300)
    batched = _fit(design, target, precision, config=config)
    for row in range(3):
        x, y, lam = design[row], target[row], precision[row]

        def residuals(beta, x=x, y=y, lam=lam):  # type: ignore[no-untyped-def]
            return (y - x @ beta) * lam.sqrt()

        def weighted_fit(beta, prec, x=x, y=y):  # type: ignore[no-untyped-def]
            del beta
            return torch.linalg.solve(x.T @ (x * prec[:, None]), x.T @ (prec * y))

        single = iteratively_reweighted_fit(
            IRLSCallbacks(residuals, weighted_fit),
            torch.zeros(3, dtype=DTYPE),
            IRLSFitConfig(
                max_iter=300,
                weight="student_t",
                weight_params={"nu": nu},
                scale="none",
                base_precision=lam,
                objective_tol=1e-15,
                weight_tol=None,
            ),
        )
        scaled = (single.state - batched.theta[row]).abs() / batched.standard_errors[row]
        assert scaled.max() < 1e-6


def test_config_and_input_validation() -> None:
    design, target, precision, _ = _data(batch=2)
    with pytest.raises(ValueError, match="step_tol"):
        BatchedIRLSFitConfig(step_tol=0.0)
    with pytest.raises(ValueError, match="nu"):
        BatchedIRLSFitConfig(weight_params={"nu": -1.0})
    with pytest.raises(ValueError, match="eta"):
        BatchedIRLSFitConfig(eta=-0.1)
    theta0 = torch.zeros(2, 3, dtype=DTYPE)
    problem = _linear_problem(design)
    with pytest.raises(ValueError, match="precision shape"):
        batched_iteratively_reweighted_fit(problem, theta0, target, precision[:, :5])
    with pytest.raises(TypeError, match="mask"):
        batched_iteratively_reweighted_fit(
            problem, theta0, target, precision, torch.ones_like(target)
        )
    with pytest.raises(ValueError, match="initial_theta"):
        batched_iteratively_reweighted_fit(problem, theta0[:1], target, precision)

    def bad_solve(theta, weights, target_, rows):  # type: ignore[no-untyped-def]
        return BatchedIRLSStep(theta[:, :2], theta[:, :2])

    with pytest.raises(ValueError, match="solve returned"):
        batched_iteratively_reweighted_fit(
            replace(problem, solve=bad_solve), theta0, target, precision
        )


def test_float32_runs() -> None:
    design, target, precision, _ = _data(seed=16)
    result = _fit(design.float(), target.float(), precision.float())
    assert result.theta.dtype == torch.float32
    assert result.converged.all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu() -> None:
    design, target, precision, _ = _data(seed=17)
    cpu = _fit(design, target, precision)
    gpu = _fit(design.cuda(), target.cuda(), precision.cuda())
    torch.testing.assert_close(gpu.theta.cpu(), cpu.theta, rtol=1e-9, atol=1e-11)
    assert torch.equal(gpu.status.cpu(), cpu.status)


def test_result_evaluated_in_row_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    import torchregress.algorithms.irls_batched as module

    design, target, precision, _ = _data(batch=7, seed=18)
    config = BatchedIRLSFitConfig(weight_params={"nu": 3.0}, eta=0.03)
    whole = _fit(design, target, precision, config=config)
    monkeypatch.setattr(module, "_RESULT_CHUNK", 3)
    chunked = _fit(design, target, precision, config=config)
    for name in ("theta", "model", "weights", "effective_precision", "objective"):
        torch.testing.assert_close(getattr(chunked, name), getattr(whole, name), rtol=0, atol=0)


def test_exact_fixed_weight_solve_is_taken_even_from_an_infeasible_start() -> None:
    # min Σ λ (y − x a)² with a ≥ 0 and y = −x: the minimiser is a = 0, but the
    # infeasible start a = −1 fits exactly (lower weighted objective than any
    # feasible point), so a decrease test would reject the exact solve.
    pixels = 40
    x = torch.linspace(0.5, 1.5, pixels, dtype=DTYPE)
    design = x.reshape(1, pixels, 1).expand(2, pixels, 1).clone()
    target = -design[..., 0]
    precision = torch.full_like(target, 4.0)
    linear = _linear_problem(design, nonnegative=torch.tensor([True]))

    def damped(theta, weights, target_, rows, damping):  # type: ignore[no-untyped-def]
        return linear.solve(theta, weights, target_, rows).theta

    problem = replace(linear, damped_solve=damped)
    config = BatchedIRLSFitConfig(weight_params={"nu": math.inf})
    start = torch.full((2, 1), -1.0, dtype=DTYPE)
    result = batched_iteratively_reweighted_fit(problem, start, target, precision, None, config)
    assert (result.status == IRLSRowStatus.CONVERGED).all()
    assert torch.equal(result.theta, torch.zeros(2, 1, dtype=DTYPE))
    assert (result.iterations == 1).all()


def test_round_off_certificate_rejects_a_predicted_increase() -> None:
    design, target, precision, _ = _data(batch=3, seed=8)
    gauss = BatchedIRLSFitConfig(weight_params={"nu": math.inf})
    exact = _fit(design, target, precision, config=gauss)
    config = replace(gauss, max_iter=5, max_damping=3)
    overshoot = _overshooting_problem(design, factor=-1e8, damped_fraction=None)

    def solve(theta, weights, target_, rows):  # type: ignore[no-untyped-def]
        step = overshoot.solve(theta, weights, target_, rows)
        q0 = (weights * (target_ - overshoot.predict(theta, rows)).square()).sum(-1)
        # The local model predicts a large increase: nothing shows the start is optimal.
        return replace(step, predicted_objective=q0 + 1e30)

    start = exact.theta + 1e-9 * exact.standard_errors
    result = batched_iteratively_reweighted_fit(
        replace(overshoot, solve=solve), start, target, precision, None, config
    )
    assert (result.status == IRLSRowStatus.STALLED).all()
    torch.testing.assert_close(result.theta, start)


@pytest.mark.parametrize("eta", [0.0, 0.03])
def test_non_finite_model_off_the_likelihood_is_ignored(eta: float) -> None:
    design, target, precision, _ = _data(batch=4, seed=19)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[:, :10] = False
    precision = precision.clone()
    precision[:, -5:] = 0.0
    off = ~mask | (precision == 0.0)
    linear = _linear_problem(design)

    def predict(theta, rows):  # type: ignore[no-untyped-def]
        # A template undefined outside the observed range.
        return linear.predict(theta, rows).masked_fill(_take(off, rows), math.nan)

    config = BatchedIRLSFitConfig(weight_params={"nu": 4.0}, eta=eta)
    theta0 = torch.zeros(4, 3, dtype=DTYPE)
    control = batched_iteratively_reweighted_fit(linear, theta0, target, precision, mask, config)
    result = batched_iteratively_reweighted_fit(
        replace(linear, predict=predict), theta0, target, precision, mask, config
    )
    assert control.converged.all()
    assert torch.equal(result.status, control.status)
    for name in ("theta", "standard_errors", "objective", "weights", "effective_precision"):
        torch.testing.assert_close(getattr(result, name), getattr(control, name), rtol=0, atol=0)
    assert torch.isnan(result.model[off]).all()
    torch.testing.assert_close(result.model[~off], control.model[~off], rtol=0, atol=0)


def test_ridge_used_by_any_solve_is_reported_per_row() -> None:
    design, target, precision, _ = _data(batch=3, seed=20)
    design = design.clone()
    design[1, :, 2] = design[1, :, 0] + design[1, :, 1]  # row 1 is not identified
    result = _fit(design, target, precision)
    assert result.ridge_used.tolist() == [False, True, False]
    assert result.ridge_used.dtype == torch.bool
    linear = _linear_problem(design)

    def bad(theta, weights, target_, rows):  # type: ignore[no-untyped-def]
        return replace(linear.solve(theta, weights, target_, rows), ridge_used=torch.ones(2))

    with pytest.raises(ValueError, match="ridge_used"):
        batched_iteratively_reweighted_fit(
            replace(linear, solve=bad), torch.zeros(3, 3, dtype=DTYPE), target, precision
        )


def test_default_config_is_student_t_nu5_and_callable_weights_need_no_params() -> None:
    design, target, precision, _ = _data(seed=21)
    default = _fit(design, target, precision)
    explicit = _fit(
        design, target, precision, config=BatchedIRLSFitConfig(weight_params={"nu": 5.0})
    )
    for name in ("theta", "objective", "weights"):
        assert torch.equal(getattr(default, name), getattr(explicit, name))

    def huber(u: torch.Tensor) -> torch.Tensor:
        return torch.where(u <= 1.345, torch.ones_like(u), 1.345 / u)

    config = BatchedIRLSFitConfig(weight=huber)
    by_name = _fit(design, target, precision, config=BatchedIRLSFitConfig(weight="huber"))
    custom = _fit(design, target, precision, config=config)
    assert custom.converged.all()
    torch.testing.assert_close(custom.theta, by_name.theta, rtol=1e-12, atol=1e-14)


def test_damped_controller_does_not_repredict_the_start_of_an_iteration() -> None:
    design, target, precision, _ = _data(batch=5, seed=22)
    linear = _linear_problem(design)
    solves: list[dict[int, torch.Tensor]] = []
    repredicted: list[int] = []  # iteration of every model evaluation at the solve's start

    def solve(theta, weights, target_, rows):  # type: ignore[no-untyped-def]
        ids = range(theta.shape[0]) if rows is None else rows.tolist()
        solves.append({int(i): theta[k].clone() for k, i in enumerate(ids)})
        return linear.solve(theta, weights, target_, rows)

    def predict(theta, rows):  # type: ignore[no-untyped-def]
        ids = range(theta.shape[0]) if rows is None else rows.tolist()
        for k, i in enumerate(ids):
            start = solves[-1].get(int(i)) if solves else None
            if start is not None and torch.equal(theta[k], start):
                repredicted.append(len(solves))
        return linear.predict(theta, rows)

    def damped(theta, weights, target_, rows, damping):  # type: ignore[no-untyped-def]
        return linear.solve(theta, weights, target_, rows).theta

    problem = replace(linear, predict=predict, solve=solve, damped_solve=damped, exact_solve=False)
    config = BatchedIRLSFitConfig(weight_params={"nu": 4.0}, step_tol=1e-6)
    result = batched_iteratively_reweighted_fit(
        problem, torch.zeros(5, 3, dtype=DTYPE), target, precision, None, config
    )
    assert result.converged.all() and int(result.iterations.max()) > 2
    # q(θ⁰) is evaluated once, at the first iteration (5 rows); afterwards the
    # weight refresh of the previous iteration provides it.
    assert repredicted == [1] * 5
    reference = _fit(design, target, precision, config=config)
    torch.testing.assert_close(result.theta, reference.theta, rtol=1e-10, atol=1e-12)


def _live_row_buffers(pixels: int, exclude: set[int]) -> float:
    """Rows of live float [*, pixels] tensor storages, deduplicated by storage."""
    import gc

    seen: dict[int, int] = {}
    for obj in gc.get_objects():
        if issubclass(type(obj), torch.Tensor) and obj.is_floating_point():
            if obj.ndim == 2 and obj.shape[-1] == pixels:
                storage = obj.untyped_storage()
                if storage.data_ptr() not in exclude:
                    seen[storage.data_ptr()] = storage.nbytes() // (pixels * obj.element_size())
    return float(sum(seen.values()))


@pytest.mark.parametrize("damped", [False, True])
def test_controller_keeps_about_three_row_buffers(damped: bool) -> None:
    import gc

    gc.collect()
    batch, pixels = 8, 160
    design, target, precision, _ = _data(batch=batch, pixels=pixels, seed=23)
    target[0] = (design[0] @ torch.ones(3, dtype=DTYPE)).clone()  # finishes early
    linear = _linear_problem(design)
    exclude = {t.untyped_storage().data_ptr() for t in (target, precision)}
    peak = {"solve": 0.0, "compacted": 0.0, "any": 0.0}

    def census(rows: torch.Tensor | None, at_solve: bool) -> None:
        units = _live_row_buffers(pixels, exclude) / batch
        peak["any"] = max(peak["any"], units)
        if at_solve:
            peak["solve"] = max(peak["solve"], units)
            if rows is not None:
                peak["compacted"] = max(peak["compacted"], units)

    def solve(theta, weights, target_, rows):  # type: ignore[no-untyped-def]
        census(rows, True)
        return linear.solve(theta, weights, target_, rows)

    def predict(theta, rows):  # type: ignore[no-untyped-def]
        census(rows, False)
        return linear.predict(theta, rows)

    def damped_solve(theta, weights, target_, rows, damping):  # type: ignore[no-untyped-def]
        return linear.solve(theta, weights, target_, rows).theta

    problem = replace(
        linear,
        solve=solve,
        predict=predict,
        damped_solve=damped_solve if damped else None,
        exact_solve=not damped,
    )
    result = batched_iteratively_reweighted_fit(
        problem, torch.zeros(batch, 3, dtype=DTYPE), target, precision
    )
    assert result.converged.all()
    assert peak["compacted"] > 0.0  # compacted iterations were observed
    # At every solve the controller holds the working target, precision and
    # weights of the running rows only: no full-size copies next to gathered ones.
    assert peak["solve"] <= 3.0
    # Elsewhere: two more transients at most (the decrease test's gathered rows,
    # or the result's chunk next to its three outputs).
    assert peak["any"] <= 5.0


@pytest.mark.parametrize("damped", [False, True])
def test_callbacks_of_one_iteration_share_the_rows_object(damped: bool) -> None:
    design, target, precision, _ = _data(batch=8, seed=24)
    target[0] = (design[0] @ torch.ones(3, dtype=DTYPE)).clone()  # finishes early
    linear = _linear_problem(design)
    calls: list[tuple[str, torch.Tensor | None]] = []

    def solve(theta, weights, target_, rows):  # type: ignore[no-untyped-def]
        calls.append(("solve", rows))
        return linear.solve(theta, weights, target_, rows)

    def predict(theta, rows):  # type: ignore[no-untyped-def]
        calls.append(("predict", rows))
        return linear.predict(theta, rows)

    def damped_solve(theta, weights, target_, rows, damping):  # type: ignore[no-untyped-def]
        return linear.solve(theta, weights, target_, rows).theta

    problem = replace(
        linear,
        solve=solve,
        predict=predict,
        damped_solve=damped_solve if damped else None,
        exact_solve=not damped,
    )
    result = batched_iteratively_reweighted_fit(
        problem, torch.zeros(8, 3, dtype=DTYPE), target, precision
    )
    assert result.converged.all()
    solves = [i for i, (name, _) in enumerate(calls) if name == "solve"]
    for start, stop in zip(solves, solves[1:] + [len(calls)], strict=True):
        rows = calls[start][1]
        # The refresh is the last model evaluation of the iteration (before the
        # result's chunked evaluation, which follows the last solve).
        refresh = [r for name, r in calls[start + 1 : stop] if name == "predict"]
        if stop != len(calls):
            assert refresh and refresh[-1] is rows
    objects = [calls[i][1] for i in solves]
    compactions = sum(a is not b for a, b in zip(objects, objects[1:], strict=False))
    sizes = [8 if r is None else r.numel() for r in objects]
    assert compactions == sum(a != b for a, b in zip(sizes, sizes[1:], strict=False))
    assert compactions >= 1
