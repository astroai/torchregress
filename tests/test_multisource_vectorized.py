"""The batched multi-source losses reproduce the per-object reference exactly.

``tests/_multisource_reference.py`` holds the original object-by-object
implementation. Randomized float64 problems with ragged missingness, several
sources, correlated noise, adaptive jitter, both factorizations, and the robust
mixture must agree in values, gradients, and diagnostics to 1e-10 relative.
"""

from __future__ import annotations

import pytest
import torch

from tests._multisource_reference import ReferenceMarginalizedLoss, ReferenceRobustLoss
from torchregress.losses.multisource import (
    MarginalizedMultiSourceGaussianNLLLoss,
    PackedObservations,
)
from torchregress.losses.multisource_robust import RobustMultiSourceGaussianNLLLoss

RTOL = 1e-10
METHODS = ("auto", "direct", "information")
BLOCK_POOL = (3, 7, 11, 20)  # local, non-contiguous block IDs


def _spd(n: int, gen: torch.Generator, negative: float = 0.0) -> torch.Tensor:
    """Random SPD matrix; with ``negative > 0`` its smallest eigenvalue is ``-negative``."""
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
    eig = 0.05 + torch.rand(n, generator=gen, dtype=torch.float64)
    if negative:
        eig[0] = -negative
    return (q * eig) @ q.T


def make_problem(
    seed: int,
    device: str,
    *,
    batch: int = 40,
    k: int = 9,
    dims: int = 3,
    block_diagonal: bool = False,
    indefinite: bool = False,
) -> dict[str, torch.Tensor]:
    """Ragged multi-source problem; unobserved values and covariance entries are NaN."""
    gen = torch.Generator().manual_seed(seed)
    observe = torch.rand(batch, 1, generator=gen) * 0.9 + 0.1
    mask = torch.rand(batch, k, generator=gen) < observe
    mask[0] = False  # empty object
    mask[1] = True  # fully observed object
    mask[2] = False
    mask[2, :2] = True  # fewer observations than latent dimensions
    block_ids = torch.tensor(BLOCK_POOL)[torch.randint(0, 4, (batch, k), generator=gen)]
    cov = torch.full((batch, k, k), float("nan"), dtype=torch.float64)
    # Indefinite observed covariances need distinct, well-separated jitter levels
    # (1e-3, 1e-2, 1e-1) and stay well conditioned once jittered.
    negatives = (5e-4, 5e-3, 5e-2)
    for i in range(batch):
        idx = mask[i].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        if block_diagonal:
            local = torch.zeros(idx.numel(), idx.numel(), dtype=torch.float64)
            ids = block_ids[i, idx]
            for b in ids.unique():
                members = (ids == b).nonzero(as_tuple=True)[0]
                negative = negatives[i % 3] if indefinite and i % 4 == 0 and b == ids[0] else 0.0
                local[members[:, None], members[None, :]] = _spd(members.numel(), gen, negative)
        else:
            negative = negatives[i % 3] if indefinite and i % 4 == 0 else 0.0
            local = _spd(idx.numel(), gen, negative)  # full cross-source correlation
        cov[i, idx[:, None], idx[None, :]] = local * (0.1 + torch.rand((), generator=gen))
    values = torch.randn(batch, k, generator=gen, dtype=torch.float64)
    values[~mask] = float("nan")
    raw = 0.4 * torch.randn(batch, dims, dims, generator=gen, dtype=torch.float64)
    lower = torch.tril(raw, -1) + torch.diag_embed(0.3 + raw.diagonal(dim1=-2, dim2=-1).abs())
    tensors = {
        "mean": torch.randn(batch, dims, generator=gen, dtype=torch.float64),
        "lower": lower,
        "jacobian": torch.randn(batch, k, dims, generator=gen, dtype=torch.float64),
        "offset": 0.1 * torch.randn(batch, k, generator=gen, dtype=torch.float64),
        "values": values,
        "cov": cov,
        "mask": mask,
        "block_ids": block_ids,
        "p": torch.rand(batch, 1, generator=gen, dtype=torch.float64).expand(batch, k) * 0.5,
        "s": 1 + 4 * torch.rand(batch, 1, generator=gen, dtype=torch.float64).expand(batch, k),
        "weights": torch.rand(batch, generator=gen, dtype=torch.float64),
    }
    # Outlier parameters are constant within each block, not only within each object.
    block_index = torch.searchsorted(torch.tensor(BLOCK_POOL), block_ids)
    tensors["p"] = tensors["p"] * (1 + block_index) / 4
    tensors["s"] = tensors["s"] + block_index
    return {name: x.to(device) for name, x in tensors.items()}


def _leaves(problem: dict[str, torch.Tensor], robust: bool) -> dict[str, torch.Tensor]:
    names = ["mean", "lower", "jacobian", "offset", "cov"]
    if robust:
        names += ["p", "s"]
    return {name: problem[name].clone().requires_grad_(True) for name in names}


def _observations(leaves: dict[str, torch.Tensor], problem: dict[str, torch.Tensor]):
    mapped = torch.einsum("bkd,bd->bk", leaves["jacobian"], leaves["mean"]) + leaves["offset"]
    return PackedObservations(
        problem["values"],
        mapped,
        leaves["jacobian"],
        leaves["cov"],
        problem["mask"],
        problem["block_ids"],
    )


def assert_relative(actual: torch.Tensor, expected: torch.Tensor, rtol: float = RTOL) -> None:
    """Agree elementwise to ``rtol`` relative to each entry and to the tensor scale."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    scale = float(expected.abs().max()) if expected.numel() else 0.0
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=rtol * max(scale, 1e-300))


def _run(loss, problem, robust: bool, reduction_weights: bool = False):
    leaves = _leaves(problem, robust)
    obs = _observations(leaves, problem)
    kwargs = {}
    if robust:
        kwargs = {"outlier_probabilities": leaves["p"], "outlier_scales": leaves["s"]}
    per = loss.per_star(leaves["mean"], leaves["lower"], obs, **kwargs)
    diagnostics = dict(loss.last_diagnostics)
    if robust:
        diagnostics["inlier"] = dict(diagnostics["inlier"])
    weights = problem["weights"] if reduction_weights else None
    total = loss(leaves["mean"], leaves["lower"], obs, weights, **kwargs)
    objective = per.square().sum() * 1e-3 + per.sum() + total.sum()
    grads = torch.autograd.grad(objective, list(leaves.values()), allow_unused=True)
    return per, total, dict(zip(leaves, grads, strict=True)), diagnostics


def _compare(new, reference, problem, robust: bool, weights: bool = False) -> None:
    per, total, grads, diagnostics = _run(new, problem, robust, weights)
    ref_per, ref_total, ref_grads, ref_diagnostics = _run(reference, problem, robust, weights)
    assert_relative(per, ref_per)
    assert_relative(total, ref_total)
    empty = ~problem["mask"].any(-1)
    assert torch.equal(per[empty], torch.zeros_like(per[empty]))
    for name, ref_grad in ref_grads.items():
        grad = grads[name]
        if ref_grad is None:
            assert grad is None or torch.equal(grad, torch.zeros_like(grad)), name
        else:
            assert grad is not None, name
            assert_relative(grad, ref_grad)
    assert diagnostics == ref_diagnostics


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_marginal_matches_per_object_reference(seed, method, device):
    problem = make_problem(seed, device, dims=1 + seed)
    for reduction in ("mean", "sum", "none"):
        _compare(
            MarginalizedMultiSourceGaussianNLLLoss(reduction, method),
            ReferenceMarginalizedLoss(reduction, method),
            problem,
            robust=False,
            weights=reduction != "none",
        )


@pytest.mark.parametrize("method", METHODS)
def test_marginal_adaptive_jitter_matches_reference_levels(method, device):
    problem = make_problem(5, device, indefinite=True)
    new = MarginalizedMultiSourceGaussianNLLLoss("sum", method, jitter=0.0, max_jitter=1.0)
    reference = ReferenceMarginalizedLoss("sum", method, jitter=0.0, max_jitter=1.0)
    _compare(new, reference, problem, robust=False)
    used = set(dict(new.last_diagnostics)["jitter"])
    assert len(used - {0.0}) >= 3  # several distinct per-object levels
    with pytest.raises(ValueError, match="positive definite"):
        MarginalizedMultiSourceGaussianNLLLoss(method=method, max_jitter=1e-2)(
            problem["mean"], problem["lower"], _observations(_leaves(problem, False), problem)
        )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("seed", [0, 3])
def test_robust_matches_per_object_reference(seed, method, device):
    problem = make_problem(seed, device, block_diagonal=True, dims=2 + seed % 2)
    _compare(
        RobustMultiSourceGaussianNLLLoss("mean", method),
        ReferenceRobustLoss("mean", method),
        problem,
        robust=True,
        weights=True,
    )


def test_robust_adaptive_jitter_and_default_parameters_match_reference(device):
    problem = make_problem(7, device, block_diagonal=True, indefinite=True)
    for kwargs in ({}, {"outlier_probability": 0.0}, {"outlier_probability": 1.0}):
        new = RobustMultiSourceGaussianNLLLoss("none", jitter=1e-13, max_jitter=100.0, **kwargs)
        reference = ReferenceRobustLoss("none", jitter=1e-13, max_jitter=100.0, **kwargs)
        leaves = _leaves(problem, robust=False)
        obs = _observations(leaves, problem)
        actual = new(leaves["mean"], leaves["lower"], obs)
        expected = reference(leaves["mean"], leaves["lower"], obs)
        assert_relative(actual, expected)
        for g, r in zip(
            torch.autograd.grad(actual.sum(), list(leaves.values()), retain_graph=True),
            torch.autograd.grad(expected.sum(), list(leaves.values())),
            strict=True,
        ):
            assert_relative(g, r)
        assert (
            dict(new.last_diagnostics)["assignment_counts"]
            == dict(reference.last_diagnostics)["assignment_counts"]
        )


def test_unobserved_entries_may_be_nan_everywhere(device):
    problem = make_problem(11, device)
    mask = problem["mask"]
    jacobian = problem["jacobian"].masked_fill(~mask[..., None], float("nan"))
    mapped = problem["offset"].masked_fill(~mask, float("nan"))
    cov = problem["cov"].clone().requires_grad_(True)
    obs = PackedObservations(problem["values"], mapped, jacobian, cov, mask)
    lower = problem["lower"].clone().requires_grad_(True)
    actual = MarginalizedMultiSourceGaussianNLLLoss("none")(problem["mean"], lower, obs)
    expected = ReferenceMarginalizedLoss("none")(problem["mean"], lower, obs)
    assert_relative(actual, expected)
    grads = torch.autograd.grad(actual.sum(), (lower, cov))
    ref_grads = torch.autograd.grad(expected.sum(), (lower, cov))
    for g, r in zip(grads, ref_grads, strict=True):
        assert torch.isfinite(g).all()
        assert_relative(g, r)


def test_mixture_dtype_wider_than_inputs_matches_reference(device):
    problem = make_problem(13, device, block_diagonal=True)
    single = {n: x.float() if x.is_floating_point() else x for n, x in problem.items()}
    leaves = _leaves(single, robust=False)
    obs = _observations(leaves, single)
    p = problem["p"]  # float64 probabilities widen the outlying assignments
    actual = RobustMultiSourceGaussianNLLLoss("none")(
        leaves["mean"], leaves["lower"], obs, outlier_probabilities=p
    )
    expected = ReferenceRobustLoss("none")(
        leaves["mean"], leaves["lower"], obs, outlier_probabilities=p
    )
    assert actual.dtype == expected.dtype == torch.float64
    # The inlier assignment is factorized in float32 by both implementations.
    assert_relative(actual, expected, rtol=1e-5)


def _invalid_cases(problem):
    mask = problem["mask"]
    first = int(mask.any(-1).nonzero()[1])  # a nonempty object
    idx = mask[first].nonzero(as_tuple=True)[0]

    def with_cov(update):
        cov = problem["cov"].clone()
        update(cov)
        return {"cov": cov}

    def asymmetric(cov):
        cov[first, idx[0], idx[1]] += 1e-3

    def indefinite(cov):
        cov[first, idx[0], idx[0]] = -1.0

    values = problem["values"].clone()
    values[first, idx[0]] = float("inf")
    yield "finite", {"values": values}
    yield "symmetric", with_cov(asymmetric)
    yield "positive definite", with_cov(indefinite)


@pytest.mark.parametrize("method", METHODS)
def test_invalid_inputs_raise_the_reference_errors(method, device):
    problem = make_problem(17, device)
    for match, change in _invalid_cases(problem):
        changed = {**problem, **change}
        obs = PackedObservations(
            changed["values"],
            changed["offset"],
            changed["jacobian"],
            changed["cov"],
            changed["mask"],
        )
        for loss in (
            MarginalizedMultiSourceGaussianNLLLoss(method=method),
            ReferenceMarginalizedLoss(method=method),
        ):
            with pytest.raises(ValueError, match=match):
                loss(changed["mean"], changed["lower"], obs)


def test_robust_invalid_block_inputs_raise_the_reference_errors(device):
    problem = make_problem(19, device, block_diagonal=True)
    leaves = _leaves(problem, robust=False)
    obs = _observations(leaves, problem)
    p = problem["p"].clone()
    nonempty = problem["mask"].any(-1).nonzero()[1, 0]
    slots = problem["mask"][nonempty].nonzero(as_tuple=True)[0]
    p[nonempty, slots[0]] = 0.9 * p[nonempty, slots[0]] + 0.01
    cross = problem["cov"].clone()
    cross[nonempty, slots[0], slots[-1]] = cross[nonempty, slots[-1], slots[0]] = 1e-3
    ids = problem["block_ids"].clone()
    ids[nonempty, slots[-1]] = BLOCK_POOL[0] if ids[nonempty, slots[0]] != 3 else BLOCK_POOL[1]
    blocks_differ = ids[nonempty, slots[0]] != ids[nonempty, slots[-1]]
    cases = [
        ("constant", obs, {"outlier_probabilities": p}, {}),
        ("bound", obs, {}, {"max_blocks": 1}),
        ("bound", obs, {}, {"max_assignments": 2}),
        ("scales", obs, {"outlier_scales": 0.5}, {}),
        ("probabilit", obs, {"outlier_probabilities": float("nan")}, {}),
    ]
    if bool(blocks_differ):
        changed = PackedObservations(
            obs.values, obs.mapped_mean, obs.jacobian, cross, obs.mask, ids
        )
        cases.append(("Cross-block", changed, {}, {}))
    for match, observations, call, init in cases:
        for cls in (RobustMultiSourceGaussianNLLLoss, ReferenceRobustLoss):
            with pytest.raises(ValueError, match=match):
                cls(**init)(leaves["mean"], leaves["lower"], observations, **call)
