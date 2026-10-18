"""Tests for variational imbalanced regression (VIR)."""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchregress.algorithms.vir import VIRLoss, VIRRegressor

DTYPE = torch.float64
_PRIOR = (0.0, 1.0, 2.0, 0.1)


# ── helpers ────────────────────────────────────────────────────────────
def _hand_nig(
    raw: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    prior: tuple[float, float, float, float] = _PRIOR,
    *,
    min_nu: float = 2.0,
    min_beta: float = 1e-6,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Reference NIG posterior and Student-t NLL written out by hand."""
    gamma0, nu0, alpha0, beta0 = prior
    pseudo_count = F.softplus(raw[:, 0]) + min_nu
    psi = raw[:, 1]
    phi = F.softplus(raw[:, 2]) + min_beta
    nu = nu0 + weight * pseudo_count
    gamma = (gamma0 * nu0 + weight * pseudo_count * psi) / nu
    alpha = alpha0 + 0.5 * weight * pseudo_count
    beta = beta0 + 0.5 * gamma0 * gamma0 * nu0 + phi
    omega = 2.0 * beta * (1.0 + nu)
    nll = (
        0.5 * torch.log(math.pi / nu)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log(nu * (target - gamma) ** 2 + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    return nll, (gamma, nu, alpha, beta)


def _make_encoder(n_features: int, hidden: int = 32) -> nn.Module:
    return nn.Sequential(
        nn.Linear(n_features, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
    )


def _imbalanced_class_data(
    n_samples: int,
    seed: int,
    *,
    probabilities: tuple[float, ...] = (0.02, 0.08, 0.80, 0.08, 0.02),
    noise: float = 0.8,
    n_features: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discrete, strongly imbalanced regression targets.

    The five target values are ordered, so neighbouring-label smoothing is
    meaningful.  The 20% balanced test variant is produced by passing uniform
    probabilities.
    """
    generator = torch.Generator().manual_seed(seed)
    classes = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    probability = torch.tensor(probabilities, dtype=torch.float64)
    probability = probability / probability.sum()
    labels = classes[
        torch.multinomial(probability, n_samples, replacement=True, generator=generator)
    ]
    direction = torch.tensor([1.0, 0.2, -0.3, 0.4, 0.1, -0.2, 0.3, 0.5][:n_features])
    features = 1.5 * labels.unsqueeze(1) * direction + noise * torch.randn(
        n_samples, n_features, generator=generator
    )
    return features.float(), labels.unsqueeze(1).float()


def _train_mse(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    encoder: nn.Module | None = None,
    epochs: int = 120,
    lr: float = 3e-3,
    batch_size: int = 64,
    seed: int = 0,
    hidden: int = 32,
) -> nn.Module:
    """Plain-MSE counterpart with the same encoder architecture and schedule.

    Pass ``encoder`` to reuse a specific (deep-copied) feature extractor, which
    keeps the comparison against VIR matched on the encoder initialisation.
    """
    torch.manual_seed(seed)
    feature_encoder = (
        _make_encoder(X.shape[1], hidden) if encoder is None else copy.deepcopy(encoder)
    )
    model = nn.Sequential(feature_encoder, nn.Linear(hidden, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed + 1)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(X.shape[0], generator=generator)
        for start in range(0, X.shape[0], batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(X[index]) - y[index]) ** 2).mean()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def _tail_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    return float((prediction - target).abs()[mask].mean())


# ── NIG mathematics ────────────────────────────────────────────────────
def test_nig_nll_matches_hand_formula() -> None:
    """The loss equals the Student-t NLL of the reweighted NIG posterior."""
    torch.manual_seed(0)
    loss_fn = VIRLoss(prior=_PRIOR, reg_weight=0.0, n_bins=4).fit(
        torch.linspace(-2.0, 2.0, 64, dtype=DTYPE)
    )
    raw = torch.randn(5, 3, dtype=DTYPE)
    target = torch.randn(5, dtype=DTYPE)
    weight = torch.tensor([1.0, 2.0, 3.0, 1.5, 4.0], dtype=DTYPE)

    loss = loss_fn(raw, target, importance_weight=weight)
    expected_nll, (gamma, nu, alpha, beta) = _hand_nig(raw, target, weight)
    assert torch.isclose(loss, expected_nll.mean(), atol=1e-12, rtol=0.0)

    posterior = loss_fn.posterior_parameters(raw, weight)
    assert torch.allclose(posterior.gamma, gamma, atol=1e-12)
    assert torch.allclose(posterior.nu, nu, atol=1e-12)
    assert torch.allclose(posterior.alpha, alpha, atol=1e-12)
    assert torch.allclose(posterior.beta, beta, atol=1e-12)

    # The tuple layout is equivalent to the packed (N, 3) layout.
    packed = loss_fn(raw, target, importance_weight=weight)
    split = loss_fn((raw[:, 0], raw[:, 1], raw[:, 2]), target, importance_weight=weight)
    assert torch.allclose(packed, split, atol=0.0)


def test_regulariser_weight_adds_paper_penalty() -> None:
    """``lambda * (nu + 2 alpha) * |y - gamma|`` is added on top of the NLL."""
    torch.manual_seed(1)
    raw = torch.randn(6, 3, dtype=DTYPE)
    target = torch.randn(6, dtype=DTYPE)
    weight = torch.tensor([1.0, 2.0, 1.0, 3.0, 1.0, 2.0], dtype=DTYPE)

    plain = VIRLoss(prior=_PRIOR, reg_weight=0.0)
    regularised = VIRLoss(prior=_PRIOR, reg_weight=0.25)
    expected_nll, (gamma, nu, alpha, _) = _hand_nig(raw, target, weight)
    expected = expected_nll + 0.25 * (nu + 2.0 * alpha) * (target - gamma).abs()
    actual = regularised(raw, target, importance_weight=weight)
    assert torch.isclose(actual, expected.mean(), atol=1e-12, rtol=0.0)
    assert not torch.isclose(
        actual, plain(raw, target, importance_weight=weight), atol=1e-12, rtol=0.0
    )


def test_predictive_mean_std_matches_student_t_variance() -> None:
    """``std`` is the total Student-t variance, not the epistemic term alone."""
    torch.manual_seed(2)
    loss_fn = VIRLoss(prior=_PRIOR, n_bins=4)
    raw = torch.randn(9, 3, dtype=DTYPE)
    weight = torch.linspace(1.0, 3.0, 9, dtype=DTYPE)
    posterior = loss_fn.posterior_parameters(raw, weight)
    mean, std = loss_fn.predictive_mean_std(posterior)

    assert torch.allclose(mean, posterior.gamma, atol=0.0)
    variance = posterior.beta * (1.0 + posterior.nu) / (posterior.nu * (posterior.alpha - 1.0))
    assert torch.allclose(std, variance.sqrt(), atol=1e-14)
    assert torch.isfinite(std).all() and bool((std > 0.0).all())


def test_reductions_shapes_and_mask() -> None:
    """Reduction, mask and weight semantics follow the shared loss contract."""
    torch.manual_seed(3)
    raw = torch.randn(7, 3)
    target = torch.randn(7)
    base = VIRLoss(prior=_PRIOR, n_bins=4)
    per_sample = base(raw, target)
    assert per_sample.dim() == 0

    none = VIRLoss(prior=_PRIOR, n_bins=4, reduction="none")
    values = none(raw, target)
    assert values.shape == (7,)
    assert torch.isclose(none(raw, target.unsqueeze(-1)), values.unsqueeze(-1), atol=1e-6).all()
    assert torch.isclose(none(raw, target).sum(), 7.0 * per_sample, atol=1e-5)

    summed = VIRLoss(prior=_PRIOR, n_bins=4, reduction="sum")
    assert torch.isclose(summed(raw, target), values.sum(), atol=1e-5)

    # Constant user weights do not change the mean; a mask removes entries.
    weights = torch.full((7,), 2.5)
    assert torch.isclose(base(raw, target, weights=weights), per_sample, atol=1e-5)
    mask = torch.tensor([True, True, False, True, True, False, True])
    masked = base(raw, target, mask=mask)
    expected = values[mask].mean()
    assert torch.isclose(masked, expected, atol=1e-5)
    assert torch.allclose(
        none(raw, target.unsqueeze(-1), mask=mask.unsqueeze(-1)).squeeze(-1)[~mask],
        torch.zeros(2),
        atol=0.0,
    )


def test_gradients_flow_to_every_nig_output() -> None:
    """All three raw head columns receive finite, non-zero gradients."""
    torch.manual_seed(4)
    loss_fn = VIRLoss(prior=_PRIOR, reg_weight=0.01, n_bins=5).fit(torch.linspace(-2.0, 2.0, 128))
    raw = torch.randn(16, 3, dtype=DTYPE, requires_grad=True)
    loss_fn(raw, torch.randn(16, dtype=DTYPE)).backward()
    assert raw.grad is not None
    assert raw.grad.shape == raw.shape
    assert torch.isfinite(raw.grad).all()
    assert bool((raw.grad.abs().sum(dim=0) > 1e-8).all())


# ── label density and latent smoothing ────────────────────────────────
def test_importance_weights_upweight_rare_labels() -> None:
    torch.manual_seed(5)
    bulk = 0.2 * torch.randn(990)
    rare = torch.full((10,), 4.0)
    targets = torch.cat([bulk, rare])
    loss_fn = VIRLoss(n_bins=10, kernel_sigma=1.0, max_importance_weight=10.0).fit(targets)
    weights = loss_fn.importance_weights(targets)

    assert weights.shape == targets.shape
    assert bool((weights > 0.0).all())
    assert float(weights[-10:].mean()) > float(weights[:990].mean())
    assert float(weights.max()) <= 10.0 + 1e-6

    # Boundary and non-finite handling.
    index = loss_fn.bin_index(torch.tensor([-1e6, 0.0, 1e6, float("nan")]))
    assert index.tolist() == [0, index[1].item(), loss_fn.n_bins - 1, 0]
    assert loss_fn.importance_weights(torch.tensor([float("nan")])).item() == 1.0
    bulk_weight = loss_fn.importance_weights(torch.tensor([0.0])).item()
    assert 0.0 < bulk_weight < 10.0
    assert bulk_weight < float(weights[-10:].mean())


def test_bin_statistics_and_occupancy_aware_smoothing() -> None:
    """Empty bins must not contaminate the smoothing of occupied neighbours."""
    n_bins = 3
    latent_dim = 2
    # Bin 0: two samples; bin 1: one sample; bin 2: empty.
    mean = torch.tensor([[1.0, 0.0], [2.0, 0.0], [4.0, 2.0]], dtype=DTYPE)
    variance = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=DTYPE)
    bin_index = torch.tensor([0, 0, 1])
    sum_mean = torch.zeros(n_bins, latent_dim, dtype=DTYPE)
    sum_variance = torch.zeros_like(sum_mean)
    sum_second = torch.zeros_like(sum_mean)
    counts = torch.zeros(n_bins, dtype=DTYPE)
    for row, bin_id in enumerate(bin_index):
        sum_mean[bin_id] += mean[row]
        sum_variance[bin_id] += variance[row]
        sum_second[bin_id] += variance[row] + mean[row] * mean[row]
        counts[bin_id] += 1

    stats = VIRRegressor._summarise_bin_statistics(sum_mean, sum_variance, sum_second, counts)
    bin_mean, bin_mean_variance, bin_covariance = stats
    assert torch.allclose(bin_mean[0], torch.tensor([1.5, 0.0], dtype=DTYPE))
    assert torch.allclose(bin_mean[1], torch.tensor([4.0, 2.0], dtype=DTYPE))
    # Empty bin 2 is imputed with the sample-weighted global mean (7/3, 2/3).
    assert torch.allclose(bin_mean[2], torch.tensor([7.0 / 3.0, 2.0 / 3.0], dtype=DTYPE))
    assert torch.allclose(bin_mean_variance[0], torch.tensor([0.25, 0.25], dtype=DTYPE))
    # Within-bin covariance is E[z^2] - E[z]^2 - Var(mean) >= 0.
    assert bool((bin_covariance > 0.0).all())

    kernel = VIRLoss(n_bins=n_bins, kernel_sigma=1.0).smoothing_matrix().to(DTYPE)
    smoothed_mean, _, smoothed_covariance = VIRRegressor._smooth_bin_statistics(
        stats, kernel, counts
    )
    # Row 0 must average bins 0 and 1 (both occupied) but NOT the empty bin 2,
    # whose imputed global mean would otherwise contaminate the result.
    weight_0, weight_1 = kernel[0, 0], kernel[0, 1]
    expected_mean = (weight_0 * bin_mean[0] + weight_1 * bin_mean[1]) / (weight_0 + weight_1)
    assert torch.allclose(smoothed_mean[0], expected_mean, atol=1e-12)
    with_empty_bin = (
        weight_0 * bin_mean[0] + weight_1 * bin_mean[1] + kernel[0, 2] * bin_mean[2]
    ) / (weight_0 + weight_1 + kernel[0, 2])
    assert not torch.allclose(smoothed_mean[0], with_empty_bin, atol=1e-6)
    expected_covariance = (weight_0 * bin_covariance[0] + weight_1 * bin_covariance[1]) / (
        weight_0 + weight_1
    )
    assert torch.allclose(smoothed_covariance[0], expected_covariance, atol=1e-12)
    # Every row still produces finite statistics.
    assert torch.isfinite(smoothed_mean).all()


def test_recolour_shrinkage_keeps_sparse_bins() -> None:
    """Large shrinkage recovers the raw representation; zero shrinkage uses the transform."""
    torch.manual_seed(6)
    mean = torch.randn(4, 3)
    variance = torch.rand(4, 3) + 0.1
    bin_index = torch.tensor([0, 0, 1, 1])
    stats = (
        torch.randn(2, 3),
        torch.rand(2, 3),
        torch.rand(2, 3) + 0.5,
    )
    smoothed = (
        torch.randn(2, 3),
        torch.rand(2, 3),
        torch.rand(2, 3) + 0.5,
    )
    counts = torch.tensor([2.0, 1.0])

    raw_mean, raw_var = VIRRegressor._recolour(
        mean, variance, bin_index, stats, smoothed, counts, 1e9
    )
    assert torch.allclose(raw_mean, mean, atol=1e-6)
    assert torch.allclose(raw_var, variance, atol=1e-6)

    transformed_mean, transformed_var = VIRRegressor._recolour(
        mean, variance, bin_index, stats, smoothed, counts, 0.0
    )
    assert not torch.allclose(transformed_mean, mean)
    assert torch.isfinite(transformed_var).all() and bool((transformed_var > 0).all())


# ── end-to-end regressor ───────────────────────────────────────────────
def test_fit_predict_shapes_attributes_and_encoder_is_cloned() -> None:
    torch.manual_seed(7)
    X, y = _imbalanced_class_data(240, seed=7, n_features=4)
    encoder = _make_encoder(4, hidden=8)
    original_state = {key: value.clone() for key, value in encoder.state_dict().items()}

    regressor = VIRRegressor(
        encoder,
        n_bins=6,
        kernel_sigma=1.0,
        smoothing_shrinkage=10.0,
        epochs=5,
        lr=1e-2,
        batch_size=32,
        seed=0,
    )
    assert regressor.fit(X, y) is regressor
    assert regressor.is_fitted
    assert regressor.model_ is not None and not regressor.model_.training
    assert regressor.loss_fn_ is not None and regressor.loss_fn_.is_fitted
    assert len(regressor.history_) == 5
    assert all(math.isfinite(entry["loss"]) for entry in regressor.history_)

    mean, std = regressor.predict(X, return_std=True)
    assert mean.shape == (X.shape[0], 1)
    assert std.shape == (X.shape[0], 1)
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()
    assert bool((std > 0.0).all())
    lower, upper = regressor.predict_interval(X, confidence=0.8)
    assert lower.shape == (X.shape[0], 1) and upper.shape == (X.shape[0], 1)
    assert bool((lower <= mean).all()) and bool((mean <= upper).all())

    # Single-sample input is accepted.
    single = regressor.predict(X[0])
    assert single.shape == (1, 1)

    assert regressor.bin_counts_ is not None
    assert regressor.bin_counts_.shape == (6,)
    assert int(regressor.bin_counts_.sum()) == X.shape[0]
    assert regressor.bin_mean_ is not None
    assert regressor.bin_mean_.shape == (6, 8)
    assert regressor.smoothed_bin_cov_ is not None
    assert regressor.smoothed_bin_cov_.shape == (6, 8)

    # The caller's encoder is untouched.
    for key, value in encoder.state_dict().items():
        assert torch.equal(value, original_state[key])


def test_single_epoch_disables_inference_recolouring() -> None:
    """A one-epoch fit never saw recoloured features, so inference skips them."""
    X, y = _imbalanced_class_data(80, seed=9, n_features=3)
    one_epoch = VIRRegressor(
        _make_encoder(3, hidden=8), n_bins=4, epochs=1, batch_size=16, seed=0
    ).fit(X, y)
    assert one_epoch.smoothing_used_ is False
    assert one_epoch.predict(X).shape == (80, 1)

    two_epochs = VIRRegressor(
        _make_encoder(3, hidden=8), n_bins=4, epochs=2, batch_size=16, seed=0
    ).fit(X, y)
    assert two_epochs.smoothing_used_ is True


def test_fit_is_reproducible_and_does_not_touch_global_rng() -> None:
    X, y = _imbalanced_class_data(180, seed=8, n_features=4)
    encoder = _make_encoder(4, hidden=8)
    torch.manual_seed(123)
    state_before = torch.get_rng_state().clone()

    first = VIRRegressor(encoder, n_bins=5, epochs=4, lr=1e-2, batch_size=32, seed=11).fit(X, y)
    assert torch.equal(torch.get_rng_state(), state_before)

    second = VIRRegressor(encoder, n_bins=5, epochs=4, lr=1e-2, batch_size=32, seed=11).fit(X, y)
    assert first.history_ == second.history_
    assert torch.equal(first.predict(X), second.predict(X))
    assert torch.equal(first.predict(X, return_std=True)[1], second.predict(X, return_std=True)[1])


def test_vir_improves_tail_error_and_calibration_over_mse() -> None:
    """Controlled comparison: same encoder, optimiser, schedule and data order.

    The synthetic problem has five ordered target values drawn with probabilities
    ``[0.02, 0.08, 0.80, 0.08, 0.02]``.  The 20 % test set is balanced, so the
    tail and rare-class errors expose how much the fit favours the bulk.  The
    thresholds are deterministic for seed 0 with generous margins; the test is
    not a claim about large-scale benchmarks.
    """
    n_features, epochs, seed = 8, 120, 0
    X_train, y_train = _imbalanced_class_data(2000, seed=seed, n_features=n_features)
    X_test, y_test = _imbalanced_class_data(
        1000,
        seed=99,
        probabilities=(0.2, 0.2, 0.2, 0.2, 0.2),
        n_features=n_features,
    )
    tail = y_test.abs().squeeze(1) >= 1.0
    rare = y_test.abs().squeeze(1) == 2.0

    # Both models use the same encoder initialisation; only the loss and head
    # differ.
    torch.manual_seed(seed)
    vir_encoder = _make_encoder(n_features)
    baseline = _train_mse(X_train, y_train, encoder=vir_encoder, epochs=epochs, seed=seed)

    with torch.no_grad():
        mse_prediction = baseline(X_test)
        residual_std = float((baseline(X_train) - y_train).std())
    mse_std = torch.full_like(mse_prediction, residual_std)
    mse_tail = _tail_mae(mse_prediction, y_test, tail)
    mse_rare = _tail_mae(mse_prediction, y_test, rare)
    mse_all = float((mse_prediction - y_test).abs().mean())
    mse_coverage = float(
        ((y_test - mse_prediction).abs() <= 1.6448536269514722 * mse_std).float().mean()
    )

    regressor = VIRRegressor(
        vir_encoder,
        n_bins=10,
        kernel_sigma=1.0,
        smoothing_shrinkage=10.0,
        epochs=epochs,
        lr=3e-3,
        batch_size=64,
        seed=seed,
        inference_smoothing="two_pass",
    ).fit(X_train, y_train)
    prediction, std = regressor.predict(X_test, return_std=True)
    lower, upper = regressor.predict_interval(X_test, confidence=0.9)
    vir_tail = _tail_mae(prediction, y_test, tail)
    vir_rare = _tail_mae(prediction, y_test, rare)
    vir_all = float((prediction - y_test).abs().mean())
    vir_coverage = float(((y_test >= lower) & (y_test <= upper)).float().mean())

    # Tail / rare-group accuracy.
    assert mse_tail > 0.40  # the imbalance really hurts the plain-MSE tail
    assert vir_tail < mse_tail - 0.02
    assert vir_rare < mse_rare - 0.02
    assert vir_all < mse_all

    # Calibration: the evidential intervals are much closer to nominal than a
    # single homoscedastic Gaussian fitted to the MSE residuals.  Both models
    # under-cover at 90 % (a known evidential-regression limitation).
    assert vir_coverage > mse_coverage + 0.05
    assert abs(vir_coverage - 0.9) < abs(mse_coverage - 0.9)
    assert bool((std > 0.0).all())


# ── validation ─────────────────────────────────────────────────────────
def test_vir_loss_validation() -> None:
    with pytest.raises(ValueError, match="4-tuple"):
        VIRLoss(prior=(0.0, 1.0, 2.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alpha0"):
        VIRLoss(prior=(0.0, 1.0, 1.0, 0.1))
    with pytest.raises(ValueError, match="nu0"):
        VIRLoss(prior=(0.0, 0.0, 2.0, 0.1))
    with pytest.raises(ValueError, match="beta0"):
        VIRLoss(prior=(0.0, 1.0, 2.0, -0.1))
    with pytest.raises(ValueError, match="reg_weight"):
        VIRLoss(reg_weight=-1.0)
    with pytest.raises(ValueError, match="n_bins"):
        VIRLoss(n_bins=1)
    with pytest.raises(ValueError, match="kernel_sigma"):
        VIRLoss(kernel_sigma=0.0)
    with pytest.raises(ValueError, match="max_importance_weight"):
        VIRLoss(max_importance_weight=0.5)
    with pytest.raises(ValueError, match="reduction"):
        VIRLoss(reduction="median")

    loss_fn = VIRLoss(n_bins=4)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        loss_fn(torch.randn(4, 4), torch.randn(4))
    with pytest.raises(ValueError, match="single regression target"):
        loss_fn(torch.randn(4, 3), torch.randn(4, 2))
    with pytest.raises(ValueError, match="same number of samples"):
        loss_fn(torch.randn(4, 3), torch.randn(5))
    with pytest.raises(RuntimeError, match="fit"):
        loss_fn.bin_index(torch.randn(3))

    no_targets = VIRLoss(n_bins=4)
    with pytest.raises(ValueError, match="finite"):
        no_targets.fit(torch.tensor([float("nan"), float("inf")]))


def test_vir_regressor_validation() -> None:
    encoder = _make_encoder(3)
    with pytest.raises(TypeError, match="nn.Module"):
        VIRRegressor(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="epochs"):
        VIRRegressor(encoder, epochs=0)
    with pytest.raises(ValueError, match="batch_size"):
        VIRRegressor(encoder, batch_size=0)
    with pytest.raises(ValueError, match="feature_dim"):
        VIRRegressor(encoder, feature_dim=0)
    with pytest.raises(ValueError, match="latent_dim"):
        VIRRegressor(encoder, latent_dim=0)
    with pytest.raises(ValueError, match="inference_smoothing"):
        VIRRegressor(encoder, inference_smoothing="nope")
    with pytest.raises(ValueError, match="n_refine"):
        VIRRegressor(encoder, n_refine=-1)
    with pytest.raises(ValueError, match="smoothing_shrinkage"):
        VIRRegressor(encoder, smoothing_shrinkage=-1.0)
    with pytest.raises(ValueError, match="kl_weight"):
        VIRRegressor(encoder, kl_weight=-1.0)
    with pytest.raises(ValueError, match="lr"):
        VIRRegressor(encoder, lr=0.0)
    with pytest.raises(ValueError, match="weight_decay"):
        VIRRegressor(encoder, weight_decay=-1.0)

    regressor = VIRRegressor(encoder, epochs=1)
    with pytest.raises(RuntimeError, match="fit"):
        regressor.predict(torch.randn(3, 3))
    with pytest.raises(ValueError, match="confidence"):
        VIRRegressor(encoder, epochs=1).predict_interval(torch.randn(2, 3), confidence=1.0)

    X, y = _imbalanced_class_data(40, seed=0, n_features=3)
    with pytest.raises(ValueError, match="2-D"):
        regressor.fit(X[:, 0], y)
    with pytest.raises(ValueError, match="same number of samples"):
        regressor.fit(X, y[:-1])
    with pytest.raises(ValueError, match="NaN"):
        bad = X.clone()
        bad[0, 0] = float("nan")
        regressor.fit(bad, y)
    with pytest.raises(ValueError, match="NaN"):
        bad_y = y.clone()
        bad_y[0, 0] = float("nan")
        regressor.fit(X, bad_y)
    with pytest.raises(ValueError, match="y must have shape"):
        regressor.fit(X, torch.randn(X.shape[0], 2))
    with pytest.raises(TypeError, match="float32/float64"):
        regressor.fit(X.to(torch.float16), y.to(torch.float16))
