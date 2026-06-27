import torch

from torchregress.algorithms.adaptive_prior_vi import (
    AdaptivePriorGuide,
    AdaptivePriorNetwork,
    VIDSRegressor,
)
from torchregress.prediction import PredictiveBatch


def test_adaptive_prior_guide_and_network() -> None:
    in_features = 4
    target_dim = 1
    param_dim = 6
    batch_size = 3

    guide = AdaptivePriorGuide(in_features, target_dim, param_dim)
    prior_net = AdaptivePriorNetwork(in_features, param_dim)

    context_X = torch.randn(2 * in_features)
    context_Y = torch.randn(2 * target_dim)
    x_query = torch.randn(batch_size, in_features)

    post_mean, post_log_var = guide(context_X, context_Y)
    assert post_mean.shape == (param_dim,)
    assert post_log_var.shape == (param_dim,)

    prior_mean, prior_log_var = prior_net(context_X, x_query)
    assert prior_mean.shape == (batch_size, param_dim)
    assert prior_log_var.shape == (batch_size, param_dim)


def test_vids_regressor() -> None:
    in_features = 3
    target_dim = 1
    N_train = 30

    x_train = torch.randn(N_train, in_features)
    y_train = torch.randn(N_train, target_dim)

    model = VIDSRegressor(
        in_features=in_features,
        target_dim=target_dim,
        hidden_dim=16,
    )

    # Fit the regressor
    model.fit(
        x_train,
        y_train,
        n_environments=8,
        bootstrap_fraction=0.3,
        epochs=3,
        lr=1e-2,
    )

    assert model.is_fitted
    assert model.x_train_mean is not None
    assert model.x_train_std is not None

    # Predict
    x_test = torch.randn(5, in_features)
    pred = model.predict_distribution(x_test, n_samples=12)

    assert isinstance(pred, PredictiveBatch)
    assert pred.mean.shape == (5, target_dim)
    assert pred.std.shape == (5, target_dim)
    assert pred.samples.shape == (5, 12)  # [B, n_samples] for 1D targets
    assert "epistemic_variance" in pred.extra
    assert "aleatoric_variance" in pred.extra
    assert pred.extra["epistemic_variance"].shape == (5, target_dim)
    assert pred.extra["aleatoric_variance"].shape == (5, target_dim)
