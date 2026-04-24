import torch
import torch.nn as nn

from torchregress.losses import InputNoiseMDNLoss, NoisyInputPredictor
from torchregress.metrics import distribution_metrics_report


def test_photoz_updates():
    # 1. NoisyInputPredictor
    model = nn.Linear(5, 1)
    predictor = NoisyInputPredictor(model, sigma_x=0.1)
    x = torch.randn(10, 5)
    out = predictor(x)
    assert out.shape == (10, 1)

    # 2. InputNoiseMDNLoss
    model_mdn = nn.Linear(5, 9)  # 3 components, 1 feature
    loss_mdn = InputNoiseMDNLoss(model_mdn, n_components=3, n_features=1, sigma_x=0.1)
    y = torch.randn(10, 1)
    loss_val = loss_mdn(x, y)
    assert loss_val.shape == ()

    # 3. Universal metrics report
    loc = torch.randn(100, 1)
    scale = torch.ones(100, 1) * 0.5
    dist = torch.distributions.Normal(loc, scale)
    report = distribution_metrics_report(dist=dist, y_true=y.repeat(10, 1))
    assert "log_prob" in report
    assert "crps" in report
    assert "pit_chi2" in report

    print("Photo-Z verification successful!")


if __name__ == "__main__":
    test_photoz_updates()
