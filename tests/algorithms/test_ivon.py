import pytest
import torch
import torch.nn as nn

from torchregress.algorithms.ivon import IVON


def test_ivon_initialization():
    model = nn.Linear(10, 1)

    # Test valid initialization
    opt = IVON(
        model.parameters(),
        lr=0.1,
        ess=100.0,
        beta1=0.9,
        beta2=0.999,
        weight_decay=1e-4,
        mc_samples=2,
        hess_approx="price",
    )
    assert opt.mc_samples == 2
    assert opt.hess_approx == "price"

    # Test invalid values trigger ValueErrors
    with pytest.raises(ValueError, match="Invalid learning rate"):
        IVON(model.parameters(), lr=-0.1, ess=100.0)

    with pytest.raises(ValueError, match="Invalid number of MC samples"):
        IVON(model.parameters(), lr=0.1, ess=100.0, mc_samples=0)

    with pytest.raises(ValueError, match="Invalid weight decay"):
        IVON(model.parameters(), lr=0.1, ess=100.0, weight_decay=-1e-4)

    with pytest.raises(ValueError, match="Invalid Hessian initialization"):
        IVON(model.parameters(), lr=0.1, ess=100.0, hess_init=-0.5)

    with pytest.raises(ValueError, match="Invalid effective sample size"):
        IVON(model.parameters(), lr=0.1, ess=-5.0)

    with pytest.raises(ValueError, match="Invalid clipping radius"):
        IVON(model.parameters(), lr=0.1, ess=100.0, clip_radius=0.0)

    with pytest.raises(ValueError, match="Invalid beta1 parameter"):
        IVON(model.parameters(), lr=0.1, ess=100.0, beta1=1.5)

    with pytest.raises(ValueError, match="Invalid beta2 parameter"):
        IVON(model.parameters(), lr=0.1, ess=100.0, beta2=-0.1)

    with pytest.raises(ValueError, match="Invalid hess_approx parameter"):
        IVON(model.parameters(), lr=0.1, ess=100.0, hess_approx="invalid_approx")


def test_ivon_step_price():
    torch.manual_seed(42)
    model = nn.Linear(2, 1)
    # Target function: y = 2*x1 - 3*x2
    # Set weights to something different to test training
    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.fill_(0.0)

    optimizer = IVON(
        model.parameters(),
        lr=0.05,
        ess=200.0,
        hess_init=1.0,
        beta1=0.9,
        beta2=0.999,
        weight_decay=1e-4,
        hess_approx="price",
    )

    # Generate some simple data
    x = torch.randn(64, 2)
    y = 2.0 * x[:, 0:1] - 3.0 * x[:, 1:2]

    initial_params = torch.cat([p.flatten() for p in model.parameters()]).clone()

    # Run optimizer steps
    for _ in range(20):
        # We need to wrap forward pass with sampled_params
        with optimizer.sampled_params(train=True):
            optimizer.zero_grad()
            out = model(x)
            loss = (out - y).pow(2).mean()
            loss.backward()
        optimizer.step()

    final_params = torch.cat([p.flatten() for p in model.parameters()]).clone()

    # Verify parameters have changed / updated
    assert not torch.allclose(initial_params, final_params)

    # Since we train towards target coefficients:
    # model.weight should move towards [2.0, -3.0]
    weights = model.weight.data.squeeze().tolist()
    assert abs(weights[0] - 0.0) > 0.1  # Moved away from 0
    assert abs(weights[1] - 0.0) > 0.1  # Moved away from 0


def test_ivon_step_gradsq():
    torch.manual_seed(42)
    model = nn.Linear(2, 1)

    optimizer = IVON(
        model.parameters(),
        lr=0.05,
        ess=200.0,
        hess_init=1.0,
        beta1=0.9,
        beta2=0.999,
        weight_decay=1e-4,
        hess_approx="gradsq",
    )

    x = torch.randn(64, 2)
    y = 2.0 * x[:, 0:1] - 3.0 * x[:, 1:2]

    initial_params = torch.cat([p.flatten() for p in model.parameters()]).clone()

    # Run steps
    for _ in range(10):
        with optimizer.sampled_params(train=True):
            optimizer.zero_grad()
            out = model(x)
            loss = (out - y).pow(2).mean()
            loss.backward()
        optimizer.step()

    final_params = torch.cat([p.flatten() for p in model.parameters()]).clone()
    assert not torch.allclose(initial_params, final_params)


def test_ivon_inference_sampling():
    model = nn.Linear(5, 2)
    optimizer = IVON(
        model.parameters(),
        lr=0.1,
        ess=100.0,
    )

    x = torch.randn(10, 5)

    # Sample different parameters for inference
    outputs = []
    for _ in range(5):
        with optimizer.sampled_params(train=False):
            with torch.no_grad():
                out = model(x)
                outputs.append(out)

    # Verify that different samples yield different predictions
    # (since weights are sampled from the posterior)
    assert not torch.allclose(outputs[0], outputs[1])
