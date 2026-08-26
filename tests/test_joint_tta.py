"""Unit tests for the joint distributional TTA prototype (ICLR MTTA)."""

from __future__ import annotations

import math

import pytest
import torch

from torchregress.test_time.joint_tta import (
    JointDistributionalTTA,
    JointTTAResult,
    gaussian_outputs,
)
from torchregress.test_time.shift_weights import (
    DomainClassifierRatioEstimator,
    OTScoreWeightEstimator,
    estimate_label_shift_weights,
)


class _DiagGaussianMLP(torch.nn.Module):
    """[n, d_in] -> ([n, 2d] = concat(mean, log_var)) diagonal-Gaussian head."""

    def __init__(self, d_in: int, d_out: int, hidden: int = 64) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(hidden, 2 * d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def _train_source_model(
    seed: int, n: int = 400
) -> tuple[_DiagGaussianMLP, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    model = _DiagGaussianMLP(3, 2).eval()
    xs = torch.randn(n, 3)

    def f(x: torch.Tensor) -> torch.Tensor:
        return (x[:, :1] * 1.5 + 0.5 * x[:, 1:2].sin()).expand(x.shape[0], 2).contiguous()

    ys = f(xs)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(300):
        opt.zero_grad()
        mu = model(xs)[:, :2]
        loss = (mu - ys).pow(2).mean()
        loss.backward()
        opt.step()
    return model, xs, ys


def test_gaussian_outputs_contract() -> None:
    model = _DiagGaussianMLP(3, 2).eval()
    x = torch.randn(7, 3)
    mu, var = gaussian_outputs(model, x)
    assert mu.shape == (7, 2) and var.shape == (7, 2)
    assert bool((var > 0).all())
    with pytest.raises(ValueError):
        gaussian_outputs(
            lambda t: torch.zeros(
                4,
            ),
            x,
        )


def test_shift_weight_estimators_sanity() -> None:
    torch.manual_seed(0)
    xs = torch.randn(400, 3)
    xt = torch.randn(600, 3) + 2.0
    est = DomainClassifierRatioEstimator(hidden=(32,), epochs=100).fit(xs, xt)
    w_far = float(est.weights(torch.tensor([[3.0, 3.0, 3.0]])))
    w_home = float(est.weights(torch.zeros(1, 3)))
    assert w_far > 5.0 * w_home
    assert bool((est.weights_for(xs) > 0).all())

    ot = OTScoreWeightEstimator(n_steps=50).fit(xs, xt)
    simplex = ot.weights(xs)
    assert simplex.shape == (400,)
    assert abs(float(simplex.sum()) - 1.0) < 1e-5

    n = 2000
    cls_s = torch.bernoulli(torch.full((n,), 0.9)).long()  # mostly class 0
    src_onehot = torch.nn.functional.one_hot(cls_s, 2).double().numpy()
    cls_t = torch.bernoulli(torch.full((n,), 0.1)).long()  # mostly class 1
    tgt_onehot = torch.nn.functional.one_hot(cls_t, 2).double().numpy()
    w, lse = estimate_label_shift_weights(src_onehot, tgt_onehot)
    assert abs(float(lse.target_prior[0]) - 0.9) < 0.05
    assert w.shape == (n,) and bool((w > 0).all())


def test_adapt_and_calibrate_all_estimator_paths_run() -> None:
    model, xs, ys = _train_source_model(0)
    xt = torch.randn(500, 3) + 0.8
    for we in ("domain_clf", "ot", "label_shift_em"):
        tta = JointDistributionalTTA(alpha=0.1, weight_estimator=we, pseudo_label_rounds=1)
        res = tta.adapt_and_calibrate(model, xs[:320], ys[:320], xt)
        assert isinstance(res, JointTTAResult)
        r = res.conformal.region_radius()
        assert r > 0.0
        lo, hi = res.diagnostics["coverage_bounds"]
        assert lo < hi
        iv = tta.predict_intervals(res, torch.randn(50, 3) + 0.8)
        assert set(iv) == {"mean", "lower", "upper", "radius"}
        assert bool((iv["upper"] >= iv["lower"]).all())


def test_synthetic_covariate_shift_improves_coverage() -> None:
    """Rotated + scaled covariate shift on synthetic d=2.

    Vanilla split CP must under-cover by more than 0.05 under the shift, and
    the joint-TTA weighted region must land inside its own NexCP bounds.
    """
    alpha = 0.1

    def make_split(seed: int):
        g = torch.Generator().manual_seed(seed)
        xs = torch.randn(900, 3, generator=g)
        theta = math.pi / 3.0
        rot = torch.tensor(
            [
                [3.0 * math.cos(theta), -math.sin(theta), 0.0],
                [3.0 * math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        xt = torch.randn(700, 3, generator=g) @ rot.T

        def f(x: torch.Tensor) -> torch.Tensor:
            return (x[:, :1].sin() + 0.5 * x[:, 1:2]).expand(x.shape[0], 2).contiguous()

        def noise(x: torch.Tensor) -> torch.Tensor:
            return (0.05 + 0.35 * x[:, :1].abs()) * torch.randn(x.shape[0], 2, generator=g)

        ys = f(xs) + noise(xs)
        yt_clean = f(xt) + noise(xt)
        return xs, ys, xt, yt_clean

    vanilla_covs = []
    joint_covs = []
    bounds_lo = []
    for seed in range(3):
        model, xs, ys = _train_source_model(seed)
        xs, ys, xt, yt = make_split(seed + 100)
        Xc, yc = xs[:640], ys[:640]

        # --- vanilla unweighted split CP on absolute residuals ---
        with torch.no_grad():
            res_cal = model(Xc)[:, :2]
            scores = (yc - res_cal).norm(dim=-1) ** 2  # chi2-like joint score
        from torchregress.losses.conformal import finite_sample_quantile

        q_vanilla = float(finite_sample_quantile(scores, alpha))
        with torch.no_grad():
            s_new = ((yt - model(xt)[:, :2]).norm(dim=-1)) ** 2
        vanilla_covs.append(float((s_new <= q_vanilla).float().mean()))

        # --- joint TTA (weights-only ablation keeps the model fixed) ---
        tta = JointDistributionalTTA(
            alpha=alpha,
            weight_estimator="domain_clf",
            align_features=False,
            pseudo_label_rounds=0,
        )
        result = tta.adapt_and_calibrate(model, Xc, yc, xt)
        covered = []
        for j in range(0, yt.shape[0], 8):  # covers() in chunks keeps memory flat
            xb, yb = xt[j : j + 8], yt[j : j + 8]
            with torch.no_grad():
                out = result.adapted_model(xb)
                mu_b, var_b = out[:, :2], out[:, 2:].exp().clamp_min(1e-6)
            covered.append(result.conformal.covers(mu_b, var_b, yb))
        joint_covs.append(float(torch.cat(covered).float().mean()))
        lo, hi = result.diagnostics["coverage_bounds"]
        bounds_lo.append(lo)

    mean_vanilla = sum(vanilla_covs) / len(vanilla_covs)
    assert mean_vanilla < alpha + 1.0 - 0.05, vanilla_covs  # under-covers by >5pts
    for cov, lo in zip(joint_covs, bounds_lo):
        assert cov >= lo, (cov, lo)


def test_pseudo_label_round_zero_is_baseline() -> None:
    model, xs, ys = _train_source_model(1)
    params_before = [p.clone() for p in model.head.parameters()]
    tta = JointDistributionalTTA(pseudo_label_rounds=0)
    res = tta.adapt_and_calibrate(model, xs[:320], ys[:320], torch.randn(200, 3))
    for before, after in zip(params_before, model.head.parameters()):
        assert torch.equal(before, after)
    assert "pseudo_label_yield" not in res.diagnostics
