import pytest
import torch

from torchregress.losses.gaussian_wasserstein import (
    GaussianWassersteinBoundLoss,
    gaussian_wasserstein_bound_loss,
    symmetric_spd_matrix_sqrt,
)
from torchregress.losses.loss_registry import create_loss_from_config


def test_zero_loss_sqrt_identity() -> None:
    b, d = 4, 3
    mu = torch.randn(b, d)
    # Pin ``torch.eye`` to ``mu`` so the fixture doesn't implicitly rely on
    # the loss module handling dtype/device of input fixtures internally.
    s = torch.eye(d, device=mu.device, dtype=mu.dtype).expand(b, d, d).clone()
    fn = GaussianWassersteinBoundLoss(covariance_parameterization="sqrt", reduction="mean")
    out = fn(mu, mu, s, s)
    assert out.item() == 0.0


def test_zero_loss_diagonal() -> None:
    mu = torch.tensor([[1.0, 2.0], [0.0, -1.0]])
    v = torch.tensor([[0.25, 1.0], [0.5, 2.0]])
    fn = GaussianWassersteinBoundLoss(covariance_parameterization="diagonal", reduction="mean")
    torch.testing.assert_close(fn(mu, mu, v, v), torch.zeros(()))


def test_diagonal_matches_hand() -> None:
    mu_p = torch.tensor([[0.0, 1.0]])
    mu_t = torch.tensor([[1.0, 0.0]])
    vp = torch.tensor([[4.0, 9.0]])
    vt = torch.tensor([[1.0, 1.0]])
    fn = GaussianWassersteinBoundLoss(
        covariance_parameterization="diagonal",
        mean_weight=1.0,
        covariance_weight=1.0,
        reduction="mean",
    )
    mean_term = ((mu_p - mu_t) ** 2).sum(dim=-1)
    cov_term = ((vp.sqrt() - vt.sqrt()) ** 2).sum(dim=-1)
    expected = (mean_term + cov_term).mean()
    torch.testing.assert_close(fn(mu_p, mu_t, vp, vt), expected)


def test_covariance_matches_cholesky_path() -> None:
    torch.manual_seed(0)
    b, d = 3, 2
    mu = torch.randn(b, d)
    spd = torch.randn(b, d, d)
    spd = spd @ spd.transpose(-1, -2) + 0.5 * torch.eye(
        d, device=spd.device, dtype=spd.dtype
    ).expand(b, d, d)
    scale_tril = torch.linalg.cholesky(spd)
    fn_cov = GaussianWassersteinBoundLoss(
        covariance_parameterization="covariance", reduction="mean"
    )
    fn_ch = GaussianWassersteinBoundLoss(covariance_parameterization="cholesky", reduction="mean")
    torch.testing.assert_close(
        fn_cov(mu, mu, spd, spd),
        fn_ch(mu, mu, scale_tril, scale_tril),
        rtol=1e-4,
        atol=1e-5,
    )


def test_full_covariance_hand_2d() -> None:
    torch.manual_seed(1)
    mu_p = torch.zeros(1, 2)
    mu_t = torch.tensor([[1.0, -1.0]])
    a = torch.tensor([[[2.0, 0.5], [0.5, 1.0]]])
    b = torch.tensor([[[1.5, 0.0], [0.0, 0.5]]])
    jitter = 1e-5
    fn = GaussianWassersteinBoundLoss(covariance_parameterization="covariance", jitter=jitter)
    # Pin ``torch.eye`` to ``a`` so the fixture doesn't implicitly rely on
    # the loss module handling dtype/device of input fixtures internally.
    eye = torch.eye(2, device=a.device, dtype=a.dtype).view(1, 2, 2)
    sa = symmetric_spd_matrix_sqrt(a + jitter * eye, eps=fn.eps)
    sb = symmetric_spd_matrix_sqrt(b + jitter * eye, eps=fn.eps)
    mean_term = ((mu_p - mu_t) ** 2).sum(dim=-1)
    cov_term = ((sa - sb) ** 2).sum(dim=(-2, -1))
    torch.testing.assert_close(
        fn(mu_p, mu_t, a, b),
        (mean_term + cov_term).mean(),
    )


def test_non_negative() -> None:
    torch.manual_seed(2)
    mu_p, mu_t = torch.randn(5, 3), torch.randn(5, 3)
    sig_p, sig_t = torch.randn(5, 3, 3), torch.randn(5, 3, 3)
    sig_p = sig_p @ sig_p.transpose(-1, -2) + 0.2 * torch.eye(
        3, device=sig_p.device, dtype=sig_p.dtype
    )
    sig_t = sig_t @ sig_t.transpose(-1, -2) + 0.2 * torch.eye(
        3, device=sig_t.device, dtype=sig_t.dtype
    )
    bs = sig_p.shape[0]
    for mode in ("diagonal", "covariance", "cholesky", "sqrt"):
        if mode == "diagonal":
            vp = torch.diagonal(sig_p, dim1=-2, dim2=-1).abs() + 0.1
            vt = torch.diagonal(sig_t, dim1=-2, dim2=-1).abs() + 0.1
            pc, tc = vp, vt
        elif mode == "cholesky":
            pc = torch.linalg.cholesky(sig_p)
            tc = torch.linalg.cholesky(sig_t)
        elif mode == "sqrt":
            j = 1e-4
            eye = torch.eye(3, device=sig_p.device, dtype=sig_p.dtype).expand(bs, 3, 3)
            pc = symmetric_spd_matrix_sqrt(sig_p + j * eye, eps=1e-8)
            tc = symmetric_spd_matrix_sqrt(sig_t + j * eye, eps=1e-8)
        else:
            pc, tc = sig_p, sig_t
            loss_fn = GaussianWassersteinBoundLoss(covariance_parameterization=mode, jitter=1e-4)
            assert loss_fn(mu_p, mu_t, pc, tc).item() >= 0.0


def test_gradients_finite() -> None:
    d = 2
    mu_p = torch.randn(2, d, requires_grad=True)
    mu_t = torch.randn(2, d)
    raw = torch.randn(2, d, d, requires_grad=True)
    sig_p = raw @ raw.transpose(-1, -2) + 0.5 * torch.eye(
        d, device=raw.device, dtype=raw.dtype
    ).expand(2, d, d)
    sig_t = torch.eye(d, device=raw.device, dtype=raw.dtype).expand(2, d, d) * 0.3
    fn = GaussianWassersteinBoundLoss(covariance_parameterization="covariance", reduction="mean")
    loss = fn(mu_p, mu_t, sig_p, sig_t)
    loss.backward()
    assert torch.isfinite(mu_p.grad).all()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_reduction_none() -> None:
    mu = torch.randn(3, 2)
    v = torch.ones_like(mu) * 0.25
    fn = GaussianWassersteinBoundLoss(covariance_parameterization="diagonal", reduction="none")
    out = fn(mu, mu, v, v)
    assert out.shape == (3,)


def test_mask_and_weights() -> None:
    mu = torch.randn(4, 2)
    v = torch.ones_like(mu)
    mask = torch.tensor([[True, True], [True, False], [False, True], [True, True]])
    fn = GaussianWassersteinBoundLoss(covariance_parameterization="diagonal", reduction="mean")
    w = torch.tensor([1.0, 2.0, 1.0, 0.5])
    assert torch.isfinite(fn(mu, mu, v, v, mask=mask, weights=w))


def test_bad_parameterization_raises() -> None:
    with pytest.raises(ValueError, match="covariance_parameterization"):
        GaussianWassersteinBoundLoss(covariance_parameterization="invalid")  # type: ignore[arg-type]


def test_mismatched_diagonal_shape_raises() -> None:
    fn = GaussianWassersteinBoundLoss(covariance_parameterization="diagonal")
    with pytest.raises(ValueError, match="diagonal mode"):
        fn(torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(2, 2), torch.ones(2, 3))


def test_bad_matrix_shape_raises() -> None:
    fn = GaussianWassersteinBoundLoss(covariance_parameterization="covariance")
    with pytest.raises(ValueError, match="covariance tensors must end"):
        # ``torch.eye`` is used here purely as a shape-stub for ``pytest.raises``
        # validation: the loss module raises on shape before consuming
        # dtype/device, so the fixture is intentionally unpinned (SKIP per
        # docs/loss_test_coverage.md rationale).
        fn(torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 3, 2), torch.eye(3).expand(2, 3, 3))  # noqa: TOR001


def test_create_loss_from_config() -> None:
    loss = create_loss_from_config({"type": "gaussian_wasserstein_bound", "jitter": 1e-5})
    assert isinstance(loss, GaussianWassersteinBoundLoss)
    assert loss.jitter == 1e-5


def test_functional_matches_class() -> None:
    mu_p = torch.randn(2, 2)
    mu_t = torch.randn(2, 2)
    a = torch.eye(2, device=mu_p.device, dtype=mu_p.dtype).expand(2, 2, 2).clone() * 0.3
    b = torch.eye(2, device=mu_p.device, dtype=mu_p.dtype).expand(2, 2, 2).clone() * 0.7
    kwargs = dict(covariance_parameterization="covariance", reduction="sum", jitter=1e-4)
    m = GaussianWassersteinBoundLoss(**kwargs)
    f = gaussian_wasserstein_bound_loss(mu_p, mu_t, a, b, **kwargs)
    torch.testing.assert_close(f, m(mu_p, mu_t, a, b))


def test_near_singular_covariance_stable() -> None:
    mu = torch.zeros(1, 2)
    # almost rank-deficient
    sig = torch.tensor([[[1e-9, 0.0], [0.0, 1.0]]])
    fn = GaussianWassersteinBoundLoss(covariance_parameterization="covariance", jitter=1e-3)
    out = fn(mu, mu, sig, sig)
    assert torch.isfinite(out).all()
