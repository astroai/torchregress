"""Trace the single-sample NaN/Inf test failure.

Reproduces the exact ``h_loss`` setup from
``tests/losses/test_gaussian.py::TestGaussianLosses::test_gaussian_losses_with_nans_infs``
and prints the intermediate tensors so we can see where the NaN slips in.
"""

import math

import torch

from torchregress.losses.gaussian import GaussianNLLLoss


def main() -> None:
    torch.manual_seed(42)
    batch_size, n_features_diag = 4, 3
    device = "cpu"

    x = torch.randn(batch_size, n_features_diag, device=device)
    x_reconstructed = torch.randn(batch_size, n_features_diag, device=device)
    mask = torch.randint(0, 2, (batch_size, n_features_diag), device=device).bool()

    x_nan = x.clone()
    x_nan[0, 0] = float("nan")
    mask_nan = mask.clone()
    mask_nan[0, 0] = False  # Mask out the NaN

    log_var = torch.zeros_like(x)
    print("x_nan:\n", x_nan)
    print("mask_nan:\n", mask_nan)

    loss_fn = GaussianNLLLoss().to(device)

    # The test calls h_loss_fn((x_nan, log_var), x_reconstructed, mask_nan).
    # 3rd positional -> ``covariance_matrices`` in the new signature.
    # The legacy-mask heuristic should promote it back to ``mask``.
    y_pred = (x_nan, log_var)
    target = x_reconstructed
    covariance_matrices = mask_nan  # what the test passes in slot 3
    mask_arg = None

    from torchregress.losses.gaussian import _is_legacy_mask_argument

    print(
        f"\n_is_legacy_mask_argument(cov={covariance_matrices.dtype}, mask={mask_arg}) "
        f"-> {_is_legacy_mask_argument(covariance_matrices, mask_arg)}"
    )

    if _is_legacy_mask_argument(covariance_matrices, mask_arg):
        print("-> promoted: mask = cov, weights = mask, cov = None")
        mask = covariance_matrices
        mask_arg if isinstance(mask_arg, torch.Tensor) else None
        covariance_matrices = None
    else:
        print("-> NOT promoted: NaN will leak through")

    mean, var = loss_fn._extract_distribution_parameters(y_pred)
    print("\nmean[0,0] is NaN?", torch.isnan(mean[0, 0]).item())
    print("var[0,0]:", var[0, 0].item())

    nll = 0.5 * (
        math.log(2 * math.pi)
        + torch.log(var + loss_fn.eps)
        + (target - mean) ** 2 / (var + loss_fn.eps)
    )
    print("\nnll has NaN?", torch.isnan(nll).any().item())
    print("nll at [0,0] (mask=False):", nll[0, 0].item())
    print("mask[0,0]:", mask[0, 0].item())

    # After _reduce
    if mask is not None:
        loss = nll[mask]
        print("\nloss[mask] numel:", loss.numel())
        print("loss[mask] has NaN?", torch.isnan(loss).any().item())
        print("loss[mask] mean:", loss.mean().item())
    else:
        print("\nNO MASK APPLIED -- nll.mean() =", nll.mean().item())


if __name__ == "__main__":
    main()
