"""
Unit tests for torchregress.utils.tensor_ops.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from torchregress.utils.tensor_ops import (
    apply_mask,
    calculate_gaussian_nll,
    calculate_propagated_variance,
    compute_model_gradients,
    convert_to_tensor,
    ensure_batch_dim,
    masked_mean,
    masked_reduction,
    masked_sum,
    prepare_cross_covariance,
    prepare_model_input_for_gradients,
)

# ═══════════════════════════════════════════════════════════════════════════════
# convert_to_tensor
# ═══════════════════════════════════════════════════════════════════════════════


class TestConvertToTensor:
    def test_numpy_array(self) -> None:
        """numpy arrays keep their dtype and are copied, not aliased."""
        arr = np.array([1.0, 2.0, 3.0])  # float64 preserved (TR-MET-14)
        result = convert_to_tensor(arr)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float64
        assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_numpy_float32_kept(self) -> None:
        """float32 numpy arrays are not upcast."""
        result = convert_to_tensor(np.array([1.0], dtype=np.float32))
        assert result.dtype == torch.float32

    def test_numpy_copied_not_aliased(self) -> None:
        """Mutating the returned tensor must not touch the caller's array."""
        arr = np.ones(3)
        t = convert_to_tensor(arr)
        t.mul_(-1)
        assert np.all(arr == 1.0)

    def test_list(self) -> None:
        """Lists of ints are promoted to float32; float lists keep precision."""
        result = convert_to_tensor([1, 2, 3])
        assert result.dtype == torch.float32
        assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_float_list_preserves_precision(self) -> None:
        result = convert_to_tensor([1.5, 2.5])
        assert result.dtype == torch.get_default_dtype()

    def test_float64_numpy_keeps_precision(self) -> None:

        result = convert_to_tensor(np.array([1.5, np.pi]))
        assert result.dtype == torch.float64

    def test_float_scalar(self) -> None:
        """Float scalars become 0-dim tensors at the default dtype."""
        result = convert_to_tensor(3.14)
        assert result.dtype == torch.get_default_dtype()

    def test_int_scalar(self) -> None:
        """Int scalars become 0-dim float32 tensors."""
        result = convert_to_tensor(42)
        assert result.dtype == torch.float32
        assert result.dim() == 0
        assert float(result.item()) == pytest.approx(42.0)

    def test_dtype_and_device_kwargs(self) -> None:
        """dtype/device kwargs are applied after conversion."""
        result = convert_to_tensor(np.array([1.0, 2.0]), dtype=torch.float32)
        assert result.dtype == torch.float32
        assert result.device.type == "cpu"

    def test_tensor_passthrough(self) -> None:
        """Tensors are returned as-is."""
        x = torch.randn(3, 4)
        result = convert_to_tensor(x)
        assert result is x

    def test_type_error(self) -> None:
        """Unsupported types raise TypeError."""
        with pytest.raises(TypeError, match="Cannot convert"):
            convert_to_tensor("hello")  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════════
# ensure_batch_dim
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnsureBatchDim:
    def test_1d_gets_batch_dim(self) -> None:
        """1D tensor gets a leading batch dimension."""
        x = torch.tensor([1.0, 2.0, 3.0])
        result = ensure_batch_dim(x)
        assert result.shape == (1, 3)

    def test_2d_unchanged(self) -> None:
        """2D tensor is returned unchanged."""
        x = torch.randn(8, 3)
        result = ensure_batch_dim(x)
        assert result.shape == (8, 3)
        assert result is x


# ═══════════════════════════════════════════════════════════════════════════════
# apply_mask
# ═══════════════════════════════════════════════════════════════════════════════


class TestApplyMask:
    def test_no_mask_returns_original(self) -> None:
        """None mask returns original tensor."""
        x = torch.tensor([1.0, 2.0, 3.0])
        result = apply_mask(x, None)
        assert result is x

    def test_with_mask_zeros_masked(self) -> None:
        """Masked positions are set to zero."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, False, True, False])
        result = apply_mask(x, mask)
        expected = torch.tensor([1.0, 0.0, 3.0, 0.0])
        assert torch.equal(result, expected)

    def test_all_false_mask(self) -> None:
        """All-false mask zeros everything."""
        x = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([False, False, False])
        result = apply_mask(x, mask)
        assert torch.equal(result, torch.zeros(3))


# ═══════════════════════════════════════════════════════════════════════════════
# masked_reduction
# ═══════════════════════════════════════════════════════════════════════════════


class TestMaskedReduction:
    # --- no mask ---
    def test_no_mask_mean(self) -> None:
        """No mask + mean reduction returns global mean."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert float(masked_reduction(x, None, "mean").item()) == pytest.approx(2.5)

    def test_no_mask_sum(self) -> None:
        """No mask + sum reduction returns global sum."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert float(masked_reduction(x, None, "sum").item()) == pytest.approx(10.0)

    def test_no_mask_max(self) -> None:
        """No mask + max reduction returns global max."""
        x = torch.tensor([1.0, 5.0, 2.0, 4.0])
        assert float(masked_reduction(x, None, "max").item()) == pytest.approx(5.0)

    def test_no_mask_min(self) -> None:
        """No mask + min reduction returns global min."""
        x = torch.tensor([1.0, 5.0, 2.0, 4.0])
        assert float(masked_reduction(x, None, "min").item()) == pytest.approx(1.0)

    def test_no_mask_none_returns_original(self) -> None:
        """No mask + none reduction returns tensor unchanged."""
        x = torch.tensor([1.0, 2.0, 3.0])
        result = masked_reduction(x, None, "none")
        assert result is x

    def test_no_mask_unknown_raises(self) -> None:
        """Unknown reduction raises ValueError."""
        x = torch.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="Unknown reduction"):
            masked_reduction(x, None, "median")  # type: ignore[arg-type]

    # --- with mask ---
    def test_mask_mean(self) -> None:
        """Masked mean ignores masked positions."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, False, True, False])
        assert float(masked_reduction(x, mask, "mean").item()) == pytest.approx(2.0)

    def test_mask_sum(self) -> None:
        """Masked sum ignores masked positions."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, False, True, False])
        assert float(masked_reduction(x, mask, "sum").item()) == pytest.approx(4.0)

    def test_mask_max(self) -> None:
        """Masked max ignores masked positions."""
        x = torch.tensor([10.0, 2.0, 3.0, 100.0])
        mask = torch.tensor([True, False, True, False])
        assert float(masked_reduction(x, mask, "max").item()) == pytest.approx(10.0)

    def test_mask_min(self) -> None:
        """Masked min ignores masked positions."""
        x = torch.tensor([10.0, 2.0, 3.0, 100.0])
        mask = torch.tensor([True, False, True, False])
        assert float(masked_reduction(x, mask, "min").item()) == pytest.approx(3.0)

    def test_mask_unknown_raises(self) -> None:
        """Unknown reduction with mask raises ValueError."""
        x = torch.tensor([1.0, 2.0])
        mask = torch.tensor([True, False])
        with pytest.raises(ValueError, match="Unknown reduction"):
            masked_reduction(x, mask, "median")  # type: ignore[arg-type]

    def test_mask_all_false_mean(self) -> None:
        """All-false mask mean returns 0 (divide by zero avoided)."""
        x = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([False, False, False])
        result = masked_reduction(x, mask, "mean")
        assert float(result.item()) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# masked_mean / masked_sum
# ═══════════════════════════════════════════════════════════════════════════════


class TestMaskedMean:
    def test_no_mask(self) -> None:
        """No mask — standard mean."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = masked_mean(x, None)
        assert float(result.item()) == pytest.approx(2.5)

    def test_with_mask(self) -> None:
        """Masked mean over all elements."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([[True, False], [True, True]])
        result = masked_mean(x, mask)
        assert float(result.item()) == pytest.approx(8.0 / 3.0)

    def test_with_dim(self) -> None:
        """Masked mean along a dimension."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([[True, False], [True, True]])
        result = masked_mean(x, mask, dim=1)
        expected = torch.tensor([1.0, 3.5])
        assert torch.allclose(result, expected)

    def test_keepdim(self) -> None:
        """Masked mean with keepdim=True."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([[True, False], [True, True]])
        result = masked_mean(x, mask, dim=1, keepdim=True)
        assert result.shape == (2, 1)


class TestMaskedSum:
    def test_no_mask(self) -> None:
        """No mask — standard sum."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = masked_sum(x, None)
        assert float(result.item()) == pytest.approx(10.0)

    def test_with_mask(self) -> None:
        """Masked sum."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([[True, False], [True, True]])
        result = masked_sum(x, mask)
        assert float(result.item()) == pytest.approx(8.0)

    def test_with_dim(self) -> None:
        """Masked sum along a dimension."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([[True, False], [True, True]])
        result = masked_sum(x, mask, dim=1)
        expected = torch.tensor([1.0, 7.0])
        assert torch.allclose(result, expected)


# ═══════════════════════════════════════════════════════════════════════════════
# prepare_cross_covariance
# ═══════════════════════════════════════════════════════════════════════════════


class TestPrepareCrossCovariance:
    def test_none_returns_zeros(self) -> None:
        """None returns zero matrix of correct shape."""
        result = prepare_cross_covariance(None, n_dims_x=3, n_dims_y=2, device=torch.device("cpu"))  # type: ignore[arg-type]
        assert result.shape == (2, 3)
        assert torch.equal(result, torch.zeros(2, 3))

    def test_correct_shape(self) -> None:
        """Correct-shape tensor returned as-is."""
        cov = torch.randn(2, 3)
        result = prepare_cross_covariance(cov, n_dims_x=3, n_dims_y=2, device=torch.device("cpu"))
        assert result.shape == (2, 3)
        assert torch.equal(result, cov.to(dtype=torch.float32))

    def test_with_dtype(self) -> None:
        """dtype is applied to the result."""
        cov = torch.randn(2, 3, dtype=torch.float64)
        result = prepare_cross_covariance(
            cov, n_dims_x=3, n_dims_y=2, device=torch.device("cpu"), dtype=torch.float32
        )
        assert result.dtype == torch.float32

    def test_wrong_shape_raises(self) -> None:
        """Wrong shape raises ValueError."""
        cov = torch.randn(2, 3)  # shape (2,3) ≠ required (3,2)
        with pytest.raises(ValueError, match="doesn't match required shape"):
            prepare_cross_covariance(cov, n_dims_x=2, n_dims_y=3, device=torch.device("cpu"))

    def test_type_error(self) -> None:
        """Non-tensor raises TypeError."""
        with pytest.raises(TypeError, match="Cross-covariance must be a tensor"):
            prepare_cross_covariance("bad", n_dims_x=2, n_dims_y=2, device=torch.device("cpu"))  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════════
# prepare_model_input_for_gradients
# ═══════════════════════════════════════════════════════════════════════════════


class TestPrepareModelInputForGradients:
    def test_already_requires_grad(self) -> None:
        """Input that already requires grad is returned as-is."""
        x = torch.randn(4, 3, requires_grad=True)
        result = prepare_model_input_for_gradients(x)
        assert result is x

    def test_no_requires_grad(self) -> None:
        """Input without grad gets requires_grad=True via clone."""
        x = torch.randn(4, 3)
        result = prepare_model_input_for_gradients(x)
        assert result.requires_grad
        assert result is not x
        assert torch.equal(result, x)


# ═══════════════════════════════════════════════════════════════════════════════
# compute_model_gradients
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeModelGradients:
    def test_single_output(self) -> None:
        """Single output returns Jacobian of shape [batch, 1, n_features]."""
        x = torch.randn(4, 3, requires_grad=True)
        # Simple linear function
        w = torch.tensor([1.0, 2.0, 3.0])
        y_pred = (x * w).sum(dim=1, keepdim=True)
        grads = compute_model_gradients(y_pred, x, n_features_y=1)
        assert grads.shape == (4, 1, 3)

    def test_single_output_correct_gradient(self) -> None:
        """Gradient matches analytical for linear function."""
        x = torch.randn(2, 3, requires_grad=True)
        w = torch.tensor([1.0, 2.0, 3.0])
        y_pred = (x * w).sum(dim=1, keepdim=True)
        grads = compute_model_gradients(y_pred, x, n_features_y=1)
        assert grads.shape == (2, 1, 3)
        # Gradient of w·x is w for each sample
        assert torch.allclose(grads[0, 0], w)

    def test_multi_output(self) -> None:
        """Multi-output returns Jacobian of shape [batch, n_features_y, n_features_x]."""
        x = torch.randn(4, 3, requires_grad=True)
        # Simple linear function with 2 outputs
        model = torch.nn.Linear(3, 2)
        y_pred = model(x)
        grads = compute_model_gradients(y_pred, x, n_features_y=2)
        assert grads.shape == (4, 2, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# calculate_gaussian_nll
# ═══════════════════════════════════════════════════════════════════════════════


class TestCalculateGaussianNLL:
    def test_empty_residuals(self) -> None:
        """Empty residuals return empty tensor."""
        residuals = torch.empty(0, 2)
        var = torch.ones(0, 2)
        result = calculate_gaussian_nll(residuals, var)
        assert result.numel() == 0

    def test_diagonal(self) -> None:
        """Diagonal covariance case computes sum of per-dim NLLs."""
        residuals = torch.tensor([[1.0, 0.0]])
        var = torch.tensor([[1.0, 1.0]])
        result = calculate_gaussian_nll(residuals, var)
        # NLL = 0.5 * (log(1) + 1^2/1 + log(1) + 0^2/1) + log(2π) = 0.5*(0+1+0+0) + 0.5*2*log(2π)
        expected = 0.5 + math.log(2 * math.pi)
        assert float(result.item()) == pytest.approx(expected)

    def test_diagonal_batch(self) -> None:
        """Batch of residuals with diagonal variance."""
        residuals = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        var = torch.tensor([[1.0, 1.0], [2.0, 1.0]])
        result = calculate_gaussian_nll(residuals, var)
        assert result.shape == (2,)
        assert torch.isfinite(result).all()

    def test_full_covariance(self) -> None:
        """Full covariance case uses MultivariateNormal."""
        residuals = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        # INPUT builder: pin ``torch.eye`` to ``residuals`` so the fixture doesn't
        # implicitly rely on the function module handling dtype/device of input
        # fixtures internally.
        var = torch.stack(
            [
                torch.eye(2, device=residuals.device, dtype=residuals.dtype),
                2 * torch.eye(2, device=residuals.device, dtype=residuals.dtype),
            ]
        )
        result = calculate_gaussian_nll(residuals, var)
        assert result.shape == (2,)
        assert torch.isfinite(result).all()
        assert float(result[0].item()) != float(result[1].item())

    def test_zero_residual_returns_log_det_term(self) -> None:
        """Zero residual gives only the log|Σ| + const term (full covariance input)."""
        residuals = torch.zeros(1, 2)
        var = torch.stack([2.0 * torch.eye(2), 2.0 * torch.eye(2)])
        result = calculate_gaussian_nll(residuals, var)
        assert torch.isfinite(result).all()

    def test_mismatched_var_shape_raises(self) -> None:
        """A [D, D] var against [B, D] residuals is rejected (TR-LOSS-39)."""
        residuals = torch.zeros(1, 2)
        var = 2.0 * torch.eye(2)
        with pytest.raises(AssertionError, match="calculate_gaussian_nll supports"):
            calculate_gaussian_nll(residuals, var)


# ═══════════════════════════════════════════════════════════════════════════════
# calculate_propagated_variance
# ═══════════════════════════════════════════════════════════════════════════════


class TestCalculatePropagatedVariance:
    def test_basic(self) -> None:
        """grad @ sigma_x @ grad^T."""
        # No co-built tensor precedes ``grad`` here — fall back to legacy
        # device/dtype so the fixture doesn't implicitly rely on the
        # function module's defaults.
        grad = torch.eye(2, device=torch.device("cpu"), dtype=torch.get_default_dtype()).unsqueeze(
            0
        )  # [1, 2, 2]
        sigma_x = torch.eye(2, device=grad.device, dtype=grad.dtype)
        result = calculate_propagated_variance(grad, sigma_x)
        assert result.shape == (1, 2, 2)
        assert torch.allclose(
            result, torch.eye(2, device=result.device, dtype=result.dtype).unsqueeze(0)
        )

    def test_with_sigma_y(self) -> None:
        """Adds sigma_y to propagated variance."""
        grad = torch.eye(2, device=torch.device("cpu"), dtype=torch.get_default_dtype()).unsqueeze(
            0
        )
        sigma_x = torch.eye(2, device=grad.device, dtype=grad.dtype)
        sigma_y = torch.eye(2, device=grad.device, dtype=grad.dtype).unsqueeze(0)
        result = calculate_propagated_variance(grad, sigma_x, sigma_y=sigma_y)
        assert torch.allclose(
            result,
            2 * torch.eye(2, device=result.device, dtype=result.dtype).unsqueeze(0),
        )

    def test_with_sigma_xy(self) -> None:
        """Subtracts cross-covariance terms."""
        grad = torch.eye(2, device=torch.device("cpu"), dtype=torch.get_default_dtype()).unsqueeze(
            0
        )
        sigma_x = torch.eye(2, device=grad.device, dtype=grad.dtype)
        sigma_xy = torch.eye(2, device=grad.device, dtype=grad.dtype).unsqueeze(0)
        result = calculate_propagated_variance(grad, sigma_x, sigma_xy=sigma_xy)
        # grad @ sigma_x @ grad^T - grad @ sigma_xy^T - sigma_xy @ grad^T
        # = I - I - I = -I
        assert torch.allclose(
            result, -torch.eye(2, device=result.device, dtype=result.dtype).unsqueeze(0)
        )

    def test_batch(self) -> None:
        """Batch of gradients."""
        grad = (
            torch.eye(2, device=torch.device("cpu"), dtype=torch.get_default_dtype())
            .unsqueeze(0)
            .expand(4, -1, -1)
        )
        sigma_x = torch.eye(2, device=grad.device, dtype=grad.dtype)
        result = calculate_propagated_variance(grad, sigma_x)
        assert result.shape == (4, 2, 2)

    def test_all_null(self) -> None:
        """All optional noise sources provided."""
        grad = torch.tensor([[[1.0, 0.0]]])  # [1, 1, 2]
        sigma_x = 2.0 * torch.eye(2, device=grad.device, dtype=grad.dtype)
        sigma_y = torch.tensor([[[0.5]]])
        sigma_xy = torch.tensor([[[1.0, 0.0]]])
        result = calculate_propagated_variance(grad, sigma_x, sigma_y=sigma_y, sigma_xy=sigma_xy)
        assert result.shape == (1, 1, 1)
        assert torch.isfinite(result).all()
