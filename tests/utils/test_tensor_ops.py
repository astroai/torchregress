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
    batched_linalg_solve,
    calculate_gaussian_nll,
    calculate_propagated_variance,
    compute_model_gradients,
    convert_to_tensor,
    ensure_batch_dim,
    masked_mean,
    masked_reduction,
    masked_sum,
    prepare_covariance,
    prepare_cross_covariance,
    prepare_model_input_for_gradients,
    prepare_param,
    prepare_sigma,
    standardize,
    unstandardize,
)

# ═══════════════════════════════════════════════════════════════════════════════
# convert_to_tensor
# ═══════════════════════════════════════════════════════════════════════════════


class TestConvertToTensor:
    def test_numpy_array(self) -> None:
        """numpy arrays are converted to float32 tensors."""
        arr = np.array([1.0, 2.0, 3.0])
        result = convert_to_tensor(arr)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_list(self) -> None:
        """Lists are converted to float32 tensors."""
        result = convert_to_tensor([1, 2, 3])
        assert result.dtype == torch.float32
        assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_float_scalar(self) -> None:
        """Float scalars become 1-element tensors."""
        result = convert_to_tensor(3.14)
        assert result.dtype == torch.float32
        assert result.shape == (1,)
        assert float(result.item()) == pytest.approx(3.14)

    def test_int_scalar(self) -> None:
        """Int scalars become 1-element float32 tensors."""
        result = convert_to_tensor(42)
        assert result.dtype == torch.float32
        assert float(result.item()) == pytest.approx(42.0)

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
# prepare_param
# ═══════════════════════════════════════════════════════════════════════════════


class TestPrepareParam:
    def test_none_uses_default(self) -> None:
        """None param fills with default_value."""
        result = prepare_param(None, n_dims=3, device=torch.device("cpu"))  # type: ignore[arg-type]
        assert result.shape == (3,)
        assert torch.equal(result, torch.ones(3))

    def test_none_custom_default(self) -> None:
        """None param with custom default_value."""
        result = prepare_param(None, n_dims=3, device=torch.device("cpu"), default_value=5.0)  # type: ignore[arg-type]
        assert torch.equal(result, torch.full((3,), 5.0))

    def test_float_scalar(self) -> None:
        """Float scalar fills n_dims."""
        result = prepare_param(2.5, n_dims=4, device=torch.device("cpu"))
        assert result.shape == (4,)
        assert torch.equal(result, torch.full((4,), 2.5))

    def test_int_scalar(self) -> None:
        """Int scalar fills n_dims."""
        result = prepare_param(3, n_dims=4, device=torch.device("cpu"))
        assert torch.equal(result, torch.full((4,), 3.0))

    def test_scalar_tensor_expands(self) -> None:
        """Scalar tensor (1 element) expands to n_dims."""
        t = torch.tensor([7.0])
        result = prepare_param(t, n_dims=5, device=torch.device("cpu"))
        assert result.shape == (5,)
        assert torch.equal(result, torch.full((5,), 7.0))

    def test_matching_tensor_reshapes(self) -> None:
        """Tensor with numel == n_dims is reshaped."""
        t = torch.tensor([1.0, 2.0, 3.0])
        result = prepare_param(t, n_dims=3, device=torch.device("cpu"))
        assert result.shape == (3,)
        assert torch.equal(result, t)

    def test_wrong_shape_raises(self) -> None:
        """Tensor with wrong number of elements raises ValueError."""
        t = torch.randn(2, 3)
        with pytest.raises(ValueError, match="doesn't match required size"):
            prepare_param(t, n_dims=5, device=torch.device("cpu"))

    def test_type_error(self) -> None:
        """Non-float, non-tensor raises TypeError."""
        with pytest.raises(TypeError, match="Parameter must be float or tensor"):
            prepare_param("bad", n_dims=3, device=torch.device("cpu"))  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════════
# prepare_sigma
# ═══════════════════════════════════════════════════════════════════════════════


class TestPrepareSigma:
    def test_default_zero(self) -> None:
        """Default behavior fills with zeros."""
        result = prepare_sigma(None, n_dims=3, device=torch.device("cpu"))  # type: ignore[arg-type]
        assert torch.equal(result, torch.zeros(3))

    def test_default_nonzero(self) -> None:
        """default_zero=False fills with ones."""
        result = prepare_sigma(None, n_dims=3, device=torch.device("cpu"), default_zero=False)  # type: ignore[arg-type]
        assert torch.equal(result, torch.ones(3))

    def test_float_scalar(self) -> None:
        """Float scalar sigma fills n_dims."""
        result = prepare_sigma(0.5, n_dims=4, device=torch.device("cpu"))
        assert torch.equal(result, torch.full((4,), 0.5))


# ═══════════════════════════════════════════════════════════════════════════════
# prepare_covariance
# ═══════════════════════════════════════════════════════════════════════════════


class TestPrepareCovariance:
    def test_none_returns_identity(self) -> None:
        """None returns identity matrix."""
        result = prepare_covariance(None, n_dims=3, device=torch.device("cpu"))  # type: ignore[arg-type]
        assert result.shape == (3, 3)
        # RESULT compare: pin ``torch.eye`` to ``result`` so the fixture doesn't
        # implicitly rely on the function module handling dtype/device of input
        # fixtures internally.
        assert torch.equal(result, torch.eye(3, device=result.device, dtype=result.dtype))

    def test_float_scalar(self) -> None:
        """Float scalar produces scaled identity."""
        result = prepare_covariance(2.0, n_dims=3, device=torch.device("cpu"))
        assert torch.equal(result, torch.eye(3, device=result.device, dtype=result.dtype) * 2.0)

    def test_int_scalar(self) -> None:
        """Int scalar produces scaled identity."""
        result = prepare_covariance(3, n_dims=2, device=torch.device("cpu"))
        assert torch.equal(result, torch.eye(2, device=result.device, dtype=result.dtype) * 3)

    def test_scalar_tensor(self) -> None:
        """1-element tensor produces scaled identity."""
        t = torch.tensor([4.0])
        result = prepare_covariance(t, n_dims=2, device=torch.device("cpu"))
        assert torch.equal(result, torch.eye(2, device=result.device, dtype=result.dtype) * 4.0)

    def test_1d_diagonal(self) -> None:
        """1D tensor becomes diagonal matrix."""
        t = torch.tensor([1.0, 2.0, 3.0])
        result = prepare_covariance(t, n_dims=3, device=torch.device("cpu"))
        # ``torch.diag`` does not accept device=/dtype= natively; pin via
        # chained ``.to`` so the fixture doesn't implicitly rely on the
        # function module handling dtype/device of input fixtures internally.
        expected = torch.diag(t).to(device=t.device, dtype=t.dtype)
        assert torch.equal(result, expected)

    def test_1d_wrong_shape_raises(self) -> None:
        """1D tensor with wrong size raises ValueError."""
        t = torch.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="doesn't match required dimensions"):
            prepare_covariance(t, n_dims=3, device=torch.device("cpu"))

    def test_2d_symmetric(self) -> None:
        """2D symmetric matrix returned as-is."""
        cov = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
        result = prepare_covariance(cov, n_dims=2, device=torch.device("cpu"))
        assert torch.allclose(result, cov)

    def test_2d_non_symmetric_symmetrized(self) -> None:
        """Non-symmetric 2D matrix is symmetrized with warning."""
        cov = torch.tensor([[2.0, 1.5], [0.5, 1.0]])
        with pytest.warns(UserWarning, match="not symmetric"):
            result = prepare_covariance(cov, n_dims=2, device=torch.device("cpu"))
        expected = (cov + cov.t()) / 2
        assert torch.allclose(result, expected)

    def test_2d_wrong_shape_raises(self) -> None:
        """2D matrix with wrong shape raises ValueError."""
        # ``torch.eye`` is used here purely as a shape-stub for ``pytest.raises``
        # validation: the function raises before consuming dtype/device, so the
        # fixture is intentionally unpinned (SKIP per docs/loss_test_coverage.md
        # rationale).
        cov = torch.eye(3)  # noqa: TOR001
        with pytest.raises(ValueError, match="doesn't match required shape"):
            prepare_covariance(cov, n_dims=2, device=torch.device("cpu"))

    def test_ndim_gt_2_raises(self) -> None:
        """Tensor with >2 dimensions raises ValueError."""
        cov = torch.randn(2, 2, 2)
        with pytest.raises(ValueError, match="must be scalar, vector or matrix"):
            prepare_covariance(cov, n_dims=2, device=torch.device("cpu"))

    def test_type_error(self) -> None:
        """Non-float, non-tensor raises TypeError."""
        with pytest.raises(TypeError, match="Covariance must be float or tensor"):
            prepare_covariance("bad", n_dims=2, device=torch.device("cpu"))  # type: ignore[arg-type]


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
# batched_linalg_solve
# ═══════════════════════════════════════════════════════════════════════════════


class TestBatchedLinalgSolve:
    def test_normal_case(self) -> None:
        """Normal well-conditioned solve."""
        # No co-built tensor precedes ``A`` here — fall back to legacy device/dtype
        # so the fixture doesn't implicitly rely on the function module's defaults.
        A = torch.eye(
            3, device=torch.device("cpu"), dtype=torch.get_default_dtype()
        ) + 0.1 * torch.randn(3, 3)
        A = A @ A.t()  # make pos-def
        b = torch.randn(3, 2)
        result = batched_linalg_solve(A, b)
        expected = torch.linalg.solve(A, b)
        assert torch.allclose(result, expected)

    def test_singular_with_jitter(self) -> None:
        """Singular matrix recovers with ridge jitter."""
        A = torch.zeros(3, 3)
        b = torch.ones(3, 1)
        result = batched_linalg_solve(A, b, ridge_factor=0.1)
        assert torch.isfinite(result).all()

    def test_severely_singular_falls_back_to_pinv(self) -> None:
        """Severely ill-conditioned matrix falls back to pseudoinverse."""
        A = torch.zeros(3, 3)
        b = torch.zeros(3, 1)
        result = batched_linalg_solve(A, b, ridge_factor=0.0)
        # pinv of zero matrix times zero rhs = zero
        assert torch.allclose(result, torch.zeros_like(b), atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# standardize / unstandardize
# ═══════════════════════════════════════════════════════════════════════════════


class TestStandardize:
    def test_compute_from_data(self) -> None:
        """Mean and std are computed from data when not provided."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        z, mean, std = standardize(x)
        assert z.shape == x.shape
        assert torch.allclose(z.mean(dim=0), torch.zeros(2), atol=1e-6)
        assert torch.allclose(z.std(dim=0), torch.ones(2), atol=1e-6)

    def test_with_provided_mean_std(self) -> None:
        """Provided mean and std are used for standardization."""
        x = torch.tensor([[1.0], [2.0], [3.0]])
        mean = torch.tensor([10.0])
        std = torch.tensor([2.0])
        z, _, _ = standardize(x, mean=mean, std=std)
        expected = (x - mean) / std
        assert torch.allclose(z, expected)

    def test_eps_prevents_division_by_zero(self) -> None:
        """eps prevents division by zero for zero-std features."""
        x = torch.tensor([[5.0], [5.0], [5.0]])
        z, mean, std = standardize(x, eps=1e-8)
        assert torch.isfinite(z).all()


class TestUnstandardize:
    def test_roundtrip(self) -> None:
        """standardize then unstandardize recovers original."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        z, mean, std = standardize(x)
        recovered = unstandardize(z, mean, std)
        assert torch.allclose(recovered, x)


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
        """Zero residual gives only the log|Σ| + const term."""
        residuals = torch.zeros(1, 2)
        var = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        result = calculate_gaussian_nll(residuals, var)
        assert torch.isfinite(result).all()


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
