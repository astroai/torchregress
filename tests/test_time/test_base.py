"""
Tests for torchregress.test_time.base — AdaptationBatch, Protocols, and helpers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchregress.prediction import PredictiveBatch
from torchregress.test_time.base import (
    AdaptationBatch,
    SupportsPredictiveBatch,
    flatten_adaptation_parameters,
)

# ═══════════════════════════════════════════════════════════════════════════════
# AdaptationBatch
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdaptationBatch:
    """AdaptationBatch — frozen dataclass for unlabeled target-time adaptation."""

    def test_construction_required_field(self) -> None:
        """Construction required field."""
        batch = AdaptationBatch(x=np.array([1.0, 2.0]))
        assert batch.x is not None
        assert batch.predictions is None
        assert batch.representations is None
        assert batch.sigma_x is None

    def test_construction_all_fields(self) -> None:
        """Construction all fields."""
        pb = PredictiveBatch(mean=np.array([1.0, 2.0], dtype=np.float32))
        batch = AdaptationBatch(
            x=np.array([[0.1, 0.2], [0.3, 0.4]]),
            predictions=pb,
            representations=np.array([[0.5, 0.6], [0.7, 0.8]]),
            sigma_x=np.array([[0.1], [0.2]]),
        )
        assert batch.x is not None
        assert batch.predictions is pb
        assert isinstance(batch.representations, np.ndarray)
        assert isinstance(batch.sigma_x, np.ndarray)

    def test_frozen_prevents_mutation(self) -> None:
        """Frozen prevents mutation."""
        batch = AdaptationBatch(x=np.array([1.0, 2.0]))
        with pytest.raises(Exception, match="cannot assign"):  # FrozenInstanceError
            batch.x = np.array([3.0, 4.0])  # type: ignore[misc]

    def test_equality(self) -> None:
        """Equality."""
        # Dataclass equality on ndarray fields triggers ValueError for
        # multi-element arrays (truth value ambiguous). Use scalars.
        a = AdaptationBatch(x=np.array(1.0))
        b = AdaptationBatch(x=np.array(1.0))
        assert a == b
        c = AdaptationBatch(x=np.array(3.0))
        assert a != c

    def test_equality_with_none_fields(self) -> None:
        """Equality with none fields."""
        a = AdaptationBatch(x=np.array(1.0))
        b = AdaptationBatch(x=np.array(1.0), predictions=None, representations=None)
        assert a == b

    def test_tensor_x_field(self) -> None:
        """Tensor x field."""
        batch = AdaptationBatch(x=torch.tensor([1.0, 2.0, 3.0]))
        assert isinstance(batch.x, torch.Tensor)
        assert tuple(batch.x.shape) == (3,)

    def test_tensor_representations_field(self) -> None:
        """Tensor representations field."""
        batch = AdaptationBatch(
            x=np.array([1.0]),
            representations=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        )
        assert isinstance(batch.representations, torch.Tensor)

    def test_tensor_sigma_x_field(self) -> None:
        """Tensor sigma x field."""
        batch = AdaptationBatch(
            x=np.array([1.0]),
            sigma_x=torch.tensor([[0.05], [0.06]]),
        )
        assert isinstance(batch.sigma_x, torch.Tensor)

    def test_repr_includes_fields(self) -> None:
        """Repr includes fields."""
        batch = AdaptationBatch(x=np.array([1.0, 2.0]), predictions=None)
        r = repr(batch)
        assert "AdaptationBatch" in r
        assert "x=" in r
        assert "predictions=" in r

    def test_predictions_field_numpy_array_compatible(self) -> None:
        """Predictions can be a PredictiveBatch with numpy arrays."""
        pb = PredictiveBatch(
            point=np.array([1.0, 2.0], dtype=np.float32),
            std=np.array([0.1, 0.2], dtype=np.float32),
        )
        batch = AdaptationBatch(x=np.array([1.0, 2.0]), predictions=pb)
        assert batch.predictions is pb

    def test_predictions_field_torch_tensor_compatible(self) -> None:
        """Predictions field torch tensor compatible."""
        pb = PredictiveBatch(
            mean=torch.tensor([1.0, 2.0]),
            std=torch.tensor([0.1, 0.2]),
        )
        batch = AdaptationBatch(x=np.array([1.0, 2.0]), predictions=pb)
        assert batch.predictions is pb


# ═══════════════════════════════════════════════════════════════════════════════
# SupportsPredictiveBatch protocol
# ═══════════════════════════════════════════════════════════════════════════════


class _PredictiveModel:
    """Concrete implementation of SupportsPredictiveBatch."""

    def predict_distribution(self, X: np.ndarray, **kwargs: object) -> PredictiveBatch:
        mean = np.mean(X, axis=1).astype(np.float32)
        std = np.full(X.shape[0], 0.1, dtype=np.float32)
        return PredictiveBatch(mean=mean, std=std)


class _NotPredictive:
    """Class that does NOT implement predict_distribution."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean(X, axis=1)


class TestSupportsPredictiveBatch:
    """SupportsPredictiveBatch — runtime-checkable protocol."""

    def test_isinstance_positive(self) -> None:
        """Isinstance positive."""
        model = _PredictiveModel()
        assert isinstance(model, SupportsPredictiveBatch)

    def test_isinstance_negative(self) -> None:
        """Isinstance negative."""
        model = _NotPredictive()
        assert not isinstance(model, SupportsPredictiveBatch)

    def test_isinstance_basic_object(self) -> None:
        """Isinstance basic object."""
        assert not isinstance(object(), SupportsPredictiveBatch)

    def test_call_method(self) -> None:
        """Call method."""
        model = _PredictiveModel()
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        result = model.predict_distribution(X)
        assert isinstance(result, PredictiveBatch)
        assert result.mean is not None
        assert result.std is not None

    def test_passes_extra_kwargs(self) -> None:
        """Passes extra kwargs."""

        class _KwargsModel:
            def predict_distribution(self, X: np.ndarray, **kwargs: object) -> PredictiveBatch:
                assert kwargs.get("temperature") == 0.5
                return PredictiveBatch(point=np.zeros(X.shape[0]))

        model = _KwargsModel()
        assert isinstance(model, SupportsPredictiveBatch)
        result = model.predict_distribution(np.ones((3, 2)), temperature=0.5)
        assert result.point is not None


# ═══════════════════════════════════════════════════════════════════════════════
# flatten_adaptation_parameters
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlattenAdaptationParameters:
    """flatten_adaptation_parameters — utility for TTA parameter grouping."""

    def test_basic_flattening(self) -> None:
        """Basic flattening."""
        p1 = torch.nn.Parameter(torch.tensor(1.0))
        p2 = torch.nn.Parameter(torch.tensor(2.0))
        p3 = torch.nn.Parameter(torch.tensor(3.0))
        groups: dict[str, list[torch.nn.Parameter]] = {
            "a": [p1, p2],
            "b": [p3],
        }
        result = flatten_adaptation_parameters(groups)
        assert len(result) == 3
        assert p1 in result
        assert p2 in result
        assert p3 in result

    def test_deduplicates_shared_parameter(self) -> None:
        """A parameter appearing in multiple groups should appear only once."""
        shared = torch.nn.Parameter(torch.tensor(1.0))
        p2 = torch.nn.Parameter(torch.tensor(2.0))
        groups: dict[str, list[torch.nn.Parameter]] = {
            "a": [shared],
            "b": [shared, p2],
        }
        result = flatten_adaptation_parameters(groups)
        assert len(result) == 2
        # shared param appears once
        assert result.count(shared) == 1

    def test_empty_dict(self) -> None:
        """Empty dict."""
        result = flatten_adaptation_parameters({})
        assert result == []

    def test_single_group(self) -> None:
        """Single group."""
        p1 = torch.nn.Parameter(torch.tensor(1.0))
        p2 = torch.nn.Parameter(torch.tensor(2.0))
        result = flatten_adaptation_parameters({"only": [p1, p2]})
        assert result == [p1, p2]

    def test_preserves_parameter_identity(self) -> None:
        """Returned parameters are the same objects (not copies)."""
        p = torch.nn.Parameter(torch.tensor(42.0))
        result = flatten_adaptation_parameters({"g": [p]})
        assert result[0] is p
        # Mutating the returned parameter mutates the original
        with torch.no_grad():
            result[0].add_(1.0)
        assert float(p) == 43.0

    def test_order_is_stable(self) -> None:
        """First occurrence of each parameter determines position."""
        p1 = torch.nn.Parameter(torch.tensor(1.0))
        p2 = torch.nn.Parameter(torch.tensor(2.0))
        p3 = torch.nn.Parameter(torch.tensor(3.0))
        groups: dict[str, list[torch.nn.Parameter]] = {
            "first": [p1],
            "second": [p2, p3],
        }
        result = flatten_adaptation_parameters(groups)
        assert result[0] is p1
        assert result[1] is p2
        assert result[2] is p3

    def test_real_module_parameters(self) -> None:
        """End-to-end: flatten parameter groups from a real nn.Module."""

        class _AdaptableModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.head = torch.nn.Linear(8, 2)
                self.embed = torch.nn.Linear(8, 8)
            def adaptation_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
                return {"head": list(self.head.parameters()), "embed": list(self.embed.parameters())}

        model = _AdaptableModule()
        groups = model.adaptation_parameter_groups()
        flat = flatten_adaptation_parameters(groups)
        # head has 2 params (weight, bias), embed has 2 params → total 4 unique
        assert len(flat) == 4
        assert all(isinstance(p, torch.nn.Parameter) for p in flat)

    def test_iterables_other_than_lists(self) -> None:
        """Groups can be tuples or other iterables."""
        p1 = torch.nn.Parameter(torch.tensor(1.0))
        p2 = torch.nn.Parameter(torch.tensor(2.0))
        groups: dict[str, tuple[torch.nn.Parameter, ...]] = {
            "a": (p1, p2),
        }
        result = flatten_adaptation_parameters(groups)
        assert len(result) == 2

    def test_all_parameters_are_trainable(self) -> None:
        """Should work with requires_grad=True parameters (default)."""
        p = torch.nn.Parameter(torch.tensor(1.0))
        result = flatten_adaptation_parameters({"g": [p]})
        assert result[0].requires_grad


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-protocol: model implementing multiple protocols
# ═══════════════════════════════════════════════════════════════════════════════


class _MultiProtocolModel(torch.nn.Module):
    """Implements all three test-time protocols."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def predict_distribution(self, X: np.ndarray, **kwargs: object) -> PredictiveBatch:
        return PredictiveBatch(point=np.zeros(X.shape[0], dtype=np.float32))

    def representation_dict(self, x: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
        return {"feat": x}

    def adaptation_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        return {"all": list(self.parameters())}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestMultiProtocol:
    """Model implementing all three protocols simultaneously."""

    def test_all_protocols_recognised(self) -> None:
        """All protocols recognised."""
        model = _MultiProtocolModel()
        assert isinstance(model, SupportsPredictiveBatch)
        assert hasattr(model, "representation_dict")
        assert hasattr(model, "adaptation_parameter_groups")

    def test_can_call_all_methods(self) -> None:
        """Can call all methods."""
        model = _MultiProtocolModel()
        # Predict
        batch = model.predict_distribution(np.ones((3, 4)))
        assert isinstance(batch, PredictiveBatch)
        # Represent
        reps = model.representation_dict(torch.randn(3, 4))
        assert "feat" in reps
        # Params
        groups = model.adaptation_parameter_groups()
        flat = flatten_adaptation_parameters(groups)
        assert len(flat) == 2  # weight + bias

    def test_flatten_adaptation_parameters_with_multi_protocol(self) -> None:
        """Flatten adaptation parameters with multi protocol."""
        model = _MultiProtocolModel()
        groups = model.adaptation_parameter_groups()
        flat = flatten_adaptation_parameters(groups)
        # Verify parameters are those from the model (by identity, not equality)
        model_ids = {id(p) for p in model.parameters()}
        assert len(flat) == len(model_ids)
        assert all(id(p) in model_ids for p in flat)
