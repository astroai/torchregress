"""Tests for BBSE target-prior estimation (F6)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchregress.test_time.label_shift import estimate_target_prior_bbse


def _one_hot(pred_labels: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((pred_labels.shape[0], k))
    out[np.arange(pred_labels.shape[0]), pred_labels] = 1.0
    return out


def test_bbse_recovers_known_prior_on_constructed_confusion():
    rng = np.random.default_rng(0)
    # Confusion C[i, j] = P(pred=j | y=i); a noisy-but-invertible classifier.
    confusion = np.array([[0.9, 0.1], [0.2, 0.8]])
    true_prior = np.array([0.7, 0.3])
    n = 40000

    y_src = rng.choice(2, size=n)
    pred_src = (rng.random(n) < confusion[y_src, 0]).astype(int)
    probs_src = _one_hot(pred_src, 2)

    y_tgt = rng.choice(2, size=n, p=true_prior)
    pred_tgt = (rng.random(n) < confusion[y_tgt, 0]).astype(int)
    probs_tgt = _one_hot(pred_tgt, 2)

    estimated = estimate_target_prior_bbse(probs_src, y_src, probs_tgt)
    np.testing.assert_allclose(estimated, true_prior, atol=0.01)


def test_bbse_three_class_recovers_prior():
    rng = np.random.default_rng(1)
    confusion = np.array([[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]])
    true_prior = np.array([0.2, 0.5, 0.3])
    n = 60000
    k = 3

    y_src = rng.choice(k, size=n)
    probs_src = rng.random((n, k)) + confusion[y_src] * 5.0  # argmax ~ confusion row
    # Deterministic argmax construction keeps the test exact:
    u = rng.random(n)
    cum = np.cumsum(confusion[y_src], axis=1)
    pred_src = (u[:, None] > cum).sum(axis=1)
    probs_src = _one_hot(pred_src, k)

    y_tgt = rng.choice(k, size=n, p=true_prior)
    u = rng.random(n)
    cum = np.cumsum(confusion[y_tgt], axis=1)
    pred_tgt = (u[:, None] > cum).sum(axis=1)
    probs_tgt = _one_hot(pred_tgt, k)

    estimated = estimate_target_prior_bbse(probs_src, y_src, probs_tgt)
    np.testing.assert_allclose(estimated, true_prior, atol=0.02)


def test_bbse_ill_conditioned_confusion_raises():
    # Classifier never predicts class 1 -> zero column -> cond ~ inf.
    probs = np.tile(np.array([1.0, 0.0]), (50, 1))
    labels = np.zeros(50, dtype=int)
    with pytest.raises(ValueError, match="ill-conditioned"):
        estimate_target_prior_bbse(probs, labels, probs)


def test_bbse_cond_threshold_is_tightenable():
    rng = np.random.default_rng(2)
    confusion = np.array([[0.99, 0.01], [0.02, 0.98]])
    n = 50000
    y_src = rng.choice(2, size=n)
    pred_src = (rng.random(n) < confusion[y_src, 0]).astype(int)
    probs_src = _one_hot(pred_src, 2)
    probs_tgt = _one_hot((rng.random(n) < 0.5).astype(int), 2)

    # A tiny threshold must reject even this mildly ill-conditioned matrix.
    with pytest.raises(ValueError, match="ill-conditioned"):
        estimate_target_prior_bbse(probs_src, y_src, probs_tgt, cond_threshold=1.0)


def test_bbse_validates_inputs():
    probs_ok = np.tile(np.array([0.9, 0.1]), (10, 1))
    bad_labels = np.full(5, 0)
    with pytest.raises(ValueError, match="align"):
        estimate_target_prior_bbse(probs_ok, bad_labels, probs_ok)
    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        estimate_target_prior_bbse(probs_ok, np.full(10, 5), probs_ok)
    with pytest.raises(ValueError, match="classes"):
        estimate_target_prior_bbse(probs_ok, np.zeros(10, int), np.ones((10, 3)) / 3.0)


def test_bbse_uses_torch_solve_path():
    # torch.linalg.solve is exercised: result solves C^T w = mu exactly on a
    # deterministic input.
    probs_src = _one_hot(np.array([0, 0, 1, 1]), 2)
    labels = np.array([0, 0, 1, 1])
    probs_tgt = _one_hot(np.array([1, 1]), 2)  # target always predicts class 1
    w = estimate_target_prior_bbse(probs_src, labels, probs_tgt)
    confusion = np.array([[1.0, 0.0], [0.0, 1.0]])
    mu = np.array([0.0, 1.0])
    expected = np.linalg.solve(torch.from_numpy(confusion.T).numpy(), mu)
    np.testing.assert_allclose(w, expected, rtol=1e-12)
