"""
Test-Time Adaptation Suite.

This script demonstrates test-time adaptation utilities under covariate shift
(feature distribution changes) and label shift (target prior distribution changes).

Specifically, it showcases:
1. Covariate shift alignment via FeatureStatNormalizer and WeightedSubspaceMomentAligner.
2. Label shift correction via PosteriorLabelShiftAdapter (discrete probabilities).
3. Continuous Gaussian label shift correction via correct_gaussian_predictions_for_label_shift.
4. Active sample selection using entropy_scores, confidence_scores, and select_high_confidence.
"""

import numpy as np

from torchregress.test_time.label_shift import (
    GaussianLabelShiftConfig,
    PosteriorLabelShiftAdapter,
    correct_gaussian_predictions_for_label_shift,
)
from torchregress.test_time.selection import (
    confidence_scores,
    entropy_scores,
    pseudo_label_targets,
    select_high_confidence,
)
from torchregress.test_time.subspace import FeatureStatNormalizer, WeightedSubspaceMomentAligner


def main():
    print("================================================================================")
    print("                 torchregress Test-Time Adaptation Suite                        ")
    print("================================================================================")

    np.random.seed(42)

    # --------------------------------------------------------------------------------
    # 1. Feature / Covariate Shift Alignment
    # --------------------------------------------------------------------------------
    print("\n--- 1. Covariate Shift Alignment ---")
    n_samples = 200
    n_features = 4

    # Generate source features (zero mean, unit variance)
    X_source = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    y_source = X_source[:, 0] * 2.0 + X_source[:, 1] * 0.5 + np.random.normal(0.0, 0.1, n_samples)

    # Target features have shifted (mean shift + variance scale)
    X_target = np.random.normal(loc=1.5, scale=2.5, size=(n_samples, n_features))

    print(
        f"Original source feature 0 stats: mean={X_source[:, 0].mean():.3f}, std={X_source[:, 0].std():.3f}"
    )
    print(
        f"Shifted target feature 0 stats : mean={X_target[:, 0].mean():.3f}, std={X_target[:, 0].std():.3f}"
    )

    # A. Feature Statistics Normalization
    normalizer = FeatureStatNormalizer()
    normalizer.fit(X_source)
    X_target_norm = normalizer.transform(X_target)
    print(
        f"After FeatureStatNormalizer    : mean={X_target_norm[:, 0].mean():.3f}, std={X_target_norm[:, 0].std():.3f}"
    )

    # B. Significant Subspace Alignment (SSA)
    aligner = WeightedSubspaceMomentAligner(variance_threshold=0.90)
    X_target_aligned = aligner.fit_transform(X_source, X_target, y_source=y_source)
    print(
        f"After SignificantSubspaceAlign : mean={X_target_aligned[:, 0].mean():.3f}, std={X_target_aligned[:, 0].std():.3f}"
    )
    print(f"Subspace Rank Selected         : {aligner.state_.rank}")

    # --------------------------------------------------------------------------------
    # 2. Discrete Label Shift Correction
    # --------------------------------------------------------------------------------
    print("\n--- 2. Discrete Label Shift Correction ---")
    n_classes = 3
    # Source prior is balanced
    source_prior = np.array([1 / 3, 1 / 3, 1 / 3])
    # Target prior has shifted heavily to class 0
    target_prior_true = np.array([0.7, 0.2, 0.1])

    # Generate target predictions (simulating a shifted target test set)
    n_test = 500
    true_labels = np.random.choice(n_classes, size=n_test, p=target_prior_true)

    # Simulate predicted probabilities with some classification noise
    probs_source = np.eye(n_classes)[true_labels] + np.random.uniform(0.0, 0.2, (n_test, n_classes))
    probs_source = probs_source / probs_source.sum(axis=1, keepdims=True)

    # Initialize and fit label shift adapter
    adapter = PosteriorLabelShiftAdapter(source_prior=source_prior)
    estimate = adapter.estimate(probs_source)

    print("True Target Prior              :", target_prior_true)
    print("EM Estimated Target Prior      :", estimate.target_prior)
    print(
        "EM Convergence Status          :",
        f"Converged in {estimate.iterations} steps" if estimate.converged else "Did not converge",
    )

    # Apply label shift correction to model posteriors
    corrected_probs = adapter.transform(probs_source)
    print("Average original probabilities :", probs_source.mean(axis=0))
    print("Average corrected probabilities:", corrected_probs.mean(axis=0))

    # --------------------------------------------------------------------------------
    # 3. Continuous Gaussian Label Shift Correction
    # --------------------------------------------------------------------------------
    print("\n--- 3. Continuous Gaussian Label Shift Correction ---")
    # Simulate Gaussian predictions: mean and std dev
    mu_pred = np.random.normal(loc=0.0, scale=1.0, size=n_test)
    std_pred = np.random.uniform(0.1, 0.5, size=n_test)

    # Source target labels (balanced distribution)
    source_targets = np.random.normal(loc=0.0, scale=1.0, size=1000)

    # Correct predictions for label shift
    corrected_mu, corrected_std, metadata = correct_gaussian_predictions_for_label_shift(
        mean=mu_pred,
        std=std_pred,
        source_targets=source_targets,
        config=GaussianLabelShiftConfig(n_bins=16),
    )

    print(f"Original mean pred stats       : mean={mu_pred.mean():.3f}, std={mu_pred.std():.3f}")
    print(
        f"Corrected mean pred stats      : mean={corrected_mu.mean():.3f}, std={corrected_mu.std():.3f}"
    )
    print("EM steps during Gaussian shift :", metadata.get("estimate_iterations"))

    # --------------------------------------------------------------------------------
    # 4. Confidence & Selection Utilities
    # --------------------------------------------------------------------------------
    print("\n--- 4. Active Sample Selection & Confidence Utilities ---")
    # Use corrected probabilities to compute entropy and confidence
    ent = entropy_scores(corrected_probs)
    conf = confidence_scores(corrected_probs)
    labels, weights = pseudo_label_targets(corrected_probs)

    # Select high-confidence samples (lowest entropy or highest confidence)
    mask = select_high_confidence(corrected_probs, min_confidence=0.75, top_fraction=0.20)

    print(f"Mean Shannon Entropy           : {ent.mean():.4f}")
    print(f"Mean Confidence Score          : {conf.mean():.4f}")
    print(
        f"Selected High-Confidence Count : {mask.sum()} out of {len(mask)} ({mask.mean() * 100:.1f}%)"
    )
    print("First 5 pseudo-labels          :", labels[:5])
    print("First 5 pseudo-label weights   :", np.round(weights[:5], 3))

    print("================================================================================")
    print("                 Test-Time Adaptation Suite completed!                          ")
    print("================================================================================")


if __name__ == "__main__":
    main()
