"""
Transforms and Augmentations Demo.

This script demonstrates utility target-space transforms and data augmentations
provided in the `torchregress.utils` module.

It showcases:
1. Target-Space Transforms: Log, Square Root, Box-Cox, and Yeo-Johnson transforms
   along with their inverses and the make_target_transform factory helper.
2. Data Augmentation Layers: GaussianNoise, MixUp, FeatureMask, Adversarial,
   and EnsemblePerturbationAugmenter.
"""

import torch
import torch.nn as nn

from torchregress.utils import (
    Adversarial,
    EnsemblePerturbationAugmenter,
    FeatureMask,
    # Augmentations
    GaussianNoise,
    MixUp,
    make_target_transform,
)


def main():
    print("================================================================================")
    print("             torchregress Transforms & Augmentations Showcase                   ")
    print("================================================================================")

    torch.manual_seed(42)

    # --------------------------------------------------------------------------------
    # 1. Target-Space Transforms
    # --------------------------------------------------------------------------------
    print("\n--- 1. Target-Space Transforms ---")

    # Positive support targets
    targets_positive = torch.tensor([0.0, 1.0, 4.0, 9.0, 16.0])
    # Signed targets (both positive and negative values)
    targets_signed = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    # A. Identity
    identity_tx = make_target_transform("identity")
    print(f"Identity Transform  : {identity_tx(targets_positive)}")

    # B. Log
    log_tx = make_target_transform("log", eps=1e-5)
    log_val = log_tx(targets_positive)
    print(f"Log Transform       : {log_val}")
    print(f"Log Inverse         : {log_tx.inverse(log_val)}")

    # C. Sqrt
    sqrt_tx = make_target_transform("sqrt")
    sqrt_val = sqrt_tx(targets_positive)
    print(f"Sqrt Transform      : {sqrt_val}")
    print(f"Sqrt Inverse        : {sqrt_tx.inverse(sqrt_val)}")

    # D. Box-Cox
    boxcox_tx = make_target_transform("boxcox", lam=0.5)
    boxcox_val = boxcox_tx(targets_positive)
    print(f"Box-Cox Transform   : {boxcox_val}")
    print(f"Box-Cox Inverse     : {boxcox_tx.inverse(boxcox_val)}")

    # E. Yeo-Johnson (Supports positive & negative values)
    yeo_tx = make_target_transform("yeojohnson", lam=1.5)
    yeo_val = yeo_tx(targets_signed)
    print(f"Yeo-Johnson Trans   : {yeo_val}")
    print(f"Yeo-Johnson Inverse : {yeo_tx.inverse(yeo_val)}")

    # --------------------------------------------------------------------------------
    # 2. Data Augmentation Techniques
    # --------------------------------------------------------------------------------
    print("\n--- 2. Data Augmentation Techniques ---")

    # Inputs: batch_size=4, n_features=3
    x_batch = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    y_batch = torch.tensor([[1.5], [4.5], [7.5], [10.5]])

    print("Original Inputs x:\n", x_batch)
    print("Original Targets y:\n", y_batch.squeeze(1))

    # A. Gaussian Noise (Probability=1.0 for demonstration)
    noise_aug = GaussianNoise(std=0.5, probability=1.0)
    x_noise, _ = noise_aug(x_batch)
    print("\nGaussianNoise Augmentation:\n", x_noise)

    # B. MixUp (Interpolates both inputs and targets)
    mixup_aug = MixUp(alpha=0.4, probability=1.0)
    x_mix, y_mix = mixup_aug(x_batch, y_batch)
    print("\nMixUp Augmentation Inputs:\n", x_mix)
    print("MixUp Augmentation Targets:\n", y_mix.squeeze(1))

    # C. Feature Masking (Sets features to 0.0 with mask ratio)
    mask_aug = FeatureMask(mask_ratio=0.33, probability=1.0)
    x_masked, _ = mask_aug(x_batch)
    print("\nFeatureMask Augmentation:\n", x_masked)

    # D. Adversarial Perturbation (Using model gradients)
    # Instantiate a dummy linear model
    dummy_model = nn.Linear(3, 1)
    dummy_loss_fn = nn.MSELoss()

    # Ensure gradients are enabled for adversarial calculation
    dummy_model.train()
    adversarial_aug = Adversarial(
        model=dummy_model, loss_fn=dummy_loss_fn, epsilon=0.1, steps=5, probability=1.0
    )
    x_adv, _ = adversarial_aug(x_batch, y_batch)
    print("\nAdversarial Augmentation:\n", x_adv)
    print("Perturbation Magnitude Max:\n", torch.max(torch.abs(x_adv - x_batch)).item())

    # E. Ensemble Perturbation Augmenter (Generates multiple copies for EIV / ensemble validation)
    perturber = EnsemblePerturbationAugmenter(n_samples=3, perturb_method="gaussian", sigma=0.05)
    perturbed_copies = perturber(x_batch)
    print(f"\nEnsemblePerturbationAugmenter (n_samples={len(perturbed_copies)}):")
    for i, copy in enumerate(perturbed_copies):
        print(f"  Copy {i}:\n", copy)

    print("================================================================================")
    print("             torchregress Transforms & Augmentations completed!                 ")
    print("================================================================================")


if __name__ == "__main__":
    main()
