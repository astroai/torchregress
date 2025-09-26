"""
Ensemble model implementations.

This module provides concrete implementations of ensemble models
for regression tasks with uncertainty estimation.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.utils.data

from .base import BaseEnsembleModel
from .layers import BatchEnsembleLinear


class HeteroscedasticEnsembleModel(BaseEnsembleModel):
    """
    Ensemble model with heteroscedastic uncertainty estimation.

    This model assumes each ensemble member predicts both mean and variance.

    Args:
        base_model: Base model class or instance to ensemble
        ensemble_size: Number of ensemble members
        device: Device to use
    """

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make prediction with uncertainty estimates.

        Args:
            x: Input tensor [batch_size, ...]

        Returns:
            Dictionary with mean, epistemic and aleatoric variance
        """
        with torch.no_grad():
            # Get predictions from all ensemble members
            predictions = self.forward(x)

            # Separate means and log_vars
            means = []
            log_vars = []

            for pred in predictions:
                if isinstance(pred, tuple) and len(pred) == 2:
                    # If output is a tuple, assume it's (mean, log_var)
                    mean, log_var = pred
                elif pred.shape[1] == 2 * pred.shape[1] // 2:  # For even number of outputs
                    # Assume first half is mean, second half is log_var
                    dim = pred.shape[1] // 2
                    mean, log_var = pred[:, :dim], pred[:, dim:]
                else:
                    raise ValueError(
                        "Model output format not recognized for heteroscedastic uncertainty"
                    )

                means.append(mean)
                log_vars.append(log_var)

            # Stack means and calculate ensemble mean [batch_size, output_dim]
            stacked_means = torch.stack(means)
            ensemble_mean = torch.mean(stacked_means, dim=0)

            # Convert log_vars to variances
            variances = [torch.exp(log_var) for log_var in log_vars]

            # Stack variances and calculate mean aleatoric uncertainty
            stacked_vars = torch.stack(variances)
            aleatoric_var = torch.mean(stacked_vars, dim=0)

            # Calculate epistemic uncertainty as variance of means
            epistemic_var = torch.var(stacked_means, dim=0, unbiased=True)

            # Total predictive variance is sum of epistemic and aleatoric
            total_var = epistemic_var + aleatoric_var

            return {
                "mean": ensemble_mean,
                "variance": total_var,
                "epistemic_variance": epistemic_var,
                "aleatoric_variance": aleatoric_var,
            }

    def predict_full_covariance(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make prediction with full-output covariance estimation, splitting aleatoric and epistemic.
        """
        with torch.no_grad():
            preds = self.forward(x)
            means, vars_ = [], []
            for pred in preds:
                # Expect (mean, log_var) or concatenated [mean|log_var]
                if isinstance(pred, tuple) and len(pred) == 2:
                    mean, log_var = pred
                elif pred.shape[1] % 2 == 0:
                    d = pred.shape[1] // 2
                    mean, log_var = pred[:, :d], pred[:, d:]
                else:
                    raise ValueError("Unexpected output format for heteroscedastic ensemble.")
                means.append(mean)
                vars_.append(torch.exp(log_var))

            # Stack [M, B, D]
            stacked_means = torch.stack(means)
            mean = torch.mean(stacked_means, dim=0)
            # Epistemic covariance: variance of means
            p = stacked_means.permute(1, 0, 2)  # [B, M, D]
            p_centered = p - mean.unsqueeze(1)
            epi_cov = torch.einsum("bmd,bnd->bmn", p_centered, p_centered) / (
                self.ensemble_size - 1
            )
            # Aleatoric covariance: mean of member variances as diagonal
            stacked_vars = torch.stack(vars_)  # [M, B, D]
            avg_vars = torch.mean(stacked_vars, dim=0)  # [B, D]
            ale_cov = torch.diag_embed(avg_vars)
            total_cov = epi_cov + ale_cov
            return {
                "mean": mean,
                "epistemic_covariance": epi_cov,
                "aleatoric_covariance": ale_cov,
                "total_covariance": total_cov,
            }


class DeepEnsemble(BaseEnsembleModel):
    """
    Implementation of deep ensembles for uncertainty estimation.

    Reference: "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
    by Lakshminarayanan, Pritzel, Blundell.

    This model creates an ensemble of independently trained models and combines
    their predictions for improved uncertainty estimation.

    Args:
        base_model: Base model class or instance to ensemble
        ensemble_size: Number of ensemble members
        device: Device to use
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer_class: type = torch.optim.Adam,
        epochs: int = 10,
        lr: float = 0.001,
        verbose: bool = True,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        **optimizer_kwargs,
    ) -> Dict[str, Any]:
        """
        Train all ensemble members independently.

        Args:
            train_loader: DataLoader for training data
            criterion: Loss function
            optimizer_class: Optimizer class
            epochs: Number of epochs
            lr: Learning rate
            verbose: Whether to print progress
            val_loader: Optional DataLoader for validation
            **optimizer_kwargs: Additional optimizer parameters

        Returns:
            Dictionary with training history
        """
        histories = []

        for i, model in enumerate(self.models):
            if verbose:
                print(f"\nTraining ensemble member {i+1}/{self.ensemble_size}")

            # Create optimizer
            optimizer = optimizer_class(model.parameters(), lr=lr, **optimizer_kwargs)

            # Train this ensemble member
            history = self._train_single_model(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                epochs=epochs,
                verbose=verbose,
                val_loader=val_loader,
            )

            histories.append(history)

        return {"member_histories": histories}

    def _train_single_model(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        verbose: bool,
        val_loader: Optional[torch.utils.data.DataLoader],
    ) -> Dict[str, List[float]]:
        """
        Train a single ensemble member.

        Args:
            model: Model to train
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            epochs: Number of training epochs
            verbose: Whether to print progress
            val_loader: Optional validation data loader

        Returns:
            Dictionary with training history
        """
        model.to(self.device)

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Training phase
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(targets, outputs)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_loss / max(num_batches, 1)
            train_losses.append(avg_train_loss)

            # Validation phase
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                val_batches = 0

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)

                        outputs = model(inputs)
                        loss = criterion(targets, outputs)

                        val_loss += loss.item()
                        val_batches += 1

                avg_val_loss = val_loss / max(val_batches, 1)
                val_losses.append(avg_val_loss)

                if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                    print(
                        f"Epoch {epoch+1}/{epochs} - Train loss: {avg_train_loss:.4f} - Val loss: {avg_val_loss:.4f}"
                    )
            else:
                if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train loss: {avg_train_loss:.4f}")

        return {"train_loss": train_losses, "val_loss": val_losses}


class HeteroscedasticBatchEnsembleModel(nn.Module):
    """
    Batch ensemble model with heteroscedastic uncertainty estimation.

    This combines batch ensembling (efficient parameter sharing) with
    heteroscedastic uncertainty estimation.

    Args:
        backbone: Base model architecture (without output head)
        input_size: Size of input features
        output_size: Size of output features (half will be used for variance)
        ensemble_size: Number of ensemble members
        device: Device to use
    """

    def __init__(
        self,
        backbone: nn.Module,
        input_size: int,
        output_size: int,
        ensemble_size: int = 4,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

        # Final layer is a BatchEnsemble layer with 2*output_size outputs
        # (mean and log_var for each output dimension)
        self.output_layer = BatchEnsembleLinear(
            in_features=input_size,
            out_features=2 * output_size,
            ensemble_size=ensemble_size,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor [batch_size, ...]

        Returns:
            Dictionary with mean and log_var predictions for each ensemble member
        """
        # Extract features using the backbone
        features = self.backbone(x)

        # Pass through batch ensemble layer to get ensemble outputs
        # Output shape: [batch_size, ensemble_size, 2*output_size]
        outputs = self.output_layer(features)

        # Split outputs into mean and log variance
        means = outputs[:, :, : self.output_size]  # [batch_size, ensemble_size, output_size]
        log_vars = outputs[:, :, self.output_size :]  # [batch_size, ensemble_size, output_size]

        return {"means": means, "log_vars": log_vars}

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make prediction with uncertainty estimates.

        Args:
            x: Input tensor [batch_size, ...]

        Returns:
            Dictionary with mean, epistemic and aleatoric variance
        """
        with torch.no_grad():
            outputs = self.forward(x)

            means = outputs["means"]  # [batch_size, ensemble_size, output_size]
            log_vars = outputs["log_vars"]  # [batch_size, ensemble_size, output_size]

            # Convert log_vars to variances
            variances = torch.exp(log_vars)

            # Calculate ensemble mean across members
            ensemble_mean = torch.mean(means, dim=1)  # [batch_size, output_size]

            # Calculate aleatoric uncertainty (mean of predicted variances)
            aleatoric_var = torch.mean(variances, dim=1)  # [batch_size, output_size]

            # Calculate epistemic uncertainty (variance of means)
            epistemic_var = torch.var(means, dim=1, unbiased=True)  # [batch_size, output_size]

            # Total predictive variance is sum of epistemic and aleatoric
            total_var = epistemic_var + aleatoric_var  # [batch_size, output_size]

            return {
                "mean": ensemble_mean,
                "variance": total_var,
                "epistemic_variance": epistemic_var,
                "aleatoric_variance": aleatoric_var,
            }
