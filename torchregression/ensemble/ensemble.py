"""
Ensemble models for regression tasks.

This module provides implementations of various ensemble techniques 
including deep ensembles, batch ensembles, and more.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from copy import deepcopy

class BatchEnsembleLinear(nn.Module):
    """
    BatchEnsemble linear layer implementation.
    
    This implements the BatchEnsemble technique from:
    "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning"
    
    Instead of maintaining M copies of a model, BatchEnsemble uses rank-1 perturbations
    to create M virtual models that share parameters.
    
    Args:
        in_features: Size of input features
        out_features: Size of output features
        ensemble_size: Number of ensemble members
        bias: Whether to use bias
        device: Device to use
        dtype: Data type
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int = 4,
        bias: bool = True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        
        # Main weight matrix - shared across ensemble
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        # Fast weight vectors for ensemble (rank-1 perturbation)
        self.r_vectors = nn.Parameter(torch.empty((ensemble_size, in_features), **factory_kwargs))
        self.s_vectors = nn.Parameter(torch.empty((ensemble_size, out_features), **factory_kwargs))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize parameters using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        
        # Initialize r and s vectors with random signs
        nn.init.normal_(self.r_vectors, mean=1.0, std=0.1)
        nn.init.normal_(self.s_vectors, mean=1.0, std=0.1)
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with BatchEnsemble computation.
        
        Input can be:
         - [batch_size, in_features]: Same input applied to all ensemble members
         - [batch_size, ensemble_size, in_features]: Different input for each ensemble member
        
        Returns:
            Output tensor [batch_size, ensemble_size, out_features]
        """
        # Handle different input shapes
        if input.dim() == 2:
            # Same input for all ensemble members: [batch_size, in_features]
            batch_size = input.shape[0]
            
            # Repeat input for each ensemble member
            # Shape becomes [ensemble_size, batch_size, in_features]
            repeated_input = input.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
            
            # Apply r vectors (element-wise) to the repeated inputs
            # [ensemble_size, batch_size, in_features]
            r_input = repeated_input * self.r_vectors.unsqueeze(1)
            
            # Reshape for batch matmul
            r_input = r_input.view(-1, self.in_features)
            
            # Apply main weight matrix
            # [ensemble_size * batch_size, out_features]
            output = F.linear(r_input, self.weight, None)
            
            # Reshape output to [ensemble_size, batch_size, out_features]
            output = output.view(self.ensemble_size, batch_size, self.out_features)
            
            # Apply s vectors (element-wise)
            output = output * self.s_vectors.unsqueeze(1)
            
            # Add bias if needed
            if self.bias is not None:
                output = output + self.bias
                
            # Reorder to [batch_size, ensemble_size, out_features]
            output = output.permute(1, 0, 2)
            
        elif input.dim() == 3:
            # Different input for each ensemble member: [batch_size, ensemble_size, in_features]
            if input.shape[1] != self.ensemble_size:
                raise ValueError(f"Input ensemble dimension size {input.shape[1]} doesn't match "
                               f"expected ensemble size {self.ensemble_size}")
                
            batch_size = input.shape[0]
            
            # Apply r vectors (element-wise) to the inputs, separately for each ensemble member
            # [batch_size, ensemble_size, in_features]
            r_input = input * self.r_vectors.unsqueeze(0)
            
            # Reshape for batch matmul
            r_input = r_input.view(-1, self.in_features)
            
            # Apply main weight matrix
            # [batch_size * ensemble_size, out_features]
            output = F.linear(r_input, self.weight, None)
            
            # Reshape output to [batch_size, ensemble_size, out_features]
            output = output.view(batch_size, self.ensemble_size, self.out_features)
            
            # Apply s vectors (element-wise)
            output = output * self.s_vectors.unsqueeze(0)
            
            # Add bias if needed
            if self.bias is not None:
                output = output + self.bias
                
        else:
            raise ValueError(f"Input must be 2D or 3D, got {input.dim()}D")
            
        return output
    
    def extra_repr(self) -> str:
        """String representation of the module."""
        return 'in_features={}, out_features={}, ensemble_size={}, bias={}'.format(
            self.in_features, self.out_features, self.ensemble_size, self.bias is not None
        )


class BaseEnsembleModel(nn.Module):
    """
    Base class for ensemble models.
    
    This class provides common functionality for different ensemble techniques.
    
    Args:
        base_model: Base model class or instance to ensemble
        ensemble_size: Number of ensemble members
        device: Device to use
    """
    def __init__(
        self,
        base_model: Union[nn.Module, type],
        ensemble_size: int = 5,
        device: str = 'cpu',
        **base_model_kwargs
    ) -> None:
        super().__init__()
        self.ensemble_size = ensemble_size
        self.device = device
        
        # Create ensemble members
        self.models = nn.ModuleList()
        for i in range(ensemble_size):
            if isinstance(base_model, type):
                # If base_model is a class, instantiate it with kwargs
                model = base_model(**base_model_kwargs)
            else:
                # Otherwise, make a deep copy of the provided instance
                model = deepcopy(base_model)
            self.models.append(model)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes predictions from all ensemble members.
        
        Args:
            x: Input tensor [batch_size, ...]
            
        Returns:
            List of predictions from each ensemble member
        """
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        return outputs
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make prediction with uncertainty estimates.
        
        Args:
            x: Input tensor [batch_size, ...]
            
        Returns:
            Dictionary with mean and variance of predictions
        """
        with torch.no_grad():
            # Get predictions from all ensemble members
            predictions = self.forward(x)
            
            # Stack predictions [ensemble_size, batch_size, output_dim]
            stacked_preds = torch.stack(predictions)
            
            # Calculate mean across ensemble dimension
            mean = torch.mean(stacked_preds, dim=0)
            
            # Calculate variance across ensemble dimension
            variance = torch.var(stacked_preds, dim=0, unbiased=True)
            
            return {'mean': mean, 'variance': variance}
            
    def predict_with_uncertainties(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make prediction with epistemic and aleatoric uncertainty estimates.
        
        Args:
            x: Input tensor [batch_size, ...]
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        with torch.no_grad():
            # For standard ensemble, this is the same as predict
            return self.predict(x)


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
                    raise ValueError("Model output format not recognized for heteroscedastic uncertainty")
                
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
                'mean': ensemble_mean,
                'variance': total_var,
                'epistemic_variance': epistemic_var,
                'aleatoric_variance': aleatoric_var
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
        **optimizer_kwargs
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
                val_loader=val_loader
            )
            
            histories.append(history)
            
        return {'member_histories': histories}
    
    def _train_single_model(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        verbose: bool,
        val_loader: Optional[torch.utils.data.DataLoader]
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
                    print(f"Epoch {epoch+1}/{epochs} - Train loss: {avg_train_loss:.4f} - Val loss: {avg_val_loss:.4f}")
            else:
                if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train loss: {avg_train_loss:.4f}")
        
        return {
            'train_loss': train_losses,
            'val_loss': val_losses
        }


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
        device: str = 'cpu'
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
            device=device
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
        means = outputs[:, :, :self.output_size]  # [batch_size, ensemble_size, output_size]
        log_vars = outputs[:, :, self.output_size:]  # [batch_size, ensemble_size, output_size]
        
        return {
            'means': means,
            'log_vars': log_vars
        }
    
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
            
            means = outputs['means']  # [batch_size, ensemble_size, output_size]
            log_vars = outputs['log_vars']  # [batch_size, ensemble_size, output_size]
            
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
                'mean': ensemble_mean,
                'variance': total_var,
                'epistemic_variance': epistemic_var,
                'aleatoric_variance': aleatoric_var
            }