import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Type, Callable
# Import the MDN ensemble functionality from mdn module
from .mixtures import MDNEnsembleModel, MDNEnsemble

class BatchEnsembleLinear(nn.Module):
    """
    BatchEnsemble linear layer as described in 'BatchEnsemble: An Alternative 
    Approach to Efficient Ensemble and Lifelong Learning' (Wen et al., 2020)
    """
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        # Shared "slow" weights
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # Per-ensemble "fast" weights (multiplicative)
        self.weight_r = nn.Parameter(torch.Tensor(ensemble_size, in_features))
        self.weight_s = nn.Parameter(torch.Tensor(ensemble_size, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize using standard methods for linear layers
        nn.init.kaiming_uniform_(self.weight, a=torch.sqrt(torch.tensor(5.0)))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / torch.sqrt(torch.tensor(fan_in, dtype=torch.float))
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Initialize fast weights to be close to identity mapping
        nn.init.normal_(self.weight_r, mean=1.0, std=0.1)
        nn.init.normal_(self.weight_s, mean=1.0, std=0.1)

    def forward(self, x):
        # More efficient implementation using broadcasting
        batch_size = x.shape[0]
        
        # Reshape for batch processing
        x_e = x.repeat(self.ensemble_size, 1)  # [e*b, in_features]
        
        # Apply fast weights using broadcasting
        r = self.weight_r.repeat_interleave(batch_size, dim=0)  # [e*b, in_features]
        x_er = x_e * r  # [e*b, in_features]
        
        # Standard linear transformation
        out = F.linear(x_er, self.weight, self.bias)  # [e*b, out_features]
        
        # Apply second fast weight
        s = self.weight_s.repeat_interleave(batch_size, dim=0)  # [e*b, out_features]
        out = out * s  # [e*b, out_features]
        
        return out


class HeteroscedasticBatchEnsembleModel(nn.Module):
    """
    A batch ensemble model that predicts mean and variance for heteroscedastic regression.
    All ensemble members share the same architecture but have different fast weights.
    """
    def __init__(self, 
                in_features: int, 
                out_features: int, 
                ensemble_size: int, 
                hidden_sizes: List[int] = [64, 64],
                activation: nn.Module = nn.ReLU()):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.out_features = out_features
        
        # Build network with configurable architecture
        self.layers = nn.ModuleList()
        layer_sizes = [in_features] + hidden_sizes
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(BatchEnsembleLinear(layer_sizes[i], layer_sizes[i+1], ensemble_size))
            
        # Output layer produces both mean and log variance
        self.output_layer = BatchEnsembleLinear(layer_sizes[-1], 2 * out_features, ensemble_size)
        self.activation = activation

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Pass through hidden layers
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            
        # Output layer
        x = self.output_layer(x)
        
        # Split and reshape outputs
        mu, log_var = torch.chunk(x, 2, dim=1)
        
        # Reshape to [ensemble_size, batch_size, output_dim]
        mu = mu.view(self.ensemble_size, batch_size, self.out_features)
        log_var = log_var.view(self.ensemble_size, batch_size, self.out_features)
        
        return mu, log_var


class BaseEnsembleModel(nn.Module):
    """Base class for a single model in a deep ensemble."""
    def __init__(self, 
                in_features: int, 
                out_features: int, 
                hidden_sizes: List[int] = [64, 64],
                activation: nn.Module = nn.ReLU()):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self._build()
    
    def _build(self):
        """Build the model architecture."""
        raise NotImplementedError
        
    def forward(self, x):
        """Forward pass through the model."""
        raise NotImplementedError

class HeteroscedasticEnsembleModel(BaseEnsembleModel):
    """A model that outputs mean and log variance for heteroscedastic regression."""
    def _build(self):
        layers = []
        layer_sizes = [self.in_features] + self.hidden_sizes
        
        # Create hidden layers
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(self.activation)
            
        self.hidden_net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(layer_sizes[-1], 2 * self.out_features)
        
    def forward(self, x):
        x = self.hidden_net(x)
        out = self.output_layer(x)
        
        # Split the output into mean and log variance
        mu, log_var = torch.split(out, self.out_features, dim=1)
        return mu, log_var

class DeepEnsemble(nn.Module):
    """
    Deep Ensemble model with optional augmentation.
    
    This implements the approach from:
    "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
    by Lakshminarayanan, Pritzel, Blundell (2017)
    
    Note: For MDN-specific ensembling, consider using MDNEnsemble from the mdn module.
    """
    def __init__(self, 
                model_class: Type[nn.Module] | Callable[..., nn.Module], 
                in_features: Optional[int] = None, 
                out_features: Optional[int] = None, 
                ensemble_size: int = 5, 
                hidden_sizes: Optional[List[int]] = None,
                activation: Optional[nn.Module] = None,
                augmentation: Optional[Callable] = None, 
                *model_args,
                **model_kwargs):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.in_features = in_features
        self.out_features = out_features
        
        # Instantiate the ensemble models
        self.models = nn.ModuleList()
        for _ in range(ensemble_size):
            if isinstance(model_class, type) and issubclass(model_class, BaseEnsembleModel):
                # If it's one of our BaseEnsembleModel classes
                kwargs = model_kwargs.copy()
                if hidden_sizes is not None:
                    kwargs['hidden_sizes'] = hidden_sizes
                if activation is not None:
                    kwargs['activation'] = activation
                model = model_class(in_features, out_features, **kwargs)
            elif isinstance(model_class, type) and issubclass(model_class, nn.Module):
                # If it's any other nn.Module class
                model = model_class(in_features, out_features, *model_args, **model_kwargs)
            else:
                # Assume it's a factory function that returns a model
                model = model_class(*model_args, **model_kwargs)
                
            self.models.append(model)
            
        self.augmentation = augmentation
        self.loss_fn = None
        
    def forward(self, x, targets=None):
        """
        Forward pass through each model in the ensemble.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            targets: Optional targets used for adversarial augmentation
            
        Returns:
            List of outputs from each model in the ensemble
        """
        outputs = []
        
        for model in self.models:
            if self.training and self.augmentation is not None and targets is not None:
                # Apply augmentation during training
                x_aug = self.augmentation(x, model, targets, self.loss_fn)
                output = model(x_aug)
            else:
                output = model(x)
                
            outputs.append(output)
            
        return outputs
    
    def predict(self, x):
        """
        Make a prediction with the ensemble by combining outputs.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            Prediction based on model type (mean and variance for heteroscedastic models,
            or averaged output for standard models)
        """
        with torch.no_grad():
            outputs = self(x)
            
            # Check the type of outputs to determine how to combine
            if isinstance(outputs[0], tuple):
                if len(outputs[0]) == 2:  # Heteroscedastic model (mean, log_var)
                    mus = torch.stack([out[0] for out in outputs])
                    log_vars = torch.stack([out[1] for out in outputs])
                    vars = torch.exp(log_vars)
                    
                    # Mean of means (predictive mean)
                    mean_mu = torch.mean(mus, dim=0)
                    
                    # Mean of variances (aleatoric uncertainty)
                    mean_var_aleatoric = torch.mean(vars, dim=0)
                    
                    # Variance of means (epistemic uncertainty)
                    var_means = torch.var(mus, dim=0, unbiased=False)
                    
                    # Total predictive variance
                    mean_var = mean_var_aleatoric + var_means
                    
                    return mean_mu, mean_var
                    
                elif len(outputs[0]) == 3:  # MDN model - delegate to MDNEnsemble
                    # For backward compatibility, handle MDN outputs similar to before
                    # but recommend using MDNEnsemble from mdn module instead
                    mixture_weights = torch.stack([out[0] for out in outputs]).mean(dim=0)
                    means = torch.stack([out[1] for out in outputs]).mean(dim=0)
                    log_scales = torch.stack([out[2] for out in outputs]).mean(dim=0)
                    
                    # Calculate overall mean and variance
                    overall_mean = torch.sum(mixture_weights.unsqueeze(-1) * means, dim=1)
                    vars = torch.exp(2 * log_scales)
                    weighted_vars = torch.sum(mixture_weights.unsqueeze(-1) * vars, dim=1)
                    mean_diffs = means - overall_mean.unsqueeze(1)
                    weighted_mean_vars = torch.sum(mixture_weights.unsqueeze(-1) * (mean_diffs ** 2), dim=1)
                    overall_var = weighted_vars + weighted_mean_vars
                    
                    return (mixture_weights, means, log_scales), (overall_mean, overall_var)
            
            try:
                # Try stacking the outputs (works for tensor outputs)
                stacked_outputs = torch.stack(outputs)
                
                # Calculate mean prediction and epistemic uncertainty
                mean_pred = torch.mean(stacked_outputs, dim=0)
                epistemic_var = torch.var(stacked_outputs, dim=0, unbiased=False)
                
                return mean_pred, epistemic_var
            except:
                # If outputs can't be stacked, just return the list
                return outputs

    def set_loss_fn(self, loss_fn):
        """
        Set the loss function used for adversarial augmentation.
        
        Args:
            loss_fn: Loss function that takes predictions and targets
        """
        self.loss_fn = loss_fn
        return self
    
    @classmethod
    def from_model_factory(cls, model_factory, ensemble_size=5, augmentation=None):
        """
        Create an ensemble from a model factory function.
        
        Args:
            model_factory: A function that returns a new model instance when called
            ensemble_size: Number of models in the ensemble
            augmentation: Optional augmentation function
            
        Returns:
            A DeepEnsemble instance
        """
        return cls(model_factory, ensemble_size=ensemble_size, augmentation=augmentation)