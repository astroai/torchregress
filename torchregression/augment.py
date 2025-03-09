import torch
import torch.nn.functional as F

class Augmentation:
    """Base class for data augmentation in deep ensembles."""
    def __call__(self, x, model=None, targets=None, loss_fn=None):
        """Apply augmentation to input data."""
        return x

class GaussianNoiseAugmentation(Augmentation):
    """Gaussian noise augmentation."""
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std
        
    def __call__(self, x, model=None, targets=None, loss_fn=None):
        """Add Gaussian noise to the input."""
        return x + torch.randn_like(x) * self.std

class AdversarialAugmentation(Augmentation):
    """Adversarial augmentation as in the Deep Ensembles paper."""
    def __init__(self, epsilon=0.01, step_size=0.001):
        super().__init__()
        self.epsilon = epsilon
        self.step_size = step_size
        
    def __call__(self, x, model, targets=None, loss_fn=None):
        """Generate adversarial examples using FGSM-like approach."""
        if targets is None or model is None:
            return x
            
        if loss_fn is None:
            # Default to MSE loss for regression
            loss_fn = F.mse_loss
            
        x_adv = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = model(x_adv)
        
        # Calculate loss based on model outputs
        if isinstance(outputs, tuple):
            # For models that return multiple outputs like (mu, log_var)
            pred = outputs[0]  # Use the mean prediction
            loss = loss_fn(pred, targets)
        else:
            loss = loss_fn(outputs, targets)
        
        # Compute gradient
        loss.backward()
        
        # Create adversarial examples
        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv + self.step_size * grad_sign
            x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)
        
        return x_adv.detach()