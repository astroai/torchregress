# Implementation Plan: Scalability & Advanced Quantile Features

This document outlines the implementation plan for four major features:
1. Native support for streaming/minibatch gradient computation
2. Online learning support
3. GPU batch optimization
4. Quantile neural networks beyond simple quantile regression

---

## 1. Streaming/Minibatch Gradient Computation

### Goal
Enable efficient processing of data streams and support for incremental gradient updates without requiring full-batch computation.

### Current State
- All losses in `torchregress/losses/base.py` compute losses over full tensors
- No streaming or incremental gradient support
- The `mask` parameter exists but doesn't provide true streaming semantics

### Implementation Plan

#### 1.1 Streaming Loss Wrapper (`torchregress/losses/streaming.py`)

Create a `StreamingLoss` wrapper that accumulates gradients across mini-batches:

```python
class StreamingLoss(nn.Module):
    """
    Wrapper for streaming/incremental loss computation.
    
    Accumulates gradients across multiple mini-batches before optimizer step.
    """
    
    def __init__(self, base_loss: BaseLoss, accumulation_steps: int = 1):
        super().__init__()
        self.base_loss = base_loss
        self.accumulation_steps = accumulation_steps
        self._step_count = 0
        self._accumulated_loss = None
        
    def forward(self, y_pred, target, mask=None, weights=None):
        # Standard forward - returns per-sample or per-batch loss
        return self.base_loss(y_pred, target, mask, weights)
        
    def accumulate(self, loss: torch.Tensor):
        """Accumulate loss for gradient accumulation."""
        if self._accumulated_loss is None:
            self._accumulated_loss = loss
        else:
            self._accumulated_loss = self._accumulated_loss + loss
        self._step_count += 1
        
    def should_update(self) -> bool:
        return self._step_count >= self.accumulation_steps
        
    def get_accumulated_loss(self) -> torch.Tensor:
        """Return normalized accumulated loss for backward."""
        return self._accumulated_loss / self._step_count
        
    def reset(self):
        """Reset accumulator state."""
        self._accumulated_loss = None
        self._step_count = 0
```

#### 1.2 Per-Sample Gradient Support

Add support for per-sample gradients using `torch.func.vmap`:

```python
from torch.func import vmap, grad

class PerSampleGradientLoss(nn.Module):
    """
    Loss that computes per-sample gradients efficiently.
    
    Uses vmap for vectorized per-sample gradient computation.
    Useful for meta-learning, differential privacy, and sample-level analysis.
    """
    
    def __init__(self, model: nn.Module, loss_fn: BaseLoss):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        
    def per_sample_grads(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute gradients for each sample in the batch."""
        def compute_loss_for_sample(xi, yi):
            pred = self.model(xi.unsqueeze(0))
            return self.loss_fn(pred, yi.unsqueeze(0))
        
        # Vectorized per-sample gradients
        grads = vmap(grad(compute_loss_for_sample))(x, y)
        return grads
```

#### 1.3 Streaming DataLoader Utilities (`torchregress/data/streaming.py`)

Create utilities for streaming data:

```python
class StreamingDataLoader:
    """
    DataLoader that supports streaming/incremental data access.
    
    Can yield data from generators, databases, or file streams.
    """
    
    def __init__(self, data_source, batch_size: int, collate_fn=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.collate_fn = collate_fn or default_collate
        
    def __iter__(self):
        batch = []
        for item in self.data_source:
            batch.append(item)
            if len(batch) >= self.batch_size:
                yield self.collate_fn(batch)
                batch = []
        if batch:
            yield self.collate_fn(batch)
```

### Files to Create/Modify

- **Create**: `torchregress/losses/streaming.py` - StreamingLoss, PerSampleGradientLoss
- **Create**: `torchregress/data/__init__.py` - Data module
- **Create**: `torchregress/data/streaming.py` - StreamingDataLoader, StreamingDataset
- **Modify**: `torchregress/losses/__init__.py` - Export new classes

---

## 2. Online Learning Support

### Goal
Support incremental/continual learning where the model updates as new data arrives without full dataset access.

### Implementation Plan

#### 2.1 Online Loss Wrapper (`torchregress/losses/online.py`)

```python
class OnlineLoss(nn.Module):
    """
    Loss that maintains running statistics for online learning.
    
    Supports exponential moving average updates for loss components
    and provides forgetting mechanisms for concept drift detection.
    """
    
    def __init__(
        self,
        base_loss: BaseLoss,
        forget_factor: float = 0.99,
        track_statistics: bool = True
    ):
        super().__init__()
        self.base_loss = base_loss
        self.forget_factor = forget_factor
        self.track_statistics = track_statistics
        
        self._running_loss = None
        self._sample_count = 0
        
    def forward(self, y_pred, target, mask=None, weights=None):
        loss = self.base_loss(y_pred, target, mask, weights)
        
        if self.track_statistics:
            self._update_statistics(loss, weights)
            
        return loss
    
    def _update_statistics(self, loss: torch.Tensor, weights: Optional[torch.Tensor]):
        """Update running statistics with exponential moving average."""
        batch_loss = loss.detach()
        
        if self._running_loss is None:
            self._running_loss = batch_loss.mean()
        else:
            self._running_loss = (
                self.forget_factor * self._running_loss + 
                (1 - self.forget_factor) * batch_loss.mean()
            )
        
        if weights is not None:
            self._sample_count += weights.sum().item()
        else:
            self._sample_count += loss.numel()
    
    def get_statistics(self) -> Dict[str, float]:
        """Return running statistics for monitoring."""
        return {
            "running_loss": self._running_loss.item() if self._running_loss else None,
            "sample_count": self._sample_count
        }
```

#### 2.2 Incremental Optimizer Wrapper

```python
class IncrementalOptimizer:
    """
    Wrapper for online/incremental optimization.
    
    Provides:
    - Partial fit support
    - Learning rate adaptation
    - Gradient clipping for stability
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer_class = torch.optim.Adam,
        lr: float = 0.001,
        gradient_clip_value: Optional[float] = 1.0
    ):
        self.model = model
        self.optimizer = optimizer_class(model.parameters(), lr=lr)
        self.gradient_clip_value = gradient_clip_value
        
    def partial_fit(self, X: torch.Tensor, y: torch.Tensor, loss_fn: BaseLoss):
        """
        Perform one incremental update.
        
        Similar to scikit-learn's partial_fit API.
        """
        self.optimizer.zero_grad()
        
        pred = self.model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        
        if self.gradient_clip_value:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.gradient_clip_value
            )
            
        self.optimizer.step()
        
    def learning_rate_schedule(self, metric: float, patience: int = 5):
        """
        Adapt learning rate based on recent performance.
        
        Reduces LR when metric plateaus.
        """
        # Implementation of learning rate adaptation
        pass
```

#### 2.3 Concept Drift Detection

```python
class DriftDetector:
    """
    Detects concept drift in online learning scenarios.
    
    Uses statistical tests to detect distribution shifts.
    """
    
    def __init__(self, window_size: int = 100, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.recent_predictions = collections.deque(maxlen=window_size)
        self.recent_targets = collections.deque(maxlen=window_size)
        
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Update with new predictions and targets."""
        self.recent_predictions.extend(y_pred.cpu().tolist())
        self.recent_targets.extend(y_true.cpu().tolist())
        
    def detect_drift(self) -> bool:
        """Return True if concept drift detected."""
        if len(self.recent_predictions) < self.window_size:
            return False
            
        # Use Kolmogorov-Smirnov test for drift detection
        # or simple statistical tests
        # ...
        return False
```

### Files to Create/Modify

- **Create**: `torchregress/losses/online.py` - OnlineLoss, OnlineTrainingLoop
- **Create**: `torchregress/optim/__init__.py` - Optimization module
- **Create**: `torchregress/optim/online.py` - IncrementalOptimizer, DriftDetector
- **Modify**: `torchregress/losses/__init__.py` - Export new classes

---

## 3. GPU Batch Optimization

### Goal
Provide utilities for efficient large-batch training on GPUs including gradient accumulation, mixed precision, and memory optimization.

### Implementation Plan

#### 3.1 Gradient Accumulation Utilities (`torchregress/optim/accumulation.py`)

```python
class GradientAccumulator:
    """
    Handles gradient accumulation for large effective batch sizes.
    
    Allows training with large effective batch sizes that don't fit in GPU memory.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 4,
        scale_loss: bool = True
    ):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.scale_loss = scale_loss
        self.step_count = 0
        
    def backward(self, loss: torch.Tensor):
        """Accumulate gradients and optionally update."""
        if self.scale_loss:
            loss = loss / self.accumulation_steps
            
        loss.backward()
        
        self.step_count += 1
        
        if self.step_count >= self.accumulation_steps:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step_count = 0
            
    def should_step(self) -> bool:
        return self.step_count >= self.accumulation_steps
```

#### 3.2 Mixed Precision Training Support

```python
class MixedPrecisionTrainer:
    """
    Utility for mixed precision (FP16/BF16) training.
    
    Wraps model and optimizer for automatic loss scaling.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        use_bf16: bool = True,
        loss_scale: Optional[float] = "dynamic"
    ):
        self.model = model
        self.optimizer = optimizer
        self.use_bf16 = use_bf16
        
        # Automatic mixed precision setup
        self.scaler = torch.cuda.amp.GradScaler() if not use_bf16 else None
        
    def train_step(self, batch, loss_fn):
        """Perform one training step with mixed precision."""
        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if self.use_bf16 else torch.float16):
            pred = self.model(batch["x"])
            loss = loss_fn(pred, batch["y"])
            
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # BF16 doesn't need scaler
            loss.backward()
            self.optimizer.step()
            
        self.optimizer.zero_grad()
```

#### 3.3 Memory-Efficient Training Utilities

```python
class MemoryEfficientTrainer:
    """
    Utilities for memory-efficient training.
    
    Features:
    - Gradient checkpointing
    - In-place operations
    - Optional context management for activations
    """
    
    @staticmethod
    def enable_gradient_checkpointing(model: nn.Module):
        """Enable gradient checkpointing to save memory."""
        # Recursively enable checkpointing on all modules
        def apply_checkpointing(module):
            if hasattr(module, 'forward'):
                module.forward = torch.utils.checkpoint.checkpoint(
                    module.forward, use_reentrant=False
                )
            for child in module.children():
                apply_checkpointing(child)
        apply_checkpointing(model)
        
    @staticmethod
    def clear_cache():
        """Clear GPU cache to free memory."""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get current GPU memory statistics."""
        return {
            "allocated": torch.cuda.memory_allocated() / 1e9,
            "reserved": torch.cuda.memory_reserved() / 1e9,
            "max_allocated": torch.cuda.max_memory_allocated() / 1e9
        }
```

#### 3.4 Batch Size Finder

```python
class BatchSizeFinder:
    """
    Automatically find maximum batch size that fits in GPU memory.
    
    Uses exponential search with binary search refinement.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        initial_batch_size: int = 32,
        growth_factor: float = 2.0
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.initial_batch_size = initial_batch_size
        self.growth_factor = growth_factor
        
    def find_max_batch_size(
        self, 
        sample_shape: Tuple[int, ...],
        max_attempts: int = 10
    ) -> int:
        """Find maximum batch size that doesn't cause OOM."""
        
        batch_size = self.initial_batch_size
        
        for attempt in range(max_attempts):
            try:
                x = torch.randn(*sample_shape, device='cuda')
                y = torch.randn(sample_shape[0], device='cuda')
                
                torch.cuda.empty_cache()
                
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                loss.backward()
                
                self.model.zero_grad()
                torch.cuda.empty_cache()
                
                batch_size = int(batch_size * self.growth_factor)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                raise
                
        return batch_size // self.growth_factor
```

### Files to Create/Modify

- **Create**: `torchregress/optim/__init__.py` - Optimization module
- **Create**: `torchregress/optim/accumulation.py` - GradientAccumulator
- **Create**: `torchregress/optim/mixed_precision.py` - MixedPrecisionTrainer
- **Create**: `torchregress/optim/memory.py` - MemoryEfficientTrainer, BatchSizeFinder
- **Modify**: `torchregress/__init__.py` - Export new modules

---

## 4. Quantile Neural Networks Beyond Quantile Regression

### Goal
Implement advanced quantile-based neural network architectures beyond simple quantile regression, including:
- Implicit Quantile Networks (IQN)
- Quantile Regression DQN (QR-DQN)
- Fully Parameterized Quantile Functions (FQF)
- CQRNN (Composite Quantile Regression NN)
- MQRNN (Multi-Quantile Recurrent NN)

### Implementation Plan

#### 4.1 Implicit Quantile Network (`torchregress/losses/iqn.py`)

```python
class ImplicitQuantileNetwork(nn.Module):
    """
    Implicit Quantile Network (IQN) for distributional RL.
    
    Learns to estimate quantile values for any quantile tau in [0, 1].
    Uses cosine embedding for continuous quantile representation.
    
    Reference: Dabney et al. "Implicit Quantile Networks for Distributional RL"
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        quantile_dim: int = 64,
        embed_dim: int = 128
    ):
        super().__init__()
        self.backbone = backbone
        self.quantile_dim = quantile_dim
        self.embed_dim = embed_dim
        
        # Cosine embedding for quantile fractions
        self.quantile_embedding = nn.Linear(quantile_dim, embed_dim)
        
        # Output head
        self.value_head = nn.Linear(embed_dim, 1)
        
    def forward(
        self, 
        x: torch.Tensor, 
        quantiles: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, ...]
            quantiles: Quantile fractions tau in [0, 1], [batch_size, n_quantiles]
                      If None, uses uniform quantiles
        
        Returns:
            Quantile predictions [batch_size, n_quantiles]
        """
        batch_size = x.shape[0]
        
        if quantiles is None:
            # Default: 100 uniform quantiles
            n_quantiles = 100
            quantiles = torch.linspace(0, 1, n_quantiles, device=x.device)
            quantiles = quantiles.unsqueeze(0).expand(batch_size, -1)
            
        # Compute cosine embedding of quantiles
        # tau -> cos(pi * tau * (1, 2, ..., K))
        quantile_embed = self._cosine_embedding(quantiles)  # [B, K, embed_dim]
        
        # Get feature representation
        features = self.backbone(x)  # [B, feature_dim]
        
        # Expand for quantile dimension
        features = features.unsqueeze(1).expand(-1, quantiles.shape[1], -1)
        
        # Combine features with quantile embedding
        combined = features * quantile_embed
        
        # Output quantile values
        quantile_values = self.value_head(combined).squeeze(-1)
        
        return quantile_values
    
    def _cosine_embedding(self, tau: torch.Tensor) -> torch.Tensor:
        """Create cosine embedding of quantile fractions."""
        # tau: [batch_size, n_quantiles]
        # Output: [batch_size, n_quantiles, quantile_dim]
        
        B, K = tau.shape
        i = torch.arange(self.quantile_dim, device=tau.device)
        
        # cos(pi * tau * i)
        embeddings = torch.cos(torch.pi * tau.unsqueeze(-1) * i)
        
        return self.quantile_embedding(embeddings)


class IQNLoss(nn.Module):
    """
    Loss function for Implicit Quantile Networks.
    
    Uses quantile regression loss with Huber smoothing for stability.
    """
    
    def __init__(self, kappa: float = 1.0):
        super().__init__()
        self.kappa = kappa
        
    def forward(
        self,
        predictions: torch.Tensor,  # [B, K]
        target: torch.Tensor,       # [B]
        quantiles: torch.Tensor      # [B, K]
    ) -> torch.Tensor:
        """
        Compute IQN loss.
        
        Args:
            predictions: Predicted quantile values [batch_size, n_quantiles]
            target: Target values [batch_size]
            quantiles: Quantile fractions tau [batch_size, n_quantiles]
        """
        # Expand target for broadcasting
        target = target.unsqueeze(1)  # [B, 1]
        
        # Compute TD error
        td_error = target - predictions  # [B, K]
        
        # Huber loss with quantile weighting
        huber = self._huber_loss(td_error, self.kappa)  # [B, K]
        
        # Quantile regression loss
        weight = torch.where(
            td_error > 0,
            quantiles,
            1 - quantiles
        )  # [B, K]
        
        loss = (weight * huber).mean()
        
        return loss
    
    @staticmethod
    def _huber_loss(x: torch.Tensor, kappa: float) -> torch.Tensor:
        """Huber loss with specified kappa."""
        return torch.where(
            x.abs() <= kappa,
            0.5 * x ** 2,
            kappa * (x.abs() - 0.5 * kappa)
        )
```

#### 4.2 Fully Parameterized Quantile Function (`torchregress/losses/fqf.py`)

```python
class FQFHead(nn.Module):
    """
    Fully Parameterized Quantile Function (FQF) head.
    
    Learns both the quantile fractions and the quantile values.
    
    Reference: Yang et al. "Fully Parameterized Quantile Function for 
               Distributional Reinforcement Learning"
    """
    
    def __init__(
        self,
        input_dim: int,
        num_quantiles: int = 200,
        tau_dim: int = 64
    ):
        super().__init__()
        self.num_quantiles = num_quantiles
        
        # Tau network: learns optimal quantile fractions
        self.tau_net = nn.Sequential(
            nn.Linear(input_dim, tau_dim),
            nn.ReLU(),
            nn.Linear(tau_dim, num_quantiles)
        )
        
        # Quantile value network
        self.value_net = nn.Sequential(
            nn.Linear(input_dim + tau_dim, tau_dim),
            nn.ReLU(),
            nn.Linear(tau_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            tau: Learned quantile fractions [batch_size, num_quantiles]
            values: Quantile values [batch_size, num_quantiles]
        """
        # Get tau logits and convert to fractions
        tau_logits = self.tau_net(x)
        tau = F.softmax(tau_logits, dim=-1)
        
        # Expand x for each quantile
        x_expanded = x.unsqueeze(1).expand(-1, self.num_quantiles, -1)
        
        # Create tau embedding
        i = torch.arange(self.num_quantiles, device=x.device)
        tau_embed = (tau.unsqueeze(-1) * i.float() / self.num_quantiles)
        
        # Combine and compute values
        combined = torch.cat([x_expanded, tau_embed], dim=-1)
        values = self.value_net(combined).squeeze(-1)
        
        return tau, values
```

#### 4.3 Composite Quantile Regression NN (`torchregress/losses/cqrnn.py`)

```python
class CompositeQuantileLoss(nn.Module):
    """
    Composite Quantile Regression Neural Network (CQRNN) loss.
    
    Combines multiple quantile losses with a regularization term
    that encourages smoothness between adjacent quantiles.
    """
    
    def __init__(
        self,
        quantiles: List[float],
        cqr_weight: float = 1.0,
        smoothness_weight: float = 0.1
    ):
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles))
        self.cqr_weight = cqr_weight
        self.smoothness_weight = smoothness_weight
        
    def forward(
        self,
        predictions: torch.Tensor,  # [B, K]
        target: torch.Tensor        # [B]
    ) -> torch.Tensor:
        """
        Compute CQRNN loss.
        
        Args:
            predictions: Predicted quantile values [batch_size, n_quantiles]
            target: Target values [batch_size]
        """
        B, K = predictions.shape
        
        # Expand target
        target = target.unsqueeze(1)  # [B, 1]
        
        # Quantile regression loss
        errors = target - predictions  # [B, K]
        
        # Asymmetric weights
        weights = torch.where(
            errors > 0,
            self.quantiles.unsqueeze(0),
            1 - self.quantiles.unsqueeze(0)
        )
        
        quantile_loss = (weights * errors.abs()).mean()
        
        # Smoothness penalty: encourage adjacent quantiles to be close
        if K > 1:
            diff = predictions[:, 1:] - predictions[:, :-1]
            smoothness_loss = (diff ** 2).mean()
        else:
            smoothness_loss = 0
            
        return self.cqr_weight * quantile_loss + self.smoothness_weight * smoothness_loss
```

#### 4.4 Multi-Quantile Recurrent Neural Network (`torchregress/losses/mqrnn.py`)

```python
class MQRNNEncoder(nn.Module):
    """
    Encoder for Multi-Quantile Recurrent Neural Network.
    
    Processes temporal sequences to create encoded representations.
    
    Reference: "A Multi-Horizon Quantile Recurrent Forecaster"
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence [batch_size, seq_len, input_dim]
            
        Returns:
            Encoded representation [batch_size, hidden_dim]
        """
        output, (hidden, cell) = self.lstm(x)
        # Use final hidden state
        return hidden[-1]


class MQRNNDecoder(nn.Module):
    """
    Decoder that predicts multiple quantiles for multiple horizons.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_quantiles: int,
        num_horizons: int,
        output_dim: int = 1
    ):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.num_horizons = num_horizons
        
        # Shared decoder
        self.decoder = nn.Linear(hidden_dim, hidden_dim)
        
        # Quantile-specific heads
        self.quantile_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_horizons * output_dim)
            for _ in range(num_quantiles)
        ])
        
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: Encoded representation [batch_size, hidden_dim]
            
        Returns:
            Quantile predictions [batch_size, num_quantiles, num_horizons]
        """
        decoded = torch.relu(self.decoder(hidden))
        
        predictions = []
        for head in self.quantile_heads:
            pred = head(decoded)
            pred = pred.view(-1, self.num_horizons)
            predictions.append(pred)
            
        return torch.stack(predictions, dim=1)


class MQRNN(nn.Module):
    """
    Full Multi-Quantile Recurrent Neural Network.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_quantiles: int = 9,
        num_horizons: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = MQRNNEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = MQRNNDecoder(
            hidden_dim, num_quantiles, num_horizons
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence [batch_size, seq_len, input_dim]
            
        Returns:
            Quantile predictions [batch_size, num_quantiles, num_horizons]
        """
        encoded = self.encoder(x)
        return self.decoder(encoded)
```

### Files to Create/Modify

- **Create**: `torchregress/losses/iqn.py` - ImplicitQuantileNetwork, IQNLoss
- **Create**: `torchregress/losses/fqf.py` - FQFHead, FQFLoss
- **Create**: `torchregress/losses/cqrnn.py` - CompositeQuantileLoss
- **Create**: `torchregress/losses/mqrnn.py` - MQRNNEncoder, MQRNNDecoder, MQRNN
- **Create**: `torchregress/models/quantile.py` - Pre-built quantile neural network models
- **Modify**: `torchregress/losses/__init__.py` - Export all new classes
- **Modify**: `torchregress/losses/loss_registry.py` - Register new losses

---

## Implementation Priority

1. **Phase 1 - Core Infrastructure**
   - Gradient Accumulation Utilities (`torchregress/optim/`)
   - Streaming Loss Wrapper (`torchregress/losses/streaming.py`)

2. **Phase 2 - Online Learning**
   - Online Loss Wrapper (`torchregress/losses/online.py`)
   - Incremental Optimizer (`torchregress/optim/online.py`)

3. **Phase 3 - GPU Optimization**
   - Mixed Precision Support (`torchregress/optim/mixed_precision.py`)
   - Memory-Efficient Utilities (`torchregress/optim/memory.py`)
   - Batch Size Finder

4. **Phase 4 - Advanced Quantile Networks**
   - Implicit Quantile Network (IQN)
   - Composite Quantile Regression NN
   - MQRNN for temporal data
   - FQF (Fully Parameterized Quantile Function)

---

## Testing Strategy

- Unit tests for each new class/module
- Integration tests with existing loss functions
- Benchmark tests comparing performance with/without optimizations
- Tests for edge cases (empty batches, NaN handling, device placement)
