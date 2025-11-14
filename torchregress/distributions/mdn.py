"""
Mixture Density Network distribution layer.
"""
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical, MixtureSameFamily


class MDN(nn.Module):
    """
    A module that produces a Mixture of Normal distributions.
    """

    def __init__(self, in_features: int, out_features: int, n_components: int) -> None:
        super().__init__()
        self.n_components = n_components
        self.out_features = out_features

        self.pi_layer = nn.Linear(in_features, n_components)
        self.mean_layer = nn.Linear(in_features, n_components * out_features)
        self.log_std_layer = nn.Linear(in_features, n_components * out_features)

    def forward(self, x: torch.Tensor) -> MixtureSameFamily:
        """
        Computes the mixture distribution from the input features.
        """
        pi_logits = self.pi_layer(x)
        means = self.mean_layer(x).view(-1, self.n_components, self.out_features)
        log_stds = self.log_std_layer(x).view(-1, self.n_components, self.out_features)
        stds = torch.exp(log_stds)

        mix = Categorical(logits=pi_logits)
        comp = Normal(loc=means, scale=stds)
        return MixtureSameFamily(mix, comp)
