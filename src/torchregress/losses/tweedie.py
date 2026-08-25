"""
Tweedie distribution loss functions for regression.

The Tweedie distribution is a family of probability distributions that includes
many common distributions like the normal, Poisson, and gamma distributions.
It's defined through the variance function V(μ) = μ^p where p is the power parameter.

Common special cases:
- p=0: Normal distribution
- p=1: Poisson distribution
- p=2: Gamma distribution
- p=3: Inverse Gaussian distribution
- 1<p<2: Compound Poisson-Gamma (useful for mixed discrete-continuous data)
"""

from typing import Any, Optional

import torch

from .base import RegressionLoss
from .loss_registry import register_regression_loss


@register_regression_loss("tweedie")
class TweedieLoss(RegressionLoss):
    """
    Tweedie loss function for regression.

    The Tweedie distribution is a family of distributions defined by the variance function:
    V(μ) = μ^p

    The loss is derived from the negative log-likelihood of the corresponding Tweedie
    distribution with the given power parameter p.

    Common special cases:
    - p=0: Normal distribution (MSE loss)
    - p=1: Poisson distribution
    - p=2: Gamma distribution
    - p=3: Inverse Gaussian distribution
    - 1<p<2: Compound Poisson-Gamma (useful for mixed discrete-continuous data)

    Args:
        p: Power parameter defining the variance function V(μ) = μ^p. Default: 1.5
        eps: Small constant for numerical stability. Default: 1e-8
        reduction: Specifies the reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        link: Link function, 'log' or 'identity'. Default is 'log' for p>=1, 'identity' for p=0.

    Example:
        >>> # Compound Poisson-Gamma (p=1.5)
        >>> loss_fn = TweedieLoss(p=1.5, link='log')
        >>> y_pred = torch.log(torch.tensor([1.0, 2.0, 3.0]))  # log(mu)
        >>> target = torch.tensor([0.0, 2.0, 5.0])
        >>> loss_fn(y_pred, target)
        tensor(1.6283)

        >>> # Gamma distribution (p=2)
        >>> loss_fn = TweedieLoss(p=2.0)
        >>> y_pred = torch.log(torch.tensor([1.0, 2.0, 3.0]))
        >>> target = torch.tensor([1.0, 2.0, 3.0])
        >>> loss_fn(y_pred, target)
        tensor(0.0000)  # Perfect prediction

    References
    ----------
    .. [1] Tweedie, M. C. K. (1984). An Index which Distinguishes between Some Important
       Exponential Families. In *Statistics: Applications and New Directions*,
       Indian Statistical Institute, Calcutta, 579–604.
       https://en.wikipedia.org/wiki/Tweedie_distribution
    """

    def __init__(
        self,
        p: float = 1.5,
        eps: float = 1e-8,
        reduction: str = "mean",
        link: Optional[str] = None,
        log_input: Optional[bool] = None,
    ) -> None:
        super().__init__(reduction=reduction)
        self.p = p
        self.eps = eps

        # Handle log_input parameter (alias for link)
        if log_input is not None:
            link = "log" if log_input else "identity"

        # Set default link function based on p
        if link is None:
            self.link = "identity" if p == 0 else "log"
        else:
            if link not in ["identity", "log"]:
                raise ValueError(f"link must be 'identity' or 'log', got {link}")
            self.link = link

        # Validate p value
        if p < 0:
            raise ValueError(f"Power parameter p must be non-negative, got {p}")
        if 0 < p < 1:
            raise ValueError(f"Power parameter p between 0 and 1 is not supported, got {p}")

    def _get_mean(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Get mean parameter from prediction based on link function.

        Args:
            y_pred: Model predictions

        Returns:
            Mean parameter μ
        """
        if self.link == "log":
            # A3: clamp in log space before exponentiation to avoid overflow
            mu = torch.exp(y_pred.clamp(max=30.0))
        else:  # identity
            mu = y_pred
        return torch.clamp(mu, min=self.eps)

    def _normal_loss(self, target: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Normal distribution loss (p=0).

        Args:
            target: Target values
            mu: Mean parameter

        Returns:
            Loss tensor
        """
        return (target - mu) ** 2 / 2

    def _poisson_loss(self, target: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Poisson distribution loss (p=1).

        Args:
            target: Target values
            mu: Mean parameter

        Returns:
            Loss tensor
        """
        target_safe = torch.where(target > 0, target, torch.ones_like(target))
        term_nz = target * torch.log(target_safe / (mu + self.eps) + self.eps) - (target - mu)
        return torch.where(target == 0, mu, term_nz)

    def _gamma_loss(self, target: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Gamma distribution loss (p=2).

        Args:
            target: Target values
            mu: Mean parameter

        Returns:
            Loss tensor
        """
        return torch.log(mu / (target + self.eps) + self.eps) + target / (mu + self.eps) - 1

    def _inverse_gaussian_loss(self, target: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Inverse Gaussian loss (p=3).

        Args:
            target: Target values
            mu: Mean parameter

        Returns:
            Loss tensor
        """
        return 0.5 * (target - mu) ** 2 / (target * mu**2 + self.eps)

    def _compound_poisson_loss(self, target: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Compound Poisson-Gamma loss (1<p<2).

        Args:
            target: Target values
            mu: Mean parameter

        Returns:
            Loss tensor
        """
        # Constants for readability
        p1 = 1.0 - self.p
        p2 = 2.0 - self.p

        # For target == 0 (half-unit-deviance)
        loss_zero = (mu**p2) / p2

        # For target > 0 (avoid negative base raised to fraction in target ** p2)
        target_safe = torch.where(target > 0, target, torch.ones_like(target))
        term1 = (target_safe**p2) / (p1 * p2)
        term2 = target * (mu**p1) / p1
        term3 = (mu**p2) / p2
        loss_nz = term1 - term2 + term3

        return torch.where(target == 0, loss_zero, loss_nz)

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate Tweedie loss.

        Args:
            y_pred: Predicted values (log(μ) if link=='log', μ if link=='identity')
                [batch_size, ...]
            target: Target values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]

        Returns:
            Tweedie loss value
        """
        # Validate inputs
        self._validate_inputs(y_pred, target, mask)

        # Get mean parameter μ
        mu = self._get_mean(y_pred)

        # Calculate loss based on Tweedie deviance
        _dispatch: dict = {
            0: self._normal_loss,
            1: self._poisson_loss,
            2: self._gamma_loss,
            3: self._inverse_gaussian_loss,
        }
        fn = _dispatch.get(self.p)
        if fn is not None:
            loss = fn(target, mu)
        elif 1 < self.p < 2:
            loss = self._compound_poisson_loss(target, mu)
        else:
            raise ValueError(
                f"Tweedie power parameter p={self.p} not supported. "
                f"Must be 0, 1, 2, 3, or between 1 and 2."
            )

        # Apply reduction with mask and weights
        return self._reduce(loss, mask, weights)


# ponytail: convenience aliases for fixed-power Tweedie losses.
@register_regression_loss("gamma")
def GammaLoss(eps: float = 1e-8, reduction: str = "mean", link: str = "log") -> TweedieLoss:
    return TweedieLoss(p=2, eps=eps, reduction=reduction, link=link)


@register_regression_loss("inverse_gaussian")
def InverseGaussianLoss(
    eps: float = 1e-8, reduction: str = "mean", link: str = "log"
) -> TweedieLoss:
    return TweedieLoss(p=3, eps=eps, reduction=reduction, link=link)


@register_regression_loss("compound_poisson")
def CompoundPoissonLoss(
    p: float = 1.5, eps: float = 1e-8, reduction: str = "mean", link: str = "log"
) -> TweedieLoss:
    if not (1 < p < 2):
        raise ValueError(f"For CompoundPoissonLoss, p must be between 1 and 2, got {p}")
    return TweedieLoss(p=p, eps=eps, reduction=reduction, link=link)


def tweedie_loss(
    y_pred: torch.Tensor,
    target: torch.Tensor,
    p: float = 1.5,
    mask: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    link: Optional[str] = None,
) -> torch.Tensor:
    """Functional wrapper for :class:`TweedieLoss`.

    Equivalent to ``TweedieLoss(p=p, reduction=reduction, link=link)``
    followed by a ``forward`` call.

    Args:
        y_pred: Predicted values (log(μ) if ``link='log'``).
        target: Non-negative target values.
        p: Tweedie power parameter (0, 1, 2, 3, or in (1, 2)).
        mask: Optional boolean mask of valid entries.
        weights: Optional per-element weights.
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.
        link: ``'log'`` (default for p>0) or ``'identity'``.

    Returns:
        Tweedie deviance loss value.
    """
    return TweedieLoss(p=p, reduction=reduction, link=link)(
        y_pred, target, mask=mask, weights=weights
    )
