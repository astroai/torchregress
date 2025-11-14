"""
This module provides nn.Module wrappers for torch.distributions objects.
"""
from .normal import Normal
from .poisson import Poisson
from .categorical import Categorical
from .evidential import Evidential
from .mdn import MDN
