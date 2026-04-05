with open("tests/test_viz.py", "r") as f:
    content = f.read()

content = content.replace("from unittest.mock import MagicMock, patch\n\nfrom torchregress.viz.utils import create_grid_figure\n", "")

content = content.replace("from unittest.mock import MagicMock, patch\nfrom torchregress.viz.utils import create_grid_figure\n", "")

new_content = """import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from unittest.mock import MagicMock, patch

from torchregress.viz.utils import create_grid_figure
from torchregress.viz.diagnostic import (
    plot_calibration_curve,
    plot_distribution_comparison,
    plot_prediction_intervals,
    plot_qq_plot,
    plot_reliability_diagram,
    plot_residual_histogram,
    plot_residuals,
)
"""

final_content = new_content + content[content.find("\nclass TestVizDiagnostic:"):]

with open("tests/test_viz.py", "w") as f:
    f.write(final_content)
