import os
from unittest import mock

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from torchregress.viz.utils import (
    add_annotations,
    add_identity_line,
    add_zero_line,
    create_color_palette,
    create_grid_figure,
    enable_latex_rendering,
    format_metric_label,
    save_figure,
    set_style,
)


class TestVizUtils:
    """Test visualization utilities."""

    def teardown_method(self):
        """Close all figures."""
        plt.close("all")

    def test_set_style(self):
        """Test set_style function."""
        # Test basic execution
        set_style(style="whitegrid", context="paper")

        # Test with custom rc parameters
        test_rc = {"lines.linewidth": 3.14}
        set_style(rc=test_rc)
        assert mpl.rcParams["lines.linewidth"] == 3.14

    def test_create_grid_figure_auto(self):
        """Test grid figure creation with auto sizing."""
        n_plots = 5
        fig, axes = create_grid_figure(n_plots=n_plots)

        assert isinstance(fig, Figure)
        assert len(axes) == n_plots
        assert isinstance(axes[0], plt.Axes)

        # In auto mode for 5 plots, it should create a 2x3 grid
        # The returned list length is exactly n_plots
        assert len(axes) == 5

    def test_create_grid_figure_explicit(self):
        """Test grid figure creation with explicit sizing."""
        n_plots = 4
        fig, axes = create_grid_figure(n_plots=n_plots, nrows=2, ncols=2)

        assert isinstance(fig, Figure)
        assert len(axes) == n_plots
        assert isinstance(axes[0], plt.Axes)

    def test_create_grid_figure_partial_explicit(self):
        """Test grid figure creation with partial explicit sizing."""
        fig, axes = create_grid_figure(n_plots=6, ncols=2)
        assert len(axes) == 6

        fig, axes = create_grid_figure(n_plots=6, nrows=2)
        assert len(axes) == 6

    def test_add_identity_line(self):
        """Test adding identity line to plot."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        add_identity_line(ax, color="red", linestyle=":", label="Identity")

        # Check that a line was added
        lines = ax.get_lines()
        assert len(lines) == 1
        assert lines[0].get_color() == "red"
        assert lines[0].get_linestyle() == ":"
        assert lines[0].get_label() == "Identity"

    def test_add_identity_line_disjoint_limits(self):
        """Test identity line when limits do not overlap."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 5)
        ax.set_ylim(10, 15)

        add_identity_line(ax)

        lines = ax.get_lines()
        assert len(lines) == 1
        # It should still add a line and restore the limits
        assert ax.get_xlim() == (0.0, 5.0)
        assert ax.get_ylim() == (10.0, 15.0)

    def test_add_zero_line_y(self):
        """Test adding horizontal zero line."""
        fig, ax = plt.subplots()
        add_zero_line(ax, axis="y", color="blue")

        lines = ax.get_lines()
        assert len(lines) == 1
        assert lines[0].get_color() == "blue"
        # Since it's axhline, it spans the x-axis, and y=0

    def test_add_zero_line_x(self):
        """Test adding vertical zero line."""
        fig, ax = plt.subplots()
        add_zero_line(ax, axis="x", color="green")

        lines = ax.get_lines()
        assert len(lines) == 1
        assert lines[0].get_color() == "green"

    def test_add_zero_line_invalid(self):
        """Test zero line with invalid axis."""
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="axis must be 'x' or 'y'"):
            add_zero_line(ax, axis="z")

    def test_save_figure(self, tmpdir):
        """Test saving a figure."""
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])

        output_dir = str(tmpdir.mkdir("figures"))
        filename = "test_fig"

        save_figure(fig, filename, directory=output_dir, formats=["png", "pdf"])

        # Check if files were created
        assert os.path.exists(os.path.join(output_dir, f"{filename}.png"))
        assert os.path.exists(os.path.join(output_dir, f"{filename}.pdf"))

    def test_add_annotations(self):
        """Test adding annotations to a plot."""
        fig, ax = plt.subplots()

        annotations = {"MSE": 0.12345, "Model": "Test"}
        add_annotations(ax, annotations, title="Metrics", loc="upper left")

        # Check that text was added to the axes
        texts = ax.texts
        assert len(texts) == 1

        text_content = texts[0].get_text()
        assert "Metrics" in text_content
        assert "MSE: 0.1235" in text_content  # 4 decimal places
        assert "Model: Test" in text_content

    def test_create_color_palette(self):
        """Test creating color palette."""
        n_colors = 5

        # Test getting hex colors
        hex_colors = create_color_palette(n_colors, as_hex=True)
        assert len(hex_colors) == n_colors
        assert isinstance(hex_colors[0], str)
        assert hex_colors[0].startswith("#")

        # Test getting RGB tuples
        rgb_colors = create_color_palette(n_colors, as_hex=False)
        assert len(rgb_colors) == n_colors
        assert isinstance(rgb_colors[0], tuple)
        assert len(rgb_colors[0]) in (3, 4)  # RGB or RGBA

        # Test getting colormap
        cmap = create_color_palette(n_colors, as_cmap=True)
        assert isinstance(cmap, mpl.colors.Colormap)

    def test_create_color_palette_single_color(self):
        """Test creating single color palette."""
        colors = create_color_palette(1, palette_name="viridis")
        assert len(colors) == 1

    def test_enable_latex_rendering(self):
        """Test latex rendering toggling."""
        # Test disable
        result_disable = enable_latex_rendering(enable=False)
        assert result_disable is False
        assert plt.rcParams["text.usetex"] is False

        # Test enable (might fail if latex not installed, but should return bool)
        result_enable = enable_latex_rendering(enable=True)
        assert isinstance(result_enable, bool)

    def test_format_metric_label(self):
        """Test metric label formatting."""
        # Test without latex
        assert format_metric_label("mse", use_latex=False) == "mse"

        # Test exact match
        assert format_metric_label("mse", use_latex=True) == r"$\mathrm{MSE}$"

        # Test partial match (test_rmse_score contains 'mse' as well, which comes first in the dictionary if order matters, let's use something unique like 'my_mape_score')
        assert format_metric_label("my_mape_score", use_latex=True) == r"$\mathrm{MAPE}$"

        # Test no match
        assert format_metric_label("unknown_metric", use_latex=True) == "unknown_metric"
