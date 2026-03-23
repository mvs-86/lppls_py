"""
Unit tests for lppls.plotting — bubble analysis visualization.

Uses the Agg (non-interactive) backend so tests run without a display.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # must be set before any other matplotlib import

import os
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
import matplotlib.collections

from lppls.formula import lppls_reformulated
from lppls.inference import calibrate
from lppls.plotting import plot_calibration, save_figure


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_prices(n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    log_p = lppls_reformulated(t, 1.05, 0.5, 9.0, 1.0, -0.5, 0.05, 0.05)
    log_p += rng.normal(0, 0.01, n)
    log_p = log_p - log_p.min() + 0.5
    prices = np.exp(log_p)
    dates = pd.date_range("2000-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates)


def _make_constant_fn(tc_norm=1.05, m=0.5, omega=9.0):
    pred = np.array([tc_norm, m, omega], dtype=float)
    return lambda x: pred.copy()


@pytest.fixture
def prices():
    return _make_prices()


@pytest.fixture
def calibration_results(prices):
    fn = _make_constant_fn()
    df = calibrate(prices, fn, step=5)
    return {"P-LNN": df}


@pytest.fixture
def multi_results(prices):
    return {
        "LM":    calibrate(prices, _make_constant_fn(tc_norm=1.04), step=5),
        "M-LNN": calibrate(prices, _make_constant_fn(tc_norm=1.05), step=5),
        "P-LNN": calibrate(prices, _make_constant_fn(tc_norm=1.06), step=5),
    }


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

class TestFigureStructure:
    def test_returns_figure(self, prices, calibration_results):
        fig = plot_calibration(prices, calibration_results)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_two_axes_created(self, prices, calibration_results):
        fig = plot_calibration(prices, calibration_results)
        axes = fig.get_axes()
        assert len(axes) == 2, f"Expected 2 axes, got {len(axes)}"
        plt.close("all")

    def test_price_line_drawn(self, prices, calibration_results):
        fig = plot_calibration(prices, calibration_results)
        ax_price = fig.get_axes()[0]
        # At least the price line should be present
        assert len(ax_price.get_lines()) >= 1
        plt.close("all")

    def test_pdf_fill_created(self, prices, calibration_results):
        fig = plot_calibration(prices, calibration_results)
        ax_pdf = fig.get_axes()[1]
        polys = [c for c in ax_pdf.get_children()
                 if isinstance(c, matplotlib.collections.PolyCollection)]
        assert len(polys) >= 1, "No filled PDF area found on right axis"
        plt.close("all")

    def test_title_set_when_provided(self, prices, calibration_results):
        fig = plot_calibration(prices, calibration_results, title="Test Title")
        ax = fig.get_axes()[0]
        assert "Test Title" in ax.get_title()
        plt.close("all")

    def test_no_title_when_none(self, prices, calibration_results):
        fig = plot_calibration(prices, calibration_results, title=None)
        ax = fig.get_axes()[0]
        assert ax.get_title() == ""
        plt.close("all")


# ---------------------------------------------------------------------------
# Vertical lines and shading
# ---------------------------------------------------------------------------

class TestDecoration:
    def test_vertical_lines_with_tc_actual_and_trough(self, prices, calibration_results):
        tc_actual = pd.Timestamp("2001-06-01")
        tc_trough = pd.Timestamp("2001-09-01")
        fig = plot_calibration(
            prices, calibration_results,
            tc_actual=tc_actual, tc_trough=tc_trough,
        )
        ax_price = fig.get_axes()[0]
        vlines = [l for l in ax_price.get_lines() if len(l.get_xdata()) == 2
                  and l.get_xdata()[0] == l.get_xdata()[1]]
        # Should have lines for first_t1, last_t2, tc_actual, tc_trough
        assert len(vlines) >= 4
        plt.close("all")

    def test_no_vertical_lines_without_optional_params(self, prices, calibration_results):
        fig = plot_calibration(prices, calibration_results)
        ax_price = fig.get_axes()[0]
        vlines = [l for l in ax_price.get_lines() if len(l.get_xdata()) == 2
                  and l.get_xdata()[0] == l.get_xdata()[1]]
        # first_t1 and last_t2 lines from calibration window
        assert len(vlines) == 2
        plt.close("all")


# ---------------------------------------------------------------------------
# show_fit_curves modes
# ---------------------------------------------------------------------------

class TestShowFitCurves:
    def test_median_mode_adds_fit_line(self, prices, calibration_results):
        fig = plot_calibration(prices, calibration_results, show_fit_curves="median")
        ax_price = fig.get_axes()[0]
        # Price line + at least one fit line
        assert len(ax_price.get_lines()) >= 2
        plt.close("all")

    def test_none_mode_only_price_line(self, prices, calibration_results):
        fig = plot_calibration(prices, calibration_results, show_fit_curves="none")
        ax_price = fig.get_axes()[0]
        non_vline = [l for l in ax_price.get_lines()
                     if not (len(l.get_xdata()) == 2 and
                             l.get_xdata()[0] == l.get_xdata()[1])]
        assert len(non_vline) == 1, "Expected only price line with show_fit_curves='none'"
        plt.close("all")

    def test_all_mode_adds_multiple_curves(self, prices, calibration_results):
        fig = plot_calibration(prices, calibration_results, show_fit_curves="all")
        ax_price = fig.get_axes()[0]
        assert len(ax_price.get_lines()) > 2
        plt.close("all")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_results_no_crash(self, prices):
        """Empty DataFrames for all models should not raise."""
        from lppls.inference import _COLUMNS
        empty = pd.DataFrame(columns=_COLUMNS)
        fig = plot_calibration(prices, {"P-LNN": empty})
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_unknown_model_name_uses_default_color(self, prices, calibration_results):
        """Model name not in _MODEL_COLORS should use fallback color without error."""
        new_results = {"MyCustomModel": list(calibration_results.values())[0]}
        fig = plot_calibration(prices, new_results)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_multiple_models(self, prices, multi_results):
        fig = plot_calibration(prices, multi_results)
        ax_pdf = fig.get_axes()[1]
        polys = [c for c in ax_pdf.get_children()
                 if isinstance(c, matplotlib.collections.PolyCollection)]
        assert len(polys) == 3
        plt.close("all")

    def test_inject_existing_axes(self, prices, calibration_results):
        fig_outer, ax = plt.subplots()
        ax_twin = ax.twinx()
        returned_fig = plot_calibration(
            prices, calibration_results, ax_price=ax, ax_pdf=ax_twin
        )
        assert returned_fig is fig_outer
        plt.close("all")

    def test_custom_figsize(self, prices, calibration_results):
        fig = plot_calibration(prices, calibration_results, figsize=(10.0, 4.0))
        w, h = fig.get_size_inches()
        assert abs(w - 10.0) < 0.1
        assert abs(h - 4.0) < 0.1
        plt.close("all")


# ---------------------------------------------------------------------------
# save_figure
# ---------------------------------------------------------------------------

class TestSaveFigure:
    def test_creates_file(self, prices, calibration_results, tmp_path):
        fig = plot_calibration(prices, calibration_results)
        out = tmp_path / "subdir" / "test_fig.png"
        save_figure(fig, out)
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close("all")

    def test_creates_parent_directories(self, prices, calibration_results, tmp_path):
        fig = plot_calibration(prices, calibration_results)
        out = tmp_path / "a" / "b" / "c" / "fig.png"
        save_figure(fig, out)
        assert out.exists()
        plt.close("all")

    def test_pdf_format(self, prices, calibration_results, tmp_path):
        fig = plot_calibration(prices, calibration_results)
        out = tmp_path / "fig.pdf"
        save_figure(fig, out)
        assert out.exists()
        plt.close("all")
