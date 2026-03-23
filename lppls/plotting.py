"""
Bubble analysis visualization — reproduces Figures 4–6 of arXiv:2405.12803v1.

Dual-axis layout
----------------
Left Y-axis  : price time series (dark line) + LPPLS fit curve(s) per model
Right Y-axis : KDE-based PDF of tc_calendar per model (filled area)
X-axis       : calendar dates

Decorations
-----------
- Grey shading: full calibration window range [first_t1, last_t2]
- Red shading : drawdown period [tc_actual, tc_trough]  (optional)
- Vertical lines:
    green  dotted-dashed  — first_t1
    red    dotted-dashed  — last_t2
    black  dotted-dashed  — tc_actual   (if provided)
    black  dashed         — tc_trough   (if provided)

Color scheme (matches paper)
-----------------------------
LM    → blue   #1f77b4
M-LNN → orange #ff7f0e
P-LNN → purple #9467bd
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dataset import _T_AXIS
from .formula import lppls_reformulated
from .inference import compute_tc_pdf

logger = logging.getLogger(__name__)

# ── Color scheme (paper Figs 4–6) ──────────────────────────────────────────
_MODEL_COLORS: dict[str, str] = {
    "LM":    "#1f77b4",   # blue
    "M-LNN": "#ff7f0e",   # orange
    "P-LNN": "#9467bd",   # purple
}
_DEFAULT_COLOR = "#2ca02c"   # green fallback for unlisted model names

# Maximum individual fit curves in "all" mode before subsampling
_MAX_ALL_CURVES = 200


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_model_color(name: str) -> str:
    color = _MODEL_COLORS.get(name)
    if color is None:
        logger.debug("Unknown model name %r — using default color.", name)
    return color or _DEFAULT_COLOR


def _compute_calibration_window_bounds(
    calibration_results: dict[str, pd.DataFrame],
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return (first_t1, last_t2) across all non-empty DataFrames."""
    starts, ends = [], []
    for df in calibration_results.values():
        if not df.empty:
            starts.append(df["window_start"].min())
            ends.append(df["window_end"].max())
    if not starts:
        return None, None
    return min(starts), max(ends)


def _build_date_grid(
    prices: pd.Series,
    calibration_results: dict[str, pd.DataFrame],
) -> pd.DatetimeIndex:
    """Date grid spanning prices plus any tc_calendar dates that lie beyond."""
    all_dates: list[pd.Timestamp] = list(prices.index)
    for df in calibration_results.values():
        if not df.empty:
            all_dates.extend(df["tc_calendar"].tolist())
    start = min(all_dates)
    end   = max(all_dates)
    # 500-point grid gives smooth KDE curves without excessive memory use
    return pd.date_range(start, end, periods=500)


def _select_representative_window(df: pd.DataFrame) -> pd.Series | None:
    """Row whose tc_calendar is nearest to the median tc_calendar."""
    if df.empty:
        return None
    median_tc = df["tc_calendar"].median()
    idx = (df["tc_calendar"] - median_tc).abs().idxmin()
    return df.loc[idx]


def _reconstruct_fit_curve(
    row: pd.Series,
    prices: pd.Series,
) -> tuple[pd.DatetimeIndex, np.ndarray] | None:
    """Reconstruct a LPPLS fit curve in original price scale.

    Steps
    -----
    1. Evaluate ``lppls_reformulated`` on the normalised time axis → fit in [0,1].
    2. Invert min-max: ``fit_raw = fit_norm * (price_max - price_min) + price_min``.
    3. If ``log_prices`` was True, ``fit_prices = exp(fit_raw)``.

    Returns
    -------
    (window_dates, fit_prices) or None if reconstruction fails.
    """
    try:
        tc_norm    = float(row["tc_norm"])
        m          = float(row["m"])
        omega      = float(row["omega"])
        A          = float(row["A"])
        B          = float(row["B"])
        C1         = float(row["C1"])
        C2         = float(row["C2"])
        price_min  = float(row["price_min"])
        price_max  = float(row["price_max"])
        log_prices = bool(row["log_prices"])

        fit_norm = lppls_reformulated(_T_AXIS, tc_norm, m, omega, A, B, C1, C2)
        if not np.all(np.isfinite(fit_norm)):
            return None

        fit_raw    = fit_norm * (price_max - price_min) + price_min
        fit_prices = np.exp(fit_raw) if log_prices else fit_raw

        window_start = pd.Timestamp(row["window_start"])
        window_end   = pd.Timestamp(row["window_end"])

        # Slice window dates from the original price index
        mask = (prices.index >= window_start) & (prices.index <= window_end)
        window_dates = prices.index[mask]
        if len(window_dates) != len(fit_prices):
            # Fallback: uniformly spaced grid within the window
            window_dates = pd.date_range(
                window_start, window_end, periods=len(fit_prices)
            )

        return window_dates, fit_prices

    except Exception as exc:
        logger.debug("_reconstruct_fit_curve failed: %s", exc)
        return None


def _draw_fit_curves(
    ax: plt.Axes,
    df: pd.DataFrame,
    color: str,
    show_fit_curves: str,
    prices: pd.Series,
) -> None:
    """Plot LPPLS fit curve(s) on *ax*."""
    if df.empty or show_fit_curves == "none":
        return

    if show_fit_curves == "median":
        row = _select_representative_window(df)
        if row is None:
            return
        result = _reconstruct_fit_curve(row, prices)
        if result is not None:
            window_dates, fit_prices = result
            ax.plot(window_dates, fit_prices, color=color,
                    linewidth=1.2, alpha=0.9)

    elif show_fit_curves == "all":
        rows_df = df if len(df) <= _MAX_ALL_CURVES else df.sample(
            n=_MAX_ALL_CURVES, random_state=0
        )
        for _, row in rows_df.iterrows():
            result = _reconstruct_fit_curve(row, prices)
            if result is not None:
                window_dates, fit_prices = result
                ax.plot(window_dates, fit_prices, color=color,
                        linewidth=0.5, alpha=0.15)


def _draw_pdf(
    ax_pdf: plt.Axes,
    dates: pd.DatetimeIndex,
    density: np.ndarray,
    color: str,
    label: str,
) -> None:
    """Draw a filled KDE density curve on the right axis."""
    ax_pdf.fill_between(dates, density, alpha=0.35, color=color,
                        label=f"PDF of $t_c$: {label}")
    ax_pdf.plot(dates, density, color=color, linewidth=1.0)


def _draw_shading_and_lines(
    ax: plt.Axes,
    first_t1: pd.Timestamp | None,
    last_t2:  pd.Timestamp | None,
    tc_actual: pd.Timestamp | None,
    tc_trough: pd.Timestamp | None,
) -> None:
    """Draw calibration window shading, drawdown shading, and vertical lines."""
    if first_t1 is not None and last_t2 is not None:
        ax.axvspan(first_t1, last_t2, alpha=0.08, color="grey",
                   label="Calibration Window")
        ax.axvline(first_t1, color="green", linestyle="-.",
                   linewidth=1.0, label="First $t_1$")
        ax.axvline(last_t2,  color="red",   linestyle="-.",
                   linewidth=1.0, label="Last $t_2$")

    if tc_actual is not None and tc_trough is not None:
        ax.axvspan(tc_actual, tc_trough, alpha=0.12, color="red",
                   label="Drawdown Period")

    if tc_actual is not None:
        ax.axvline(tc_actual, color="black", linestyle="-.",
                   linewidth=1.0, label="$t_c$: Actual")
    if tc_trough is not None:
        ax.axvline(tc_trough, color="black", linestyle="--",
                   linewidth=1.0, label="Post $t_c$ Trough")


def _format_axes(
    ax_price: plt.Axes,
    ax_pdf:   plt.Axes,
    title:    str | None,
) -> None:
    """Apply labels, combined legend, date formatting, and optional title."""
    ax_price.set_xlabel("Date")
    ax_price.set_ylabel("Price")
    ax_pdf.set_ylabel("Density of $t_c$")
    ax_pdf.set_ylim(bottom=0)

    # Merge legend entries from both axes
    h1, l1 = ax_price.get_legend_handles_labels()
    h2, l2 = ax_pdf.get_legend_handles_labels()
    ax_price.legend(h1 + h2, l1 + l2,
                    loc="upper left", fontsize=8, framealpha=0.7)

    locator   = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax_price.xaxis.set_major_locator(locator)
    ax_price.xaxis.set_major_formatter(formatter)

    if title:
        ax_price.set_title(title)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_calibration(
    prices: pd.Series,
    calibration_results: dict[str, pd.DataFrame],
    *,
    tc_actual: pd.Timestamp | None = None,
    tc_trough: pd.Timestamp | None = None,
    show_fit_curves: str = "median",
    kde_bandwidth: float | str = "scott",
    figsize: tuple[float, float] = (14.0, 6.0),
    price_color: str = "#1a1a1a",
    title: str | None = None,
    ax_price: plt.Axes | None = None,
    ax_pdf:   plt.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Reproduce Figures 4–6 from arXiv:2405.12803v1.

    Parameters
    ----------
    prices : pd.Series
        Original raw price series with DatetimeIndex. Plotted as-is on the
        left axis. Pass ``np.log(prices)`` if you prefer log-scale display.
    calibration_results : dict[str, pd.DataFrame]
        Output of :func:`~lppls.inference.calibrate_multi` or a manually
        assembled ``{model_name: calibrate_df}`` dict.
    tc_actual : pd.Timestamp, optional
        Known critical time. Draws a black dotted-dashed vertical line and,
        if *tc_trough* is also given, starts a red-shaded drawdown region.
    tc_trough : pd.Timestamp, optional
        Post-peak price trough. Draws a black dashed vertical line and ends
        the red-shaded drawdown region.
    show_fit_curves : {"median", "all", "none"}
        ``"median"`` — one representative LPPLS fit per model (default).
        ``"all"``    — all windows plotted with low alpha (subsampled to 200).
        ``"none"``   — no fit curves.
    kde_bandwidth : float or str
        Bandwidth for ``scipy.stats.gaussian_kde`` (``"scott"`` by default).
    figsize : (width, height) in inches.  Default (14, 6).
    price_color : str
        Colour of the price line.
    title : str, optional
        Figure title.
    ax_price, ax_pdf : matplotlib.axes.Axes, optional
        Inject existing axes (e.g. to embed in a multi-panel figure).
        *ax_pdf* must be a twin of *ax_price*. If both are ``None`` a new
        figure is created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax_price is None:
        fig, ax_price = plt.subplots(figsize=figsize)
        ax_pdf = ax_price.twinx()
    else:
        if ax_pdf is None:
            ax_pdf = ax_price.twinx()
        fig = ax_price.get_figure()

    # ── Price line ────────────────────────────────────────────────────────
    ax_price.plot(prices.index, prices.values,
                  color=price_color, linewidth=1.0, label="Price")

    # ── Calibration window bounds ─────────────────────────────────────────
    first_t1, last_t2 = _compute_calibration_window_bounds(calibration_results)
    date_grid = _build_date_grid(prices, calibration_results)

    # ── Shading and vertical lines ────────────────────────────────────────
    _draw_shading_and_lines(ax_price, first_t1, last_t2, tc_actual, tc_trough)

    # ── Per-model fit curves and PDFs ─────────────────────────────────────
    for model_name, df in calibration_results.items():
        color = _get_model_color(model_name)
        _draw_fit_curves(ax_price, df, color, show_fit_curves, prices)
        density = compute_tc_pdf(df, date_grid, bandwidth=kde_bandwidth)
        if density.max() > 0:
            _draw_pdf(ax_pdf, date_grid, density, color, model_name)

    # ── Formatting ────────────────────────────────────────────────────────
    _format_axes(ax_price, ax_pdf, title)
    fig.tight_layout()
    return fig


def save_figure(
    fig: matplotlib.figure.Figure,
    path: str | Path,
    dpi: int = 150,
) -> None:
    """Save *fig* to *path*, creating parent directories as needed.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str or Path
        Output file path. Format inferred from extension (png, pdf, svg…).
    dpi : int
        Dots per inch for raster formats. Default 150.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info("Figure saved to %s", path)
