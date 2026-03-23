"""
Sliding-window LPPLS calibration and tc probability density estimation.

For a given price series, calibrate() runs a trained model over every
252-point window, converts the predicted tc_norm to a calendar date, and
recovers the linear LPPLS parameters (A, B, C1, C2) for optional fit-curve
plotting.  compute_tc_pdf() then estimates the probability density of tc
across windows via KDE.

Typical P-LNN usage
-------------------
    import keras, pandas as pd
    from lppls.inference import calibrate, compute_tc_pdf

    model = keras.models.load_model("plnn.keras", compile=False)
    model_fn = lambda x: model(x[None], training=False)[0].numpy()

    prices = pd.read_csv("nasdaq.csv", index_col=0, parse_dates=True)["Close"]
    df = calibrate(prices, model_fn, step=5)
    # df columns: window_start, window_end, tc_norm, m, omega,
    #             tc_calendar, A, B, C1, C2,
    #             series_norm, price_min, price_max, log_prices
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from .dataset import SERIES_LEN, _TC_MIN, _TC_MAX, _T_AXIS
from .formula import recover_linear_params

logger = logging.getLogger(__name__)

# Nonlinear parameter validity bounds used during window filtering.
# tc bounds come from dataset.py (P-LNN training distribution).
# m and omega from spec §3 / mlnn.py.
_M_MIN, _M_MAX = 0.1, 0.9
_OMEGA_MIN, _OMEGA_MAX = 6.0, 13.0

# Small offset to avoid tc == t2 exactly (which makes dt=0 at _T_AXIS[-1],
# producing NaN in lppls_reformulated and recover_linear_params).
_TC_EPS = 1e-6

# Ordered column list used to produce an empty DataFrame with correct schema.
_COLUMNS = [
    "window_start", "window_end",
    "tc_norm", "m", "omega",
    "tc_calendar",
    "A", "B", "C1", "C2",
    "series_norm",
    "price_min", "price_max",
    "log_prices",
]


@dataclass
class WindowResult:
    """Result of calibrating a single 252-point window."""
    window_start: pd.Timestamp
    window_end:   pd.Timestamp
    tc_norm:      float          # model output in (_TC_MIN, _TC_MAX]
    m:            float
    omega:        float
    tc_calendar:  pd.Timestamp   # critical time in calendar space
    A:            float          # recovered linear LPPLS parameters
    B:            float
    C1:           float
    C2:           float
    series_norm:  np.ndarray     # (252,) float32 min-max-normalised window
    price_min:    float          # raw (pre-log) window minimum
    price_max:    float          # raw (pre-log) window maximum
    log_prices:   bool           # whether log() was applied before normalise


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _minmax_normalize(arr: np.ndarray) -> np.ndarray | None:
    """Scale arr to [0, 1]. Returns None if max-min < 1e-12 (degenerate)."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return None
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _tc_norm_to_calendar(
    tc_norm: float,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.Timestamp:
    """Map tc_norm (in normalised time) to a calendar date.

    The normalised axis maps window_start → 0 and window_end → 1.
    tc_norm = 1.0 corresponds to window_end; values above 1 lie in the future.

        tc_calendar = window_end + (tc_norm - 1.0) * (window_end - window_start)
    """
    window_span = window_end - window_start  # pd.Timedelta
    return window_end + (tc_norm - 1.0) * window_span


def _is_valid_result(tc_norm: float, m: float, omega: float) -> bool:
    """Return True iff all parameters are finite and within spec bounds.

    tc_norm uses P-LNN training bounds from dataset.py (_TC_MIN, _TC_MAX).
    A small epsilon guards against tc == t2 which causes dt=0 in formula.py.
    """
    return (
        np.isfinite(tc_norm)
        and (_TC_MIN + _TC_EPS) <= tc_norm <= _TC_MAX
        and np.isfinite(m) and _M_MIN <= m <= _M_MAX
        and np.isfinite(omega) and _OMEGA_MIN <= omega <= _OMEGA_MAX
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calibrate(
    prices: pd.Series,
    model_fn: Callable[[np.ndarray], np.ndarray],
    step: int = 1,
    window_len: int = SERIES_LEN,
    log_prices: bool = True,
) -> pd.DataFrame:
    """Sliding-window LPPLS calibration over a real price time series.

    Parameters
    ----------
    prices : pd.Series
        Price series with a DatetimeIndex. Values must be positive raw prices.
    model_fn : Callable[[np.ndarray], np.ndarray]
        Maps a min-max-normalised ``(window_len,)`` float32 array to a
        ``(3,)`` array ``[tc_norm, m, omega]`` in normalised units.
        Compatible with any model (P-LNN, M-LNN, LM wrapper, etc.).
    step : int
        Observations to advance between windows. ``step=1`` gives daily
        resolution; ``step=5`` gives approximately weekly.
    window_len : int
        Fixed window length in observations. Must match the model input size.
    log_prices : bool
        If True (default), apply ``np.log`` before normalisation. Recommended
        for LPPLS since the model expects log-price dynamics.

    Returns
    -------
    pd.DataFrame
        One row per successfully calibrated window with columns:
        ``window_start``, ``window_end``, ``tc_norm``, ``m``, ``omega``,
        ``tc_calendar``, ``A``, ``B``, ``C1``, ``C2``,
        ``series_norm``, ``price_min``, ``price_max``, ``log_prices``.
        Windows with NaN prices, degenerate range, model exceptions, or
        out-of-bound predictions are silently dropped.

    Raises
    ------
    ValueError
        If ``len(prices) < window_len`` or prices contain non-positive values
        when ``log_prices=True``.
    """
    if len(prices) < window_len:
        raise ValueError(
            f"Series length {len(prices)} is shorter than window_len {window_len}."
        )

    vals = prices.values.copy().astype(float)
    dates_idx = prices.index

    if log_prices:
        if np.any(vals[np.isfinite(vals)] <= 0):
            raise ValueError(
                "prices contain non-positive values; cannot apply log(). "
                "Pass log_prices=False or clean the series first."
            )
        vals = np.where(np.isfinite(vals), np.log(vals), np.nan)

    results: list[WindowResult] = []
    n = len(vals)

    for i in range(0, n - window_len + 1, step):
        raw_window = vals[i : i + window_len]

        if not np.all(np.isfinite(raw_window)):
            continue

        price_min = float(raw_window.min())
        price_max = float(raw_window.max())

        series_norm = _minmax_normalize(raw_window)
        if series_norm is None:
            continue

        try:
            pred = np.asarray(model_fn(series_norm), dtype=float).ravel()
        except Exception as exc:
            logger.debug("model_fn raised on window %d: %s", i, exc)
            continue

        if pred.shape != (3,) or not np.all(np.isfinite(pred)):
            continue

        tc_norm = float(pred[0])
        m       = float(pred[1])
        omega   = float(pred[2])

        if not _is_valid_result(tc_norm, m, omega):
            continue

        try:
            A, B, C1, C2 = recover_linear_params(
                _T_AXIS, tc_norm, m, omega, series_norm
            )
        except Exception as exc:
            logger.debug("recover_linear_params failed on window %d: %s", i, exc)
            continue

        if not all(np.isfinite(v) for v in (A, B, C1, C2)):
            continue

        window_start = dates_idx[i]
        window_end   = dates_idx[i + window_len - 1]
        tc_calendar  = _tc_norm_to_calendar(tc_norm, window_start, window_end)

        results.append(WindowResult(
            window_start=window_start,
            window_end=window_end,
            tc_norm=tc_norm,
            m=m,
            omega=omega,
            tc_calendar=tc_calendar,
            A=float(A), B=float(B), C1=float(C1), C2=float(C2),
            series_norm=series_norm,
            price_min=price_min,
            price_max=price_max,
            log_prices=log_prices,
        ))

    if not results:
        return pd.DataFrame(columns=_COLUMNS)

    return pd.DataFrame([
        {
            "window_start": r.window_start,
            "window_end":   r.window_end,
            "tc_norm":      r.tc_norm,
            "m":            r.m,
            "omega":        r.omega,
            "tc_calendar":  r.tc_calendar,
            "A":            r.A,
            "B":            r.B,
            "C1":           r.C1,
            "C2":           r.C2,
            "series_norm":  r.series_norm,
            "price_min":    r.price_min,
            "price_max":    r.price_max,
            "log_prices":   r.log_prices,
        }
        for r in results
    ])


def calibrate_multi(
    prices: pd.Series,
    models: dict[str, Callable[[np.ndarray], np.ndarray]],
    step: int = 1,
    window_len: int = SERIES_LEN,
    log_prices: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run :func:`calibrate` for each named model.

    Parameters
    ----------
    prices : pd.Series
    models : dict[str, Callable]
        e.g. ``{"M-LNN": mlnn_fn, "P-LNN": plnn_fn}``
    step, window_len, log_prices : forwarded to :func:`calibrate`.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys match ``models``. Empty DataFrames are always included.
    """
    return {
        name: calibrate(
            prices, fn,
            step=step, window_len=window_len, log_prices=log_prices,
        )
        for name, fn in models.items()
    }


def compute_tc_pdf(
    df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    bandwidth: float | str = "scott",
) -> np.ndarray:
    """Evaluate a KDE of tc_calendar values on a date grid.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`calibrate`; must contain column ``tc_calendar``.
    dates : pd.DatetimeIndex
        Grid of dates at which to evaluate the density.
    bandwidth : float or str
        KDE bandwidth — ``"scott"`` (default), ``"silverman"``, or a float
        passed directly to ``scipy.stats.gaussian_kde``.

    Returns
    -------
    np.ndarray, shape (len(dates),)
        Density values. Returns an all-zeros array when ``df`` has fewer
        than 2 rows (KDE is undefined for n < 2).
    """
    zeros = np.zeros(len(dates))
    if df.empty or len(df) < 2:
        return zeros

    # Use ordinal (integer days since year 1) for numerical stability.
    tc_ordinals  = np.array([ts.toordinal() for ts in df["tc_calendar"]], dtype=float)
    grid_ordinals = np.array([d.toordinal() for d in dates], dtype=float)

    try:
        kde = gaussian_kde(tc_ordinals, bw_method=bandwidth)
        return kde(grid_ordinals)
    except Exception as exc:
        logger.debug("gaussian_kde failed: %s", exc)
        return zeros
