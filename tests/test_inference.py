"""
Unit tests for lppls.inference — sliding-window calibration and tc PDF.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from lppls.dataset import _T_AXIS, _TC_MIN, _TC_MAX, SERIES_LEN
from lppls.formula import lppls_reformulated
from lppls.inference import (
    _tc_norm_to_calendar,
    calibrate,
    calibrate_multi,
    compute_tc_pdf,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_synthetic_prices(n: int = 300, seed: int = 42) -> pd.Series:
    """Realistic LPPLS bubble price series with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    tc    = 1.05
    m     = 0.5
    omega = 9.0
    A, B, C1, C2 = 1.0, -0.5, 0.05, 0.05

    t = np.linspace(0.0, 1.0, n)
    log_prices = lppls_reformulated(t, tc, m, omega, A, B, C1, C2)
    log_prices += rng.normal(0, 0.01, size=n)

    # Shift so all values are positive before exp
    log_prices = log_prices - log_prices.min() + 0.5
    prices = np.exp(log_prices)

    dates = pd.date_range("2000-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates, name="price")


def make_constant_model_fn(
    tc_norm: float = 1.05,
    m: float = 0.5,
    omega: float = 9.0,
):
    """Model function that always returns the same valid parameters."""
    pred = np.array([tc_norm, m, omega], dtype=float)

    def fn(x: np.ndarray) -> np.ndarray:
        return pred.copy()

    return fn


# ---------------------------------------------------------------------------
# _tc_norm_to_calendar
# ---------------------------------------------------------------------------

class TestTcNormToCalendar:
    def test_tc_norm_one_returns_window_end(self):
        t1 = pd.Timestamp("2020-01-01")
        t2 = pd.Timestamp("2021-01-01")
        result = _tc_norm_to_calendar(1.0, t1, t2)
        assert result == t2

    def test_tc_norm_above_one_lies_in_future(self):
        t1 = pd.Timestamp("2020-01-01")
        t2 = pd.Timestamp("2020-07-01")
        tc_norm = 1.1
        result = _tc_norm_to_calendar(tc_norm, t1, t2)
        assert result > t2

    def test_tc_norm_offset_proportional_to_span(self):
        t1 = pd.Timestamp("2020-01-01")
        t2 = pd.Timestamp("2021-01-01")
        span = t2 - t1
        tc_norm = 1.0 + 0.5
        result = _tc_norm_to_calendar(tc_norm, t1, t2)
        expected = t2 + 0.5 * span
        assert result == expected


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

class TestCalibrate:
    def test_returns_dataframe_correct_columns(self):
        prices = make_synthetic_prices()
        fn = make_constant_model_fn()
        df = calibrate(prices, fn, step=10)
        expected_cols = {
            "window_start", "window_end", "tc_norm", "m", "omega",
            "tc_calendar", "A", "B", "C1", "C2",
            "series_norm", "price_min", "price_max", "log_prices",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_window_count_step1(self):
        n = 300
        prices = make_synthetic_prices(n=n)
        fn = make_constant_model_fn()
        df = calibrate(prices, fn, step=1)
        expected = n - SERIES_LEN + 1
        assert len(df) == expected

    def test_window_count_step5(self):
        n = 300
        prices = make_synthetic_prices(n=n)
        fn = make_constant_model_fn()
        df = calibrate(prices, fn, step=5)
        expected = math.ceil((n - SERIES_LEN + 1) / 5)
        # Allow ±1 for off-by-one in range() vs ceil
        assert abs(len(df) - expected) <= 1

    def test_tc_calendar_at_or_after_window_end(self):
        prices = make_synthetic_prices()
        fn = make_constant_model_fn(tc_norm=1.05)
        df = calibrate(prices, fn, step=10)
        assert not df.empty
        assert (df["tc_calendar"] >= df["window_end"]).all()

    def test_nan_in_window_dropped(self):
        prices = make_synthetic_prices(n=300)
        prices.iloc[100] = np.nan
        fn = make_constant_model_fn()
        df_clean = calibrate(make_synthetic_prices(n=300), fn, step=1)
        df_nan   = calibrate(prices, fn, step=1)
        assert len(df_nan) < len(df_clean)

    def test_degenerate_window_dropped(self):
        """A constant window has zero variance — should be silently skipped."""
        prices = make_synthetic_prices(n=300)
        # Make window 0..SERIES_LEN-1 constant
        prices.iloc[:SERIES_LEN] = 1.0
        fn = make_constant_model_fn()
        df = calibrate(prices, fn, step=1)
        # At minimum window 0 should be absent
        assert len(df) < 300 - SERIES_LEN + 1

    def test_model_exception_skipped(self):
        """Model that raises on even-indexed call; odd calls succeed."""
        prices = make_synthetic_prices(n=300)
        pred = np.array([1.05, 0.5, 9.0], dtype=float)
        call_count = {"n": 0}

        def fn(x):
            call_count["n"] += 1
            if call_count["n"] % 2 == 0:
                raise RuntimeError("deliberate failure")
            return pred.copy()

        df = calibrate(prices, fn, step=1)
        # Roughly half the windows should be present
        total_windows = 300 - SERIES_LEN + 1
        assert 0 < len(df) < total_windows

    def test_short_series_raises(self):
        prices = make_synthetic_prices(n=100)
        fn = make_constant_model_fn()
        with pytest.raises(ValueError, match="shorter than"):
            calibrate(prices, fn)

    def test_non_positive_prices_raises_with_log(self):
        prices = make_synthetic_prices(n=300)
        prices.iloc[0] = -1.0
        fn = make_constant_model_fn()
        with pytest.raises(ValueError, match="non-positive"):
            calibrate(prices, fn, log_prices=True)

    def test_linear_params_finite(self):
        prices = make_synthetic_prices()
        fn = make_constant_model_fn()
        df = calibrate(prices, fn, step=10)
        assert not df.empty
        for col in ("A", "B", "C1", "C2"):
            assert df[col].apply(np.isfinite).all(), f"{col} has non-finite values"

    def test_series_norm_unit_range(self):
        prices = make_synthetic_prices()
        fn = make_constant_model_fn()
        df = calibrate(prices, fn, step=10)
        assert not df.empty
        for arr in df["series_norm"]:
            assert arr.min() >= 0.0 - 1e-6
            assert arr.max() <= 1.0 + 1e-6

    def test_log_prices_false_accepted(self):
        """log_prices=False should work without ValueError."""
        prices = make_synthetic_prices(n=300)
        fn = make_constant_model_fn()
        df = calibrate(prices, fn, step=10, log_prices=False)
        assert isinstance(df, pd.DataFrame)

    def test_empty_dataframe_correct_schema_when_all_filtered(self):
        """Model returns OOB params → all windows dropped → empty df with schema."""
        prices = make_synthetic_prices(n=300)
        bad_fn = lambda x: np.array([0.0, 0.0, 0.0])  # all OOB
        df = calibrate(prices, bad_fn, step=1)
        assert df.empty
        assert "tc_calendar" in df.columns


# ---------------------------------------------------------------------------
# compute_tc_pdf
# ---------------------------------------------------------------------------

class TestComputeTcPdf:
    def _make_df(self, n: int = 50) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
        prices = make_synthetic_prices(n=300)
        fn = make_constant_model_fn()
        df = calibrate(prices, fn, step=1)
        dates = pd.date_range(
            df["tc_calendar"].min(), df["tc_calendar"].max(), periods=200
        )
        return df, dates

    def test_shape_matches_dates(self):
        df, dates = self._make_df()
        density = compute_tc_pdf(df, dates)
        assert density.shape == (len(dates),)

    def test_nonnegative(self):
        df, dates = self._make_df()
        density = compute_tc_pdf(df, dates)
        assert (density >= 0).all()

    def test_empty_df_returns_zeros(self):
        from lppls.inference import _COLUMNS
        df = pd.DataFrame(columns=_COLUMNS)
        dates = pd.date_range("2020-01-01", periods=50)
        density = compute_tc_pdf(df, dates)
        assert (density == 0).all()

    def test_single_row_returns_zeros(self):
        """KDE requires n >= 2."""
        prices = make_synthetic_prices(n=300)
        fn = make_constant_model_fn()
        df = calibrate(prices, fn, step=1).iloc[:1]
        dates = pd.date_range("2000-01-01", periods=50)
        density = compute_tc_pdf(df, dates)
        assert (density == 0).all()

    def test_integrates_approximately_to_one(self):
        """KDE density × date_width should ≈ 1."""
        prices = make_synthetic_prices(n=300)
        fn = make_constant_model_fn()
        df = calibrate(prices, fn, step=1)
        dates = pd.date_range(
            df["tc_calendar"].min() - pd.Timedelta(days=60),
            df["tc_calendar"].max() + pd.Timedelta(days=60),
            periods=500,
        )
        density = compute_tc_pdf(df, dates)
        # Convert date axis to fractional years for integration width
        ordinals = np.array([d.toordinal() for d in dates], dtype=float)
        integral = np.trapz(density, ordinals)
        assert 0.8 <= integral <= 1.2, f"Integral = {integral:.3f}, expected ≈ 1.0"

    def test_custom_bandwidth_accepted(self):
        df, dates = self._make_df()
        density = compute_tc_pdf(df, dates, bandwidth=0.5)
        assert density.shape == (len(dates),)
        assert (density >= 0).all()


# ---------------------------------------------------------------------------
# calibrate_multi
# ---------------------------------------------------------------------------

class TestCalibrateMulti:
    def test_returns_all_keys(self):
        prices = make_synthetic_prices(n=300)
        models = {
            "M-LNN": make_constant_model_fn(tc_norm=1.05),
            "P-LNN": make_constant_model_fn(tc_norm=1.10),
        }
        results = calibrate_multi(prices, models, step=10)
        assert set(results.keys()) == {"M-LNN", "P-LNN"}

    def test_empty_model_preserved(self):
        """Model returning OOB params yields empty DataFrame under its key."""
        prices = make_synthetic_prices(n=300)
        models = {
            "good":  make_constant_model_fn(tc_norm=1.05),
            "bad":   lambda x: np.array([0.0, 0.0, 0.0]),
        }
        results = calibrate_multi(prices, models, step=10)
        assert "bad" in results
        assert results["bad"].empty
        assert not results["good"].empty

    def test_each_value_is_dataframe(self):
        prices = make_synthetic_prices(n=300)
        models = {"A": make_constant_model_fn()}
        results = calibrate_multi(prices, models, step=10)
        assert isinstance(results["A"], pd.DataFrame)
