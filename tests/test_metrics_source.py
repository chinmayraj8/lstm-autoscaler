"""Unit tests for the metrics-ingestion interface (src/autoscaler/metrics_source.py).

No TensorFlow, no real dataset, no network -- MetricsSource is interface-
only at this stage (Step 20): no concrete real connector exists yet, so
these tests exercise the ABC contract via StaticMetricsSource (a synthetic
reference implementation, not a stand-in for a real system) and check
resample_readings against data._prepare_timeseries's resample step
directly, on the same synthetic data, to back up the "byte-identical
convention" claim in the module docstring.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.autoscaler import config
from src.autoscaler.data import _prepare_timeseries
from src.autoscaler.metrics_source import (
    MetricsSource,
    StaticMetricsSource,
    resample_readings,
    synthetic_readings_series,
)

T0 = datetime(2026, 1, 1)


# ── resample_readings vs. data._prepare_timeseries: same convention ────────

def test_resample_readings_matches_prepare_timeseries_on_the_same_data():
    series = synthetic_readings_series(T0, n_points=500, cadence_minutes=1, seed=3)

    # Path A: metrics_source.resample_readings on a raw-readings DataFrame.
    raw = series.to_frame(name=config.FEATURE_COL)
    via_metrics_source = resample_readings(raw)

    # Path B: data._prepare_timeseries on an equivalent raw CSV-shaped frame
    # (epoch-second time_stamp, machine_id, both feature columns -- the
    # shape _prepare_timeseries actually expects).
    csv_shaped = pd.DataFrame({
        "machine_id": "m_test",
        "time_stamp": (series.index.astype("int64") // 10**9),
        "cpu_util_percent": series.values,
        "mem_util_percent": series.values,  # unused by FEATURE_COL, value irrelevant here
    })
    via_prepare_timeseries = _prepare_timeseries(csv_shaped, "m_test")[[config.FEATURE_COL]]

    pd.testing.assert_frame_equal(via_metrics_source, via_prepare_timeseries,
                                  check_freq=False, check_names=False)


def test_resample_readings_on_empty_input_returns_empty():
    empty = pd.DataFrame({config.FEATURE_COL: []}, index=pd.DatetimeIndex([]))
    out = resample_readings(empty)
    assert out.empty


def test_resample_readings_ffills_gaps_same_as_prepare_timeseries():
    # A finer-than-5min series with a gap should ffill through the gap,
    # not introduce NaNs -- matches _prepare_timeseries's convention exactly.
    idx = pd.date_range(T0, periods=20, freq="1min")
    values = np.full(20, 50.0)
    raw = pd.Series(values, index=idx, name=config.FEATURE_COL).to_frame()
    # Drop a chunk in the middle to create a real gap between resampled bins.
    raw = raw.drop(raw.index[8:12])

    out = resample_readings(raw)
    assert not out[config.FEATURE_COL].isna().any()


# ── MetricsSource / StaticMetricsSource contract ────────────────────────────

def test_metrics_source_is_abstract():
    with pytest.raises(TypeError):
        MetricsSource()  # can't instantiate the ABC directly


def test_static_metrics_source_slices_to_requested_range():
    series = synthetic_readings_series(T0, n_points=100, cadence_minutes=1, seed=1)
    source = StaticMetricsSource({"m_test": series})

    start = T0 + timedelta(minutes=10)
    end = T0 + timedelta(minutes=20)
    result = source.fetch_readings("m_test", start, end)

    assert (result.index >= start).all()
    assert (result.index <= end).all()
    assert len(result) == 11  # inclusive, 1-min cadence


def test_static_metrics_source_unknown_machine_returns_empty():
    source = StaticMetricsSource({"m_test": synthetic_readings_series(T0, 10)})
    result = source.fetch_readings("m_never_seen", T0, T0 + timedelta(hours=1))
    assert result.empty


def test_static_metrics_source_empty_overlap_returns_empty():
    series = synthetic_readings_series(T0, n_points=10, cadence_minutes=1, seed=2)
    source = StaticMetricsSource({"m_test": series})
    # Request a range entirely after the series ends.
    result = source.fetch_readings("m_test", T0 + timedelta(days=1), T0 + timedelta(days=2))
    assert result.empty


# ── fetch_lookback_window: the convenience a future live loop would call ───

def test_fetch_lookback_window_returns_lookback_steps_rows():
    # 1-min cadence raw data, resampled to 5-min -- plenty of history before `now`.
    series = synthetic_readings_series(T0, n_points=200, cadence_minutes=1, seed=4)
    source = StaticMetricsSource({"m_test": series})
    now = T0 + timedelta(minutes=150)

    window = source.fetch_lookback_window("m_test", now, lookback_steps=config.LOOKBACK_STEPS)

    assert len(window) >= config.LOOKBACK_STEPS
    assert window.index.max() <= now
    assert not window[config.FEATURE_COL].isna().any()


def test_fetch_lookback_window_on_machine_with_no_data_is_empty():
    source = StaticMetricsSource({})
    window = source.fetch_lookback_window("m_never_seen", T0)
    assert window.empty
