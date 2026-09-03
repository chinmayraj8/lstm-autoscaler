"""Unit tests for the metrics-ingestion interface (src/autoscaler/metrics_source.py).

No TensorFlow, no real dataset, no network. Two groups of tests:

- The interface/StaticMetricsSource tests from Step 20, unchanged: exercise
  the ABC contract via StaticMetricsSource (a synthetic reference
  implementation, not a stand-in for a real system) and check
  resample_readings against data._prepare_timeseries's resample step
  directly, on the same synthetic data.
- PrometheusMetricsSource tests (Step 21 / Stage 3): a mocked
  `requests.Session` returning the fixture in
  tests/fixtures/prometheus_instant_query_response.json (see that file's
  own note on provenance) -- no live cluster dependency, matching this
  step's explicit instruction.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.autoscaler import config
from src.autoscaler.data import _prepare_timeseries
from src.autoscaler.metrics_source import (
    MetricsSource,
    PrometheusMetricsSource,
    PrometheusMetricsSourceError,
    StaticMetricsSource,
    build_cpu_util_query,
    instance_to_machine_id,
    resample_readings,
    synthetic_readings_series,
)

T0 = datetime(2026, 1, 1)
_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "prometheus_instant_query_response.json"


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


# ── PrometheusMetricsSource (Step 21 / Stage 3) ─────────────────────────────

def _load_fixture() -> dict:
    with open(_FIXTURE_PATH) as f:
        return json.load(f)


def _mock_session(payload: dict) -> MagicMock:
    """A `requests.Session`-shaped mock whose `.get(...)` always returns
    `payload` as JSON with a 200 status (`raise_for_status` a no-op)."""
    session = MagicMock()
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    session.get.return_value = response
    return session


def test_build_cpu_util_query_matches_the_verified_query():
    assert build_cpu_util_query() == (
        '100 * (1 - avg by (instance) '
        '(rate(node_cpu_seconds_total{mode="idle"}[5m])))'
    )


def test_build_cpu_util_query_parameterizes_range_vector_and_idle_label():
    q = build_cpu_util_query(range_vector="1m", idle_mode_label="idle_mode")
    assert "[1m]" in q
    assert 'mode="idle_mode"' in q
    assert "[5m]" not in q


def test_instance_to_machine_id_strips_port():
    assert instance_to_machine_id("10.244.1.5:9100") == "10.244.1.5"


def test_instance_to_machine_id_leaves_bare_label_unchanged():
    assert instance_to_machine_id("node-a") == "node-a"


def test_fetch_readings_queries_instant_endpoint_and_filters_to_machine_id():
    payload = _load_fixture()
    session = _mock_session(payload)
    source = PrometheusMetricsSource(session=session, query_step_minutes=5)

    result = source.fetch_readings("10.244.1.6", T0, T0)  # single-step window

    assert session.get.call_args.args[0].endswith("/api/v1/query")
    assert session.get.call_args.kwargs["params"]["query"] == source.query
    assert "time" in session.get.call_args.kwargs["params"]
    assert list(result[config.FEATURE_COL]) == pytest.approx([34.102756])
    assert result.index[0] == T0


def test_fetch_readings_unknown_machine_returns_empty():
    session = _mock_session(_load_fixture())
    source = PrometheusMetricsSource(session=session)
    result = source.fetch_readings("10.244.9.9", T0, T0)
    assert result.empty
    assert list(result.columns) == [config.FEATURE_COL]


def test_fetch_readings_multiple_steps_issues_one_call_per_step():
    session = _mock_session(_load_fixture())
    source = PrometheusMetricsSource(session=session, query_step_minutes=5)

    result = source.fetch_readings("10.244.1.5", T0, T0 + timedelta(minutes=15))

    assert session.get.call_count == 4  # T0, +5, +10, +15
    assert len(result) == 4
    assert (result[config.FEATURE_COL] == 12.483921).all()


def test_fetch_readings_rejects_end_before_start():
    source = PrometheusMetricsSource(session=_mock_session(_load_fixture()))
    with pytest.raises(ValueError):
        source.fetch_readings("10.244.1.5", T0, T0 - timedelta(minutes=5))


def test_fetch_readings_raises_on_non_success_status():
    session = _mock_session({"status": "error", "error": "bad query"})
    source = PrometheusMetricsSource(session=session)
    with pytest.raises(PrometheusMetricsSourceError):
        source.fetch_readings("10.244.1.5", T0, T0)


def test_fetch_readings_raises_on_non_vector_result_type():
    session = _mock_session({"status": "success", "data": {"resultType": "matrix", "result": []}})
    source = PrometheusMetricsSource(session=session)
    with pytest.raises(PrometheusMetricsSourceError):
        source.fetch_readings("10.244.1.5", T0, T0)


def test_list_machine_ids_returns_sorted_deduplicated_ids():
    session = _mock_session(_load_fixture())
    source = PrometheusMetricsSource(session=session)
    assert source.list_machine_ids() == ["10.244.1.5", "10.244.1.6", "10.244.1.7"]


def test_fetch_readings_output_feeds_resample_readings_unchanged():
    # The whole point of Stage 3: real (here, mocked) Prometheus data must
    # land in the exact shape data._prepare_timeseries already produces, via
    # the SAME resample_readings Step 20 built and verified.
    session = _mock_session(_load_fixture())
    source = PrometheusMetricsSource(session=session, query_step_minutes=5)
    raw = source.fetch_readings("10.244.1.5", T0, T0 + timedelta(minutes=25))
    out = resample_readings(raw)
    assert list(out.columns) == [config.FEATURE_COL]
    assert not out[config.FEATURE_COL].isna().any()


def test_default_prometheus_url_matches_the_confirmed_in_cluster_target():
    source = PrometheusMetricsSource(session=_mock_session(_load_fixture()))
    assert source.prometheus_url == (
        "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090"
    )
