"""Integration tests for GET /forecast/confidence (src/api/main.py).

Real end-to-end path: Prometheus (mocked here, matching test_metrics_source.py's
own convention of never hitting a real network) -> resample_readings
(unmodified) -> arima_baseline._arima_forecast_with_ci (unmodified) ->
inverse-transform -> JSON. Nothing here touches _shadow_store or
LSTM_AUTOSCALER_SHADOW_DB.
"""

import os

os.environ["LSTM_AUTOSCALER_SHADOW_DB"] = ":memory:"

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.api.main as main_module
from src.autoscaler import config

client = TestClient(main_module.app)


def _synthetic_cpu_dataframe(n=40, seed=3):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=n, freq="5min")
    values = 5.0 + 0.5 * np.sin(np.arange(n) / 3.0) + rng.normal(0, 0.05, n)
    return pd.DataFrame({config.FEATURE_COL: values}, index=idx)


def test_returns_a_real_interval_that_straddles_the_point_forecast():
    df = _synthetic_cpu_dataframe()
    with patch("src.api.main.PrometheusMetricsSource") as MockSource:
        MockSource.return_value.fetch_readings.return_value = df
        r = client.get(
            "/forecast/confidence",
            params={"machine_id": "m_test", "prometheus_url": "http://fake:9090"},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["machine_id"] == "m_test"
    assert body["confidence_level"] == 0.95
    assert len(body["forecast"]) == 3
    for point in body["forecast"]:
        assert point["lower_pct"] <= point["cpu_pct"] <= point["upper_pct"]
    # step_minutes must be 5, 10, 15 -- HORIZON_STEPS at 5-minute cadence.
    assert [p["step_minutes"] for p in body["forecast"]] == [5, 10, 15]


def test_narrower_confidence_level_gives_a_narrower_interval():
    df = _synthetic_cpu_dataframe()
    with patch("src.api.main.PrometheusMetricsSource") as MockSource:
        MockSource.return_value.fetch_readings.return_value = df
        r_95 = client.get(
            "/forecast/confidence",
            params={"machine_id": "m_test", "prometheus_url": "http://fake:9090", "confidence_level": 0.95},
        )
        r_50 = client.get(
            "/forecast/confidence",
            params={"machine_id": "m_test", "prometheus_url": "http://fake:9090", "confidence_level": 0.50},
        )

    width_95 = [p["upper_pct"] - p["lower_pct"] for p in r_95.json()["forecast"]]
    width_50 = [p["upper_pct"] - p["lower_pct"] for p in r_50.json()["forecast"]]
    assert all(w50 < w95 for w50, w95 in zip(width_50, width_95))


def test_422_when_not_enough_real_history_to_fit_arima():
    # 3 points is below MIN_FIT_POINTS (LOOKBACK_STEPS + 2 = 8) -- a real
    # scrape gap or brand-new node, same convention as the live loop.
    df = _synthetic_cpu_dataframe(n=3)
    with patch("src.api.main.PrometheusMetricsSource") as MockSource:
        MockSource.return_value.fetch_readings.return_value = df
        r = client.get(
            "/forecast/confidence",
            params={"machine_id": "m_test", "prometheus_url": "http://fake:9090"},
        )
    assert r.status_code == 422


@pytest.mark.parametrize("bad_level", [0.0, 1.0, 1.5, -0.2])
def test_400_for_confidence_level_outside_open_unit_interval(bad_level):
    r = client.get(
        "/forecast/confidence",
        params={"machine_id": "m_test", "prometheus_url": "http://fake:9090", "confidence_level": bad_level},
    )
    assert r.status_code == 400


def test_prometheus_connection_failure_returns_502():
    import requests

    with patch("src.api.main.PrometheusMetricsSource") as MockSource:
        MockSource.return_value.fetch_readings.side_effect = requests.exceptions.ConnectionError("boom")
        r = client.get(
            "/forecast/confidence",
            params={"machine_id": "m_test", "prometheus_url": "http://fake:9090"},
        )
    assert r.status_code == 502
