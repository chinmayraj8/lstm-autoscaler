"""Integration tests for the /shadow/* endpoints (src/api/main.py).

Needs TensorFlow (main.py imports `_build_lstm_model` at module level, same
as it always has) but no real dataset, no network, and no lstm_model.keras
file -- the /shadow endpoints don't touch the `lifespan`-loaded model at
all, and the fixture below never calls it.

Uses `LSTM_AUTOSCALER_SHADOW_DB=:memory:` (set before importing src.api.main)
so these tests never touch the real project-root shadow_state.db file.
"""

import os

os.environ["LSTM_AUTOSCALER_SHADOW_DB"] = ":memory:"

import pytest
from fastapi.testclient import TestClient

import src.api.main as main_module

client = TestClient(main_module.app)


def _window_payload(hybrid_scale=1.0, arima_scale=1.5, n_seq=10):
    actual = [[150.0, 152.0, 155.0] for _ in range(n_seq)]
    hybrid_fc = [[v * hybrid_scale for v in row] for row in actual]
    arima_fc = [[v * arima_scale for v in row] for row in actual]
    return dict(
        window_start="2026-01-01T00:00:00Z", window_end="2026-01-02T00:00:00Z",
        demand_scale=1.0, y_actual=actual,
        arima_forecast=arima_fc, arima_under_prov_weight=10, arima_safety_margin=0.2,
        hybrid_forecast=hybrid_fc, hybrid_under_prov_weight=10, hybrid_safety_margin=0.2,
    )


@pytest.fixture(autouse=True)
def _fresh_store():
    """Each test gets an isolated in-memory ShadowStore -- main.py's module-level
    _shadow_store is replaced directly rather than re-importing the module
    (re-import wouldn't give a fresh :memory: DB anyway, since Python caches
    imported modules)."""
    from src.autoscaler.shadow_store import ShadowStore
    main_module._shadow_store = ShadowStore(":memory:")
    yield


def test_shadow_status_404_for_unknown_machine():
    r = client.get("/shadow/m_never_seen")
    assert r.status_code == 404


def test_shadow_windows_404_for_unknown_machine():
    r = client.get("/shadow/m_never_seen/windows")
    assert r.status_code == 404


def test_submit_window_creates_machine_and_returns_result():
    r = client.post("/shadow/m_test/window", json=_window_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["machine_id"] == "m_test"
    assert body["current_forecaster"] == "arima"  # only 1 window, below min_windows
    assert body["assignment_changed"] is False
    assert body["n_banked_windows"] == 1
    assert "arima_cost" in body["window_result"]
    assert "hybrid_cost" in body["window_result"]


def test_three_consistent_hybrid_wins_trigger_assignment_change():
    for _ in range(3):
        r = client.post("/shadow/m_test/window", json=_window_payload(hybrid_scale=1.0, arima_scale=1.5))
    body = r.json()
    assert body["current_forecaster"] == "hybrid"
    assert body["assignment_changed"] is True
    assert body["verdict"] == "confirmed cheaper"


def test_status_endpoint_reflects_assignment_and_survives_across_requests():
    for _ in range(3):
        client.post("/shadow/m_test/window", json=_window_payload(hybrid_scale=1.0, arima_scale=1.5))

    r = client.get("/shadow/m_test")
    assert r.status_code == 200
    body = r.json()
    assert body["current_forecaster"] == "hybrid"
    assert body["n_banked_windows"] == 3
    assert len(body["assignment_history"]) == 1
    assert body["assignment_history"][0]["old_forecaster"] == "arima"
    assert body["assignment_history"][0]["new_forecaster"] == "hybrid"
    assert "timestamp" in body["assignment_history"][0]
    assert "evidence" in body["assignment_history"][0]


def test_windows_endpoint_lists_every_banked_window():
    for _ in range(4):
        client.post("/shadow/m_test/window", json=_window_payload())

    r = client.get("/shadow/m_test/windows")
    assert r.status_code == 200
    body = r.json()
    assert body["machine_id"] == "m_test"
    assert len(body["windows"]) == 4
    for w in body["windows"]:
        assert set(w.keys()) == {
            "window_start", "window_end", "arima_cost", "arima_sla_pct", "hybrid_cost", "hybrid_sla_pct",
        }


def test_mismatched_shapes_returns_400():
    payload = _window_payload()
    payload["hybrid_forecast"] = [[1.0, 2.0]]  # wrong shape vs y_actual/arima_forecast
    r = client.post("/shadow/m_test/window", json=payload)
    assert r.status_code == 400


def test_machines_are_isolated_over_http():
    client.post("/shadow/m_a/window", json=_window_payload(hybrid_scale=1.0, arima_scale=1.5))
    client.post("/shadow/m_b/window", json=_window_payload(hybrid_scale=1.5, arima_scale=1.0))

    a = client.get("/shadow/m_a").json()
    b = client.get("/shadow/m_b").json()
    assert a["cumulative"]["n_windows"] == 1
    assert b["cumulative"]["n_windows"] == 1
    assert a["cumulative"] != b["cumulative"]
