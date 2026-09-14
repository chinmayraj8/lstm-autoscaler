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
            "hybrid_model_version",
        }
        # Hand-submitted via /shadow/{id}/window -- not tied to any real
        # loaded .keras file, so no version to report.
        assert w["hybrid_model_version"] is None


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


# ── /shadow/{machine_id}/decisions (Step 22, Stage 4 live loop) ────────────

def test_decisions_endpoint_empty_for_unseen_machine_not_404():
    # Unlike /shadow/{id} and /shadow/{id}/windows, a log endpoint returns
    # an empty list rather than 404 -- "never observed yet" is an ordinary
    # state here, not evidence the shadow-comparison gate has an opinion.
    r = client.get("/shadow/m_never_seen/decisions")
    assert r.status_code == 200
    assert r.json() == {"machine_id": "m_never_seen", "decisions": []}


def test_decisions_endpoint_reflects_observed_decisions_from_the_store():
    from datetime import datetime, timezone
    main_module._shadow_store.record_observed_decision(
        "m_test", datetime(2026, 1, 1, tzinfo=timezone.utc), forecaster="arima",
        forecast_cpu_pct=[5.1, 5.3, 5.2], planned_load_pct=142.5,
        current_servers=2, recommended_servers=2, action="hold",
    )
    r = client.get("/shadow/m_test/decisions")
    assert r.status_code == 200
    body = r.json()
    assert body["machine_id"] == "m_test"
    assert len(body["decisions"]) == 1
    d = body["decisions"][0]
    assert d["forecaster"] == "arima"
    assert d["forecast_cpu_pct"] == [5.1, 5.3, 5.2]
    assert d["action"] == "hold"
    assert d["recommended_servers"] == 2


# ── /health (Step 22 additions) ─────────────────────────────────────────────
# `client` (module-level, no `with`) never runs `lifespan` in this test
# file's established style (see the module docstring: these tests need
# neither lstm_model.keras nor the real dataset), so `_state` stays `{}`
# and `/health` correctly 503s over HTTP here -- that's the SAME behavior
# `/health` always had before Step 22, not a regression. These two tests
# call the route function directly against a manually-populated `_state`,
# the same way `_fresh_store` pokes `main_module._shadow_store` directly
# rather than trying to make a real lifespan run in this file.

def test_health_503_before_state_is_populated():
    main_module._state.clear()
    with pytest.raises(main_module.HTTPException) as exc_info:
        main_module.health()
    assert exc_info.value.status_code == 503


def test_health_reports_live_loop_and_model_flags_once_started():
    main_module._state.clear()
    main_module._state["started_at"] = "2026-01-01T00:00:00+00:00"
    try:
        body = main_module.health()
    finally:
        main_module._state.clear()
    assert body.status == "ok"
    assert body.lstm_model_loaded is False   # "model" never put into _state above
    assert body.live_loop_running is False   # no live-loop thread started in this test
    assert body.machine_id is None
