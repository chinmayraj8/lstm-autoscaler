"""Unit tests for the live forecasting loop (src/autoscaler/live_loop.py).

No TensorFlow, no real dataset, no network, no real cluster -- exercises
`observe_node_once`/`run_tick`/`_build_hybrid_window`/the scheduler against
`StaticMetricsSource` (synthetic, Step 20) and a `ShadowStore(":memory:")`
(Step 19). The hybrid path is exercised against a STUB residual model (a
plain object with a `.predict` method) injected via `model_loader` --
proving the plumbing is correct without requiring a real trained artifact,
which does not exist (see live_loop.py's module docstring).
"""

import threading
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.autoscaler import config
from src.autoscaler.live_loop import (
    HybridModelUnavailable,
    LiveLoopConfig,
    _build_hybrid_window,
    observe_node_once,
    resolve_tracked_machine_ids,
    run_scheduler_loop,
    run_tick,
)
from src.autoscaler.metrics_source import StaticMetricsSource, synthetic_readings_series
from src.autoscaler.shadow import AssignmentChange, ShadowWindowResult
from src.autoscaler.shadow_store import ShadowStore

T0 = datetime(2026, 1, 1)


def _source(n_hours=4, cadence_minutes=5, seed=1, machine_id="m_test"):
    n_points = int(n_hours * 60 / cadence_minutes) + 1
    series = synthetic_readings_series(T0, n_points=n_points, cadence_minutes=cadence_minutes,
                                       mean=40.0, amplitude=8.0, noise_std=1.5, seed=seed)
    return StaticMetricsSource({machine_id: series})


def _seed_hybrid_assignment(store, machine_id):
    """Put `machine_id` straight into a hybrid-assigned state with
    `last_evaluated_at` left unset -- i.e. "already assigned, due for its
    first live re-evaluation" -- rather than going through
    `record_and_evaluate` (which would also stamp `last_evaluated_at`,
    making the next `run_shadow_cycle` call a same-day no-op per Step 19's
    existing cadence logic, and defeating the point of these tests)."""
    store._save_assignment_change(AssignmentChange(
        timestamp=T0, machine_id=machine_id, old_forecaster="arima", new_forecaster="hybrid",
        verdict="confirmed cheaper", n_windows=3, evidence="seeded for test",
    ))


class _StubResidualModel:
    """Always predicts zero residual -- makes hybrid == ARIMA exactly, a
    deterministic, easy-to-assert-on stand-in for a real trained model.
    Output shape is (n_sequences, HORIZON_STEPS), matching a real model's
    `Dense(horizon)` output (forecasting._build_lstm_model) -- NOT the
    input's own (n_sequences, LOOKBACK_STEPS, 1) shape."""

    def predict(self, X, verbose=0):
        return np.zeros((X.shape[0], config.HORIZON_STEPS), dtype=np.float64)


# ── observe_node_once ────────────────────────────────────────────────────

def test_observe_node_once_logs_and_returns_a_decision():
    source = _source()
    store = ShadowStore(":memory:")
    now = T0 + timedelta(hours=3)
    cfg = LiveLoopConfig(fit_lookback_hours=2.0)

    result = observe_node_once("m_test", source, store, now, cfg)

    assert result is not None
    assert result["machine_id"] == "m_test"
    assert len(result["forecast_cpu_pct"]) == 3  # config.HORIZON_STEPS
    assert result["action"] in {"hold"} or "scale_" in result["action"]

    logged = store.get_observed_decisions("m_test")
    assert len(logged) == 1
    assert logged[0]["forecaster"] == "arima"
    assert logged[0]["recommended_servers"] == result["recommended_servers"]
    assert store.get_last_recommended_servers("m_test", default=-1) == result["recommended_servers"]


def test_observe_node_once_skips_on_insufficient_data():
    source = StaticMetricsSource({"m_sparse": synthetic_readings_series(T0, n_points=2, cadence_minutes=5)})
    store = ShadowStore(":memory:")
    result = observe_node_once("m_sparse", source, store, T0 + timedelta(hours=1), LiveLoopConfig())
    assert result is None
    assert store.get_observed_decisions("m_sparse") == []


def test_observe_node_once_carries_shadow_server_count_forward():
    source = _source(n_hours=6)
    store = ShadowStore(":memory:")
    cfg = LiveLoopConfig(fit_lookback_hours=2.0)

    r1 = observe_node_once("m_test", source, store, T0 + timedelta(hours=3), cfg)
    r2 = observe_node_once("m_test", source, store, T0 + timedelta(hours=3, minutes=5), cfg)

    assert r2["current_servers"] == r1["recommended_servers"]


# ── run_tick: two separate paths ────────────────────────────────────────────

def test_run_tick_arima_only_nodes_never_touch_shadow_windows():
    source = StaticMetricsSource({
        "m_a": synthetic_readings_series(T0, n_points=60, cadence_minutes=5, seed=1),
        "m_b": synthetic_readings_series(T0, n_points=60, cadence_minutes=5, seed=2),
    })
    store = ShadowStore(":memory:")
    now = T0 + timedelta(hours=3)
    cfg = LiveLoopConfig(fit_lookback_hours=2.0)

    summary = run_tick(store, source, ["m_a", "m_b"], now, cfg)

    assert set(summary["observed"]) == {"m_a", "m_b"}
    assert summary["shadow_evaluated"] == []
    assert summary["shadow_skipped"] == []
    assert store.get_full_window_history("m_a") == []
    assert store.get_full_window_history("m_b") == []
    assert len(store.get_observed_decisions("m_a")) == 1


def test_run_tick_hybrid_assigned_node_without_model_is_skipped_not_crashed():
    source = _source(n_hours=6)
    store = ShadowStore(":memory:")
    # Seed a hybrid assignment as if made via the existing manual path
    # (Steps 18-19) -- exactly as this stage's scoping assumes a real
    # assignment would arrive (the live loop never makes this assignment
    # itself; see live_loop.py's module docstring).
    _seed_hybrid_assignment(store, "m_test")
    assert store.load_state("m_test").current_forecaster == "hybrid"

    now = T0 + timedelta(hours=3)
    cfg = LiveLoopConfig(fit_lookback_hours=1.0, shadow_fit_hours=1.0, shadow_window_hours=1.5)
    summary = run_tick(store, source, ["m_test"], now, cfg)

    assert summary["observed"] == ["m_test"]  # ARIMA-only path still ran
    assert summary["shadow_skipped"] == ["m_test"]  # no real model -> skipped, not crashed
    assert summary["shadow_evaluated"] == []


def test_run_tick_hybrid_assigned_node_with_stub_model_banks_a_window():
    source = _source(n_hours=4, cadence_minutes=5)
    store = ShadowStore(":memory:")
    _seed_hybrid_assignment(store, "m_test")
    assert store.load_state("m_test").current_forecaster == "hybrid"

    now = T0 + timedelta(hours=3, minutes=30)
    cfg = LiveLoopConfig(fit_lookback_hours=1.0, shadow_fit_hours=1.0, shadow_window_hours=1.5)

    def _stub_loader(machine_id, model_dir):
        return _StubResidualModel()

    summary = run_tick(store, source, ["m_test"], now, cfg, model_loader=_stub_loader)

    assert summary["shadow_evaluated"] == ["m_test"]
    assert summary["shadow_skipped"] == []
    history = store.get_full_window_history("m_test")
    assert len(history) == 1
    # Zero-residual stub -> hybrid forecast == ARIMA forecast exactly -> equal cost.
    assert history[0].hybrid_cost == pytest.approx(history[0].arima_cost)


def test_build_hybrid_window_raises_when_model_missing():
    source = _source(n_hours=4)
    cfg = LiveLoopConfig(shadow_fit_hours=1.0, shadow_window_hours=1.5)
    with pytest.raises(HybridModelUnavailable):
        _build_hybrid_window("m_test", source, cfg, T0 + timedelta(hours=3, minutes=30))


def test_build_hybrid_window_raises_on_insufficient_history():
    source = _source(n_hours=1)  # far too little for shadow_fit + shadow_window
    cfg = LiveLoopConfig(shadow_fit_hours=6.0, shadow_window_hours=24.0)
    with pytest.raises(HybridModelUnavailable):
        _build_hybrid_window("m_test", source, cfg, T0 + timedelta(minutes=30),
                             model_loader=lambda mid, d: _StubResidualModel())


# ── Node discovery ────────────────────────────────────────────────────────

def test_resolve_tracked_machine_ids_prefers_env_var(monkeypatch):
    monkeypatch.setenv("LSTM_AUTOSCALER_TRACKED_NODES", "m_a, m_b ,m_c")
    source = StaticMetricsSource({})
    assert resolve_tracked_machine_ids(source) == ["m_a", "m_b", "m_c"]


def test_resolve_tracked_machine_ids_falls_back_to_discovery(monkeypatch):
    monkeypatch.delenv("LSTM_AUTOSCALER_TRACKED_NODES", raising=False)

    class _DiscoverableSource(StaticMetricsSource):
        def list_machine_ids(self):
            return ["m_x", "m_y"]

    assert resolve_tracked_machine_ids(_DiscoverableSource({})) == ["m_x", "m_y"]


def test_resolve_tracked_machine_ids_raises_without_either(monkeypatch):
    monkeypatch.delenv("LSTM_AUTOSCALER_TRACKED_NODES", raising=False)
    with pytest.raises(ValueError):
        resolve_tracked_machine_ids(StaticMetricsSource({}))


# ── Scheduler loop ──────────────────────────────────────────────────────────

def test_run_scheduler_loop_stops_after_max_ticks():
    source = StaticMetricsSource({})
    store = ShadowStore(":memory:")
    calls = {"n": 0}

    def get_ids():
        calls["n"] += 1
        return []

    run_scheduler_loop(store, source, get_ids, LiveLoopConfig(tick_seconds=0),
                       stop_event=threading.Event(), max_ticks=3)
    assert calls["n"] == 3


def test_run_scheduler_loop_survives_a_failing_tick():
    store = ShadowStore(":memory:")

    def get_ids():
        raise RuntimeError("boom")

    # Must not raise -- a bad tick is logged, not fatal to the loop.
    run_scheduler_loop(store, StaticMetricsSource({}), get_ids, LiveLoopConfig(tick_seconds=0),
                       stop_event=threading.Event(), max_ticks=2)
