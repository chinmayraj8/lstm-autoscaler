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
from unittest.mock import patch

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
from src.autoscaler.shadow import AssignmentChange
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


# ── run_tick: real actuation (Step 26), off by default ──────────────────────

def test_run_tick_actuation_disabled_by_default_never_calls_actuator():
    source = StaticMetricsSource({
        "m_a": synthetic_readings_series(T0, n_points=60, cadence_minutes=5, seed=1),
    })
    store = ShadowStore(":memory:")
    now = T0 + timedelta(hours=3)
    cfg = LiveLoopConfig(fit_lookback_hours=2.0)  # enable_actuation defaults to False

    with patch("src.autoscaler.actuator.set_replicas") as fake_set_replicas:
        summary = run_tick(store, source, ["m_a"], now, cfg)

    fake_set_replicas.assert_not_called()
    assert summary["actuated"] == []
    assert summary["actuation_skipped"] == []


def test_run_tick_actuation_enabled_calls_actuator_only_for_the_designated_machine():
    source = StaticMetricsSource({
        "m_a": synthetic_readings_series(T0, n_points=60, cadence_minutes=5, seed=1),
        "m_b": synthetic_readings_series(T0, n_points=60, cadence_minutes=5, seed=2),
    })
    store = ShadowStore(":memory:")
    now = T0 + timedelta(hours=3)
    cfg = LiveLoopConfig(
        fit_lookback_hours=2.0, enable_actuation=True, actuation_machine_id="m_a",
        actuation_deployment="demo-workload", actuation_namespace="lstm-autoscaler",
    )

    with patch("src.autoscaler.actuator.set_replicas") as fake_set_replicas:
        summary = run_tick(store, source, ["m_a", "m_b"], now, cfg)

    assert summary["actuated"] == ["m_a"]
    assert summary["actuation_skipped"] == []
    fake_set_replicas.assert_called_once()
    call_args = fake_set_replicas.call_args[0]
    assert call_args[0] == "demo-workload"
    assert call_args[1] == "lstm-autoscaler"
    expected_replicas = store.get_observed_decisions("m_a")[0]["recommended_servers"]
    assert call_args[2] == expected_replicas


def test_run_tick_actuation_failure_is_caught_and_logged_not_crashed():
    source = StaticMetricsSource({
        "m_a": synthetic_readings_series(T0, n_points=60, cadence_minutes=5, seed=1),
    })
    store = ShadowStore(":memory:")
    now = T0 + timedelta(hours=3)
    cfg = LiveLoopConfig(fit_lookback_hours=2.0, enable_actuation=True, actuation_machine_id="m_a")

    from src.autoscaler.actuator import ActuationError
    with patch("src.autoscaler.actuator.set_replicas", side_effect=ActuationError("boom")):
        summary = run_tick(store, source, ["m_a"], now, cfg)  # must not raise

    assert summary["actuated"] == []
    assert summary["actuation_skipped"] == ["m_a"]
    assert summary["observed"] == ["m_a"]  # the rest of the tick still ran


# ── run_tick: read-before-write reconciliation (Step 26 follow-up) ─────────
# `get_current_replicas` is unmocked in the three tests above -- against
# this test suite's real environment (no in-cluster token, no kubeconfig)
# it genuinely raises ActuationError via `_load_config` (see
# test_actuator.py), which the reconciliation code below treats as
# "couldn't verify" and falls back on, unreconciled -- exactly reproducing
# this loop's pre-reconciliation behavior. That fallback path is what kept
# those three tests passing unmodified; the tests below cover the new
# behavior explicitly, with `get_current_replicas` mocked.

def test_run_tick_actuation_reconciles_when_real_replicas_disagree_with_the_ledger():
    source = StaticMetricsSource({
        "m_a": synthetic_readings_series(T0, n_points=60, cadence_minutes=5, seed=1),
    })
    store = ShadowStore(":memory:")
    now = T0 + timedelta(hours=3)
    cfg = LiveLoopConfig(
        fit_lookback_hours=2.0, enable_actuation=True, actuation_machine_id="m_a",
        actuation_deployment="demo-workload", actuation_namespace="lstm-autoscaler",
    )

    # A plausible in-bounds drift (still within [min_servers, max_servers])
    # from the ledger's default `config.SIM_INITIAL_SERVERS` -- big enough
    # that whatever `_decide_scaling` recomputes from it is guaranteed to
    # differ from this tick's un-reconciled recommendation, but not so far
    # outside [min_servers, max_servers] that the candidate-clamping in
    # `_decide_scaling` itself (not this reconciliation code) produces a
    # confusing edge case unrelated to what this test is checking.
    real_replicas = config.DEC_MAX_SERVERS - 3
    assert config.DEC_MIN_SERVERS < real_replicas < config.DEC_MAX_SERVERS
    assert real_replicas != config.SIM_INITIAL_SERVERS

    with patch("src.autoscaler.actuator.get_current_replicas", return_value=real_replicas) as fake_get,          patch("src.autoscaler.actuator.set_replicas") as fake_set:
        summary = run_tick(store, source, ["m_a"], now, cfg)

    fake_get.assert_called_once_with("demo-workload", "lstm-autoscaler")
    assert summary["actuated"] == ["m_a"]
    assert summary["reconciled"] == ["m_a"]

    un_reconciled = store.get_observed_decisions("m_a")[0]["recommended_servers"]
    actuated_value = fake_set.call_args[0][2]
    assert actuated_value != un_reconciled  # the real count changed what got sent
    # The scale step cap still applies to the RECONCILED base, not the stale one.
    assert abs(actuated_value - real_replicas) <= config.DEC_SCALE_STEP
    # The ledger is corrected to match what was actually just applied.
    assert store.get_last_recommended_servers("m_a", default=-1) == actuated_value


def test_run_tick_actuation_skips_reconciliation_when_real_replicas_already_match():
    source = StaticMetricsSource({
        "m_a": synthetic_readings_series(T0, n_points=60, cadence_minutes=5, seed=1),
    })
    store = ShadowStore(":memory:")
    now = T0 + timedelta(hours=3)
    cfg = LiveLoopConfig(
        fit_lookback_hours=2.0, enable_actuation=True, actuation_machine_id="m_a",
        actuation_deployment="demo-workload", actuation_namespace="lstm-autoscaler",
    )

    with patch("src.autoscaler.actuator.get_current_replicas", return_value=config.SIM_INITIAL_SERVERS),          patch("src.autoscaler.actuator.set_replicas") as fake_set:
        summary = run_tick(store, source, ["m_a"], now, cfg)

    assert summary["reconciled"] == []
    un_reconciled = store.get_observed_decisions("m_a")[0]["recommended_servers"]
    assert fake_set.call_args[0][2] == un_reconciled


def test_run_tick_actuation_falls_back_when_reading_real_replicas_fails():
    source = StaticMetricsSource({
        "m_a": synthetic_readings_series(T0, n_points=60, cadence_minutes=5, seed=1),
    })
    store = ShadowStore(":memory:")
    now = T0 + timedelta(hours=3)
    cfg = LiveLoopConfig(fit_lookback_hours=2.0, enable_actuation=True, actuation_machine_id="m_a")

    from src.autoscaler.actuator import ActuationError
    with patch("src.autoscaler.actuator.get_current_replicas", side_effect=ActuationError("no cluster")),          patch("src.autoscaler.actuator.set_replicas") as fake_set:
        summary = run_tick(store, source, ["m_a"], now, cfg)  # must not raise

    assert summary["actuated"] == ["m_a"]
    assert summary["reconciled"] == []
    un_reconciled = store.get_observed_decisions("m_a")[0]["recommended_servers"]
    assert fake_set.call_args[0][2] == un_reconciled



# -- run_tick: actuation circuit breaker (Step 26 follow-up) ----------------
# get_current_replicas is mocked to succeed (real_current == the ledger's
# value) in every test below, so these isolate the breaker itself from the
# reconciliation behavior already covered above.

def test_run_tick_actuation_circuit_breaker_trips_after_threshold_and_then_skips():
    source = StaticMetricsSource({
        "m_a": synthetic_readings_series(T0, n_points=200, cadence_minutes=5, seed=1),
    })
    store = ShadowStore(":memory:")
    cfg = LiveLoopConfig(
        fit_lookback_hours=2.0, enable_actuation=True, actuation_machine_id="m_a",
        actuation_circuit_breaker_threshold=2, actuation_circuit_breaker_cooldown=timedelta(hours=1),
    )

    from src.autoscaler.actuator import ActuationError
    with patch("src.autoscaler.actuator.get_current_replicas", return_value=config.SIM_INITIAL_SERVERS):
        with patch("src.autoscaler.actuator.set_replicas", side_effect=ActuationError("boom")) as fake_set:
            # Two real, failed attempts -- the second crosses the threshold and trips it.
            run_tick(store, source, ["m_a"], T0 + timedelta(hours=3), cfg)
            run_tick(store, source, ["m_a"], T0 + timedelta(hours=3, minutes=5), cfg)
            assert fake_set.call_count == 2
        assert cfg._actuation_consecutive_failures == 2
        assert cfg._actuation_circuit_opened_at is not None

        # A third tick, still well inside the cooldown window: breaker is
        # OPEN, so set_replicas must NOT be called again this tick.
        with patch("src.autoscaler.actuator.set_replicas") as fake_set_during_cooldown:
            summary = run_tick(store, source, ["m_a"], T0 + timedelta(hours=3, minutes=10), cfg)

    fake_set_during_cooldown.assert_not_called()
    assert summary["actuation_skipped"] == ["m_a"]
    assert summary["actuated"] == []


def test_run_tick_actuation_circuit_breaker_probes_after_cooldown_and_closes_on_success():
    source = StaticMetricsSource({
        "m_a": synthetic_readings_series(T0, n_points=1000, cadence_minutes=5, seed=1),
    })
    store = ShadowStore(":memory:")
    cfg = LiveLoopConfig(
        fit_lookback_hours=2.0, enable_actuation=True, actuation_machine_id="m_a",
        actuation_circuit_breaker_threshold=1, actuation_circuit_breaker_cooldown=timedelta(hours=1),
    )

    from src.autoscaler.actuator import ActuationError
    with patch("src.autoscaler.actuator.get_current_replicas", return_value=config.SIM_INITIAL_SERVERS):
        with patch("src.autoscaler.actuator.set_replicas", side_effect=ActuationError("boom")):
            run_tick(store, source, ["m_a"], T0 + timedelta(hours=3), cfg)  # 1 failure -> trips (threshold=1)
        assert cfg._actuation_circuit_opened_at is not None

        # Still well inside the cooldown: breaker open, no attempt.
        with patch("src.autoscaler.actuator.set_replicas") as fake_set_during_cooldown:
            run_tick(store, source, ["m_a"], T0 + timedelta(hours=3, minutes=10), cfg)
        fake_set_during_cooldown.assert_not_called()

        # Past the cooldown: exactly one probe attempt, and it succeeds.
        with patch("src.autoscaler.actuator.set_replicas") as fake_set_probe:
            summary = run_tick(store, source, ["m_a"], T0 + timedelta(hours=4, minutes=5), cfg)

    fake_set_probe.assert_called_once()
    assert summary["actuated"] == ["m_a"]
    assert cfg._actuation_consecutive_failures == 0
    assert cfg._actuation_circuit_opened_at is None  # breaker fully closed again


def test_run_tick_actuation_circuit_breaker_open_does_not_skip_the_hybrid_shadow_path():
    # Regression check: an earlier draft of this feature used `continue`
    # to skip the rest of the per-machine loop body when the breaker was
    # open, which silently also skipped the UNRELATED hybrid shadow-window
    # path for the same machine on that tick -- these two paths must stay
    # independent, per this module's own docstring. The breaker is forced
    # open directly (white-box) rather than tripped via a real failing
    # run_tick call first, so this test is decoupled from shadow.py's own
    # reassignment logic (a real prior tick can legitimately revert m_a
    # back to "arima" -- a different, already-covered behavior, not what
    # this test is checking).
    source = _source(n_hours=6, cadence_minutes=5, machine_id="m_a")
    store = ShadowStore(":memory:")
    _seed_hybrid_assignment(store, "m_a")
    assert store.load_state("m_a").current_forecaster == "hybrid"

    cfg = LiveLoopConfig(
        fit_lookback_hours=1.0, shadow_fit_hours=1.0, shadow_window_hours=1.5,
        enable_actuation=True, actuation_machine_id="m_a",
        actuation_circuit_breaker_threshold=1, actuation_circuit_breaker_cooldown=timedelta(hours=1),
    )
    cfg._actuation_consecutive_failures = 1
    cfg._actuation_circuit_opened_at = T0 + timedelta(hours=3)  # forced open

    def _stub_loader(machine_id, model_dir):
        return _StubResidualModel()

    with patch("src.autoscaler.actuator.set_replicas") as fake_set:
        summary = run_tick(store, source, ["m_a"], T0 + timedelta(hours=3, minutes=10), cfg,
                           model_loader=_stub_loader)

    fake_set.assert_not_called()  # breaker open -- confirms this tick exercised that branch
    assert summary["actuation_skipped"] == ["m_a"]
    # The hybrid path must still have run for m_a this tick, unaffected.
    assert summary["shadow_evaluated"] == ["m_a"] or summary["shadow_skipped"] == ["m_a"]

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
