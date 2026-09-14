"""Unit tests for GET /metrics instrumentation (src/autoscaler/instrumentation.py).

The Counters/Gauges here live on one module-level `CollectorRegistry` for
the whole test process (same as production -- see that module's own
docstring), so:
  - Counters (DECISIONS_TOTAL, ACTUATION_TOTAL) are asserted by DELTA
    (value before vs. after the call under test), never by absolute value
    -- other tests in this same process may have already incremented the
    same label combination.
  - Gauges (RECOMMENDED_SERVERS, CIRCUIT_BREAKER_OPEN, SHADOW_WINDOW_COST)
    are safe to assert absolutely: `.set()` always overwrites, and each
    test below uses a machine_id unique to itself, so no other test's
    write can be the one this test's assertion observes.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

from src.autoscaler import config
from src.autoscaler.instrumentation import (
    ACTUATION_TOTAL,
    CIRCUIT_BREAKER_OPEN,
    DECISIONS_TOTAL,
    RECOMMENDED_SERVERS,
    SHADOW_WINDOW_COST,
    render_metrics,
)
from src.autoscaler.live_loop import REACTIVE_FALLBACK_FORECASTER, LiveLoopConfig, observe_node_once, run_tick
from src.autoscaler.metrics_source import StaticMetricsSource, synthetic_readings_series
from src.autoscaler.shadow import ShadowWindowResult, cumulative_summary
from src.autoscaler.shadow_store import ShadowStore, run_shadow_cycle

T0 = datetime(2026, 1, 1)


def _source(machine_id, n_hours=4, cadence_minutes=5, seed=1):
    n_points = int(n_hours * 60 / cadence_minutes) + 1
    series = synthetic_readings_series(T0, n_points=n_points, cadence_minutes=cadence_minutes,
                                       mean=40.0, amplitude=8.0, noise_std=1.5, seed=seed)
    return StaticMetricsSource({machine_id: series})


def _counter_value(counter, **labels):
    return counter.labels(**labels)._value.get()


def test_render_metrics_includes_all_five_metric_families():
    body = render_metrics().decode()
    for name in (
        "autoscaler_decisions_total",
        "autoscaler_recommended_servers",
        "autoscaler_actuation_total",
        "autoscaler_circuit_breaker_open",
        "autoscaler_shadow_window_cost",
    ):
        assert name in body


def test_observe_node_once_arima_path_increments_decisions_and_sets_recommended():
    machine_id = "m_instr_arima"
    source = _source(machine_id)
    store = ShadowStore(":memory:")
    cfg = LiveLoopConfig(fit_lookback_hours=2.0)

    before = _counter_value(DECISIONS_TOTAL, machine_id=machine_id, forecaster="arima", action="hold") + \
        _counter_value(DECISIONS_TOTAL, machine_id=machine_id, forecaster="arima", action="scale_up") + \
        _counter_value(DECISIONS_TOTAL, machine_id=machine_id, forecaster="arima", action="scale_down")

    result = observe_node_once(machine_id, source, store, T0 + timedelta(hours=3), cfg)

    after = _counter_value(DECISIONS_TOTAL, machine_id=machine_id, forecaster="arima", action="hold") + \
        _counter_value(DECISIONS_TOTAL, machine_id=machine_id, forecaster="arima", action="scale_up") + \
        _counter_value(DECISIONS_TOTAL, machine_id=machine_id, forecaster="arima", action="scale_down")

    assert after == before + 1
    assert RECOMMENDED_SERVERS.labels(machine_id=machine_id)._value.get() == result["recommended_servers"]


def test_observe_node_once_reactive_fallback_labels_the_counter_distinctly():
    machine_id = "m_instr_fallback"
    source = _source(machine_id)
    store = ShadowStore(":memory:")
    cfg = LiveLoopConfig(fit_lookback_hours=2.0)

    before = _counter_value(DECISIONS_TOTAL, machine_id=machine_id,
                            forecaster=REACTIVE_FALLBACK_FORECASTER, action="hold") + \
        _counter_value(DECISIONS_TOTAL, machine_id=machine_id,
                       forecaster=REACTIVE_FALLBACK_FORECASTER, action="scale_up") + \
        _counter_value(DECISIONS_TOTAL, machine_id=machine_id,
                       forecaster=REACTIVE_FALLBACK_FORECASTER, action="scale_down")

    with patch("src.autoscaler.live_loop._arima_forecast_once", side_effect=RuntimeError("boom")):
        observe_node_once(machine_id, source, store, T0 + timedelta(hours=3), cfg)

    after = _counter_value(DECISIONS_TOTAL, machine_id=machine_id,
                           forecaster=REACTIVE_FALLBACK_FORECASTER, action="hold") + \
        _counter_value(DECISIONS_TOTAL, machine_id=machine_id,
                       forecaster=REACTIVE_FALLBACK_FORECASTER, action="scale_up") + \
        _counter_value(DECISIONS_TOTAL, machine_id=machine_id,
                       forecaster=REACTIVE_FALLBACK_FORECASTER, action="scale_down")

    assert after == before + 1
    # And the plain "arima" counter for this same machine did NOT move.
    arima_total = _counter_value(DECISIONS_TOTAL, machine_id=machine_id, forecaster="arima", action="hold") + \
        _counter_value(DECISIONS_TOTAL, machine_id=machine_id, forecaster="arima", action="scale_up") + \
        _counter_value(DECISIONS_TOTAL, machine_id=machine_id, forecaster="arima", action="scale_down")
    assert arima_total == 0


def test_run_tick_actuation_success_records_result_and_closes_breaker_gauge():
    machine_id = "m_instr_act_ok"
    source = _source(machine_id)
    store = ShadowStore(":memory:")
    cfg = LiveLoopConfig(
        fit_lookback_hours=2.0, enable_actuation=True, actuation_machine_id=machine_id,
        actuation_deployment="demo-workload", actuation_namespace="lstm-autoscaler",
    )

    before = _counter_value(ACTUATION_TOTAL, machine_id=machine_id, result="success")

    with patch("src.autoscaler.actuator.get_current_replicas", return_value=config.SIM_INITIAL_SERVERS), \
         patch("src.autoscaler.actuator.set_replicas"):
        run_tick(store, source, [machine_id], T0 + timedelta(hours=3), cfg)

    after = _counter_value(ACTUATION_TOTAL, machine_id=machine_id, result="success")
    assert after == before + 1
    assert CIRCUIT_BREAKER_OPEN.labels(machine_id=machine_id)._value.get() == 0.0


def test_run_tick_actuation_failure_records_result_and_opens_breaker_gauge():
    machine_id = "m_instr_act_fail"
    source = _source(machine_id, n_hours=6)
    store = ShadowStore(":memory:")
    cfg = LiveLoopConfig(
        fit_lookback_hours=2.0, enable_actuation=True, actuation_machine_id=machine_id,
        actuation_circuit_breaker_threshold=1, actuation_circuit_breaker_cooldown=timedelta(hours=1),
    )

    before = _counter_value(ACTUATION_TOTAL, machine_id=machine_id, result="failure")

    from src.autoscaler.actuator import ActuationError
    with patch("src.autoscaler.actuator.get_current_replicas", return_value=config.SIM_INITIAL_SERVERS), \
         patch("src.autoscaler.actuator.set_replicas", side_effect=ActuationError("boom")):
        run_tick(store, source, [machine_id], T0 + timedelta(hours=3), cfg)

    after = _counter_value(ACTUATION_TOTAL, machine_id=machine_id, result="failure")
    assert after == before + 1
    # Threshold=1 -> this single failure trips the breaker immediately.
    assert CIRCUIT_BREAKER_OPEN.labels(machine_id=machine_id)._value.get() == 1.0


def test_run_shadow_cycle_banking_a_window_sets_the_cost_gauge():
    machine_id = "m_instr_shadow"
    store = ShadowStore(":memory:")

    window = ShadowWindowResult(
        machine_id=machine_id, window_start=T0, window_end=T0 + timedelta(hours=24),
        arima_cost=0.4, arima_sla_pct=2.0, hybrid_cost=0.1, hybrid_sla_pct=0.5,
    )
    run_shadow_cycle(store, machine_id, T0, lambda: window, min_windows=3, cadence_days=30)

    full_history = store.get_full_window_history(machine_id)
    expected = cumulative_summary(full_history)

    assert SHADOW_WINDOW_COST.labels(machine_id=machine_id, forecaster="arima")._value.get() == \
        expected["arima_cost_total"]
    assert SHADOW_WINDOW_COST.labels(machine_id=machine_id, forecaster="hybrid")._value.get() == \
        expected["hybrid_cost_total"]
