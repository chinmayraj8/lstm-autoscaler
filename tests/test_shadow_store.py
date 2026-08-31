"""Unit tests for SQLite persistence (src/autoscaler/shadow_store.py).

No TensorFlow, no real dataset, no network -- uses pytest's `tmp_path` for
a real-but-temporary SQLite file (proving persistence actually survives a
"restart" -- a fresh ShadowStore pointed at the same file) and ":memory:"
for fast in-process tests where restart-survival isn't the point.

The central claim under test: shadow_store.py is a storage change, not a
logic change. Every scenario here mirrors one already covered against
plain in-memory `MachineShadowState` in test_shadow.py; the assertions
check that persisting and reloading produces the identical decision
shadow.py's own (untouched) functions would have produced in memory.
"""

from datetime import datetime, timedelta

import pytest

from src.autoscaler.shadow import (
    DEFAULT_MIN_WINDOWS,
    ShadowWindowResult,
    cumulative_summary,
)
from src.autoscaler.shadow_store import ShadowStore, record_and_evaluate, run_shadow_cycle

T0 = datetime(2026, 1, 1)


def _window(mid, hybrid_cost, arima_cost, day_offset=0):
    return ShadowWindowResult(
        machine_id=mid,
        window_start=T0 + timedelta(days=day_offset),
        window_end=T0 + timedelta(days=day_offset, hours=24),
        arima_cost=arima_cost, arima_sla_pct=1.0,
        hybrid_cost=hybrid_cost, hybrid_sla_pct=1.0,
    )


@pytest.fixture
def store():
    return ShadowStore(":memory:")


# ── Basic schema / empty-state behavior ─────────────────────────────────────

def test_fresh_store_has_no_machines():
    store = ShadowStore(":memory:")
    assert store.has_machine("m_test") is False
    assert store.list_machine_ids() == []


def test_load_state_for_unseen_machine_returns_default():
    store = ShadowStore(":memory:")
    state = store.load_state("m_never_seen")
    assert state.machine_id == "m_never_seen"
    assert state.current_forecaster == "arima"
    assert state.last_evaluated_at is None
    assert state.window_results == []
    assert state.assignment_history == []


# ── record_and_evaluate: mirrors the /shadow/{id}/window endpoint's flow ───

def test_record_and_evaluate_persists_windows_and_assigns_hybrid(store):
    for i, c in enumerate([0.10, 0.11, 0.09]):
        state, change = record_and_evaluate(store, "m_test", _window("m_test", c, 0.30, day_offset=i), T0)

    assert change is not None
    assert change.old_forecaster == "arima"
    assert change.new_forecaster == "hybrid"
    assert state.current_forecaster == "hybrid"

    # A completely fresh ShadowStore over the same file (simulated here via
    # the same in-memory connection, which the fixture keeps alive) sees
    # the identical state -- proving this isn't just returned-value state.
    reloaded = store.load_state("m_test")
    assert reloaded.current_forecaster == "hybrid"
    assert len(reloaded.window_results) == 3
    assert reloaded.assignment_history[-1].new_forecaster == "hybrid"


def test_record_and_evaluate_survives_a_real_restart(tmp_path):
    db_path = str(tmp_path / "shadow.db")
    store1 = ShadowStore(db_path)
    for i, c in enumerate([0.10, 0.11, 0.09]):
        record_and_evaluate(store1, "m_test", _window("m_test", c, 0.30, day_offset=i), T0)

    # Simulate a process restart: a brand-new ShadowStore instance, same file.
    store2 = ShadowStore(db_path)
    state = store2.load_state("m_test")
    assert state.current_forecaster == "hybrid"
    assert len(state.window_results) == 3
    assert len(state.assignment_history) == 1
    assert state.assignment_history[0].evidence == "hybrid confirmed cheaper across all 3 windows"


def test_record_and_evaluate_no_change_below_min_windows(store):
    state, change = record_and_evaluate(store, "m_test", _window("m_test", 0.05, 0.20), T0)
    assert change is None
    assert state.current_forecaster == "arima"
    reloaded = store.load_state("m_test")
    assert reloaded.current_forecaster == "arima"
    assert reloaded.last_evaluated_at == T0  # still stamped


def test_record_and_evaluate_reverts_after_drift(store):
    # First establish a hybrid assignment ...
    for i, c in enumerate([0.10, 0.11, 0.09]):
        record_and_evaluate(store, "m_test", _window("m_test", c, 0.30, day_offset=i), T0)
    assert store.load_state("m_test").current_forecaster == "hybrid"

    # ... then feed windows (a later re-evaluation) where ARIMA now wins.
    # With a 3-window cap, the reversion fires as soon as enough of the
    # freshly-banked "bad for hybrid" windows have pushed the old
    # hybrid-favoring ones out of the capped set -- not necessarily on the
    # very last window submitted, so collect every change across the loop
    # rather than asserting only on the final iteration's return value.
    later = T0 + timedelta(days=31)
    changes = []
    for i, c in enumerate([0.10, 0.11, 0.09]):
        _, change = record_and_evaluate(
            store, "m_test", _window("m_test", 0.30, c, day_offset=10 + i), later,
        )
        if change is not None:
            changes.append(change)

    assert len(changes) == 1
    assert changes[0].old_forecaster == "hybrid"
    assert changes[0].new_forecaster == "arima"
    assert store.load_state("m_test").current_forecaster == "arima"

    # Both the hybrid assignment and the reversion are in the durable log.
    history = store.get_assignment_history("m_test")
    assert [c.new_forecaster for c in history] == ["hybrid", "arima"]


# ── run_shadow_cycle: mirrors shadow.maybe_run_shadow_cycle, persisted ──────

def test_run_shadow_cycle_pulls_and_persists_a_window(store):
    calls = {"n": 0}

    def get_window():
        calls["n"] += 1
        return _window("m_test", 0.1, 0.2)

    run_shadow_cycle(store, "m_test", T0, get_window, min_windows=3, cadence_days=30)

    assert calls["n"] == 1
    assert len(store.load_state("m_test").window_results) == 1


def test_run_shadow_cycle_noops_once_stable_and_not_due(store):
    for i, c in enumerate([0.10, 0.11, 0.09]):
        record_and_evaluate(store, "m_test", _window("m_test", c, 0.30, day_offset=i), T0)
    assert store.load_state("m_test").current_forecaster == "hybrid"

    calls = {"n": 0}

    def get_window():
        calls["n"] += 1
        return _window("m_test", 0.1, 0.2)

    result = run_shadow_cycle(store, "m_test", T0 + timedelta(days=5), get_window,
                              min_windows=3, cadence_days=30)
    assert result is None
    assert calls["n"] == 0  # no wasted shadow compute, same guarantee as the in-memory version


def test_run_shadow_cycle_runs_again_once_cadence_elapses(store):
    for i, c in enumerate([0.10, 0.11, 0.09]):
        record_and_evaluate(store, "m_test", _window("m_test", c, 0.30, day_offset=i), T0)

    calls = {"n": 0}

    def get_window():
        calls["n"] += 1
        return _window("m_test", 0.10, 0.30, day_offset=100)

    run_shadow_cycle(store, "m_test", T0 + timedelta(days=31), get_window,
                     min_windows=3, cadence_days=30)
    assert calls["n"] == 1


# ── Full audit log vs. the decision-relevant capped window ─────────────────

def test_full_window_history_is_not_capped_even_though_decisions_use_recent_n(store):
    for i, c in enumerate([0.10, 0.11, 0.12, 0.13, 0.14]):
        record_and_evaluate(store, "m_test", _window("m_test", c, 0.30, day_offset=i),
                            T0, keep_last_n=3)

    # decision-relevant state only keeps the most recent 3 ...
    assert len(store.load_state("m_test", window_limit=3).window_results) == 3
    # ... but the full audit log keeps every window ever banked.
    full_history = store.get_full_window_history("m_test")
    assert len(full_history) == 5
    assert [round(w.hybrid_cost, 2) for w in full_history] == [0.10, 0.11, 0.12, 0.13, 0.14]


def test_load_state_window_cap_keeps_most_recent_by_window_start(store):
    for i, c in enumerate([0.10, 0.11, 0.12, 0.13, 0.14]):
        record_and_evaluate(store, "m_test", _window("m_test", c, 0.30, day_offset=i),
                            T0, keep_last_n=3)

    state = store.load_state("m_test", window_limit=3)
    assert [round(w.hybrid_cost, 2) for w in state.window_results] == [0.12, 0.13, 0.14]


# ── Multi-machine isolation ──────────────────────────────────────────────────

def test_machines_are_isolated_from_each_other(store):
    for i, c in enumerate([0.10, 0.11, 0.09]):
        record_and_evaluate(store, "m_a", _window("m_a", c, 0.30, day_offset=i), T0)
    record_and_evaluate(store, "m_b", _window("m_b", 0.25, 0.10), T0)  # arima clearly wins

    assert store.load_state("m_a").current_forecaster == "hybrid"
    assert store.load_state("m_b").current_forecaster == "arima"
    assert set(store.list_machine_ids()) == {"m_a", "m_b"}


# ── cumulative_summary still works unmodified against store-loaded windows ──

def test_cumulative_summary_on_reloaded_windows_matches_in_memory_shape(store):
    for i, c in enumerate([0.10, 0.11, 0.09]):
        record_and_evaluate(store, "m_test", _window("m_test", c, 0.30, day_offset=i), T0)
    windows = store.load_state("m_test").window_results
    summary = cumulative_summary(windows)
    assert summary["n_windows"] == 3
    assert summary["hybrid_cost_mean"] == pytest.approx((0.10 + 0.11 + 0.09) / 3)
    assert summary["arima_cost_mean"] == pytest.approx(0.30)
