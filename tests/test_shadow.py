"""Unit tests for the shadow-mode evaluation harness (src/autoscaler/shadow.py).

No TensorFlow, no ARIMA fitting, no real dataset -- same convention as
test_arima_baseline.py and test_decision.py: shadow.py only needs numpy
(via decision.py/simulation.py), so these tests run fast with pure
synthetic (y_pred_real, y_actual_real) arrays and hand-constructed
ShadowWindowResult records, exercising the windowing, statistical-bar
gating, and reassignment logic directly.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.autoscaler.decision import DecisionConfig
from src.autoscaler.simulation import SimConfig
from src.autoscaler.shadow import (
    AssignmentChange,
    MachineShadowState,
    ShadowWindowResult,
    aggregate_verdict,
    cumulative_summary,
    decide_assignment,
    evaluate_and_maybe_reassign,
    hybrid_wins_every_window,
    is_reevaluation_due,
    maybe_run_shadow_cycle,
    record_shadow_window,
    run_shadow_window,
    score_forecaster_window,
)

T0 = datetime(2026, 1, 1)


def _window(hybrid_cost, arima_cost, hybrid_sla=1.0, arima_sla=1.0, mid="m_test"):
    return ShadowWindowResult(
        machine_id=mid, window_start=T0, window_end=T0 + timedelta(hours=24),
        arima_cost=arima_cost, arima_sla_pct=arima_sla,
        hybrid_cost=hybrid_cost, hybrid_sla_pct=hybrid_sla,
    )


# ── score_forecaster_window / run_shadow_window: reuses the real sim machinery ──

def test_score_forecaster_window_perfect_forecast_has_near_zero_cost():
    # A forecast that exactly matches actual demand should never under- or
    # over-provision by more than the fixed safety margin/capacity granularity.
    n_seq, horizon = 20, 3
    rng = np.random.RandomState(0)
    actual = rng.uniform(40, 60, size=(n_seq, horizon)).astype(np.float32)
    dec_cfg = DecisionConfig(under_prov_weight=5)
    sim_cfg = SimConfig()

    cost, sla = score_forecaster_window(actual.copy(), actual, dec_cfg, sim_cfg,
                                        demand_scale=1.0, safety_margin=0.5)
    assert cost < 0.05
    assert sla < 5.0


def test_score_forecaster_window_bad_forecast_costs_more_than_good_forecast():
    # Demand held near a level that needs ~3 servers (80%/server capacity) --
    # a forecast that tracks it lets the decision engine ramp up and cover
    # it; a forecast stuck near zero keeps the fleet at minimum size and
    # causes sustained under-provisioning.
    n_seq, horizon = 20, 3
    rng = np.random.RandomState(1)
    actual = rng.uniform(150, 160, size=(n_seq, horizon)).astype(np.float32)
    good_forecast = actual.copy()
    bad_forecast = np.full_like(actual, 1.0)

    dec_cfg = DecisionConfig(under_prov_weight=20)
    sim_cfg = SimConfig()
    good_cost, _ = score_forecaster_window(good_forecast, actual, dec_cfg, sim_cfg, 1.0, 0.2)
    bad_cost, bad_sla = score_forecaster_window(bad_forecast, actual, dec_cfg, sim_cfg, 1.0, 0.2)

    assert bad_cost > good_cost
    assert bad_sla > 0


def test_run_shadow_window_scores_both_forecasters_against_the_same_actual():
    n_seq, horizon = 15, 3
    rng = np.random.RandomState(2)
    actual = rng.uniform(30, 70, size=(n_seq, horizon)).astype(np.float32)
    arima_pred = actual + rng.normal(0, 2, size=actual.shape).astype(np.float32)
    hybrid_pred = actual.copy()  # hybrid predicts perfectly here -- should win

    dec_cfg = DecisionConfig(under_prov_weight=10)
    result = run_shadow_window(
        "m_test", T0, T0 + timedelta(hours=24), actual,
        arima_pred, dec_cfg, 0.25,
        hybrid_pred, dec_cfg, 0.25,
        demand_scale=1.0,
    )
    assert result.machine_id == "m_test"
    assert result.hybrid_cost <= result.arima_cost


# ── hybrid_wins_every_window: strict per-window consistency, not just mean ──

def test_hybrid_wins_every_window_true_when_all_windows_favor_hybrid():
    windows = [_window(hybrid_cost=0.1, arima_cost=0.2) for _ in range(3)]
    assert hybrid_wins_every_window(windows) is True


def test_hybrid_wins_every_window_false_if_any_single_window_loses():
    windows = [
        _window(hybrid_cost=0.05, arima_cost=0.20),
        _window(hybrid_cost=0.05, arima_cost=0.20),
        _window(hybrid_cost=0.30, arima_cost=0.20),  # hybrid loses this one
    ]
    # mean hybrid cost (0.1333) is still well below mean arima cost (0.20) --
    # this is exactly the single-bad-window case the strict rule must catch.
    assert np.mean([w.hybrid_cost for w in windows]) < np.mean([w.arima_cost for w in windows])
    assert hybrid_wins_every_window(windows) is False


def test_hybrid_wins_every_window_false_on_empty_list():
    assert hybrid_wins_every_window([]) is False


# ── aggregate_verdict: the same combined-+/-1sigma rule as everywhere else ──

def test_aggregate_verdict_confirmed_cheaper_when_gap_exceeds_combined_std():
    windows = [_window(hybrid_cost=c, arima_cost=0.30) for c in [0.10, 0.11, 0.09]]
    assert aggregate_verdict(windows) == "confirmed cheaper"


def test_aggregate_verdict_tied_when_gap_within_combined_std():
    # hybrid costs vary enough that the small mean gap doesn't clear ±1sigma
    windows = [_window(hybrid_cost=c, arima_cost=0.20) for c in [0.05, 0.35, 0.10]]
    assert aggregate_verdict(windows) in ("tied (within ±1σ)", "directional (within ±1σ)")


def test_aggregate_verdict_confirmed_more_expensive_when_arima_clearly_wins():
    windows = [_window(hybrid_cost=0.30, arima_cost=c) for c in [0.10, 0.11, 0.09]]
    assert aggregate_verdict(windows) == "confirmed more expensive"


# ── decide_assignment: consistency AND significance, both required ─────────

def test_decide_assignment_defaults_to_arima_below_min_windows():
    windows = [_window(hybrid_cost=0.05, arima_cost=0.20)]  # only 1, need >=3
    forecaster, verdict_label, evidence = decide_assignment(windows, min_windows=3)
    assert forecaster == "arima"
    assert verdict_label == "insufficient_windows"
    assert "1/3" in evidence


def test_decide_assignment_hybrid_when_consistent_and_confirmed():
    windows = [_window(hybrid_cost=c, arima_cost=0.30) for c in [0.10, 0.11, 0.09]]
    forecaster, verdict_label, _ = decide_assignment(windows, min_windows=3)
    assert forecaster == "hybrid"
    assert verdict_label == "confirmed cheaper"


def test_decide_assignment_stays_arima_when_inconsistent_even_if_mean_favors_hybrid():
    windows = [
        _window(hybrid_cost=0.05, arima_cost=0.20),
        _window(hybrid_cost=0.05, arima_cost=0.20),
        _window(hybrid_cost=0.30, arima_cost=0.20),  # one loss breaks consistency
    ]
    forecaster, _, evidence = decide_assignment(windows, min_windows=3)
    assert forecaster == "arima"
    assert "inconsistent" in evidence


def test_decide_assignment_stays_arima_when_consistent_but_not_statistically_confirmed():
    # hybrid wins every window (all three costs are below arima's constant
    # 0.30) but with high variance, so the mean gap doesn't clear the
    # combined-std bar -- consistency alone must not be enough without
    # significance.
    windows = [_window(hybrid_cost=c, arima_cost=0.30) for c in [0.28, 0.29, 0.05]]
    assert hybrid_wins_every_window(windows) is True
    forecaster, verdict_label, _ = decide_assignment(windows, min_windows=3)
    assert forecaster == "arima"
    assert verdict_label != "confirmed cheaper"


# ── record_shadow_window: rolling window cap ────────────────────────────────

def test_record_shadow_window_keeps_only_most_recent_n():
    state = MachineShadowState(machine_id="m_test")
    for i in range(5):
        record_shadow_window(state, _window(hybrid_cost=0.1 + i, arima_cost=0.2), keep_last_n=3)
    assert len(state.window_results) == 3
    # the oldest two (hybrid_cost 0.1, 1.1) should have been dropped
    assert [round(w.hybrid_cost, 2) for w in state.window_results] == [2.1, 3.1, 4.1]


# ── is_reevaluation_due ──────────────────────────────────────────────────────

def test_is_reevaluation_due_true_when_never_evaluated():
    state = MachineShadowState(machine_id="m_test")
    assert is_reevaluation_due(state, T0, cadence_days=30) is True


def test_is_reevaluation_due_false_within_cadence():
    state = MachineShadowState(machine_id="m_test", last_evaluated_at=T0)
    assert is_reevaluation_due(state, T0 + timedelta(days=10), cadence_days=30) is False


def test_is_reevaluation_due_true_after_cadence_elapses():
    state = MachineShadowState(machine_id="m_test", last_evaluated_at=T0)
    assert is_reevaluation_due(state, T0 + timedelta(days=31), cadence_days=30) is True


# ── evaluate_and_maybe_reassign: the audit trail ────────────────────────────

def test_evaluate_and_maybe_reassign_logs_first_assignment_to_hybrid():
    state = MachineShadowState(machine_id="m_test")
    for c in [0.10, 0.11, 0.09]:
        record_shadow_window(state, _window(hybrid_cost=c, arima_cost=0.30))

    change = evaluate_and_maybe_reassign(state, T0)

    assert change is not None
    assert isinstance(change, AssignmentChange)
    assert change.old_forecaster == "arima"
    assert change.new_forecaster == "hybrid"
    assert change.timestamp == T0
    assert state.current_forecaster == "hybrid"
    assert state.last_evaluated_at == T0
    assert state.assignment_history == [change]


def test_evaluate_and_maybe_reassign_no_change_when_bar_not_cleared():
    state = MachineShadowState(machine_id="m_test")  # starts on "arima"
    record_shadow_window(state, _window(hybrid_cost=0.05, arima_cost=0.20))  # only 1 window

    change = evaluate_and_maybe_reassign(state, T0)

    assert change is None
    assert state.current_forecaster == "arima"
    assert state.assignment_history == []
    assert state.last_evaluated_at == T0  # still stamped, even with no change


def test_evaluate_and_maybe_reassign_reverts_to_arima_after_drift():
    # Machine was previously (correctly) assigned to the hybrid ...
    state = MachineShadowState(machine_id="m_test", current_forecaster="hybrid")
    # ... but its most recent shadow windows now show ARIMA winning instead
    # (simulating workload drift discovered at a later re-evaluation).
    for c in [0.10, 0.11, 0.09]:
        record_shadow_window(state, _window(hybrid_cost=0.30, arima_cost=c))

    change = evaluate_and_maybe_reassign(state, T0 + timedelta(days=30))

    assert change is not None
    assert change.old_forecaster == "hybrid"
    assert change.new_forecaster == "arima"
    assert state.current_forecaster == "arima"


# ── maybe_run_shadow_cycle: caller-driven scheduling, no wasted shadow compute ──

def test_maybe_run_shadow_cycle_pulls_a_window_when_still_accumulating():
    state = MachineShadowState(machine_id="m_test")
    calls = {"n": 0}

    def get_window():
        calls["n"] += 1
        return _window(hybrid_cost=0.1, arima_cost=0.2)

    maybe_run_shadow_cycle(state, T0, get_window, min_windows=3, cadence_days=30)
    assert calls["n"] == 1
    assert len(state.window_results) == 1


def test_maybe_run_shadow_cycle_noops_once_stable_and_not_due():
    state = MachineShadowState(machine_id="m_test")
    for c in [0.10, 0.11, 0.09]:
        record_shadow_window(state, _window(hybrid_cost=c, arima_cost=0.30))
    evaluate_and_maybe_reassign(state, T0)  # now evaluated + stable (assigned to hybrid)

    calls = {"n": 0}

    def get_window():
        calls["n"] += 1
        return _window(hybrid_cost=0.1, arima_cost=0.2)

    result = maybe_run_shadow_cycle(state, T0 + timedelta(days=5), get_window,
                                    min_windows=3, cadence_days=30)
    assert result is None
    assert calls["n"] == 0  # no wasted shadow-window compute


def test_maybe_run_shadow_cycle_runs_again_once_cadence_elapses():
    state = MachineShadowState(machine_id="m_test")
    for c in [0.10, 0.11, 0.09]:
        record_shadow_window(state, _window(hybrid_cost=c, arima_cost=0.30))
    evaluate_and_maybe_reassign(state, T0)

    calls = {"n": 0}

    def get_window():
        calls["n"] += 1
        return _window(hybrid_cost=0.10, arima_cost=0.30)

    maybe_run_shadow_cycle(state, T0 + timedelta(days=31), get_window,
                           min_windows=3, cadence_days=30)
    assert calls["n"] == 1


# ── cumulative_summary ───────────────────────────────────────────────────────

def test_cumulative_summary_empty():
    assert cumulative_summary([]) == {"n_windows": 0}


def test_cumulative_summary_computes_mean_and_total():
    windows = [_window(hybrid_cost=c, arima_cost=a) for c, a in [(0.1, 0.2), (0.3, 0.4)]]
    summary = cumulative_summary(windows)
    assert summary["n_windows"] == 2
    assert summary["hybrid_cost_mean"] == pytest.approx(0.2)
    assert summary["hybrid_cost_total"] == pytest.approx(0.4)
    assert summary["arima_cost_mean"] == pytest.approx(0.3)
