"""Unit tests for the decision engine (src/autoscaler/decision.py): both the
original greedy (max-of-horizon) engine and the new multi-step-aware one
added in Step 12.

No TensorFlow needed -- this module is pure numpy/dataclass logic.
"""

import numpy as np

from src.autoscaler.decision import (
    DecisionConfig,
    _build_lstm_targets,
    _build_multistep_targets,
    _compute_penalty,
    _compute_penalty_multistep,
    _decide_scaling,
    _decide_scaling_multistep,
)


def test_penalty_is_zero_when_capacity_exactly_matches_load():
    cfg = DecisionConfig()  # capacity_pct=80, over_w=1.0, under_w=20.0
    assert _compute_penalty(n_servers=2, predicted_load=160.0, cfg=cfg) == 0.0


def test_penalty_weights_under_provisioning_more_than_over_provisioning():
    cfg = DecisionConfig()
    under = _compute_penalty(n_servers=1, predicted_load=100.0, cfg=cfg)   # 20% under capacity
    over  = _compute_penalty(n_servers=3, predicted_load=100.0, cfg=cfg)   # 140% over capacity
    assert under == 4.0     # 20 * (20/100)
    assert round(over, 4) == 1.4  # 1.0 * (140/100)
    # under_prov_weight (20.0) >> over_prov_weight (1.0) is the whole point of
    # the asymmetric penalty -- a much smaller under-provisioning fraction
    # still outweighs a much larger over-provisioning fraction here.
    assert under > over


def test_decide_scaling_holds_when_hold_is_cheapest():
    cfg = DecisionConfig()
    action, new_n = _decide_scaling(current_servers=2, predicted_load=150.0, cfg=cfg)
    assert action == "hold"
    assert new_n == 2


def test_decide_scaling_scales_up_when_under_provisioned():
    cfg = DecisionConfig()
    action, new_n = _decide_scaling(current_servers=1, predicted_load=200.0, cfg=cfg)
    assert action == "scale_up +1"
    assert new_n == 2


def test_decide_scaling_scales_down_when_over_provisioned():
    cfg = DecisionConfig()
    action, new_n = _decide_scaling(current_servers=5, predicted_load=50.0, cfg=cfg)
    assert action == "scale_down -1"
    assert new_n == 4


def test_decide_scaling_clamps_at_max_servers():
    cfg = DecisionConfig(max_servers=10)
    action, new_n = _decide_scaling(current_servers=10, predicted_load=100_000.0, cfg=cfg)
    assert new_n == 10  # can't exceed max_servers even when wildly under-provisioned


def test_decide_scaling_clamps_at_min_servers():
    cfg = DecisionConfig(min_servers=1)
    action, new_n = _decide_scaling(current_servers=1, predicted_load=0.0, cfg=cfg)
    assert new_n == 1  # can't go below min_servers


def test_build_lstm_targets_uses_max_of_forecast_and_demand_scale():
    # Two steps; forecast horizon of 3 each. The engine only ever looks at
    # max() of each row -- this is the documented "throws away the shape of
    # the horizon" limitation, and the test locks in that current behavior.
    y_pred_real = np.array([
        [10.0, 50.0, 20.0],   # max = 50
        [5.0, 5.0, 5.0],      # max = 5
    ])
    y_actual_real = np.array([
        [30.0, 0.0, 0.0],
        [40.0, 0.0, 0.0],
    ])
    dec_cfg = DecisionConfig()
    sim_cfg = type("SimCfgStub", (), {"initial_servers": 2})()  # duck-typed, only needs .initial_servers

    targets, demand = _build_lstm_targets(
        y_pred_real, y_actual_real, dec_cfg, sim_cfg, demand_scale=1.0, safety_margin=0.0
    )

    # demand comes from actual (column 0) * demand_scale, unaffected by forecast
    np.testing.assert_array_equal(demand, np.array([30.0, 40.0]))
    assert len(targets) == 2


# ── Multi-step-aware engine (Step 12) ────────────────────────────────────────

def test_multistep_penalty_uses_uniform_mean_of_per_step_penalties():
    cfg = DecisionConfig()  # over_w=1.0, under_w=20.0, capacity_pct=80
    # n=1 -> cap=80. Step-by-step: 10 -> over_frac=0.70 -> penalty 0.70;
    # 100 -> under_frac=0.20 -> penalty 4.0; 10 -> 0.70 again. Mean of
    # [0.70, 4.0, 0.70] = 5.4/3 = 1.8.
    penalty = _compute_penalty_multistep(n_servers=1, predicted_loads=np.array([10.0, 100.0, 10.0]), cfg=cfg)
    assert round(penalty, 6) == 1.8


def test_multistep_penalty_dilutes_a_single_step_spike_relative_to_a_sustained_plateau():
    # The whole point of Step 12: the OLD engine only ever sees max(), so a
    # one-tick spike to 100 and a plateau steady at 100 look identical to it.
    cfg = DecisionConfig()
    spike = np.array([70.0] * 9 + [100.0])     # 9 calm steps, 1 spike step
    plateau = np.array([100.0] * 10)           # elevated at every step

    old_engine_penalty_both_cases = _compute_penalty(n_servers=1, predicted_load=100.0, cfg=cfg)
    spike_multistep = _compute_penalty_multistep(1, spike, cfg)
    plateau_multistep = _compute_penalty_multistep(1, plateau, cfg)

    # Old engine can't tell these apart -- both collapse to max()=100.
    assert _compute_penalty(1, float(np.max(spike)), cfg) == old_engine_penalty_both_cases
    assert _compute_penalty(1, float(np.max(plateau)), cfg) == old_engine_penalty_both_cases
    # New engine scores the spike far below the plateau.
    assert spike_multistep < plateau_multistep
    assert plateau_multistep == old_engine_penalty_both_cases  # plateau: mean == max when uniform


def test_decide_scaling_multistep_holds_for_brief_spike_where_greedy_scales_up():
    cfg = DecisionConfig()
    spike = np.array([70.0] * 9 + [100.0])

    old_action, old_n = _decide_scaling(current_servers=1, predicted_load=float(np.max(spike)), cfg=cfg)
    new_action, new_n = _decide_scaling_multistep(current_servers=1, predicted_loads=spike, cfg=cfg)

    assert (old_action, old_n) == ("scale_up +1", 2)
    assert (new_action, new_n) == ("hold", 1)


def test_decide_scaling_multistep_agrees_with_greedy_for_sustained_plateau():
    cfg = DecisionConfig()
    plateau = np.array([100.0] * 10)

    old_action, old_n = _decide_scaling(current_servers=1, predicted_load=float(np.max(plateau)), cfg=cfg)
    new_action, new_n = _decide_scaling_multistep(current_servers=1, predicted_loads=plateau, cfg=cfg)

    assert (old_action, old_n) == (new_action, new_n) == ("scale_up +1", 2)


def test_build_multistep_targets_diverges_from_greedy_on_a_spiky_forecast():
    # Same spike shape as above, run through both target-builders end to end
    # (two "sequences" in a row, so current_servers carries over between
    # them) -- proves the divergence survives the full _build_*_targets loop,
    # not just the single-call decision functions above.
    y_pred_real = np.array([
        [70.0] * 9 + [100.0],
        [70.0] * 9 + [100.0],
    ])
    y_actual_real = np.array([
        [30.0] + [0.0] * 9,
        [40.0] + [0.0] * 9,
    ])
    dec_cfg = DecisionConfig()
    sim_cfg = type("SimCfgStub", (), {"initial_servers": 1})()

    old_targets, old_demand = _build_lstm_targets(
        y_pred_real, y_actual_real, dec_cfg, sim_cfg, demand_scale=1.0, safety_margin=0.0
    )
    new_targets, new_demand = _build_multistep_targets(
        y_pred_real, y_actual_real, dec_cfg, sim_cfg, demand_scale=1.0, safety_margin=0.0
    )

    np.testing.assert_array_equal(old_targets, np.array([2, 2]))    # greedy scales up and stays up
    np.testing.assert_array_equal(new_targets, np.array([1, 1]))    # multistep holds throughout
    # demand extraction is identical either way -- only the decision differs
    np.testing.assert_array_equal(old_demand, new_demand)
    np.testing.assert_array_equal(new_demand, np.array([30.0, 40.0]))
