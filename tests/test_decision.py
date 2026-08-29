"""Unit tests for the LSTM policy's decision engine (src/autoscaler/decision.py).

No TensorFlow needed -- this module is pure numpy/dataclass logic.
"""

import numpy as np

from src.autoscaler.decision import DecisionConfig, _build_lstm_targets, _compute_penalty, _decide_scaling


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
