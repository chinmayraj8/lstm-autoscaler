"""Unit tests for the fleet simulator and the Reactive policy
(src/autoscaler/simulation.py).

No TensorFlow needed -- pure numpy/dataclass logic.
"""

import numpy as np

from src.autoscaler.decision import DecisionConfig
from src.autoscaler.simulation import (
    SimConfig,
    _compute_cost_score,
    _reactive_autoscaler,
    _run_simulation,
)


def test_scale_up_is_delayed_by_startup_delay_steps():
    # initial_servers=2, startup_delay_steps=1 (the project's actual default:
    # a newly-requested server becomes active exactly one 5-min tick later).
    sim_cfg = SimConfig(initial_servers=2, server_capacity=80.0, startup_delay_steps=1)
    demand = np.array([1.0, 1.0, 1.0])       # demand value doesn't matter for this test
    targets = np.array([3, 3, 3])            # request 3 servers from the very first tick

    metrics = _run_simulation(demand, targets, sim_cfg)

    # Tick 1: still only 2 active (the new server hasn't started up yet)
    # Tick 2: the queued server has become active -> 3
    # Tick 3: stays at 3 (no further scaling requested)
    assert metrics.server_counts == [2, 3, 3]
    assert metrics.capacity_trace == [160.0, 240.0, 240.0]


def test_scale_down_is_immediate_not_delayed():
    sim_cfg = SimConfig(initial_servers=5, server_capacity=80.0, startup_delay_steps=1)
    demand = np.array([1.0, 1.0])
    targets = np.array([2, 2])   # request fewer servers

    metrics = _run_simulation(demand, targets, sim_cfg)

    # Scaling down has no startup delay -- takes effect the same tick.
    assert metrics.server_counts == [2, 2]


def test_scale_down_floors_at_one_server():
    sim_cfg = SimConfig(initial_servers=2, server_capacity=80.0, startup_delay_steps=1)
    demand = np.array([1.0])
    targets = np.array([0])   # asking for 0 servers should floor at 1

    metrics = _run_simulation(demand, targets, sim_cfg)
    assert metrics.server_counts == [1]


def test_sla_violation_recorded_when_demand_exceeds_capacity():
    sim_cfg = SimConfig(initial_servers=2, server_capacity=80.0, startup_delay_steps=1)
    demand = np.array([170.0])   # capacity is 2*80=160 < 170
    targets = np.array([2])

    metrics = _run_simulation(demand, targets, sim_cfg)
    assert metrics.sla_violations == 1
    assert metrics.under_prov_steps == 1
    assert metrics.over_prov_steps == 0


def test_over_provisioning_recorded_when_capacity_more_than_double_demand():
    sim_cfg = SimConfig(initial_servers=5, server_capacity=80.0, startup_delay_steps=1)
    demand = np.array([100.0])   # capacity 400 > 2*100=200
    targets = np.array([5])

    metrics = _run_simulation(demand, targets, sim_cfg)
    assert metrics.over_prov_steps == 1
    assert metrics.sla_violations == 0


def test_compute_cost_score_matches_weighted_formula():
    sim_cfg = SimConfig(initial_servers=2, server_capacity=80.0, startup_delay_steps=1)
    # 1 over-provisioned step, 1 under-provisioned (SLA-violating) step, 4 total
    demand = np.array([100.0, 170.0, 100.0, 100.0])
    targets = np.array([5, 2, 2, 2])

    metrics = _run_simulation(demand, targets, sim_cfg)
    cfg = DecisionConfig(over_prov_weight=1.0, under_prov_weight=20.0)
    cost = _compute_cost_score(metrics, cfg)

    expected = round(1.0 * (metrics.over_prov_steps / 4) + 20.0 * (metrics.under_prov_steps / 4), 6)
    assert cost == expected


def test_reactive_autoscaler_scales_up_above_threshold_and_down_below():
    sim_cfg = SimConfig(initial_servers=2)
    demand = np.array([900.0, 900.0, 10.0])

    targets = _reactive_autoscaler(demand, sim_cfg, scale_up_threshold=80, scale_down_threshold=30)

    # tick1: 900/2=450 > 80  -> active=3
    # tick2: 900/3=300 > 80  -> active=4
    # tick3: 10/4=2.5 < 30   -> active=3
    np.testing.assert_array_equal(targets, np.array([3, 4, 3]))


def test_reactive_autoscaler_clamps_between_one_and_ten():
    sim_cfg = SimConfig(initial_servers=10)
    demand = np.array([100_000.0])  # would want to scale up far past 10
    targets = _reactive_autoscaler(demand, sim_cfg, scale_up_threshold=80, scale_down_threshold=30)
    assert targets[0] == 10
