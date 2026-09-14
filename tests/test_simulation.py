"""Unit tests for the fleet simulator and the Reactive policy
(src/autoscaler/simulation.py).

No TensorFlow needed -- pure numpy/dataclass logic.
"""

import numpy as np

from src.autoscaler import config
from src.autoscaler.decision import DecisionConfig
from src.autoscaler.simulation import (
    SimConfig,
    _compute_cost_score,
    _reactive_autoscaler,
    _reactive_decide_once,
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


# ── _reactive_decide_once (live_loop.py's ARIMA-failure fallback) ──────────
# Same threshold rule as _reactive_autoscaler above, applied once against a
# real current reading/replica count instead of looping over a whole
# offline series from SimConfig.initial_servers.

def test_reactive_decide_once_scales_up_above_threshold():
    # current_servers=2, cpu=10% -> demand=10*20=200, load=200/2=100 > 80
    action, servers = _reactive_decide_once(
        current_cpu_pct=10.0, current_servers=2,
        demand_scale=20.0, scale_up_threshold=80.0, scale_down_threshold=30.0,
    )
    assert action == "scale_up +1"
    assert servers == 3


def test_reactive_decide_once_scales_down_below_threshold():
    # current_servers=4, cpu=2% -> demand=2*20=40, load=40/4=10 < 30
    action, servers = _reactive_decide_once(
        current_cpu_pct=2.0, current_servers=4,
        demand_scale=20.0, scale_up_threshold=80.0, scale_down_threshold=30.0,
    )
    assert action == "scale_down -1"
    assert servers == 3


def test_reactive_decide_once_holds_within_the_band():
    # current_servers=3, cpu=5% -> demand=5*20=100, load=100/3=33.3, between 30 and 80
    action, servers = _reactive_decide_once(
        current_cpu_pct=5.0, current_servers=3,
        demand_scale=20.0, scale_up_threshold=80.0, scale_down_threshold=30.0,
    )
    assert action == "hold"
    assert servers == 3


def test_reactive_decide_once_clamps_at_max_servers():
    action, servers = _reactive_decide_once(
        current_cpu_pct=100.0, current_servers=config.DEC_MAX_SERVERS,
        demand_scale=20.0, scale_up_threshold=80.0, scale_down_threshold=30.0,
        max_servers=config.DEC_MAX_SERVERS,
    )
    # Already at the ceiling -- clamped target equals current, so this is
    # a "hold", not a "scale_up" that silently does nothing.
    assert action == "hold"
    assert servers == config.DEC_MAX_SERVERS


def test_reactive_decide_once_clamps_at_min_servers():
    action, servers = _reactive_decide_once(
        current_cpu_pct=0.1, current_servers=config.DEC_MIN_SERVERS,
        demand_scale=20.0, scale_up_threshold=80.0, scale_down_threshold=30.0,
        min_servers=config.DEC_MIN_SERVERS,
    )
    assert action == "hold"
    assert servers == config.DEC_MIN_SERVERS


def test_reactive_decide_once_defaults_reuse_the_real_config_constants():
    # No thresholds/bounds passed explicitly -- must reuse
    # config.REACTIVE_UP_THRESHOLD/_DOWN_THRESHOLD and config.DEC_* rather
    # than a fresh set of magic numbers, per this function's own contract.
    action, servers = _reactive_decide_once(current_cpu_pct=10.0, current_servers=2)
    demand = 10.0 * config.DEMAND_SCALE
    expected_up = demand / 2 > config.REACTIVE_UP_THRESHOLD
    assert expected_up  # sanity check on the fixture itself
    assert action == f"scale_up +{config.DEC_SCALE_STEP}"
    assert servers == 2 + config.DEC_SCALE_STEP
