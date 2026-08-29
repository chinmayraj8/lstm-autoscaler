"""Unit tests for per-machine demand-scale calibration and feasibility
(src/autoscaler/calibration.py).

Locks in the two numbers this project got wrong once already in an earlier
reconstruction (see progress/ and docs/): the calibration target is 115% of
ONE server's capacity, and feasibility is a p99 check, not p95.
"""

import numpy as np
import pandas as pd

from src.autoscaler.calibration import calibrate_demand_scale, check_feasibility


def test_calibrate_demand_scale_formula():
    # target_mean_load_pct defaults to 115.0
    assert calibrate_demand_scale(machine_mean_cpu=23.0) == 115.0 / 23.0


def test_calibrate_demand_scale_respects_custom_target():
    assert calibrate_demand_scale(machine_mean_cpu=50.0, target_mean_load_pct=200.0) == 4.0


def _make_ts(values):
    return pd.DataFrame({"cpu_util_percent": values})


def test_check_feasibility_uses_p99_not_p95():
    # 0..100 inclusive (101 points): np.percentile is exact here, no
    # interpolation ambiguity -- p95=95, p99=99.
    ts = _make_ts(list(range(0, 101)))

    result = check_feasibility(ts, demand_scale=10.0, max_servers=10, server_capacity=80.0)

    assert result["demand_p95"] == 950.0   # 95 * 10
    assert result["demand_p99"] == 990.0   # 99 * 10 -- must be p99, not p95
    assert result["demand_mean"] == 500.0  # mean(0..100)=50 * 10
    assert result["max_fleet_cap"] == 800.0  # 10 servers * 80% capacity


def test_check_feasibility_infeasible_when_p99_exceeds_max_capacity():
    ts = _make_ts(list(range(0, 101)))
    result = check_feasibility(ts, demand_scale=10.0, max_servers=10, server_capacity=80.0)
    assert result["infeasible"] is True
    assert "p99" in result["reason"]


def test_check_feasibility_feasible_when_p99_within_capacity():
    ts = _make_ts(list(range(0, 101)))
    result = check_feasibility(ts, demand_scale=1.0, max_servers=10, server_capacity=80.0)
    # p99=99 << max_fleet_cap=800
    assert result["infeasible"] is False
    assert result["reason"] == "OK"
