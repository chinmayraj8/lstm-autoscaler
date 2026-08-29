"""
Per-machine demand-scale calibration and feasibility checking (Step 5+).

Logic copied unchanged from experiments/pipeline.py.
"""

import numpy as np
import pandas as pd

from . import config


def calibrate_demand_scale(machine_mean_cpu: float,
                           target_mean_load_pct: float = config.TARGET_MEAN_LOAD_PCT) -> float:
    """Return a demand scale so mean aggregate demand == target_mean_load_pct.

    Uses the full resampled series mean — this is a normalisation constant,
    not a tunable hyperparameter.
    """
    return target_mean_load_pct / machine_mean_cpu


def check_feasibility(ts: pd.DataFrame, demand_scale: float,
                      max_servers: int = config.DEC_MAX_SERVERS,
                      server_capacity: float = config.SIM_SERVER_CAPACITY) -> dict:
    """Return feasibility stats for a machine at the given demand_scale.

    Infeasible when p99 calibrated demand > max fleet capacity.  In that
    regime both policies fail structurally (capacity ceiling), not because of
    forecasting quality, so the machine should be skipped or re-scaled.
    """
    cpu     = ts[config.FEATURE_COL].values * demand_scale
    max_cap = max_servers * server_capacity
    p99     = float(np.percentile(cpu, 99))
    return {
        "demand_mean":   round(float(cpu.mean()), 2),
        "demand_p95":    round(float(np.percentile(cpu, 95)), 2),
        "demand_p99":    round(p99, 2),
        "max_fleet_cap": max_cap,
        "infeasible":    p99 > max_cap,
        "reason":        (f"p99 {p99:.1f}% > max fleet {max_cap:.0f}%"
                          if p99 > max_cap else "OK"),
    }
