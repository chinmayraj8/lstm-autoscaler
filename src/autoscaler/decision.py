"""
The LSTM policy's decision engine: turns a forecast into a target server
count via a one-step-ahead greedy penalty minimization.

Logic copied unchanged from experiments/pipeline.py. See the project's own
docs (progress/, README) for the known limitation that this only uses
max() of the forecast horizon rather than its full shape.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from . import config


@dataclass
class DecisionConfig:
    server_capacity_pct: float = config.DEC_SERVER_CAPACITY
    min_servers: int = config.DEC_MIN_SERVERS
    max_servers: int = config.DEC_MAX_SERVERS
    over_prov_weight: float = config.DEC_OVER_WEIGHT
    under_prov_weight: float = config.DEC_UNDER_WEIGHT
    scale_step: int = config.DEC_SCALE_STEP


def _compute_penalty(n_servers: int, predicted_load: float, cfg: DecisionConfig) -> float:
    total_cap  = n_servers * cfg.server_capacity_pct
    over_frac  = max(0.0, total_cap - predicted_load) / 100.0
    under_frac = max(0.0, predicted_load - total_cap) / 100.0
    return cfg.over_prov_weight * over_frac + cfg.under_prov_weight * under_frac


def _decide_scaling(current_servers: int, predicted_load: float,
                    cfg: DecisionConfig) -> Tuple[str, int]:
    candidates = {
        "hold":       current_servers,
        "scale_up":   min(current_servers + cfg.scale_step, cfg.max_servers),
        "scale_down": max(current_servers - cfg.scale_step, cfg.min_servers),
    }
    best  = min(candidates, key=lambda a: _compute_penalty(candidates[a], predicted_load, cfg))
    new_n = candidates[best]
    if best == "scale_up":
        return f"scale_up +{new_n - current_servers}", new_n
    elif best == "scale_down":
        return f"scale_down -{current_servers - new_n}", new_n
    return "hold", new_n


def _build_lstm_targets(y_pred_real, y_actual_real, dec_cfg, sim_cfg,
                        demand_scale, safety_margin):
    targets = []
    current_servers = sim_cfg.initial_servers
    for i in range(len(y_pred_real)):
        planned_load = float(np.max(y_pred_real[i])) * demand_scale * (1.0 + safety_margin)
        _, new_target = _decide_scaling(current_servers, planned_load, dec_cfg)
        targets.append(new_target)
        current_servers = new_target
    demand = y_actual_real[:, 0] * demand_scale
    return np.array(targets), demand
