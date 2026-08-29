"""
The discrete-time fleet simulator (shared by both policies) and the
Reactive (threshold-based) policy itself.

Logic copied unchanged from experiments/pipeline.py.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from . import config


@dataclass
class SimConfig:
    initial_servers: int = config.SIM_INITIAL_SERVERS
    server_capacity: float = config.SIM_SERVER_CAPACITY
    startup_delay_steps: int = config.SIM_STARTUP_DELAY


@dataclass
class SimMetrics:
    sla_violations: int = 0
    total_steps: int = 0
    over_prov_steps: int = 0
    under_prov_steps: int = 0
    server_counts: List[int] = field(default_factory=list)
    demand_trace: List[float] = field(default_factory=list)
    capacity_trace: List[float] = field(default_factory=list)


def _run_simulation(demand_series: np.ndarray, target_series: np.ndarray,
                    sim_cfg: SimConfig) -> SimMetrics:
    metrics = SimMetrics()
    active  = sim_cfg.initial_servers
    pending: List[Tuple[int, int]] = []

    for demand, target in zip(demand_series, target_series):
        next_pending = []
        for n, ticks in pending:
            ticks -= 1
            if ticks <= 0:
                active += n
            else:
                next_pending.append((n, ticks))
        pending = next_pending

        needed = int(target)
        if needed > active:
            pending.append((needed - active, sim_cfg.startup_delay_steps))
        elif needed < active:
            active = max(1, needed)

        total_cap = active * sim_cfg.server_capacity
        if demand > total_cap:
            metrics.sla_violations   += 1
            metrics.under_prov_steps += 1
        if total_cap > demand * 2.0:
            metrics.over_prov_steps += 1

        metrics.total_steps += 1
        metrics.server_counts.append(active)
        metrics.demand_trace.append(demand)
        metrics.capacity_trace.append(total_cap)

    return metrics


def _compute_cost_score(metrics: SimMetrics, cfg) -> float:
    n          = metrics.total_steps
    over_cost  = cfg.over_prov_weight  * (metrics.over_prov_steps  / n)
    under_cost = cfg.under_prov_weight * (metrics.under_prov_steps / n)
    return round(over_cost + under_cost, 6)


def _reactive_autoscaler(demand_series, sim_cfg,
                          scale_up_threshold:   float = config.REACTIVE_UP_THRESHOLD,
                          scale_down_threshold: float = config.REACTIVE_DOWN_THRESHOLD):
    active  = sim_cfg.initial_servers
    targets = []
    for demand in demand_series:
        load_per_server = demand / active
        if load_per_server > scale_up_threshold:
            active = min(active + 1, 10)
        elif load_per_server < scale_down_threshold:
            active = max(active - 1, 1)
        targets.append(active)
    return np.array(targets)
