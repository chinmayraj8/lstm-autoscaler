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


def verdict(cost_a: float, cost_b: float, std_a: float, std_b: float) -> str:
    """Is `a` confirmed cheaper than `b`? The combined-+/-1-sigma rule this
    project has used since Step 11 (run_multimachine_v2._verdict,
    generalized in run_arima_full13._verdict) to classify a cost comparison
    -- promoted here (Step 18) into the package proper so shadow.py (and
    any future caller) reuses the exact same statistical bar instead of a
    fresh reimplementation. `run_arima_full13.py`'s own `_verdict` is left
    as-is (not worth touching already-validated experiment scripts), but
    is byte-for-byte the same formula.

    `std_a`/`std_b` are 0.0 for a deterministic quantity (e.g. ARIMA, or a
    single shadow-window's raw cost with no repeats yet) -- callers pass
    0.0 explicitly rather than None to keep the comparison's basis
    unambiguous in logs/tests.
    """
    gap = cost_b - cost_a   # positive => a is cheaper
    combined = (std_a or 0.0) + (std_b or 0.0)
    if gap > combined:
        return "confirmed cheaper"
    elif gap > 0:
        return "directional (within ±1σ)"
    elif abs(gap) <= combined:
        return "tied (within ±1σ)"
    else:
        return "confirmed more expensive"


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


def _reactive_decide_once(current_cpu_pct: float, current_servers: int,
                          demand_scale: float = config.DEMAND_SCALE,
                          scale_up_threshold: float = config.REACTIVE_UP_THRESHOLD,
                          scale_down_threshold: float = config.REACTIVE_DOWN_THRESHOLD,
                          min_servers: int = config.DEC_MIN_SERVERS,
                          max_servers: int = config.DEC_MAX_SERVERS,
                          scale_step: int = config.DEC_SCALE_STEP) -> Tuple[str, int]:
    """One-tick equivalent of `_reactive_autoscaler`'s per-step rule
    (live_loop.py's ARIMA-failure fallback). `_reactive_autoscaler` above
    is built for offline replay over a whole series and always starts
    from `SimConfig.initial_servers` -- not "wherever the real fleet
    actually is right now". This takes a single current real CPU reading
    plus the current real replica count directly, applies the IDENTICAL
    `demand = current_cpu_pct * demand_scale` / `load_per_server = demand
    / current_servers` threshold rule against the SAME
    `config.REACTIVE_UP_THRESHOLD`/`REACTIVE_DOWN_THRESHOLD` constants
    (no new thresholds), and returns `(action, recommended_servers)` in
    the exact string convention `decision._decide_scaling` uses ("hold" |
    "scale_up +N" | "scale_down -N"), so a caller/log line/the frontend
    treats this identically to an ARIMA-produced decision.

    `min_servers`/`max_servers`/`scale_step` default to the live decision
    engine's own bounds (`config.DEC_*`), not `_reactive_autoscaler`'s
    hardcoded 1/10 above -- both currently equal the same values, but
    this keeps the fallback honestly bounded by the same real system
    limits the primary policy uses, not a second, coincidentally-matching
    set of magic numbers.
    """
    demand = current_cpu_pct * demand_scale
    load_per_server = demand / current_servers
    if load_per_server > scale_up_threshold:
        new_servers = min(current_servers + scale_step, max_servers)
    elif load_per_server < scale_down_threshold:
        new_servers = max(current_servers - scale_step, min_servers)
    else:
        new_servers = current_servers

    if new_servers > current_servers:
        return f"scale_up +{new_servers - current_servers}", new_servers
    if new_servers < current_servers:
        return f"scale_down -{current_servers - new_servers}", new_servers
    return "hold", current_servers
