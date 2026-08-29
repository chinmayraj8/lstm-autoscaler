"""
The decision engine: turns a forecast into a target server count via a
greedy penalty minimization. Forecaster-agnostic -- takes whatever
(y_pred_real, y_actual_real) array pair a forecaster produces (LSTM,
ARIMA, ...) with no forecaster-specific logic in it. See Step 10's note
in arima_baseline.py: `_build_lstm_targets` already had no LSTM-specific
logic despite the name, which is exactly what let ARIMA plug into it
unchanged.

Two engines live here side by side (Step 12, Phase 3):

- `_decide_scaling` / `_compute_penalty` / `_build_lstm_targets` -- the
  ORIGINAL one-step-ahead greedy rule, unchanged since experiments/
  pipeline.py. It collapses the forecast horizon to a single number
  (`max(y_pred_real[i])`) before penalizing, so a one-tick spike and a
  sustained plateau of the same peak height are scored identically. This
  is a known, documented limitation (progress/00_INDEX.md, the audit doc)
  -- kept in place because existing tests and the API depend on it, not
  because it's believed to be the better engine.
- `_decide_scaling_multistep` / `_compute_penalty_multistep` /
  `_build_multistep_targets` -- NEW. Computes the per-step penalty at
  every step of the forecast horizon (not just the max) and averages them
  uniformly, so a brief spike (elevated at 1 of `horizon` steps) is scored
  at roughly 1/horizon of a sustained plateau's (elevated at every step)
  penalty, instead of identically. See progress/2026-08-29_step12-*.md for
  why this was tried and what it changed.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

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


# ── Multi-step-aware engine (Step 12) ────────────────────────────────────────

def _compute_penalty_multistep(n_servers: int, predicted_loads: np.ndarray,
                               cfg: DecisionConfig,
                               horizon_weights: Optional[np.ndarray] = None) -> float:
    """Same per-step penalty formula as `_compute_penalty`, applied at every
    step of `predicted_loads` (shape (horizon,)) and combined with
    `horizon_weights` (uniform mean by default) instead of first collapsing
    the horizon to its max. A one-tick spike therefore contributes roughly
    `1/horizon` of a sustained plateau's penalty, rather than the identical
    penalty `_compute_penalty(n_servers, max(predicted_loads), cfg)` would
    assign to both.
    """
    horizon = len(predicted_loads)
    if horizon_weights is None:
        horizon_weights = np.full(horizon, 1.0 / horizon)
    per_step = np.array([_compute_penalty(n_servers, load, cfg) for load in predicted_loads])
    return float(np.dot(horizon_weights, per_step))


def _decide_scaling_multistep(current_servers: int, predicted_loads: np.ndarray,
                              cfg: DecisionConfig,
                              horizon_weights: Optional[np.ndarray] = None) -> Tuple[str, int]:
    """Same hold/scale_up/scale_down candidate search as `_decide_scaling`,
    scored with `_compute_penalty_multistep` instead of `_compute_penalty`."""
    candidates = {
        "hold":       current_servers,
        "scale_up":   min(current_servers + cfg.scale_step, cfg.max_servers),
        "scale_down": max(current_servers - cfg.scale_step, cfg.min_servers),
    }
    best = min(candidates,
               key=lambda a: _compute_penalty_multistep(candidates[a], predicted_loads, cfg, horizon_weights))
    new_n = candidates[best]
    if best == "scale_up":
        return f"scale_up +{new_n - current_servers}", new_n
    elif best == "scale_down":
        return f"scale_down -{current_servers - new_n}", new_n
    return "hold", new_n


def _build_multistep_targets(y_pred_real, y_actual_real, dec_cfg, sim_cfg,
                             demand_scale, safety_margin,
                             horizon_weights: Optional[np.ndarray] = None):
    """Drop-in replacement for `_build_lstm_targets` using the full forecast
    horizon instead of its max. Identical signature and return shape
    (targets, demand) -- swap this in for `_build_lstm_targets` wherever a
    `target_builder` is accepted (experiment.py, arima_baseline.py) and
    nothing else about the calling code needs to change. Forecaster-agnostic
    exactly like `_build_lstm_targets`: `y_pred_real` is whatever shape
    (n_sequences, horizon) array the forecaster produced, LSTM or ARIMA.
    """
    targets = []
    current_servers = sim_cfg.initial_servers
    for i in range(len(y_pred_real)):
        planned_loads = np.asarray(y_pred_real[i], dtype=float) * demand_scale * (1.0 + safety_margin)
        _, new_target = _decide_scaling_multistep(current_servers, planned_loads, dec_cfg, horizon_weights)
        targets.append(new_target)
        current_servers = new_target
    demand = y_actual_real[:, 0] * demand_scale
    return np.array(targets), demand
