"""
Public orchestration API: tune both policies on validation, then run one
seed's full evaluation on the held-out test split.

Split: 60% train / 20% validation / 20% test (chronological).
- Scaler fits on train only.
- Validation is used exclusively for hyperparameter tuning (both policies).
- Test is touched once, at the end, for the final reported numbers.

Logic copied unchanged from experiments/pipeline.py.
"""

import os

import numpy as np
import tensorflow as tf

from . import config
from .calibration import calibrate_demand_scale  # noqa: F401  (re-exported for convenience)
from .data import _load_and_prepare, _make_sequences, _split_three_way
from .decision import DecisionConfig, _build_lstm_targets
from .forecasting import _build_lstm_model, _evaluate_lstm, _evaluate_naive, _train_lstm
from .simulation import SimConfig, _compute_cost_score, _reactive_autoscaler, _run_simulation


def tune_on_validation(seed: int = 42, machine_id=None, nrows=config.NROWS,
                       df_raw=None, demand_scale: float = config.DEMAND_SCALE) -> dict:
    """Grid-search both policies on the validation split only.

    Returns a dict with the best params for Reactive and for the LSTM
    decision engine, plus their validation cost scores and which grid
    values were tried.  Never touches the test split.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    ts, machine_id = _load_and_prepare(machine_id, nrows, df_raw)
    train_data, val_data, _, scaler = _split_three_way(ts, config.FEATURE_COL)

    X_train, y_train = _make_sequences(train_data, config.LOOKBACK_STEPS, config.HORIZON_STEPS)
    X_val,   y_val   = _make_sequences(val_data,   config.LOOKBACK_STEPS, config.HORIZON_STEPS)

    # Train LSTM on train split (Keras val_split is last 10 % of TRAIN for early-stopping)
    lstm_model = _build_lstm_model(config.LOOKBACK_STEPS, config.HORIZON_STEPS)
    exp_dir    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(exp_dir, "experiments", f"_tmp_tune_seed_{seed}.keras")
    _train_lstm(lstm_model, X_train, y_train, model_path)

    # LSTM predictions on VAL (never touches test)
    y_pred_val, y_val_real, _, _ = _evaluate_lstm(lstm_model, X_val, y_val, scaler)

    # Demand on val for Reactive (uses actual, not predicted)
    val_demand = y_val_real[:, 0] * demand_scale

    sim_cfg  = SimConfig()
    eval_cfg = DecisionConfig()   # fixed evaluation weights for Reactive search

    # ── Grid-search Reactive thresholds ──────────────────────────────────────
    best_reactive_cost = float("inf")
    best_up   = config.REACTIVE_UP_THRESHOLD
    best_down = config.REACTIVE_DOWN_THRESHOLD
    reactive_grid_results = []

    for up in config.REACTIVE_UP_GRID:
        for down in config.REACTIVE_DOWN_GRID:
            if down >= up:
                continue
            targets = _reactive_autoscaler(val_demand, sim_cfg, up, down)
            metrics = _run_simulation(val_demand, targets, sim_cfg)
            cost    = _compute_cost_score(metrics, eval_cfg)
            reactive_grid_results.append((up, down, cost))
            if cost < best_reactive_cost:
                best_reactive_cost = cost
                best_up, best_down = up, down

    # ── Grid-search LSTM decision-engine params ───────────────────────────────
    best_lstm_cost = float("inf")
    best_upw = config.DEC_UNDER_WEIGHT
    best_sm  = config.SAFETY_MARGIN
    lstm_grid_results = []

    for upw in config.LSTM_UPW_GRID:
        for sm in config.LSTM_SM_GRID:
            dec_cfg = DecisionConfig(under_prov_weight=upw)
            targets, _ = _build_lstm_targets(
                y_pred_val, y_val_real, dec_cfg, sim_cfg, demand_scale, sm
            )
            metrics = _run_simulation(val_demand, targets, sim_cfg)
            cost    = _compute_cost_score(metrics, dec_cfg)
            lstm_grid_results.append((upw, sm, cost))
            if cost < best_lstm_cost:
                best_lstm_cost = cost
                best_upw, best_sm = upw, sm

    return {
        "machine_id":             machine_id,
        "reactive_up":            best_up,
        "reactive_down":          best_down,
        "reactive_val_cost":      round(best_reactive_cost, 6),
        "lstm_under_prov_weight": best_upw,
        "lstm_safety_margin":     best_sm,
        "lstm_val_cost":          round(best_lstm_cost, 6),
        "reactive_grid":          reactive_grid_results,
        "lstm_grid":              lstm_grid_results,
    }


def run_single_experiment(
    seed: int,
    under_prov_weight: float = config.DEC_UNDER_WEIGHT,
    safety_margin:     float = config.SAFETY_MARGIN,
    reactive_up:       float = config.REACTIVE_UP_THRESHOLD,
    reactive_down:     float = config.REACTIVE_DOWN_THRESHOLD,
    machine_id=None,
    nrows=config.NROWS,
    df_raw=None,
    demand_scale: float = config.DEMAND_SCALE,
) -> dict:
    """Run the full pipeline for one seed, evaluate on the TEST split only.

    All policy parameters must be provided from outside (typically from
    tune_on_validation) so the test split is never involved in tuning.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    ts, machine_id = _load_and_prepare(machine_id, nrows, df_raw)
    train_data, _, test_data, scaler = _split_three_way(ts, config.FEATURE_COL)

    X_train, y_train = _make_sequences(train_data, config.LOOKBACK_STEPS, config.HORIZON_STEPS)
    X_test,  y_test  = _make_sequences(test_data,  config.LOOKBACK_STEPS, config.HORIZON_STEPS)

    lstm_model = _build_lstm_model(config.LOOKBACK_STEPS, config.HORIZON_STEPS)
    exp_dir    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(exp_dir, "experiments", f"_tmp_model_seed_{seed}.keras")
    history, wall_clock_secs = _train_lstm(lstm_model, X_train, y_train, model_path)
    epochs_trained = len(history.history["loss"])

    # Evaluate forecast on TEST
    y_pred_real, y_test_real, lstm_rmse, lstm_mae = _evaluate_lstm(
        lstm_model, X_test, y_test, scaler
    )
    naive_rmse, naive_mae = _evaluate_naive(X_test, y_test, scaler, config.HORIZON_STEPS)

    # Simulations on TEST with tuned params
    dec_cfg = DecisionConfig(under_prov_weight=under_prov_weight)
    sim_cfg = SimConfig()

    lstm_targets, demand_series = _build_lstm_targets(
        y_pred_real, y_test_real, dec_cfg, sim_cfg, demand_scale, safety_margin
    )
    lstm_metrics = _run_simulation(demand_series, lstm_targets, sim_cfg)

    reactive_targets = _reactive_autoscaler(demand_series, sim_cfg, reactive_up, reactive_down)
    reactive_metrics = _run_simulation(demand_series, reactive_targets, sim_cfg)

    lstm_cost     = _compute_cost_score(lstm_metrics,     dec_cfg)
    reactive_cost = _compute_cost_score(reactive_metrics, dec_cfg)

    n = lstm_metrics.total_steps
    return {
        "seed":                           seed,
        "lstm_forecast_rmse":             round(lstm_rmse,  6),
        "lstm_forecast_mae":              round(lstm_mae,   6),
        "naive_baseline_rmse":            round(naive_rmse, 6),
        "lstm_sla_violation_rate_pct":    round(lstm_metrics.sla_violations   / n * 100, 4),
        "reactive_sla_violation_rate_pct":round(reactive_metrics.sla_violations/ n * 100, 4),
        "lstm_over_prov_waste_pct":       round(lstm_metrics.over_prov_steps   / n * 100, 4),
        "reactive_over_prov_waste_pct":   round(reactive_metrics.over_prov_steps/ n * 100, 4),
        "lstm_cost_score":                lstm_cost,
        "reactive_cost_score":            reactive_cost,
        "epochs_trained":                 epochs_trained,
        "wall_clock_secs":                round(wall_clock_secs, 1),
    }
