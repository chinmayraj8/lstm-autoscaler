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
                       df_raw=None, demand_scale: float = config.DEMAND_SCALE,
                       target_builder=_build_lstm_targets,
                       horizon_weights_grid=None) -> dict:
    """Grid-search both policies on the validation split only.

    `target_builder` selects the decision engine the LSTM's grid search is
    scored against -- defaults to the original greedy `_build_lstm_targets`
    (max-of-horizon). Pass `_build_multistep_targets` (decision.py, Step 12)
    to tune the multi-step-aware engine instead; same signature, same
    (targets, demand) return shape, so nothing else here needs to change.

    `horizon_weights_grid` (Step 13): optional list of `horizon_weights`
    arrays (or `None` for the target_builder's own default) to add as a
    third grid dimension alongside `under_prov_weight`/`safety_margin` --
    only meaningful for `_build_multistep_targets`, which is the only
    target_builder that accepts `horizon_weights`. Left as `None` (the
    default), behavior is byte-identical to before this parameter existed:
    a single implicit candidate (`None`, i.e. the target_builder's own
    default) is tried, and `lstm_grid` keeps its original 3-tuple
    `(under_prov_weight, safety_margin, cost)` shape. Passing 2+ candidates
    switches `lstm_grid` to 4-tuples `(under_prov_weight, safety_margin,
    horizon_weights, cost)` instead -- callers that only care about the
    default single-candidate behavior are unaffected either way.

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
    multi_hw = horizon_weights_grid is not None and len(horizon_weights_grid) > 1
    hw_candidates = horizon_weights_grid if horizon_weights_grid is not None else [None]

    best_lstm_cost = float("inf")
    best_upw = config.DEC_UNDER_WEIGHT
    best_sm  = config.SAFETY_MARGIN
    best_hw  = None
    lstm_grid_results = []

    for upw in config.LSTM_UPW_GRID:
        for sm in config.LSTM_SM_GRID:
            for hw in hw_candidates:
                dec_cfg = DecisionConfig(under_prov_weight=upw)
                hw_kwargs = {} if hw is None else {"horizon_weights": np.asarray(hw)}
                targets, _ = target_builder(
                    y_pred_val, y_val_real, dec_cfg, sim_cfg, demand_scale, sm, **hw_kwargs
                )
                metrics = _run_simulation(val_demand, targets, sim_cfg)
                cost    = _compute_cost_score(metrics, dec_cfg)
                lstm_grid_results.append((upw, sm, hw, cost) if multi_hw else (upw, sm, cost))
                if cost < best_lstm_cost:
                    best_lstm_cost = cost
                    best_upw, best_sm, best_hw = upw, sm, hw

    return {
        "machine_id":             machine_id,
        "reactive_up":            best_up,
        "reactive_down":          best_down,
        "reactive_val_cost":      round(best_reactive_cost, 6),
        "lstm_under_prov_weight": best_upw,
        "lstm_safety_margin":     best_sm,
        "lstm_horizon_weights":   best_hw,
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
    target_builder=_build_lstm_targets,
    horizon_weights=None,
) -> dict:
    """Run the full pipeline for one seed, evaluate on the TEST split only.

    All policy parameters must be provided from outside (typically from
    tune_on_validation) so the test split is never involved in tuning.
    `target_builder` must match whatever was used to produce
    `under_prov_weight`/`safety_margin` in tuning -- see tune_on_validation's
    docstring. `horizon_weights` (Step 13): the single winning candidate from
    `tune_on_validation`'s `horizon_weights_grid` (or `None`, unchanged
    behavior) -- only meaningful for `_build_multistep_targets`.
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

    hw_kwargs = {} if horizon_weights is None else {"horizon_weights": np.asarray(horizon_weights)}
    lstm_targets, demand_series = target_builder(
        y_pred_real, y_test_real, dec_cfg, sim_cfg, demand_scale, safety_margin, **hw_kwargs
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
