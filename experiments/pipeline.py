"""
Pipeline for the LSTM autoscaler experiment harness.

Split: 60 % train / 20 % validation / 20 % test (chronological).
- Scaler fits on train only.
- Validation is used exclusively for hyperparameter tuning (both policies).
- Test is touched once, at the end, for the final reported numbers.

Public API
----------
tune_on_validation(seed)          -> dict of best params for both policies
run_single_experiment(seed, ...)  -> dict of all metrics on the TEST split
"""

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# ── Frozen constants ──────────────────────────────────────────────────────────
DATA_PATH = os.path.expanduser("~/Desktop/machine_usage_bigger.csv")
DEMAND_SCALE = 20.0
LOOKBACK_STEPS = 6
HORIZON_STEPS = 3
FEATURE_COL = "cpu_util_percent"
VAL_RATIO = 0.2
TEST_RATIO = 0.2
NROWS = 500_000

LSTM_UNITS = 128
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
MAX_EPOCHS = 60
BATCH_SIZE = 64
VAL_SPLIT = 0.1   # Keras internal val split used for early-stopping only
ES_PATIENCE = 10

DEC_SERVER_CAPACITY = 80.0
DEC_MIN_SERVERS = 1
DEC_MAX_SERVERS = 10
DEC_OVER_WEIGHT = 1.0
DEC_UNDER_WEIGHT = 20.0   # original notebook value; may be overridden by tuning
DEC_SCALE_STEP = 1
SAFETY_MARGIN = 0.25      # original notebook value; may be overridden by tuning

SIM_INITIAL_SERVERS = 2
SIM_SERVER_CAPACITY = 80.0
SIM_STARTUP_DELAY = 1

# Original (untuned) Reactive thresholds; may be overridden by tuning
REACTIVE_UP_THRESHOLD = 80.0
REACTIVE_DOWN_THRESHOLD = 30.0

# Tuning search spaces
_REACTIVE_UP_GRID   = [60, 65, 70, 75, 80, 85, 90]
_REACTIVE_DOWN_GRID = [10, 15, 20, 25, 30, 35, 40]
_LSTM_UPW_GRID      = [5, 10, 15, 20, 25, 30, 40]
_LSTM_SM_GRID       = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class DecisionConfig:
    server_capacity_pct: float = DEC_SERVER_CAPACITY
    min_servers: int = DEC_MIN_SERVERS
    max_servers: int = DEC_MAX_SERVERS
    over_prov_weight: float = DEC_OVER_WEIGHT
    under_prov_weight: float = DEC_UNDER_WEIGHT
    scale_step: int = DEC_SCALE_STEP


@dataclass
class SimConfig:
    initial_servers: int = SIM_INITIAL_SERVERS
    server_capacity: float = SIM_SERVER_CAPACITY
    startup_delay_steps: int = SIM_STARTUP_DELAY


@dataclass
class SimMetrics:
    sla_violations: int = 0
    total_steps: int = 0
    over_prov_steps: int = 0
    under_prov_steps: int = 0
    server_counts: List[int] = field(default_factory=list)
    demand_trace: List[float] = field(default_factory=list)
    capacity_trace: List[float] = field(default_factory=list)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _pick_best_machine(df: pd.DataFrame) -> str:
    return df["machine_id"].value_counts().idxmax()


def _prepare_timeseries(df: pd.DataFrame, machine_id: str) -> pd.DataFrame:
    mdf = df[df["machine_id"] == machine_id].copy()
    mdf["time_stamp"] = pd.to_datetime(mdf["time_stamp"], unit="s")
    mdf = mdf.set_index("time_stamp").sort_index()
    mdf = mdf[["cpu_util_percent", "mem_util_percent"]]
    mdf = mdf.resample("5min").mean()
    mdf = mdf.ffill().dropna()
    return mdf


def _split_three_way(ts: pd.DataFrame, feature: str, val_ratio: float = VAL_RATIO,
                     test_ratio: float = TEST_RATIO):
    """Chronological 60/20/20 split. Scaler fits on train only."""
    values = ts[[feature]].values.astype(np.float32)
    n = len(values)
    train_end = int(n * (1.0 - val_ratio - test_ratio))
    val_end   = int(n * (1.0 - test_ratio))

    train_raw = values[:train_end]
    val_raw   = values[train_end:val_end]
    test_raw  = values[val_end:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_raw)
    return (
        scaler.transform(train_raw),
        scaler.transform(val_raw),
        scaler.transform(test_raw),
        scaler,
    )


def _make_sequences(data: np.ndarray, lookback: int, horizon: int):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback : i + lookback + horizon, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Model helpers ─────────────────────────────────────────────────────────────

def _build_lstm_model(lookback: int, horizon: int) -> tf.keras.Model:
    model = Sequential([
        LSTM(LSTM_UNITS, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS, return_sequences=False),
        Dropout(DROPOUT_RATE),
        Dense(horizon),
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")
    return model


def _train_lstm(lstm_model, X_train, y_train, model_path: str):
    early_stop = EarlyStopping(
        monitor="val_loss", patience=ES_PATIENCE, restore_best_weights=True, verbose=0
    )
    checkpoint = ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=0)
    t0 = time.time()
    history = lstm_model.fit(
        X_train, y_train,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        callbacks=[early_stop, checkpoint],
        verbose=0,
    )
    elapsed = time.time() - t0
    if os.path.exists(model_path):
        os.remove(model_path)
    return history, elapsed


def _inv(arr: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    out = np.zeros_like(arr)
    for h in range(arr.shape[1]):
        out[:, h] = scaler.inverse_transform(arr[:, h].reshape(-1, 1)).flatten()
    return out


def _evaluate_lstm(model, X, y, scaler):
    y_pred_scaled = model.predict(X, verbose=0)
    y_pred_real   = _inv(y_pred_scaled, scaler)
    y_real        = _inv(y, scaler)
    rmse = float(np.sqrt(mean_squared_error(y_real.flatten(), y_pred_real.flatten())))
    mae  = float(mean_absolute_error(y_real.flatten(), y_pred_real.flatten()))
    return y_pred_real, y_real, rmse, mae


def _evaluate_naive(X, y, scaler, horizon):
    last_val       = X[:, -1, 0]
    y_naive_scaled = np.repeat(last_val[:, None], horizon, axis=1)
    y_naive_real   = _inv(y_naive_scaled, scaler)
    y_real         = _inv(y, scaler)
    rmse = float(np.sqrt(mean_squared_error(y_real.flatten(), y_naive_real.flatten())))
    mae  = float(mean_absolute_error(y_real.flatten(), y_naive_real.flatten()))
    return rmse, mae


# ── Decision engine ───────────────────────────────────────────────────────────

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


# ── Simulation ────────────────────────────────────────────────────────────────

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


def _compute_cost_score(metrics: SimMetrics, cfg: DecisionConfig) -> float:
    n          = metrics.total_steps
    over_cost  = cfg.over_prov_weight  * (metrics.over_prov_steps  / n)
    under_cost = cfg.under_prov_weight * (metrics.under_prov_steps / n)
    return round(over_cost + under_cost, 6)


def _reactive_autoscaler(demand_series, sim_cfg,
                          scale_up_threshold:   float = REACTIVE_UP_THRESHOLD,
                          scale_down_threshold: float = REACTIVE_DOWN_THRESHOLD):
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


# ── Data loading (shared by tuning and experiment runs) ───────────────────────

def _load_and_prepare(machine_id=None, nrows=NROWS, df_raw=None):
    """Load data and prepare time-series for one machine.

    Accepts a pre-loaded df_raw to avoid redundant CSV reads when calling
    this for many machines in a loop.  Returns (ts, machine_id_used).
    """
    if df_raw is None:
        df_raw = pd.read_csv(
            DATA_PATH,
            nrows=nrows,
            usecols=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"],
        )
    if machine_id is None:
        machine_id = _pick_best_machine(df_raw)
    ts = _prepare_timeseries(df_raw, machine_id)
    return ts, machine_id


# ── Public API ────────────────────────────────────────────────────────────────

def tune_on_validation(seed: int = 42, machine_id=None, nrows=NROWS,
                       df_raw=None) -> dict:
    """Grid-search both policies on the validation split only.

    Returns a dict with the best params for Reactive and for the LSTM
    decision engine, plus their validation cost scores and which grid
    values were tried.  Never touches the test split.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    ts, machine_id = _load_and_prepare(machine_id, nrows, df_raw)
    train_data, val_data, _, scaler = _split_three_way(ts, FEATURE_COL)

    X_train, y_train = _make_sequences(train_data, LOOKBACK_STEPS, HORIZON_STEPS)
    X_val,   y_val   = _make_sequences(val_data,   LOOKBACK_STEPS, HORIZON_STEPS)

    # Train LSTM on train split (Keras val_split is last 10 % of TRAIN for early-stopping)
    lstm_model = _build_lstm_model(LOOKBACK_STEPS, HORIZON_STEPS)
    exp_dir    = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(exp_dir, f"_tmp_tune_seed_{seed}.keras")
    _train_lstm(lstm_model, X_train, y_train, model_path)

    # LSTM predictions on VAL (never touches test)
    y_pred_val, y_val_real, _, _ = _evaluate_lstm(lstm_model, X_val, y_val, scaler)

    # Demand on val for Reactive (uses actual, not predicted)
    val_demand = y_val_real[:, 0] * DEMAND_SCALE

    sim_cfg  = SimConfig()
    eval_cfg = DecisionConfig()   # fixed evaluation weights for Reactive search

    # ── Grid-search Reactive thresholds ──────────────────────────────────────
    best_reactive_cost = float("inf")
    best_up   = REACTIVE_UP_THRESHOLD
    best_down = REACTIVE_DOWN_THRESHOLD
    reactive_grid_results = []

    for up in _REACTIVE_UP_GRID:
        for down in _REACTIVE_DOWN_GRID:
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
    best_upw = DEC_UNDER_WEIGHT
    best_sm  = SAFETY_MARGIN
    lstm_grid_results = []

    for upw in _LSTM_UPW_GRID:
        for sm in _LSTM_SM_GRID:
            dec_cfg = DecisionConfig(under_prov_weight=upw)
            targets, _ = _build_lstm_targets(
                y_pred_val, y_val_real, dec_cfg, sim_cfg, DEMAND_SCALE, sm
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
    under_prov_weight: float = DEC_UNDER_WEIGHT,
    safety_margin:     float = SAFETY_MARGIN,
    reactive_up:       float = REACTIVE_UP_THRESHOLD,
    reactive_down:     float = REACTIVE_DOWN_THRESHOLD,
    machine_id=None,
    nrows=NROWS,
    df_raw=None,
) -> dict:
    """Run the full pipeline for one seed, evaluate on the TEST split only.

    All policy parameters must be provided from outside (typically from
    tune_on_validation) so the test split is never involved in tuning.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    ts, machine_id = _load_and_prepare(machine_id, nrows, df_raw)
    train_data, _, test_data, scaler = _split_three_way(ts, FEATURE_COL)

    X_train, y_train = _make_sequences(train_data, LOOKBACK_STEPS, HORIZON_STEPS)
    X_test,  y_test  = _make_sequences(test_data,  LOOKBACK_STEPS, HORIZON_STEPS)

    lstm_model = _build_lstm_model(LOOKBACK_STEPS, HORIZON_STEPS)
    exp_dir    = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(exp_dir, f"_tmp_model_seed_{seed}.keras")
    history, wall_clock_secs = _train_lstm(lstm_model, X_train, y_train, model_path)
    epochs_trained = len(history.history["loss"])

    # Evaluate forecast on TEST
    y_pred_real, y_test_real, lstm_rmse, lstm_mae = _evaluate_lstm(
        lstm_model, X_test, y_test, scaler
    )
    naive_rmse, naive_mae = _evaluate_naive(X_test, y_test, scaler, HORIZON_STEPS)

    # Simulations on TEST with tuned params
    dec_cfg = DecisionConfig(under_prov_weight=under_prov_weight)
    sim_cfg = SimConfig()

    lstm_targets, demand_series = _build_lstm_targets(
        y_pred_real, y_test_real, dec_cfg, sim_cfg, DEMAND_SCALE, safety_margin
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
