"""
Precomputes per-step replay traces for the live-monitor dashboard.

Why precompute instead of training inside the Streamlit app
-------------------------------------------------------------
Training an LSTM on every page load/rerun would make the dashboard
unusably slow (Streamlit reruns the whole script on most interactions).
Instead, this script runs the exact same pipeline pieces used by
`experiments/pipeline.py` (same split, same model, same decision engine
and simulation loop) ONCE per machine, using tuned parameters already
established in Steps 3/5/8, and writes a per-timestep CSV that the
dashboard just reads and animates. No TensorFlow import is needed at
dashboard-serve time.

Run from the project root:
    venv/bin/python -m src.dashboard.prepare_replay_data
"""

import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from src.autoscaler import (  # noqa: E402
    DATA_PATH,
    DEMAND_SCALE,
    FEATURE_COL,
    HORIZON_STEPS,
    LOOKBACK_STEPS,
    TEST_RATIO,
    DecisionConfig,
    SimConfig,
    _build_lstm_model,
    _build_lstm_targets,
    _load_and_prepare,
    _make_sequences,
    _reactive_autoscaler,
    _run_simulation,
    _split_three_way,
    _train_lstm,
    calibrate_demand_scale,
)

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(_DATA_DIR, exist_ok=True)

# ── Curated machine set, spanning the different regimes found in the project ──
# Tuned params come from previously-completed grid searches (Step 3 for
# m_1933, Step 5/8 for the rest) — no re-tuning here, just the final
# test-split run that produces per-step traces instead of aggregates.
MACHINES = {
    "m_1933": dict(
        label="m_1933 — original baseline (Step 3): Reactive wins outright",
        nrows=500_000,
        demand_scale=DEMAND_SCALE,          # 20.0, uncalibrated (Steps 1-3 default)
        reactive_up=60, reactive_down=40,
        lstm_upw=5, lstm_sm=0.10,
        cluster_note="not clustered (pre-Step-4 default machine)",
    ),
    "m_2189": dict(
        label="m_2189 — cluster 3: LSTM cost advantage, Reactive wins SLA",
        nrows=5_000_000,
        demand_scale=None,                  # calibrated at runtime
        reactive_up=75, reactive_down=30,
        lstm_upw=5, lstm_sm=0.25,
        cluster_note="cluster 3, burstiness=14.49",
    ),
    "m_2065": dict(
        label="m_2065 — cluster 0: LSTM cost advantage, Reactive wins SLA",
        nrows=5_000_000,
        demand_scale=None,
        reactive_up=65, reactive_down=30,
        lstm_upw=5, lstm_sm=0.40,
        cluster_note="cluster 0, burstiness=16.33",
    ),
    "m_2085": dict(
        label="m_2085 — cluster 2: LSTM wins BOTH metrics (Step 8 standout)",
        nrows=5_000_000,
        demand_scale=None,
        reactive_up=90, reactive_down=10,
        lstm_upw=5, lstm_sm=0.30,
        cluster_note="cluster 2, burstiness=1.50",
    ),
    "m_2134": dict(
        label="m_2134 — stress test: Reactive wins both, decisively",
        nrows=5_000_000,
        demand_scale=None,
        reactive_up=75, reactive_down=20,
        lstm_upw=5, lstm_sm=0.35,
        cluster_note="stress-test pick, burstiness=38.20",
    ),
}

SEED = 42


def _action_labels(server_counts: np.ndarray) -> list:
    """Turns a server-count trace into per-step 'hold'/'scale_up +N'/'scale_down -N' labels."""
    actions = ["hold"]  # first step has no prior state to compare against
    for prev, cur in zip(server_counts[:-1], server_counts[1:]):
        if cur > prev:
            actions.append(f"scale_up +{cur - prev}")
        elif cur < prev:
            actions.append(f"scale_down -{prev - cur}")
        else:
            actions.append("hold")
    return actions


def build_replay_trace(machine_id: str, cfg: dict, seed: int = SEED) -> tuple:
    """Runs one train+test pass and returns (per-step DataFrame, meta dict)."""
    import tensorflow as tf

    np.random.seed(seed)
    tf.random.set_seed(seed)

    print(f"\n=== {machine_id} ===")
    df_raw = pd.read_csv(
        DATA_PATH,
        nrows=cfg["nrows"],
        usecols=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"],
    )
    ts, machine_id = _load_and_prepare(machine_id, cfg["nrows"], df_raw)
    print(f"  timeseries length: {len(ts)} (5-min steps)")

    demand_scale = cfg["demand_scale"]
    if demand_scale is None:
        demand_scale = calibrate_demand_scale(float(ts[FEATURE_COL].mean()))
    print(f"  demand_scale = {demand_scale:.4f}x")

    train_data, _, test_data, scaler = _split_three_way(ts, FEATURE_COL)
    X_train, y_train = _make_sequences(train_data, LOOKBACK_STEPS, HORIZON_STEPS)
    X_test, y_test = _make_sequences(test_data, LOOKBACK_STEPS, HORIZON_STEPS)

    model = _build_lstm_model(LOOKBACK_STEPS, HORIZON_STEPS)
    model_path = os.path.join(_DATA_DIR, f"_tmp_{machine_id}.keras")
    history, wall_clock_secs = _train_lstm(model, X_train, y_train, model_path)
    epochs_trained = len(history.history["loss"])
    print(f"  trained {epochs_trained} epochs in {wall_clock_secs:.1f}s")

    y_pred_scaled = model.predict(X_test, verbose=0)

    def _inv(arr):
        out = np.zeros_like(arr)
        for h in range(arr.shape[1]):
            out[:, h] = scaler.inverse_transform(arr[:, h].reshape(-1, 1)).flatten()
        return out

    y_pred_real = _inv(y_pred_scaled)
    y_test_real = _inv(y_test)

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    lstm_rmse = float(np.sqrt(mean_squared_error(y_test_real.flatten(), y_pred_real.flatten())))
    lstm_mae = float(mean_absolute_error(y_test_real.flatten(), y_pred_real.flatten()))

    dec_cfg = DecisionConfig(under_prov_weight=cfg["lstm_upw"])
    sim_cfg = SimConfig()

    lstm_targets, demand_series = _build_lstm_targets(
        y_pred_real, y_test_real, dec_cfg, sim_cfg, demand_scale, cfg["lstm_sm"]
    )
    lstm_metrics = _run_simulation(demand_series, lstm_targets, sim_cfg)

    reactive_targets = _reactive_autoscaler(
        demand_series, sim_cfg, cfg["reactive_up"], cfg["reactive_down"]
    )
    reactive_metrics = _run_simulation(demand_series, reactive_targets, sim_cfg)

    # ── Recover real timestamps for the test-split steps ──────────────────────
    n_total = len(ts)
    val_end = int(n_total * (1.0 - TEST_RATIO))
    test_index = ts.index[val_end:]
    n_steps = len(demand_series)
    step_timestamps = test_index[LOOKBACK_STEPS: LOOKBACK_STEPS + n_steps]

    # The trace's raw time_stamp is seconds-since-experiment-start, not a real
    # epoch (hence the 1970 dates) — shift the whole window so the last step
    # lands at "now", making the replay read as recent activity rather than a
    # dataset artifact. Relative 5-min spacing between steps is unaffected.
    now = pd.Timestamp.now().replace(microsecond=0)
    step_timestamps = step_timestamps + (now - step_timestamps[-1])

    forecast_trace = y_pred_real[:, 0] * demand_scale  # 1-step-ahead forecast, same units as demand

    reactive_capacity = np.array(reactive_metrics.capacity_trace)
    lstm_capacity = np.array(lstm_metrics.capacity_trace)
    reactive_servers = np.array(reactive_metrics.server_counts)
    lstm_servers = np.array(lstm_metrics.server_counts)

    reactive_sla_violation = (demand_series > reactive_capacity).astype(int)
    lstm_sla_violation = (demand_series > lstm_capacity).astype(int)
    reactive_over_prov = (reactive_capacity > demand_series * 2.0).astype(int)
    lstm_over_prov = (lstm_capacity > demand_series * 2.0).astype(int)

    steps_1idx = np.arange(1, n_steps + 1)
    reactive_cum_sla_pct = np.cumsum(reactive_sla_violation) / steps_1idx * 100
    lstm_cum_sla_pct = np.cumsum(lstm_sla_violation) / steps_1idx * 100

    over_w, under_w = dec_cfg.over_prov_weight, dec_cfg.under_prov_weight
    reactive_cum_cost = (
        over_w * np.cumsum(reactive_over_prov) / steps_1idx
        + under_w * np.cumsum(reactive_sla_violation) / steps_1idx
    )
    lstm_cum_cost = (
        over_w * np.cumsum(lstm_over_prov) / steps_1idx
        + under_w * np.cumsum(lstm_sla_violation) / steps_1idx
    )

    trace = pd.DataFrame({
        "step": np.arange(n_steps),
        "timestamp": step_timestamps,
        "actual_demand": demand_series,
        "lstm_forecast": forecast_trace,
        "reactive_servers": reactive_servers,
        "lstm_servers": lstm_servers,
        "reactive_capacity": reactive_capacity,
        "lstm_capacity": lstm_capacity,
        "reactive_sla_violation": reactive_sla_violation,
        "lstm_sla_violation": lstm_sla_violation,
        "reactive_over_prov": reactive_over_prov,
        "lstm_over_prov": lstm_over_prov,
        "reactive_action": _action_labels(reactive_servers),
        "lstm_action": _action_labels(lstm_servers),
        "reactive_cum_sla_pct": reactive_cum_sla_pct,
        "lstm_cum_sla_pct": lstm_cum_sla_pct,
        "reactive_cum_cost": reactive_cum_cost,
        "lstm_cum_cost": lstm_cum_cost,
    })

    meta = {
        "machine_id": machine_id,
        "label": cfg["label"],
        "cluster_note": cfg["cluster_note"],
        "demand_scale": round(float(demand_scale), 4),
        "reactive_up": cfg["reactive_up"],
        "reactive_down": cfg["reactive_down"],
        "lstm_upw": cfg["lstm_upw"],
        "lstm_sm": cfg["lstm_sm"],
        "lstm_forecast_rmse": round(lstm_rmse, 4),
        "lstm_forecast_mae": round(lstm_mae, 4),
        "epochs_trained": epochs_trained,
        "wall_clock_secs": round(wall_clock_secs, 1),
        "n_steps": int(n_steps),
        "step_minutes": 5,
        "final_reactive_sla_pct": round(float(reactive_cum_sla_pct[-1]), 4),
        "final_lstm_sla_pct": round(float(lstm_cum_sla_pct[-1]), 4),
        "final_reactive_cost": round(float(reactive_cum_cost[-1]), 4),
        "final_lstm_cost": round(float(lstm_cum_cost[-1]), 4),
    }

    if os.path.exists(model_path):
        os.remove(model_path)

    return trace, meta


def main() -> None:
    all_meta = {}
    for machine_id, cfg in MACHINES.items():
        trace, meta = build_replay_trace(machine_id, cfg)
        out_path = os.path.join(_DATA_DIR, f"replay_{machine_id}.csv")
        trace.to_csv(out_path, index=False)
        all_meta[machine_id] = meta
        print(f"  -> wrote {out_path} ({len(trace)} steps)")
        print(f"  final: LSTM SLA={meta['final_lstm_sla_pct']}%  "
              f"React SLA={meta['final_reactive_sla_pct']}%  "
              f"LSTM cost={meta['final_lstm_cost']}  "
              f"React cost={meta['final_reactive_cost']}")

    meta_path = os.path.join(_DATA_DIR, "machines_meta.json")
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"\nWrote metadata for {len(all_meta)} machines -> {meta_path}")


if __name__ == "__main__":
    main()
