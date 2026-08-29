"""
Step 6: ARIMA baseline comparison and LSTM forecast sweep.

Tasks
-----
1. 3-way comparison (m_1933, seeds 42-46):
   Naive persistence vs ARIMA(2,0,1) vs LSTM on RMSE and MAE.
   → experiments/outputs/06_3way_comparison.png

2. LSTM forecast-horizon sweep (m_1933, seed=42):
   Horizons 1/3/6/12 steps (5/15/30/60 min ahead), fixed lookback=6 steps (30 min).
   → experiments/outputs/07_sweep_horizon_lookback.png

3. LSTM lookback-window sweep (m_1933, seed=42):
   Lookbacks 3/6/12/24 steps (15/30/60/120 min), fixed horizon=3 steps (15 min).
   → same chart as (2), second subplot
"""

import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.autoscaler import (
    DATA_PATH,
    FEATURE_COL,
    HORIZON_STEPS,
    LOOKBACK_STEPS,
    NROWS,
    _build_lstm_model,
    _evaluate_lstm,
    _evaluate_naive,
    _load_and_prepare,
    _make_sequences,
    _pick_best_machine,
    _split_three_way,
    _train_lstm,
    run_single_experiment,
)

import tensorflow as tf

warnings.filterwarnings("ignore")

SEEDS             = [42, 43, 44, 45, 46]
ARIMA_ORDER       = (2, 0, 1)

HORIZON_SWEEP     = [1, 3, 6, 12]   # steps → 5, 15, 30, 60 min
LOOKBACK_SWEEP    = [3, 6, 12, 24]  # steps → 15, 30, 60, 120 min

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


# ── ARIMA evaluation ──────────────────────────────────────────────────────────

def _evaluate_arima(train_scaled, val_scaled, test_scaled,
                    lookback, horizon, scaler, order=ARIMA_ORDER):
    """Rolling horizon-step-ahead ARIMA forecasts aligned with LSTM sequences.

    Fits on train only.  Updates state through val (no refit).  Rolls
    through test conditioning on actual observations at each step, which
    mirrors how the LSTM receives its lookback context.  Targets are the
    same time-steps as X_test / y_test from _make_sequences.
    """
    from statsmodels.tsa.arima.model import ARIMA

    train_flat = train_scaled.flatten()
    val_flat   = val_scaled.flatten()
    test_flat  = test_scaled.flatten()

    print(f"    Fitting ARIMA{order} on {len(train_flat)} train points ...", end="", flush=True)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(train_flat, order=order).fit()
    print(f" done ({time.time()-t0:.1f}s, AIC={fit.aic:.1f})")

    # Update state through val
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(val_flat) > 0:
            fit = fit.append(val_flat, refit=False)
        # Update through first 'lookback' test points (state init to match LSTM context)
        if lookback > 0 and len(test_flat) >= lookback:
            fit = fit.append(test_flat[:lookback], refit=False)

    n_sequences = len(test_flat) - lookback - horizon + 1
    if n_sequences <= 0:
        raise ValueError(f"Not enough test data for lookback={lookback} horizon={horizon}")

    y_pred_sc = []
    y_true_sc = []

    for i in range(n_sequences):
        fc = fit.forecast(steps=horizon)
        y_pred_sc.append(np.clip(fc, 0.0, 1.0))   # predictions are in [0,1] scaled space
        y_true_sc.append(test_flat[i + lookback : i + lookback + horizon])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = fit.append([test_flat[i + lookback]], refit=False)

    y_pred_arr = np.array(y_pred_sc)   # (n_sequences, horizon)
    y_true_arr = np.array(y_true_sc)

    y_pred_real = scaler.inverse_transform(
        y_pred_arr.reshape(-1, 1)).reshape(y_pred_arr.shape)
    y_true_real = scaler.inverse_transform(
        y_true_arr.reshape(-1, 1)).reshape(y_true_arr.shape)

    rmse = float(np.sqrt(mean_squared_error(y_true_real.flatten(), y_pred_real.flatten())))
    mae  = float(mean_absolute_error(y_true_real.flatten(), y_pred_real.flatten()))
    return rmse, mae


# ── LSTM sweep helper ────────────────────────────────────────────────────────

def _lstm_rmse_for_config(train_scaled, test_scaled, scaler,
                           lookback, horizon, seed=42):
    """Train and evaluate one LSTM with custom lookback/horizon. Returns (rmse, mae)."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_train, y_train = _make_sequences(train_scaled, lookback, horizon)
    X_test,  y_test  = _make_sequences(test_scaled,  lookback, horizon)
    if len(X_train) < 10 or len(X_test) < 3:
        return float("nan"), float("nan")

    model      = _build_lstm_model(lookback, horizon)
    exp_dir    = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(exp_dir, f"_tmp_sweep_lb{lookback}_h{horizon}.keras")
    _train_lstm(model, X_train, y_train, model_path)
    _, _, rmse, mae = _evaluate_lstm(model, X_test, y_test, scaler)
    return rmse, mae


# ── Task 1: 3-way comparison ─────────────────────────────────────────────────

def run_3way_comparison(machine_id, df_raw):
    print("\n" + "=" * 66)
    print("  TASK 1: 3-way comparison (Naive / ARIMA / LSTM)")
    print("=" * 66)

    ts, mid = _load_and_prepare(machine_id=machine_id, df_raw=df_raw)
    print(f"  Machine: {mid}, {len(ts)} resampled points")

    train_scaled, val_scaled, test_scaled, scaler = _split_three_way(ts, FEATURE_COL)
    X_test, y_test = _make_sequences(test_scaled, LOOKBACK_STEPS, HORIZON_STEPS)

    # Naive baseline
    naive_rmse, naive_mae = _evaluate_naive(X_test, y_test, scaler, HORIZON_STEPS)
    print(f"\n  Naive:            RMSE={naive_rmse:.4f}  MAE={naive_mae:.4f}")

    # ARIMA baseline
    arima_rmse, arima_mae = _evaluate_arima(
        train_scaled, val_scaled, test_scaled,
        LOOKBACK_STEPS, HORIZON_STEPS, scaler,
    )
    print(f"  ARIMA{ARIMA_ORDER}: RMSE={arima_rmse:.4f}  MAE={arima_mae:.4f}")

    # LSTM (5 seeds) — reuse run_single_experiment which handles seeds/splits correctly
    print(f"\n  LSTM (seeds {SEEDS[0]}–{SEEDS[-1]}):")
    lstm_rmse_list, lstm_mae_list = [], []
    for seed in SEEDS:
        result = run_single_experiment(seed, machine_id=mid, df_raw=df_raw)
        lstm_rmse_list.append(result["lstm_forecast_rmse"])
        lstm_mae_list.append(result["lstm_forecast_mae"])
        print(f"    seed={seed}: RMSE={result['lstm_forecast_rmse']:.4f}  "
              f"MAE={result['lstm_forecast_mae']:.4f}  "
              f"epochs={result['epochs_trained']}")

    lstm_rmse_mean = float(np.mean(lstm_rmse_list))
    lstm_rmse_std  = float(np.std(lstm_rmse_list))
    lstm_mae_mean  = float(np.mean(lstm_mae_list))
    lstm_mae_std   = float(np.std(lstm_mae_list))
    print(f"  LSTM mean:        RMSE={lstm_rmse_mean:.4f}±{lstm_rmse_std:.4f}  "
          f"MAE={lstm_mae_mean:.4f}±{lstm_mae_std:.4f}")

    return {
        "machine_id":      mid,
        "naive_rmse":      naive_rmse,
        "naive_mae":       naive_mae,
        "arima_rmse":      arima_rmse,
        "arima_mae":       arima_mae,
        "lstm_rmse_mean":  lstm_rmse_mean,
        "lstm_rmse_std":   lstm_rmse_std,
        "lstm_mae_mean":   lstm_mae_mean,
        "lstm_mae_std":    lstm_mae_std,
        "lstm_rmse_seeds": lstm_rmse_list,
        "lstm_mae_seeds":  lstm_mae_list,
    }


# ── Task 2: horizon sweep ────────────────────────────────────────────────────

def run_horizon_sweep(machine_id, df_raw):
    print("\n" + "=" * 66)
    print("  TASK 2: Horizon sweep (fixed lookback=6 steps / 30 min, seed=42)")
    print("=" * 66)

    ts, mid = _load_and_prepare(machine_id=machine_id, df_raw=df_raw)
    train_scaled, _, test_scaled, scaler = _split_three_way(ts, FEATURE_COL)

    rows = []
    for horizon in HORIZON_SWEEP:
        X_test, y_test = _make_sequences(test_scaled, LOOKBACK_STEPS, horizon)
        naive_rmse, naive_mae = _evaluate_naive(X_test, y_test, scaler, horizon)

        print(f"  horizon={horizon:2d} steps ({horizon*5:3d} min)  ", end="", flush=True)
        t0 = time.time()
        lstm_rmse, lstm_mae = _lstm_rmse_for_config(
            train_scaled, test_scaled, scaler,
            lookback=LOOKBACK_STEPS, horizon=horizon, seed=42,
        )
        print(f"LSTM RMSE={lstm_rmse:.4f}  MAE={lstm_mae:.4f} | "
              f"Naive RMSE={naive_rmse:.4f}  ({time.time()-t0:.0f}s)")

        rows.append({
            "horizon_steps": horizon,
            "horizon_min":   horizon * 5,
            "lstm_rmse":     lstm_rmse,
            "lstm_mae":      lstm_mae,
            "naive_rmse":    naive_rmse,
            "naive_mae":     naive_mae,
        })

    return pd.DataFrame(rows)


# ── Task 3: lookback sweep ───────────────────────────────────────────────────

def run_lookback_sweep(machine_id, df_raw):
    print("\n" + "=" * 66)
    print("  TASK 3: Lookback sweep (fixed horizon=3 steps / 15 min, seed=42)")
    print("=" * 66)

    ts, mid = _load_and_prepare(machine_id=machine_id, df_raw=df_raw)
    train_scaled, _, test_scaled, scaler = _split_three_way(ts, FEATURE_COL)

    rows = []
    for lookback in LOOKBACK_SWEEP:
        X_test, y_test = _make_sequences(test_scaled, lookback, HORIZON_STEPS)
        if len(X_test) < 3:
            print(f"  lookback={lookback:2d}: too few test sequences, skipping.")
            continue
        naive_rmse, naive_mae = _evaluate_naive(X_test, y_test, scaler, HORIZON_STEPS)

        print(f"  lookback={lookback:2d} steps ({lookback*5:3d} min)  ", end="", flush=True)
        t0 = time.time()
        lstm_rmse, lstm_mae = _lstm_rmse_for_config(
            train_scaled, test_scaled, scaler,
            lookback=lookback, horizon=HORIZON_STEPS, seed=42,
        )
        print(f"LSTM RMSE={lstm_rmse:.4f}  MAE={lstm_mae:.4f} | "
              f"Naive RMSE={naive_rmse:.4f}  ({time.time()-t0:.0f}s)")

        rows.append({
            "lookback_steps": lookback,
            "lookback_min":   lookback * 5,
            "lstm_rmse":      lstm_rmse,
            "lstm_mae":       lstm_mae,
            "naive_rmse":     naive_rmse,
            "naive_mae":      naive_mae,
        })

    return pd.DataFrame(rows)


# ── Charts ───────────────────────────────────────────────────────────────────

COLORS = {"naive": "#4C72B0", "arima": "#DD8452", "lstm": "#55A868"}


def plot_3way_comparison(comp, path):
    methods  = ["Naive\nPersistence", f"ARIMA{ARIMA_ORDER}", "LSTM\n(mean±std)"]
    rmse_val = [comp["naive_rmse"], comp["arima_rmse"], comp["lstm_rmse_mean"]]
    rmse_err = [0.0, 0.0, comp["lstm_rmse_std"]]
    mae_val  = [comp["naive_mae"],  comp["arima_mae"],  comp["lstm_mae_mean"]]
    mae_err  = [0.0, 0.0, comp["lstm_mae_std"]]

    x     = np.arange(3)
    cols  = [COLORS["naive"], COLORS["arima"], COLORS["lstm"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, vals, errs, metric in zip(
        axes,
        [rmse_val, mae_val],
        [rmse_err, mae_err],
        ["RMSE (CPU %)", "MAE (CPU %)"],
    ):
        bars = ax.bar(x, vals, 0.5, yerr=errs, capsize=6,
                      color=cols, alpha=0.85, error_kw={"linewidth": 1.5})
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(errs) * 0.1 + max(vals) * 0.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_ylabel(metric)
        ax.set_title(f"Forecast {metric.split()[0]}")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(vals) * 1.25 + max(errs) * 1.5)

    mid   = comp["machine_id"]
    seeds = f"seeds {SEEDS[0]}–{SEEDS[-1]}"
    fig.suptitle(
        f"3-Way Forecast Comparison — Machine {mid}\n"
        f"LSTM: {seeds}  |  horizon={HORIZON_STEPS} steps ({HORIZON_STEPS*5} min)  "
        f"|  lookback={LOOKBACK_STEPS} steps ({LOOKBACK_STEPS*5} min)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {path}")


def plot_sweep(h_df, lb_df, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Horizon sweep
    h_min = h_df["horizon_min"].tolist()
    ax1.plot(h_min, h_df["lstm_rmse"],  "o-",  color=COLORS["lstm"],  linewidth=2,
             markersize=7, label="LSTM")
    ax1.plot(h_min, h_df["naive_rmse"], "s--", color=COLORS["naive"], linewidth=1.5,
             markersize=6, label="Naive persistence")
    for x, y in zip(h_min, h_df["lstm_rmse"]):
        ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color=COLORS["lstm"])
    ax1.set_xlabel("Forecast Horizon (minutes)")
    ax1.set_ylabel("RMSE (CPU %)")
    ax1.set_title(f"LSTM RMSE vs Forecast Horizon\n"
                  f"(lookback={LOOKBACK_STEPS*5} min, seed=42)")
    ax1.set_xticks(h_min)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0)

    # Lookback sweep
    lb_min = lb_df["lookback_min"].tolist()
    ax2.plot(lb_min, lb_df["lstm_rmse"],  "o-",  color=COLORS["lstm"],  linewidth=2,
             markersize=7, label="LSTM")
    ax2.plot(lb_min, lb_df["naive_rmse"], "s--", color=COLORS["naive"], linewidth=1.5,
             markersize=6, label="Naive persistence")
    for x, y in zip(lb_min, lb_df["lstm_rmse"]):
        ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color=COLORS["lstm"])
    ax2.set_xlabel("Lookback Window (minutes)")
    ax2.set_ylabel("RMSE (CPU %)")
    ax2.set_title(f"LSTM RMSE vs Lookback Window\n"
                  f"(horizon={HORIZON_STEPS*5} min, seed=42)")
    ax2.set_xticks(lb_min)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0)

    fig.suptitle("LSTM Hyperparameter Sweep — Machine m_1933  (60/20/20 split)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Summary printer ──────────────────────────────────────────────────────────

def _print_summary(comp, h_df, lb_df):
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    lstm_vs_naive_rmse = comp["naive_rmse"] - comp["lstm_rmse_mean"]
    lstm_vs_arima_rmse = comp["arima_rmse"] - comp["lstm_rmse_mean"]
    print(f"\n  3-WAY COMPARISON  (m_1933, horizon={HORIZON_STEPS*5} min)")
    print(f"  {'Method':<22} {'RMSE':>8} {'MAE':>8}")
    print(f"  {'-'*42}")
    print(f"  {'Naive persistence':<22} {comp['naive_rmse']:>8.4f} {comp['naive_mae']:>8.4f}")
    print(f"  {'ARIMA'+str(ARIMA_ORDER):<22} {comp['arima_rmse']:>8.4f} {comp['arima_mae']:>8.4f}")
    print(f"  {'LSTM (mean±std)':<22} "
          f"{comp['lstm_rmse_mean']:>8.4f}±{comp['lstm_rmse_std']:.4f}  "
          f"{comp['lstm_mae_mean']:>8.4f}±{comp['lstm_mae_std']:.4f}")
    print(f"\n  LSTM vs Naive: RMSE gap = {lstm_vs_naive_rmse:+.4f} "
          f"({'LSTM better' if lstm_vs_naive_rmse > 0 else 'Naive better'})")
    print(f"  LSTM vs ARIMA: RMSE gap = {lstm_vs_arima_rmse:+.4f} "
          f"({'LSTM better' if lstm_vs_arima_rmse > 0 else 'ARIMA better'})")

    print(f"\n  HORIZON SWEEP  (lookback={LOOKBACK_STEPS*5} min, seed=42)")
    print(f"  {'Horizon':>10} {'LSTM RMSE':>12} {'Naive RMSE':>12} {'LSTM/Naive':>12}")
    print(f"  {'-'*50}")
    for _, r in h_df.iterrows():
        ratio = r["lstm_rmse"] / r["naive_rmse"] if r["naive_rmse"] > 0 else float("nan")
        print(f"  {int(r['horizon_min']):>8} min {r['lstm_rmse']:>12.4f} "
              f"{r['naive_rmse']:>12.4f} {ratio:>12.3f}")

    print(f"\n  LOOKBACK SWEEP  (horizon={HORIZON_STEPS*5} min, seed=42)")
    print(f"  {'Lookback':>11} {'LSTM RMSE':>12} {'Naive RMSE':>12} {'LSTM/Naive':>12}")
    print(f"  {'-'*50}")
    for _, r in lb_df.iterrows():
        ratio = r["lstm_rmse"] / r["naive_rmse"] if r["naive_rmse"] > 0 else float("nan")
        print(f"  {int(r['lookback_min']):>9} min {r['lstm_rmse']:>12.4f} "
              f"{r['naive_rmse']:>12.4f} {ratio:>12.3f}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {NROWS:,} rows ...")
    df_raw = pd.read_csv(
        DATA_PATH,
        nrows=NROWS,
        usecols=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"],
    )
    machine_id = _pick_best_machine(df_raw)
    print(f"Best machine: {machine_id}")

    comp  = run_3way_comparison(machine_id, df_raw)
    h_df  = run_horizon_sweep(machine_id, df_raw)
    lb_df = run_lookback_sweep(machine_id, df_raw)

    plot_3way_comparison(comp,  os.path.join(OUTPUT_DIR, "06_3way_comparison.png"))
    plot_sweep(h_df, lb_df,     os.path.join(OUTPUT_DIR, "07_sweep_horizon_lookback.png"))

    _print_summary(comp, h_df, lb_df)

    # Persist sweep tables for the progress log
    h_df.to_csv(os.path.join(OUTPUT_DIR, "horizon_sweep.csv"), index=False)
    lb_df.to_csv(os.path.join(OUTPUT_DIR, "lookback_sweep.csv"), index=False)


if __name__ == "__main__":
    main()
