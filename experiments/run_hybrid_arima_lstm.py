"""
Step 16: combine ARIMA and the LSTM instead of pitting them head-to-head.

Steps 8-15 treated ARIMA and the LSTM as competitors under an identical
decision engine. Step 14 diagnosed the LSTM's actual weakness as too little
per-machine data (~1,300-2,300 points) to exploit extra signal without
overfitting -- a hybrid gives the LSTM a smaller, more tractable problem
(ARIMA's residuals, which should contain only whatever nonlinear structure
a linear ARMA model misses) instead of the full raw series.

Two approaches, both under the SAME unmodified decision engine
(`_build_lstm_targets`, the original greedy max-of-horizon engine -- same
one Step 8/11/15 used, so results are directly comparable to
`results_v7_arima_full13.csv`'s ARIMA/LSTM/Reactive numbers, reused
unchanged) and the SAME architecture/lookback/horizon as every prior LSTM
in this project:

1. RESIDUAL HYBRID: fit ARIMA(2,0,1) on train (order held at the Step 6
   default, not Step 14's per-machine AIC/BIC choice -- Step 14's own
   audit found that re-selection changes only 2/13 head-to-head verdicts,
   both via ARIMA overfitting its order to train likelihood with no
   held-out check, and one of those two reverses under BIC; that isn't a
   reliable enough foundation to build a new experiment on, so the frozen,
   already-thoroughly-audited (2,0,1) order is used here instead).
   Compute ARIMA's rolling multi-step forecast and residuals across
   train/val/test. Train the LSTM -- identical architecture, same
   `_make_sequences`-built input window -- to predict those residuals
   instead of raw `cpu_util_percent`. Final forecast = ARIMA forecast +
   LSTM residual forecast, in real units. See arima_baseline.py's new
   `_arima_train_walkforward` (train-residual labels, walk-forward from
   the deployed model's own fixed parameters -- no re-estimation leakage,
   no fit-mismatch with the deployed ARIMA) and `_inv_residual` (correct
   affine-residual inverse-scaling, NOT the same operation as a raw
   value's inverse-transform).

2. BLEND BASELINE: a per-machine, validation-tuned weighted average of the
   STANDALONE ARIMA and STANDALONE LSTM forecasts: `w * LSTM + (1-w) *
   ARIMA`, `w` grid-searched jointly with `under_prov_weight`/
   `safety_margin` on validation (11 candidates, 0.0 to 1.0 in steps of
   0.1) -- a much simpler combination than the residual hybrid, kept as a
   comparison point per the task.

Both need the STANDALONE LSTM's own forecast (for the blend) in addition
to the residual-predicting LSTM (for the hybrid) -- two distinct LSTM
models are trained per (machine, seed), since they have different training
targets. ARIMA is fit once per machine (deterministic, re-run and
asserted bit-identical, same discipline as every prior ARIMA step).

Same tuning discipline as every other step: grid-searched on validation
only, test touched once, 5 seeds for anything LSTM-stochastic. Run on all
13 feasible machines. Writes experiments/results_v14_hybrid.csv (130 rows)
and experiments/results_v14_hybrid_summary.csv (13 rows: standalone
ARIMA/LSTM/Reactive from results_v7, reused unchanged, vs. residual-hybrid
and blend, both variants).
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.autoscaler import config
from src.autoscaler.arima_baseline import (
    ARIMA_ORDER,
    _arima_rolling_forecast,
    _arima_train_walkforward,
    _inv_flat,
    _inv_residual,
)
from src.autoscaler.data import _load_and_prepare, _make_sequences, _split_three_way
from src.autoscaler.decision import DecisionConfig, _build_lstm_targets
from src.autoscaler.forecasting import _build_lstm_model, _evaluate_lstm, _train_lstm
from src.autoscaler.simulation import SimConfig, _compute_cost_score, _run_simulation

SEEDS = [42, 43, 44, 45, 46]
W_GRID = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_V7 = os.path.join(EXP_DIR, "results_v7_arima_full13.csv")
RESULTS_V14 = os.path.join(EXP_DIR, "results_v14_hybrid.csv")
SUMMARY_V14 = os.path.join(EXP_DIR, "results_v14_hybrid_summary.csv")


def _fit_arima(train_flat, order=ARIMA_ORDER):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(train_flat, order=order).fit()


def _arima_forecasts_for_machine(mid, df_raw):
    """Everything ARIMA-related for one machine: train/val/test splits,
    the deployed fit, its rolling forecasts (real units) on train/val/test,
    and a determinism check (re-run val+test once, compare)."""
    ts, _ = _load_and_prepare(mid, config.NROWS, df_raw)
    train_data, val_data, test_data, scaler = _split_three_way(ts, config.FEATURE_COL)
    train_flat, val_flat, test_flat = train_data.flatten(), val_data.flatten(), test_data.flatten()

    fit_full = _fit_arima(train_flat)
    y_pred_train_sc, y_true_train_sc = _arima_train_walkforward(
        fit_full, train_flat, config.LOOKBACK_STEPS, config.HORIZON_STEPS)
    y_pred_val_sc, y_true_val_sc = _arima_rolling_forecast(
        train_flat, np.array([]), val_flat, config.LOOKBACK_STEPS, config.HORIZON_STEPS)
    y_pred_test_sc, y_true_test_sc = _arima_rolling_forecast(
        train_flat, val_flat, test_flat, config.LOOKBACK_STEPS, config.HORIZON_STEPS)

    # determinism check (same discipline as Steps 10-15)
    y_pred_test_sc_2, _ = _arima_rolling_forecast(
        train_flat, val_flat, test_flat, config.LOOKBACK_STEPS, config.HORIZON_STEPS)
    assert np.array_equal(y_pred_test_sc, y_pred_test_sc_2), f"{mid}: ARIMA test forecast not deterministic"

    return {
        "train_data": train_data, "val_data": val_data, "test_data": test_data, "scaler": scaler,
        "train_pred_sc": y_pred_train_sc, "train_true_sc": y_true_train_sc,
        "val_pred_sc": y_pred_val_sc, "val_true_sc": y_true_val_sc,
        "val_pred_real": _inv_flat(y_pred_val_sc, scaler), "val_true_real": _inv_flat(y_true_val_sc, scaler),
        "test_pred_sc": y_pred_test_sc, "test_true_sc": y_true_test_sc,
        "test_pred_real": _inv_flat(y_pred_test_sc, scaler), "test_true_real": _inv_flat(y_true_test_sc, scaler),
    }


def _train_standard_lstm(train_data, seed, tag):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    X_train, y_train = _make_sequences(train_data, config.LOOKBACK_STEPS, config.HORIZON_STEPS)
    model = _build_lstm_model(config.LOOKBACK_STEPS, config.HORIZON_STEPS)
    model_path = os.path.join(EXP_DIR, f"_tmp_{tag}_seed_{seed}.keras")
    _train_lstm(model, X_train, y_train, model_path)
    return model


def _train_residual_lstm(train_data, resid_train_sc, seed, tag):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    X_train, _ = _make_sequences(train_data, config.LOOKBACK_STEPS, config.HORIZON_STEPS)
    assert len(X_train) == len(resid_train_sc), "X/residual length mismatch"
    model = _build_lstm_model(config.LOOKBACK_STEPS, config.HORIZON_STEPS)
    model_path = os.path.join(EXP_DIR, f"_tmp_{tag}_seed_{seed}.keras")
    _train_lstm(model, X_train, resid_train_sc.astype(np.float32), model_path)
    return model


def _predict_real(model, data_scaled, scaler):
    X, _ = _make_sequences(data_scaled, config.LOOKBACK_STEPS, config.HORIZON_STEPS)
    y_pred_sc = model.predict(X, verbose=0)
    return _inv_flat(y_pred_sc, scaler)


def _predict_residual_real(model, data_scaled, scaler):
    X, _ = _make_sequences(data_scaled, config.LOOKBACK_STEPS, config.HORIZON_STEPS)
    resid_pred_sc = model.predict(X, verbose=0)
    return _inv_residual(resid_pred_sc, scaler)


def _tune_and_eval_forecast(y_pred_val_real, y_val_true_real, y_pred_test_real, y_test_true_real, demand_scale):
    """Grid-search (upw, sm) on val for an already-combined (real-units)
    forecast, evaluate the winner once on test. Forecaster-agnostic --
    `_build_lstm_targets` only ever consumes a (y_pred_real, y_actual_real)
    pair, whatever produced it (hybrid or blend, here)."""
    val_demand = y_val_true_real[:, 0] * demand_scale
    sim_cfg = SimConfig()
    best_cost, best_upw, best_sm = float("inf"), config.DEC_UNDER_WEIGHT, config.SAFETY_MARGIN
    for upw in config.LSTM_UPW_GRID:
        for sm in config.LSTM_SM_GRID:
            dec_cfg = DecisionConfig(under_prov_weight=upw)
            targets, _ = _build_lstm_targets(y_pred_val_real, y_val_true_real, dec_cfg, sim_cfg, demand_scale, sm)
            metrics = _run_simulation(val_demand, targets, sim_cfg)
            cost = _compute_cost_score(metrics, dec_cfg)
            if cost < best_cost:
                best_cost, best_upw, best_sm = cost, upw, sm

    test_demand = y_test_true_real[:, 0] * demand_scale
    dec_cfg = DecisionConfig(under_prov_weight=best_upw)
    targets, _ = _build_lstm_targets(y_pred_test_real, y_test_true_real, dec_cfg, sim_cfg, demand_scale, best_sm)
    metrics = _run_simulation(test_demand, targets, sim_cfg)
    n = metrics.total_steps
    return {
        "under_prov_weight": best_upw, "safety_margin": best_sm, "val_cost": round(best_cost, 6),
        "cost_score": _compute_cost_score(metrics, dec_cfg),
        "sla_violation_rate_pct": round(metrics.sla_violations / n * 100, 4),
    }


def _tune_and_eval_blend(lstm_val_real, arima_val_real, y_val_true_real,
                         lstm_test_real, arima_test_real, y_test_true_real, demand_scale):
    """Same as _tune_and_eval_forecast but with `w` as a third grid dimension:
    blended = w*LSTM + (1-w)*ARIMA, in real units."""
    val_demand = y_val_true_real[:, 0] * demand_scale
    sim_cfg = SimConfig()
    best_cost, best_upw, best_sm, best_w = float("inf"), config.DEC_UNDER_WEIGHT, config.SAFETY_MARGIN, 0.5
    for w in W_GRID:
        blended_val = w * lstm_val_real + (1.0 - w) * arima_val_real
        for upw in config.LSTM_UPW_GRID:
            for sm in config.LSTM_SM_GRID:
                dec_cfg = DecisionConfig(under_prov_weight=upw)
                targets, _ = _build_lstm_targets(blended_val, y_val_true_real, dec_cfg, sim_cfg, demand_scale, sm)
                metrics = _run_simulation(val_demand, targets, sim_cfg)
                cost = _compute_cost_score(metrics, dec_cfg)
                if cost < best_cost:
                    best_cost, best_upw, best_sm, best_w = cost, upw, sm, w

    blended_test = best_w * lstm_test_real + (1.0 - best_w) * arima_test_real
    test_demand = y_test_true_real[:, 0] * demand_scale
    dec_cfg = DecisionConfig(under_prov_weight=best_upw)
    targets, _ = _build_lstm_targets(blended_test, y_test_true_real, dec_cfg, sim_cfg, demand_scale, best_sm)
    metrics = _run_simulation(test_demand, targets, sim_cfg)
    n = metrics.total_steps
    return {
        "under_prov_weight": best_upw, "safety_margin": best_sm, "w": best_w, "val_cost": round(best_cost, 6),
        "cost_score": _compute_cost_score(metrics, dec_cfg),
        "sla_violation_rate_pct": round(metrics.sla_violations / n * 100, 4),
    }


def main() -> None:
    v7 = pd.read_csv(RESULTS_V7)
    machines = sorted(v7["machine_id"].unique())
    demand_scales = v7.set_index("machine_id")["demand_scale"].to_dict()
    print(f"{len(machines)} feasible machines: {machines}")

    print("\nLoading 5,000,000 rows ...")
    df_raw = pd.read_csv(
        config.DATA_PATH, nrows=5_000_000,
        usecols=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"],
    )

    out_rows = []

    for mid in machines:
        ds = demand_scales[mid]
        print(f"\n{'='*78}\n  {mid}  (demand_scale={ds:.4f}x)\n{'='*78}")
        arima = _arima_forecasts_for_machine(mid, df_raw)
        resid_train_sc = arima["train_true_sc"] - arima["train_pred_sc"]
        print(f"  ARIMA fit OK. train residual std={resid_train_sc.std():.4f}  "
              f"val residual std={(arima['val_true_sc']-arima['val_pred_sc']).std():.4f}")

        for seed in SEEDS:
            # ── Standalone LSTM (for the blend) ──────────────────────────────
            std_model = _train_standard_lstm(arima["train_data"], seed, f"std_{mid}")
            lstm_val_real = _predict_real(std_model, arima["val_data"], arima["scaler"])
            lstm_test_real = _predict_real(std_model, arima["test_data"], arima["scaler"])

            # ── Residual LSTM (for the hybrid) ───────────────────────────────
            resid_model = _train_residual_lstm(arima["train_data"], resid_train_sc, seed, f"resid_{mid}")
            resid_val_real = _predict_residual_real(resid_model, arima["val_data"], arima["scaler"])
            resid_test_real = _predict_residual_real(resid_model, arima["test_data"], arima["scaler"])
            hybrid_val_real = arima["val_pred_real"] + resid_val_real
            hybrid_test_real = arima["test_pred_real"] + resid_test_real

            r_hybrid = _tune_and_eval_forecast(
                hybrid_val_real, arima["val_true_real"], hybrid_test_real, arima["test_true_real"], ds)
            r_blend = _tune_and_eval_blend(
                lstm_val_real, arima["val_pred_real"], arima["val_true_real"],
                lstm_test_real, arima["test_pred_real"], arima["test_true_real"], ds)

            out_rows.append({"machine_id": mid, "seed": seed, "variant": "residual_hybrid", **r_hybrid})
            out_rows.append({"machine_id": mid, "seed": seed, "variant": "blend", **r_blend})
            print(f"    seed={seed}: hybrid cost={r_hybrid['cost_score']:.4f} SLA={r_hybrid['sla_violation_rate_pct']:.2f}%   "
                  f"blend(w={r_blend['w']}) cost={r_blend['cost_score']:.4f} SLA={r_blend['sla_violation_rate_pct']:.2f}%")

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(RESULTS_V14, index=False)
    print(f"\nSaved: {RESULTS_V14}  ({len(out_df)} rows)")

    # ── Summary vs Step 11 baseline ──────────────────────────────────────────
    summary = []
    for mid in machines:
        row = v7[v7["machine_id"] == mid]
        lstm_cost, lstm_std = row["lstm_cost_score"].mean(), row["lstm_cost_score"].std()
        arima_cost = row["arima_cost_score"].iloc[0]
        reactive_cost = row["reactive_cost_score"].iloc[0]

        h_rows = out_df[(out_df.machine_id == mid) & (out_df.variant == "residual_hybrid")]
        b_rows = out_df[(out_df.machine_id == mid) & (out_df.variant == "blend")]
        h_mean, h_std = h_rows["cost_score"].mean(), h_rows["cost_score"].std()
        b_mean, b_std = b_rows["cost_score"].mean(), b_rows["cost_score"].std()
        w_modal = b_rows["w"].mode().iloc[0]

        def h2h(cost, std):
            gap = arima_cost - cost
            if gap > std: return "confirmed cheaper than ARIMA"
            elif gap > 0: return "directional (unconfirmed)"
            elif gap == 0: return "tied (exact)"
            elif gap > -std: return "ARIMA directional cheaper"
            else: return "ARIMA confirmed cheaper"

        summary.append({
            "machine_id": mid,
            "lstm_baseline_cost": lstm_cost, "lstm_baseline_std": lstm_std,
            "arima_cost": arima_cost, "reactive_cost": reactive_cost,
            "hybrid_cost": h_mean, "hybrid_std": h_std, "hybrid_vs_arima": h2h(h_mean, h_std),
            "blend_cost": b_mean, "blend_std": b_std, "blend_w_modal": w_modal, "blend_vs_arima": h2h(b_mean, b_std),
        })

    sm_df = pd.DataFrame(summary)
    sm_df.to_csv(SUMMARY_V14, index=False)
    print(f"Saved: {SUMMARY_V14}")

    print("\n" + "=" * 120)
    print("  HYBRID/BLEND vs standalone ARIMA (Step 11 numbers) and standalone LSTM (Step 11 numbers)")
    print("=" * 120)
    for _, r in sm_df.sort_values("machine_id").iterrows():
        print(f"  {r['machine_id']:<10} LSTM={r['lstm_baseline_cost']:.4f}±{r['lstm_baseline_std']:.4f}  "
              f"ARIMA={r['arima_cost']:.4f}  Reactive={r['reactive_cost']:.4f}   "
              f"hybrid={r['hybrid_cost']:.4f}±{r['hybrid_std']:.4f} [{r['hybrid_vs_arima']}]   "
              f"blend(w~{r['blend_w_modal']})={r['blend_cost']:.4f}±{r['blend_std']:.4f} [{r['blend_vs_arima']}]")
    print("\n  hybrid_vs_arima counts:", sm_df["hybrid_vs_arima"].value_counts().to_dict())
    print("  blend_vs_arima counts:", sm_df["blend_vs_arima"].value_counts().to_dict())
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
