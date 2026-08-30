"""
Step 15: does pooling training data across similar machines beat training
one LSTM from scratch per machine?

Every LSTM in this project so far has been trained on a single machine's
~1,300-2,300 resampled points (Step 14's multivariate-input experiment
suggested this may just not be enough data for a 128-unit 2-layer LSTM to
exploit extra signal without overfitting). This step tests the other lever
for the same underlying problem: instead of adding features, add DATA, by
pooling several similar machines' training series into one shared model.

Architecture, LOOKBACK_STEPS/HORIZON_STEPS, and the decision engine are held
completely unchanged from Step 8/11's original setup (`_build_lstm_targets`,
the greedy max-of-horizon engine, LSTM_UPW_GRID x LSTM_SM_GRID) -- this is a
data-quantity experiment, not another architecture/engine change, so
Step 11's numbers (results_v7_arima_full13.csv: LSTM, ARIMA(2,0,1), Reactive,
all under the original engine) are reused unchanged as the baseline to beat.

Pool selection
--------------
Reproducing run_multimachine_v2.py's exact K-means (k=4, standardized
mean/std/burstiness, random_state=42, n_init=10) on the full 733-machine
qualifying candidate pool (not just the 13 sampled representatives) gives
cluster sizes of 353, 329, 35, and 16 machines. The 353-machine cluster is
both the single largest full cluster AND has the most already-feasible
sampled representatives among Step 8's 13 (5, vs 4 each for the other two
non-infeasible clusters): m_2183, m_2104, m_2065, m_2355, m_2134. That's the
pool used here.

Calendar-overlap check (the leakage risk this step was explicitly asked to
check for)
-----------------------------------------------------------------------------
All 5 pool machines' resampled series span the identical ~8-day window
(1970-01-01 -> 1970-01-08, epoch-relative timestamps -- this is the
Alibaba-cluster-trace-style fixed observation window, not per-machine
calendar coverage) and have nearly identical lengths (2281-2304 points), so
each machine's own chronological 60/20/20 split lands at nearly the same
ABSOLUTE timestamp regardless of which machine. Checked explicitly: the
LATEST train-split end across all 5 pool machines is 1970-01-05 19:50; the
EARLIEST test-split start across all 5 is 1970-01-07 09:35 -- a ~1.5-day
gap with zero pairwise overlap. Pooling only each machine's own TRAIN split
(never touching any pool machine's val/test) is therefore safe in this
dataset: no pooled machine's training data reaches into any machine's
(including its own) test-period calendar time. This is a property of this
specific trace (a fixed shared observation window with near-identical
per-machine coverage) and would NOT hold automatically in a dataset where
different machines have different, possibly-overlapping-but-offset date
ranges -- worth re-checking if this approach is ever reused elsewhere.

Sequence construction: concatenated vs. interleaved
-----------------------------------------------------
Chosen: build (X, y) sliding-window sequences INDEPENDENTLY per pool
machine from that machine's own scaled train array (a window never spans
two machines), then merge the resulting per-machine sequence arrays into
one pooled training set (shuffled once, seeded -- order doesn't matter once
each row is an independent supervised example). This is the "interleaved"
option in the sense that no single sequence is contaminated by a
machine-boundary seam. The alternative -- concatenating the raw
(unwindowed) per-machine series end-to-end before sliding the window over
the whole thing -- would create a handful of nonsense windows whose
lookback spans machine A's tail and machine B's head; avoiding that entirely
by windowing first, merging second, is strictly simpler than concatenating
and then dropping seam windows, so that's what's implemented.

Scaling: ONE shared MinMaxScaler, fit on the pooled TRAIN values only (the
union of all 5 pool machines' own train splits), used to transform every
pool machine's val/test AND every one of the 13 evaluation machines'
train/val/test (in cluster or not) -- because the pooled model's weights
are only meaningful in the scaled space they were trained in. For
out-of-cluster evaluation machines this means their true values may fall
outside [0, 1] under this scaler (no clipping applied, same convention as
every other input path in this project) -- an expected, not a bug,
consequence of evaluating a shared model on a machine it may have very
different scale/burstiness from.

Two variants evaluated per machine (5 seeds each -- a full pretrain-and/or
-finetune cycle per seed, not just re-scoring the same weights):
  (a) pooled, no fine-tuning: the pretrained pooled model applied directly
      to the target machine's own val (for decision-param tuning) and test
      (final score) splits.
  (b) pooled + fine-tuned: a copy of the pretrained pooled model, warm-started
      from its weights, continues training (`_train_lstm`, same
      MAX_EPOCHS/ES_PATIENCE, unchanged) on the target machine's OWN train
      split only, then evaluated the same way.

Writes experiments/results_v13_pooled_lstm.csv (130 rows: 13 machines x 2
variants x 5 seeds) and experiments/results_v13_pooled_summary.csv (13 rows:
both variants' mean+-std vs. Step 11's per-machine LSTM/ARIMA/Reactive).
"""

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.autoscaler import config
from src.autoscaler.data import _load_and_prepare, _make_sequences
from src.autoscaler.decision import DecisionConfig, _build_lstm_targets
from src.autoscaler.forecasting import _build_lstm_model, _evaluate_lstm, _train_lstm
from src.autoscaler.simulation import SimConfig, _compute_cost_score, _run_simulation

SEEDS = [42, 43, 44, 45, 46]
POOL_MACHINES = ["m_2183", "m_2104", "m_2065", "m_2355", "m_2134"]

RESULTS_V7 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "results_v7_arima_full13.csv")
RESULTS_V13 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_v13_pooled_lstm.csv")
SUMMARY_V13 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_v13_pooled_summary.csv")

EXP_DIR = os.path.dirname(os.path.abspath(__file__))


def _machine_raw_splits(mid, df_raw):
    """Chronological 60/20/20 split of a machine's own RAW (unscaled) values."""
    ts, _ = _load_and_prepare(mid, config.NROWS, df_raw)
    values = ts[[config.FEATURE_COL]].values.astype(np.float32)
    n = len(values)
    train_end = int(n * (1.0 - config.VAL_RATIO - config.TEST_RATIO))
    val_end   = int(n * (1.0 - config.TEST_RATIO))
    return values[:train_end], values[train_end:val_end], values[val_end:]


def _fit_pool_scaler(pool_train_raws):
    pooled = np.concatenate(pool_train_raws, axis=0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(pooled)
    return scaler


def _build_pooled_sequences(pool_train_raws, scaler, seed):
    X_list, y_list = [], []
    for train_raw in pool_train_raws:
        train_scaled = scaler.transform(train_raw)
        X, y = _make_sequences(train_scaled, config.LOOKBACK_STEPS, config.HORIZON_STEPS)
        X_list.append(X)
        y_list.append(y)
    X_pool = np.concatenate(X_list, axis=0)
    y_pool = np.concatenate(y_list, axis=0)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X_pool))
    return X_pool[perm], y_pool[perm]


def _tune_and_eval(model, scaler, val_raw, test_raw, demand_scale):
    """Grid-search (upw, sm) on val using `model`'s forecast (frozen, no
    training here), then evaluate the winning params on test once. Same
    logic as experiment.py's tune_on_validation/run_single_experiment, just
    reusable against an externally-supplied (already trained) model instead
    of training fresh inside this function."""
    X_val, y_val = _make_sequences(scaler.transform(val_raw), config.LOOKBACK_STEPS, config.HORIZON_STEPS)
    X_test, y_test = _make_sequences(scaler.transform(test_raw), config.LOOKBACK_STEPS, config.HORIZON_STEPS)

    y_pred_val, y_val_real, _, _ = _evaluate_lstm(model, X_val, y_val, scaler)
    val_demand = y_val_real[:, 0] * demand_scale
    sim_cfg = SimConfig()

    best_cost = float("inf")
    best_upw, best_sm = config.DEC_UNDER_WEIGHT, config.SAFETY_MARGIN
    for upw in config.LSTM_UPW_GRID:
        for sm in config.LSTM_SM_GRID:
            dec_cfg = DecisionConfig(under_prov_weight=upw)
            targets, _ = _build_lstm_targets(y_pred_val, y_val_real, dec_cfg, sim_cfg, demand_scale, sm)
            metrics = _run_simulation(val_demand, targets, sim_cfg)
            cost = _compute_cost_score(metrics, dec_cfg)
            if cost < best_cost:
                best_cost = cost
                best_upw, best_sm = upw, sm

    y_pred_test, y_test_real, rmse, mae = _evaluate_lstm(model, X_test, y_test, scaler)
    demand_series = y_test_real[:, 0] * demand_scale
    dec_cfg = DecisionConfig(under_prov_weight=best_upw)
    targets, _ = _build_lstm_targets(y_pred_test, y_test_real, dec_cfg, sim_cfg, demand_scale, best_sm)
    metrics = _run_simulation(demand_series, targets, sim_cfg)
    n = metrics.total_steps
    return {
        "under_prov_weight": best_upw, "safety_margin": best_sm,
        "val_cost": round(best_cost, 6),
        "cost_score": _compute_cost_score(metrics, dec_cfg),
        "sla_violation_rate_pct": round(metrics.sla_violations / n * 100, 4),
        "over_prov_waste_pct": round(metrics.over_prov_steps / n * 100, 4),
        "forecast_rmse": round(rmse, 6), "forecast_mae": round(mae, 6),
    }


def main() -> None:
    v7 = pd.read_csv(RESULTS_V7)
    machines = sorted(v7["machine_id"].unique())
    demand_scales = v7.set_index("machine_id")["demand_scale"].to_dict()
    print(f"{len(machines)} feasible machines: {machines}")
    print(f"Pool (largest K-means cluster, {len(POOL_MACHINES)} machines): {POOL_MACHINES}")

    print("\nLoading 5,000,000 rows ...")
    df_raw = pd.read_csv(
        config.DATA_PATH, nrows=5_000_000,
        usecols=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"],
    )

    # Cache every machine's raw splits once (used both for pool-building and per-machine eval)
    raw_splits = {mid: _machine_raw_splits(mid, df_raw) for mid in machines}
    pool_train_raws = [raw_splits[mid][0] for mid in POOL_MACHINES]
    scaler = _fit_pool_scaler(pool_train_raws)
    print(f"Pooled scaler fit on {sum(len(a) for a in pool_train_raws)} pooled train points "
          f"(range seen: [{scaler.data_min_[0]:.2f}, {scaler.data_max_[0]:.2f}])")

    out_rows = []

    for seed in SEEDS:
        print(f"\n{'='*78}\n  SEED {seed}: pretraining pooled model on {POOL_MACHINES}\n{'='*78}")
        np.random.seed(seed)
        tf.random.set_seed(seed)
        X_pool, y_pool = _build_pooled_sequences(pool_train_raws, scaler, seed)
        print(f"  pooled training sequences: {len(X_pool)}")
        pooled_model = _build_lstm_model(config.LOOKBACK_STEPS, config.HORIZON_STEPS)
        model_path = os.path.join(EXP_DIR, f"_tmp_pooled_seed_{seed}.keras")
        history, secs = _train_lstm(pooled_model, X_pool, y_pool, model_path)
        pretrained_weights = pooled_model.get_weights()
        print(f"  pretrained in {secs:.1f}s, {len(history.history['loss'])} epochs")

        for mid in machines:
            ds = demand_scales[mid]
            train_raw, val_raw, test_raw = raw_splits[mid]

            # ── Variant (a): pooled, no fine-tuning ──────────────────────────
            r_a = _tune_and_eval(pooled_model, scaler, val_raw, test_raw, ds)
            out_rows.append({"machine_id": mid, "seed": seed, "variant": "pooled_no_finetune", **r_a})

            # ── Variant (b): pooled + fine-tuned on this machine's own train ──
            np.random.seed(seed)
            tf.random.set_seed(seed)
            ft_model = _build_lstm_model(config.LOOKBACK_STEPS, config.HORIZON_STEPS)
            ft_model.set_weights(pretrained_weights)
            X_train_mid, y_train_mid = _make_sequences(
                scaler.transform(train_raw), config.LOOKBACK_STEPS, config.HORIZON_STEPS)
            ft_path = os.path.join(EXP_DIR, f"_tmp_finetune_{mid}_seed_{seed}.keras")
            _train_lstm(ft_model, X_train_mid, y_train_mid, ft_path)
            r_b = _tune_and_eval(ft_model, scaler, val_raw, test_raw, ds)
            out_rows.append({"machine_id": mid, "seed": seed, "variant": "pooled_finetuned", **r_b})

            print(f"    {mid:<10} (a) no-finetune cost={r_a['cost_score']:.4f} SLA={r_a['sla_violation_rate_pct']:.2f}%   "
                  f"(b) finetuned cost={r_b['cost_score']:.4f} SLA={r_b['sla_violation_rate_pct']:.2f}%")

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(RESULTS_V13, index=False)
    print(f"\nSaved: {RESULTS_V13}  ({len(out_df)} rows)")

    # ── Summary vs Step 11 baseline ──────────────────────────────────────────
    summary = []
    for mid in machines:
        row = v7[v7["machine_id"] == mid]
        lstm_base_cost, lstm_base_std = row["lstm_cost_score"].mean(), row["lstm_cost_score"].std()
        arima_cost = row["arima_cost_score"].iloc[0]
        reactive_cost = row["reactive_cost_score"].iloc[0]

        a_rows = out_df[(out_df.machine_id == mid) & (out_df.variant == "pooled_no_finetune")]
        b_rows = out_df[(out_df.machine_id == mid) & (out_df.variant == "pooled_finetuned")]
        a_mean, a_std = a_rows["cost_score"].mean(), a_rows["cost_score"].std()
        b_mean, b_std = b_rows["cost_score"].mean(), b_rows["cost_score"].std()

        def verdict(cost_after, std_after):
            gap = lstm_base_cost - cost_after
            combined = lstm_base_std + std_after
            if gap > combined:
                return "pooled confirmed better"
            elif gap > 0:
                return "pooled directional better"
            elif abs(gap) <= combined:
                return "tied"
            else:
                return "pooled confirmed worse"

        summary.append({
            "machine_id": mid, "in_pool_cluster": mid in POOL_MACHINES,
            "lstm_baseline_cost": lstm_base_cost, "lstm_baseline_std": lstm_base_std,
            "arima_cost": arima_cost, "reactive_cost": reactive_cost,
            "pooled_no_finetune_cost": a_mean, "pooled_no_finetune_std": a_std,
            "pooled_no_finetune_verdict": verdict(a_mean, a_std),
            "pooled_finetuned_cost": b_mean, "pooled_finetuned_std": b_std,
            "pooled_finetuned_verdict": verdict(b_mean, b_std),
        })

    sm_df = pd.DataFrame(summary)
    sm_df.to_csv(SUMMARY_V13, index=False)
    print(f"Saved: {SUMMARY_V13}")

    print("\n" + "=" * 116)
    print("  POOLED LSTM: per-machine baseline (Step 11) vs pooled (no finetune) vs pooled+finetuned")
    print("=" * 116)
    for _, r in sm_df.sort_values("machine_id").iterrows():
        tag = "[POOL]" if r["in_pool_cluster"] else "      "
        print(f"  {tag} {r['machine_id']:<10} baseline={r['lstm_baseline_cost']:.4f}±{r['lstm_baseline_std']:.4f}  "
              f"no-ft={r['pooled_no_finetune_cost']:.4f}±{r['pooled_no_finetune_std']:.4f} [{r['pooled_no_finetune_verdict']}]  "
              f"ft={r['pooled_finetuned_cost']:.4f}±{r['pooled_finetuned_std']:.4f} [{r['pooled_finetuned_verdict']}]  "
              f"(ARIMA={r['arima_cost']:.4f} Reactive={r['reactive_cost']:.4f})")
    print("\n  no-finetune verdict counts:", sm_df["pooled_no_finetune_verdict"].value_counts().to_dict())
    print("  finetuned verdict counts:", sm_df["pooled_finetuned_verdict"].value_counts().to_dict())
    print("=" * 116 + "\n")


if __name__ == "__main__":
    main()
