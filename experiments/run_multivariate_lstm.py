"""
Step 14 (Phase 2): give the LSTM input signal ARIMA structurally can't use
-- time-of-day/day-of-week (cyclically encoded) and a short rolling
mean/std -- as extra input channels (multivariate input instead of
univariate CPU%), and see whether that produces any new, confirmed
LSTM-vs-ARIMA cost advantage. Everything else (architecture, grids, decision
engine, tuning discipline) is held fixed to isolate "does more input signal
help" from any of the confounds Steps 10/13/14a-c already caught.

LSTM: `tune_on_validation(multivariate=True, target_builder=_build_multistep_targets,
horizon_weights_grid=HW_GRID)` then `run_single_experiment(..., multivariate=True)`
x 5 seeds -- real retraining, same fair (multistep + horizon_weights-searched)
setup Step 13 established. ARIMA: reused UNCHANGED from Step 14a's per-machine
AIC-order + horizon_weights-searched numbers (results_v11_arima_reordered.csv)
-- ARIMA structurally cannot use these extra channels (it's the whole point
of this experiment), so there's nothing to re-run there. Reactive: reused
unchanged from results_v7 (deterministic since Step 8).

Writes experiments/results_v12_multivariate_lstm.csv (65 rows) and
experiments/results_v12_multivariate_summary.csv (13 rows: univariate vs
multivariate LSTM cost, head-to-head vs (order-corrected) ARIMA, vs Reactive).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.autoscaler import (
    DATA_PATH,
    _build_multistep_targets,
    run_single_experiment,
    tune_on_validation,
)

SEEDS = [42, 43, 44, 45, 46]

HORIZON_WEIGHT_SCHEMES = {
    "uniform":        None,
    "linear_321":     [0.5, 1.0 / 3.0, 1.0 / 6.0],
    "front_60_30_10": [0.6, 0.3, 0.1],
    "front_80_15_05": [0.8, 0.15, 0.05],
}
HW_GRID = list(HORIZON_WEIGHT_SCHEMES.values())


def _hw_name(hw):
    if hw is None:
        return "uniform"
    for name, val in HORIZON_WEIGHT_SCHEMES.items():
        if val is not None and np.allclose(val, hw):
            return name
    return str(hw)


RESULTS_V10_SUMMARY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "results_v10_horizon_weights_summary.csv")
RESULTS_V11 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_v11_arima_reordered.csv")
RESULTS_V7 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "results_v7_arima_full13.csv")
RESULTS_V12 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_v12_multivariate_lstm.csv")
SUMMARY_V12 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_v12_multivariate_summary.csv")


def _own_verdict(cost_before, cost_after, std_before, std_after):
    gap = cost_before - cost_after
    combined = (std_before or 0.0) + (std_after or 0.0)
    if gap > combined:
        return "multivariate confirmed better"
    elif gap > 0:
        return "multivariate directional improvement"
    elif abs(gap) <= combined:
        return "tied"
    else:
        return "multivariate confirmed worse"


def _h2h_verdict(cost_L, std_L, cost_A):
    gap = cost_A - cost_L
    combined = std_L
    if gap > combined:
        return "LSTM confirmed cheaper"
    elif gap > 0:
        return "LSTM directional (unconfirmed)"
    elif gap == 0:
        return "tied (exact)"
    elif gap > -combined:
        return "ARIMA directional (unconfirmed)"
    else:
        return "ARIMA confirmed cheaper"


def _vs_reactive_verdict(cost_policy, std_policy, cost_reactive):
    gap = cost_reactive - cost_policy
    combined = std_policy
    if gap > combined:
        return "beats Reactive (confirmed)"
    elif gap > 0:
        return "beats Reactive (directional)"
    elif gap == 0:
        return "tied w/ Reactive (exact)"
    elif gap > -combined:
        return "Reactive directional"
    else:
        return "Reactive wins (confirmed)"


def main() -> None:
    v10 = pd.read_csv(RESULTS_V10_SUMMARY)
    v11 = pd.read_csv(RESULTS_V11)
    v7 = pd.read_csv(RESULTS_V7)
    machines = sorted(v10["machine_id"].unique())
    print(f"{len(machines)} feasible machines: {machines}")

    print("\nLoading 5,000,000 rows ...")
    df_raw = pd.read_csv(
        DATA_PATH, nrows=5_000_000,
        usecols=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"],
    )

    out_rows = []
    summary = []

    for mid in machines:
        ds = float(v7[v7["machine_id"] == mid]["demand_scale"].iloc[0])
        burst = float(v10[v10["machine_id"] == mid]["burstiness"].iloc[0])
        print(f"\n{'='*78}\n  {mid}  (demand_scale={ds:.4f}x, burstiness={burst:.4f})\n{'='*78}")

        print("  Tuning multivariate LSTM (multistep + horizon_weights grid) on validation ...")
        lstm_best = tune_on_validation(
            seed=42, machine_id=mid, df_raw=df_raw, demand_scale=ds,
            target_builder=_build_multistep_targets, horizon_weights_grid=HW_GRID,
            multivariate=True,
        )
        lstm_hw = lstm_best["lstm_horizon_weights"]
        print(f"    upw={lstm_best['lstm_under_prov_weight']} sm={lstm_best['lstm_safety_margin']} "
              f"hw={_hw_name(lstm_hw)}  val_cost={lstm_best['lstm_val_cost']:.4f}")

        lstm_rows = []
        for seed in SEEDS:
            r = run_single_experiment(
                seed,
                under_prov_weight=lstm_best["lstm_under_prov_weight"],
                safety_margin=lstm_best["lstm_safety_margin"],
                reactive_up=lstm_best["reactive_up"],
                reactive_down=lstm_best["reactive_down"],
                machine_id=mid, df_raw=df_raw, demand_scale=ds,
                target_builder=_build_multistep_targets, horizon_weights=lstm_hw,
                multivariate=True,
            )
            lstm_rows.append(r)
            print(f"    seed={seed}: cost={r['lstm_cost_score']:.4f}  SLA={r['lstm_sla_violation_rate_pct']:.4f}%")
            out_rows.append({
                "machine_id": mid, "burstiness": burst, "seed": seed,
                "lstm_under_prov_weight": lstm_best["lstm_under_prov_weight"],
                "lstm_safety_margin": lstm_best["lstm_safety_margin"],
                "lstm_horizon_weights": _hw_name(lstm_hw),
                "lstm_cost_score_multivariate": r["lstm_cost_score"],
                "lstm_sla_violation_rate_pct_multivariate": r["lstm_sla_violation_rate_pct"],
            })

        mv_costs = np.array([r["lstm_cost_score"] for r in lstm_rows])
        mv_cost_mean, mv_cost_std = mv_costs.mean(), mv_costs.std()
        mv_sla_mean = np.mean([r["lstm_sla_violation_rate_pct"] for r in lstm_rows])

        v10_row = v10[v10["machine_id"] == mid].iloc[0]
        v11_row = v11[v11["machine_id"] == mid].iloc[0]
        uni_cost, uni_std = v10_row["lstm_hw_cost"], v10_row["lstm_hw_cost_std"]
        arima_cost = v11_row["arima_cost_new_order"]
        reactive_cost = v11_row["reactive_cost"]

        own_verdict = _own_verdict(uni_cost, mv_cost_mean, uni_std, mv_cost_std)
        h2h = _h2h_verdict(mv_cost_mean, mv_cost_std, arima_cost)
        vs_reactive = _vs_reactive_verdict(mv_cost_mean, mv_cost_std, reactive_cost)
        h2h_uni = _h2h_verdict(uni_cost, uni_std, arima_cost)

        summary.append({
            "machine_id": mid, "burstiness": burst,
            "lstm_uni_cost": uni_cost, "lstm_uni_cost_std": uni_std,
            "lstm_mv_cost": mv_cost_mean, "lstm_mv_cost_std": mv_cost_std,
            "lstm_mv_sla": mv_sla_mean,
            "own_verdict": own_verdict,
            "arima_cost_order_corrected": arima_cost, "reactive_cost": reactive_cost,
            "h2h_verdict_univariate": h2h_uni, "h2h_verdict_multivariate": h2h,
            "h2h_changed": h2h_uni != h2h,
            "vs_reactive_multivariate": vs_reactive,
        })
        print(f"  LSTM cost: univariate {uni_cost:.4f}±{uni_std:.4f}  ->  multivariate {mv_cost_mean:.4f}±{mv_cost_std:.4f}  [{own_verdict}]")
        print(f"  Head-to-head vs ARIMA (order-corrected, {arima_cost:.4f}): "
              f"univariate [{h2h_uni}]  ->  multivariate [{h2h}]{'  ** CHANGED **' if h2h_uni != h2h else ''}")
        print(f"  vs Reactive ({reactive_cost:.4f}): [{vs_reactive}]")

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(RESULTS_V12, index=False)
    print(f"\nSaved: {RESULTS_V12}  ({len(out_df)} rows)")

    sm_df = pd.DataFrame(summary)
    sm_df.to_csv(SUMMARY_V12, index=False)
    print(f"Saved: {SUMMARY_V12}")

    print("\n" + "=" * 110)
    print("  MULTIVARIATE LSTM: univariate (Step 13) vs +time/rolling-stat channels (Step 14)")
    print("=" * 110)
    for _, r in sm_df.sort_values("burstiness").iterrows():
        print(f"  {r['machine_id']:<10} burst={r['burstiness']:>6.2f}  "
              f"cost: {r['lstm_uni_cost']:.4f}->{r['lstm_mv_cost']:.4f} [{r['own_verdict']:<30}]  "
              f"h2h: {r['h2h_verdict_univariate']:<28} -> {r['h2h_verdict_multivariate']:<28}"
              f"{'  ** CHANGED **' if r['h2h_changed'] else ''}  vsReactive:[{r['vs_reactive_multivariate']}]")
    print(f"\n  own-verdict counts:", sm_df["own_verdict"].value_counts().to_dict())
    print(f"  NEW confirmed LSTM-vs-ARIMA wins under multivariate:",
          sm_df[(sm_df["h2h_verdict_multivariate"] == "LSTM confirmed cheaper") &
                (sm_df["h2h_verdict_univariate"] != "LSTM confirmed cheaper")]["machine_id"].tolist())
    print("=" * 110 + "\n")


if __name__ == "__main__":
    main()
