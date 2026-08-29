"""
Step 10 (Phase 2): ARIMA wired into the actual autoscaler simulation.

Step 6 (run_baselines_and_sweep.py) only ever compared ARIMA's point-forecast
RMSE/MAE against the LSTM's, on one machine (m_1933, smooth demand). It never
ran ARIMA's forecasts through the decision engine + fleet simulator, so its
cost score and SLA rate were unknown -- and it was never tested on a bursty
machine.

This script does both, in one pass, for three machines:
- m_1933   -- Step 6's original machine (smooth demand), now under the
              project's current calibrated-demand-scale convention (Step 5+)
              instead of Step 6's global DEMAND_SCALE=20, since we're running
              the full simulation, not just forecast RMSE.
- m_2189   -- bursty (Step 8: burstiness=14.49), LSTM's clearest confirmed
              cost advantage over Reactive.
- m_2134   -- the most bursty machine tested so far (Step 8: burstiness=38.2),
              the stress-test case where Reactive wins both metrics outright.

For each machine:
  1. tune_on_validation()       -- LSTM + Reactive decision params, grid-
                                    searched on the val split (existing, Step
                                    3+ discipline unchanged).
  2. tune_arima_on_validation() -- ARIMA's decision params, grid-searched on
                                    the SAME val split with the SAME grids
                                    (LSTM_UPW_GRID x LSTM_SM_GRID) -- so ARIMA
                                    is tuned exactly as hard as the LSTM, no
                                    harder and no easier.
  3. run_single_experiment() x5 seeds  -- LSTM (stochastic) + Reactive
     (deterministic, kept for continuity with existing multi-seed CSVs).
  4. run_arima_experiment()             -- ARIMA (deterministic; run once,
     not averaged over 5 identical values -- see arima_baseline.py's
     docstring for why).

Writes experiments/results_v6_arima_in_sim.csv (wide format, one row per
seed, consistent with results_v5's schema plus arima_* columns).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.autoscaler import (
    DATA_PATH,
    FEATURE_COL,
    _load_and_prepare,
    calibrate_demand_scale,
    check_feasibility,
    run_arima_experiment,
    run_single_experiment,
    tune_arima_on_validation,
    tune_on_validation,
)

SEEDS = [42, 43, 44, 45, 46]

# (machine_id, nrows) -- m_1933 uses the 500K-row default (Step 6's original
# load size, and _pick_best_machine's top pick within it); the two bursty
# machines use 5M rows to match how they were selected/calibrated in Step 8,
# so ARIMA is compared against the identical demand series LSTM/Reactive
# already have recorded numbers for in results_v5_expanded_multimachine.csv.
MACHINES = [
    ("m_1933", 500_000),
    ("m_2189", 5_000_000),
    ("m_2134", 5_000_000),
]

RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_v6_arima_in_sim.csv")

METRIC_COLS = [
    "machine_id", "burstiness_score", "demand_scale",
    "demand_mean_pct", "demand_p95_pct", "demand_p99_pct",
    "reactive_up_tuned", "reactive_down_tuned",
    "lstm_under_prov_weight_tuned", "lstm_safety_margin_tuned",
    "arima_under_prov_weight_tuned", "arima_safety_margin_tuned", "arima_order",
    "seed",
    "lstm_forecast_rmse", "lstm_forecast_mae",
    "arima_forecast_rmse", "arima_forecast_mae",
    "naive_baseline_rmse",
    "lstm_sla_violation_rate_pct", "reactive_sla_violation_rate_pct",
    "arima_sla_violation_rate_pct",
    "lstm_over_prov_waste_pct", "reactive_over_prov_waste_pct",
    "arima_over_prov_waste_pct",
    "lstm_cost_score", "reactive_cost_score", "arima_cost_score",
    "epochs_trained", "wall_clock_secs", "arima_wall_clock_secs",
]


def _burstiness(ts) -> float:
    """Same diff-p95 formula as run_multimachine._compute_machine_stats."""
    cpu = ts[FEATURE_COL].values
    diffs = np.abs(np.diff(cpu))
    return float(np.percentile(diffs, 95)) if len(diffs) >= 20 else 0.0


def _append_row(row: dict) -> None:
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        pd.DataFrame([{c: row.get(c) for c in METRIC_COLS}]).to_csv(
            f, header=write_header, index=False
        )


def _verdict(gap: float, combined_sigma: float) -> str:
    if gap > combined_sigma:
        return "wins (confirmed, gap > combined ±1σ)"
    elif gap > 0:
        return "directional advantage (within ±1σ)"
    elif abs(gap) <= combined_sigma:
        return "tied (within ±1σ)"
    else:
        return "loses"


def run_machine(mid: str, nrows: int, df_raw: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print(f"  MACHINE: {mid}")
    print("=" * 78)

    ts, mid = _load_and_prepare(machine_id=mid, df_raw=df_raw)
    burst = _burstiness(ts)
    ds = calibrate_demand_scale(float(ts[FEATURE_COL].mean()))
    feas = check_feasibility(ts, ds)
    print(f"  {len(ts)} resampled points | burstiness={burst:.4f} | demand_scale={ds:.4f}x")
    print(f"  demand mean/p95/p99 = {feas['demand_mean']:.1f}/{feas['demand_p95']:.1f}/"
          f"{feas['demand_p99']:.1f}%  ({feas['reason']})")
    if feas["infeasible"]:
        print(f"  SKIPPING {mid} -- infeasible at this demand_scale.")
        return

    print("\n  Tuning LSTM + Reactive on validation (seed=42) ...")
    lr_best = tune_on_validation(seed=42, machine_id=mid, df_raw=df_raw, demand_scale=ds)
    print(f"    Reactive: up={lr_best['reactive_up']} down={lr_best['reactive_down']}  "
          f"val_cost={lr_best['reactive_val_cost']:.4f}")
    print(f"    LSTM dec: upw={lr_best['lstm_under_prov_weight']} "
          f"sm={lr_best['lstm_safety_margin']}  val_cost={lr_best['lstm_val_cost']:.4f}")

    print("\n  Tuning ARIMA decision params on validation (same grids) ...")
    a_best = tune_arima_on_validation(seed=42, machine_id=mid, df_raw=df_raw, demand_scale=ds)
    print(f"    ARIMA:    upw={a_best['arima_under_prov_weight']} "
          f"sm={a_best['arima_safety_margin']}  val_cost={a_best['arima_val_cost']:.4f}  "
          f"val_rmse={a_best['arima_val_rmse']:.4f}")

    print("\n  Running ARIMA on TEST (deterministic, once) ...")
    arima_result = run_arima_experiment(
        seed=42,
        under_prov_weight=a_best["arima_under_prov_weight"],
        safety_margin=a_best["arima_safety_margin"],
        machine_id=mid, df_raw=df_raw, demand_scale=ds,
    )
    # Determinism spot-check: re-run once and confirm bit-identical cost score.
    arima_result_2 = run_arima_experiment(
        seed=42,
        under_prov_weight=a_best["arima_under_prov_weight"],
        safety_margin=a_best["arima_safety_margin"],
        machine_id=mid, df_raw=df_raw, demand_scale=ds,
    )
    assert arima_result["arima_cost_score"] == arima_result_2["arima_cost_score"], (
        "ARIMA cost score was NOT deterministic across two identical runs -- "
        "the module docstring's claim doesn't hold, investigate before trusting these numbers."
    )
    print(f"    ARIMA:    RMSE={arima_result['arima_forecast_rmse']:.4f}  "
          f"SLA={arima_result['arima_sla_violation_rate_pct']:.4f}%  "
          f"cost={arima_result['arima_cost_score']:.4f}  "
          f"(determinism check: re-run matched exactly)")

    print(f"\n  Running LSTM + Reactive on TEST, seeds {SEEDS[0]}-{SEEDS[-1]} ...")
    lstm_rows = []
    for seed in SEEDS:
        result = run_single_experiment(
            seed,
            under_prov_weight=lr_best["lstm_under_prov_weight"],
            safety_margin=lr_best["lstm_safety_margin"],
            reactive_up=lr_best["reactive_up"],
            reactive_down=lr_best["reactive_down"],
            machine_id=mid, df_raw=df_raw, demand_scale=ds,
        )
        lstm_rows.append(result)
        row = {
            "machine_id": mid, "burstiness_score": burst, "demand_scale": ds,
            "demand_mean_pct": feas["demand_mean"], "demand_p95_pct": feas["demand_p95"],
            "demand_p99_pct": feas["demand_p99"],
            "reactive_up_tuned": lr_best["reactive_up"],
            "reactive_down_tuned": lr_best["reactive_down"],
            "lstm_under_prov_weight_tuned": lr_best["lstm_under_prov_weight"],
            "lstm_safety_margin_tuned": lr_best["lstm_safety_margin"],
            "arima_under_prov_weight_tuned": a_best["arima_under_prov_weight"],
            "arima_safety_margin_tuned": a_best["arima_safety_margin"],
            "arima_order": arima_result["arima_order"],
            **result,
            "arima_forecast_rmse": arima_result["arima_forecast_rmse"],
            "arima_forecast_mae": arima_result["arima_forecast_mae"],
            "arima_sla_violation_rate_pct": arima_result["arima_sla_violation_rate_pct"],
            "arima_over_prov_waste_pct": arima_result["arima_over_prov_waste_pct"],
            "arima_cost_score": arima_result["arima_cost_score"],
            "arima_wall_clock_secs": arima_result["arima_wall_clock_secs"],
        }
        _append_row(row)
        print(f"    seed={seed}: LSTM SLA={result['lstm_sla_violation_rate_pct']:.4f}%  "
              f"cost={result['lstm_cost_score']:.4f}  |  "
              f"Reactive SLA={result['reactive_sla_violation_rate_pct']:.4f}%  "
              f"cost={result['reactive_cost_score']:.4f}  |  "
              f"epochs={result['epochs_trained']}")

    # ── Per-machine 3-way summary ─────────────────────────────────────────────
    lstm_sla = np.array([r["lstm_sla_violation_rate_pct"] for r in lstm_rows])
    lstm_cost = np.array([r["lstm_cost_score"] for r in lstm_rows])
    react_sla = np.array([r["reactive_sla_violation_rate_pct"] for r in lstm_rows])
    react_cost = np.array([r["reactive_cost_score"] for r in lstm_rows])
    lstm_rmse = np.array([r["lstm_forecast_rmse"] for r in lstm_rows])

    print(f"\n  {'-'*70}")
    print(f"  3-WAY SUMMARY -- {mid}  (burstiness={burst:.4f})")
    print(f"  {'-'*70}")
    print(f"  {'Method':<10} {'RMSE':>10} {'SLA %':>16} {'Cost score':>18}")
    print(f"  {'Naive':<10} {'-':>10} {'-':>16} {'-':>18}")
    print(f"  {'ARIMA':<10} {arima_result['arima_forecast_rmse']:>10.4f} "
          f"{arima_result['arima_sla_violation_rate_pct']:>15.4f}% "
          f"{arima_result['arima_cost_score']:>18.4f}  (deterministic, std=0)")
    print(f"  {'LSTM':<10} {lstm_rmse.mean():>6.4f}±{lstm_rmse.std():.4f} "
          f"{lstm_sla.mean():>10.4f}%±{lstm_sla.std():.4f} "
          f"{lstm_cost.mean():>12.4f}±{lstm_cost.std():.4f}")
    print(f"  {'Reactive':<10} {'-':>10} "
          f"{react_sla.mean():>15.4f}%±{react_sla.std():.4f} "
          f"{react_cost.mean():>18.4f}±{react_cost.std():.4f}")

    cost_gap_arima_vs_lstm = arima_result["arima_cost_score"] - lstm_cost.mean()
    sla_gap_arima_vs_lstm = arima_result["arima_sla_violation_rate_pct"] - lstm_sla.mean()
    cost_gap_arima_vs_react = arima_result["arima_cost_score"] - react_cost.mean()
    sla_gap_arima_vs_react = arima_result["arima_sla_violation_rate_pct"] - react_sla.mean()

    print(f"\n  ARIMA vs LSTM:     cost gap (ARIMA-LSTM) = {cost_gap_arima_vs_lstm:+.4f}  "
          f"({'ARIMA cheaper' if cost_gap_arima_vs_lstm < 0 else 'LSTM cheaper'})  "
          f"| SLA gap = {sla_gap_arima_vs_lstm:+.4f}pp "
          f"({'ARIMA fewer violations' if sla_gap_arima_vs_lstm < 0 else 'LSTM fewer violations'})")
    print(f"  ARIMA vs Reactive: cost gap (ARIMA-React) = {cost_gap_arima_vs_react:+.4f}  "
          f"({'ARIMA cheaper' if cost_gap_arima_vs_react < 0 else 'Reactive cheaper'})  "
          f"| SLA gap = {sla_gap_arima_vs_react:+.4f}pp "
          f"({'ARIMA fewer violations' if sla_gap_arima_vs_react < 0 else 'Reactive fewer violations'})")


def main() -> None:
    df_cache = {}
    for mid, nrows in MACHINES:
        if nrows not in df_cache:
            print(f"\nLoading {nrows:,} rows ...")
            df_cache[nrows] = pd.read_csv(
                DATA_PATH, nrows=nrows,
                usecols=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"],
            )
        run_machine(mid, nrows, df_cache[nrows])

    if os.path.exists(RESULTS_CSV):
        print(f"\nSaved: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
