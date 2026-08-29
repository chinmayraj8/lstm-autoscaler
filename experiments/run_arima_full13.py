"""
Step 11 (Phase 2): does the LSTM have a genuine, forecaster-specific cost
advantage over Reactive, or does ARIMA match/beat it on the same machines?

Step 10 wired ARIMA into the decision engine + simulator on 3 machines and
found ARIMA beat the LSTM on m_2189 -- Step 8's clearest confirmed LSTM cost
advantage over Reactive. That raises the question this script answers
directly, across all 13 of Step 8's feasible machines: does the LSTM beat
BOTH Reactive AND ARIMA on cost, or does ARIMA also do as well as (or
better than) the LSTM wherever the LSTM beats Reactive?

To keep this cheap and honest, LSTM and Reactive are NOT re-tuned or
re-run here: their already-recorded, correctly-tuned numbers (5 seeds each,
tuned on validation only, per Step 8's discipline) are reused directly from
experiments/results_v5_expanded_multimachine.csv. Only ARIMA is computed
fresh: for each machine, `tune_arima_on_validation` grid-searches ARIMA's
own decision params on validation with the *same* grids the LSTM was tuned
with (LSTM_UPW_GRID x LSTM_SM_GRID) -- so ARIMA is tuned exactly as hard as
the LSTM was, no harder and no easier -- then `run_arima_experiment` runs
the tuned ARIMA once on test. ARIMA's fit is deterministic (statsmodels,
no seed-dependent randomness -- see arima_baseline.py's docstring); this
script re-runs it once per machine and asserts the cost score is
bit-identical, exactly like Step 10 did, rather than assuming it.

Writes experiments/results_v7_arima_full13.csv: all 13 machines' existing
results_v5 rows (5 per machine, unchanged) plus the new arima_* columns
(the single ARIMA value repeated across all 5 rows for that machine, same
convention results_v6 used).

Then classifies each machine's COST result into one of three buckets:
  - LSTM has a confirmed cost advantage over BOTH Reactive and ARIMA
    (gap > combined +/-1 sigma against each) -- a genuine, forecaster-
    specific advantage.
  - LSTM has a confirmed cost advantage over Reactive, but ARIMA also
    matches or beats the LSTM -- Step 8's original "LSTM cost advantage"
    framing was real (Reactive does lose), but the advantage isn't
    LSTM-specific: any reasonable forecaster gets it.
  - Reactive has a confirmed cost advantage over both, or the results are
    mixed/tied -- no forecaster-based advantage on this machine.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.autoscaler import (
    DATA_PATH,
    run_arima_experiment,
    tune_arima_on_validation,
)

RESULTS_V5 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "results_v5_expanded_multimachine.csv")
RESULTS_V7 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "results_v7_arima_full13.csv")

ARIMA_COLS = [
    "arima_under_prov_weight_tuned", "arima_safety_margin_tuned", "arima_order",
    "arima_forecast_rmse", "arima_forecast_mae",
    "arima_sla_violation_rate_pct", "arima_over_prov_waste_pct",
    "arima_cost_score", "arima_wall_clock_secs",
]


def _verdict(cost_a: float, cost_b: float, std_a: float, std_b: float) -> str:
    """Is `a` confirmed cheaper than `b`? Same combined-+/-1-sigma rule as
    run_multimachine_v2._verdict, generalized to any two policies."""
    gap = cost_b - cost_a   # positive => a is cheaper
    combined = (std_a or 0.0) + (std_b or 0.0)
    if gap > combined:
        return "confirmed cheaper"
    elif gap > 0:
        return "directional (within ±1σ)"
    elif abs(gap) <= combined:
        return "tied (within ±1σ)"
    else:
        return "confirmed more expensive"


def main() -> None:
    v5 = pd.read_csv(RESULTS_V5)
    machines = sorted(v5["machine_id"].unique())
    print(f"{len(machines)} feasible machines from Step 8: {machines}")

    print("\nLoading 5,000,000 rows ...")
    df_raw = pd.read_csv(
        DATA_PATH, nrows=5_000_000,
        usecols=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"],
    )

    out_rows = []
    summary = []

    for mid in machines:
        grp = v5[v5["machine_id"] == mid]
        ds = float(grp["demand_scale"].iloc[0])
        burst = float(grp["burstiness_score"].iloc[0])

        print(f"\n{'='*70}\n  {mid}  (demand_scale={ds:.4f}x, burstiness={burst:.4f})\n{'='*70}")

        print("  Tuning ARIMA decision params on validation (same grids as LSTM) ...")
        a_best = tune_arima_on_validation(seed=42, machine_id=mid, df_raw=df_raw, demand_scale=ds)
        print(f"    ARIMA: upw={a_best['arima_under_prov_weight']} "
              f"sm={a_best['arima_safety_margin']}  val_cost={a_best['arima_val_cost']:.4f}  "
              f"val_rmse={a_best['arima_val_rmse']:.4f}")

        arima_result = run_arima_experiment(
            seed=42,
            under_prov_weight=a_best["arima_under_prov_weight"],
            safety_margin=a_best["arima_safety_margin"],
            machine_id=mid, df_raw=df_raw, demand_scale=ds,
        )
        arima_result_2 = run_arima_experiment(
            seed=42,
            under_prov_weight=a_best["arima_under_prov_weight"],
            safety_margin=a_best["arima_safety_margin"],
            machine_id=mid, df_raw=df_raw, demand_scale=ds,
        )
        assert arima_result["arima_cost_score"] == arima_result_2["arima_cost_score"], (
            f"{mid}: ARIMA cost score was NOT deterministic across two identical runs."
        )
        print(f"    ARIMA test: RMSE={arima_result['arima_forecast_rmse']:.4f}  "
              f"SLA={arima_result['arima_sla_violation_rate_pct']:.4f}%  "
              f"cost={arima_result['arima_cost_score']:.4f}  (determinism check: matched)")

        arima_cols_values = {
            "arima_under_prov_weight_tuned": a_best["arima_under_prov_weight"],
            "arima_safety_margin_tuned": a_best["arima_safety_margin"],
            "arima_order": str(arima_result["arima_order"]),
            "arima_forecast_rmse": arima_result["arima_forecast_rmse"],
            "arima_forecast_mae": arima_result["arima_forecast_mae"],
            "arima_sla_violation_rate_pct": arima_result["arima_sla_violation_rate_pct"],
            "arima_over_prov_waste_pct": arima_result["arima_over_prov_waste_pct"],
            "arima_cost_score": arima_result["arima_cost_score"],
            "arima_wall_clock_secs": arima_result["arima_wall_clock_secs"],
        }
        for _, row in grp.iterrows():
            new_row = row.to_dict()
            new_row.update(arima_cols_values)
            out_rows.append(new_row)

        lstm_cost_mean = grp["lstm_cost_score"].mean()
        lstm_cost_std = grp["lstm_cost_score"].std()
        reactive_cost_mean = grp["reactive_cost_score"].mean()
        reactive_cost_std = grp["reactive_cost_score"].std()
        lstm_sla_mean = grp["lstm_sla_violation_rate_pct"].mean()
        lstm_sla_std = grp["lstm_sla_violation_rate_pct"].std()
        reactive_sla_mean = grp["reactive_sla_violation_rate_pct"].mean()
        reactive_sla_std = grp["reactive_sla_violation_rate_pct"].std()
        arima_cost = arima_result["arima_cost_score"]
        arima_sla = arima_result["arima_sla_violation_rate_pct"]

        lstm_vs_reactive = _verdict(lstm_cost_mean, reactive_cost_mean, lstm_cost_std, reactive_cost_std)
        lstm_vs_arima = _verdict(lstm_cost_mean, arima_cost, lstm_cost_std, 0.0)
        arima_vs_reactive = _verdict(arima_cost, reactive_cost_mean, 0.0, reactive_cost_std)

        # lstm_vs_reactive == "confirmed more expensive" <=> Reactive is confirmed
        # cheaper than LSTM (same gap, mirrored); same logic for ARIMA below.
        if lstm_vs_reactive == "confirmed cheaper" and lstm_vs_arima == "confirmed cheaper":
            bucket = "LSTM-specific advantage"
        elif lstm_vs_reactive == "confirmed cheaper":
            bucket = "forecaster-general advantage (ARIMA matches/beats LSTM)"
        elif lstm_vs_reactive == "confirmed more expensive" and arima_vs_reactive == "confirmed more expensive":
            bucket = "Reactive wins outright"
        else:
            bucket = "mixed / no confirmed winner"

        summary.append({
            "machine_id": mid, "burstiness": burst,
            "lstm_cost_mean": lstm_cost_mean, "lstm_cost_std": lstm_cost_std,
            "reactive_cost_mean": reactive_cost_mean, "reactive_cost_std": reactive_cost_std,
            "arima_cost": arima_cost,
            "lstm_sla_mean": lstm_sla_mean, "reactive_sla_mean": reactive_sla_mean,
            "arima_sla": arima_sla,
            "lstm_vs_reactive": lstm_vs_reactive, "lstm_vs_arima": lstm_vs_arima,
            "arima_vs_reactive": arima_vs_reactive, "bucket": bucket,
        })
        print(f"  Cost: LSTM {lstm_cost_mean:.4f}±{lstm_cost_std:.4f}  |  "
              f"Reactive {reactive_cost_mean:.4f}±{reactive_cost_std:.4f}  |  "
              f"ARIMA {arima_cost:.4f}")
        print(f"  LSTM vs Reactive: {lstm_vs_reactive}  |  LSTM vs ARIMA: {lstm_vs_arima}  |  "
              f"ARIMA vs Reactive: {arima_vs_reactive}")
        print(f"  >>> {bucket}")

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(RESULTS_V7, index=False)
    print(f"\nSaved: {RESULTS_V7}  ({len(out_df)} rows, {len(machines)} machines)")

    # ── Final classification table ────────────────────────────────────────
    sm = pd.DataFrame(summary)
    print("\n" + "=" * 100)
    print("  FINAL CLASSIFICATION: does the LSTM beat BOTH Reactive AND ARIMA on cost?")
    print("=" * 100)
    print(f"  {'Machine':<10} {'Burst':>8} {'LSTM cost':>14} {'React cost':>14} {'ARIMA cost':>11}  Bucket")
    print("  " + "-" * 96)
    for _, r in sm.sort_values("burstiness").iterrows():
        print(f"  {r['machine_id']:<10} {r['burstiness']:>8.2f} "
              f"{r['lstm_cost_mean']:>7.4f}±{r['lstm_cost_std']:.4f} "
              f"{r['reactive_cost_mean']:>7.4f}±{r['reactive_cost_std']:.4f} "
              f"{r['arima_cost']:>11.4f}  {r['bucket']}")

    counts = sm["bucket"].value_counts()
    print("\n  " + "-" * 70)
    for bucket, n in counts.items():
        print(f"  {bucket}: {n}/{len(sm)}")
    print("=" * 100 + "\n")

    sm.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_v7_classification_summary.csv"), index=False)


if __name__ == "__main__":
    main()
