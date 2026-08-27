"""
Step 5: per-machine demand-scale calibration experiment.

Workflow
--------
1. Load 5 M rows (same as Step 4).
2. For each of the 5 machines selected in Step 4, compute a per-machine
   demand_scale = TARGET_MEAN_LOAD_PCT / machine_mean_cpu so that every
   machine targets 115 % mean aggregate demand instead of applying the
   global DEMAND_SCALE = 20.0.
3. Run check_feasibility(); skip infeasible machines (p99 > max fleet cap)
   with a printed explanation.
4. For each feasible machine: tune on val (seed=42), then run seeds [42-46]
   on the test split.  Save to experiments/results_v4_calibrated_demand.csv.
5. Print:
   - Feasibility table (all 5 machines)
   - Per-machine comparison (feasible machines only)
   - Burstiness vs LSTM advantage correlation
   - v3 vs v4 direct comparison per machine
"""

import csv
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.pipeline import (
    DATA_PATH,
    DEC_MAX_SERVERS,
    FEATURE_COL,
    SIM_SERVER_CAPACITY,
    _prepare_timeseries,
    calibrate_demand_scale,
    check_feasibility,
    run_single_experiment,
    tune_on_validation,
)

SEEDS = [42, 43, 44, 45, 46]
NROWS_SELECT = 5_000_000

RESULTS_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results_v4_calibrated_demand.csv",
)

# Machines and metadata hardcoded from Step 4 (no re-running K-means)
MACHINES = [
    {"machine_id": "m_2087", "burstiness_score": 1.2500,  "cluster": 2, "is_stress_test": False},
    {"machine_id": "m_2101", "burstiness_score": 7.5000,  "cluster": 1, "is_stress_test": False},
    {"machine_id": "m_2189", "burstiness_score": 14.4857, "cluster": 3, "is_stress_test": False},
    {"machine_id": "m_2065", "burstiness_score": 16.3342, "cluster": 0, "is_stress_test": False},
    {"machine_id": "m_2134", "burstiness_score": 38.2000, "cluster": -1, "is_stress_test": True},
]

METRIC_COLS = [
    "machine_id", "burstiness_score", "cluster", "is_stress_test",
    "demand_scale", "demand_mean_pct", "demand_p95_pct", "demand_p99_pct",
    "reactive_up_tuned", "reactive_down_tuned",
    "lstm_under_prov_weight_tuned", "lstm_safety_margin_tuned",
    "seed",
    "lstm_forecast_rmse", "lstm_forecast_mae", "naive_baseline_rmse",
    "lstm_sla_violation_rate_pct", "reactive_sla_violation_rate_pct",
    "lstm_over_prov_waste_pct", "reactive_over_prov_waste_pct",
    "lstm_cost_score", "reactive_cost_score",
    "epochs_trained", "wall_clock_secs",
]


def _append_row(row: dict) -> None:
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow({col: row[col] for col in METRIC_COLS})


def _verdict(gap: float, combined_sigma: float) -> str:
    if gap > combined_sigma:
        return "LSTM wins (confirmed, gap > combined ±1σ)"
    elif gap > 0:
        return "LSTM directional advantage (within ±1σ)"
    elif abs(gap) <= combined_sigma:
        return "tied (within ±1σ)"
    else:
        return "Reactive wins"


def _print_feasibility_table(feas_map: dict) -> None:
    max_cap = DEC_MAX_SERVERS * SIM_SERVER_CAPACITY
    print("\n" + "=" * 88)
    print("  FEASIBILITY TABLE  (demand_scale = TARGET_MEAN_LOAD_PCT / machine_mean_cpu)")
    print(f"  Max fleet capacity = {max_cap:.0f}%  (10 servers × 80%)")
    print("=" * 88)
    hdr = (f"  {'Machine':<14} {'Scale':>7} {'Mean%':>7} {'P95%':>7} {'P99%':>7} "
           f"{'MaxCap%':>8}  Status")
    print(hdr)
    print("  " + "-" * 74)
    for mid, f in feas_map.items():
        status = "INFEASIBLE" if f["infeasible"] else "feasible"
        print(f"  {mid:<14} {f['demand_scale']:>7.2f} {f['demand_mean']:>7.1f} "
              f"{f['demand_p95']:>7.1f} {f['demand_p99']:>7.1f} "
              f"{f['max_fleet_cap']:>8.0f}  {status}"
              + (f"  ({f['reason']})" if f["infeasible"] else ""))
    print("=" * 88)


def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 82)
    print("  PER-MACHINE TEST RESULTS  (v4 calibrated demand, seeds 42-46)")
    print("=" * 82)

    pm_rows = []
    for mid, grp in df.groupby("machine_id", sort=False):
        burst   = float(grp["burstiness_score"].iloc[0])
        cluster = int(grp["cluster"].iloc[0])
        is_st   = bool(grp["is_stress_test"].iloc[0])
        label   = "stress-test" if is_st else f"cluster {cluster}"

        up   = grp["reactive_up_tuned"].iloc[0]
        down = grp["reactive_down_tuned"].iloc[0]
        upw  = grp["lstm_under_prov_weight_tuned"].iloc[0]
        sm   = grp["lstm_safety_margin_tuned"].iloc[0]
        ds   = grp["demand_scale"].iloc[0]

        ls_mean = grp["lstm_sla_violation_rate_pct"].mean()
        ls_std  = grp["lstm_sla_violation_rate_pct"].std()
        rs_mean = grp["reactive_sla_violation_rate_pct"].mean()
        rs_std  = grp["reactive_sla_violation_rate_pct"].std()
        lc_mean = grp["lstm_cost_score"].mean()
        lc_std  = grp["lstm_cost_score"].std()
        rc_mean = grp["reactive_cost_score"].mean()
        rc_std  = grp["reactive_cost_score"].std()

        sla_gap  = rs_mean - ls_mean
        cost_gap = rc_mean - lc_mean

        print(f"\n  {mid}  [{label}]  burstiness={burst:.4f}  demand_scale={ds:.2f}x")
        print(f"    Tuned: Reactive up={up} down={down} | LSTM upw={upw} sm={sm}")
        print(f"    SLA : LSTM {ls_mean:.4f}%±{ls_std:.4f}%  "
              f"vs  Reactive {rs_mean:.4f}%±{rs_std:.4f}%  "
              f"(gap={sla_gap:+.4f} pp)")
        print(f"    Cost: LSTM {lc_mean:.4f}±{lc_std:.4f}  "
              f"vs  Reactive {rc_mean:.4f}±{rc_std:.4f}  "
              f"(gap={cost_gap:+.4f})")
        print(f"    SLA  verdict: {_verdict(sla_gap,  ls_std + rs_std)}")
        print(f"    Cost verdict: {_verdict(cost_gap, lc_std + rc_std)}")

        pm_rows.append({
            "machine_id": mid, "burstiness": burst, "label": label,
            "sla_gap": sla_gap, "cost_gap": cost_gap,
        })

    # Burstiness correlation
    print("\n" + "=" * 82)
    print("  BURSTINESS vs LSTM RELATIVE ADVANTAGE (positive = LSTM better)")
    print("  " + "-" * 70)
    pm = pd.DataFrame(pm_rows).sort_values("burstiness")
    print(f"  {'Machine':<14} {'Burstiness':>12} {'SLA gap (R-L)':>16} {'Cost gap (R-L)':>16}  Label")
    print("  " + "-" * 70)
    for _, r in pm.iterrows():
        print(f"  {r['machine_id']:<14} {r['burstiness']:>12.4f} "
              f"{r['sla_gap']:>+16.4f} {r['cost_gap']:>+16.4f}  {r['label']}")

    if len(pm) >= 3:
        corr_sla  = pm["burstiness"].corr(pm["sla_gap"])
        corr_cost = pm["burstiness"].corr(pm["cost_gap"])
        print(f"\n  Pearson r(burstiness, SLA gap)  = {corr_sla:+.3f}")
        print(f"  Pearson r(burstiness, cost gap) = {corr_cost:+.3f}")

    # Hypothesis verdict
    print("\n  HYPOTHESIS: LSTM wins on bursty, loses/ties on calm")
    print("  " + "-" * 50)
    bursty_half = pm[pm["burstiness"] >= pm["burstiness"].median()]
    calm_half   = pm[pm["burstiness"] <  pm["burstiness"].median()]
    if (bursty_half["sla_gap"] > 0).all() and (calm_half["sla_gap"] <= 0).all():
        print("  SLA:  hypothesis HOLDS")
    elif (bursty_half["sla_gap"] > 0).any() or (calm_half["sla_gap"] <= 0).any():
        print("  SLA:  hypothesis PARTIALLY holds")
    else:
        print("  SLA:  hypothesis FAILS")

    if (bursty_half["cost_gap"] > 0).all() and (calm_half["cost_gap"] <= 0).all():
        print("  Cost: hypothesis HOLDS")
    elif (bursty_half["cost_gap"] > 0).any() or (calm_half["cost_gap"] <= 0).any():
        print("  Cost: hypothesis PARTIALLY holds")
    else:
        print("  Cost: hypothesis FAILS")
    print()


def _print_v3_vs_v4(df_v4: pd.DataFrame, v3_path: str) -> None:
    if not os.path.exists(v3_path):
        print(f"\n  (v3 CSV not found at {v3_path} — skipping direct comparison)")
        return

    df_v3 = pd.read_csv(v3_path)
    print("\n" + "=" * 92)
    print("  v3 (DEMAND_SCALE=20) vs v4 (calibrated) — mean across seeds 42-46")
    print("=" * 92)
    hdr = (f"  {'Machine':<14} {'v3 Scale':>9} {'v4 Scale':>9} "
           f"{'v3 LSTM SLA%':>14} {'v4 LSTM SLA%':>14} "
           f"{'v3 React SLA%':>15} {'v4 React SLA%':>15}")
    print(hdr)
    print("  " + "-" * 92)

    for mid in [m["machine_id"] for m in MACHINES]:
        g3 = df_v3[df_v3["machine_id"] == mid]
        g4 = df_v4[df_v4["machine_id"] == mid]
        if g3.empty:
            continue
        v3_scale = 20.0
        if g4.empty:
            # infeasible — skipped
            v4_scale = g4["demand_scale"].iloc[0] if not g4.empty else float("nan")
            print(f"  {mid:<14} {v3_scale:>9.2f} {'(skip)':>9} "
                  f"{'---':>14} {'INFEASIBLE':>14} {'---':>15} {'INFEASIBLE':>15}")
            continue
        v4_scale = float(g4["demand_scale"].iloc[0])
        v3_ls = g3["lstm_sla_violation_rate_pct"].mean()
        v4_ls = g4["lstm_sla_violation_rate_pct"].mean()
        v3_rs = g3["reactive_sla_violation_rate_pct"].mean()
        v4_rs = g4["reactive_sla_violation_rate_pct"].mean()
        print(f"  {mid:<14} {v3_scale:>9.2f} {v4_scale:>9.2f} "
              f"{v3_ls:>14.4f} {v4_ls:>14.4f} "
              f"{v3_rs:>15.4f} {v4_rs:>15.4f}")
    print()


def main() -> None:
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"Loading {NROWS_SELECT:,} rows ...")
    df_raw = pd.read_csv(
        DATA_PATH,
        nrows=NROWS_SELECT,
        usecols=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"],
    )
    print(f"Loaded: {len(df_raw):,} rows")

    # ── 2. Compute per-machine demand scales and feasibility ──────────────────
    print("\nComputing per-machine demand scales and feasibility ...")
    feas_map   = {}
    ts_cache   = {}

    for m in MACHINES:
        mid = m["machine_id"]
        ts  = _prepare_timeseries(df_raw, mid)
        ts_cache[mid] = ts
        ds  = calibrate_demand_scale(float(ts[FEATURE_COL].mean()))
        feas = check_feasibility(ts, ds)
        feas["demand_scale"] = round(ds, 4)
        feas_map[mid] = feas

    _print_feasibility_table(feas_map)

    # ── 3. Resume check ───────────────────────────────────────────────────────
    already_done: set = set()
    if os.path.exists(RESULTS_CSV):
        existing = pd.read_csv(RESULTS_CSV)
        already_done = set(zip(existing["machine_id"], existing["seed"].astype(int)))
        print(f"\nExisting results: {len(already_done)} (machine, seed) pairs already done.")

    # ── 4. Per-machine pipeline ───────────────────────────────────────────────
    infeasible_machines = []

    for m_info in MACHINES:
        mid        = m_info["machine_id"]
        burstiness = float(m_info["burstiness_score"])
        cluster    = int(m_info["cluster"])
        is_st      = bool(m_info["is_stress_test"])
        feas       = feas_map[mid]
        ds         = feas["demand_scale"]
        label      = "stress-test" if is_st else f"cluster {cluster}"

        print(f"\n{'='*66}")
        print(f"  {mid}  [{label}]  burstiness={burstiness:.4f}  demand_scale={ds:.2f}x")
        print(f"{'='*66}")

        if feas["infeasible"]:
            print(f"  SKIPPING — infeasible: {feas['reason']}")
            infeasible_machines.append({"machine_id": mid, "reason": feas["reason"]})
            continue

        print("  Tuning on this machine's val split (seed=42) ...")
        best = tune_on_validation(seed=42, machine_id=mid, df_raw=df_raw, demand_scale=ds)
        print(f"  Reactive best: up={best['reactive_up']} down={best['reactive_down']}  "
              f"val_cost={best['reactive_val_cost']:.4f}")
        print(f"  LSTM dec best: upw={best['lstm_under_prov_weight']} "
              f"sm={best['lstm_safety_margin']}  "
              f"val_cost={best['lstm_val_cost']:.4f}")

        for seed in SEEDS:
            if (mid, seed) in already_done:
                print(f"  Seed {seed}: already in CSV, skipping.")
                continue
            result = run_single_experiment(
                seed,
                under_prov_weight=best["lstm_under_prov_weight"],
                safety_margin=best["lstm_safety_margin"],
                reactive_up=best["reactive_up"],
                reactive_down=best["reactive_down"],
                machine_id=mid,
                df_raw=df_raw,
                demand_scale=ds,
            )
            row = {
                "machine_id":                   mid,
                "burstiness_score":             burstiness,
                "cluster":                      cluster,
                "is_stress_test":               int(is_st),
                "demand_scale":                 ds,
                "demand_mean_pct":              feas["demand_mean"],
                "demand_p95_pct":               feas["demand_p95"],
                "demand_p99_pct":               feas["demand_p99"],
                "reactive_up_tuned":            best["reactive_up"],
                "reactive_down_tuned":          best["reactive_down"],
                "lstm_under_prov_weight_tuned": best["lstm_under_prov_weight"],
                "lstm_safety_margin_tuned":     best["lstm_safety_margin"],
                **result,
            }
            _append_row(row)
            print(
                f"  Seed {seed}: LSTM SLA={result['lstm_sla_violation_rate_pct']:.4f}%  "
                f"React SLA={result['reactive_sla_violation_rate_pct']:.4f}%  "
                f"LSTM cost={result['lstm_cost_score']:.4f}  "
                f"React cost={result['reactive_cost_score']:.4f}  "
                f"epochs={result['epochs_trained']}  "
                f"time={result['wall_clock_secs']:.0f}s"
            )

    # ── 5. Summary ────────────────────────────────────────────────────────────
    if infeasible_machines:
        print("\n" + "=" * 60)
        print("  INFEASIBLE MACHINES (skipped, both policies fail structurally)")
        for im in infeasible_machines:
            print(f"    {im['machine_id']}: {im['reason']}")
        print("=" * 60)

    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
        if not df.empty:
            _print_summary(df)
            v3_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "results_v3_multimachine.csv")
            _print_v3_vs_v4(df, v3_path)
    else:
        print("\nNo feasible machines produced results.")


if __name__ == "__main__":
    main()
