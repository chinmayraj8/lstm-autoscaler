"""
Step 4: multi-machine burstiness experiment.

Workflow
--------
1. Load 5 M rows; compute per-machine stats (mean, std, burstiness_p95) for
   machines with >= 2000 raw rows and >= 200 resampled 5-min points.
2. K-means (k=4) on standardized (mean, std, burstiness); pick the machine
   closest to each centroid.  Also always include the single highest-
   burstiness qualifying machine as a labelled stress-test case.
3. For each selected machine: tune_on_validation(seed=42) on that machine's
   own val split, then run_single_experiment for seeds [42-46] on its test
   split.  Policy params can differ per machine.
4. Save all results to experiments/results_v3_multimachine.csv.
5. Print per-machine summary and test whether LSTM relative performance
   correlates with burstiness.
"""

import csv
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.autoscaler import (
    DATA_PATH,
    _prepare_timeseries,
    run_single_experiment,
    tune_on_validation,
)

SEEDS = [42, 43, 44, 45, 46]
NROWS_SELECT = 5_000_000
MIN_RAW_ROWS = 2000   # minimum raw rows in the 5 M sample to qualify
MIN_TS_LEN   = 200    # minimum 5-min resampled points to run the ML pipeline
K = 4

RESULTS_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results_v3_multimachine.csv",
)

METRIC_COLS = [
    "machine_id", "burstiness_score", "cluster", "is_stress_test",
    "reactive_up_tuned", "reactive_down_tuned",
    "lstm_under_prov_weight_tuned", "lstm_safety_margin_tuned",
    "seed",
    "lstm_forecast_rmse", "lstm_forecast_mae", "naive_baseline_rmse",
    "lstm_sla_violation_rate_pct", "reactive_sla_violation_rate_pct",
    "lstm_over_prov_waste_pct", "reactive_over_prov_waste_pct",
    "lstm_cost_score", "reactive_cost_score",
    "epochs_trained", "wall_clock_secs",
]


# ── Machine-selection helpers ─────────────────────────────────────────────────

def _compute_machine_stats(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Return per-machine (mean, std, burstiness_p95) for qualifying machines."""
    counts = df_raw["machine_id"].value_counts()
    qualifying_ids = counts[counts >= MIN_RAW_ROWS].index.tolist()
    print(f"  Machines with >= {MIN_RAW_ROWS} raw rows: {len(qualifying_ids)}")

    rows = []
    for mid in qualifying_ids:
        mdf = df_raw[df_raw["machine_id"] == mid]
        ts  = _prepare_timeseries(mdf, mid)
        if len(ts) < MIN_TS_LEN:
            continue
        cpu    = ts["cpu_util_percent"].values
        diffs  = np.abs(np.diff(cpu))
        burst  = float(np.percentile(diffs, 95)) if len(diffs) >= 20 else 0.0
        rows.append({
            "machine_id":  mid,
            "mean":        float(cpu.mean()),
            "std":         float(cpu.std()),
            "burstiness":  burst,
            "n_raw_rows":  int(counts[mid]),
            "n_ts_points": int(len(ts)),
        })

    return pd.DataFrame(rows)


def _select_machines(stats_df: pd.DataFrame, k: int):
    """Return a list of dicts describing the selected machines.

    Picks the centroid-closest machine per k-means cluster, then always
    appends the highest-burstiness machine (labelled stress-test) if it is
    not already in the centroid selection.
    """
    X       = stats_df[["mean", "std", "burstiness"]].values
    X_std   = StandardScaler().fit_transform(X)
    km      = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels  = km.fit_predict(X_std)

    stats_df = stats_df.copy().reset_index(drop=True)
    stats_df["cluster"]       = labels
    stats_df["is_stress_test"] = False

    selected = []
    for c in range(k):
        mask       = stats_df["cluster"] == c
        cluster_df = stats_df[mask].reset_index(drop=True)
        cluster_X  = X_std[mask]
        dists      = np.linalg.norm(cluster_X - km.cluster_centers_[c], axis=1)
        selected.append(cluster_df.iloc[dists.argmin()].to_dict())

    selected_ids = {m["machine_id"] for m in selected}

    top_row = stats_df.sort_values("burstiness", ascending=False).iloc[0].to_dict()
    if top_row["machine_id"] not in selected_ids:
        top_row["is_stress_test"] = True
        selected.append(top_row)
        print(f"  Stress-test added: {top_row['machine_id']}  "
              f"burstiness={top_row['burstiness']:.4f}  "
              f"(not in k-means centroid selection)")
    else:
        print(f"  Highest-burstiness machine {top_row['machine_id']} "
              f"already in k-means selection")

    return selected, stats_df


# ── CSV helper ────────────────────────────────────────────────────────────────

def _append_row(row: dict) -> None:
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow({col: row[col] for col in METRIC_COLS})


# ── Summary printer ───────────────────────────────────────────────────────────

def _verdict(gap: float, combined_sigma: float) -> str:
    if gap > combined_sigma:
        return "LSTM wins (confirmed, gap > combined ±1σ)"
    elif gap > 0:
        return "LSTM directional advantage (within ±1σ)"
    elif abs(gap) <= combined_sigma:
        return "tied (within ±1σ)"
    else:
        return "Reactive wins"


def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 82)
    print("  PER-MACHINE TEST RESULTS  (seeds 42-46, test split only)")
    print("=" * 82)

    pm_rows = []
    for mid, grp in df.groupby("machine_id", sort=False):
        burst    = float(grp["burstiness_score"].iloc[0])
        cluster  = int(grp["cluster"].iloc[0])
        is_st    = bool(grp["is_stress_test"].iloc[0])
        label    = "stress-test" if is_st else f"cluster {cluster}"

        up   = grp["reactive_up_tuned"].iloc[0]
        down = grp["reactive_down_tuned"].iloc[0]
        upw  = grp["lstm_under_prov_weight_tuned"].iloc[0]
        sm   = grp["lstm_safety_margin_tuned"].iloc[0]

        ls_mean = grp["lstm_sla_violation_rate_pct"].mean()
        ls_std  = grp["lstm_sla_violation_rate_pct"].std()
        rs_mean = grp["reactive_sla_violation_rate_pct"].mean()
        rs_std  = grp["reactive_sla_violation_rate_pct"].std()
        lc_mean = grp["lstm_cost_score"].mean()
        lc_std  = grp["lstm_cost_score"].std()
        rc_mean = grp["reactive_cost_score"].mean()
        rc_std  = grp["reactive_cost_score"].std()

        sla_gap  = rs_mean - ls_mean   # positive = LSTM better on SLA
        cost_gap = rc_mean - lc_mean   # positive = LSTM better on cost

        print(f"\n  {mid}  [{label}]  burstiness={burst:.4f}")
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

    # Burstiness correlation table
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
    bursty_sla_advantage  = (bursty_half["sla_gap"]  > 0).all()
    bursty_cost_advantage = (bursty_half["cost_gap"] > 0).all()
    calm_no_sla_advantage  = (calm_half["sla_gap"]  <= 0).all()
    calm_no_cost_advantage = (calm_half["cost_gap"] <= 0).all()

    if bursty_sla_advantage and calm_no_sla_advantage:
        print("  SLA:  hypothesis HOLDS (LSTM wins on all bursty, loses/ties on all calm)")
    elif bursty_sla_advantage or calm_no_sla_advantage:
        print("  SLA:  hypothesis PARTIALLY holds")
    else:
        print("  SLA:  hypothesis FAILS")

    if bursty_cost_advantage and calm_no_cost_advantage:
        print("  Cost: hypothesis HOLDS (LSTM wins on all bursty, loses/ties on all calm)")
    elif bursty_cost_advantage or calm_no_cost_advantage:
        print("  Cost: hypothesis PARTIALLY holds")
    else:
        print("  Cost: hypothesis FAILS")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Load data and compute stats ────────────────────────────────────────
    print(f"Loading {NROWS_SELECT:,} rows for machine selection ...")
    df_raw = pd.read_csv(
        DATA_PATH,
        nrows=NROWS_SELECT,
        usecols=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"],
    )
    print(f"Loaded: {len(df_raw):,} rows | {df_raw['machine_id'].nunique()} unique machines")

    print(f"\nComputing per-machine stats ...")
    stats_df = _compute_machine_stats(df_raw)
    print(f"Machines with >= {MIN_TS_LEN} resampled points: {len(stats_df)}")
    if len(stats_df) == 0:
        print("No qualifying machines. Exiting.")
        return

    # ── 2. K-means selection ──────────────────────────────────────────────────
    k_actual = min(K, len(stats_df))
    print(f"\nK-means (k={k_actual}) on standardized (mean, std, burstiness) ...")
    selected, stats_df = _select_machines(stats_df, k_actual)

    print(f"\n{'='*72}")
    print("  SELECTED MACHINES  (sorted by burstiness)")
    print(f"{'='*72}")
    hdr = f"  {'Machine':<14} {'Mean':>8} {'Std':>8} {'Burstiness':>12} {'N_ts':>7}  Label"
    print(hdr)
    print("  " + "-" * 60)
    for m in sorted(selected, key=lambda x: x["burstiness"]):
        lbl = "stress-test" if m["is_stress_test"] else f"cluster {int(m['cluster'])}"
        print(f"  {m['machine_id']:<14} {m['mean']:>8.4f} {m['std']:>8.4f} "
              f"{m['burstiness']:>12.4f} {int(m['n_ts_points']):>7}  {lbl}")
    print(f"{'='*72}")

    # ── 3. Per-machine pipeline ────────────────────────────────────────────────
    already_done: set = set()
    if os.path.exists(RESULTS_CSV):
        existing = pd.read_csv(RESULTS_CSV)
        already_done = set(zip(existing["machine_id"], existing["seed"].astype(int)))
        print(f"\nExisting results: {len(already_done)} (machine, seed) pairs already done.")

    for m_info in selected:
        mid        = m_info["machine_id"]
        burstiness = float(m_info["burstiness"])
        cluster    = int(m_info["cluster"])
        is_st      = bool(m_info["is_stress_test"])

        print(f"\n{'='*62}")
        print(f"  {mid}  burstiness={burstiness:.4f}  "
              f"{'[stress-test]' if is_st else f'[cluster {cluster}]'}")
        print(f"{'='*62}")

        print("  Tuning on this machine's val split (seed=42) ...")
        best = tune_on_validation(seed=42, machine_id=mid, df_raw=df_raw)
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
            )
            row = {
                "machine_id":                   mid,
                "burstiness_score":             burstiness,
                "cluster":                      cluster,
                "is_stress_test":               int(is_st),
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

    # ── 4. Summary ────────────────────────────────────────────────────────────
    df = pd.read_csv(RESULTS_CSV)
    _print_summary(df)


if __name__ == "__main__":
    main()
