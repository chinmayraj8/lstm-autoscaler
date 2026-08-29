"""
Step 8: expanded multi-machine experiment (15-20 machines).

Workflow
--------
1. Load 5 M rows (same as Steps 4-5).
2. Compute per-machine stats (mean, std, burstiness_p95) for machines with
   >= 2000 raw rows and >= 200 resampled 5-min points (same qualifying
   criteria as Step 4, reusing `run_multimachine._compute_machine_stats`).
3. K-means (k=4) on standardized (mean, std, burstiness) -- same clustering
   as Step 4. Instead of picking only the single centroid-closest machine
   per cluster, sample the 3-4 machines closest to each centroid (falls
   back to fewer only if a cluster has < 3 qualifying machines). Always add
   the single highest-burstiness qualifying machine as a labelled
   stress-test case (Step 4's deliberate pick), if not already selected.
4. For each selected machine, compute a per-machine demand_scale via
   calibrate_demand_scale(); run check_feasibility(); skip infeasible
   machines (p99 calibrated demand > max fleet capacity) with a printed
   reason, same as Step 5.
5. For each feasible machine: tune_on_validation(seed=42) on its own val
   split, then run_single_experiment for seeds [42-46] on its test split.
   Save to experiments/results_v5_expanded_multimachine.csv.
6. Print:
   - Selected-machines table (cluster + rank within cluster)
   - Feasibility table
   - Per-machine comparison (feasible machines only)
   - Burstiness vs LSTM advantage correlation
   - Pattern-generalization classification: across all feasible machines,
     how many show a confirmed LSTM cost advantage (gap > combined ±1σ) at
     tied/near-tied SLA (the m_2189 pattern), how many show a confirmed
     LSTM cost advantage despite Reactive winning SLA (the m_2065 pattern),
     and how many show Reactive winning both metrics outright. This is the
     question Step 8 exists to answer: did Step 5's two cost-advantage
     machines generalize, or were they two lucky picks out of five?
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
    DEC_MAX_SERVERS,
    FEATURE_COL,
    SIM_SERVER_CAPACITY,
    _prepare_timeseries,
    calibrate_demand_scale,
    check_feasibility,
    run_single_experiment,
    tune_on_validation,
)
from experiments.run_multimachine import _compute_machine_stats

SEEDS = [42, 43, 44, 45, 46]
NROWS_SELECT = 5_000_000
K = 4
N_PER_CLUSTER_MAX = 4
N_PER_CLUSTER_MIN = 3

RESULTS_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results_v5_expanded_multimachine.csv",
)

METRIC_COLS = [
    "machine_id", "burstiness_score", "cluster", "cluster_rank", "is_stress_test",
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


# ── Machine selection: K-means (k=4) + N-per-cluster sampling ────────────────

def _select_representative_machines(stats_df: pd.DataFrame, k: int):
    """Generalizes Step 4's 1-per-cluster centroid pick to 3-4-per-cluster.

    For each cluster, ranks members by distance to the centroid and keeps
    the closest N_PER_CLUSTER_MAX (or fewer, down to all members, if the
    cluster has fewer than N_PER_CLUSTER_MIN qualifying machines). Always
    adds the single highest-burstiness qualifying machine as a labelled
    stress-test case if it isn't already selected.
    """
    X     = stats_df[["mean", "std", "burstiness"]].values
    X_std = StandardScaler().fit_transform(X)
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_std)

    stats_df = stats_df.copy().reset_index(drop=True)
    stats_df["cluster"] = labels
    stats_df["is_stress_test"] = False

    selected = []
    for c in range(k):
        mask       = (stats_df["cluster"] == c).values
        cluster_df = stats_df[mask].reset_index(drop=True)
        cluster_X  = X_std[mask]
        dists      = np.linalg.norm(cluster_X - km.cluster_centers_[c], axis=1)
        order      = np.argsort(dists)

        n_avail = len(cluster_df)
        if n_avail < N_PER_CLUSTER_MIN:
            n_pick = n_avail
            print(f"  Cluster {c}: only {n_avail} qualifying machine(s) "
                  f"(< {N_PER_CLUSTER_MIN}) -- taking all.")
        else:
            n_pick = min(N_PER_CLUSTER_MAX, n_avail)

        for rank, idx in enumerate(order[:n_pick]):
            row = cluster_df.iloc[idx].to_dict()
            row["cluster_rank"] = rank   # 0 = centroid-closest
            selected.append(row)

    selected_ids = {m["machine_id"] for m in selected}

    top_row = stats_df.sort_values("burstiness", ascending=False).iloc[0].to_dict()
    if top_row["machine_id"] not in selected_ids:
        top_row["is_stress_test"] = True
        top_row["cluster_rank"]   = -1
        selected.append(top_row)
        print(f"  Stress-test added: {top_row['machine_id']}  "
              f"burstiness={top_row['burstiness']:.4f}  "
              f"(not in per-cluster selection)")
    else:
        for m in selected:
            if m["machine_id"] == top_row["machine_id"]:
                m["is_stress_test"] = True
        print(f"  Highest-burstiness machine {top_row['machine_id']} "
              f"already in per-cluster selection (flagged as stress-test too)")

    return selected, stats_df


# ── CSV helper ────────────────────────────────────────────────────────────────

def _append_row(row: dict) -> None:
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow({col: row[col] for col in METRIC_COLS})


# ── Verdict + printers ────────────────────────────────────────────────────────

def _verdict(gap: float, combined_sigma: float) -> str:
    if gap > combined_sigma:
        return "LSTM wins (confirmed, gap > combined ±1σ)"
    elif gap > 0:
        return "LSTM directional advantage (within ±1σ)"
    elif abs(gap) <= combined_sigma:
        return "tied (within ±1σ)"
    else:
        return "Reactive wins"


def _print_selected_machines(selected: list) -> None:
    print(f"\n{'='*78}")
    print(f"  SELECTED MACHINES ({len(selected)} total)")
    print(f"{'='*78}")
    hdr = (f"  {'Machine':<12} {'Cluster':>7} {'Rank':>5} {'Mean':>8} {'Std':>8} "
           f"{'Burstiness':>11}  Label")
    print(hdr)
    print("  " + "-" * 66)
    for m in sorted(selected, key=lambda x: (x["cluster"], x["cluster_rank"])):
        lbl = "stress-test" if m["is_stress_test"] else ""
        print(f"  {m['machine_id']:<12} {int(m['cluster']):>7} {int(m['cluster_rank']):>5} "
              f"{m['mean']:>8.4f} {m['std']:>8.4f} {m['burstiness_score']:>11.4f}  {lbl}")
    print(f"{'='*78}")


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
    print("  PER-MACHINE TEST RESULTS  (v5 expanded multimachine, seeds 42-46)")
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
    print()


def _print_pattern_generalization(df: pd.DataFrame) -> pd.DataFrame:
    """Answers the question Step 8 exists to answer.

    Buckets every feasible machine into:
      (A) confirmed LSTM cost advantage + tied/near-tied SLA   (m_2189-style)
      (B) confirmed LSTM cost advantage + Reactive wins SLA    (m_2065-style)
      (C) Reactive wins BOTH metrics outright
      (D) everything else (e.g. LSTM wins/ties cost without confirming it,
          or LSTM wins SLA too)
    """
    print("\n" + "=" * 88)
    print("  PATTERN GENERALIZATION: does the m_2189 / m_2065 cost-advantage pattern hold?")
    print("=" * 88)

    rows = []
    for mid, grp in df.groupby("machine_id", sort=False):
        ls_mean, ls_std = grp["lstm_sla_violation_rate_pct"].mean(), grp["lstm_sla_violation_rate_pct"].std()
        rs_mean, rs_std = grp["reactive_sla_violation_rate_pct"].mean(), grp["reactive_sla_violation_rate_pct"].std()
        lc_mean, lc_std = grp["lstm_cost_score"].mean(), grp["lstm_cost_score"].std()
        rc_mean, rc_std = grp["reactive_cost_score"].mean(), grp["reactive_cost_score"].std()

        sla_gap  = rs_mean - ls_mean
        cost_gap = rc_mean - lc_mean
        sla_verdict  = _verdict(sla_gap,  (ls_std or 0) + (rs_std or 0))
        cost_verdict = _verdict(cost_gap, (lc_std or 0) + (rc_std or 0))

        rows.append({
            "machine_id":   mid,
            "burstiness":   float(grp["burstiness_score"].iloc[0]),
            "sla_gap":      sla_gap,
            "cost_gap":     cost_gap,
            "sla_verdict":  sla_verdict,
            "cost_verdict": cost_verdict,
        })

    pm = pd.DataFrame(rows)

    cost_confirmed    = pm["cost_verdict"].str.startswith("LSTM wins")
    sla_reactive_wins = pm["sla_verdict"] == "Reactive wins"

    bucket_a = pm[cost_confirmed & ~sla_reactive_wins]
    bucket_b = pm[cost_confirmed & sla_reactive_wins]
    bucket_c = pm[(pm["cost_verdict"] == "Reactive wins") & sla_reactive_wins]
    used_idx = set(bucket_a.index) | set(bucket_b.index) | set(bucket_c.index)
    bucket_d = pm[~pm.index.isin(used_idx)]

    print(f"\n  Feasible machines evaluated: {len(pm)}")

    print(f"\n  (A) Confirmed LSTM cost advantage + tied/near-tied SLA "
          f"(m_2189-style): {len(bucket_a)}")
    for _, r in bucket_a.iterrows():
        print(f"      {r['machine_id']:<10} burstiness={r['burstiness']:>8.4f}  "
              f"cost_gap={r['cost_gap']:+.4f}  sla_gap={r['sla_gap']:+.4f}  ({r['sla_verdict']})")

    print(f"\n  (B) Confirmed LSTM cost advantage + Reactive wins SLA "
          f"(m_2065-style): {len(bucket_b)}")
    for _, r in bucket_b.iterrows():
        print(f"      {r['machine_id']:<10} burstiness={r['burstiness']:>8.4f}  "
              f"cost_gap={r['cost_gap']:+.4f}  sla_gap={r['sla_gap']:+.4f}")

    print(f"\n  (C) Reactive wins BOTH metrics outright: {len(bucket_c)}")
    for _, r in bucket_c.iterrows():
        print(f"      {r['machine_id']:<10} burstiness={r['burstiness']:>8.4f}  "
              f"cost_gap={r['cost_gap']:+.4f}  sla_gap={r['sla_gap']:+.4f}")

    if len(bucket_d):
        print(f"\n  (D) Other: {len(bucket_d)}")
        for _, r in bucket_d.iterrows():
            print(f"      {r['machine_id']:<10} burstiness={r['burstiness']:>8.4f}  "
                  f"cost_gap={r['cost_gap']:+.4f} ({r['cost_verdict']})  "
                  f"sla_gap={r['sla_gap']:+.4f} ({r['sla_verdict']})")

    n_cost_adv_total = len(bucket_a) + len(bucket_b)
    print("\n  " + "-" * 70)
    print(f"  TOTAL confirmed LSTM cost advantage (any SLA outcome): "
          f"{n_cost_adv_total} / {len(pm)}")
    print(f"  TOTAL Reactive wins both metrics outright: "
          f"{len(bucket_c)} / {len(pm)}")
    print("=" * 88 + "\n")

    return pm


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"Loading {NROWS_SELECT:,} rows ...")
    df_raw = pd.read_csv(
        DATA_PATH,
        nrows=NROWS_SELECT,
        usecols=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"],
    )
    print(f"Loaded: {len(df_raw):,} rows | {df_raw['machine_id'].nunique()} unique machines")

    # ── 2. Machine selection: K-means (k=4) + 3-4 per cluster + stress-test ───
    print("\nComputing per-machine stats (mean, std, burstiness) ...")
    stats_df = _compute_machine_stats(df_raw)
    print(f"Machines with >= 200 resampled points: {len(stats_df)}")
    if len(stats_df) == 0:
        print("No qualifying machines. Exiting.")
        return

    k_actual = min(K, len(stats_df))
    print(f"\nK-means (k={k_actual}) + {N_PER_CLUSTER_MIN}-{N_PER_CLUSTER_MAX} "
          f"representative machines per cluster ...")
    selected, stats_df = _select_representative_machines(stats_df, k_actual)

    # normalize field name: stats_df/selected uses "burstiness"; downstream
    # (CSV columns, summary printers) uses "burstiness_score" for consistency
    # with Steps 4-5.
    for m in selected:
        m["burstiness_score"] = m.pop("burstiness")

    _print_selected_machines(selected)

    # ── 3. Per-machine demand scale + feasibility ─────────────────────────────
    print("\nComputing per-machine demand scales and feasibility ...")
    feas_map = {}
    for m in selected:
        mid = m["machine_id"]
        ts  = _prepare_timeseries(df_raw, mid)
        ds  = calibrate_demand_scale(float(ts[FEATURE_COL].mean()))
        feas = check_feasibility(ts, ds)
        feas["demand_scale"] = round(ds, 4)
        feas_map[mid] = feas

    _print_feasibility_table(feas_map)

    # ── 4. Resume check ────────────────────────────────────────────────────────
    already_done: set = set()
    if os.path.exists(RESULTS_CSV):
        existing = pd.read_csv(RESULTS_CSV)
        already_done = set(zip(existing["machine_id"], existing["seed"].astype(int)))
        print(f"\nExisting results: {len(already_done)} (machine, seed) pairs already done.")

    # ── 5. Per-machine pipeline ────────────────────────────────────────────────
    infeasible_machines = []

    for m_info in selected:
        mid          = m_info["machine_id"]
        burstiness   = float(m_info["burstiness_score"])
        cluster      = int(m_info["cluster"])
        cluster_rank = int(m_info["cluster_rank"])
        is_st        = bool(m_info["is_stress_test"])
        feas         = feas_map[mid]
        ds           = feas["demand_scale"]
        label        = "stress-test" if is_st else f"cluster {cluster} (rank {cluster_rank})"

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
                "cluster_rank":                 cluster_rank,
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

    # ── 6. Summary ────────────────────────────────────────────────────────────
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
            _print_pattern_generalization(df)
    else:
        print("\nNo feasible machines produced results.")


if __name__ == "__main__":
    main()
