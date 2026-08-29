"""
Run the LSTM autoscaler pipeline for seeds [42, 43, 44, 45, 46].

Each seed trains the model fully from scratch. One row per run is appended
to experiments/results.csv (header written on first run). After all 5 runs,
a summary table of mean ± std is printed for every metric.
"""

import csv
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.autoscaler import run_single_experiment

SEEDS = [42, 43, 44, 45, 46]

RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")

METRIC_COLS = [
    "seed",
    "lstm_forecast_rmse",
    "lstm_forecast_mae",
    "naive_baseline_rmse",
    "lstm_sla_violation_rate_pct",
    "reactive_sla_violation_rate_pct",
    "lstm_over_prov_waste_pct",
    "reactive_over_prov_waste_pct",
    "lstm_cost_score",
    "reactive_cost_score",
    "epochs_trained",
    "wall_clock_secs",
]


def _append_row(row: dict) -> None:
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row[k] for k in METRIC_COLS})


def _print_summary(df: pd.DataFrame) -> None:
    numeric = [c for c in METRIC_COLS if c != "seed"]
    means = df[numeric].mean()
    stds = df[numeric].std()

    print("\n" + "=" * 74)
    print(f"  MULTI-SEED SUMMARY  (seeds: {', '.join(str(s) for s in SEEDS)})")
    print("=" * 74)
    print(f"  {'Metric':<42} {'Mean':>10}  {'Std':>10}")
    print("  " + "-" * 66)
    for col in numeric:
        print(f"  {col:<42} {means[col]:>10.4f}  {stds[col]:>10.4f}")
    print("=" * 74)

    lstm_sla_mean = means["lstm_sla_violation_rate_pct"]
    lstm_sla_std = stds["lstm_sla_violation_rate_pct"]
    react_sla_mean = means["reactive_sla_violation_rate_pct"]
    react_sla_std = stds["reactive_sla_violation_rate_pct"]
    lstm_cost_mean = means["lstm_cost_score"]
    lstm_cost_std = stds["lstm_cost_score"]
    react_cost_mean = means["reactive_cost_score"]
    react_cost_std = stds["reactive_cost_score"]

    sla_gap = react_sla_mean - lstm_sla_mean
    cost_gap = react_cost_mean - lstm_cost_mean

    print("\n  VERDICT")
    print("  -------")
    print(
        f"  SLA Violations : LSTM {lstm_sla_mean:.4f}% ± {lstm_sla_std:.4f}%"
        f"  vs  Reactive {react_sla_mean:.4f}% ± {react_sla_std:.4f}%"
    )
    if sla_gap > lstm_sla_std + react_sla_std:
        print(
            "  -> LSTM SLA advantage (gap larger than combined ±1σ): the difference is real."
        )
    elif sla_gap > 0:
        print(
            "  -> LSTM has lower mean SLA violations, but the gap is within one combined σ"
            " -- advantage is not firmly established."
        )
    else:
        print("  -> No SLA advantage for LSTM (Reactive is equal or better on average).")

    print(
        f"  Cost Score     : LSTM {lstm_cost_mean:.4f} ± {lstm_cost_std:.4f}"
        f"  vs  Reactive {react_cost_mean:.4f} ± {react_cost_std:.4f}"
    )
    if cost_gap > lstm_cost_std + react_cost_std:
        print(
            "  -> LSTM cost advantage (gap larger than combined ±1σ): the difference is real."
        )
    elif cost_gap > 0:
        print(
            "  -> LSTM has lower mean cost score, but the gap is within one combined σ."
        )
    else:
        print("  -> No cost advantage for LSTM.")
    print()


def main() -> None:
    already_done: set = set()
    if os.path.exists(RESULTS_CSV):
        existing = pd.read_csv(RESULTS_CSV)
        already_done = set(existing["seed"].astype(int))
        print(f"Existing results found for seeds: {sorted(already_done)}")

    for seed in SEEDS:
        if seed in already_done:
            print(f"Seed {seed}: already in results.csv, skipping.")
            continue
        print(f"\n{'='*52}\nSeed {seed} — training from scratch ...\n{'='*52}")
        result = run_single_experiment(seed)
        _append_row(result)
        print(
            f"  LSTM SLA={result['lstm_sla_violation_rate_pct']:.4f}%  "
            f"Reactive SLA={result['reactive_sla_violation_rate_pct']:.4f}%  "
            f"LSTM cost={result['lstm_cost_score']:.4f}  "
            f"Reactive cost={result['reactive_cost_score']:.4f}  "
            f"epochs={result['epochs_trained']}  "
            f"time={result['wall_clock_secs']:.0f}s"
        )

    df = pd.read_csv(RESULTS_CSV)
    _print_summary(df)


if __name__ == "__main__":
    main()
