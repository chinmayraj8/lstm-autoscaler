"""
Step 3: symmetric-tuning multi-seed experiment.

Workflow
--------
1. tune_on_validation(seed=42): grid-search both policies on the VAL split,
   never touching test data.
2. run_single_experiment(seed, **best_params): evaluate both policies on the
   TEST split with those frozen params, for seeds [42..46].
3. Append one row per seed to experiments/results_v2_symmetric_tuning.csv.
4. Print mean ± std summary and a plain-English verdict.
"""

import csv
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.autoscaler import (
    DEC_UNDER_WEIGHT, REACTIVE_DOWN_THRESHOLD, REACTIVE_UP_THRESHOLD,
    SAFETY_MARGIN, run_single_experiment, tune_on_validation,
)

SEEDS = [42, 43, 44, 45, 46]

RESULTS_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results_v2_symmetric_tuning.csv",
)

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


def _print_tuning_report(best: dict) -> None:
    print("\n" + "=" * 70)
    print("  TUNING RESULTS (validation split, seed=42)")
    print("=" * 70)
    print(f"  Reactive  best up={best['reactive_up']}  "
          f"down={best['reactive_down']}  "
          f"val_cost={best['reactive_val_cost']:.4f}")
    orig_r_changed = (
        best["reactive_up"] != REACTIVE_UP_THRESHOLD
        or best["reactive_down"] != REACTIVE_DOWN_THRESHOLD
    )
    if orig_r_changed:
        print(f"    (original was up={REACTIVE_UP_THRESHOLD}  "
              f"down={REACTIVE_DOWN_THRESHOLD} — changed by tuning)")
    else:
        print(f"    (same as original up={REACTIVE_UP_THRESHOLD}  "
              f"down={REACTIVE_DOWN_THRESHOLD} — tuning confirmed)")

    print(f"  LSTM dec  best under_prov_weight={best['lstm_under_prov_weight']}  "
          f"safety_margin={best['lstm_safety_margin']}  "
          f"val_cost={best['lstm_val_cost']:.4f}")
    orig_l_changed = (
        best["lstm_under_prov_weight"] != DEC_UNDER_WEIGHT
        or best["lstm_safety_margin"] != SAFETY_MARGIN
    )
    if orig_l_changed:
        print(f"    (original was under_prov_weight={DEC_UNDER_WEIGHT}  "
              f"safety_margin={SAFETY_MARGIN} — changed by tuning)")
    else:
        print(f"    (same as original under_prov_weight={DEC_UNDER_WEIGHT}  "
              f"safety_margin={SAFETY_MARGIN} — tuning confirmed)")
    print("=" * 70)


def _print_summary(df: pd.DataFrame, best: dict) -> None:
    numeric = [c for c in METRIC_COLS if c != "seed"]
    means = df[numeric].mean()
    stds  = df[numeric].std()

    print("\n" + "=" * 74)
    print(f"  MULTI-SEED TEST RESULTS  (seeds: {', '.join(str(s) for s in SEEDS)})")
    print(f"  Split: 60/20/20 train/val/test  |  evaluated on TEST only")
    print(f"  Reactive params: up={best['reactive_up']}  down={best['reactive_down']}")
    print(f"  LSTM dec params: under_prov_weight={best['lstm_under_prov_weight']}  "
          f"safety_margin={best['lstm_safety_margin']}")
    print("=" * 74)
    print(f"  {'Metric':<42} {'Mean':>10}  {'Std':>10}")
    print("  " + "-" * 66)
    for col in numeric:
        print(f"  {col:<42} {means[col]:>10.4f}  {stds[col]:>10.4f}")
    print("=" * 74)

    lstm_sla_mean   = means["lstm_sla_violation_rate_pct"]
    lstm_sla_std    = stds["lstm_sla_violation_rate_pct"]
    react_sla_mean  = means["reactive_sla_violation_rate_pct"]
    react_sla_std   = stds["reactive_sla_violation_rate_pct"]
    lstm_cost_mean  = means["lstm_cost_score"]
    lstm_cost_std   = stds["lstm_cost_score"]
    react_cost_mean = means["reactive_cost_score"]
    react_cost_std  = stds["reactive_cost_score"]

    sla_gap  = react_sla_mean - lstm_sla_mean
    cost_gap = react_cost_mean - lstm_cost_mean

    print("\n  VERDICT (after symmetric tuning)")
    print("  ---------------------------------")
    print(f"  SLA Violations : LSTM {lstm_sla_mean:.4f}% ± {lstm_sla_std:.4f}%"
          f"  vs  Reactive {react_sla_mean:.4f}% ± {react_sla_std:.4f}%")
    combined_sla_sigma = lstm_sla_std + react_sla_std
    if sla_gap > combined_sla_sigma:
        print(f"  -> LSTM SLA advantage confirmed (gap {sla_gap:.4f} pp > combined ±1σ {combined_sla_sigma:.4f} pp).")
    elif sla_gap > 0:
        print(f"  -> LSTM has lower mean SLA violations but gap ({sla_gap:.4f} pp) is within"
              f" combined ±1σ ({combined_sla_sigma:.4f} pp) — not firmly established.")
    else:
        print(f"  -> No SLA advantage for LSTM (gap {sla_gap:.4f} pp, Reactive equal or better).")

    print(f"  Cost Score     : LSTM {lstm_cost_mean:.4f} ± {lstm_cost_std:.4f}"
          f"  vs  Reactive {react_cost_mean:.4f} ± {react_cost_std:.4f}")
    combined_cost_sigma = lstm_cost_std + react_cost_std
    if cost_gap > combined_cost_sigma:
        print(f"  -> LSTM cost advantage confirmed (gap {cost_gap:.4f} > combined ±1σ {combined_cost_sigma:.4f}).")
    elif cost_gap > 0:
        print(f"  -> LSTM has lower mean cost but gap ({cost_gap:.4f}) is within"
              f" combined ±1σ ({combined_cost_sigma:.4f}) — not firmly established.")
    else:
        print(f"  -> No cost advantage for LSTM (gap {cost_gap:.4f}, Reactive equal or better).")
    print()


def main() -> None:
    # ── Step 1: tune on validation ────────────────────────────────────────────
    print("Tuning both policies on validation split (seed=42, never touching test)...")
    best = tune_on_validation(seed=42)
    _print_tuning_report(best)

    # Print top-5 Reactive grid results
    reactive_sorted = sorted(best["reactive_grid"], key=lambda x: x[2])
    print("\n  Top-5 Reactive (up, down, val_cost):")
    for up, down, cost in reactive_sorted[:5]:
        marker = " <-- best" if (up == best["reactive_up"] and down == best["reactive_down"]) else ""
        print(f"    up={up:2d}  down={down:2d}  cost={cost:.4f}{marker}")

    lstm_sorted = sorted(best["lstm_grid"], key=lambda x: x[2])
    print("\n  Top-5 LSTM decision-engine (under_prov_weight, safety_margin, val_cost):")
    for upw, sm, cost in lstm_sorted[:5]:
        marker = (
            " <-- best"
            if (upw == best["lstm_under_prov_weight"] and sm == best["lstm_safety_margin"])
            else ""
        )
        print(f"    upw={upw:2d}  sm={sm:.2f}  cost={cost:.4f}{marker}")

    # ── Step 2: run 5 seeds on test ───────────────────────────────────────────
    already_done: set = set()
    if os.path.exists(RESULTS_CSV):
        existing = pd.read_csv(RESULTS_CSV)
        already_done = set(existing["seed"].astype(int))
        print(f"\nExisting test results for seeds: {sorted(already_done)}")

    for seed in SEEDS:
        if seed in already_done:
            print(f"Seed {seed}: already in results_v2, skipping.")
            continue
        print(f"\n{'='*54}\nSeed {seed} — training from scratch, evaluating on TEST ...\n{'='*54}")
        result = run_single_experiment(
            seed,
            under_prov_weight=best["lstm_under_prov_weight"],
            safety_margin=best["lstm_safety_margin"],
            reactive_up=best["reactive_up"],
            reactive_down=best["reactive_down"],
        )
        _append_row(result)
        print(
            f"  LSTM SLA={result['lstm_sla_violation_rate_pct']:.4f}%  "
            f"Reactive SLA={result['reactive_sla_violation_rate_pct']:.4f}%  "
            f"LSTM cost={result['lstm_cost_score']:.4f}  "
            f"Reactive cost={result['reactive_cost_score']:.4f}  "
            f"epochs={result['epochs_trained']}  "
            f"time={result['wall_clock_secs']:.0f}s"
        )

    # ── Step 3: summary ───────────────────────────────────────────────────────
    df = pd.read_csv(RESULTS_CSV)
    _print_summary(df, best)


if __name__ == "__main__":
    main()
