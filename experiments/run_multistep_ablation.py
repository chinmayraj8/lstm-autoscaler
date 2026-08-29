"""
Step 12 (Phase 3): does a multi-step-aware decision engine help the LSTM
specifically, or does it help ARIMA about equally?

decision.py's original engine (`_build_lstm_targets`) collapses each
forecast row to `max(y_pred_real[i])` before penalizing -- a one-tick spike
and a sustained plateau of the same peak height are scored identically.
`_build_multistep_targets` (new, decision.py) instead penalizes every step
of the horizon and averages, so a brief spike scores far below a plateau.
Both target-builders are forecaster-agnostic (same (y_pred_real,
y_actual_real) -> (targets, demand) signature) -- exactly like Step 10 noted
`_build_lstm_targets` already was, which is what lets ARIMA and LSTM both
be run through either engine here.

Given Step 11's finding (ARIMA matches/beats the LSTM's cost on 9 of the
10 machines where Step 8 found an "LSTM cost advantage"), the real question
this script settles: if a smarter decision engine helps, does it help the
LSTM's multi-step forecast MORE than ARIMA's rolling one-step forecast
(because their error shapes across the horizon differ), or does it help
both about equally (in which case Step 11's "any decent forecaster" story
just gets one layer more accurate, not reversed)?

To keep compute proportional to what's actually new, the GREEDY (original
engine) numbers for LSTM, Reactive, and ARIMA are reused unchanged from
experiments/results_v7_arima_full13.csv (Steps 8 + 11) -- not re-tuned or
re-run. Only the MULTISTEP cells are computed fresh, for both forecasters,
on all 13 of Step 8's feasible machines:
  - LSTM multistep: `tune_on_validation(target_builder=_build_multistep_targets)`
    then `run_single_experiment(..., target_builder=_build_multistep_targets)`
    x 5 seeds (LSTM training is genuinely stochastic -- this is real retraining,
    not reused).
  - ARIMA multistep: `tune_arima_on_validation(target_builder=_build_multistep_targets)`
    then `run_arima_experiment(..., target_builder=_build_multistep_targets)`
    once (deterministic; re-run and asserted bit-identical per machine, same
    discipline as Steps 10-11).
Both are tuned on the SAME grids (LSTM_UPW_GRID x LSTM_SM_GRID) the greedy
engine used -- multistep is tuned exactly as hard as greedy, no harder, no
easier, and LSTM and ARIMA are tuned exactly as hard as each other.

Writes experiments/results_v8_multistep_ablation.csv (65 rows: 13 machines
x 5 seeds, greedy columns reused from v7 + new multistep columns) and
experiments/results_v8_ablation_summary.csv (one row per machine: greedy
vs multistep cost/SLA for each forecaster, plus verdicts).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.autoscaler import (
    DATA_PATH,
    _build_multistep_targets,
    run_arima_experiment,
    run_single_experiment,
    tune_arima_on_validation,
    tune_on_validation,
)

SEEDS = [42, 43, 44, 45, 46]

RESULTS_V7 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "results_v7_arima_full13.csv")
RESULTS_V8 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "results_v8_multistep_ablation.csv")
SUMMARY_V8 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "results_v8_ablation_summary.csv")


def _verdict(cost_before: float, cost_after: float, std_before: float, std_after: float) -> str:
    """Is `after` confirmed cheaper than `before`? Same combined-+/-1-sigma
    rule used throughout this project (run_multimachine_v2._verdict)."""
    gap = cost_before - cost_after   # positive => after is cheaper
    combined = (std_before or 0.0) + (std_after or 0.0)
    if gap > combined:
        return "multistep confirmed better"
    elif gap > 0:
        return "multistep directional improvement (within ±1σ)"
    elif abs(gap) <= combined:
        return "tied (within ±1σ)"
    else:
        return "multistep confirmed worse"


def main() -> None:
    v7 = pd.read_csv(RESULTS_V7)
    machines = sorted(v7["machine_id"].unique())
    print(f"{len(machines)} feasible machines: {machines}")

    print("\nLoading 5,000,000 rows ...")
    df_raw = pd.read_csv(
        DATA_PATH, nrows=5_000_000,
        usecols=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"],
    )

    out_rows = []
    summary = []

    for mid in machines:
        grp = v7[v7["machine_id"] == mid]
        ds = float(grp["demand_scale"].iloc[0])
        burst = float(grp["burstiness_score"].iloc[0])
        print(f"\n{'='*74}\n  {mid}  (demand_scale={ds:.4f}x, burstiness={burst:.4f})\n{'='*74}")

        # ── LSTM, multistep engine: real retraining ─────────────────────────
        print("  Tuning LSTM under the multistep engine on validation ...")
        lstm_best = tune_on_validation(
            seed=42, machine_id=mid, df_raw=df_raw, demand_scale=ds,
            target_builder=_build_multistep_targets,
        )
        print(f"    LSTM multistep: upw={lstm_best['lstm_under_prov_weight']} "
              f"sm={lstm_best['lstm_safety_margin']}  val_cost={lstm_best['lstm_val_cost']:.4f}")

        lstm_multistep_rows = []
        for seed in SEEDS:
            r = run_single_experiment(
                seed,
                under_prov_weight=lstm_best["lstm_under_prov_weight"],
                safety_margin=lstm_best["lstm_safety_margin"],
                reactive_up=lstm_best["reactive_up"],
                reactive_down=lstm_best["reactive_down"],
                machine_id=mid, df_raw=df_raw, demand_scale=ds,
                target_builder=_build_multistep_targets,
            )
            lstm_multistep_rows.append(r)
            print(f"    seed={seed}: LSTM multistep cost={r['lstm_cost_score']:.4f}  "
                  f"SLA={r['lstm_sla_violation_rate_pct']:.4f}%  epochs={r['epochs_trained']}")

        # ── ARIMA, multistep engine: cheap, deterministic ───────────────────
        print("  Tuning ARIMA under the multistep engine on validation ...")
        arima_best = tune_arima_on_validation(
            seed=42, machine_id=mid, df_raw=df_raw, demand_scale=ds,
            target_builder=_build_multistep_targets,
        )
        arima_result = run_arima_experiment(
            seed=42,
            under_prov_weight=arima_best["arima_under_prov_weight"],
            safety_margin=arima_best["arima_safety_margin"],
            machine_id=mid, df_raw=df_raw, demand_scale=ds,
            target_builder=_build_multistep_targets,
        )
        arima_result_2 = run_arima_experiment(
            seed=42,
            under_prov_weight=arima_best["arima_under_prov_weight"],
            safety_margin=arima_best["arima_safety_margin"],
            machine_id=mid, df_raw=df_raw, demand_scale=ds,
            target_builder=_build_multistep_targets,
        )
        assert arima_result["arima_cost_score"] == arima_result_2["arima_cost_score"], (
            f"{mid}: ARIMA multistep cost score was NOT deterministic across two identical runs."
        )
        print(f"    ARIMA multistep: upw={arima_best['arima_under_prov_weight']} "
              f"sm={arima_best['arima_safety_margin']}  "
              f"cost={arima_result['arima_cost_score']:.4f}  "
              f"SLA={arima_result['arima_sla_violation_rate_pct']:.4f}%  (determinism check: matched)")

        # ── Assemble per-seed rows (greedy columns reused from v7) ──────────
        for seed, ms_row in zip(SEEDS, lstm_multistep_rows):
            base = grp[grp["seed"] == seed].iloc[0].to_dict()
            base.update({
                "lstm_under_prov_weight_multistep": lstm_best["lstm_under_prov_weight"],
                "lstm_safety_margin_multistep": lstm_best["lstm_safety_margin"],
                "lstm_cost_score_multistep": ms_row["lstm_cost_score"],
                "lstm_sla_violation_rate_pct_multistep": ms_row["lstm_sla_violation_rate_pct"],
                "lstm_over_prov_waste_pct_multistep": ms_row["lstm_over_prov_waste_pct"],
                "arima_under_prov_weight_multistep": arima_best["arima_under_prov_weight"],
                "arima_safety_margin_multistep": arima_best["arima_safety_margin"],
                "arima_cost_score_multistep": arima_result["arima_cost_score"],
                "arima_sla_violation_rate_pct_multistep": arima_result["arima_sla_violation_rate_pct"],
                "arima_over_prov_waste_pct_multistep": arima_result["arima_over_prov_waste_pct"],
            })
            out_rows.append(base)

        # ── Per-machine before/after summary + verdicts ─────────────────────
        lstm_greedy_cost_mean = grp["lstm_cost_score"].mean()
        lstm_greedy_cost_std = grp["lstm_cost_score"].std()
        lstm_ms_costs = np.array([r["lstm_cost_score"] for r in lstm_multistep_rows])
        lstm_ms_cost_mean, lstm_ms_cost_std = lstm_ms_costs.mean(), lstm_ms_costs.std()

        lstm_greedy_sla_mean = grp["lstm_sla_violation_rate_pct"].mean()
        lstm_ms_sla_mean = np.mean([r["lstm_sla_violation_rate_pct"] for r in lstm_multistep_rows])

        arima_greedy_cost = grp["arima_cost_score"].iloc[0]
        arima_ms_cost = arima_result["arima_cost_score"]
        arima_greedy_sla = grp["arima_sla_violation_rate_pct"].iloc[0]
        arima_ms_sla = arima_result["arima_sla_violation_rate_pct"]

        lstm_verdict = _verdict(lstm_greedy_cost_mean, lstm_ms_cost_mean, lstm_greedy_cost_std, lstm_ms_cost_std)
        arima_verdict = _verdict(arima_greedy_cost, arima_ms_cost, 0.0, 0.0)

        summary.append({
            "machine_id": mid, "burstiness": burst,
            "lstm_greedy_cost": lstm_greedy_cost_mean, "lstm_greedy_cost_std": lstm_greedy_cost_std,
            "lstm_multistep_cost": lstm_ms_cost_mean, "lstm_multistep_cost_std": lstm_ms_cost_std,
            "lstm_greedy_sla": lstm_greedy_sla_mean, "lstm_multistep_sla": lstm_ms_sla_mean,
            "arima_greedy_cost": arima_greedy_cost, "arima_multistep_cost": arima_ms_cost,
            "arima_greedy_sla": arima_greedy_sla, "arima_multistep_sla": arima_ms_sla,
            "lstm_verdict": lstm_verdict, "arima_verdict": arima_verdict,
        })
        print(f"  LSTM  cost: greedy {lstm_greedy_cost_mean:.4f}±{lstm_greedy_cost_std:.4f}  ->  "
              f"multistep {lstm_ms_cost_mean:.4f}±{lstm_ms_cost_std:.4f}   [{lstm_verdict}]")
        print(f"  ARIMA cost: greedy {arima_greedy_cost:.4f}  ->  "
              f"multistep {arima_ms_cost:.4f}   [{arima_verdict}]")

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(RESULTS_V8, index=False)
    print(f"\nSaved: {RESULTS_V8}  ({len(out_df)} rows)")

    sm = pd.DataFrame(summary)
    sm.to_csv(SUMMARY_V8, index=False)
    print(f"Saved: {SUMMARY_V8}")

    # ── Final ablation table ────────────────────────────────────────────────
    print("\n" + "=" * 108)
    print("  MULTISTEP ABLATION: greedy (max-of-horizon) vs multistep (mean-of-horizon) decision engine")
    print("=" * 108)
    print(f"  {'Machine':<10} {'Burst':>7} {'LSTM greedy':>13} {'LSTM multistep':>16} {'LSTM verdict':<34} "
          f"{'ARIMA greedy':>13} {'ARIMA multistep':>16}  ARIMA verdict")
    print("  " + "-" * 104)
    for _, r in sm.sort_values("burstiness").iterrows():
        print(f"  {r['machine_id']:<10} {r['burstiness']:>7.2f} "
              f"{r['lstm_greedy_cost']:>7.4f}±{r['lstm_greedy_cost_std']:.4f} "
              f"{r['lstm_multistep_cost']:>10.4f}±{r['lstm_multistep_cost_std']:.4f} "
              f"{r['lstm_verdict']:<34} "
              f"{r['arima_greedy_cost']:>13.4f} {r['arima_multistep_cost']:>16.4f}  {r['arima_verdict']}")

    print("\n  LSTM verdict counts:")
    for v, n in sm["lstm_verdict"].value_counts().items():
        print(f"    {v}: {n}/{len(sm)}")
    print("  ARIMA verdict counts:")
    for v, n in sm["arima_verdict"].value_counts().items():
        print(f"    {v}: {n}/{len(sm)}")
    print("=" * 108 + "\n")


if __name__ == "__main__":
    main()
