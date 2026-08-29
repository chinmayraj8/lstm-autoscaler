"""
Step 13 (Phase 2): does front-loading the multistep engine's horizon_weights
toward the near-term forecast step beat the uniform mean Step 12 shipped
with?

`_build_multistep_targets` (decision.py, Step 12) defaults `horizon_weights`
to a uniform mean (1/horizon each) -- every one of the HORIZON_STEPS=3
forecast steps counts equally toward the scaling decision. But
SIM_STARTUP_DELAY=1 means a scale-up decided *this* tick only becomes active
*next* tick, so the near-term forecast step is mechanically the one that
determines next-tick SLA risk most directly; steps 2-3 matter for smoothing
ahead but shouldn't necessarily be weighted equally to step 1. This was
flagged as unexplored in Steps 12 and 13's "still open" sections.

`horizon_weights_grid` (new optional param, Step 13, on `tune_on_validation` /
`tune_arima_on_validation`) adds this as a third grid dimension alongside
`under_prov_weight` x `safety_margin` -- same discipline as everything else
in this project: grid-searched on validation only, test touched once, same
grid for LSTM and ARIMA so neither is tuned harder than the other. Backward
compatible by construction (see experiment.py / arima_baseline.py
docstrings) -- confirmed by the full pytest suite passing unchanged.

Candidate schemes (HORIZON_STEPS=3, all sum to 1.0):
  uniform          [0.333, 0.333, 0.333]  -- Step 12's default, included as
                                             the baseline candidate so "no
                                             front-loading helps" is itself
                                             a reachable grid-search outcome,
                                             not assumed away.
  linear_321       [0.500, 0.333, 0.167]  -- weight proportional to 3:2:1
  front_60_30_10   [0.600, 0.300, 0.100]
  front_80_15_05   [0.800, 0.150, 0.050]  -- near-exclusively near-term

Runs LSTM (real retraining, 5 seeds) and ARIMA (deterministic) through this
enlarged grid on all 13 of Step 8's feasible machines, via the SAME
`_build_multistep_targets` target_builder Step 12 used -- only the
horizon_weights dimension is new. Writes
experiments/results_v10_horizon_weights.csv (65 rows) and
experiments/results_v10_horizon_weights_summary.csv (13 rows, one per
machine, comparing against Step 12's uniform-only multistep numbers from
results_v8).
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


RESULTS_V8_SUMMARY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "results_v8_ablation_summary.csv")
RESULTS_V7 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "results_v7_arima_full13.csv")
RESULTS_V10 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_v10_horizon_weights.csv")
SUMMARY_V10 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_v10_horizon_weights_summary.csv")


def _verdict(cost_before: float, cost_after: float, std_before: float, std_after: float) -> str:
    """Same combined-+/-1-sigma rule used throughout this project
    (run_multimachine_v2._verdict, reused in run_multistep_ablation._verdict)."""
    gap = cost_before - cost_after   # positive => after is cheaper
    combined = (std_before or 0.0) + (std_after or 0.0)
    if gap > combined:
        return "hw search confirmed better"
    elif gap > 0:
        return "hw search directional improvement (within +/-1sigma)"
    elif abs(gap) <= combined:
        return "tied (within +/-1sigma)"
    else:
        return "hw search confirmed worse"


def main() -> None:
    v8 = pd.read_csv(RESULTS_V8_SUMMARY)
    v7 = pd.read_csv(RESULTS_V7)
    machines = sorted(v8["machine_id"].unique())
    print(f"{len(machines)} feasible machines: {machines}")
    print(f"horizon_weights candidates: {list(HORIZON_WEIGHT_SCHEMES.keys())}")

    print("\nLoading 5,000,000 rows ...")
    df_raw = pd.read_csv(
        DATA_PATH, nrows=5_000_000,
        usecols=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"],
    )

    out_rows = []
    summary = []

    for mid in machines:
        ds = float(v7[v7["machine_id"] == mid]["demand_scale"].iloc[0])
        burst = float(v8[v8["machine_id"] == mid]["burstiness"].iloc[0])
        print(f"\n{'='*78}\n  {mid}  (demand_scale={ds:.4f}x, burstiness={burst:.4f})\n{'='*78}")

        # ── LSTM: tune (upw x sm x horizon_weights), then 5-seed test eval ──
        print("  Tuning LSTM (multistep, horizon_weights grid) on validation ...")
        lstm_best = tune_on_validation(
            seed=42, machine_id=mid, df_raw=df_raw, demand_scale=ds,
            target_builder=_build_multistep_targets, horizon_weights_grid=HW_GRID,
        )
        lstm_hw = lstm_best["lstm_horizon_weights"]
        print(f"    LSTM: upw={lstm_best['lstm_under_prov_weight']} "
              f"sm={lstm_best['lstm_safety_margin']} hw={_hw_name(lstm_hw)}  "
              f"val_cost={lstm_best['lstm_val_cost']:.4f}")

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
            )
            lstm_rows.append(r)
            print(f"    seed={seed}: LSTM cost={r['lstm_cost_score']:.4f}  "
                  f"SLA={r['lstm_sla_violation_rate_pct']:.4f}%")

        # ── ARIMA: tune (upw x sm x horizon_weights), then deterministic eval ──
        print("  Tuning ARIMA (multistep, horizon_weights grid) on validation ...")
        arima_best = tune_arima_on_validation(
            seed=42, machine_id=mid, df_raw=df_raw, demand_scale=ds,
            target_builder=_build_multistep_targets, horizon_weights_grid=HW_GRID,
        )
        arima_hw = arima_best["arima_horizon_weights"]
        arima_result = run_arima_experiment(
            seed=42,
            under_prov_weight=arima_best["arima_under_prov_weight"],
            safety_margin=arima_best["arima_safety_margin"],
            machine_id=mid, df_raw=df_raw, demand_scale=ds,
            target_builder=_build_multistep_targets, horizon_weights=arima_hw,
        )
        arima_result_2 = run_arima_experiment(
            seed=42,
            under_prov_weight=arima_best["arima_under_prov_weight"],
            safety_margin=arima_best["arima_safety_margin"],
            machine_id=mid, df_raw=df_raw, demand_scale=ds,
            target_builder=_build_multistep_targets, horizon_weights=arima_hw,
        )
        assert arima_result["arima_cost_score"] == arima_result_2["arima_cost_score"], (
            f"{mid}: ARIMA (horizon_weights search) cost was NOT deterministic across two runs."
        )
        print(f"    ARIMA: upw={arima_best['arima_under_prov_weight']} "
              f"sm={arima_best['arima_safety_margin']} hw={_hw_name(arima_hw)}  "
              f"cost={arima_result['arima_cost_score']:.4f}  "
              f"SLA={arima_result['arima_sla_violation_rate_pct']:.4f}%  (determinism check: matched)")

        # ── Assemble per-seed rows ───────────────────────────────────────────
        for seed, r in zip(SEEDS, lstm_rows):
            out_rows.append({
                "machine_id": mid, "burstiness": burst, "demand_scale": ds, "seed": seed,
                "lstm_under_prov_weight": lstm_best["lstm_under_prov_weight"],
                "lstm_safety_margin": lstm_best["lstm_safety_margin"],
                "lstm_horizon_weights": _hw_name(lstm_hw),
                "lstm_cost_score": r["lstm_cost_score"],
                "lstm_sla_violation_rate_pct": r["lstm_sla_violation_rate_pct"],
                "arima_under_prov_weight": arima_best["arima_under_prov_weight"],
                "arima_safety_margin": arima_best["arima_safety_margin"],
                "arima_horizon_weights": _hw_name(arima_hw),
                "arima_cost_score": arima_result["arima_cost_score"],
                "arima_sla_violation_rate_pct": arima_result["arima_sla_violation_rate_pct"],
            })

        # ── Compare against Step 12's uniform-only multistep baseline (v8) ──
        v8_row = v8[v8["machine_id"] == mid].iloc[0]
        lstm_uniform_cost, lstm_uniform_std = v8_row["lstm_multistep_cost"], v8_row["lstm_multistep_cost_std"]
        arima_uniform_cost = v8_row["arima_multistep_cost"]

        lstm_hw_costs = np.array([r["lstm_cost_score"] for r in lstm_rows])
        lstm_hw_cost_mean, lstm_hw_cost_std = lstm_hw_costs.mean(), lstm_hw_costs.std()
        arima_hw_cost = arima_result["arima_cost_score"]

        lstm_verdict = _verdict(lstm_uniform_cost, lstm_hw_cost_mean, lstm_uniform_std, lstm_hw_cost_std)
        arima_verdict = _verdict(arima_uniform_cost, arima_hw_cost, 0.0, 0.0)

        head_to_head_uniform = "LSTM" if v8_row["lstm_multistep_cost"] < v8_row["arima_multistep_cost"] else "ARIMA"
        head_to_head_hw = "LSTM" if lstm_hw_cost_mean < arima_hw_cost else "ARIMA"

        summary.append({
            "machine_id": mid, "burstiness": burst,
            "lstm_uniform_cost": lstm_uniform_cost, "lstm_uniform_cost_std": lstm_uniform_std,
            "lstm_hw_chosen": _hw_name(lstm_hw), "lstm_hw_cost": lstm_hw_cost_mean, "lstm_hw_cost_std": lstm_hw_cost_std,
            "lstm_verdict": lstm_verdict,
            "arima_uniform_cost": arima_uniform_cost,
            "arima_hw_chosen": _hw_name(arima_hw), "arima_hw_cost": arima_hw_cost,
            "arima_verdict": arima_verdict,
            "head_to_head_uniform_winner": head_to_head_uniform,
            "head_to_head_hw_winner": head_to_head_hw,
            "head_to_head_changed": head_to_head_uniform != head_to_head_hw,
        })
        print(f"  LSTM  cost: uniform {lstm_uniform_cost:.4f}±{lstm_uniform_std:.4f}  ->  "
              f"{_hw_name(lstm_hw)} {lstm_hw_cost_mean:.4f}±{lstm_hw_cost_std:.4f}   [{lstm_verdict}]")
        print(f"  ARIMA cost: uniform {arima_uniform_cost:.4f}  ->  "
              f"{_hw_name(arima_hw)} {arima_hw_cost:.4f}   [{arima_verdict}]")
        print(f"  Head-to-head winner: uniform={head_to_head_uniform}  hw-search={head_to_head_hw}"
              f"{'  ** CHANGED **' if head_to_head_uniform != head_to_head_hw else ''}")

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(RESULTS_V10, index=False)
    print(f"\nSaved: {RESULTS_V10}  ({len(out_df)} rows)")

    sm_df = pd.DataFrame(summary)
    sm_df.to_csv(SUMMARY_V10, index=False)
    print(f"Saved: {SUMMARY_V10}")

    print("\n" + "=" * 100)
    print("  HORIZON_WEIGHTS SEARCH: uniform (Step 12) vs best-of-4-schemes (Step 13)")
    print("=" * 100)
    for _, r in sm_df.sort_values("burstiness").iterrows():
        print(f"  {r['machine_id']:<10} burst={r['burstiness']:>6.2f}  "
              f"LSTM: uniform {r['lstm_uniform_cost']:.4f} -> {r['lstm_hw_chosen']:<15} {r['lstm_hw_cost']:.4f} [{r['lstm_verdict']}]  "
              f"ARIMA: uniform {r['arima_uniform_cost']:.4f} -> {r['arima_hw_chosen']:<15} {r['arima_hw_cost']:.4f} [{r['arima_verdict']}]")
    print("\n  LSTM chosen-scheme counts:", sm_df["lstm_hw_chosen"].value_counts().to_dict())
    print("  ARIMA chosen-scheme counts:", sm_df["arima_hw_chosen"].value_counts().to_dict())
    print("  LSTM verdict counts:", sm_df["lstm_verdict"].value_counts().to_dict())
    print("  ARIMA verdict counts:", sm_df["arima_verdict"].value_counts().to_dict())
    print("  Head-to-head winner changed on:", sm_df[sm_df["head_to_head_changed"]]["machine_id"].tolist())
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
