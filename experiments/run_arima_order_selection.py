"""
Step 14 (Phase 1): ARIMA's order has been frozen at (2,0,1) since Step 6,
reused unchanged through Steps 8-13 -- flagged repeatedly in "Still open" as
never re-selected per machine. This re-selects it properly.

`select_arima_order` (arima_baseline.py, new) grid-searches ARIMA(p,d,q) via
AIC on each machine's TRAIN split only (p,q in 0..4, d in {0,1} -- 45
candidates, standard Box-Jenkins order selection, no leakage: the search
never touches val or test). Each candidate is a single MLE fit, not the full
rolling forecast, so this is cheap (~5-10s/machine).

Once each machine's best-fit order is chosen, this script re-runs ARIMA
through the exact same "fair" setup Step 13 established for the LSTM
comparison: `tune_arima_on_validation(target_builder=_build_multistep_targets,
horizon_weights_grid=HW_GRID, order=best_order)` then
`run_arima_experiment(..., order=best_order)`. LSTM numbers are reused
UNCHANGED from Step 13's results_v10 (already tuned+run under the identical
fair multistep+horizon_weights setup) -- no reason to retrain the LSTM for
an ARIMA-only change. Reactive numbers reused unchanged from results_v7.

Writes experiments/results_v11_arima_order_selection.csv (order-search
detail: chosen order, AIC, whether it differs from (2,0,1), per machine) and
experiments/results_v11_arima_reordered.csv (13 rows: old-order ARIMA cost
vs new-order ARIMA cost vs LSTM (Step 13 fair) vs Reactive, with head-to-head
and vs-Reactive verdicts recomputed under the new order).
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.autoscaler import (
    DATA_PATH,
    NROWS,
    _build_multistep_targets,
    _load_and_prepare,
    _split_three_way,
    ARIMA_ORDER,
    run_arima_experiment,
    select_arima_order,
    tune_arima_on_validation,
)
from src.autoscaler import config

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
RESULTS_V7 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "results_v7_arima_full13.csv")
RESULTS_V11_ORDERS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "results_v11_arima_order_selection.csv")
RESULTS_V11 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_v11_arima_reordered.csv")


def _h2h_verdict(cost_L, std_L, cost_A):
    """LSTM vs ARIMA head-to-head, sigma-gated (same rule used throughout,
    ARIMA std=0). Returns 'LSTM confirmed cheaper' / 'LSTM directional' /
    'tied (exact)' / 'ARIMA directional' / 'ARIMA confirmed cheaper'."""
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
    gap = cost_reactive - cost_policy   # positive => policy cheaper than Reactive
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
    v7 = pd.read_csv(RESULTS_V7)
    machines = sorted(v10["machine_id"].unique())
    print(f"{len(machines)} feasible machines: {machines}")

    print("\nLoading 5,000,000 rows ...")
    df_raw = pd.read_csv(
        DATA_PATH, nrows=5_000_000,
        usecols=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"],
    )

    order_rows = []
    out_rows = []

    for mid in machines:
        ds = float(v7[v7["machine_id"] == mid]["demand_scale"].iloc[0])
        burst = float(v10[v10["machine_id"] == mid]["burstiness"].iloc[0])
        print(f"\n{'='*78}\n  {mid}  (demand_scale={ds:.4f}x, burstiness={burst:.4f})\n{'='*78}")

        # ── Order selection on TRAIN only ────────────────────────────────
        ts, _ = _load_and_prepare(mid, NROWS, df_raw)
        train_data, _, _, _ = _split_three_way(ts, config.FEATURE_COL)
        train_flat = train_data.flatten()

        best_order, results = select_arima_order(
            train_flat, p_range=range(0, 5), d_range=(0, 1), q_range=range(0, 5))
        results_sorted = sorted(results, key=lambda r: r[1])
        old_order_entry = next((r for r in results_sorted if r[0] == ARIMA_ORDER), None)
        old_rank = results_sorted.index(old_order_entry) + 1 if old_order_entry else None
        print(f"  Best-fit order (AIC, {len(results)} converged of 45 candidates): "
              f"{best_order}  (old order (2,0,1) rank: {old_rank}/{len(results_sorted)})")
        order_rows.append({
            "machine_id": mid, "burstiness": burst,
            "best_order": str(best_order), "best_aic": results_sorted[0][1],
            "old_order_aic": old_order_entry[1] if old_order_entry else None,
            "old_order_rank": old_rank, "n_converged": len(results),
            "order_changed": best_order != ARIMA_ORDER,
        })

        # ── Re-tune + re-run ARIMA under the new order, same fair setup as Step 13 ──
        arima_best = tune_arima_on_validation(
            seed=42, machine_id=mid, df_raw=df_raw, demand_scale=ds,
            target_builder=_build_multistep_targets, horizon_weights_grid=HW_GRID,
            order=best_order,
        )
        arima_hw = arima_best["arima_horizon_weights"]
        arima_result = run_arima_experiment(
            seed=42,
            under_prov_weight=arima_best["arima_under_prov_weight"],
            safety_margin=arima_best["arima_safety_margin"],
            machine_id=mid, df_raw=df_raw, demand_scale=ds,
            target_builder=_build_multistep_targets, horizon_weights=arima_hw,
            order=best_order,
        )
        arima_result_2 = run_arima_experiment(
            seed=42,
            under_prov_weight=arima_best["arima_under_prov_weight"],
            safety_margin=arima_best["arima_safety_margin"],
            machine_id=mid, df_raw=df_raw, demand_scale=ds,
            target_builder=_build_multistep_targets, horizon_weights=arima_hw,
            order=best_order,
        )
        assert arima_result["arima_cost_score"] == arima_result_2["arima_cost_score"], (
            f"{mid}: ARIMA (new order) cost was NOT deterministic across two runs."
        )
        print(f"    ARIMA (order={best_order}): upw={arima_best['arima_under_prov_weight']} "
              f"sm={arima_best['arima_safety_margin']} hw={_hw_name(arima_hw)}  "
              f"cost={arima_result['arima_cost_score']:.4f}  (determinism check: matched)")

        # ── Compare against Step 13's (2,0,1)-order ARIMA + reused LSTM/Reactive ──
        v10_row = v10[v10["machine_id"] == mid].iloc[0]
        v7_row = v7[v7["machine_id"] == mid].iloc[0]
        old_arima_cost = v10_row["arima_hw_cost"]         # (2,0,1), fair hw-search, Step 13
        lstm_cost, lstm_std = v10_row["lstm_hw_cost"], v10_row["lstm_hw_cost_std"]
        reactive_cost = v7_row["reactive_cost_score"]      # unchanged since Step 8, deterministic

        new_arima_cost = arima_result["arima_cost_score"]

        h2h_old = _h2h_verdict(lstm_cost, lstm_std, old_arima_cost)
        h2h_new = _h2h_verdict(lstm_cost, lstm_std, new_arima_cost)
        arima_reactive_old = _vs_reactive_verdict(old_arima_cost, 0.0, reactive_cost)
        arima_reactive_new = _vs_reactive_verdict(new_arima_cost, 0.0, reactive_cost)

        out_rows.append({
            "machine_id": mid, "burstiness": burst,
            "old_order": str(ARIMA_ORDER), "new_order": str(best_order),
            "order_changed": best_order != ARIMA_ORDER,
            "arima_cost_old_order": old_arima_cost, "arima_cost_new_order": new_arima_cost,
            "arima_cost_delta": new_arima_cost - old_arima_cost,
            "lstm_cost": lstm_cost, "lstm_cost_std": lstm_std, "reactive_cost": reactive_cost,
            "h2h_verdict_old_order": h2h_old, "h2h_verdict_new_order": h2h_new,
            "h2h_changed": h2h_old != h2h_new,
            "arima_vs_reactive_old_order": arima_reactive_old,
            "arima_vs_reactive_new_order": arima_reactive_new,
        })
        print(f"  ARIMA cost: order(2,0,1)={old_arima_cost:.4f}  ->  order{best_order}={new_arima_cost:.4f}  "
              f"(delta={new_arima_cost - old_arima_cost:+.4f})")
        print(f"  Head-to-head vs LSTM ({lstm_cost:.4f}±{lstm_std:.4f}): "
              f"old-order [{h2h_old}]  ->  new-order [{h2h_new}]"
              f"{'  ** CHANGED **' if h2h_old != h2h_new else ''}")

    order_df = pd.DataFrame(order_rows)
    order_df.to_csv(RESULTS_V11_ORDERS, index=False)
    print(f"\nSaved: {RESULTS_V11_ORDERS}")

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(RESULTS_V11, index=False)
    print(f"Saved: {RESULTS_V11}")

    print("\n" + "=" * 108)
    print("  ARIMA ORDER RE-SELECTION: (2,0,1) everywhere (Steps 6-13) vs per-machine AIC-best (Step 14)")
    print("=" * 108)
    for _, r in out_df.sort_values("burstiness").iterrows():
        print(f"  {r['machine_id']:<10} burst={r['burstiness']:>6.2f}  "
              f"order: {r['old_order']}->{r['new_order']:<12} "
              f"cost: {r['arima_cost_old_order']:.4f}->{r['arima_cost_new_order']:.4f} ({r['arima_cost_delta']:+.4f})  "
              f"h2h: {r['h2h_verdict_old_order']:<28} -> {r['h2h_verdict_new_order']:<28}"
              f"{'  ** CHANGED **' if r['h2h_changed'] else ''}")
    print(f"\n  Orders changed: {order_df['order_changed'].sum()}/{len(order_df)}")
    print(f"  Head-to-head verdict changed: {out_df['h2h_changed'].sum()}/{len(out_df)}  "
          f"-> {out_df[out_df['h2h_changed']]['machine_id'].tolist()}")
    print("=" * 108 + "\n")


if __name__ == "__main__":
    main()
