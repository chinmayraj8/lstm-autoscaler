"""
Diagnostic: for each of Step 8's 13 feasible machines, compute the raw
(demand_scale-multiplied) demand ceiling on the validation split vs the
test split, and the % growth from val to test.

This directly replicates the mechanism Step 10 used to explain why m_2085's
"LSTM wins both metrics" result was a tuning-generalization artifact (its
val-tuned Reactive threshold didn't survive a test split whose demand
ceiling grew 26.6% past what validation showed). Step 10 flagged this as
"still open": whether the same growth-vs-verdict-flip pattern predicts
which OTHER machines have unreliable single-machine verdicts, including the
four Step 12 flagged as needing this exact check (m_2380, m_2189, m_2163,
m_2183) before treating their new/strengthened LSTM-vs-ARIMA multistep wins
as real forecasting-shape effects rather than another generalization
artifact.

No TensorFlow needed -- pure data diagnostic, reusing the project's own
_prepare_timeseries / config.FEATURE_COL and the same 60/20/20 chronological
split math as _split_three_way (but on raw values, not scaled -- percentiles
here are directly comparable to check_feasibility's convention).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.autoscaler import config
from src.autoscaler.data import _prepare_timeseries

RESULTS_V5 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "results_v5_expanded_multimachine.csv")

FLAGGED = {"m_2380", "m_2189", "m_2163", "m_2183"}  # Step 12's unverified new/strengthened wins
KNOWN = {"m_2085": "known confound (Step 10)", "m_2087": "cluster-mate control", "m_2241": "cluster-mate control"}


def split_raw(ts_values, val_ratio=config.VAL_RATIO, test_ratio=config.TEST_RATIO):
    n = len(ts_values)
    train_end = int(n * (1.0 - val_ratio - test_ratio))
    val_end = int(n * (1.0 - test_ratio))
    return ts_values[:train_end], ts_values[train_end:val_end], ts_values[val_end:]


def main():
    v5 = pd.read_csv(RESULTS_V5)
    machines = sorted(v5["machine_id"].unique())
    print(f"{len(machines)} feasible machines: {machines}\n", flush=True)

    print(f"Loading data from {config.DATA_PATH} ...", flush=True)
    df_raw = pd.read_csv(
        config.DATA_PATH, nrows=5_000_000,
        usecols=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"],
    )
    print(f"Loaded {len(df_raw):,} rows.\n", flush=True)

    rows = []
    for mid in machines:
        grp = v5[v5["machine_id"] == mid]
        ds = float(grp["demand_scale"].iloc[0])
        burst = float(grp["burstiness_score"].iloc[0])

        ts = _prepare_timeseries(df_raw, mid)
        vals = ts[config.FEATURE_COL].values.astype(np.float64) * ds  # raw demand %, calibrated

        _, val_raw, test_raw = split_raw(vals)

        val_max, test_max = float(val_raw.max()), float(test_raw.max())
        val_p99, test_p99 = float(np.percentile(val_raw, 99)), float(np.percentile(test_raw, 99))
        growth_max = (test_max - val_max) / val_max * 100.0
        growth_p99 = (test_p99 - val_p99) / val_p99 * 100.0

        tag = KNOWN.get(mid, "FLAGGED (Step 12 multistep win/strengthen)" if mid in FLAGGED else "")
        rows.append(dict(
            machine_id=mid, burstiness=burst, demand_scale=ds,
            val_max=round(val_max, 2), test_max=round(test_max, 2), growth_max_pct=round(growth_max, 1),
            val_p99=round(val_p99, 2), test_p99=round(test_p99, 2), growth_p99_pct=round(growth_p99, 1),
            tag=tag,
        ))
        print(f"  {mid:10s} burst={burst:6.2f}  val_max={val_max:7.2f}%  test_max={test_max:7.2f}%  "
              f"growth={growth_max:+6.1f}%   {tag}", flush=True)

    out = pd.DataFrame(rows).sort_values("growth_max_pct", ascending=False)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v9_val_test_gap.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}\n")

    print("Ranked by val->test max-demand growth (descending):")
    print(out[["machine_id", "burstiness", "growth_max_pct", "growth_p99_pct", "tag"]].to_string(index=False))

    print("\nSanity check against Step 10's reported numbers for the known machines:")
    print("  m_2085 expected val_max=173.47  test_max=219.73  growth=+26.6%")
    print("  m_2087 expected val_max=134.27  test_max=163.42  growth=+21.7%")
    print("  m_2241 expected val_max=167.58  test_max=167.58  growth=+0.0%")


if __name__ == "__main__":
    main()
