"""
Step 17: does ARIMA(2,0,1) residual autocorrelation predict which machines
show a real residual-hybrid advantage (Step 16)?

Step 16 found the residual hybrid (LSTM predicting ARIMA's residuals)
"confirmed cheaper than ARIMA" on 4/13 machines. One (m_2085) was explained
away as inheriting an already-known ARIMA anomaly (Step 10). Of the 3 new
candidates, a full Steps-10/13/14-style audit found m_2104 robust (survives
every check) and m_2380/m_2647 real-but-thin. This script tests the natural
mechanistic hypothesis: an LSTM can only extract genuine signal from ARIMA's
residuals if those residuals still contain autocorrelation ARIMA(2,0,1)
failed to capture (i.e. the model is underfit for that machine's structure).
Machines whose residuals are already close to white noise should be exactly
the ones where the hybrid finds nothing to add.

Diagnostic: fit ARIMA(2,0,1) on each machine's TRAIN split only (no test
leakage -- everything here is computable at model-selection time, before
ever touching val or test), take the model's own one-step-ahead in-sample
residuals (`fit.resid`, the standard object for this exact question -- NOT
the walk-forward multi-step residuals Step 16 used to build the hybrid's
training labels, which have induced overlap-autocorrelation from
overlapping horizon windows even under a perfectly-specified model and so
are the wrong input for a Ljung-Box test), then run the Ljung-Box test
(H0: no residual autocorrelation up to lag k -- rejecting H0 means the model
is missing structure) and report ACF/PACF at the first few lags.

Cross-references against every other per-machine feature already computed
in this project: burstiness score, demand_scale, K-means cluster membership
(reproducing the exact same clustering used throughout, see
run_pooled_lstm.py's docstring for the full-cluster-size context), and the
val->test demand-ceiling growth diagnostic (results_v9, Step 13).

Writes experiments/results_v15_residual_autocorrelation.csv.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.autoscaler import ARIMA_ORDER, config
from src.autoscaler.data import _load_and_prepare, _split_three_way

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_V7 = os.path.join(EXP_DIR, "results_v7_arima_full13.csv")
RESULTS_V9 = os.path.join(EXP_DIR, "results_v9_val_test_gap.csv")
RESULTS_V15 = os.path.join(EXP_DIR, "results_v15_residual_autocorrelation.csv")

LB_LAGS = [5, 10]
ACF_LAGS_TO_REPORT = 5

# From Step 16's confound audit + Step 15's exact K-means reproduction
HYBRID_WIN = {"m_2104": "robust", "m_2380": "thin", "m_2647": "thin", "m_2085": "confound (excluded)"}
CLUSTER_OF = {
    "m_2183": "A", "m_2104": "A", "m_2065": "A", "m_2134": "A", "m_2355": "A",  # 353 members
    "m_2380": "B", "m_2189": "B", "m_2163": "B", "m_2647": "B",                # 329 members
    "m_2056": "C", "m_2085": "C", "m_2241": "C", "m_2087": "C",                # 35 members
}


def main() -> None:
    v7 = pd.read_csv(RESULTS_V7)
    v9 = pd.read_csv(RESULTS_V9)
    machines = sorted(v7["machine_id"].unique())
    print(f"{len(machines)} feasible machines: {machines}\n")

    print("Loading 5,000,000 rows ...")
    df_raw = pd.read_csv(
        config.DATA_PATH, nrows=5_000_000,
        usecols=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"],
    )

    rows = []
    for mid in machines:
        ts, _ = _load_and_prepare(mid, config.NROWS, df_raw)
        train_data, _, _, _ = _split_three_way(ts, config.FEATURE_COL)
        train_flat = train_data.flatten()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ARIMA(train_flat, order=ARIMA_ORDER).fit()
        resid = fit.resid

        lb = acorr_ljungbox(resid, lags=LB_LAGS, return_df=True)
        acf_vals = acf(resid, nlags=ACF_LAGS_TO_REPORT, fft=True)[1:]  # drop lag-0 (always 1.0)
        pacf_vals = pacf(resid, nlags=ACF_LAGS_TO_REPORT)[1:]
        max_abs_acf = float(np.max(np.abs(acf_vals)))

        burst = v7[v7.machine_id == mid]["burstiness_score"].iloc[0]
        ds = v7[v7.machine_id == mid]["demand_scale"].iloc[0]
        growth_max = v9[v9.machine_id == mid]["growth_max_pct"].iloc[0]
        growth_p99 = v9[v9.machine_id == mid]["growth_p99_pct"].iloc[0]

        row = {
            "machine_id": mid,
            "hybrid_result": HYBRID_WIN.get(mid, "no advantage"),
            "cluster": CLUSTER_OF.get(mid, "?"),
            "burstiness": burst, "demand_scale": ds,
            "growth_max_pct": growth_max, "growth_p99_pct": growth_p99,
            "lb_stat_lag5": lb.loc[5, "lb_stat"], "lb_pvalue_lag5": lb.loc[5, "lb_pvalue"],
            "lb_stat_lag10": lb.loc[10, "lb_stat"], "lb_pvalue_lag10": lb.loc[10, "lb_pvalue"],
            "max_abs_acf_lag1to5": max_abs_acf,
        }
        for i in range(ACF_LAGS_TO_REPORT):
            row[f"acf_lag{i+1}"] = acf_vals[i]
            row[f"pacf_lag{i+1}"] = pacf_vals[i]
        rows.append(row)

        print(f"  {mid:<10} hybrid={row['hybrid_result']:<22} burst={burst:>6.2f}  "
              f"LB(lag10) stat={row['lb_stat_lag10']:>7.2f} p={row['lb_pvalue_lag10']:.4f}  "
              f"max|ACF(1-5)|={max_abs_acf:.4f}")

    out = pd.DataFrame(rows).sort_values("lb_pvalue_lag10")
    out.to_csv(RESULTS_V15, index=False)
    print(f"\nSaved: {RESULTS_V15}")

    print("\nRanked by Ljung-Box p-value at lag 10 (ascending = most residual autocorrelation left):")
    print(out[["machine_id", "hybrid_result", "cluster", "burstiness", "demand_scale",
              "growth_max_pct", "lb_stat_lag10", "lb_pvalue_lag10", "max_abs_acf_lag1to5"]].to_string(index=False))


if __name__ == "__main__":
    main()
