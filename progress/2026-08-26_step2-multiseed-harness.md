# Step 2: Multi-seed experiment harness
Date: 2026-08-26
Status: done

## What changed

1. **`experiments/pipeline.py` created.** All pipeline logic extracted from
   `lstm_autoscaler.ipynb` into a single importable module with one public
   function, `run_single_experiment(seed: int) -> dict`. The function runs the
   full pipeline (data load → prepare_timeseries → scale_and_split →
   make_sequences → build_lstm_model → train → evaluate_lstm →
   naive_persistence_forecast → DecisionConfig/decide_scaling →
   SimConfig/run_simulation → reactive_autoscaler) end to end for one seed
   and returns every metric. Every hyperparameter is frozen at the notebook's
   exact values (LOOKBACK=6, HORIZON=3, DEMAND_SCALE=20, under_prov_weight=20,
   safety_margin=0.25, ES_PATIENCE=10, etc.). The model is always trained from
   scratch; no cached model is ever reloaded.

2. **`experiments/run_multiseed.py` created.** Runs `run_single_experiment`
   for seeds [42, 43, 44, 45, 46], appends one row per run to
   `experiments/results.csv` (idempotent — already-done seeds are skipped),
   then prints a summary table of mean ± std and a plain-English verdict.

3. **`experiments/results.csv` created.** Holds the 5-run results (real
   numbers, no placeholders).

4. **Notebook updated.** A new markdown cell was inserted before the final
   comparison table pointing to `experiments/results.csv` as the multi-run
   source of truth and embedding the key summary numbers.

## Why

Step 1 found that two identically-seeded runs (seed=42) disagreed on whether
the LSTM beats the Reactive baseline on SLA violations. A 0.003-percentage-
point difference in forecast RMSE was enough to flip a single simulation tick
across a server-count threshold. The single-run comparison in the notebook
therefore cannot be trusted. The only way to know which outcome is
representative is to run the pipeline multiple times with different seeds and
report mean ± std.

## Before -> After

| Metric | Before (single notebook run, seed=42) | After (mean ± std, seeds 42–46) |
|---|---:|---:|
| LSTM Forecast RMSE | 0.5778% | 0.5827% ± 0.0027% |
| LSTM Forecast MAE | 0.4370% | 0.4387% ± 0.0009% |
| Naive Baseline RMSE | 0.7531% | 0.7531% ± 0.0000% |
| LSTM SLA Violation Rate | 0.22% (one run) | 0.3532% ± 0.1209% |
| Reactive SLA Violation Rate | 0.44% (one run) | 0.4415% ± 0.0000% |
| LSTM Over-Prov Waste | 0.66% (one run) | 0.3091% ± 0.3348% |
| Reactive Over-Prov Waste | 4.64% (one run) | 4.6358% ± 0.0000% |
| LSTM Cost Score | 0.0508 (one run) | 0.0737 ± 0.0210 |
| Reactive Cost Score | 0.1347 (one run) | 0.1347 ± 0.0000 |

Per-seed breakdown (from `experiments/results.csv`):

| Seed | LSTM RMSE | LSTM SLA% | React SLA% | LSTM Cost | React Cost | Epochs |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.578563 | 0.2208 | 0.4415 | 0.050773 | 0.134658 | 60 |
| 43 | 0.584596 | 0.4415 | 0.4415 | 0.090508 | 0.134658 | 60 |
| 44 | 0.581392 | 0.2208 | 0.4415 | 0.050773 | 0.134658 | 57 |
| 45 | 0.584997 | 0.4415 | 0.4415 | 0.088300 | 0.134658 | 60 |
| 46 | 0.583814 | 0.4415 | 0.4415 | 0.088300 | 0.134658 | 46 |

## Impact

**Cost score: the LSTM advantage is real.** LSTM 0.0737 ± 0.0210 vs Reactive
0.1347 ± 0.0000. The gap (0.0610) is nearly 3× the LSTM's own standard
deviation and far outside combined ±1σ. This holds across all 5 seeds without
exception.

**SLA violations: the advantage exists on average but is not firmly
established.** The LSTM wins on SLA in 2 of 5 seeds (42, 44) and ties in 3
(43, 45, 46). The mean gap (0.0883 percentage points) is smaller than the
LSTM's own ±1σ (0.1209 pp). The direction is consistently correct — the LSTM
never performs *worse* than Reactive on SLA — but the gap is narrow enough
that it could plausibly be noise at this sample size.

**The single-run comparison is now correctly framed.** The notebook note
makes clear that a one-off run should not be the claim; the multi-run CSV is.

**Forecast RMSE variance is low (±0.003 pp).** This is where the
nondeterminism lives, but it is tiny in absolute terms. The naive baseline
RMSE is perfectly stable (0.7531% every run), as expected — it doesn't involve
any random components.

## Still open

- Reactive baseline threshold tuning: still untouched (§7.2 of the audit).
  The LSTM-vs-Reactive comparison is still not a fair fight; the Reactive
  thresholds (80 / 30%) were never tuned against the same data and objective
  as the LSTM decision engine.
- Multi-machine validation: still only `m_1933`, picked by row count (§7.3).
- SLA advantage uncertain: 5 seeds is not enough to definitively separate the
  LSTM's SLA advantage from noise. More seeds, or a paired statistical test,
  would strengthen (or refute) this claim.
- Phase 2B onward (src/ extraction, API, Kubernetes, dashboard): not started.
