# Step 6: ARIMA baseline + forecast-horizon / lookback sweep
Date: 2026-08-27
Status: done

## What changed

1. **`experiments/run_baselines_and_sweep.py` created**: three tasks in one script.
   - **Task 1** — `run_3way_comparison`: fits ARIMA(2,0,1) with rolling evaluation on the
     60/20/20 train/val/test split used by all previous steps; runs LSTM on seeds [42–46] via
     `run_single_experiment`; computes RMSE and MAE for all three methods.
   - **Task 2** — `run_horizon_sweep`: trains fresh LSTM with LOOKBACK=6 (fixed) and horizon
     ∈ {1, 3, 6, 12} steps (5/15/30/60 min); seed=42.
   - **Task 3** — `run_lookback_sweep`: trains fresh LSTM with HORIZON=3 (fixed) and lookback
     ∈ {3, 6, 12, 24} steps (15/30/60/120 min); seed=42.
   - Saves `experiments/outputs/06_3way_comparison.png` and
     `experiments/outputs/07_sweep_horizon_lookback.png`.

2. **ARIMA implementation**: fit on train only; state updated through val with
   `ARIMAResults.append(refit=False)` (Kalman filter update, no refit); rolling
   horizon-step-ahead forecasts over the test window aligned to the same
   target time-steps as LSTM sequence evaluation.  Order (2,0,1) selected a priori
   (ARMA(2,1) — handles short-range autocorrelation in stationary CPU series).
   AIC = −3038.5 on m_1933 train set.

3. **No changes to `experiments/pipeline.py`**: sweep uses private functions
   (`_build_lstm_model`, `_train_lstm`, `_evaluate_lstm`, `_evaluate_naive`,
   `_make_sequences`, `_split_three_way`) imported directly.

## Why

Step 1 reported that LSTM beats a naive-persistence baseline by ~23% RMSE. This is
meaningful only if naive persistence is a hard baseline to beat. Adding a classical
ARIMA forecaster tests whether the LSTM advantage is real or whether any decent
univariate model closes the gap.

The horizon and lookback sweeps address two open methodological questions:
- Does the LSTM's advantage over baselines depend on the forecast horizon? (If it only
  works at 15-min ahead but not 60-min, the finding is narrower than it looks.)
- Is the default lookback (30 min / 6 steps) near-optimal, or was it under-tuned?

## Before → After

### 3-way comparison (m_1933, 15 min / 3-step horizon, seeds 42–46 for LSTM)

| Metric | Naive persistence | ARIMA(2,0,1) | LSTM (mean±std) |
|---|---:|---:|---:|
| **RMSE (CPU %)** | 0.7531 | **0.5774** | 0.5788 ± 0.0046 |
| **MAE (CPU %)**  | 0.5509 | **0.4355** | 0.4396 ± 0.0013 |

LSTM vs Naive:  RMSE −0.1744 (−23.2 %) — LSTM better  
LSTM vs ARIMA:  RMSE −0.0014 (−0.24%) — **ARIMA wins by one σ width** (σ=0.0046)

### Horizon sweep (lookback=30 min, seed=42)

| Horizon | LSTM RMSE | Naive RMSE | LSTM/Naive |
|---:|---:|---:|---:|
|  5 min (1 step) | 0.5844 | 0.6935 | 0.843 |
| **15 min (3 steps)** | **0.5746** | 0.7531 | **0.763** |
| 30 min (6 steps) | 0.5825 | 0.7456 | 0.781 |
| 60 min (12 steps) | 0.6418 | 0.7727 | 0.831 |

### Lookback sweep (horizon=15 min, seed=42)

| Lookback | LSTM RMSE | Naive RMSE | LSTM/Naive |
|---:|---:|---:|---:|
|  15 min (3 steps) | 0.6955 | 0.7667 | 0.907 |
| **30 min (6 steps)** | **0.5747** | 0.7531 | **0.763** |
|  60 min (12 steps) | 0.5793 | 0.7422 | 0.781 |
| 120 min (24 steps) | 0.5793 | 0.7465 | 0.776 |

## Impact

### The LSTM does not beat a well-fitted ARIMA on point-forecast accuracy

ARIMA(2,0,1) achieves RMSE=0.5774 and MAE=0.4355.  The LSTM's mean is
RMSE=0.5788 ± 0.0046 — 0.0014 worse, which is 0.3 σ.  This is within noise;
no claim of LSTM superiority on point-forecast accuracy can be made for m_1933
at 15-min horizon.

Both ARIMA and LSTM beat naive persistence by **~23% RMSE** and **~21% MAE**.
The gap to naive is real; the gap between ARIMA and LSTM is not.

This partially undermines the project's original motivation: if ARIMA matches the
LSTM on forecasting, the autoscaler improvement (when it exists) must come from the
decision engine's use of uncertainty / multi-step forecasts, not from raw forecast
accuracy.

### Horizon sweep: LSTM advantage is horizon-dependent

The LSTM's relative RMSE advantage over naive varies:
- 5 min: −15.7% (weakest — 1 step ahead is easy for naive)
- 15 min: −23.7% (strongest — the default horizon)
- 30 min: −21.9%
- 60 min: −16.9%

The LSTM's absolute RMSE degrades at 60 min (0.5746 → 0.6418). The naive
baseline also degrades, but less sharply in percentage terms. The 15-min horizon
is near-optimal for the LSTM on this machine; extending to 60 min does not provide
proportionally more autoscaler lead-time.

### Lookback sweep: 30 min is the sweet spot; more context does not help

With lookback=15 min (3 steps), LSTM RMSE rises to 0.6955 — the model lacks enough
context.  At 30 min (6 steps, default), RMSE is 0.5747.  At 60 and 120 min, RMSE
plateaus at 0.5793 — essentially identical within seed noise.  The default
`LOOKBACK_STEPS=6` is at the point of diminishing returns; tuning it further will
not improve forecast accuracy on m_1933.

### ARIMA is a competitive practical alternative

ARIMA fits in 0.2 seconds and requires no GPU or hyperparameter tuning.  It matches
the LSTM's RMSE while being orders of magnitude simpler.  For machines like m_1933
with smooth, slow-varying demand the LSTM's advantage in the autoscaler comes from
the decision engine (safety margin, penalty weighting), not from forecasting skill.
On machines with sharper demand spikes (e.g., m_2134, m_2189) the LSTM may
outperform ARIMA — that comparison has not been run.

## Still open

- ARIMA vs LSTM comparison has only been run on m_1933 (smooth demand). Whether
  ARIMA remains competitive on bursty machines (m_2189, m_2134) is unknown.
- Horizon/lookback sweep was run on m_1933 only (seed=42). Multi-seed sweep or
  sweep on bursty machines is not done.
- ARIMA baseline not yet integrated into the autoscaler simulation (we only measured
  its forecast RMSE; its autoscaling cost score is not measured).
- No confirmed LSTM SLA advantage on any machine (Step 5 retracted the only
  confirmed instance). LSTM cost advantage confirmed on m_2189 and m_2065.
- Phase 2B onward (src/ extraction, API, Kubernetes, dashboard): not started.
