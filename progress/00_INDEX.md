# Progress Index (newest first)

See `README.md` in this folder for the convention these entries follow.

| Date | Step | Summary | Key result |
|---|---|---|---|
| 2026-08-27 | [Step 6: ARIMA baseline + sweep](./2026-08-27_step6-arima-baseline-and-sweep.md) | ARIMA(2,0,1) baseline; 3-way RMSE/MAE comparison; horizon sweep 5–60 min; lookback sweep 15–120 min | **ARIMA matches LSTM on RMSE** (0.5774 vs 0.5788±0.0046 — within 1σ). Both beat naive by 23%. LSTM sweet spot: 15-min horizon, 30-min lookback; plateau beyond. |
| 2026-08-27 | [Step 5: Per-machine demand-scale calibration](./2026-08-27_step5-demand-scale-calibration.md) | Replace global DEMAND_SCALE=20 with calibrate_demand_scale(mean_cpu, target=115%); re-run Step 4's 5 machines; skip infeasible (p99 > 800%) | **Step 4 LSTM SLA advantage on m_2134 retracted** — was ceiling-collision artifact. **First confirmed LSTM cost advantage** on m_2189 (+0.1572) and m_2065 (+0.0561). Burstiness hypothesis now disconfirmed for SLA (r=−0.91). |
| 2026-08-27 | [Step 4: Multi-machine burstiness](./2026-08-27_step4-multimachine-burstiness.md) | 5M-row machine selection; K-means (k=4) + stress-test; 5 machines × 5 seeds with per-machine tuning | Reactive wins on 4/5 machines on both metrics. ~~First confirmed LSTM SLA advantage on m_2134~~ **(retracted by Step 5 — was demand-calibration artifact)**. Hypothesis partially holds for SLA (r=+0.33), fails for cost. |
| 2026-08-27 | [Step 3: Symmetric baseline tuning](./2026-08-27_step3-symmetric-baseline-tuning.md) | 60/20/20 split; grid-searched both policies on val; ran 5-seed test on held-out test | **LSTM advantage reversed.** After fair tuning, Reactive achieves 0.0% SLA violations and cost=0.0000 on all 5 seeds. LSTM gets 1.766% SLA violations and cost=0.0883. Step 2's LSTM advantage was an artifact of tuning asymmetry. |
| 2026-08-26 | [Step 2: Multi-seed harness](./2026-08-26_step2-multiseed-harness.md) | Extracted pipeline into `experiments/pipeline.py`; ran 5 seeds (42–46); wrote results to `experiments/results.csv` | LSTM cost advantage is firmly established (0.0737 ± 0.0210 vs 0.1347, gap > combined ±1σ). SLA advantage has correct direction (0.3532% vs 0.4415%) but is within one combined σ — not yet confirmed. (Result invalidated by Step 3 — was tuning-asymmetry artifact.) |
| 2026-08-25 | [Step 1: Leakage fix + naive baseline](./2026-08-25_step1-leakage-fix-and-naive-baseline.md) | Fixed train/test scaler leakage; added naive persistence baseline; retrained model from scratch; re-executed notebook for real | LSTM beats naive baseline by 23% lower RMSE (new, real evidence). LSTM-vs-Reactive SLA result flipped between two identically-seeded runs (0.22% vs 0.44% in one, 0.44% vs 0.44% tie in another) — proves the single-run comparison isn't trustworthy yet. |

## What's confirmed so far (updated after Step 6)
- Scaler leakage bug: **fixed** (§7.1 of the audit).
- Naive persistence baseline: **added** (§7.5).
- Model retrains from scratch instead of silently reloading a stale cached model.
- Multi-seed harness: **done** — `experiments/results.csv` (Step 2), `results_v2_symmetric_tuning.csv` (Step 3), `results_v3_multimachine.csv` (Step 4), `results_v4_calibrated_demand.csv` (Step 5).
- LSTM vs Reactive with symmetric tuning: **Reactive wins** on both SLA violations and cost score, across all 5 seeds (Step 3, m_1933).
- ~~First confirmed LSTM SLA advantage: m_2134 (+0.22 pp)~~ — **retracted (Step 5)**; was a demand-calibration artifact (both policies hitting capacity ceiling identically).
- **First confirmed LSTM cost advantage**: m_2189 (burstiness=14.49, gap=+0.1572, confirmed > combined ±1σ=0.0065). SLA tied.
- **Second confirmed LSTM cost advantage**: m_2065 (burstiness=16.33, gap=+0.0561, confirmed > combined ±1σ=0.0132). Reactive wins SLA.
- **No confirmed LSTM SLA advantage on any machine** after proper demand calibration.
- **ARIMA(2,0,1) matches LSTM on point-forecast RMSE** (0.5774 vs 0.5788±0.0046 on m_1933) — within 1σ. ARIMA also wins MAE (0.4355 vs 0.4396).
- LSTM beats naive persistence by ~23% RMSE: **confirmed** (both ARIMA and LSTM share this advantage).
- **LSTM horizon sweet spot: 15 min** (LSTM/Naive ratio=0.763, best); degrades at 60 min (ratio=0.831).
- **LSTM lookback sweet spot: 30 min** (6 steps, default); plateaus at 60–120 min (no gain from more context).
- Burstiness–cost correlation with calibrated demand: **r=−0.64 (cost)** — negative, direction opposite to original hypothesis.
- Burstiness–SLA correlation with calibrated demand: **r=−0.91 (SLA)** — strongly negative; hypothesis fails.

## What's still open (in rough order)
- ARIMA vs LSTM comparison run on m_1933 only. Whether ARIMA remains competitive on bursty machines (m_2189, m_2134) is unknown.
- ARIMA not yet used in the autoscaler simulation (no cost-score comparison, only RMSE/MAE).
- LSTM cost advantage mechanism: on m_2189 and m_2065 the LSTM wins cost by avoiding over-provisioning waste, at the price of more SLA violations on m_2065.
- m_2101 infeasible at any reasonable scale (p99=965% at 115% mean target).
- Burstiness hypothesis fully disconfirmed for SLA; partially holds for cost.
- Phase 2B onward (src/ extraction, API, Kubernetes, dashboard): not started.
