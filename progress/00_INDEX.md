# Progress Index (newest first)

See `README.md` in this folder for the convention these entries follow.

| Date | Step | Summary | Key result |
|---|---|---|---|
| 2026-08-29 | [Step 9: src/ package extraction + pytest suite](./2026-08-29_step9-src-refactor.md) | Split `experiments/pipeline.py` (515 lines, no tests) into a proper `src/autoscaler/` package (config/data/decision/simulation/calibration/forecasting/experiment); `pipeline.py` is now a backward-compatible shim; added 25 pytest tests for the non-TensorFlow logic | **No numeric behavior changed** — 25/25 tests pass for real, and a fresh real training run (seed 999, real TensorFlow/Metal GPU) reproduced Reactive's numbers exactly and landed the LSTM's numbers within normal seed variance. Fully verified. |
| 2026-08-27 | [Step 8: Expanded multi-machine validation](./2026-08-27_step8-expanded-multimachine.md) | Same K-means (k=4) + stress-test selection as Step 4, but 3-4 machines per cluster (17 total, 13 feasible) instead of 1 | **LSTM cost advantage generalizes**: 10/13 feasible machines (77%) show a confirmed cost advantage, up from 2/4 in Step 5 — not two lucky machines. But the "tied SLA" framing does NOT generalize: 0/13 machines show tied SLA; the real pattern is a trade-off (LSTM wins cost, Reactive wins SLA) on 9/10, with one exception (m_2085) where LSTM wins both. Entire cluster 1 (4/4 sampled) is infeasible, not just m_2101. Burstiness-cost correlation collapses (r=-0.639 → -0.218); burstiness-SLA correlation holds (r=-0.876). |
| 2026-08-27 | [Step 6: ARIMA baseline + sweep](./2026-08-27_step6-arima-baseline-and-sweep.md) | ARIMA(2,0,1) baseline; 3-way RMSE/MAE comparison; horizon sweep 5–60 min; lookback sweep 15–120 min | **ARIMA matches LSTM on RMSE** (0.5774 vs 0.5788±0.0046 — within 1σ). Both beat naive by 23%. LSTM sweet spot: 15-min horizon, 30-min lookback; plateau beyond. |
| 2026-08-27 | [Step 5: Per-machine demand-scale calibration](./2026-08-27_step5-demand-scale-calibration.md) | Replace global DEMAND_SCALE=20 with calibrate_demand_scale(mean_cpu, target=115%); re-run Step 4's 5 machines; skip infeasible (p99 > 800%) | **Step 4 LSTM SLA advantage on m_2134 retracted** — was ceiling-collision artifact. **First confirmed LSTM cost advantage** on m_2189 (+0.1572) and m_2065 (+0.0561). Burstiness hypothesis now disconfirmed for SLA (r=−0.91). |
| 2026-08-27 | [Step 4: Multi-machine burstiness](./2026-08-27_step4-multimachine-burstiness.md) | 5M-row machine selection; K-means (k=4) + stress-test; 5 machines × 5 seeds with per-machine tuning | Reactive wins on 4/5 machines on both metrics. ~~First confirmed LSTM SLA advantage on m_2134~~ **(retracted by Step 5 — was demand-calibration artifact)**. Hypothesis partially holds for SLA (r=+0.33), fails for cost. |
| 2026-08-27 | [Step 3: Symmetric baseline tuning](./2026-08-27_step3-symmetric-baseline-tuning.md) | 60/20/20 split; grid-searched both policies on val; ran 5-seed test on held-out test | **LSTM advantage reversed.** After fair tuning, Reactive achieves 0.0% SLA violations and cost=0.0000 on all 5 seeds. LSTM gets 1.766% SLA violations and cost=0.0883. Step 2's LSTM advantage was an artifact of tuning asymmetry. |
| 2026-08-26 | [Step 2: Multi-seed harness](./2026-08-26_step2-multiseed-harness.md) | Extracted pipeline into `experiments/pipeline.py`; ran 5 seeds (42–46); wrote results to `experiments/results.csv` | LSTM cost advantage is firmly established (0.0737 ± 0.0210 vs 0.1347, gap > combined ±1σ). SLA advantage has correct direction (0.3532% vs 0.4415%) but is within one combined σ — not yet confirmed. (Result invalidated by Step 3 — was tuning-asymmetry artifact.) |
| 2026-08-25 | [Step 1: Leakage fix + naive baseline](./2026-08-25_step1-leakage-fix-and-naive-baseline.md) | Fixed train/test scaler leakage; added naive persistence baseline; retrained model from scratch; re-executed notebook for real | LSTM beats naive baseline by 23% lower RMSE (new, real evidence). LSTM-vs-Reactive SLA result flipped between two identically-seeded runs (0.22% vs 0.44% in one, 0.44% vs 0.44% tie in another) — proves the single-run comparison isn't trustworthy yet. |

## What's confirmed so far (updated after Step 9)
- Scaler leakage bug: **fixed** (§7.1 of the audit).
- Naive persistence baseline: **added** (§7.5).
- Model retrains from scratch instead of silently reloading a stale cached model.
- Multi-seed harness: **done** — `experiments/results.csv` (Step 2), `results_v2_symmetric_tuning.csv` (Step 3), `results_v3_multimachine.csv` (Step 4), `results_v4_calibrated_demand.csv` (Step 5), `results_v5_expanded_multimachine.csv` (Step 8, 17 machines / 13 feasible).
- LSTM vs Reactive with symmetric tuning: **Reactive wins** on both SLA violations and cost score, across all 5 seeds (Step 3, m_1933).
- ~~First confirmed LSTM SLA advantage: m_2134 (+0.22 pp)~~ — **retracted (Step 5)**; was a demand-calibration artifact (both policies hitting capacity ceiling identically).
- **LSTM cost advantage generalizes (Step 8)**: 10/13 feasible machines (77%) show a confirmed LSTM cost advantage (gap > combined ±1σ), across 3 of 4 clusters. Not two lucky machines.
- ~~m_2189's SLA was "tied" (Step 5)~~ — **did not reproduce on re-run (Step 8)**: resolves to Reactive winning SLA by -0.57pp. Likely TensorFlow/Metal training non-determinism near a close call. **0/13 machines show tied SLA in Step 8** — the dominant pattern is a trade-off (LSTM wins cost, Reactive wins SLA), true on 9/10 cost-advantage machines.
- **One machine (m_2085) has LSTM winning BOTH metrics outright** — first such case in this project. Unexplained mechanism; same cluster as two machines where Reactive wins both.
- **Reactive wins both metrics outright on 3/13 (23%)**: m_2087, m_2241 (calm cluster), m_2134 (stress test).
- **No confirmed LSTM SLA advantage on any machine** except m_2085, after proper demand calibration.
- **ARIMA(2,0,1) matches LSTM on point-forecast RMSE** (0.5774 vs 0.5788±0.0046 on m_1933) — within 1σ. ARIMA also wins MAE (0.4355 vs 0.4396).
- LSTM beats naive persistence by ~23% RMSE: **confirmed** (both ARIMA and LSTM share this advantage).
- **LSTM horizon sweet spot: 15 min** (LSTM/Naive ratio=0.763, best); degrades at 60 min (ratio=0.831).
- **LSTM lookback sweet spot: 30 min** (6 steps, default); plateaus at 60–120 min (no gain from more context).
- Burstiness–SLA correlation: **r=−0.876 (n=13, Step 8)**, consistent with Step 5's r=−0.911 (n=4) — strongly negative; hypothesis fails, holds up with more data.
- Burstiness–cost correlation: **collapses from r=−0.639 (n=4, Step 5) to r=−0.218 (n=13, Step 8)** — Step 5's cost correlation was largely a small-sample artifact.
- **Entire cluster 1 is infeasible** (Step 8), not just m_2101 (Step 5): all 4 sampled members (m_2101, m_2315, m_2281, m_2049) have p99 demand 830-965% > 800% fleet cap. Structural, not machine-specific.
- **Core pipeline is now a tested package** (Step 9): `src/autoscaler/`, 25 passing pytest tests on the decision engine, simulator, and calibration math. `experiments/pipeline.py` kept as a compatibility shim.

## What's still open (in rough order)
- ARIMA vs LSTM comparison run on m_1933 only. Whether ARIMA remains competitive on bursty machines (m_2189, m_2134, or the other 12 Step 8 machines) is unknown.
- ARIMA not yet used in the autoscaler simulation (no cost-score comparison, only RMSE/MAE).
- **m_2085 (LSTM wins both metrics) is unexplained** — what differs about its demand shape vs m_2087/m_2241 in the same cluster where Reactive wins both.
- **Training non-determinism**: re-running the same machine/seed can shift tuned params and flip close verdicts (m_2189 SLA tied → Reactive-wins between Step 5 and Step 8). Treat single-machine verdicts near a tie as directional, not exact; aggregate patterns across many machines are more trustworthy.
- Cluster 1 (m_2101 and neighbors) infeasible at any reasonable scale — needs a higher `max_servers` ceiling or a burst-absorbing capacity tier to evaluate at all.
- Only 13/733 qualifying machines evaluated (1.8%) — K-means clusters are coarse over 3 features; within-cluster diversity (cluster 2 alone has 3 distinct outcomes) suggests even 13 may undersample the real variance.
- Burstiness hypothesis fully disconfirmed for SLA; fails for cost too now (correlation collapsed with more data).
- Containerization (Docker) and a real Kubernetes/KEDA deployment: not started. (The FastAPI service and the Streamlit dashboard are both already done, ahead of this index — see `src/api/` and `src/dashboard/`.)
