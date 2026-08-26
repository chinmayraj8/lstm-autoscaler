# Progress Index (newest first)

See `README.md` in this folder for the convention these entries follow.

| Date | Step | Summary | Key result |
|---|---|---|---|
| 2026-08-27 | [Step 3: Symmetric baseline tuning](./2026-08-27_step3-symmetric-baseline-tuning.md) | 60/20/20 split; grid-searched both policies on val; ran 5-seed test on held-out test | **LSTM advantage reversed.** After fair tuning, Reactive achieves 0.0% SLA violations and cost=0.0000 on all 5 seeds. LSTM gets 1.766% SLA violations and cost=0.0883. Step 2's LSTM advantage was an artifact of tuning asymmetry. |
| 2026-08-26 | [Step 2: Multi-seed harness](./2026-08-26_step2-multiseed-harness.md) | Extracted pipeline into `experiments/pipeline.py`; ran 5 seeds (42–46); wrote results to `experiments/results.csv` | LSTM cost advantage is firmly established (0.0737 ± 0.0210 vs 0.1347, gap > combined ±1σ). SLA advantage has correct direction (0.3532% vs 0.4415%) but is within one combined σ — not yet confirmed. (Result invalidated by Step 3 — was tuning-asymmetry artifact.) |
| 2026-08-25 | [Step 1: Leakage fix + naive baseline](./2026-08-25_step1-leakage-fix-and-naive-baseline.md) | Fixed train/test scaler leakage; added naive persistence baseline; retrained model from scratch; re-executed notebook for real | LSTM beats naive baseline by 23% lower RMSE (new, real evidence). LSTM-vs-Reactive SLA result flipped between two identically-seeded runs (0.22% vs 0.44% in one, 0.44% vs 0.44% tie in another) — proves the single-run comparison isn't trustworthy yet. |

## What's confirmed so far
- Scaler leakage bug: **fixed** (§7.1 of the audit).
- Naive persistence baseline: **added** (§7.5).
- Model retrains from scratch instead of silently reloading a stale cached model.
- Multi-seed harness: **done** — `experiments/results.csv` (Step 2) and `results_v2_symmetric_tuning.csv` (Step 3).
- LSTM vs Reactive with symmetric tuning: **Reactive wins** on both SLA violations and cost score, across all 5 seeds, with no variance. The Step 2 LSTM advantage was a tuning-asymmetry artifact.
- LSTM beats naive persistence baseline by ~23% RMSE: **still holds** (the LSTM is learning something; the problem is the decision/simulation layer, not the forecast).

## What's still open (in rough order)
- Fair evaluation setting: the test period's demand is easy enough for a tuned Reactive to achieve zero cost. An LSTM advantage (if any) requires a dataset with demand spikes that exceed Reactive reaction time, or a multi-machine setting with genuine pattern diversity (§7.3).
- Multi-machine validation — still only `m_1933`, picked by row count not representativeness (§7.3).
- Everything in Phase 2B onward (src/ extraction, API, Kubernetes, dashboard) — not started.
