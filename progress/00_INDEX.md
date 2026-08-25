# Progress Index (newest first)

See `README.md` in this folder for the convention these entries follow.

| Date | Step | Summary | Key result |
|---|---|---|---|
| 2026-08-26 | [Step 2: Multi-seed harness](./2026-08-26_step2-multiseed-harness.md) | Extracted pipeline into `experiments/pipeline.py`; ran 5 seeds (42–46); wrote results to `experiments/results.csv` | LSTM cost advantage is firmly established (0.0737 ± 0.0210 vs 0.1347, gap > combined ±1σ). SLA advantage has correct direction (0.3532% vs 0.4415%) but is within one combined σ — not yet confirmed. |
| 2026-08-25 | [Step 1: Leakage fix + naive baseline](./2026-08-25_step1-leakage-fix-and-naive-baseline.md) | Fixed train/test scaler leakage; added naive persistence baseline; retrained model from scratch; re-executed notebook for real | LSTM beats naive baseline by 23% lower RMSE (new, real evidence). LSTM-vs-Reactive SLA result flipped between two identically-seeded runs (0.22% vs 0.44% in one, 0.44% vs 0.44% tie in another) — proves the single-run comparison isn't trustworthy yet. |

## What's confirmed so far
- Scaler leakage bug: **fixed** (§7.1 of the audit).
- Naive persistence baseline: **added** (§7.5).
- Model retrains from scratch instead of silently reloading a stale cached model.
- Multi-seed harness: **done** — `experiments/results.csv` holds 5-run ground truth.
- LSTM cost score advantage: **firmly established** across all 5 seeds (gap > combined ±1σ).

## What's still open (in rough order)
- SLA violation advantage: direction is correct but within one combined σ — needs more seeds or a paired test to confirm.
- Reactive baseline threshold tuning — currently untouched, so the LSTM-vs-Reactive fight is still not fair (§7.2).
- Multi-machine validation — still only tested on `m_1933`, picked by row count not representativeness (§7.3).
- Everything in Phase 2B onward (src/ extraction, API, Kubernetes, dashboard) — not started.
