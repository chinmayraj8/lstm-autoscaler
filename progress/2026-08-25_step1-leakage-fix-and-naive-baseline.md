# Step 1: Fix scaler leakage, add naive persistence baseline
Date: 2026-08-25
Status: done


What changed in `lstm_autoscaler.ipynb` today, and what the real before/after numbers are.

## What was fixed

1. **Scaler leakage (§7.1 of the audit).** `scale_and_split` now splits chronologically *first*, fits `MinMaxScaler` on the training partition *only*, and uses that same scaler to transform the test partition. Previously the scaler was fit on the full series (train+test together) before splitting.
2. **Naive persistence baseline added (§7.5).** A new "Step 1.6" section forecasts `predict(t+h) = value(t)` and reports its RMSE/MAE next to the LSTM's, so there's finally something to check the LSTM's accuracy against.
3. **MAE added** alongside RMSE throughout.
4. **The model was retrained from scratch.** The old cached `lstm_model.keras` was trained under the leaky preprocessing, so it's no longer valid — the notebook now always retrains rather than silently reloading a stale model.
5. **Reactive baseline thresholds were *not* touched.** Fixing the tuning asymmetry (§7.2 of the audit) is a bigger job — a proper grid search on both policies — and is intentionally scheduled separately, not bundled into this pass.

## Numbers: before vs. after

| Metric | Before (leaky, original) | After (fixed, this run) |
|---|---:|---:|
| LSTM Forecast RMSE | 0.5794% | 0.5778% |
| LSTM Forecast MAE | *(not measured)* | 0.4370% |
| Naive baseline RMSE | *(didn't exist)* | 0.7531% |
| **LSTM beats naive baseline by** | *(unknown)* | **23.3% lower RMSE** |
| SLA Violation Rate — LSTM vs Reactive | 0.22% vs 0.44% | 0.22% vs 0.44% |
| Over-Provisioning Waste — LSTM vs Reactive | 0.66% vs 4.64% | 0.66% vs 4.64% |
| Cost Score — LSTM vs Reactive | 0.0508 vs 0.1347 | 0.0508 vs 0.1347 |

Two genuinely good pieces of news: the leak itself turned out not to be hiding a big number (RMSE barely moved), and **the LSTM legitimately beats a naive "predict no change" baseline by ~23%** — that's new, real evidence the model is learning something, which the project never had before today.

## The uncomfortable finding (read this one)

I ran the corrected pipeline **twice** — once as a standalone script, once as the actual notebook re-execution above — with the same code, same seed (42), same data. The two runs did not agree:

| Run | RMSE | SLA: LSTM vs Reactive | Over-prov: LSTM vs Reactive | Cost: LSTM vs Reactive |
|---|---:|---:|---:|---:|
| Script run | 0.5811% | 0.44% vs 0.44% (**tied**) | 0.22% vs 4.64% | 0.0905 vs 0.1347 |
| Notebook run (shipped) | 0.5778% | 0.22% vs 0.44% | 0.66% vs 4.64% | 0.0508 vs 0.1347 |

A 0.003-percentage-point difference in forecast RMSE — noise-level, from ordinary TensorFlow run-to-run nondeterminism, not a code difference — was enough to flip the SLA-violation comparison from "LSTM wins" to "exact tie," because the decision engine only cares about whether a forecast crosses a discrete server-count threshold, and one particular 5-minute tick landed on the wrong side of that threshold in one run and the right side in the other.

This is exactly the risk the audit flagged in §7.4 ("no statistical treatment anywhere") and in the risk register ("fixing the bugs might change the headline result") — except now it's not hypothetical, it happened, from our own two runs. **The single-run comparison in the notebook right now (which happens to still show LSTM winning) cannot be trusted further than the standalone script's run (which shows a tie), because there is no principled reason to prefer one run over the other.** Only running this ≥5 times with different seeds and reporting mean ± std (already on the roadmap, Week 2–4) will tell us which behavior is real.

## What's still open

- Reactive baseline thresholds are still untuned (§7.2) — today's LSTM-vs-Reactive comparison is still not a fair fight.
- Only one machine (`m_1933`), still picked by "most rows," not representativeness.
- Still a single run per configuration — the variance shown above is exactly why the multi-seed harness is next.
