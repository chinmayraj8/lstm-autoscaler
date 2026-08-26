# Step 3: Symmetric baseline tuning (fair train/val/test split)
Date: 2026-08-27
Status: done

## What changed

1. **`experiments/pipeline.py` updated**: split changed from 80/20 (train/test)
   to 60/20/20 (train/val/test). The scaler still fits on train only. A new
   public function `tune_on_validation(seed)` grid-searches both policies on
   the validation split and never touches the test split. `run_single_experiment`
   now accepts tuned params and evaluates only on the test split.

2. **`experiments/run_multiseed_v2.py` created**: drives the full workflow —
   tune on validation (seed=42), freeze the best params for both policies, then
   run seeds [42-46] on the held-out test split. Results written to
   `experiments/results_v2_symmetric_tuning.csv`.

3. **Both policies tuned on the validation split (search on val, never test):**

   | Policy | Parameter | Original | Tuned (val best) | Val cost |
   |---|---|---:|---:|---:|
   | Reactive | scale_up_threshold | 80.0 % | **60** % | — |
   | Reactive | scale_down_threshold | 30.0 % | **40** % | **0.0000** |
   | LSTM dec | under_prov_weight | 20.0 | **5** | — |
   | LSTM dec | safety_margin | 0.25 | **0.10** | **0.0022** |

   Both policies changed substantially. The Reactive found a lower up-threshold
   (scales up earlier) and a higher down-threshold (scales down later), which
   keeps more servers available and achieves zero cost on validation. The LSTM
   decision engine moved to much lower conservatism weights; even at its best
   val-set configuration it still could not match Reactive's perfect val cost.

## Why

Step 2 established that the LSTM had an advantage on cost score (confirmed) and
a directional advantage on SLA violations (within one σ). But both results came
from a pipeline where the LSTM decision-engine weights (under_prov_weight=20,
safety_margin=0.25) had been tuned against the same 20% test window the results
were reported on, while the Reactive thresholds (80/30%) were never tuned at
all. This step gives both policies an equally rigorous tuning process using a
proper held-out validation window.

## Before -> After

Comparison: Step 2 results (80/20, asymmetric tuning) vs Step 3 results (60/20/20,
symmetric tuning). Both compare the mean across seeds 42–46. Step 3 test split
is the same chronological slice as Step 2's test split (last 20 % of data); the
train split is shorter (60 % vs 80 %).

| Metric | Step 2 mean (asymmetric) | Step 3 mean (symmetric) |
|---|---:|---:|
| LSTM Forecast RMSE | 0.5827% | 0.5811% |
| Naive Baseline RMSE | 0.7531% | 0.7531% |
| **LSTM SLA Violation Rate** | **0.3532%** | **1.7660%** |
| **Reactive SLA Violation Rate** | **0.4415%** | **0.0000%** |
| **LSTM Over-Prov Waste** | 0.3091% | **0.0000%** |
| **Reactive Over-Prov Waste** | 4.6358% | **0.0000%** |
| **LSTM Cost Score** | **0.0737** | **0.0883** |
| **Reactive Cost Score** | **0.1347** | **0.0000** |

Per-seed test results (from `experiments/results_v2_symmetric_tuning.csv`):

| Seed | LSTM RMSE | LSTM SLA% | React SLA% | LSTM Cost | React Cost | Epochs |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.573777 | 1.766 | 0.000 | 0.0883 | 0.0000 | 48 |
| 43 | 0.593518 | 1.766 | 0.000 | 0.0883 | 0.0000 | 14 |
| 44 | 0.575011 | 1.766 | 0.000 | 0.0883 | 0.0000 | 28 |
| 45 | 0.588468 | 1.766 | 0.000 | 0.0883 | 0.0000 | 18 |
| 46 | 0.574512 | 1.766 | 0.000 | 0.0883 | 0.0000 | 33 |

Mean ± std: LSTM SLA 1.7660% ± 0.0000%, Reactive 0.0000% ± 0.0000%. LSTM cost
0.0883 ± 0.0000, Reactive 0.0000 ± 0.0000.

## Impact

**The LSTM's advantage from Step 2 does not survive symmetric tuning. It reverses.**

After giving the Reactive baseline a fair grid-search on validation, it achieves
zero SLA violations and zero over-provisioning cost on the test split across all
five seeds. The LSTM, with its validation-preferred parameters (upw=5, sm=0.10),
has 1.766% SLA violations and a cost score of 0.0883 in all five seeds.

The gap is absolute and perfectly stable: Reactive wins on every seed, on both
metrics, with zero variance. The gap is larger than any combined σ.

This means the Step 2 result ("LSTM cost advantage confirmed, gap > combined
±1σ") was an artifact of the tuning asymmetry: the LSTM's decision engine was
tuned against the test window, while Reactive's thresholds were never tuned at
all. Once tuning is symmetric, the picture inverts.

Several additional observations worth noting:

- The LSTM's simulation outcomes are perfectly stable across seeds (0.000 σ)
  even though RMSE varies by ±0.0093 pp. With the lower upw=5/sm=0.10 params,
  the decision engine is less sensitive to exact prediction values, so the
  seed-to-seed RMSE variance no longer flips simulation outcomes. This is the
  opposite of Step 2.

- The Reactive baseline achieving cost=0.0000 on both val and test means the
  tuned thresholds (up=60, down=40) keep the server count at exactly the right
  level for this machine's demand pattern — no SLA violations, no period where
  capacity exceeds double the demand.

- The fact that four different Reactive configurations all achieve cost=0.0000 on
  validation (up=60/65/70/75, down=40) suggests the validation period's demand
  is easy to serve: many thresholds work perfectly. The key finding is that
  the Reactive, given any reasonable chance to find a good threshold, can match
  or beat the LSTM on this data.

- This does NOT mean the LSTM is useless — it means the current evaluation
  (one machine, one demand pattern, 5-minute granularity, simple decision
  engine) is not hard enough for the LSTM to demonstrate value. A Reactive
  autoscaler can solve the problem.

## Still open

- **LSTM vs Reactive with a harder evaluation**: the current test period is
  apparently easy enough for a well-tuned Reactive to achieve zero cost. A
  fair evaluation would require a dataset with genuine demand spikes that
  exceed Reactive reaction time — or a multi-machine setting where pattern
  diversity forces the LSTM to actually predict rather than just react.
- Reactive threshold tuning asymmetry was the immediate problem and is now fixed,
  but the evaluation itself (§7.3 of the audit — one machine, not
  representative) remains unresolved.
- Multi-machine validation: still only `m_1933` (§7.3).
- Phase 2B onward (src/ extraction, API, Kubernetes, dashboard): not started.
