# Step 15: pooled training across a machine cluster — no net gain, and a new instability mode
Date: 2026-08-30
Status: done

## What changed

**`experiments/run_pooled_lstm.py` created.** Instead of training one LSTM
from scratch per machine (every prior step), this pools training data
across several similar machines and trains one shared LSTM, then evaluates
it two ways per machine: applied directly, and briefly fine-tuned on that
machine's own train split. Architecture, `LOOKBACK_STEPS`/`HORIZON_STEPS`,
and the decision engine are held **completely unchanged** from Step 8/11's
original setup (`_build_lstm_targets`, the greedy max-of-horizon engine,
`LSTM_UPW_GRID`×`LSTM_SM_GRID`) — deliberately, per the task, to isolate
"does more training data help" from any of the architecture/engine/feature
changes Steps 12–14 already tried and found didn't help. Step 11's numbers
(`results_v7_arima_full13.csv`: per-machine LSTM, ARIMA(2,0,1), Reactive,
all under this same original engine) are reused unchanged as the baseline.

**Pool selection.** Reproduced `run_multimachine_v2.py`'s exact K-means
(k=4, standardized mean/std/burstiness, `random_state=42`, `n_init=10`) on
the full 733-machine qualifying candidate pool — not just the 13 sampled
representatives. Full cluster sizes: 353, 329, 35, 16. The 353-machine
cluster is both the single largest full cluster *and* has the most
already-feasible sampled representatives among Step 8's 13 (5, vs 4 each
for the other two non-infeasible clusters): **m_2183, m_2104, m_2065,
m_2355, m_2134** — the pool used here.

**Calendar-overlap leakage check (done explicitly, as asked).** All 5 pool
machines' resampled series span the identical ~8-day observation window
(1970-01-01 → 1970-01-08, epoch-relative — this trace gives every machine
the same fixed calendar window, it isn't per-machine coverage) with nearly
identical lengths (2281–2304 points). Checked directly: the **latest**
train-split end across all 5 pool machines is **1970-01-05 19:50**; the
**earliest** test-split start across all 5 is **1970-01-07 09:35** — a
~1.5-day gap, zero pairwise overlap between any machine's train split and
any machine's (including its own) test split. **Verdict: this specific
confound is not a real risk in this dataset**, because every machine's
chronological 60/20/20 split lands at nearly the same absolute timestamp
regardless of which machine — a property of this trace's fixed shared
observation window, not something that would hold automatically in a
dataset where different machines' date ranges differ or overlap unevenly.

**Sequence construction: interleaved, not concatenated.** (X, y) sequences
are built independently per pool machine from that machine's own scaled
train array — no sliding window ever spans two machines — then the
resulting per-machine sequence arrays are merged and shuffled into one
pooled training set. This is the "interleaved" option: no single training
example is contaminated by a machine-boundary seam. The alternative
(concatenating raw per-machine series end-to-end before windowing) would
create a handful of nonsense windows whose lookback spans one machine's
tail and the next machine's head; windowing first and merging second
avoids that by construction, more simply than concatenating and then
having to drop seam windows.

**Scaling.** One shared `MinMaxScaler`, fit on the pooled TRAIN values only
(union of all 5 pool machines' own train splits), used to transform every
evaluation machine's train/val/test — in-pool or not — because the shared
model's weights are only meaningful in the scaled space they were trained
in. For out-of-cluster machines this means true values can fall outside
[0, 1] under this scaler (not clipped, same convention as every other input
path in this project).

**Two variants, 5 seeds each (a full pretrain-and/or-finetune cycle per
seed):**
- **(a) pooled, no fine-tuning**: the pretrained pooled model applied
  directly to each target machine's own val (decision-param tuning) and
  test (final score) splits.
- **(b) pooled + fine-tuned**: a copy of the pretrained pooled model, warm-started
  from its weights, continues training (`_train_lstm`, same
  `MAX_EPOCHS`/`ES_PATIENCE`, unchanged) on the target machine's own train
  split only, then evaluated the same way.

Run on all 13 feasible machines (5 in-pool, 8 out-of-pool — a genuine
out-of-cluster transfer test, not just within-cluster). Writes
`experiments/results_v13_pooled_lstm.csv` (130 rows) and
`experiments/results_v13_pooled_summary.csv` (13 rows).

## Before → After

### Own verdict: pooled variant's cost vs. that same machine's from-scratch baseline (Step 11)

| Machine | In pool? | Baseline | No fine-tune | No-ft verdict | Fine-tuned | Ft verdict |
|---|---|---:|---:|---|---:|---|
| m_2065 | ✓ | 0.2254±0.0111 | 0.2267±0.0103 | tied | 0.2428±0.0375 | tied |
| m_2104 | ✓ | 0.3046±0.0083 | 0.3011±0.0020 | directional better | 0.2945±0.0065 | directional better |
| m_2134 | ✓ | 0.9196±0.0302 | 0.9620±**0.1077** | tied | 0.9082±0.0237 | directional better |
| m_2183 | ✓ | 0.2971±0.0192 | 0.2971±0.0051 | tied | 0.3333±0.0229 | tied |
| m_2355 | ✓ | 0.3341±0.0031 | 0.3274±0.0069 | directional better | 0.3318±0.0042 | directional better |
| m_2056 |  | 0.1205±0.0065 | 0.3748±**0.3569** | tied | 0.1214±0.0170 | tied |
| m_2085 |  | 0.2129±0.0059 | 0.3548±**0.3614** | tied | 0.3942±0.0407 | **confirmed worse** |
| m_2087 |  | 0.0221±0.0000 | 0.2543±**0.4137** | tied | 0.0221±0.0000 | tied |
| m_2163 |  | 0.1488±0.0051 | 0.1766±0.0346 | tied | 0.1563±0.0010 | **confirmed worse** |
| m_2189 |  | 0.1391±0.0044 | 0.1289±0.0160 | directional better | 0.1333±0.0304 | directional better |
| m_2241 |  | 0.0110±0.0000 | 0.3558±**0.3751** | tied | 0.0110±0.0000 | tied |
| m_2380 |  | 0.2000±0.0064 | 0.2053±0.0135 | tied | 0.1965±0.0216 | directional better |
| m_2647 |  | 0.1925±0.0036 | 0.1788±0.0167 | directional better | 0.1841±0.0073 | directional better |

**No fine-tune: 0/13 confirmed better, 4/13 directional better, 9/13 tied,
0/13 confirmed worse. Fine-tuned: 0/13 confirmed better, 6/13 directional
better, 5/13 tied, 2/13 confirmed worse.** Neither variant produces a
single machine where pooling *confirms* an improvement over training from
scratch.

### Head-to-head vs. ARIMA (Step 11's numbers, same original engine)

| Variant | LSTM confirmed cheaper | ARIMA confirmed cheaper | Unconfirmed / tied |
|---|---:|---:|---:|
| Baseline (Step 11) | 1/13 (m_2085 — known confound) | 6/13 | 6/13 |
| Pooled, no fine-tune | 1/13 (m_2183 — see caveat below) | 2/13 | 10/13 |
| Pooled, fine-tuned | **0/13** | 5/13 | 8/13 |

## Impact

### The bold-looking no-fine-tune numbers hide a real instability, not a real improvement

The **out-of-cluster calm machines (m_2056, m_2085, m_2087, m_2241 — the
same 4-machine cluster from Step 10's m_2085 investigation, burstiness
1.0–1.6) show enormous no-fine-tune variance (σ up to 0.41)** — an order of
magnitude larger than anything else measured in this project. Tracing this
to individual seeds: **seed 42's pretrained pooled model produces cost
scores of 0.99–1.0 (catastrophic — near the worst structurally possible)
on all four of these machines simultaneously**, while seeds 43–46 give
much more reasonable costs (0.03–0.32) on the exact same four machines.
Seed 42's forecast RMSE on these machines (8.3–9.8) isn't dramatically
worse than its RMSE on in-pool machines (8.0–17.9) — the catastrophe is
mechanical, not a broken forecast: these four calm machines have a much
higher calibrated `demand_scale` (16–23×, vs. 2.6–3.3× for the pool
cluster) because `calibrate_demand_scale` is inversely proportional to a
machine's mean CPU%, and calm machines run at much lower mean utilization.
The same raw forecast noise gets amplified 5–9× more for these machines
before it reaches the decision engine, so a pretrained model whose
out-of-distribution behavior happens to be slightly biased for the calm
regime (as one pretraining run out of five was) turns a modest forecast
error into a catastrophic decision-engine cost. **This is a genuine new
failure mode introduced specifically by out-of-cluster transfer without
fine-tuning — not present in any prior step of this project, because every
prior LSTM was trained and evaluated on the same single machine's own
scale.**

Fine-tuning fixes this cleanly: all four calm machines' fine-tuned costs
return close to their from-scratch baseline (m_2087, m_2241: exact match,
0.0221 and 0.0110 respectively; m_2056: 0.1214 vs. baseline's 0.1205) —
warm-starting from pooled weights and then training on the target
machine's own data re-adapts the forecast to that machine's own scale
before evaluation, erasing most of the out-of-distribution artifact. This
means the *practical* recommendation, if pooling were adopted at all, is
unambiguous: **never deploy a pooled model to an out-of-cluster machine
without fine-tuning it first** — the no-fine-tune numbers are not a usable
end state on their own, so the fine-tuned variant is the one that actually
matters for the "does this help" question.

### The one new "confirmed" LSTM-vs-ARIMA win doesn't survive its own paired comparison

m_2183 (in-pool) shows `LSTM confirmed cheaper` under the no-fine-tune
variant — new relative to baseline (`LSTM directional`, not confirmed).
But this "confirmation" comes entirely from **variance shrinkage, not a
cost improvement**: the pooled no-fine-tune mean cost (0.297130) is
essentially identical to the baseline's mean (0.297130 to 6 decimal
places) — only the standard deviation shrank (0.0192 → 0.0051), which is
what pushes the fixed gap to ARIMA (0.0097) over the now-smaller combined-σ
threshold. And critically, **the fine-tuned variant on the same machine
reverses this entirely to `ARIMA confirmed cheaper`** (cost rises to
0.3333). A result that flips depending on which of two variants of the
same underlying model you evaluate, and that traces to a variance artifact
rather than a mean shift, is exactly the kind of fragile "win" this
project's Steps 10, 13, and 14 have repeatedly found doesn't survive
scrutiny — this one wasn't given the full cluster-swap/test-optimal audit
treatment (out of scope for this step), but should be treated with the
same skepticism as m_2056's and m_2134's unaudited flips from Step 14
rather than counted as a new data point.

### Pooling with fine-tuning: never confirmed worse than from-scratch, rarely confirmed better, and produces zero new LSTM-vs-ARIMA wins

Table 1's fine-tuned column is the honest bottom line for "does pooling
help, evaluated the way that actually matters (with fine-tuning, so the
instability above doesn't dominate the picture)": 6/13 machines improve
directionally (never confirmed), 5/13 are tied, and 2/13 (m_2085, m_2163)
are **confirmed worse** than training from scratch — including m_2085,
this project's one previously-"confirmed" LSTM-vs-ARIMA win, which
*degrades* under pooling+fine-tuning (cost 0.2129 → 0.3942, and its
head-to-head verdict against ARIMA drops from `LSTM confirmed cheaper` to
`ARIMA directional`). Zero machines show a confirmed head-to-head win over
ARIMA under the fine-tuned variant — down from 1/13 at baseline (itself a
known confound). **Pooling five times as much training data, while holding
architecture and decision engine fixed, does not produce a single
confirmed improvement anywhere in this project's evidence, and it actively
erodes the one previously-standing (already-confounded) LSTM advantage.**

### Honest conclusion

This is now the third distinct lever tried specifically to find a genuine
LSTM-specific advantage in this pipeline — after the multi-step decision
engine (Steps 12–13, retracted), multivariate input features (Step 14,
made things broadly worse), and now pooled training data (this step, no
net gain, new instability mode on out-of-cluster transfer). None of the
three produced a result that survived scrutiny. Combined with Step 6's
original finding that a 2-parameter ARIMA(2,0,1) statistically ties this
same 128-unit 2-layer LSTM on point-forecast RMSE, the straightforward
reading of this project's accumulated evidence is that **at this dataset's
scale (~1,300–2,300 points per machine, even pooled to ~7,000–9,000) and
for this forecasting task's actual complexity, the LSTM is not the right
tool to reach for over a simple ARIMA baseline** — not because it performs
badly in an absolute sense (it's usually close to ARIMA, sometimes
directionally ahead), but because none of the standard levers for
extracting more value from a neural forecaster (a smarter loss/decision
coupling, richer input features, more training data via pooling) have
produced a result this project's own significance bar treats as real. That
is a legitimate, substantive conclusion in its own right, not a failure to
find one.

## Still open

- m_2183's new no-fine-tune "confirmed" LSTM-vs-ARIMA win was not given the
  full val-grid-rank/test-optimal-sweep/cluster-swap audit Steps 13–14 gave
  every other new flip — given it reverses under the fine-tuned variant of
  the exact same underlying model and traces to variance shrinkage rather
  than a cost improvement, it should not be trusted without that same
  scrutiny, and the prior established pattern suggests it likely wouldn't
  survive it either.
- Fine-tuning was operationalized as "the same `_train_lstm` call,
  unchanged `MAX_EPOCHS`/`ES_PATIENCE`, just warm-started from pretrained
  weights" — the simplest, least-parameterized definition available, but
  not the only reasonable one. A smaller learning rate, frozen early
  layers, or a shorter epoch budget specifically for fine-tuning were not
  tried and might change the outcome.
- Only the single largest K-means cluster (5 machines) was pooled. Whether
  a smaller, tighter pool (e.g., just the 3-4 closest-to-centroid machines,
  excluding the stress-test outlier m_2134) or pooling across all 13
  feasible machines regardless of cluster would behave differently is
  untested.
- The out-of-cluster instability mechanism (demand_scale amplifying
  pretrained-model noise) was diagnosed for the no-fine-tune variant on 4
  machines from one cluster — whether it's a general property of any
  large demand_scale mismatch between pool and target, or specific to
  these particular machines/seeds, wasn't tested systematically (e.g., by
  deliberately varying demand_scale mismatch magnitude).
- SLA was recorded in `results_v13_pooled_lstm.csv` but not run through the
  same three-way "confirmed" classification the cost analysis used here.
- Containerization (Docker) and a real Kubernetes/KEDA deployment: still
  not started.
