# Step 14: m_2380 fails its own swap test, ARIMA order re-selection, multivariate LSTM — net result is 0/13
Date: 2026-08-29
Status: done

## What changed

1. **Full Step-10-style confound audit on m_2380** (val-grid rank check,
   test-optimal sweep from its actual test forecast, cluster-mate
   counterfactual swap) — the same treatment Step 13 gave m_2183, extended
   to the one machine Step 13 left unaudited as "the strongest remaining
   candidate for a genuine LSTM-specific multistep advantage."

2. **`select_arima_order` added to `arima_baseline.py`**: grid-searches
   ARIMA(p,d,q) via AIC (or BIC) on each machine's TRAIN split only (p,q in
   0..4, d in {0,1}, 45 candidates — standard Box-Jenkins order selection,
   no leakage, never touches val/test). `experiments/run_arima_order_selection.py`
   runs this on all 13 feasible machines, then re-tunes+re-runs ARIMA under
   its per-machine best order through the exact "fair" setup Step 13
   established (`_build_multistep_targets` + `horizon_weights_grid`
   search), reusing LSTM (Step 13, results_v10) and Reactive (results_v7)
   numbers unchanged. Writes `results_v11_arima_order_selection.csv` (order
   search detail) and `results_v11_arima_reordered.csv` (13-row comparison).

3. **Multivariate LSTM input** (`data.py`: `_prepare_multivariate`,
   `_split_three_way_multivariate`, `_make_multivariate_sequences`;
   `forecasting.py`: `_build_lstm_model` gained an `n_features` param;
   `experiment.py`: `tune_on_validation`/`run_single_experiment` gained a
   `multivariate` param, default `False`, byte-identical behavior
   unchanged — confirmed by the full pytest suite). Adds time-of-day +
   day-of-week (cyclically sin/cos-encoded, deterministic from the
   timestamp — no leakage) and a short (6-step) rolling mean/std of the
   target feature as extra input channels — features ARIMA structurally
   can't use. Column 0 stays the original target feature, scaled by its
   own `MinMaxScaler` (fit on train only, exactly as before), so the
   existing inverse-scaling path is untouched; `y` is still built from
   that one channel only. Architecture (units, layers, dropout, optimizer),
   grids, and decision engine all held fixed — the only thing that changes
   is what the LSTM sees at its input. `experiments/run_multivariate_lstm.py`
   runs this (real retraining, 5 seeds, same fair multistep+horizon_weights
   setup) on all 13 machines, reusing Step 14's order-corrected ARIMA
   numbers and Reactive unchanged. Writes `results_v12_multivariate_lstm.csv`
   and `results_v12_multivariate_summary.csv`.

All three pieces of new package code (`select_arima_order`, the
multivariate data/model/experiment plumbing) are backward compatible by
construction — every new parameter defaults to the pre-existing behavior,
and the full 35-test pytest suite passes unchanged throughout.

## Before → After

### Task 1 — m_2380 confound audit (uniform-weights multistep engine, same checks as m_2183 in Step 13)

| Check | m_2380 result |
|---|---:|
| LSTM val-grid rank of tuned (5, 0.4) | 1/56, margin **0.0022** (thin — unlike m_2183's 0.0243) |
| ARIMA val-grid rank of tuned (5, 0.3) | 1/56, margin 0.0066 |
| LSTM: val-tuned param vs test-optimal | 0.1898 vs **0.1678** — (30, 0.35), **0.0221 left on the table** |
| ARIMA: val-tuned param vs test-optimal | 0.2230 vs **0.1766** — (5, 0.35), **0.0464 left on the table** (largest generalization gap seen in this project) |
| Head-to-head at own-tuned params | LSTM 0.1898 vs ARIMA 0.2230 → LSTM cheaper (reproduces Step 13's number) |
| Head-to-head at both test-optimal | LSTM 0.1678 vs ARIMA 0.1766 → LSTM cheaper, but margin nearly gone (0.0088) |
| Cluster-mate (m_2189, m_2163, m_2647) swap | **ARIMA cheaper in 2 of 3** (using m_2189's params: 0.2296 vs 0.2230; using m_2163's or m_2647's ARIMA sm=0.35: 0.1898 vs **0.1766**) |

Unlike m_2183 — which won under literally every counterfactual tried —
**m_2380 loses to ARIMA under 2 of its 3 cluster-mate parameter swaps**,
and both of its own val-tuned params leave real cost on the table on test
(LSTM: 0.022; ARIMA: 0.046, the single largest val→test generalization gap
measured anywhere in this project). The σ-margin that made m_2380 look
decisive in Step 13's horizon_weights table (gap/σ≈6.5) only reflects the
LSTM's own seed noise — it says nothing about how sensitive the *ARIMA
side* of the comparison is to which val-optimal grid cell the search
happened to land on, and that sensitivity is exactly what the swap test
exposes.

### Task 2 — ARIMA order re-selection (AIC on train, all 13 machines)

| Machine | Old order (rank/48) | New order (AIC-best) | Cost: old→new | Head-to-head: old→new |
|---|---|---|---:|---|
| m_2241 | (2,0,1), 12/48 | (2,0,3) | 0.0110→0.0110 | tied → tied |
| m_2087 | (2,0,1), 3/48 | (1,0,1) | 0.0221→0.0221 | tied → tied |
| m_2085 | (2,0,1), 17/48 | (3,1,3) | 0.3636→0.3348 | LSTM confirmed → LSTM confirmed |
| m_2056 | (2,0,1), 24/48 | (4,0,4) | 0.1038→0.1656 | **ARIMA confirmed → LSTM confirmed** |
| m_2380 | (2,0,1), 2/48 | (1,0,2) | 0.2230→0.2230 | LSTM confirmed → LSTM confirmed |
| m_2189/m_2647/m_2163/m_2065/m_2355/m_2183/m_2104 | (2,0,1), rank 1/48 (already AIC-optimal) or no cost change | unchanged or cost-neutral | — | unchanged |
| m_2134 | (2,0,1), 11/48 | (3,0,3) | 0.8565→0.9161 | **ARIMA confirmed → LSTM confirmed** |

**7/13 machines get a different AIC-best order; only 4/13 see the decision
cost actually move; only 2/13 (m_2056, m_2134) flip the head-to-head
verdict — and both flips go to ARIMA's *disadvantage* (its cost got
worse), not because a new LSTM strength emerged.**

**BIC robustness check on the 2 flipped machines**: m_2134's AIC-best
(3,0,3) and BIC-best (1,0,1) *disagree* — refitting under BIC's order gives
ARIMA cost 0.8698 (vs LSTM's 0.8980±0.0092, gap 0.028 > combined σ 0.0092)
→ **reverts to ARIMA confirmed cheaper**, undoing the AIC-driven flip
entirely. m_2056's AIC-best and BIC-best agree ((4,0,4) both), so this
particular check doesn't rescue it — but its val-grid margin is thin
(0.0022 of 224 cells) and its val→test demand direction runs the *opposite*
way from the classic generalization mechanism (test demand ceiling
*shrank* 18.2% from val, not grew), so a higher-order model overfitting
train's in-sample likelihood without any held-out check on rolling-forecast
quality is the more likely explanation than a genuine LSTM edge. Neither
flip is treated as confirmed evidence of anything LSTM-specific.

### Task 3 — multivariate LSTM (time-of-day/day-of-week + rolling mean/std, all 13 machines, 5 seeds)

| Machine | Burst | Univariate (Step 13) | Multivariate (Step 14) | Own verdict | Head-to-head w/ ARIMA |
|---|---:|---:|---:|---|---|
| m_2241 | 1.00 | 0.0110±0.0000 | 0.0177±0.0014 | confirmed **worse** | tied → ARIMA confirmed cheaper |
| m_2087 | 1.25 | 0.0221±0.0000 | 0.0221±0.0000 | tied | tied → tied |
| m_2085 | 1.50 | 0.2151±0.0020 | 0.2333±0.0466 | tied (huge new variance) | LSTM confirmed → LSTM confirmed |
| m_2056 | 1.60 | 0.1249±0.0077 | 0.2031±0.0266 | confirmed **worse** | LSTM confirmed → ARIMA confirmed |
| m_2380 | 14.33 | 0.1982±0.0038 | 0.2221±0.0208 | tied | LSTM confirmed → unconfirmed |
| m_2189 | 14.49 | 0.1280±0.0092 | 0.1395±0.0132 | tied | unconfirmed → ARIMA confirmed |
| m_2647 | 14.56 | 0.1868±0.0069 | 0.2278±0.0158 | confirmed **worse** | ARIMA confirmed → ARIMA confirmed |
| m_2163 | 14.80 | 0.1603±0.0041 | 0.2486±0.0588 | confirmed **worse** | ARIMA confirmed → ARIMA confirmed |
| m_2065 | 16.33 | 0.2290±0.0085 | 0.5096±0.0845 | confirmed **worse** | unconfirmed → ARIMA confirmed |
| m_2355 | 16.60 | 0.3350±0.0068 | 0.4410±0.0507 | confirmed **worse** | ARIMA confirmed → ARIMA confirmed |
| m_2183 | 16.67 | 0.3236±0.0106 | 0.7700±0.0726 | confirmed **worse** | unconfirmed → ARIMA confirmed |
| m_2104 | 17.50 | 0.3068±0.0104 | 0.3453±0.0294 | tied | ARIMA confirmed → ARIMA confirmed |
| m_2134 | 38.20 | 0.8980±0.0092 | **2.1351**±0.0487 | confirmed **worse** | LSTM confirmed → ARIMA confirmed |

**8/13 confirmed worse, 5/13 tied, 0/13 improved (confirmed or
directional). 0/13 new confirmed LSTM-vs-ARIMA wins.** Several previously
LSTM-favorable machines (m_2056, m_2065, m_2183, m_2134) flip to ARIMA's
favor purely because the multivariate LSTM got worse, not because ARIMA
improved.

## Impact

### m_2380 does not survive its own audit — 0/13 machines show a confirmed, confound-free LSTM-vs-ARIMA cost advantage

Combining this with Step 13: **of Step 12's original 5 "LSTM confirmed
cheaper than ARIMA under multistep" machines, none stand as clean evidence
of a genuine LSTM-specific structural advantage.** m_2189 and m_2183
already downgraded to unconfirmed under Step 13's horizon_weights fairness
check; m_2163 already reversed outright; m_2085 has been treated as a known
confound since Step 10 (a Reactive/tuning-generalization artifact, never a
clean win to begin with); and now **m_2380 — the one machine that survived
every check so far — fails its own cluster-mate swap test**, losing to
ARIMA under 2 of 3 counterfactual parameter substitutions and showing the
largest val→test generalization gap (0.046 of cost left on the table)
measured anywhere in this project. Its "confirmed" status in Step 13's
table reflected only the LSTM's own seed-to-seed noise floor, not how
sensitive ARIMA's side of the comparison was to which grid cell its
validation search happened to land on — exactly the kind of fragility the
swap test exists to catch, and did.

**Plainly: 0/13 machines in this project show a confirmed LSTM-vs-ARIMA
cost advantage once every confound found so far (val→test demand-ceiling
generalization, horizon_weights tuning-fairness, and now ARIMA
parameter-choice fragility exposed by cluster-mate swaps) is corrected
for.** Every apparent LSTM-specific edge this project has found, across
Steps 8–14, has either been explained away by a tuning-rigor asymmetry
between the two forecasters or has failed a direct robustness check when
one was applied.

### ARIMA order re-selection moves ARIMA's cost on only 4/13 machines, and where it moves the head-to-head verdict, it's not evidence of anything LSTM-specific either

Most machines' AIC-optimal order either matches (2,0,1) or produces an
economically identical decision-engine outcome despite a different order —
order mismatch alone rarely matters once forecasts get compressed through
the small discrete server-count decision space. The two machines where it
does move the needle (m_2056, m_2134) both move in ARIMA's *disadvantage*
— a higher-order, better-in-sample-fit model producing a *worse* rolling
multi-step-ahead forecast is the textbook overfitting risk of selecting
order via AIC/BIC alone, with no held-out check on actual forecast quality
(this project validates every other tuning choice — Reactive's threshold,
the LSTM's decision params, horizon_weights — on the validation split
specifically because in-sample criteria don't predict held-out performance;
order selection here used only train-split AIC/BIC, which is standard
Box-Jenkins practice but doesn't carry that same validation guarantee).
The BIC check on m_2134 makes this concrete: switching from AIC's (3,0,3)
to BIC's more parsimonious (1,0,1) — a completely defensible alternative
choice using the *same* train-only data — reverses the flip back to ARIMA's
favor. **Neither new "LSTM confirmed cheaper" verdict from order
re-selection should be trusted without the same audit treatment m_2380 just
failed**, and the BIC check on m_2134 already suggests it wouldn't survive
one either.

### Multivariate input made the LSTM broadly worse, not better — a genuine result, checked for bugs

Given the severity of the degradation (m_2134: 0.898→2.135, more than 2x;
m_2183: 0.324→0.770), this was checked for an implementation bug before
being reported as a finding: the multivariate feature frame has no NaNs,
correctly-scaled columns bounded in [0,1] (or [-1,1] for the cyclical
features), and a manually-run sanity check at plausible decision params
(upw=5, sm=0.4) on m_2134 produces a normal-looking forecast (RMSE 17.7,
comparable to the univariate LSTM's ~17.3–17.6 on the same machine) and a
normal-looking cost (0.843, close to the univariate baseline) — the
pipeline itself is not broken. The catastrophic m_2134 number specifically
traces to the validation grid search choosing `under_prov_weight=20` (the
*only* machine, out of 13 × this-and-Step-13's-prior-runs, where the search
picked anything other than the grid's minimum, 5) — a val-cost-minimizing
choice that happened to generalize disastrously to test, given
`under_prov_weight` directly multiplies the under-provisioning penalty. But
**12 of the 13 machines used the same `under_prov_weight=5` this project's
tuning has selected everywhere else, and 7 of those 12 still degraded
(several sharply) — so the m_2134 tuning fluke explains one machine's
magnitude, not the broad pattern.** The more likely general mechanism: this
project's own Step 6 finding that a 2-parameter ARIMA(2,0,1) statistically
ties a 128-unit 2-layer LSTM on point-forecast RMSE already suggested this
forecasting task has limited genuine complexity for a model this large to
exploit; adding 6 more input channels to a fixed-capacity, fixed-budget
model trained on only ~1,300–2,300 points per machine increases the
model's effective degrees of freedom without giving it more data or tuning
budget to use them well, and the added channels appear to have mostly
contributed noise/overfitting risk rather than exploitable signal, exactly
as the "give the LSTM features ARIMA structurally can't use" framing
predicted might happen if the extra signal ended up not being useful.
**0/13 machines show a new confirmed LSTM edge from this change — so it did
not need the Steps 10/13 confound-audit treatment, since there was no new
win to audit.**

## Still open

- **m_2056's ARIMA-order-driven "LSTM confirmed cheaper" flip was not given
  the full m_2380/m_2183 treatment** (no test-optimal sweep, no
  cluster-mate swap) — only a val-grid rank/margin check (thin, 1/224)
  and the AIC/BIC agreement check were done. Given the pattern established
  by every other flip audited so far, it should be treated as unconfirmed
  pending that same scrutiny, not as a new data point.
- ARIMA order selection used only train-split AIC/BIC (standard, no
  leakage, per the task) but never checked selected orders against
  held-out (validation-split) rolling-forecast RMSE the way this project
  validates every other tuning choice — a systematic gap between this
  step's order-selection rigor and the rest of the project's discipline.
  Worth closing before trusting any order-driven verdict change.
- The multivariate LSTM's degradation was diagnosed at a mechanistic level
  (fixed capacity/budget vs. added channels, one outlier tuning choice on
  m_2134) but not exhaustively — e.g., whether scaling the rolling-std
  channel differently, using a longer rolling window, or giving the model
  more capacity/epochs to compensate would change the outcome is untested.
  The task asked to hold architecture/tuning fixed specifically to isolate
  "does more signal help *as-is*" — it doesn't, but "could it help with
  appropriate capacity/regularization changes" is a different, unanswered
  question.
- SLA was reported for the multivariate run (it got dramatically worse
  alongside cost, e.g. m_2183: 13.16%) but not run through the same
  three-way "confirmed" classification the cost analysis used anywhere in
  this step.
- Containerization (Docker) and a real Kubernetes/KEDA deployment: still
  not started.
