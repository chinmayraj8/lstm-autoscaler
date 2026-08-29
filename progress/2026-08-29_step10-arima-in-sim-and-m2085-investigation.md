# Step 10: ARIMA wired into the simulation + the m_2085 anomaly explained
Date: 2026-08-29
Status: done

## What changed

1. **`src/autoscaler/arima_baseline.py` created.** Reuses Step 6's exact
   rolling-forecast procedure (`_evaluate_arima` in
   `experiments/run_baselines_and_sweep.py`: fit ARIMA on train only,
   advance state through val/test with `.append(refit=False)`, roll
   horizon-step-ahead forecasts) but returns the raw `(y_pred_real,
   y_actual_real)` arrays instead of just RMSE/MAE. Those arrays are the
   exact shape `_build_lstm_targets` (the decision engine, `decision.py`)
   already consumes — that function has no LSTM-specific logic in it, so
   ARIMA's forecasts now drive the identical decision engine and
   `_run_simulation`/`_compute_cost_score` (`simulation.py`) that the LSTM
   and Reactive policies use. Two public functions, mirroring
   `experiment.py`'s shape:
   - `tune_arima_on_validation(seed, machine_id, df_raw, demand_scale)` —
     grid-searches ARIMA's decision params on the validation split, using
     the *same* grids (`LSTM_UPW_GRID` × `LSTM_SM_GRID`) the LSTM is
     tuned with, so ARIMA is never tuned harder or easier than the LSTM.
     Never touches test.
   - `run_arima_experiment(seed, under_prov_weight, safety_margin,
     machine_id, df_raw, demand_scale)` — runs the tuned ARIMA through
     test, returns `arima_forecast_rmse/mae`, `arima_sla_violation_rate_pct`,
     `arima_over_prov_waste_pct`, `arima_cost_score`, mirroring
     `run_single_experiment`'s LSTM fields.
   - Registered as lazy imports in `src/autoscaler/__init__.py` (needs
     statsmodels, not TensorFlow, but follows the same "don't pay the
     import cost unless asked" pattern already used for the TF-dependent
     names).
   - **Determinism, stated explicitly and checked, not assumed**:
     statsmodels' ARIMA MLE fit has no random restarts and no seed
     dependence, so `run_arima_experiment` produces bit-identical output
     across repeated calls on the same data. `experiments/
     run_arima_in_sim.py` asserts this at runtime for every machine (re-runs
     ARIMA once and compares); the assertion held on all three machines
     below. Because of this, ARIMA is reported as a single deterministic
     number, not a fabricated "mean±std" over 5 identical values — the
     project's mean±std rule exists to average out LSTM's real stochasticity
     and Reactive's real absence of it; averaging 5 identical ARIMA runs
     would manufacture a false appearance of either variance or confidence,
     neither of which would be true.
2. **`tests/test_arima_baseline.py` added** (5 tests, all pass, no
   TensorFlow or real dataset needed — pure synthetic AR(1) series, small
   `order=(1,0,0)` to keep fits fast): rolling-forecast window count matches
   `_make_sequences`' convention, context points are excluded from scoring,
   determinism (the same claim above, checked on synthetic data too),
   prediction clipping to `[0,1]`, and the local `_inv_flat` inverse-scaling
   helper. **25 → 30 passing tests total.**
3. **`experiments/run_arima_in_sim.py` created**: runs the full LSTM +
   Reactive + ARIMA three-way comparison (tune-on-val, then 5-seed test
   evaluation for LSTM/Reactive, one deterministic test evaluation for
   ARIMA) on three machines and writes `experiments/
   results_v6_arima_in_sim.csv`:
   - **m_1933** — Step 6's original machine (smooth demand, burstiness=1.33),
     now under the project's current calibrated-demand-scale convention
     (`calibrate_demand_scale`, Step 5+) instead of Step 6's uncalibrated
     global `DEMAND_SCALE=20`, since this run exercises the full simulation
     rather than just forecast RMSE. **These m_1933 numbers are therefore
     not directly comparable to Steps 2–3's uncalibrated results for the
     same machine_id** — different demand_scale, different simulated
     capacity regime.
   - **m_2189** — bursty (burstiness=14.49), the clearest confirmed LSTM
     cost advantage over Reactive in Step 8.
   - **m_2134** — most bursty (burstiness=38.2), Step 8's stress-test case
     where Reactive won both metrics outright.
4. **m_2085 anomaly investigated** (ad hoc analysis, not a new permanent
   script — the commands are reproducible from this doc): a counterfactual
   Reactive-threshold swap, a validation-grid audit, a val→test
   distribution-shift check, and an LSTM sensitivity diagnostic, all run
   against the real dataset. See "Impact" below for the full chain of
   evidence.

## Before → After

### Task 1+2: ARIMA in the simulation (cost score + SLA rate), 3 machines

| Machine | Burstiness | Method | Forecast RMSE | SLA % | Cost score |
|---|---:|---|---:|---:|---:|
| **m_1933** (smooth) | 1.33 | ARIMA(2,0,1) | 0.5774 | 5.0773% | **0.2539** (deterministic) |
| | | LSTM (mean±std, 5 seeds) | 0.5798±0.0050 | 5.0773%±0.0000 | 0.2539±0.0000 |
| | | Reactive | — | 0.2208%±0.0000 | 0.2583±0.0000 |
| **m_2189** (bursty) | 14.49 | ARIMA(2,0,1) | 6.2994 | **0.8830%** | **0.1038** (deterministic) |
| | | LSTM (mean±std, 5 seeds) | 6.4393±0.0120 | 1.1479%±0.0883 | 0.1219±0.0035 |
| | | Reactive | — | 1.1038%±0.0000 | 0.2759±0.0000 |
| **m_2134** (most bursty, stress test) | 38.20 | ARIMA(2,0,1) | 17.3095 | 12.8035% | 0.9139 (deterministic) |
| | | LSTM (mean±std, 5 seeds) | 17.5731±0.0281 | **11.4349%±0.5654** | **0.8746±0.0210** |
| | | Reactive | — | **2.4283%±0.0000** | **0.6711±0.0000** |

Tuned params — Reactive `(up, down)` / LSTM `(under_prov_weight, safety_margin)` /
ARIMA `(under_prov_weight, safety_margin)`, all grid-searched on validation
only, same grids for LSTM and ARIMA:
m_1933: Reactive (80,10), LSTM (5, 0.10), ARIMA (5, 0.10).
m_2189: Reactive (75,30), LSTM (5, 0.30), ARIMA (5, 0.30).
m_2134: Reactive (75,20), LSTM (5, 0.40), ARIMA (5, 0.35).

Full per-seed rows: `experiments/results_v6_arima_in_sim.csv`.

### Task 3: m_2085 anomaly — evidence chain

| Check | m_2085 | m_2087 (cluster-mate) | m_2241 (cluster-mate) |
|---|---:|---:|---:|
| Reactive's tuned threshold (val-optimal) | up=90, down=10 | up=60, down=40 | up=60, down=40 |
| Val-grid rank of (90,10) | **1 / 49** (val_cost=0.348889) | 31 / 49 (tied at 0.0) | 20 / 49 |
| Val-grid rank of (60,40) | 41 / 49 (val_cost=0.977778) | **1 / 49** (val_cost=0.0) | **1 / 49** (val_cost=0.0) |
| Demand max, val split | 173.47% | 134.27% | 167.58% |
| Demand max, test split | 219.73% (**+26.6%**) | 163.42% (+21.7%) | 167.58% (**+0.0%**) |
| Reactive test cost, own tuned threshold | **0.286031** | 0.0 | 0.0 |
| Reactive test cost, counterfactual (60,40) | **0.099778** | (already 60,40) | (already 60,40) |
| LSTM test cost (mean, 5 seeds) | 0.2129 | 0.0221 | 0.0110 |
| LSTM test cost, best achievable on test (diagnostic peek, sm swept) | 0.1175 (sm=0.40, vs. its actual val-tuned sm=0.30 → 0.2040) | — | — |

## Impact

### ARIMA in the simulation: the LSTM's Step 8 cost advantage is not LSTM-specific

The headline result: **on m_2189 — the exact machine Step 8 called out as
the LSTM's clearest confirmed cost advantage — ARIMA beats the LSTM too**,
on both cost (0.1038 vs 0.1219±0.0035) and SLA (0.883% vs 1.148%±0.088),
using the identical decision engine and simulator. ARIMA also beats
Reactive by a wider margin than the LSTM does. This means Step 8's finding
should be reframed: **the cost advantage on m_2189 comes from having *any*
reasonable forecast feeding the decision engine, not from the LSTM
specifically.** A 60-year-old linear ARMA model does at least as well.
This directly answers what Step 6 left open ("the LSTM's advantage in the
autoscaler comes from the decision engine... on machines with sharper
demand spikes... the LSTM may outperform ARIMA — that comparison has not
been run") — and the answer, on m_2189, is that the LSTM does *not*
outperform ARIMA there.

On m_1933 (smooth demand), ARIMA and the LSTM land on nearly identical
autoscaling decisions: 452/453 and 450/453 test-window ticks respectively
choose the same 2-server target, differing on only 4 of 453 steps, and
their cost scores and SLA rates match to 4+ decimal places as a direct
consequence. This is consistent with — and extends into the simulation —
Step 6's original RMSE finding that ARIMA and LSTM are statistically tied
on smooth demand.

On m_2134 (the most bursty machine tested, Step 8's stress-test case), the
LSTM narrowly beats ARIMA on both metrics (cost 0.8746±0.021 vs 0.9139; SLA
11.43%±0.57 vs 12.80%) — the one place in this run where the LSTM shows any
edge over ARIMA. But **both forecast-driven policies lose decisively to
Reactive here** (cost 0.6711, SLA 2.43%), unchanged from Step 8's finding.
So the honest summary across all three machines: forecaster choice
(ARIMA vs LSTM) barely matters for the cost-vs-Reactive story at moderate
burstiness, and only starts to matter — modestly, in the LSTM's favor — at
the most extreme burstiness tested, where neither forecaster beats Reactive
anyway.

### m_2085: the "LSTM wins both metrics" result is a tuning-generalization artifact, not a demand-shape effect

Step 8 flagged m_2085 as the one machine (of 13) where LSTM won both cost
and SLA outright, unlike its cluster-mates m_2087/m_2241 (similar
burstiness, same cluster) where Reactive won both, and left the mechanism
as an open question.

The mechanism, established by direct counterfactual simulation on the real
test demand series: **Reactive's validation-tuned threshold for m_2085
(up=90, down=10) is legitimately val-optimal — rank 1 of 49 grid
combinations, by a wide, non-tied margin (val_cost=0.349 vs the runner-up
region) — but it generalizes catastrophically to test** (test cost=0.286,
SLA=4.88%). Re-running the *identical* test demand series through the
threshold m_2085's own cluster-mates converged to (up=60, down=40) gives
cost=0.0998 and SLA=2.00% — **both numbers better than the LSTM's actual
test performance on m_2085** (cost=0.213±0.008, SLA≈2.88%±0.16%). In other
words: give m_2085's Reactive policy the threshold its neighbors use, and
Reactive wins both metrics on m_2085 too, matching the cluster's dominant
pattern instead of breaking it.

Why does (90, 10) fail to generalize specifically for m_2085? Its test
split's demand ceiling grows sharply past what validation ever showed it:
max demand rises from 173.5% (val) to 219.7% (test), a 26.6% jump — bigger
than m_2087's (+21.7%) and far bigger than m_2241's, whose test max is
*identical* to its val max (+0.0%, no growth at all). A threshold tuned to
be slow both ways (wait for 90% load/server before adding a server, wait
for under 10% before removing one) is cheapest when the demand range stays
inside what it was tuned on; it under-provisions once the tail extends past
that range, which is exactly what happened to m_2085 and not to its
cluster-mates.

This generalization gap is not unique to Reactive: a diagnostic sweep of
the LSTM's own decision params against m_2085's actual test forecasts (not
used to change any reported number — a peek, for mechanism only) shows its
val-tuned choice (safety_margin=0.30) leaves real cost on the table too —
the best achievable on test would be safety_margin=0.40, cost=0.1175
vs. the val-tuned 0.2040. But the LSTM's gap (0.086 of cost left on the
table) is much smaller than Reactive's (0.186) on this machine, which is
why the LSTM's val-tuned choice still ends up ahead of Reactive's val-tuned
choice in the officially reported Step 8 numbers, even though neither
policy's tuning generalized perfectly.

**Conclusion: m_2085 is not evidence that the LSTM has some structural
edge tied to a demand-shape property Step 8's burstiness/mean/std features
failed to capture.** It is evidence that this project's single
train/val/test split — with no cross-validation on the threshold/margin
search — occasionally produces a validation-optimal Reactive threshold that
does not survive contact with test, on machines where the val and test
windows happen to sample meaningfully different demand ceilings. That
risk exists for the LSTM's tuning too, just less severely on this
particular machine. This is a second, distinct reason (alongside the
project's already-documented "training non-determinism") to treat any
single machine's verdict as directional rather than exact — and it
specifically undermines using m_2085 as a positive example of "when
does the LSTM win."

## Still open

- The m_2085 mechanism (val→test tail growth outpacing what the tuned
  Reactive threshold can absorb) was diagnosed on 3 machines. Whether it
  predicts which *other* machines will show anomalous single-machine
  verdicts — e.g. by computing a "val→test demand-ceiling growth" feature
  and checking whether it correlates with verdict flips across Step 8's
  13 machines — has not been tested.
- The LSTM sensitivity diagnostic on m_2085 (best-achievable-on-test
  safety_margin) was a one-off peek at one seed, not a systematic study.
  Whether the LSTM's tuning gap is *generally* smaller than Reactive's
  (a real hypothesis this step's evidence points toward) or whether m_2085
  is again a special case is unverified beyond this one machine.
- ARIMA has only been run through the simulation on 3 machines (m_1933,
  m_2189, m_2134) — not the other 10 feasible Step 8 machines. Whether
  "ARIMA matches or beats LSTM except at extreme burstiness" generalizes,
  the way Step 8 checked burstiness-vs-cost-advantage generalization for
  LSTM-vs-Reactive, is unknown.
- `run_arima_in_sim.py`'s m_1933 numbers use the current calibrated-demand
  convention and are not comparable to Steps 2–3's uncalibrated m_1933
  results. No attempt was made to reconcile or re-derive those old numbers
  under calibration.
- ARIMA's order (2,0,1) was reused unchanged from Step 6, not re-selected
  per machine (e.g. via AIC grid search) — consistent with the LSTM's
  architecture also not being tuned per machine, but worth flagging: a
  machine-specific ARIMA order might change the m_2134 result where ARIMA
  currently trails the LSTM.
- Containerization (Docker) and a real Kubernetes/KEDA deployment: still
  not started.
