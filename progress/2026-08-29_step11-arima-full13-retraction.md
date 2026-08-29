# Step 11: ARIMA across all 13 feasible machines — Step 8's "LSTM cost advantage" retracted as LSTM-specific
Date: 2026-08-29
Status: done

## What changed

**`experiments/run_arima_full13.py` created.** Extends Step 10's ARIMA
wiring to all 13 of Step 8's feasible machines (Step 10 only covered
m_1933, m_2189, m_2134). To keep this cheap and avoid re-litigating
already-settled tuning, LSTM and Reactive are **not** re-tuned or re-run:
their existing, correctly-tuned numbers (5 seeds each, tuned on validation
only) are pulled straight from `experiments/results_v5_expanded_multimachine.csv`
(Step 8) unchanged. Only ARIMA is new work per machine:
`tune_arima_on_validation` grid-searches ARIMA's decision params on
validation with the *same* grids the LSTM was tuned with
(`LSTM_UPW_GRID` × `LSTM_SM_GRID`), then `run_arima_experiment` runs the
tuned ARIMA once on test. Determinism (statsmodels ARIMA has no
seed-dependent randomness) is re-checked at runtime for every one of the
13 machines, not assumed from Step 10's earlier check — all 13 passed.

Writes:
- `experiments/results_v7_arima_full13.csv` — all 13 machines' existing
  results_v5 rows (65 rows, unchanged) plus 9 new `arima_*` columns.
- `experiments/results_v7_classification_summary.csv` — one row per
  machine: LSTM/Reactive/ARIMA cost and SLA means (±std where applicable),
  and the three pairwise verdicts (LSTM vs Reactive, LSTM vs ARIMA, ARIMA
  vs Reactive), each using the project's existing "confirmed" rule (gap >
  combined ±1σ, same rule as `run_multimachine_v2._verdict`).

Each machine is classified into exactly one bucket based on cost:
- **LSTM-specific advantage** — LSTM confirmed cheaper than *both*
  Reactive and ARIMA.
- **Forecaster-general advantage** — LSTM confirmed cheaper than Reactive,
  but ARIMA also matches or beats the LSTM (tied, directional, or
  confirmed cheaper) — i.e. any reasonable forecaster gets the win over
  Reactive, not something specific to the LSTM.
- **Reactive wins outright** — Reactive confirmed cheaper than both.
- (a residual "mixed" bucket exists in the code for anything that fits
  none of the above; it was empty on this run.)

## Before → After

### Full 13-machine classification (sorted by burstiness)

| Machine | Burstiness | LSTM cost | Reactive cost | ARIMA cost | LSTM vs ARIMA | Bucket |
|---|---:|---:|---:|---:|---|---|
| m_2241 | 1.00 | 0.0110±0.0000 | 0.0000±0.0000 | 0.0110 | tied | Reactive wins outright |
| m_2087 | 1.25 | 0.0221±0.0000 | 0.0000±0.0000 | 0.0221 | tied | Reactive wins outright |
| m_2085 | 1.50 | 0.2129±0.0059 | 0.2860±0.0000 | 0.3636 | **confirmed cheaper (LSTM)** | **LSTM-specific advantage** |
| m_2056 | 1.60 | 0.1205±0.0065 | 0.2362±0.0000 | 0.1038 | confirmed cheaper (ARIMA) | forecaster-general |
| m_2380 | 14.33 | 0.2000±0.0064 | 0.2119±0.0000 | 0.2053 | directional (LSTM, unconfirmed) | forecaster-general |
| m_2189 | 14.49 | 0.1391±0.0044 | 0.2759±0.0000 | 0.1038 | confirmed cheaper (ARIMA) | forecaster-general |
| m_2647 | 14.56 | 0.1925±0.0036 | 0.5055±0.0000 | 0.1832 | confirmed cheaper (ARIMA) | forecaster-general |
| m_2163 | 14.80 | 0.1488±0.0051 | 0.2230±0.0000 | 0.1435 | confirmed cheaper (ARIMA) | forecaster-general |
| m_2065 | 16.33 | 0.2254±0.0111 | 0.2984±0.0000 | 0.2004 | confirmed cheaper (ARIMA) | forecaster-general |
| m_2355 | 16.60 | 0.3341±0.0031 | 0.6548±0.0000 | 0.3229 | confirmed cheaper (ARIMA) | forecaster-general |
| m_2183 | 16.67 | 0.2971±0.0192 | 0.3731±0.0000 | 0.3068 | directional (LSTM, unconfirmed) | forecaster-general |
| m_2104 | 17.50 | 0.3046±0.0083 | 0.6004±0.0000 | 0.3002 | tied | forecaster-general |
| m_2134 | 38.20 | 0.9196±0.0302 | 0.6711±0.0000 | 0.9139 | tied | Reactive wins outright |

**Final counts: LSTM-specific advantage 1/13, forecaster-general advantage
9/13, Reactive wins outright 3/13, mixed 0/13.**

Cross-check against Step 8's original numbers: 1 + 9 = **10/13 machines
show a confirmed cost advantage over Reactive** — matches Step 8's
headline "10/13 feasible machines (77%) show a confirmed cost advantage"
exactly. The reactive-wins-outright set (m_2087, m_2241, m_2134) also
matches Step 8's "Reactive wins both metrics outright on 3/13" exactly.
**The set of machines and the count of the advantage are correct and
reproduced. What was wrong was attributing that advantage to the LSTM.**

Of the 9 forecaster-general machines, ARIMA is *confirmed* cheaper than
the LSTM outright on 6 (m_2056, m_2065, m_2163, m_2189, m_2355, m_2647),
tied on 1 (m_2104), and the LSTM has only a small, *unconfirmed*
directional edge on 2 (m_2183, m_2380) — never a confirmed LSTM edge on
any of these 9.

## Impact

### Step 8's headline claim is retracted as an LSTM-specific finding — this is the project's third retraction

Step 8 reported: *"LSTM cost advantage generalizes: 10/13 feasible
machines (77%) show a confirmed cost advantage, up from 2/4 in Step 5 —
not two lucky machines."* That statement is true about the existence and
size of the advantage over Reactive. **It is false about attribution.**
Running the exact same decision engine and simulator with ARIMA(2,0,1)
substituted for the LSTM — tuned with equal rigor, on the same grids,
never touching test during tuning — matches or beats the LSTM's cost on
9 of those 10 machines. On 6 of those 9 ARIMA is not just competitive but
*confirmed* cheaper than the LSTM.

That leaves exactly one machine, m_2085, where the LSTM is confirmed
cheaper than both Reactive and ARIMA. **Step 10 already established what
that one case is**: Reactive's validation-tuned threshold on m_2085
(up=90, down=10) is a legitimate rank-1-of-49 choice on validation that
generalizes catastrophically to test, because m_2085's test-split demand
ceiling grows 26.6% past what validation ever showed it — a bigger jump
than its cluster-mates saw. Swapping in the cluster-mates' threshold
(60/40) on m_2085's actual test demand produces a Reactive result
(cost=0.0998) that beats the LSTM's actual result (cost≈0.213) too. So the
one surviving "LSTM-specific advantage" is itself a Reactive
tuning-generalization artifact, not a demonstrated LSTM mechanism — Step
10's finding was published before this step ran, and this step's result
is fully consistent with it, not contradicted by it.

**Putting the two steps together: zero of Step 8's 13 feasible machines
demonstrate a cost advantage that is both (a) attributable to the LSTM
specifically rather than to "any reasonable forecaster," and (b) not
explainable by a known tuning-generalization confound.** The advantage
over Reactive is real, reproducible, and generalizes across burstiness
levels (this step reconfirms Step 8's 10/13 count exactly) — but it is a
property of *forecast-driven decision-making versus purely reactive
threshold control*, not a property of the LSTM. A model with two
autoregressive terms and one moving-average term, fit in under a second
with no GPU, gets the same result on 9 of the 10 machines where the LSTM
"wins."

This joins Step 3 ("LSTM advantage reversed... was an artifact of tuning
asymmetry") and Step 5 ("Step 4 LSTM SLA advantage retracted... was a
ceiling-collision artifact") as the project's third instance of an
LSTM-favorable headline result not surviving a more careful check. As
with those two, this is reported plainly rather than reframed to preserve
the original claim — that discipline is what makes the earlier two
retractions, and this one, worth trusting.

### What does survive: a forecast-driven decision engine beats Reactive, robustly

This is not a purely negative result. The pattern that DOES hold up:
whenever there's meaningful multi-step forecast signal (moderate-to-high
burstiness, 9/13 machines here), feeding *either* forecaster into this
project's greedy decision engine (`decision.py`) beats pure threshold-based
Reactive control on cost, often by a wide margin (e.g. m_2647: LSTM/ARIMA
≈0.18–0.19 vs Reactive 0.51; m_2355: ≈0.32–0.33 vs 0.65). That's a real,
now doubly-confirmed finding about the *decision engine's* value, distinct
from any claim about the LSTM's forecasting quality. On calm machines
(burstiness < 2, m_2087/m_2241/m_2056-adjacent) and at extreme burstiness
(m_2134), Reactive still wins outright or ties, unchanged from Step 8.

## Still open

- ARIMA's order (2,0,1) was reused unchanged across all 13 machines, never
  re-selected per machine (e.g. via AIC). A per-machine-tuned order could
  only strengthen ARIMA's showing further, not weaken it — so this doesn't
  threaten the conclusion above, but it means "ARIMA(2,0,1) matches the
  LSTM" is a lower bound on what a properly-tuned classical baseline could
  do.
- SLA was reported for context (`results_v7_classification_summary.csv`
  has `lstm_sla_mean`/`reactive_sla_mean`/`arima_sla`) but not formally
  classified into the same three-way bucket system this step used for
  cost — Reactive wins SLA on 12/13 machines (all except m_2189, where
  ARIMA has the lowest SLA), consistent with Step 8's established
  cost-vs-SLA trade-off pattern, but a full ARIMA-inclusive SLA
  classification (mirroring this step's cost analysis) wasn't done.
- Whether a different classical baseline (e.g. exponential smoothing,
  Prophet) or a simpler LSTM (fewer units/layers) would show the same
  "any reasonable forecaster suffices" pattern, or whether the current
  128-unit 2-layer LSTM is specifically over-provisioned for how easy
  these forecasting tasks are, is unexplored.
- Containerization (Docker) and a real Kubernetes/KEDA deployment: still
  not started.
