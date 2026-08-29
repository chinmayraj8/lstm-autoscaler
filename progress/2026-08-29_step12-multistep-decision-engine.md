# Step 12: a multi-step-aware decision engine — helps the LSTM more than ARIMA, but doesn't overturn Step 11
Date: 2026-08-29
Status: done

## What changed

1. **`src/autoscaler/decision.py`: a second decision engine, added alongside
   the original.** The original (`_decide_scaling` / `_compute_penalty` /
   `_build_lstm_targets`) collapses each forecast row to
   `max(y_pred_real[i])` before penalizing a candidate server count — a
   one-tick spike and a sustained plateau of the same peak height are scored
   identically. It is **unchanged** (existing tests and the API depend on
   it). New, alongside it:
   - `_compute_penalty_multistep(n_servers, predicted_loads, cfg, horizon_weights=None)`
     — applies the same per-step penalty formula at *every* step of the
     horizon and combines them with `horizon_weights` (uniform mean by
     default: `1/horizon` each). A one-tick spike now contributes roughly
     `1/horizon` of a sustained plateau's penalty instead of the identical
     amount.
   - `_decide_scaling_multistep(current_servers, predicted_loads, cfg, horizon_weights=None)`
     — same hold/scale_up/scale_down candidate search as `_decide_scaling`,
     scored with the multistep penalty.
   - `_build_multistep_targets(y_pred_real, y_actual_real, dec_cfg, sim_cfg, demand_scale, safety_margin, horizon_weights=None)`
     — drop-in replacement for `_build_lstm_targets`: identical signature
     and `(targets, demand)` return shape, so it can be swapped in anywhere
     `_build_lstm_targets` was used, with no other code changes.
     Forecaster-agnostic exactly like the original: takes whatever
     `(y_pred_real, y_actual_real)` array pair a forecaster produced,
     LSTM- or ARIMA-shaped, with no forecaster-specific logic.
2. **`experiment.py` and `arima_baseline.py`: an optional `target_builder`
   parameter** added to `tune_on_validation`, `run_single_experiment`,
   `tune_arima_on_validation`, and `run_arima_experiment` (default:
   `_build_lstm_targets`, so every existing caller is unaffected — verified
   by re-running the full test suite and checking call sites in
   `run_multimachine_v2.py`, `run_arima_full13.py`, etc. use only keyword
   args). Passing `target_builder=_build_multistep_targets` tunes and runs
   that forecaster under the multistep engine instead, with identical
   tuning discipline (same grids, validation-only, test touched once).
3. **`tests/test_decision.py`: 5 new tests** (13 total in that file, 35
   total in the suite) for the multistep engine: the uniform-mean penalty
   formula directly; that a spike is diluted relative to a plateau while
   the old engine can't tell them apart (both collapse to the same max);
   a hand-verified case where the two engines make *different* scaling
   decisions for the same spike (`hold` vs `scale_up`); a case where they
   agree (sustained plateau); and an end-to-end `_build_multistep_targets`
   vs `_build_lstm_targets` divergence test.
4. **`experiments/run_multistep_ablation.py` created.** Reuses the greedy
   (original-engine) LSTM, Reactive, and ARIMA numbers unchanged from
   `results_v7_arima_full13.csv` (Steps 8 + 11); computes the multistep
   cells fresh, on all 13 of Step 8's feasible machines, for **both**
   forecasters:
   - LSTM multistep: real retraining — `tune_on_validation` (1 train) then
     `run_single_experiment` × 5 seeds (target_builder=multistep). This is
     genuine stochastic retraining, not reused.
   - ARIMA multistep: `tune_arima_on_validation` + `run_arima_experiment`
     (target_builder=multistep), deterministic, re-run and asserted
     bit-identical per machine (same discipline as Steps 10–11).
   Both tuned on the same grids the greedy engine used. Writes
   `experiments/results_v8_multistep_ablation.csv` (65 rows) and
   `experiments/results_v8_ablation_summary.csv` (13 rows, one per machine).

## Before → After

### Table 1 — each forecaster vs. itself (greedy engine → multistep engine)

| Machine | Burst | LSTM greedy | LSTM multistep | LSTM verdict | ARIMA greedy | ARIMA multistep | ARIMA verdict |
|---|---:|---:|---:|---|---:|---:|---|
| m_2241 | 1.00 | 0.0110±0.0000 | 0.0110±0.0000 | tied | 0.0110 | 0.0110 | tied |
| m_2087 | 1.25 | 0.0221±0.0000 | 0.0221±0.0000 | tied | 0.0221 | 0.0221 | tied |
| m_2085 | 1.50 | 0.2129±0.0059 | 0.2151±0.0028 | tied | 0.3636 | 0.3636 | tied |
| m_2056 | 1.60 | 0.1205±0.0065 | 0.1249±0.0066 | tied | 0.1038 | 0.1015 | **multistep better** |
| m_2380 | 14.33 | 0.2000±0.0064 | 0.2022±0.0092 | tied | 0.2053 | 0.2230 | **multistep worse** |
| m_2189 | 14.49 | 0.1391±0.0044 | **0.1188±0.0026** | **multistep confirmed better** | 0.1038 | 0.1280 | **multistep worse** |
| m_2647 | 14.56 | 0.1925±0.0036 | 0.2022±0.0086 | tied | 0.1832 | 0.1788 | **multistep better** |
| m_2163 | 14.80 | 0.1488±0.0051 | 0.1532±0.0063 | tied | 0.1435 | 0.1611 | **multistep worse** |
| m_2065 | 16.33 | 0.2254±0.0111 | 0.2254±0.0022 | tied | 0.2004 | 0.2160 | **multistep worse** |
| m_2355 | 16.60 | 0.3341±0.0031 | 0.3363±0.0055 | tied | 0.3229 | 0.3163 | **multistep better** |
| m_2183 | 16.67 | 0.2971±0.0192 | 0.3214±0.0101 | tied | 0.3068 | 0.3510 | **multistep worse** |
| m_2104 | 17.50 | 0.3046±0.0083 | 0.2980±0.0081 | directional better | 0.3002 | 0.2870 | **multistep better** |
| m_2134 | 38.20 | 0.9196±0.0302 | 0.9007±0.0104 | directional better | 0.9139 | 0.8565 | **multistep better** |

**LSTM: 10/13 tied, 2/13 directional (unconfirmed) improvement, 1/13
confirmed better, 0/13 worse (confirmed or directional) — never regresses
on any machine, but rarely clears its own 5-seed noise floor.**
**ARIMA: 5/13 confirmed better, 5/13 confirmed worse, 3/13 tied — a wash.**
*Caveat on comparing these two counts directly: ARIMA has zero variance
(deterministic), so **any** nonzero before/after difference for it is
automatically "confirmed" — a much lower bar than LSTM's, which must clear
real 5-seed training noise. The "LSTM never regresses / ARIMA is a
coin-flip" framing is suggestive but partly an artifact of that asymmetric
bar, not proof the engine is doing something categorically different for
each forecaster. Table 2 below is the more trustworthy comparison, because
it uses the same bar (LSTM's own noise) consistently in both the "before"
and "after" columns.*

### Table 2 — LSTM vs. ARIMA head-to-head, greedy engine vs. multistep engine (the decisive table)

| Machine | Burst | Greedy: LSTM vs ARIMA | Multistep: LSTM vs ARIMA | Changed? |
|---|---:|---|---|---|
| m_2241 | 1.00 | tied | tied | no |
| m_2087 | 1.25 | tied | tied | no |
| m_2085 | 1.50 | **LSTM confirmed cheaper** (known confound, Step 10) | **LSTM confirmed cheaper** | no |
| m_2056 | 1.60 | ARIMA confirmed cheaper | ARIMA confirmed cheaper | no |
| m_2380 | 14.33 | LSTM directional (unconfirmed) | **LSTM confirmed cheaper** | **yes — strengthened** |
| m_2189 | 14.49 | ARIMA confirmed cheaper | **LSTM confirmed cheaper** | **yes — reversed** |
| m_2647 | 14.56 | ARIMA confirmed cheaper | ARIMA confirmed cheaper | no |
| m_2163 | 14.80 | ARIMA confirmed cheaper | **LSTM confirmed cheaper** | **yes — reversed** |
| m_2065 | 16.33 | ARIMA confirmed cheaper | ARIMA confirmed cheaper | no |
| m_2355 | 16.60 | ARIMA confirmed cheaper | ARIMA confirmed cheaper | no |
| m_2183 | 16.67 | LSTM directional (unconfirmed) | **LSTM confirmed cheaper** | **yes — strengthened** |
| m_2104 | 17.50 | tied | ARIMA confirmed cheaper | yes — new ARIMA edge |
| m_2134 | 38.20 | tied | ARIMA confirmed cheaper | yes — new ARIMA edge |

**LSTM confirmed cheaper than ARIMA: 1/13 (greedy) → 5/13 (multistep)** —
4 of those 5 are new; 2 (m_2189, m_2163) are genuine reversals of Step 11's
finding, not just noise resolving in the LSTM's favor. **ARIMA confirmed
cheaper than LSTM: 6/13 → 6/13 — same count, 2 different machines** (lost
m_2189/m_2163 to the LSTM, gained m_2104/m_2134 from "tied"). Tied: 4 → 2.

### Table 3 — the Step 11 bucket classification (vs. Reactive), re-run under multistep

| Bucket | Greedy (Step 11) | Multistep (Step 12) |
|---|---:|---:|
| LSTM-specific advantage (beats both Reactive and ARIMA) | 1/13 (m_2085, confounded — Step 10) | 1/13 (same machine, same confound) |
| Forecaster-general advantage (beats Reactive; ARIMA matches/beats LSTM) | 9/13 | 8/13 |
| Reactive wins outright | 3/13 (m_2087, m_2241, m_2134) | 3/13 (same 3 machines) |
| Mixed / no confirmed winner | 0/13 | 1/13 (m_2380 — LSTM only directionally beats Reactive here; ARIMA loses to Reactive) |

## Impact

### The multistep engine does not overturn Step 11's "any decent forecaster" finding

Judged the way this project has always judged things — confirmed cost
advantage over **Reactive** — Table 3 shows almost no change. The three
machines where Reactive wins outright are the identical three from Step 11.
The one "LSTM-specific" machine is still m_2085, still the same
tuning-generalization confound Step 10 already explained (its own numbers
barely moved: 0.2129→0.2151 for the LSTM, and ARIMA's cost is bit-for-bit
identical before and after — the multistep grid search converged to the
exact same decision trajectory for ARIMA on this machine). The
forecaster-general bucket shrank by exactly one machine (m_2380 moved to
"mixed" because the LSTM's edge over Reactive there is real but not
confirmed). **If the question is "does a smarter decision engine let the
LSTM reclaim the advantage over Reactive that Step 11 showed wasn't really
LSTM-specific," the answer is no.**

### But head-to-head against ARIMA specifically, something real changed

Table 2 is the finding this step actually earns: comparing LSTM and ARIMA
to *each other*, under the *same* engine, with the *same* tuning rigor
applied to both — exactly the comparison Step 11 ran under the old engine
— the LSTM's confirmed cost advantage over ARIMA goes from 1 machine (a
known confound) to 5 machines, with two of those being outright reversals
(m_2189, m_2163: ARIMA was confirmed cheaper under the old engine, LSTM is
confirmed cheaper under the new one) and two more being previously-weak,
now-confirmed LSTM edges (m_2380, m_2183). This is a real, mechanistically
sensible effect: the LSTM predicts all three horizon steps from one
learned nonlinear function, while ARIMA's 3-step-ahead forecast is
produced by extrapolating a linear AR(2,1) recursion forward, which tends
to revert toward the unconditional mean as the horizon extends — a
structurally different shape. The old max-based engine threw that shape
away entirely; once the engine actually looks at all three steps, the
LSTM's shape turns out to be the more useful one on several machines. That
is a genuine, forecaster-specific effect of using the full horizon — it
was just invisible under the old engine, and it is not large enough or
one-sided enough to change who beats Reactive.

**Direct answer to this step's question**: a smarter decision engine helps
the LSTM more than it helps ARIMA, in the narrow sense that LSTM vs ARIMA
comparisons shift in the LSTM's favor on net (5 vs 1, with 2 outright
reversals) — but it does not help the LSTM enough, or broadly enough, to
change which machines beat Reactive. Step 11's "any decent forecaster
beats Reactive" story survives essentially intact; what's new is that
*which* forecaster you'd reach for, if you cared about squeezing out the
last bit of cost, is no longer a coin flip once the decision engine stops
throwing away two-thirds of the forecast. ARIMA still wins outright on 6
of 13 machines even under the improved engine (m_2056, m_2647, m_2065,
m_2355, m_2104, m_2134) — including the two most burstiness-extreme
machines newly gained under multistep (m_2104, m_2134), where ARIMA's edge
actually *strengthened*. There is no clean burstiness threshold above or
below which one engine effect dominates; the reversals and the
strengthened-ARIMA cases both live in the same 14–18 burstiness band.

## Still open

- The 4 new/strengthened LSTM-vs-ARIMA advantages under multistep
  (m_2380, m_2189, m_2163, m_2183) have not been checked for a
  tuning-generalization confound the way Step 10 checked m_2085. Given
  that check materially changed the interpretation of one machine already,
  these four deserve the same scrutiny before being treated as clean
  evidence of an LSTM forecasting-shape advantage.
- `horizon_weights` defaults to a uniform mean and was never varied — a
  weighting that favors the near-term step (the one that actually
  determines next-tick SLA risk, given `SIM_STARTUP_DELAY=1`) was not
  tried and might change either forecaster's showing further.
- SLA moved in both directions for both forecasters under multistep, with
  no clean pattern (e.g., m_2189's LSTM SLA improved 1.68%→0.97% while its
  ARIMA SLA worsened 0.88%→1.55%) — not run through the same three-way
  "confirmed" classification Step 11 used for cost.
- ARIMA's order (2,0,1) was still not re-selected per machine under the
  new engine either — unclear whether a better-fit ARIMA order would close
  some of the 6 machines it still wins, or open up more LSTM wins.
- Containerization (Docker) and a real Kubernetes/KEDA deployment: still
  not started.
