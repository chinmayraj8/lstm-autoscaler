# Step 13: confound audit of Step 12's multistep wins + horizon_weights search — most of the "LSTM beats ARIMA under multistep" finding retracts
Date: 2026-08-29
Status: done

## What changed

1. **`experiments/check_val_test_gap.py` + `experiments/results_v9_val_test_gap.csv`**
   (already present untracked, reviewed and committed as Step 13a). Extends
   Step 10's m_2085 val→test demand-ceiling-growth diagnostic to all 13
   feasible machines. Reviewed for correctness: reproduces its own committed
   CSV byte-for-byte on re-run, and its sanity check against Step 10's
   reported m_2085/m_2087/m_2241 numbers matches (m_2085 growth computed
   here as 26.7% vs Step 10's reported 26.6% — a rounding artifact:
   173.47→219.73 computes to 26.667%, not a bug).

2. **Confound audit (ad hoc, Step-10-style — not a permanent script, see
   "Impact" for the full evidence chain) of the four machines Step 12
   flagged as needing scrutiny before their new/strengthened multistep
   LSTM-vs-ARIMA wins are treated as real.** Per the task, full audit on
   m_2183, lighter-touch on m_2189/m_2163 (m_2380 out of this round's
   scope — see "Still open"). Three checks per machine, mirroring Step 10's
   m_2085 evidence chain: (a) validation-grid rank/margin check for the
   tuned LSTM and ARIMA multistep decision params, (b) the val→test
   demand-growth numbers from check 1, (c) for m_2183 only, a full
   (under_prov_weight × safety_margin) TEST-cost grid built directly from
   each forecaster's actual test forecast (no retraining per cell), plus a
   counterfactual swap using each of m_2183's cluster-0 mates'
   (m_2065/m_2355/m_2104/m_2134) tuned params instead of its own.

3. **`horizon_weights_grid` (new optional parameter) on `tune_on_validation`
   (experiment.py) and `tune_arima_on_validation` (arima_baseline.py).**
   Adds `horizon_weights` as a third grid dimension alongside
   `under_prov_weight` × `safety_margin`, only meaningful for
   `_build_multistep_targets` (decision.py, Step 12), whose `horizon_weights`
   param it already accepted but Step 12 never varied. Backward compatible
   by construction: default `horizon_weights_grid=None` behaves
   byte-identically to before this parameter existed (single implicit
   candidate `None` → target_builder's own default → `_build_lstm_targets`
   callers, which don't accept `horizon_weights` at all, are never passed
   the kwarg) — confirmed by the full 35-test pytest suite passing
   unchanged. `run_single_experiment` / `run_arima_experiment` gained a
   matching `horizon_weights` parameter to run the tuned winner on test.
   Four candidate schemes tried (`HORIZON_STEPS=3`, all sum to 1.0):
   `uniform` [.333,.333,.333] (Step 12's default, included as a reachable
   grid-search outcome, not assumed away), `linear_321` [.5,.333,.167],
   `front_60_30_10` [.6,.3,.1], `front_80_15_05` [.8,.15,.05] — all
   front-load the near-term step, motivated by `SIM_STARTUP_DELAY=1`
   (a scale-up decided this tick only becomes active next tick, so the
   near-term forecast step determines next-tick SLA risk most directly).

4. **`experiments/run_horizon_weights_search.py` created and run on all 13
   feasible machines, for both LSTM (real retraining, 5 seeds) and ARIMA
   (deterministic, re-run and asserted bit-identical — same discipline as
   Steps 10-12), through `_build_multistep_targets`.** Same tuning
   discipline as everything else: grid-searched on validation only (now a
   7×8×4=224-cell grid), test touched once. Writes
   `experiments/results_v10_horizon_weights.csv` (65 rows) and
   `experiments/results_v10_horizon_weights_summary.csv` (13 rows).

## Before → After

### Confound audit — m_2183 (full), m_2189/m_2163 (light), uniform-weights multistep engine

| Check | m_2183 | m_2189 | m_2163 |
|---|---:|---:|---:|
| LSTM val-grid rank of tuned (upw, sm) | 1/56, margin **0.0243** (clear) | 1/56, margin 0.0022 (thin) | 1/56, margin 0.0110 (moderate) |
| ARIMA val-grid rank of tuned (upw, sm) | 1/56, margin 0.0022 (thin) | 1/56, margin 0.0022 (thin) | 1/56, margin 0.0044 (thin) |
| Val→test demand growth (max / p99) | +13.6% / **+26.8%** | +0.3% / −3.7% | +4.7% / −1.9% |
| Closest cluster-mate's p99 growth | m_2065: **+33.6%** (higher, no reversal there) | n/a (near-zero) | n/a (low) |
| LSTM: val-tuned param vs test-optimal | **identical** (0 cost left on table) | not swept (light audit) | not swept (light audit) |
| ARIMA: val-tuned param vs test-optimal | 0.351 vs 0.333 (0.018 left on table) | not swept | not swept |
| Head-to-head at own-tuned params (test) | LSTM 0.3046 vs ARIMA 0.3510 → **LSTM cheaper** | (existing) 0.1188±0.0026 vs 0.1280 → gap/σ≈3.6 | (existing) 0.1532±0.0063 vs 0.1435 → gap/σ≈1.25 (thinnest) |
| Cluster-mate counterfactual swap (4 tried) | **LSTM cheaper in all 4** | not run | not run |
| **Verdict on classic tuning-generalization confound** | **not this confound** — survives every check | demand-growth low risk; margin decisive | demand-growth low risk; margin thinnest of the three from the start |

(m_2183's own fresh LSTM retrain landed at test cost 0.3046, not the
0.3113 results_v8 reports for seed 42 — expected, attributable to this
project's already-documented GPU/Metal training non-determinism, not a bug:
ARIMA, which is deterministic, reproduced its officially reported 0.350993
bit-for-bit in this independent script, which is what validates the
audit's methodology rather than the exact LSTM number.)

### Table 1 — horizon_weights search: each forecaster vs. its own uniform-weights multistep baseline

| Machine | Burst | LSTM: uniform → best-of-4 | LSTM chosen scheme | ARIMA: uniform → best-of-4 | ARIMA chosen scheme |
|---|---:|---:|---|---:|---|
| m_2241 | 1.00 | 0.0110→0.0110 tied | uniform | 0.0110→0.0110 tied | uniform |
| m_2087 | 1.25 | 0.0221→0.0221 tied | uniform | 0.0221→0.0221 tied | uniform |
| m_2085 | 1.50 | 0.2151→0.2151 tied | linear_321 | 0.3636→0.3636 tied | uniform |
| m_2056 | 1.60 | 0.1249→0.1249 tied | front_80_15_05 | 0.1015→0.1038 **worse** | front_60_30_10 |
| m_2380 | 14.33 | 0.2022→0.1982 directional better | linear_321 | 0.2230→0.2230 tied | uniform |
| m_2189 | 14.49 | 0.1188→0.1280 tied (nominally worse) | front_80_15_05 | 0.1280→0.1258 **confirmed better** | linear_321 |
| m_2647 | 14.56 | 0.2022→0.1868 directional better | uniform | 0.1788→0.1788 tied | uniform |
| m_2163 | 14.80 | 0.1532→0.1603 tied (nominally worse) | uniform | 0.1611→0.1369 **confirmed better** | front_80_15_05 |
| m_2065 | 16.33 | 0.2254→0.2290 tied | front_80_15_05 | 0.2160→0.2294 **worse** | front_60_30_10 |
| m_2355 | 16.60 | 0.3363→0.3350 directional better | front_80_15_05 | 0.3163→0.3185 **worse** | linear_321 |
| m_2183 | 16.67 | 0.3214→0.3236 tied | linear_321 | 0.3510→0.3311 **confirmed better** | front_80_15_05 |
| m_2104 | 17.50 | 0.2980→0.3068 tied | uniform | 0.2870→0.2870 tied | uniform |
| m_2134 | 38.20 | 0.9007→0.8980 directional better | uniform | 0.8565→0.8565 tied | uniform |

**LSTM: 0/13 confirmed better, 4/13 directional (unconfirmed) better, 9/13
tied, 0/13 worse — never regresses, rarely clears its own noise floor
(same pattern Step 12's Table 1 already showed for greedy→multistep).**
**ARIMA: 3/13 confirmed better (m_2163, m_2189, m_2183 — exactly the three
audited above), 3/13 confirmed worse (m_2056, m_2065, m_2355 — cases where
the val-chosen scheme didn't generalize to test, the same
tuning-generalization theme Step 10 documented, now showing up on a new
grid dimension), 7/13 tied.** Uniform remains the single most common
winner for both forecasters (6/13 LSTM, 7/13 ARIMA) but a front-loaded
scheme wins outright on 7/13 (LSTM) and 6/13 (ARIMA) — front-loading is a
real, non-trivial improvement lever, not a non-effect.

### Table 2 — LSTM vs. ARIMA head-to-head, uniform engine vs. fair horizon_weights search (properly σ-gated)

Using this project's standard rule (gap over the combined ±1σ, ARIMA's σ=0
since it's deterministic) applied *consistently* to both the uniform-only
comparison and the horizon_weights-search comparison — not the raw
mean-cheaper label, which overstates several of these:

| Machine | Burst | Uniform engine (Step 12) | Fair horizon_weights search (Step 13) | Changed? |
|---|---:|---|---|---|
| m_2241 | 1.00 | tied | tied | no |
| m_2087 | 1.25 | tied | tied | no |
| m_2085 | 1.50 | **LSTM confirmed** (known confound) | **LSTM confirmed** (same confound) | no |
| m_2056 | 1.60 | ARIMA confirmed | ARIMA confirmed | no |
| m_2380 | 14.33 | **LSTM confirmed** | **LSTM confirmed** (gap/σ≈6.5) | no |
| m_2189 | 14.49 | **LSTM confirmed** | **downgraded to unconfirmed** (gap/σ≈−0.24, essentially a toss-up) | **yes — no longer confirmed** |
| m_2647 | 14.56 | ARIMA confirmed | ARIMA confirmed | no |
| m_2163 | 14.80 | **LSTM confirmed** | **reversed — ARIMA confirmed** (gap/σ≈−5.7) | **yes — full reversal** |
| m_2065 | 16.33 | ARIMA confirmed | **downgraded to unconfirmed** (gap/σ≈0.05, a coin flip) | **yes — no longer confirmed** |
| m_2355 | 16.60 | ARIMA confirmed | ARIMA confirmed | no |
| m_2183 | 16.67 | **LSTM confirmed** | **downgraded to unconfirmed** (gap/σ≈0.71, LSTM nominally still ahead but not significant) | **yes — no longer confirmed** |
| m_2104 | 17.50 | ARIMA confirmed | ARIMA confirmed | no |
| m_2134 | 38.20 | ARIMA confirmed | ARIMA confirmed | no |

**LSTM confirmed cheaper than ARIMA: 5/13 (Step 12, uniform) → 2/13 (Step
13, fair horizon_weights) — only m_2085 (known confound) and m_2380
survive.** ARIMA confirmed cheaper: 6/13 → 6/13 (same count, different
membership: loses m_2065, gains m_2163). Unconfirmed/tied: 2/13 → 5/13.

## Impact

### The classic val→test generalization confound: cleared for m_2183, plausibly not the mechanism for m_2189/m_2163 either

Applying Step 10's exact m_2085 playbook to m_2183 gives the opposite
verdict from m_2085: m_2183's val-tuned LSTM param is *also* the test-optimal
param (zero generalization gap, vs. m_2085's Reactive threshold leaving
0.186 of cost on the table), and the LSTM beats ARIMA on m_2183's actual
test data under every parameter choice tried — its own tuning, ARIMA's own
best-case test-optimal tuning, and four different cluster-mates' tuned
choices (none of which are derived from m_2183's own validation split).
The val→test demand-ceiling-growth number that flagged m_2183
(+26.8% p99, the highest of the 13) turns out **not** to predict this
particular mechanism either: m_2183's closest cluster-mate by burstiness,
m_2065, has an even higher p99 growth (+33.6%) with no LSTM-vs-ARIMA
reversal at all under the uniform engine — undermining demand-growth as the
explanation, even though m_2183's own numbers looked superficially similar
to m_2085's story. m_2189 and m_2163 show near-zero val→test demand growth
(±5%), so this specific confound mechanism essentially doesn't apply to
either of them. **Conclusion: none of the three audited machines' original
uniform-engine LSTM-vs-ARIMA verdicts are explained by Step 10's
tuning-generalization mechanism.** That mechanism is real (m_2085) but does
not generalize to Step 12's other flagged machines.

### But a different, more consequential confound was hiding in Step 12's methodology: `horizon_weights` was fixed to uniform for both forecasters

This is the headline finding of this step. Step 12's comparison held
`horizon_weights` fixed at uniform for both LSTM and ARIMA and concluded the
LSTM's forecast shape benefits more from a multistep engine because ARIMA's
linear extrapolation reverts toward the mean across the horizon. That
framing implicitly assumed uniform averaging is the "neutral" choice for
both forecasters — but it isn't: **a front-loaded weighting is exactly the
kind of engine adjustment that offsets ARIMA's mean-reversion tendency**, by
counting the near-term step (where ARIMA's forecast hasn't reverted much
yet) more heavily than the reverted far steps. Once ARIMA is given the same
grid-search freedom over `horizon_weights` that this step also gave the
LSTM — same discipline, same four candidate schemes, tuned on validation
only — **ARIMA's cost drops enough on 3 of the 4 previously-flagged
machines to erase or reverse the LSTM's edge**: m_2163 fully reverses
(ARIMA confirmed cheaper, gap/σ≈5.7 — decisive), and m_2189/m_2183 both
downgrade from confirmed to statistical toss-ups. Only m_2380 survives with
its LSTM-confirmed-cheaper verdict intact (gap/σ≈6.5, and here it's the
LSTM's own cost that improved under a front-loaded scheme, not ARIMA's that
failed to). A third, previously-unflagged machine (m_2065) also flips from
"ARIMA confirmed cheaper" to an unconfirmed coin-flip, driven by ARIMA's
val-chosen `horizon_weights` scheme generalizing *worse* to test than
uniform did there — the same tuning-generalization-gap phenomenon Step 10
described, now showing up on this new grid dimension instead of Reactive's
threshold.

**Net honest count: of Step 12's 5 "LSTM confirmed cheaper than ARIMA under
multistep" machines, only 2 remain confirmed after this step's two audits —
m_2085 (already a known confound, Step 10) and m_2380 (not yet
confound-audited this round, see "Still open"). m_2189 and m_2183 downgrade
to unconfirmed; m_2163 reverses outright.** This is this project's fourth
retraction (after Steps 3, 5, 11), and it lands on the same root cause as
Step 11's: an apparent LSTM-specific advantage that turns out to be an
artifact of comparing the two forecasters under a decision-engine
configuration that wasn't actually tuned as hard for one of them as it was
for the other — mirroring almost exactly Step 11's finding that Step 8's
"LSTM beats Reactive" advantage wasn't LSTM-specific once ARIMA got equal
tuning rigor.

### horizon_weights front-loading is a real, useful lever — just not an LSTM-specific one

Separately from the head-to-head story, Table 1 shows front-loading the
near-term horizon step is a genuine improvement lever for *both*
forecasters: a non-uniform scheme wins the validation grid search on 7/13
machines for the LSTM and 6/13 for ARIMA, and produces the single largest
cost improvement of anything in this step (ARIMA on m_2163: 0.1611→0.1369,
a 15% relative cost reduction). But it helps ARIMA at least as often and as
much as it helps the LSTM — the SIM_STARTUP_DELAY=1 mechanism motivating
front-loading (the near-term forecast step is what actually determines
next-tick SLA risk) is forecaster-agnostic, exactly as `_build_multistep_targets`
itself was designed to be (Step 12's docstring). There is no forecaster-specific
benefit to front-loading beyond what already showed up in Table 2.

## Still open

- **m_2380** was not run through this step's confound audit (val-grid rank
  check, cluster-mate swap) — it's the sole machine whose LSTM-confirmed-cheaper
  verdict survived the horizon_weights fairness check, which makes it the
  best remaining candidate for a genuine LSTM-specific multistep advantage
  in this project, but it hasn't been checked for the classic
  val→test-generalization confound the way m_2183 was here.
- The head-to-head reclassification in Table 2 was not re-run against
  **Reactive** (the project's primary "beats Reactive" bucket, Step 11/12's
  Table 3) under the horizon_weights-optimized engine — only the
  LSTM-vs-ARIMA comparison was redone. Whether front-loading changes which
  machines beat Reactive at all is unknown.
- Only 4 horizon_weights schemes were tried, all monotonically front-loaded
  and hand-picked, not themselves grid-searched over a continuous weight
  space or checked for schemes that de-emphasize (not just re-weight) the
  far horizon step entirely (i.e., horizon=1 collapse, which would just be
  the original greedy engine under a different name — worth noting as a
  sanity boundary that wasn't explicitly tested).
- LSTM's own `under_prov_weight` grid choice was `5` (the grid's minimum)
  on effectively every machine across all of Steps 8-13 — never verified
  whether a lower value would do better; same pre-existing grid-boundary
  caveat noted for `safety_margin=0.40` (the grid's maximum) on several
  bursty machines including m_2183.
- ARIMA's order (2,0,1), unchanged since Step 6, still was not re-selected
  per machine even under this step's expanded grid — unclear whether a
  better-fit order would close more of the 6/13 machines ARIMA still wins,
  or erode more of the LSTM's remaining 2/13.
- SLA was not run through the same head-to-head reclassification this step
  applied to cost.
- Containerization (Docker) and a real Kubernetes/KEDA deployment: still
  not started.
