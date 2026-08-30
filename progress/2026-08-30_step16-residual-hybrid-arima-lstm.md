# Step 16: residual hybrid — the first LSTM-related win to survive full scrutiny
Date: 2026-08-30
Status: done

## What changed

Steps 8–15 treated ARIMA and the LSTM as competitors under an identical
decision engine; none of the levers tried since (multistep engine,
multivariate features, pooled training) produced a confound-free
LSTM-specific advantage. This step combines them instead.

**`arima_baseline.py`: two new functions.**
- `_arima_train_walkforward(fit_full, train_flat, lookback, horizon)` —
  generates honest walk-forward residual TRAINING labels for the LSTM.
  Reuses the deployed ARIMA(2,0,1) model's already-estimated parameters
  (via statsmodels' `.apply()`, which reuses fixed parameters without
  re-estimating) but replays the observation history from scratch via the
  same `.append(refit=False)` walk-forward loop `_arima_rolling_forecast`
  already uses for val/test, so each forecast at step `i` only reflects
  points revealed so far — never later points in train. This sidesteps the
  "the model already saw this point while being fit" leakage worry for
  residual labels specifically, while keeping parameters identical to the
  deployed model (no fit-mismatch between label-generation and inference).
  Sanity-checked before use: train's walk-forward residual std (0.1026,
  scaled space, m_2189) is comparable to val's genuinely-held-out residual
  std (0.1002) — not artificially small, i.e. not an obvious leak.
- `_inv_residual(residual_scaled, scaler)` — correct affine inverse-scaling
  for a *difference* of two scaled values (`real_a - real_b == (scaled_a -
  scaled_b) * scaler_range`, no additive offset — unlike `_inv_flat` on a
  raw value, which does include the offset). Getting this wrong would
  silently corrupt every hybrid forecast.

**`experiments/run_hybrid_arima_lstm.py` created.** Order held at (2,0,1)
— not Step 14's per-machine AIC/BIC choice, whose own audit found it
overfits train likelihood with no held-out check and one of its two
verdict flips reversed under BIC; not a reliable enough foundation to
build a new experiment on. Architecture/lookback/horizon and the decision
engine (`_build_lstm_targets`, the original greedy engine, same as Step
8/11/15) held completely unchanged — Step 11's numbers
(`results_v7_arima_full13.csv`) are the baseline, reused unchanged.

Two approaches, both evaluated per machine per seed:
1. **Residual hybrid**: ARIMA's rolling forecast + a same-architecture
   LSTM trained to predict ARIMA's *residuals* (not raw `cpu_util_percent`)
   from the identical lookback-window input. Final forecast = ARIMA +
   LSTM-residual, in real units, fed unmodified into the decision engine.
2. **Blend baseline**: `w * standalone-LSTM + (1-w) * standalone-ARIMA`,
   `w` grid-searched jointly with `under_prov_weight`/`safety_margin` on
   validation (11 candidates, 0.0–1.0 step 0.1).

Both need a **standalone** LSTM's own forecast too (for the blend), so two
distinct LSTM models are trained per (machine, seed) — one on raw values,
one on residuals. ARIMA fit once per machine, determinism asserted (re-run,
bit-identical). Run on all 13 feasible machines, 5 seeds. Writes
`experiments/results_v14_hybrid.csv` (130 rows) and
`experiments/results_v14_hybrid_summary.csv` (13 rows).

## Before → After

### Hybrid & blend vs. standalone ARIMA and standalone LSTM (Step 11 numbers, reused unchanged)

| Machine | LSTM (Step 11) | ARIMA | Hybrid | Hybrid vs ARIMA | Blend (w) | Blend vs ARIMA |
|---|---:|---:|---:|---|---:|---|
| m_2056 | 0.1205±0.0065 | 0.1038 | 0.1082±0.0056 | ARIMA directional | 0.1108 (w=0.1) | ARIMA directional |
| m_2065 | 0.2254±0.0111 | 0.2004 | 0.2125±0.0100 | ARIMA confirmed | 0.2004 (w=0.0) | ARIMA directional |
| m_2085 | 0.2129±0.0059 | 0.3636 | 0.3455±0.0108 | **confirmed cheaper** | 0.3636 (w=0.0) | tied |
| m_2087 | 0.0221±0.0000 | 0.0221 | 0.0221±0.0000 | tied | 0.0221 (w=0.0) | tied |
| m_2104 | 0.3046±0.0083 | 0.3002 | 0.2861±0.0125 | **confirmed cheaper** | 0.3002 (w=0.0) | tied |
| m_2134 | 0.9196±0.0302 | 0.9139 | 0.9042±0.0214 | directional | 0.9241 (w=0.1) | ARIMA confirmed |
| m_2163 | 0.1488±0.0051 | 0.1435 | 0.1426±0.0089 | directional | 0.1651 (w=0.0) | ARIMA directional |
| m_2183 | 0.2971±0.0192 | 0.3068 | 0.3002±0.0192 | directional | 0.3091 (w=0.1) | ARIMA directional |
| m_2189 | 0.1391±0.0044 | 0.1038 | 0.1117±0.0102 | ARIMA directional | 0.1073 (w=0.4) | ARIMA confirmed |
| m_2241 | 0.0110±0.0000 | 0.0110 | 0.0110±0.0000 | tied | 0.0110 (w=0.0) | tied |
| m_2355 | 0.3341±0.0031 | 0.3229 | 0.3225±0.0010 | directional | 0.3318 (w=0.4) | ARIMA confirmed |
| m_2380 | 0.2000±0.0064 | 0.2053 | 0.2035±0.0010 | **confirmed cheaper** (thin) | 0.2040 (w=0.0) | directional |
| m_2647 | 0.1925±0.0036 | 0.1832 | 0.1801±0.0030 | **confirmed cheaper** (thin) | 0.1815 (w=0.2) | **confirmed cheaper** |

**Hybrid vs ARIMA: 4/13 confirmed cheaper, 4/13 directional, 2/13 ARIMA
directional, 2/13 tied, 1/13 ARIMA confirmed cheaper.** **Blend vs ARIMA:
1/13 confirmed cheaper, 1/13 directional, 4/13 ARIMA directional, 4/13
tied, 3/13 ARIMA confirmed cheaper.** The blend's tuned `w` lands near 0
(favoring ARIMA almost exclusively) on 9/13 machines — the val search
mostly rediscovers "trust ARIMA," rarely finding a beneficial linear
combination.

### Hybrid vs. standalone LSTM (does residual-framing help the LSTM itself?)

**4/13 confirmed better, 4/13 directional better, 4/13 tied, 1/13
confirmed worse (m_2085 only).** The hybrid rarely hurts and often helps
the LSTM's own usefulness in this pipeline — a broader, more consistent
signal than the narrower "beats ARIMA" comparison above (e.g. m_2189:
hybrid confirmed better than standalone LSTM, 0.1117 vs 0.1391, even
though it still doesn't catch ARIMA at 0.1038).

### Confound audit on the 4 "confirmed cheaper than ARIMA" machines (Steps 10/13/14 methodology)

| Check | m_2085 | m_2104 | m_2380 | m_2647 |
|---|---|---:|---:|---:|
| Explanation needed? | **No — inherits a known confound** | Yes | Yes | Yes |
| Val→test demand growth (max/p99) | (already known: +26.6%/+14.3%, Step 10) | −2.4% / −8.7% | −6.0% / −9.7% | +0.4% / −7.5% |
| Val-grid rank of tuned params | — | 1/56, margin 0.011 | 1/56, margin 0.009 | 1/56, margin 0.009 |
| Own params vs test-optimal | — | **0 gap** (test-optimal exactly) | 0.031 left on table | 0.024 left on table |
| Individual seeds beating ARIMA | — | **5/5** | 4/5 (1 exact tie) | 3/5 (2 exact ties) |
| Cluster-mate swap (own vs. neighbors' tuned params) | — | **4/4 still beat ARIMA** | 2/3 beat, 1/3 ties | 2/3 beat, 1/3 ties-or-loses |
| **Verdict** | Not a hybrid finding — beats an already-anomalous ARIMA number, not evidence of hybrid quality | **Robust — survives every check** | Real but thin/borderline | Real but thin/borderline |

## Impact

### m_2085 doesn't count — it beats a machine ARIMA is already known to be bad on

m_2085's ARIMA cost (0.3636) is far and away the highest of any machine in
this project's ARIMA results — the same machine Step 10 already explained
as a Reactive/tuning-generalization anomaly (test demand ceiling grows
26.6% past what validation showed). The hybrid's cost here (0.3455) is
actually *worse* than the plain standalone LSTM (0.2129) — it only
"beats ARIMA" because ARIMA is unusually bad on this specific machine for
reasons already on record, unrelated to anything the hybrid does well.
Not counted as a hybrid-specific finding.

### m_2104: the strongest LSTM-related result this project has found, and it survives everything

Every check that has previously deflated a "confirmed" LSTM win comes back
clean here: val→test demand growth runs the *safe* direction (test's
ceiling shrank, not grew); the val-tuned params are exactly the test-optimal
params (zero generalization gap, vs. m_2085's 0.186 or m_2380's own 0.031
in this same table); **all 5 individual seeds beat ARIMA**, not just the
mean (0.2715–0.2980 vs. ARIMA's 0.3002, no exceptions); and the win
survives being handed all 4 cluster-mates' independently-tuned parameters
instead of its own. This is the first time in this project's 16-step
history that a claimed LSTM-vs-ARIMA advantage has passed the full
audit treatment used since Step 10 without any erosion.

### m_2380 and m_2647: real, small, but genuinely thin

Both show a consistent *direction* (most seeds beat ARIMA by a small
margin) but real cracks under closer inspection: their val-tuned params
leave real cost on the table relative to their own test-optimal (0.024–
0.031, comparable in size to m_2085's original 0.086 LSTM-side gap from
Step 10), and one of three cluster-mate substitutions ties or loses to
ARIMA rather than beating it, for each machine. Individually, not every
seed clears ARIMA (m_2380: 1/5 ties exactly; m_2647: 2/5 tie exactly).
Reported honestly as directionally-supported-but-fragile, not placed in
the same tier of confidence as m_2104.

### The blend baseline mostly just rediscovers "trust ARIMA" — a legitimate, informative negative result

The blend's validation search chose `w` near 0 (predominantly-ARIMA) on
9/13 machines, and produced only 1/13 confirmed wins over ARIMA (m_2647 —
the same machine, and the same modest margin, the residual hybrid also
found there). This says something real and useful: **simply averaging two
independently-trained forecasts' raw outputs does not expose much useful
complementary signal on this dataset** — whatever the LSTM can add on top
of ARIMA is apparently concentrated in the *residual structure itself*
(the way the hybrid frames the problem), not accessible via a linear
combination of the two models' independent point forecasts.

### Honest bottom line

Combining the two forecasters, rather than pitting them head-to-head,
finally produces a result that survives this project's own audit standard
— but the honest scope of that result is narrow: **1 of 13 machines
(m_2104) shows a robust, confound-audited residual-hybrid advantage over
standalone ARIMA; 2 more (m_2380, m_2647) show a real but thin, partially
fragile signal in the same direction; the other 9 (including the trivially
explained m_2085) show no advantage or ARIMA still wins.** This does not
overturn this project's broader conclusion from Step 15 that the LSTM
isn't the natural first tool to reach for on this task at this data
scale — but it does show that when the LSTM is given the SPECIFIC,
narrower problem of modeling what a linear ARMA model misses (rather than
the raw series, extra features, or more pooled raw data), there IS a small
amount of genuinely exploitable nonlinear residual structure on at least
one machine, and probably a couple more, on this dataset. That's a
meaningfully different, more specific answer than "the LSTM doesn't help
here" — it's "the LSTM doesn't help with the raw series, but there is a
small amount of nonlinear signal in ARIMA's residuals that a
narrowly-scoped LSTM can partially extract, on a minority of machines."

## Still open

- m_2380 and m_2647's "real but thin" verdicts were not pushed further
  (e.g., a second-seed cluster-swap spot check, or a test-optimal sweep
  averaged across all 5 seeds instead of seed 42 only) — the thin margins
  found here are suggestive, not exhaustively confirmed or ruled out.
- m_2104's robustness was checked at seed 42 only for the test-optimal
  sweep and cluster-mate swap (the 5-seed "every seed beats ARIMA" check
  used the main run's already-computed numbers, which is a genuinely
  independent piece of evidence, but the sweep/swap mechanics themselves
  weren't repeated across all 5 seeds).
- Why m_2104 specifically shows the strongest signal — some structural
  property of its demand pattern that makes ARIMA's residuals unusually
  learnable — was not investigated. Might correlate with burstiness,
  autocorrelation structure, or something else; unexplored.
- The residual hybrid used ARIMA(2,0,1) globally (deliberately, given Step
  14's per-machine order re-selection wasn't judged reliable) — whether a
  more carefully-validated per-machine order (with a held-out rolling-
  forecast check, the gap Step 14 flagged as still open) would change
  which machines show a hybrid advantage is untested.
- The blend's `w` grid was coarse (11 candidates, 0.1 steps) — a finer
  grid was not tried and is unlikely to change the qualitative "mostly
  favors pure ARIMA" finding but wasn't verified.
- SLA was recorded in `results_v14_hybrid.csv` but not run through the
  same three-way "confirmed" classification the cost analysis used here.
- Containerization (Docker) and a real Kubernetes/KEDA deployment: still
  not started.
