# Step 17: why m_2104 works — hypothesis falsified, no clean predictor found, draft production recommendation
Date: 2026-08-31
Status: done

## What changed

**`experiments/check_arima_residual_autocorrelation.py` created.** Tests
the natural mechanistic hypothesis for Step 16's one robust result: that
machines where ARIMA(2,0,1)'s own residuals still show real
autocorrelation (the model is underfit for that machine's structure) are
exactly the ones where an LSTM has genuine nonlinear pattern left to
learn, and machines whose residuals are already close to white noise are
the ones where the hybrid should find nothing. For each of the 13 feasible
machines: fits ARIMA(2,0,1) on the TRAIN split only (no test leakage —
everything here is computable before val or test is ever touched), takes
the model's own one-step-ahead in-sample residuals (`fit.resid` — the
standard object for this exact diagnostic question, distinct from Step
16's walk-forward multi-step residuals, which have induced
overlap-autocorrelation from overlapping horizon windows even under a
perfectly-specified model and so would be the wrong input for a
Ljung-Box test), and runs the Ljung-Box test (H0: no autocorrelation up to
lag k) plus ACF/PACF at the first 5 lags. Cross-referenced against every
other per-machine feature already computed in this project: burstiness
score, `demand_scale`, K-means cluster membership (reproducing the exact
clustering from Step 15), and val→test demand-ceiling growth (Step 13's
`results_v9_val_test_gap.csv`). Writes
`experiments/results_v15_residual_autocorrelation.csv`.

A follow-up check (ad hoc, not committed as a separate script — the
commands are reproducible from this doc, same convention as Step 10's
investigations) tests a second, nonlinear-dependence hypothesis after the
first one failed: Ljung-Box on *squared* residuals (the standard
diagnostic for ARCH effects / volatility clustering — nonlinear dependence
a linear ACF cannot detect).

## Before → After

### Ljung-Box test on ARIMA(2,0,1)'s train-split residuals, all 13 machines

| Machine | Hybrid result (Step 16) | Cluster | Burstiness | demand_scale | Val→test growth (max) | LB stat (lag 10) | LB p-value (lag 10) | max\|ACF(1-5)\| |
|---|---|---|---:|---:|---:|---:|---:|---:|
| m_2056 | no advantage | C | 1.60 | 23.31 | −18.2% | 23.30 | **0.0097** | 0.070 |
| m_2134 | no advantage | A | 38.20 | 2.89 | +14.5% | 19.38 | **0.0356** | 0.057 |
| m_2647 | **thin** | B | 14.56 | 2.59 | +0.4% | 14.07 | 0.1700 | 0.042 |
| m_2163 | no advantage | B | 14.80 | 2.64 | +4.7% | 13.84 | 0.1806 | 0.019 |
| m_2085 | confound (excl.) | C | 1.50 | 23.13 | +26.7% | 12.12 | 0.2771 | 0.063 |
| m_2380 | **thin** | B | 14.33 | 2.64 | −6.0% | 11.48 | 0.3211 | 0.024 |
| m_2183 | no advantage | A | 16.67 | 3.13 | +13.6% | 10.09 | 0.4328 | 0.030 |
| m_2241 | no advantage | C | 1.00 | 16.76 | 0.0% | 9.88 | 0.4512 | 0.048 |
| m_2355 | no advantage | A | 16.60 | 3.26 | 0.0% | 8.53 | 0.5768 | 0.041 |
| m_2087 | no advantage | C | 1.25 | 16.11 | +21.7% | 7.24 | 0.7025 | 0.030 |
| m_2189 | no advantage | B | 14.49 | 2.64 | +0.3% | 6.18 | 0.7999 | 0.023 |
| m_2065 | no advantage | A | 16.33 | 3.10 | +13.1% | 5.59 | 0.8488 | 0.029 |
| m_2104 | **robust** | A | 17.50 | 3.12 | −2.4% | 2.40 | **0.9923** | **0.020** |

### Follow-up: Ljung-Box on squared residuals (nonlinear/ARCH-effect check)

| Machine | Hybrid result | LB-sq p-value (lag 10) |
|---|---|---:|
| m_2134 | no advantage | 3.0×10⁻²⁴³ |
| m_2189 | no advantage | 1.1×10⁻³⁵ |
| m_2647 | thin | 1.1×10⁻³² |
| m_2355 | no advantage | 2.5×10⁻²⁰ |
| m_2065 | no advantage | 4.4×10⁻¹⁴ |
| m_2087 | no advantage | 2.9×10⁻¹¹ |
| m_2163 | no advantage | 1.1×10⁻¹⁰ |
| m_2241 | no advantage | 4.3×10⁻¹⁰ |
| m_2183 | no advantage | 5.3×10⁻¹⁰ |
| m_2380 | thin | 3.2×10⁻⁶ |
| m_2104 | **robust** | 0.0164 |
| m_2056 | no advantage | 0.0572 |
| m_2085 | confound (excl.) | 0.0646 |

## Impact

### The stated hypothesis is not just unconfirmed — it's inverted

**m_2104, Step 16's one robust, fully-audited hybrid win, has the highest
Ljung-Box p-value of all 13 machines (0.9923) and the lowest max|ACF| in
the first 5 lags (0.020).** Its ARIMA(2,0,1) residuals are the closest to
textbook white noise of any machine in this dataset — the exact opposite
of "ARIMA is underfit here, leaving structure for the LSTM to find." The
two machines with the *most* significant linear residual autocorrelation
(m_2056, p=0.0097; m_2134, p=0.0356 — the only two below the conventional
0.05 threshold) both show **no hybrid advantage at all**. If the
hypothesis were even weakly directionally correct, m_2104 should have
ranked near the bottom of this table, not the top.

The follow-up nonlinear check doesn't rescue the story either, but for a
different reason: **squared-residual autocorrelation (volatility
clustering) turns out to be present, overwhelmingly, in almost every
machine's ARIMA residuals** — 9 of 13 machines show p-values below
10⁻¹⁰, including several "no advantage" machines with p-values
effectively at machine precision (m_2134: 3×10⁻²⁴³). This makes the
squared-residual test uninformative as a discriminator here: it isn't
absent in the non-winners and specifically present in the winners, it's
close to universal in this dataset regardless of outcome — an interesting
fact about CPU-utilization time series in general (heteroskedasticity is
the norm, not the exception), but not a usable predictor for this
question. And the direction is backwards here too: m_2104's own
squared-residual p-value (0.0164) is *less* extreme (closer to
non-significant) than both "thin" machines' (m_2647: 1.1×10⁻³²; m_2380:
3.2×10⁻⁶) — if ARCH-type nonlinear structure were the real mechanism,
m_2104 (the strongest hybrid win) should show the most extreme departure
from independence, not one of the least extreme among the three. Even
restricted to the ARCH-effect angle, m_2104 isn't distinguished from the
pack in a clean, consistent direction.

### No other already-computed feature separates the three winners from the other ten either

- **Burstiness**: m_2104 (17.5), m_2380 (14.3), m_2647 (14.6) sit in the
  middle of the range (1.0–38.2), and m_2189/m_2163 — burstiness 14.5/14.8,
  essentially identical to m_2380/m_2647 — show no advantage at all.
- **`demand_scale`**: all three winners sit in the same 2.6–3.1 band
  shared by 6 other non-winning machines (m_2065, m_2183, m_2134, m_2355,
  m_2189, m_2163) — not a discriminator.
- **Cluster membership**: m_2104 is in cluster A (with m_2065, m_2183,
  m_2134, m_2355 — all "no advantage"); m_2380/m_2647 are in cluster B
  (with m_2189, m_2163 — also "no advantage"). The three winners split
  across two different K-means clusters, and each of those clusters
  contains multiple non-winning machines. Zero separation power.
- **Val→test demand-ceiling growth**: the three winners do cluster near
  zero/negative growth (−2.4%, −6.0%, +0.4%), but so do m_2189 (+0.3%) and
  m_2241 (0.0%), both "no advantage," and m_2056 (−18.2%, the most
  negative growth of any machine) is also "no advantage." Suggestive at
  best, not clean.

No search for a *combination* of these features that happens to separate
the 3 positive examples from the 10 negative ones was attempted: with only
3 positive cases in a 13-machine sample, mining for a conjunction of
thresholds that happens to fit would be overfitting to noise, not a
finding — any such rule would need validation on a much larger machine
sample before it could be trusted, which this project doesn't have.

### Honest conclusion

**No clean predictor emerged.** The specific, plausible-sounding mechanism
proposed for this investigation — leftover linear autocorrelation in
ARIMA's residuals — is actively contradicted by the data: the machine
with the cleanest, whitest residuals is exactly the one where the hybrid
works best. The likely (though unverified) implication is that whatever
the LSTM is exploiting on m_2104 is *nonlinear* structure invisible to a
linear ACF/Ljung-Box diagnostic — plausible in principle (an LSTM is a
nonlinear model; a residual series can be linearly uncorrelated yet still
contain nonlinear functional dependence on its own lags), but the
follow-up ARCH-effect check that would have been the natural next
diagnostic for exactly that idea didn't discriminate either, because
volatility clustering turned out to be nearly universal across this
dataset's machines regardless of hybrid outcome. **This project does not
know what makes m_2104 special.** That's a legitimate place to land: the
result itself (Step 16's audit) stands on its own evidence — 5/5 seeds,
zero generalization gap, robust to cluster-mate parameter swaps — and
doesn't require a mechanism explanation to be trusted as real. It just
means "deploy per-machine, validated empirically" rather than "predict
eligibility from a static feature" is the only currently-supportable
policy (see recommendation below).

## Draft production recommendation (not implemented — analysis only)

`src/api/main.py` currently serves a single hardcoded LSTM model for a
single hardcoded machine (`_load_and_prepare()`'s default pick, currently
whichever machine has the most rows — historically m_1933), loaded from a
static `lstm_model.keras` file, with no ARIMA support and no per-machine
selection logic at all. Any change here is a substantial rewrite, not a
tweak — appropriately out of scope to implement in the same step as the
analysis that motivates it. This section is the case for what *should*
happen, for review before any code changes.

**Recommendation: default to plain ARIMA(2,0,1), tuned per-machine on
validation, for every machine. Do not deploy the residual hybrid broadly.
Treat it as an opt-in, per-machine, empirically-gated enhancement.**

Reasoning:
1. **ARIMA already wins or ties on the large majority of machines.** Per
   Step 11, ARIMA beats or matches the standalone LSTM's cost on 9/13
   machines under identical tuning rigor. Nothing since — the multistep
   engine (Steps 12–13), multivariate features (Step 14), or pooled
   training (Step 15) — changed that picture after confound-correction.
2. **The residual hybrid's benefit is real but narrow, and there is no way
   to predict which machine will see it *before deploying and measuring*.**
   This step's entire purpose was to find that predictor, specifically so
   deployment could be gated on it. It didn't materialize. Deploying the
   hybrid fleet-wide would mean carrying its operational cost (a second
   model type, a second training/monitoring/retraining pipeline, LSTM
   training-seed variance to manage — none of which ARIMA needs) on 13/13
   machines for a payoff that materializes on roughly 1/13, with 2 more
   uncertain/thin cases, and even where it works the improvement is modest
   (m_2104: 0.2861 vs. 0.3002, a ~4.7% relative cost reduction).
3. **ARIMA is simpler to operate.** It is deterministic (no training-seed
   variance to monitor or explain to on-call engineers — a real property
   Steps 9/13 had to repeatedly account for with the LSTM), has no
   GPU/TensorFlow dependency, and Step 6 already established it
   statistically ties the LSTM on point-forecast RMSE. A plain-ARIMA
   default is not a compromise on quality; on this evidence, it's very
   close to the best broadly-supportable choice.
4. **Where the hybrid is worth trying**, the only supportable process is
   the same validation-only, test-touched-once tuning discipline this
   project has used everywhere: for a specific target machine, fit both
   plain ARIMA and the residual hybrid, compare their tuned cost on that
   machine's own validation split, and only switch that specific machine
   to the hybrid if the margin clears the same combined-±1σ bar used
   throughout this project (re-checked periodically as more data
   accumulates, since a single validation split's verdict — this
   project's own repeated finding, Steps 10/13/14 — can itself fail to
   generalize). This is more operational overhead than a static
   feature-based gate would have been, which is exactly why finding a
   static predictor mattered — but with none available, this is the only
   defensible middle ground between "ignore the real m_2104 result
   entirely" and "deploy an unpredictable, usually-inert extra model
   fleet-wide."

**Alternative considered and not recommended**: deploying the hybrid
fleet-wide "just in case." Rejected because it adds real operational
complexity (a second model class, its own training/monitoring lifecycle)
for a benefit that Step 16's own evidence says is absent on roughly 9 of
13 machines and only clearly present on 1 — the added complexity isn't
paid for by the typical machine.

## Still open

- No code changes were made to `src/api/main.py` — this section is a
  recommendation for review, not an implementation. Rewriting the service
  to support per-machine forecaster selection (ARIMA default, opt-in
  hybrid) would be a substantial follow-up task in its own right.
- The "nonlinear structure invisible to linear ACF" explanation for
  m_2104 is speculation consistent with the evidence, not itself tested.
  A direct nonlinearity test on the raw residual series (e.g. BDS test,
  or checking whether a shallow feedforward net on lagged residuals alone
  beats a linear AR model on those same residuals) was not run and would
  be the natural next diagnostic if this line of investigation continues.
- The squared-residual (ARCH-effect) check used only lag 10; a more
  thorough treatment (multiple lag windows, or an explicit GARCH fit)
  might behave differently, though given how uniformly significant the
  effect already is across nearly all 13 machines, it seems unlikely to
  suddenly discriminate cleanly at a different lag choice.
- Whether a larger machine sample (beyond these 13) would reveal a real,
  smaller-effect-size predictor drowned out by this project's small
  sample is unknown and untestable with the current dataset scope (13/733
  qualifying machines evaluated, a limitation flagged since Step 8).
- Containerization (Docker) and a real Kubernetes/KEDA deployment: still
  not started.
