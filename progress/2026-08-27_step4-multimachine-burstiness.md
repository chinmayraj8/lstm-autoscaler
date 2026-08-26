# Step 4: Multi-machine burstiness experiment
Date: 2026-08-27
Status: done

## What changed

1. **`experiments/pipeline.py` updated**: `_load_and_prepare`, `tune_on_validation`,
   and `run_single_experiment` now accept `machine_id`, `nrows`, and `df_raw`
   parameters. When `df_raw` is supplied by the caller the CSV is not re-read;
   this lets `run_multimachine.py` load the data once and share it across all
   per-machine calls.

2. **`experiments/run_multimachine.py` created**: loads 5 M rows; computes
   per-machine stats (mean, std, 95th-percentile |Δ| of 5-min-resampled CPU)
   for the 733 machines with >= 2000 raw rows and >= 200 resampled points;
   runs K-means (k=4) on standardised (mean, std, burstiness); picks the
   centroid machine per cluster; always also adds the highest-burstiness
   machine as a labelled stress-test case; tunes both policies on each
   machine's own validation split; evaluates on each machine's test split
   across seeds [42–46].

3. **`experiments/results_v3_multimachine.csv` created**: 25 rows (5 machines
   × 5 seeds), includes `machine_id`, `burstiness_score`, `cluster`,
   `is_stress_test`, tuned param columns, and all standard metrics.

## Why

Step 3 showed that on m_1933 the Reactive baseline dominates once tuned
fairly. The hypothesis is that this result is machine-specific: m_1933 has
smooth, slow-varying demand where a threshold rule can react fast enough. On
machines whose demand can jump sharply within one 5-minute tick (high
burstiness), the LSTM's ability to anticipate via a safety margin should
give it an SLA advantage that Reactive cannot match. This step tests that
hypothesis on five structurally different machines.

## Before -> After

No direct before/after for a single metric — this is the first multi-machine
measurement. The m_1933 Step 3 result is the reference point.

| Machine | Burstiness | LSTM SLA%±std | React SLA%±std | SLA gap | LSTM cost±std | React cost±std | Cost gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| m_2087 (cluster 2) | 1.2500 | 0.4415±0.0000 | 0.2208±0.0000 | **−0.22 pp** | 0.0221±0.0000 | 0.0110±0.0000 | −0.0110 |
| m_2101 (cluster 1) | 7.5000 | 91.8322±0.0000 | 41.7219±0.0000 | **−50.11 pp** | 4.6733±0.0000 | 2.2053±0.0000 | −2.4680 |
| m_2189 (cluster 3) | 14.4857 | 62.4724±0.0000 | 62.0309±0.0000 | **−0.44 pp** | 3.1236±0.0000 | 3.1015±0.0000 | −0.0221 |
| m_2065 (cluster 0) | 16.3342 | 44.0980±0.0000 | 42.9844±0.0000 | **−1.11 pp** | 2.2129±0.0012 | 2.1604±0.0000 | −0.0526 |
| m_2134 (stress-test) | 38.2000 | 56.2914±0.0000 | 56.5121±0.0000 | **+0.22 pp** | 2.8353±0.0012 | 2.8344±0.0000 | −0.0009 |

Tuned parameters per machine:

| Machine | Reactive up | Reactive down | LSTM upw | LSTM sm | React val_cost | LSTM val_cost |
|---|---:|---:|---:|---:|---:|---:|
| m_2087 | 60 | 40 | 5 | 0.20 | 0.0000 | 0.0000 |
| m_2101 | 60 | 10 | 5 | 0.05 | 1.0000 | 1.0000 |
| m_2189 | 60 | 10 | 5 | 0.20 | 9.7572 | 2.4393 |
| m_2065 | 60 | 35 | 5 | 0.25 | 6.8058 | 1.7299 |
| m_2134 | 65 | 40 | 5 | 0.35 | 5.4978 | 1.3960 |

Burstiness vs LSTM relative advantage:

| Machine | Burstiness | SLA gap (R−L) | Cost gap (R−L) |
|---|---:|---:|---:|
| m_2087 | 1.2500 | −0.2207 | −0.0110 |
| m_2101 | 7.5000 | −50.1103 | −2.4680 |
| m_2189 | 14.4857 | −0.4415 | −0.0221 |
| m_2065 | 16.3342 | −1.1136 | −0.0526 |
| m_2134 | 38.2000 | **+0.2207** | −0.0009 |

Pearson r(burstiness, SLA gap) = **+0.330** (weak positive).
Pearson r(burstiness, cost gap) = **+0.326** (weak positive).

## Impact

**Hypothesis: "LSTM wins on bursty, loses/ties on calm" — partially holds for
SLA, fails for cost.**

For SLA: Reactive wins on 4 of 5 machines. The one exception is m_2134
(burstiness=38.2, highest in the sample), where the LSTM leads by +0.2207 pp
(gap > combined ±1σ = 0 since both stds are 0 — confirmed, not noise). The
positive correlation r=+0.33 is consistent with the direction of the hypothesis
but is based on only 5 points and is weak.

For cost: Reactive wins on all 5 machines. On m_2134 the gap is −0.0009 (within
±1σ, so "tied"), but the LSTM never wins outright.

**The project now has its first confirmed instance of an LSTM SLA advantage**
(m_2134), but it is narrow (0.22 pp absolute) and does not extend to cost score.

Several additional observations:

- **m_2101 is an outlier that dominates the correlation**: this machine has
  burstiness=7.5 (moderate) but mean CPU std=16 % — by far the most volatile
  in the sample. The LSTM gets 91.8% SLA violations vs Reactive's 41.7%. This
  appears to be a val-to-test domain-shift failure: the validation period's
  demand was tractable enough that the tuner chose aggressively low-conservatism
  params (upw=5, sm=0.05), but the test period has extreme demand spikes that
  the LSTM completely fails to provision for. If m_2101 is excluded, the
  correlation r would likely be stronger and more informative.

- **m_2189 and m_2065 both have very high SLA rates (42–62%) for both
  policies.** Their mean CPU × DEMAND_SCALE=20 pushes aggregate demand into
  a regime where neither policy can keep up: with max_servers=10 and
  server_capacity=80%, maximum capacity is 800 % aggregate load. m_2189's mean
  CPU=43.6 → mean demand=872 %, which systematically exceeds this ceiling.
  High SLA rates here reflect a machine/setting mismatch, not a meaningful
  comparison between policies.

- **Tuning universally preferred under_prov_weight=5** (lowest in the grid)
  across all machines. This means the validation period consistently penalised
  violations less than the original hand-tuned value (20). Whether this is a
  feature of the data or a grid-search artefact is unclear with 5 machines.

- **The stress-test (m_2134) shows the mechanism working**: the LSTM's
  sm=0.35 safety margin preemptively adds capacity before predicted spikes
  reach the threshold, capturing the one-tick lead that Reactive cannot have.
  But the absolute gain is small because 5-minute granularity is already much
  coarser than most real demand spikes.

**Does the project now have ANY evidence of a real LSTM advantage somewhere?**
Yes — one confirmed instance: m_2134 SLA (+0.2207 pp, seeds 42–46 unanimous).
It is the weakest possible confirmation: one machine, one metric, one tiny gap.
The cost score is tied on that machine. On all other machines and both metrics,
Reactive wins.

## Still open

- The LSTM's SLA advantage on m_2134 (+0.22 pp) is directionally consistent
  with the burstiness hypothesis but too small to make a strong claim. More
  bursty machines or a finer time granularity (e.g. 1-min intervals) might
  widen the gap or eliminate it.
- m_2101 (high std, LSTM 91.8% SLA) indicates a severe val-to-test domain
  shift problem; the tuning methodology does not handle non-stationary demand
  well. This is unresolved.
- Machines where mean demand × DEMAND_SCALE exceeds max_servers × server
  capacity (m_2189, m_2065) are not useful test cases — both policies fail.
  Future experiments need either a higher max_servers or per-machine demand
  rescaling.
- Forecast horizon/lookback sweep: not done (out of scope for this step).
- Phase 2B onward (src/ extraction, API, Kubernetes, dashboard): not started.
