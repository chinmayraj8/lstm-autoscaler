# Step 5: Per-machine demand-scale calibration
Date: 2026-08-27
Status: done

## What changed

1. **`experiments/pipeline.py` updated**:
   - Added `TARGET_MEAN_LOAD_PCT = 115.0` constant alongside the existing `DEMAND_SCALE = 20.0`
     (kept for backward compat with Steps 1–3 scripts).
   - Added `calibrate_demand_scale(machine_mean_cpu, target_mean_load_pct=115.0)` — returns
     `target / mean_cpu`; this is a normalisation constant, not a tunable hyperparameter.
   - Added `check_feasibility(ts, demand_scale)` — returns demand stats (mean, p95, p99) and
     flags infeasible when p99 calibrated demand > max fleet capacity (10 × 80% = 800%).
   - Added `demand_scale: float = DEMAND_SCALE` parameter to both `tune_on_validation` and
     `run_single_experiment`; replaced all 3 internal `DEMAND_SCALE` references with the
     parameter. Default = old constant; all Steps 1–3 scripts unaffected.

2. **`experiments/run_multimachine_v2.py` created**: loads 5 M rows; uses the same 5 machines
   from Step 4 (metadata hardcoded — no re-running K-means); computes per-machine demand_scale
   via `calibrate_demand_scale`; calls `check_feasibility` and skips infeasible machines with
   a printed explanation; tunes and evaluates feasible machines across seeds [42–46]; saves to
   `experiments/results_v4_calibrated_demand.csv`.

3. **`experiments/results_v4_calibrated_demand.csv` created**: 20 rows (4 feasible machines ×
   5 seeds). Additional columns vs v3: `demand_scale`, `demand_mean_pct`, `demand_p95_pct`,
   `demand_p99_pct`.

## Why

Step 4 applied `DEMAND_SCALE = 20.0` uniformly. That constant was calibrated for m_1933 (mean
CPU ~5.7% → mean demand ~114%), but for higher-CPU machines it pushed aggregate demand into or
past the fleet ceiling (10 servers × 80% = 800%):

| Machine | v3 mean demand | v3 p99 demand | Status |
|---|---:|---:|---|
| m_2189 (mean CPU 43.6%) | ~872% | far above cap | both policies fail at ceiling |
| m_2065 (mean CPU 37.1%) | ~742% | peaks above cap | near-ceiling all the time |
| m_2134 (mean CPU 39.8%) | ~796% | exactly at cap | both policies fail identically |

In that regime both policies' SLA rates reflected a capacity mismatch, not forecasting quality.
The "first confirmed LSTM SLA advantage" on m_2134 (+0.22 pp) in Step 4 was an artifact of both
policies being pinned at the ceiling — the tiny gap was noise, not signal.

## Before → After

Feasibility table (v4 calibrated demand):

| Machine | v3 scale | v4 scale | v3 mean demand% | v4 mean demand% | v3 p99% | v4 p99% | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| m_2087 | 20.0 | 16.11 | 141.3 | 115.0 | 147.6 | 147.6 | feasible |
| m_2101 | 20.0 | 16.18 | ~598 | 115.0 | — | 965.4 | **INFEASIBLE** |
| m_2189 | 20.0 | 2.64 | ~872 | 115.0 | — | 185.3 | feasible |
| m_2065 | 20.0 | 3.10 | ~742 | 115.0 | — | 273.0 | feasible |
| m_2134 | 20.0 | 2.89 | ~796 | 115.0 | — | 241.7 | feasible |

Per-machine v4 test results (mean ± std across seeds 42–46):

| Machine | Burstiness | LSTM SLA%±std | React SLA%±std | SLA gap | LSTM cost±std | React cost±std | Cost gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| m_2087 (cluster 2) | 1.2500 | 0.4415±0.0000 | 0.0000±0.0000 | −0.44 pp | 0.0221±0.0000 | 0.0000±0.0000 | −0.0221 |
| m_2101 (cluster 1) | 7.5000 | — | — | INFEASIBLE | — | — | — |
| m_2189 (cluster 3) | 14.4857 | 1.1038±0.1561 | 1.1038±0.0000 | **0.00 pp** | 0.1188±0.0065 | 0.2759±0.0000 | **+0.1572** |
| m_2065 (cluster 0) | 16.3342 | 2.1826±0.2904 | 0.6682±0.0000 | −1.51 pp | 0.2423±0.0132 | 0.2984±0.0000 | **+0.0561** |
| m_2134 (stress-test) | 38.2000 | 11.6115±0.4578 | 2.4283±0.0000 | −9.18 pp | 0.8848±0.0191 | 0.6711±0.0000 | −0.2137 |

Tuned parameters per machine (v4):

| Machine | Reactive up | Reactive down | LSTM upw | LSTM sm | React val_cost | LSTM val_cost |
|---|---:|---:|---:|---:|---:|---:|
| m_2087 | 60 | 40 | 5 | 0.05 | 0.0000 | 0.0000 |
| m_2189 | 75 | 30 | 5 | 0.30 | 0.2627 | 0.2163 |
| m_2065 | 65 | 30 | 5 | 0.35 | 0.5045 | 0.4219 |
| m_2134 | 75 | 20 | 5 | 0.40 | 0.5841 | 0.4226 |

Burstiness vs LSTM relative advantage (v4 feasible machines, r on 4 points):

| Machine | Burstiness | SLA gap (R−L) | Cost gap (R−L) |
|---|---:|---:|---:|
| m_2087 | 1.2500 | −0.4415 | −0.0221 |
| m_2189 | 14.4857 | 0.0000 | **+0.1572** |
| m_2065 | 16.3342 | −1.5144 | **+0.0561** |
| m_2134 | 38.2000 | −9.1832 | −0.2137 |

Pearson r(burstiness, SLA gap) = **−0.911** (strongly negative — opposite of hypothesis).
Pearson r(burstiness, cost gap) = **−0.639** (moderately negative).

## Impact

**Calibration invalidated Step 4's key result and produced new, cleaner findings.**

### Step 4 result retracted

The "first confirmed LSTM SLA advantage" on m_2134 (+0.22 pp, all seeds) does not survive
calibration. In v3, ×20 scale pushed m_2134's mean demand to ~796% — essentially at the 800%
fleet ceiling. Both policies failed almost identically for structural reasons. The 0.22 pp gap
was noise on top of mutual failure, not evidence of LSTM predictive advantage.

In v4 with ×2.89 scale (mean demand = 115%), m_2134 has real headroom for both policies to
operate. Reactive now dominates: SLA gap = −9.18 pp (Reactive wins). This is the true
comparison on this machine.

### New confirmed findings

1. **First confirmed LSTM COST advantage: m_2189** (burstiness=14.49). Cost gap = +0.1572
   (0.2759 − 0.1188), gap >> combined ±1σ (0.0065). SLA is exactly tied (0.0 pp gap, both
   1.1038%). The LSTM achieves the same SLA at less than half the Reactive cost score by
   avoiding the Reactive's aggressive over-provisioning (22.1% vs 6.6% over-prov steps).

2. **Second confirmed LSTM COST advantage: m_2065** (burstiness=16.33). Cost gap = +0.0561
   (0.2984 − 0.2423), gap > combined ±1σ (0.0132). Reactive wins SLA (0.67% vs 2.18%), but
   the LSTM's cost score is lower because it wastes far less capacity. The LSTM accepts more
   SLA violations in exchange for dramatically less over-provisioning waste.

### Mechanism on m_2189 and m_2065

Both machines have moderate-to-high burstiness but with calibrated demand their p99 is well
within fleet capacity (185% and 273% vs 800% max). The Reactive over-provisions heavily to
avoid SLA violations (over_prov_waste=22% and 27%), because its threshold logic scales up
aggressively and scales down slowly. The LSTM, with sm=0.30–0.35, tracks the forecast more
tightly and wastes less capacity — at the cost of slightly more SLA violations on m_2065.

### Burstiness hypothesis now fails (opposite direction)

With 4 comparable data points (m_2101 correctly excluded as infeasible at any reasonable
scale):
- Pearson r(burstiness, SLA gap) = −0.911. Higher burstiness correlates with the LSTM losing
  MORE on SLA, not winning.
- Pearson r(burstiness, cost gap) = −0.639. Direction is also opposite to the hypothesis.
- The only cost advantage is on the two mid-burstiness machines (m_2189, m_2065), not the
  highest-burstiness one (m_2134).

### m_2101 status

Even after calibration to 115% mean demand, m_2101 has p99 = 965.4% — far above the 800%
ceiling. This machine has extremely fat-tailed demand (mean 115% but p99 is 8.4× the mean);
no reasonable fleet size can absorb its spikes without either an unbounded max_servers or a
per-spike elastic capacity model. Correctly excluded as infeasible.

## Still open

- **No confirmed LSTM SLA advantage anywhere** (the Step 4 instance was retracted). Cost
  advantage confirmed on 2 machines (m_2189, m_2065), but Reactive wins SLA on 3/4 machines.
- **Cost advantage mechanism on mid-burstiness machines**: the LSTM's lower cost score comes
  from less over-provisioning waste, but this trade-off (more SLA violations for less waste) is
  configurable by tuning `under_prov_weight`. A deeper sweep of this parameter might close the
  SLA gap further.
- **m_2101 (extreme tail demand)**: not addressable without a higher max_servers ceiling or a
  burst-absorbing capacity tier. Out of scope for this step.
- **Burstiness hypothesis fully disconfirmed** for SLA; needs reformulation for cost.
- **Forecast horizon/lookback sweep**: not done.
- **Phase 2B onward (src/ extraction, API, Kubernetes, dashboard)**: not started.
