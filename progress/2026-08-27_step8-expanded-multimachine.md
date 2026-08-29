# Step 8: Expanded multi-machine validation (15-20 machines)
Date: 2026-08-27
Status: done

## What changed

1. **`experiments/run_multimachine_v2.py` extended in place**:
   - Replaced the Step 4 hardcoded 5-machine list with a live re-run of the same
     K-means (k=4) clustering on standardized (mean, std, burstiness) over the
     733 qualifying machines in the 5 M-row sample (reuses
     `run_multimachine._compute_machine_stats`, same `random_state=42`).
   - Added `_select_representative_machines()`: instead of keeping only the
     single centroid-closest machine per cluster, it ranks each cluster's
     members by distance to the centroid and keeps the closest 4 (or fewer,
     down to all members, if a cluster has < 3 qualifying machines). The
     single highest-burstiness qualifying machine is still added separately
     as the labelled stress-test case, exactly as in Step 4.
   - `calibrate_demand_scale` and `check_feasibility` applied to every
     selected machine, unchanged from Step 5; infeasible machines are
     skipped with a printed reason.
   - Tuning (`tune_on_validation`, seed=42) and evaluation
     (`run_single_experiment`, seeds 42-46) per feasible machine, unchanged
     from Steps 4-5.
   - Added `_print_pattern_generalization()`: classifies every feasible
     machine into (A) confirmed LSTM cost advantage with SLA tied or
     LSTM-favorable, (B) confirmed LSTM cost advantage with Reactive winning
     SLA, (C) Reactive winning both metrics outright, (D) everything else.
   - Output CSV renamed to `experiments/results_v5_expanded_multimachine.csv`;
     added `cluster_rank` column (0 = centroid-closest) so per-cluster
     picks are traceable.
   - Dropped the old `_print_v3_vs_v4` comparison (not meaningful once the
     machine set itself has expanded well beyond Step 4's five).

2. **`experiments/results_v5_expanded_multimachine.csv` created**: 65 rows
   (13 feasible machines x 5 seeds).

## Why

Step 5's "confirmed LSTM cost advantage" result rested on 2 machines out of
4 feasible ones (m_2189, m_2065) — far too small a sample to tell whether
that was a real regime or two lucky draws from the K-means centroid picks.
Step 8 exists to answer that directly: keep the same selection *method*
(K-means k=4 + stress test) but sample 3-4 machines per cluster instead of
1, so each cluster's picture isn't determined by a single point.

## Before -> After

Selected machines: 4 clusters x 4 candidates (centroid-closest 4) + 1
stress-test = **17 machines** (up from Step 4/5's 5).

Feasibility (same 115%-mean-demand calibration and 800% fleet-cap rule as
Step 5):

| Cluster | Candidates | Feasible | Infeasible |
|---|---:|---:|---:|
| 0 (mid-mean, mid-burst) | 4 | 4 | 0 |
| 1 (low-mean, fat-tailed) | 4 | 0 | **4 — entire cluster** |
| 2 (calm, low-burst) | 4 | 4 | 0 |
| 3 (high-mean, mid-burst) | 4 | 4 | 0 |
| stress-test | 1 | 1 | 0 |
| **Total** | **17** | **13** | **4** |

All 4 infeasible machines are in cluster 1 (m_2101, m_2315, m_2281, m_2049)
— same p99 965%/917%/952%/842% >> 800% cap pattern Step 5 found for m_2101
alone. With 3 more samples from the same cluster now also failing, this is
no longer a one-off: **cluster 1's demand profile is structurally
infeasible at a 115% mean-demand target**, not just m_2101 specifically.

Per-machine test results (mean ± std across seeds 42-46), sorted by cluster:

| Machine | Cluster | Burstiness | LSTM SLA%±std | React SLA% | SLA verdict | LSTM cost±std | React cost | Cost verdict |
|---|---|---:|---:|---:|---|---:|---:|---|
| m_2065 | 0 (rank 0) | 16.33 | 1.3363±0.1575 | 0.6682 | Reactive wins | 0.2254±0.0111 | 0.2984 | **LSTM wins (confirmed)** |
| m_2355 | 0 (rank 1) | 16.60 | 3.2517±0.1220 | 0.4454 | Reactive wins | 0.3341±0.0031 | 0.6548 | **LSTM wins (confirmed)** |
| m_2183 | 0 (rank 2) | 16.67 | 2.7815±0.3348 | 1.1038 | Reactive wins | 0.2971±0.0192 | 0.3731 | **LSTM wins (confirmed)** |
| m_2104 | 0 (rank 3) | 17.50 | 3.7086±0.1847 | 0.2208 | Reactive wins | 0.3046±0.0083 | 0.6004 | **LSTM wins (confirmed)** |
| m_2087 | 2 (rank 0) | 1.25 | 0.4415±0.0000 | 0.0000 | Reactive wins | 0.0221±0.0000 | 0.0000 | Reactive wins |
| m_2056 | 2 (rank 1) | 1.60 | 0.8389±0.1847 | 0.2208 | Reactive wins | 0.1205±0.0065 | 0.2362 | **LSTM wins (confirmed)** |
| m_2241 | 2 (rank 2) | 1.00 | 0.2208±0.0000 | 0.0000 | Reactive wins | 0.0110±0.0000 | 0.0000 | Reactive wins |
| m_2085 | 2 (rank 3) | 1.50 | 2.8825±0.1568 | 4.8780 | **LSTM wins (confirmed)** | 0.2129±0.0059 | 0.2860 | **LSTM wins (confirmed)** |
| m_2189 | 3 (rank 0) | 14.49 | 1.6777±0.1209 | 1.1038 | Reactive wins | 0.1391±0.0044 | 0.2759 | **LSTM wins (confirmed)** |
| m_2380 | 3 (rank 1) | 14.33 | 1.4570±0.3348 | 0.2208 | Reactive wins | 0.2000±0.0064 | 0.2119 | **LSTM wins (confirmed)** |
| m_2163 | 3 (rank 2) | 14.80 | 1.8543±0.1209 | 0.4415 | Reactive wins | 0.1488±0.0051 | 0.2230 | **LSTM wins (confirmed)** |
| m_2647 | 3 (rank 3) | 14.56 | 1.2804±0.0987 | 0.0000 | Reactive wins | 0.1925±0.0036 | 0.5055 | **LSTM wins (confirmed)** |
| m_2134 | stress-test | 38.20 | 12.7594±0.8171 | 2.4283 | Reactive wins | 0.9196±0.0302 | 0.6711 | Reactive wins |

Pearson correlations (13 feasible machines, vs Step 5's 4):

| | Step 5 (n=4) | Step 8 (n=13) |
|---|---:|---:|
| r(burstiness, SLA gap) | -0.911 | **-0.876** (holds) |
| r(burstiness, cost gap) | -0.639 | **-0.218** (weakens sharply) |

Pattern-generalization classification (13 feasible machines):

| Bucket | Count | Machines |
|---|---:|---|
| (A) Confirmed LSTM cost advantage + SLA tied/LSTM-favorable | 1 | m_2085 (LSTM wins **both** metrics) |
| (B) Confirmed LSTM cost advantage + Reactive wins SLA | 9 | m_2065, m_2355, m_2183, m_2104, m_2056, m_2189, m_2380, m_2163, m_2647 |
| (C) Reactive wins both metrics outright | 3 | m_2087, m_2241, m_2134 |
| **Total confirmed LSTM cost advantage (A+B)** | **10 / 13 (77%)** | |
| **Total Reactive wins both** | **3 / 13 (23%)** | |

## Impact

**The LSTM cost advantage generalizes; the "tied SLA, free lunch" framing
does not.**

1. **Cost advantage is now the modal outcome, not a coincidence.** 10 of 13
   feasible machines (77%) show a confirmed LSTM cost advantage (gap >
   combined ±1σ), up from 2 of 4 (50%) in Step 5. It shows up in every
   feasible machine in clusters 0 and 3, plus half of cluster 2. This is not
   "two lucky machines" — it is the dominant pattern once more machines are
   sampled per cluster.

2. **But the mechanism is a trade-off, not a free lunch, in 9 of those 10
   cases.** Step 5's headline framing of m_2189 was "same SLA, lower cost."
   Re-run here with 3 more cluster-3 neighbors (m_2380, m_2163, m_2647) plus
   4 cluster-0 neighbors of m_2065 (m_2355, m_2183, m_2104), the actual
   pattern is m_2065's flavor: **the LSTM buys a lower cost score by
   accepting materially more SLA violations** (gaps from -0.57pp to -3.49pp
   in this run), not by matching Reactive's SLA for free. Zero of the 13
   machines show a tied SLA outcome in this run.

3. **One new best-case machine appeared: m_2085**, where the LSTM wins
   confirmed on *both* SLA (+2.00pp better) and cost (+0.073 better) —
   the first machine in this project where the LSTM strictly dominates
   Reactive. It's a calm machine (burstiness 1.50, same cluster as m_2087
   and m_2241, which both go the other way), so burstiness alone doesn't
   explain it; worth a follow-up look at what's different about its demand
   shape.

4. **Reactive still wins both metrics outright on 3/13 (23%)**: 2 of the 4
   calmest machines (m_2087, m_2241, cluster 2) and the stress test
   (m_2134). So "LSTM cost advantage" is common but not universal even
   within the calmest cluster — cluster 2 alone contains one Reactive-sweep
   (m_2087), one LSTM-sweep (m_2085), and one plain LSTM-cost-advantage
   (m_2056), showing real machine-to-machine heterogeneity that burstiness
   doesn't capture.

5. **Burstiness-SLA correlation holds up (r=-0.876 vs -0.911)**: higher
   burstiness still predicts a larger Reactive SLA advantage. **Burstiness-
   cost correlation collapses (r=-0.639 -> -0.218)**: Step 5's fairly strong
   cost correlation was itself largely a 4-point artifact; with 13 points
   it's closer to noise.

6. **Cluster 1 is entirely infeasible**, not just m_2101. All 4 sampled
   members (m_2101, m_2315, m_2281, m_2049) have p99 calibrated demand
   830-965% against an 800% fleet cap. This cluster's demand shape (very
   low mean CPU, very high variance) can't be served by the current
   10-server/80%-capacity fleet model at a 115% mean-demand target,
   regardless of which specific machine is picked.

### Answering the question directly

**Does the m_2189/m_2065 cost-advantage pattern generalize, or was it two
lucky machines?** The *cost advantage* itself generalizes — it's now seen
on 10/13 machines across 3 of 4 clusters. It was not two lucky machines.
But the specific claim that made it look like a "free lunch" — SLA staying
tied while cost drops — was the lucky part: m_2189 itself, re-run here,
resolves to Reactive winning SLA by -0.57pp rather than the exact tie Step
5 reported (see caveat below), and none of the other 12 machines show a
tied SLA either. The honest generalized finding is: **the LSTM usually
wins on cost by trading away SLA compliance, on this fleet-capacity
model**, with m_2085 as a single exception where it wins on both.

## Still open

- **Run-to-run noise beyond the 5-seed variance.** m_2189 was tuned to
  `sm=0.25` here vs `sm=0.30` in Step 5, and its SLA verdict flipped from
  "tied" to "Reactive wins" despite being the same machine, same demand
  scale, same seed range. TensorFlow/Keras training on the Metal GPU
  backend isn't bit-exact across process runs even with
  `np.random.seed`/`tf.random.set_seed` fixed, so individual per-machine
  verdicts near a tie can shift on re-run. The *aggregate* pattern across
  13 machines is far more robust than any single machine's verdict, so
  treat single-machine claims (like Step 5's "m_2189 tied SLA") as
  directional, not exact.
- **m_2085 (LSTM wins both) is unexplained.** Worth a follow-up: what
  about its demand shape differs from m_2087/m_2241 (same cluster,
  Reactive wins both) despite similar burstiness/mean/std?
- **Cluster 1 (infeasible) still unaddressed** — needs either a higher
  `max_servers` ceiling or a burst-absorbing capacity tier to even be
  evaluable; out of scope here as in Step 5.
- **Only 13 of 733 qualifying machines evaluated (1.8%).** The K-means
  clusters themselves are coarse (4 clusters over 3 features); a machine
  could plausibly be selected that isn't representative of its cluster's
  full internal diversity (cluster 2 alone showed 3 different outcomes
  among its 4 members).
- **ARIMA baseline (Step 6) not re-run on any of these 12 new machines** —
  the RMSE/MAE parity with LSTM was only established on m_1933.
- **Phase 2B onward (src/ extraction, API, Kubernetes, dashboard): not
  started.**
