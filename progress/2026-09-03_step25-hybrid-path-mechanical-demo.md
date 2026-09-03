# Step 25: hybrid shadow-window scoring — a mechanical demo, not a real result
Date: 2026-09-03
Status: done

Traced why `shadow_windows`/`assignment_changes` stayed at 0/0 in Step 24
even after real uptime — Step 24's own doc blamed elapsed time ("needs
~30h more uptime"), but that explanation was wrong. This step finds and
documents the real, structural reason, then proves the scoring pipeline
itself is sound by mechanically forcing both blockers open by hand, using
an explicitly-flagged placeholder (untrained, random-weight) residual
model. **This is proof the wiring works, not a forecasting result** — no
real trained hybrid model exists anywhere in this codebase, before or
after this step.

## The two structural blockers (why 0/0 could never resolve on its own)

1. **A closed loop with no entry point.** `run_tick` (`live_loop.py`)
   only attempts shadow-window scoring `if state.current_forecaster ==
   "hybrid"`. The only thing that ever sets `current_forecaster` to
   `"hybrid"` is `evaluate_and_maybe_reassign` — itself only ever reached
   from INSIDE that same hybrid-gated branch (`run_tick` → `run_shadow_cycle`
   → `maybe_run_shadow_cycle` → `evaluate_and_maybe_reassign`). A machine
   starting on the database default (`"arima"`, `shadow.py`'s own
   established default since Step 17) can never organically become
   eligible for evaluation. Not a slow ramp — a closed loop.
2. **No residual-hybrid model exists.** Even a machine manually pushed to
   `"hybrid"` immediately hits `HybridModelUnavailable`
   (`live_loop._load_hybrid_residual_model`) — no `.keras` file for any
   machine exists anywhere in this repository. Only the unrelated original
   `lstm_model.keras` (early single-machine steps, a different model
   entirely) is present.

`tests/test_live_loop.py` already documents and exercises the intended
workaround for both, purely against synthetic data
(`_seed_hybrid_assignment`, and a stub `model_loader`). This step does the
real-cluster equivalent of both, against the actual deployed Step 23/24
observer service.

## What changed

**`src/api/main.py`**: `LSTM_AUTOSCALER_SHADOW_FIT_HOURS`/
`LSTM_AUTOSCALER_SHADOW_WINDOW_HOURS` env vars now override
`LiveLoopConfig.shadow_fit_hours`/`shadow_window_hours`, mirroring exactly
how `LSTM_AUTOSCALER_TICK_SECONDS` already overrode `tick_seconds` (Step
22). Defaults unchanged (6h/24h — the real 30h-of-history methodology);
these exist so a deliberately-flagged demo run can shrink them without
editing code, not so they get left small.

**`src/api/main.py`**: added `logging.basicConfig(level=logging.INFO, ...)`.
**Found while debugging this step, not before it**: without a configured
handler, every `logger.info`/`.warning`/`.exception` call in `shadow.py`
and `live_loop.py` — including the live loop's per-tick "observed" line
and any exception it catches — was silently dropped. `kubectl logs`
showed nothing but uvicorn's own HTTP access lines, no matter what the
live loop was actually doing internally. This is a real, previously
unnoticed observability gap: a project whose stated goal since Step 18 has
been "watch it running, not just trust it" had, in practice, no visibility
into its own live loop's per-tick reasoning on the one place that matters
— the real cluster. Fixed with one line, not scoped down for this demo.

**`k8s/observer.yaml`**:
- `LSTM_AUTOSCALER_HYBRID_MODEL_DIR=/data/hybrid_residual` — on the PVC,
  so the placeholder model (like `shadow_state.db`) survives a pod
  restart. (`live_loop.DEFAULT_HYBRID_MODEL_DIR` already read this env var
  — Step 22 wired the constant, this step is the first time it's actually
  set to anything.)
- **Demo-only overrides**, clearly commented as such: `LSTM_AUTOSCALER_SHADOW_FIT_HOURS`/
  `_WINDOW_HOURS`, iterated twice (`1.0`/`2.0` → `0.75`/`1.0` → `0.7`/`0.8`)
  before verification succeeded — see "Real Prometheus data has real gaps"
  below for why — then removed entirely after (see "Decision" below).
- **Memory bumped**: 256Mi/512Mi → 512Mi/1Gi (requests/limits). Directly
  observed: 512Mi OOMKilled the container while a second, separate
  TensorFlow process (spawned via `kubectl exec` alongside the
  already-running server) loaded the placeholder model. The main server's
  own in-process tick did NOT crash at 512Mi, but that headroom had never
  actually been exercised before — the observer-only ARIMA path (Steps
  21-24) never loads a Keras model at all. Left at the higher value going
  forward rather than reverted after the demo, since the hybrid path stays
  wired (if dormant) and could be exercised again.
- Image tag: `latest` → `step25`. See "The `:latest` staleness trap"
  below.

**PVC**: `172.18.0.2.keras` (a placeholder model, `_build_lstm_model`
(`forecasting.py`) with `config.LOOKBACK_STEPS=6`/`config.HORIZON_STEPS=3`,
untrained/random weights, saved via `model.save(path)` — NOT
`model.save_weights`, which requires a `.weights.h5` suffix and doesn't
match how `live_loop._load_hybrid_residual_model` calls `.load_weights()`
on a `.keras` path) placed at `/data/hybrid_residual/172.18.0.2.keras` via
`kubectl cp` (built locally in this project's own dev venv, not inside the
resource-constrained pod — see the OOM finding above for why not).

**Database**: `machines.172.18.0.2.current_forecaster` set to `"hybrid"`
via `kubectl exec` running a short script that calls
`ShadowStore._save_assignment_change(AssignmentChange(...))` directly —
the real-cluster equivalent of `tests/test_live_loop.py`'s
`_seed_hybrid_assignment` helper, evidence field explicit that this is
`"manually seeded -- no real trained model behind this, placeholder
weights only"`.

## Real Prometheus data has real gaps — a genuine, unplanned finding

The first attempt (`shadow_fit_hours=1.0`, `shadow_window_hours=2.0`, 3h
total) failed with `fit=0 pts` — not a code bug, a real data availability
problem. Investigation via `/api/v1/status/tsdb` and direct
`/api/v1/query` calls at historical timestamps found `node_cpu_seconds_total`
for our tracked instance had an actual ~15-20 minute scrape/data gap
roughly 45-90 minutes before the check (most likely caused by this same
debugging session's own memory pressure on the cluster — see the OOM
finding above, which happened on the same node right around then), even
though Prometheus itself had been running continuously for 6h48m with zero
restarts and a 10-day configured retention. The naive assumption "pod
uptime ≈ available history" was wrong; the correct check is "does data
actually exist, queried directly," which is exactly the kind of gap this
project's own connector (`PrometheusMetricsSource`, Step 21) has to
tolerate in real production use, not just in this one debugging session.

A second attempt (`0.75h` fit / `1.0h` window, 1h45m total) fixed the
`fit=0` problem but was chosen with more headroom than strictly needed,
which meant waiting for real time to pass until the gap fully cleared the
larger window — asked directly whether the ~50-minute poll timeout meant
a genuine 50-minute wait, the honest answer was no (that was a safety
margin, not the expected duration), and the values were tightened once
more to `0.7h` fit / `0.8h` window (1.5h total) — the smallest values
clearing each hard minimum (`MIN_FIT_POINTS`=8, `LOOKBACK_STEPS+HORIZON_STEPS`=9)
by exactly one point of margin — cutting the actual remaining wait from
~30 minutes to ~10.

## The `:latest` staleness trap

A rebuilt image (containing the logging fix above) pushed to
`localhost:5000/lstm-autoscaler-observer:latest` did NOT take effect after
`kubectl apply` + `kubectl rollout restart` — the pod kept running the
OLD build. `imagePullPolicy: IfNotPresent` checks whether that image
REFERENCE (repo+tag) is already present on the node, not whether the
registry's content under that tag changed since the last pull; since the
node had already pulled `:latest` once, it never re-pulled, silently
serving stale code. Fixed by tagging the actual demo build `:step25`
(unique per meaningfully-different build) and updating the manifest —
`k8s/observer.yaml`'s own comments now document this so it isn't
rediscovered the hard way again.

## What was verified (against the real cluster, not a mock)

- **Both structural blockers confirmed real** by direct code trace, not
  just theorized: `run_tick`'s hybrid gate and `HybridModelUnavailable`
  both behave exactly as read from source, on the real deployed service.
- **The placeholder model round-trips correctly**: built and re-loaded via
  `model.load_weights()` locally before ever touching the pod, confirming
  the file format matches what `_load_hybrid_residual_model` actually
  calls.
- **The seeded hybrid assignment persisted correctly** in the PVC-backed
  `shadow_state.db`, survived an (unplanned) OOM-triggered pod restart
  along with the rest of `machines`/`observed_decisions` — incidental but
  genuine evidence toward Step 24's still-open "does shadow state actually
  survive a real reschedule" item, for a crash-restart specifically (not
  yet a full pod eviction/reschedule to a different node).
- **`_build_hybrid_window` reached, attempted, and logged a specific,
  correct diagnostic** (`not enough real history ... fit=N pts, target=M
  pts, need >=9 target pts`) once real logging was wired up — this is the
  function doing exactly what its own docstring says: fail cheaply and
  legibly on insufficient real data, not crash, not fabricate a result.
  Watched `fit` grow tick over tick (1 → 6 → 7 → 9 points) as real time
  advanced past the Prometheus gap, confirming the retry-every-tick
  behavior works exactly as `is_reevaluation_due`'s design intends.
- **A shadow window banked, for real, at 2026-09-03T16:29:07.755562+00:00**:
  `window=[15:41:07, 16:29:07]` (the full 48-minute `shadow_window_hours`
  span), `arima_cost=0.0`, `hybrid_cost=0.0` (both zero — a real, if
  unremarkable, number: at this cluster's low, mostly-idle CPU% and this
  window's short length/small sequence count, neither forecaster ever hit
  an over- or under-provisioning step in the simulated fleet; not a bug,
  just an uninformative demo-scale result, exactly as flagged up front).
  Confirmed three independent ways: the SQLite `shadow_windows` table
  directly, the pod's own log line
  (`autoscaler.shadow: shadow window banked machine=172.18.0.2 ...`,
  visible for the first time thanks to this step's logging fix), and the
  real `GET /shadow/172.18.0.2/windows` HTTP endpoint.
- **The safety gate fired correctly on the very same tick — the actual
  best evidence this step produced.** `decide_assignment` requires
  `DEFAULT_MIN_WINDOWS=3` banked windows before recommending anything
  other than `"arima"`; with only 1 window banked, it correctly
  recommended `"arima"`, and since the machine's `current_forecaster` was
  still `"hybrid"` (from the manual seed), `evaluate_and_maybe_reassign`
  correctly detected the mismatch and logged a REAL, second
  `AssignmentChange`: `hybrid -> arima`, `verdict=insufficient_windows`,
  `evidence="only 1/3 shadow windows completed"`. This is a stronger
  result than "the hybrid assignment stuck" would have been — it proves
  the gating logic's own safety property (don't trust a single window,
  revert to the safe default) survives contact with the real pipeline,
  not just synthetic test data. Confirmed via `GET /shadow/172.18.0.2`'s
  `assignment_history` (two entries: the manual seed, then this real
  reversion) and the matching pod log line.
- **The whole cycle ran inside the main server process without crashing
  it** (the earlier `kubectl exec`-triggered OOMs were a SECOND,
  co-resident TF process problem — see above — not a problem with the
  in-process tick itself), at the bumped 1Gi memory limit.

## Verification result

A shadow window banked at **2026-09-03T16:29:07.755562+00:00**, roughly 30
minutes after the demo values were deployed (most of that time spent
waiting for real Prometheus data to clear the gap described above, plus
one further tightening of the window sizes partway through). Full detail
in "What was verified" above. The pipeline — pull real data, roll an ARIMA
forecast, apply the placeholder residual, score both forecasters via the
unmodified decision engine/simulator, bank the window, re-run the
assignment decision, log everything — is confirmed working end-to-end
against the real cluster. **Nothing about this demonstrates the residual
hybrid forecasts anything well** — the model's weights are random and
untrained; the cost numbers happening to be 0.0 for both forecasters this
time is a fact about this particular short, low-traffic window, not a
claim about forecast quality.

## Before → After

| | Before Step 25 | After Step 25 |
|---|---|---|
| Why 0/0 shadow_windows | Attributed (Step 24, incorrectly) to needing ~30h more uptime | Traced to two structural blockers; both mechanically worked around for this demo |
| `logger.info`/`.warning`/`.exception` visibility | Silently dropped, no configured handler | Visible in `kubectl logs` (`logging.basicConfig` added) |
| `LSTM_AUTOSCALER_SHADOW_FIT_HOURS`/`_WINDOW_HOURS` | Not overridable without editing code | Env-var overridable, same pattern as `TICK_SECONDS` |
| Placeholder residual-hybrid model | None exists anywhere | One placeholder (untrained) model, one machine, on the PVC — explicitly NOT a real result |
| Container memory limit | 512Mi (never exercised against a real TF model load) | 1Gi (directly informed by an observed OOMKilled event) |
| Image tag discipline | `:latest` (silently stale after a rebuild) | Versioned per meaningfully-different build (`:step25`) |
| shadow_windows / assignment_changes | 0 / 0 | 1 / 2 (seed, then a real automatic reversion — see below) |

## Decision: what stays seeded after this demo

**Reverted, and confirmed via a real redeploy**: `LSTM_AUTOSCALER_SHADOW_FIT_HOURS`/
`_WINDOW_HOURS` — removed entirely from `k8s/observer.yaml` (not just set
back to 6/24, so there's no lingering demo-shaped config to misread later);
`kubectl apply` + rollout confirmed the running pod now has neither env var
set, so `LiveLoopConfig`'s code defaults (6h fit / 24h window) are what's
actually running.

**Reverted**: the placeholder model file,
`/data/hybrid_residual/172.18.0.2.keras` — deleted from the PVC via
`kubectl exec` before the redeploy above. Reasoning: this file is a
disposable artifact whose only purpose was proving a correctly-shaped
`.keras` file at the right path gets loaded and used — it has no ongoing
value, and leaving it in place risked a future person re-seeding a hybrid
assignment (deliberately or by accident) and getting a "real-looking"
shadow window off random, untrained weights without the context in this
document. `LSTM_AUTOSCALER_HYBRID_MODEL_DIR` itself was left set (see
below) since the directory being configured is harmless — it's just
currently empty again.

**Kept, deliberately, NOT reverted**:
- **The `assignment_changes`/`shadow_windows` database rows** (the seed,
  the banked window, and the automatic reversion). These are a factual
  record of what actually happened, each one explicit in its own evidence
  field about being a manual/demo action (`"seeded for mechanical demo"`,
  `"manually seeded -- no real trained model behind this, placeholder
  weights only"`) or the real, automatic gating response
  (`"insufficient_windows"`). Deleting this history would erase legitimate
  provenance and contradicts this project's own progress-log convention
  (`progress/README.md`: never quietly hide what actually happened).
  `172.18.0.2`'s `current_forecaster` is already back to `"arima"` in the
  database — not because anyone reverted it by hand, but because the real
  gating logic did that itself on the same tick the window banked. There
  was nothing left to manually revert there.
- **`LSTM_AUTOSCALER_HYBRID_MODEL_DIR=/data/hybrid_residual`** and the
  logging fix (`logging.basicConfig`), the `LSTM_AUTOSCALER_SHADOW_FIT_HOURS`/
  `_WINDOW_HOURS` override-wiring code itself, the 1Gi memory limit, and
  the `:step25` image tag. None of these are demo-specific — they're real,
  permanent improvements this step happened to need in order to run the
  demo at all (visibility into what the loop is actually doing; the
  ability to override the shadow timing window without a code change, for
  whenever it's next legitimately needed; enough memory headroom to
  actually load a Keras model in this process; not silently serving a
  stale build). Reverting these would just reintroduce the same problems
  next time this path is touched.

## Still open

- **No real trained residual-hybrid model exists anywhere in this
  codebase.** Unchanged by this step, and this step's placeholder must
  never be mistaken for one — it has random, untrained weights and was
  built purely to prove a file at the right path with the right shape gets
  loaded and used correctly, nothing about forecast quality.
- **No automatic path from ARIMA to hybrid exists**, and this step doesn't
  add one — assignment still requires the manual intervention this step
  itself performed (or the existing `/shadow/{id}/window` endpoint,
  Steps 18-19). This is intentional (Step 17's caution against a
  static/automatic hybrid-assignment rule), now additionally confirmed
  structurally enforced rather than just a design choice nobody had
  reason to route around yet.
- **Memory/resource sizing for the hybrid path is still a rough estimate**,
  informed by one OOM event during a debugging session, not systematic
  profiling.
- **The Prometheus data gap's root cause wasn't conclusively identified**
  — plausibly this session's own memory-pressure debugging, not confirmed.
  Worth knowing if it recurs without an obvious cause next time.
- **This demo only ever banked 1 shadow window, by design** (a single
  short window was the goal — proving the mechanism, not running a real
  multi-window evaluation). `decide_assignment` requires 3 *consecutive*
  wins before ever recommending a switch TO hybrid — that specific path
  (an actual promotion decision, not just the reversion this demo
  produced) remains exercised only by `tests/test_shadow.py`'s synthetic
  data, never against real windows end-to-end. A true test of that would
  need 3 real windows in a row, each with the hybrid genuinely
  outperforming ARIMA — not attempted here, and not meaningful to attempt
  with an untrained placeholder model regardless.
- Every Step 21-24 "still open" item not touched by this step remains
  open: HTTP call volume, no retry/backoff on a transient Prometheus
  failure, ARIMA refit-from-scratch-every-tick, PVC `storageClassName`/
  backup policy, server-side manifest validation.
- **No scale-up/scale-down call to real infrastructure exists anywhere in
  this codebase.** This step doesn't change that boundary and doesn't move
  it any closer without being asked to.
