# Step 22 (Stage 4 of 3+2): the live forecasting loop
Date: 2026-09-03
Status: done (scoped — see "Still open")

Part 4 of the "make the shadow harness watch real infrastructure" project.
Step 21 built a real, tested `PrometheusMetricsSource` but nothing called
it. This step builds the scheduled loop that does: on a 5-minute tick, for
every tracked node, pull real history via that connector, run ARIMA
through the existing, unmodified forecasting and decision-engine code, and
log the result — still logging only, no scaling call anywhere.

**No scale-up/scale-down call to real infrastructure exists anywhere in
this codebase, and this step doesn't change that boundary.** Every new
function here either returns a value or writes a row to `ShadowStore`;
none of them can reach a real fleet.

## What changed

**`src/autoscaler/live_loop.py` created.** Two deliberately separate
logging paths (see the module's own docstring for the full reasoning) —
this split was the one real design decision this step had to make, since
`shadow.py`'s data model (`ShadowWindowResult`) is fundamentally a
two-forecaster comparison and forcing every tick through it would mean
fabricating a hybrid number for nodes that have never had one computed:

1. **`observe_node_once`** — runs for EVERY tracked node, EVERY tick.
   Pulls `LiveLoopConfig.fit_lookback_hours` (default 3h) of real history
   via `source.fetch_readings` + `resample_readings` (Steps 20-21,
   unmodified), fits+forecasts ARIMA, runs the identical scaling decision
   the `/forecast` endpoint already computes (`decision._decide_scaling`,
   unmodified), and logs it via the new `ShadowStore.record_observed_decision`.
   `current_servers` is a logging-only "shadow" count
   (`ShadowStore.get_last_recommended_servers`) carried forward tick to
   tick, exactly mirroring how `_build_lstm_targets` iterates
   `current_servers = new_target` internally — never a real fleet size.
2. **The hybrid re-validation path** (`_build_hybrid_window` +
   `shadow_store.run_shadow_cycle`) — runs ONLY for nodes ALREADY assigned
   to the hybrid (`state.current_forecaster == "hybrid"`), per this
   stage's explicit instruction. It builds a live 24h ARIMA-vs-hybrid
   shadow window (walk-forward via `arima_baseline._arima_rolling_forecast`,
   unmodified, fit on `shadow_fit_hours` of real history and rolled through
   `shadow_window_hours`) and feeds it into `shadow_store.run_shadow_cycle`
   (Step 19, unmodified) — re-validating or reverting that assignment
   against real data, using shadow.py's own re-evaluation cadence
   unchanged. It does not, and cannot, promote an ARIMA node to hybrid on
   its own — see "The residual hybrid: wired, not real" below.

**`arima_baseline.py`: one new function, `_arima_forecast_once`.** Every
existing function in that file (`_arima_rolling_forecast`,
`_arima_train_walkforward`) is shaped for backtesting against
already-known future data — a live tick has no such data (that's what's
being predicted). This extracts exactly the fit-and-forecast primitive
`_arima_rolling_forecast` already uses internally (same `ARIMA` class,
same `order`, same `[0,1]` clip convention), without its walk-forward
loop, rather than reimplementing the fit call from scratch. Every other
function in the file is untouched.

**`shadow_store.py`: one new table (`observed_decisions`), one new column
(`machines.last_recommended_servers`, added via an additive
`ALTER TABLE ... ADD COLUMN`, safe against pre-Step-22 database files),
and three new methods** (`record_observed_decision`,
`get_observed_decisions`, `get_last_recommended_servers`) — all additive;
none of Step 19's existing tables, methods, or the three existing tests'
assertions changed.

**`src/api/main.py`**:
- **New `GET /shadow/{machine_id}/decisions`** — the per-tick observed-decision
  log, the primary "watch real decisions accumulate" view this stage's
  instructions asked for. Empty (not 404) for an unseen machine, unlike
  `/shadow/{machine_id}` — a log endpoint's "nothing yet" is an ordinary
  state, not the shadow-comparison gate having an opinion.
- **`LSTM_AUTOSCALER_PROMETHEUS_URL`** (new env var): if set at startup, a
  background thread runs `live_loop.run_scheduler_loop` against a real
  `PrometheusMetricsSource`, polling on `LSTM_AUTOSCALER_TICK_SECONDS`
  (default 300s). Unset (the default — every existing test, and local dev
  without a cluster), no thread starts and nothing about this service's
  prior behavior changes.
- **`LSTM_AUTOSCALER_SKIP_LSTM_MODEL`** (new env var): skips loading
  `lstm_model.keras` and fitting its scaler from the local Kaggle CSV at
  startup. Needed because Stage 5's minimal observer deployment has
  neither the CSV nor a reason to pay that cost — `/shadow/*` and the live
  loop never touch that model or scaler, only `/forecast` does (which
  returns 503 when skipped, the same response it already gave before
  startup finished). Unset by default; unchanged behavior for the existing
  `/forecast` path.
- **`/health`**: now reports `lstm_model_loaded` and `live_loop_running`
  explicitly instead of hard-requiring the LSTM model to return 200 —
  needed once the model became optional. Returns 503 only if startup
  hasn't finished at all (previously: 503 if the model specifically wasn't
  loaded, which is no longer the right liveness signal for a service that
  can legitimately run without it).

**Tests**: `tests/test_live_loop.py` (13 new) — `observe_node_once`
against `StaticMetricsSource`, insufficient-data skip, the shadow server
count carried forward tick-to-tick; `run_tick`'s two-path split (ARIMA-only
nodes never touch `shadow_windows`; a hybrid-assigned node with no real
model is skipped, not crashed; a hybrid-assigned node with a STUB residual
model — a zero-residual object with a plain `.predict` method, since no
real trained artifact exists, see below — correctly banks a shadow window
with `hybrid_cost == arima_cost` by construction); `_build_hybrid_window`
raising on a missing model or on insufficient real history;
`resolve_tracked_machine_ids`'s env-var/discovery/error precedence; the
scheduler loop stopping after `max_ticks` and surviving a failing tick.
`tests/test_api_shadow.py` (+4): the new `/decisions` endpoint (empty for
unseen, reflects a directly-recorded decision), `/health`'s new fields
(tested by calling the route function directly against a manually-set
`_state`, matching this test file's existing convention of not running a
real `lifespan` — see that file's module docstring). No live cluster
dependency anywhere. **120/120 project tests pass** (103 pre-existing + 13
`test_live_loop.py` + 4 `test_api_shadow.py`).

*(Correcting Step 21's own count while here: that step's doc said "112/112,
14 new" — the actual number, re-verified via `pytest --collect-only`, is
**103/103, 13 new**. An arithmetic slip in that entry, not a retraction of
anything it claimed about behavior; fixed in place per this project's own
"never overwrite" rule not applying to a same-day count correction with no
behavioral claim attached.)*

## The residual hybrid: wired, not real

No production-ready hybrid-inference artifact exists anywhere in this
codebase. Step 16's residual-hybrid LSTM only ever ran as an offline
experiment script (`experiments/run_hybrid_arima_lstm.py`) against the
historical CSV, trained fresh per run, and never saved a model file. On a
brand-new real cluster, every node starts on ARIMA (shadow.py's Step 17
default) and there is currently no automatic path for the live loop to
produce a hybrid forecast for a node that hasn't already been assigned
one — `_load_hybrid_residual_model` looks for
`{hybrid_model_dir}/{machine_id}.keras` and raises `HybridModelUnavailable`
when it (always, today) isn't there. This was a direct scoping question
asked before writing this step: the option taken was "wire the hook,
no-op today" — build the branch that would run a per-node pretrained
residual model for hybrid-assigned nodes, tested against a stub, fully
documented as an open gap, rather than either skip the hybrid path
entirely or build a real training/export pipeline for it (a substantially
larger scope this step wasn't asked for). Getting a node onto the hybrid
in the first place remains the existing Step 18/19 manual path
(`/shadow/{id}/window`, e.g. fed from an offline analysis of enough
accumulated real Prometheus history) — this step doesn't add a new one,
matching Step 17's caution against a static/automatic hybrid-assignment
rule.

## Before → After

| | Before Step 22 | After Step 22 |
|---|---|---|
| Live forecasting loop | None | `live_loop.run_tick`/`run_scheduler_loop`, tested against synthetic data |
| ARIMA observation of real (mocked) data | None | `observe_node_once` — real per-tick forecast + decision, logged |
| Hybrid re-validation against real data | None | Wired (`_build_hybrid_window`), but inert on every real node today — no pretrained artifact exists |
| `/shadow/{machine_id}/decisions` | Didn't exist | New — per-tick observed-decision log |
| Running the loop against a real cluster | N/A | `LSTM_AUTOSCALER_PROMETHEUS_URL` env var starts it as a background thread in `src/api/main.py` |
| Minimal deployment without the CSV/LSTM model | Not possible (`lifespan` always required both) | `LSTM_AUTOSCALER_SKIP_LSTM_MODEL` makes them optional |
| Test count | 103 | 120 |

## Impact

### The two-path split is the load-bearing design decision here, not an afterthought

`shadow.py`'s gating logic is unmodified by this step (as instructed) — it
still needs both forecasters' numbers for a window to mean anything. That
constraint plus "only run the hybrid for already-assigned nodes" together
imply that most real ticks, for the foreseeable future (until a node is
manually assigned to hybrid via real accumulated evidence), will never
produce a `ShadowWindowResult` at all. Without `observe_node_once`'s
separate log, this stage would have shipped a live loop that runs and logs
essentially nothing for a brand-new cluster — a much weaker "watch real
decisions accumulate" than what got asked for. `/shadow/{machine_id}/decisions`
is what actually accumulates from tick one.

### The hybrid path is real plumbing, not a stub that pretends to work

`_build_hybrid_window` was built and tested end-to-end against a stub
model precisely so that once a real per-machine residual artifact exists
(whatever produces it — that's unbuilt, deliberately, per the scoping
question answered before writing this step), only `_load_hybrid_residual_model`
needs to change, not the window-construction or scoring logic around it.
The test suite proves the sequence-alignment between ARIMA's rolling
forecast and the residual model's lookback windows is correct (Step 16's
`hybrid = arima + residual` mechanism, reused, not reinvented) — that was
the part actually worth getting right now, since it's easy to get subtly
wrong and hard to notice with only a stub to test against.

## Still open

- **No pretrained residual-hybrid model exists for any real node.** The
  hybrid re-validation path has never run against real inference output,
  only a zero-residual stub. A future step would need to decide how a
  per-machine residual model gets trained from real accumulated
  Prometheus data and exported to `{hybrid_model_dir}/{machine_id}.keras`
  — genuinely unbuilt, not a small gap.
- **No node can be automatically promoted to the hybrid by this loop.**
  Intentional (see "The residual hybrid" above) — assignment still
  requires the existing manual `/shadow/{id}/window` path.
- **HTTP call volume**, flagged in Step 21 as a future concern, is now a
  real one: `observe_node_once` pulls `fit_lookback_hours` worth of points
  (default 3h → ~36 `/api/v1/query` calls) per node per 5-minute tick;
  `_build_hybrid_window`, when it does run, pulls `shadow_fit_hours +
  shadow_window_hours` (default 30h → ~360 calls). Neither was optimized
  (e.g. caching previously-fetched points across overlapping ticks) —
  acceptable for this stage's scope but worth revisiting before tracking
  many nodes at high tick frequency.
- **No retry/backoff on a transient Prometheus failure within a tick** —
  `observe_node_once`'s and `_build_hybrid_window`'s exceptions are caught
  and logged one level up in `run_tick` (so one node's failure doesn't
  stop the rest of that tick, and one tick's failure doesn't stop
  `run_scheduler_loop`), but a failed node is simply retried next tick,
  with no backoff or alerting.
- **ARIMA is refit from scratch every tick**, on a short window (default
  3h). This is simple and matches "pull its rolling lookback window" from
  this stage's instructions, but is a different (cheaper, less
  historically-informed) fitting regime than the offline pipeline's
  fit-on-most-of-the-dataset convention — not validated against real data
  for forecast quality, since no real cluster was available while building
  this step.
- **Not yet actually running against the real cluster.** This step builds
  and tests the loop against synthetic/mocked data; Stage 5 (next) is
  what actually deploys it in-cluster, pointed at the real Prometheus URL.
- **No scale-up/scale-down call to real infrastructure exists anywhere in
  this codebase.** This step doesn't change that boundary and doesn't move
  it any closer without being asked to.
