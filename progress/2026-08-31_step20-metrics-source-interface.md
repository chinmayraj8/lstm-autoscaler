# Step 20: swappable metrics-ingestion interface — deliberately interface-only
Date: 2026-08-31
Status: done (scoped)

Part 2 of the "make the shadow harness watch real infrastructure" project.
Before writing this step, the user was asked directly which real
monitoring stack the eventual connector should target (Prometheus, a
cloud provider's metrics API, or something custom), per the prior step's
explicit instruction not to guess. **The answer: not available yet — build
the swappable interface only, hold off on a concrete connector and on
Stage 3 (the live forecasting loop) until there's a real system to point
it at.** This step is scoped to exactly that. **No scale-up/scale-down
call to real infrastructure exists anywhere in this codebase, and this
step doesn't change that boundary.**

## What changed

**`src/autoscaler/metrics_source.py` created.** A `MetricsSource`
abstract base class with one required method:
`fetch_readings(machine_id, start, end) -> pd.DataFrame` — raw (not yet
resampled), whatever-cadence readings for one machine over a bounded time
range, as a DataFrame with a DatetimeIndex and at least a
`config.FEATURE_COL` column. Pull-based, not push: every existing consumer
in this project already works on bounded historical windows (a lookback
window, a 24-hour shadow window), never a live event stream, so a pull
interface keeps a future real connector a drop-in replacement for "read a
bounded slice of history" — nothing downstream (resampling, forecasting,
the decision engine, the shadow harness) needs to know or care where that
slice came from.

A concrete method, `fetch_lookback_window(machine_id, now, lookback_steps, resample_minutes)`,
is provided on the base class (not abstract — built from `fetch_readings` +
`resample_readings`): exactly enough history ending at `now` to cover
`config.LOOKBACK_STEPS` 5-minute readings, with a small margin so `ffill`
has real data at the window's start rather than immediately dropping
those rows. This is the call a future live forecasting loop would make
once per tracked machine per tick — not built yet (see "Still open").

**`resample_readings(raw, feature_col, resample_rule)`**: turns raw
readings into the standard cadence shape the rest of this project's
pipeline was built and tested against. Intentionally duplicates (does not
import) `data._prepare_timeseries`'s exact three-line resample convention
(`.resample("5min").mean()` → `.ffill()` → `.dropna()`) rather than
risking a signature change to that already-validated, widely-used
function just to make a piece of it reusable from a new module — the two
are checked directly against each other in the test suite instead, on
identical synthetic data, so any future drift between them would be
caught rather than assumed away.

**`StaticMetricsSource`**: a synthetic reference implementation of
`MetricsSource`, backed by an in-memory `{machine_id: pandas.Series}`
mapping. Explicitly labeled in its own docstring as NOT a real connector
and NOT production-ready — it exists only to prove the interface's
contract is implementable and to test `resample_readings`/
`fetch_lookback_window` without any real infrastructure. Paired with
`synthetic_readings_series(...)`, a small helper generating a smooth
diurnal-ish sine-plus-noise series at whatever cadence a test needs
(finer than 5 minutes, so `resample_readings`' averaging/ffill behavior is
actually exercised, not just passed through unchanged).

**Tests (`tests/test_metrics_source.py`, 9 tests, no TensorFlow, no
network)**: the ABC can't be instantiated directly; `resample_readings`
matches `data._prepare_timeseries`'s resample step exactly on the same
synthetic data (`pd.testing.assert_frame_equal`, the concrete check behind
the "byte-identical convention" claim); empty-input and gap-filling edge
cases; `StaticMetricsSource` correctly slices to a requested range,
returns empty for an unknown machine or a non-overlapping range;
`fetch_lookback_window` returns the right shape and never leaves NaNs.
**90/90 project tests pass** (81 pre-existing + 9 new).

## Before → After

| | Before Step 20 | After Step 20 |
|---|---|---|
| Where forecast input comes from | Nowhere live — every shadow window submitted by hand via `/shadow/{machine_id}/window` | Still nowhere live — but a defined, swappable contract exists for where it eventually WILL come from |
| Concrete real connector | None | Still none (deliberately — no real system named yet) |
| Resampling convention for live data | Undefined | `resample_readings`, checked to match `data._prepare_timeseries` exactly |
| Live forecasting loop | None | Still none (deliberately — Stage 3, waits on Stage 2 having something real to pull from) |
| Test count | 81 | 90 |

## Impact

### The interface exists, and it's provably compatible with the rest of the pipeline — that's the whole deliverable this step promises

This step does not make the harness observe anything real; it makes the
*shape* of a future real connector concrete and checked against the
pipeline it will eventually feed. The direct `assert_frame_equal` against
`_prepare_timeseries` is the load-bearing piece of evidence here: it's not
enough for `resample_readings` to look similar to the existing convention,
it has to produce identical output on identical input, and now that's
verified by a test rather than asserted in a docstring.

### Scope was deliberately narrowed by direct instruction, not by default caution

The task offered a fourth option — "not available yet, build the
interface only" — specifically to avoid the failure mode of guessing at a
Prometheus query shape or a cloud API's auth model that would then need
to be thrown away once the real system was named. That option was chosen.
Nothing about `MetricsSource`'s design should be read as "this is what the
real connector will look like" beyond the one method signature every
implementation must satisfy — the concrete shape of a Prometheus or
CloudWatch connector (query construction, auth, rate limiting, retry
behavior) is entirely unwritten and will be designed once that answer
exists.

## Still open

- **The concrete real connector does not exist.** Once the real monitoring
  stack is known, implementing `MetricsSource.fetch_readings` for it is
  the next piece of work — likely including things this interface
  intentionally didn't design for yet (auth, retries, rate limits, how the
  real system identifies "a machine" versus this project's `machine_id`
  convention).
- **The live forecasting loop (Stage 3) does not exist.** Nothing calls
  `fetch_lookback_window` on a schedule, runs ARIMA/the hybrid against the
  result, or feeds that into `shadow_store.run_shadow_cycle`. That's
  explicitly deferred until there's a real connector to drive it —
  building it against only `StaticMetricsSource` now would mean testing a
  loop against data it will never actually see in production.
- `main.py` was not touched this step — there is nothing live to expose
  yet, so no new endpoints were added. The existing `/shadow/*` endpoints
  from Steps 18–19 are unaffected.
- **No scale-up/scale-down call to real infrastructure exists anywhere in
  this codebase.** This step doesn't change that boundary and doesn't
  move it any closer without being asked to.
