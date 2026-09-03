# Step 21 (Stage 3 of 3): PrometheusMetricsSource — the real connector
Date: 2026-09-03
Status: done

Part 3 of the "make the shadow harness watch real infrastructure" project,
and the first step in it that names an actual real system. Step 20
deliberately stopped at "interface only" because no real monitoring stack
had been named yet. It's now named: a Kubernetes cluster running
kube-prometheus-stack, Prometheus reachable in-cluster at
`http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090`,
scraping node-exporter. Machine unit = node (not workload/pod). The
per-node CPU% query was verified working against the real cluster:

```
100 * (1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])))
```

**No scale-up/scale-down call to real infrastructure exists anywhere in
this codebase, and this step doesn't change that boundary.** This step
only reads from Prometheus; it does not write to or call anything in the
cluster besides that one read-only query.

## What changed

**`src/autoscaler/metrics_source.py`: `PrometheusMetricsSource(MetricsSource)`
added** (the interface itself, `resample_readings`, `StaticMetricsSource`
— all Step 20 — are untouched). Three supporting pieces, each isolated so
the query string is written exactly once anywhere in this codebase:

- `build_cpu_util_query(range_vector="5m", idle_mode_label="idle")` —
  builds the verified query with both the range-vector window and the
  idle-mode label as parameters, not hardcoded magic strings. Calling it
  with the defaults reproduces the confirmed query byte-for-byte (tested).
- `instance_to_machine_id(instance_label)` — maps Prometheus's `instance`
  label (conventionally `<host>:<port>`, e.g. `"10.244.1.5:9100"` for
  node-exporter's default port) to this project's port-free `machine_id`
  convention, by stripping the port suffix if present.
- `PrometheusMetricsSourceError` — raised for a Prometheus response that
  parses as JSON but isn't the shape this connector expects (a
  non-`"success"` status, or a `resultType` other than `"vector"`).
  Distinct from `requests`' own network/HTTP exceptions, which propagate
  unchanged.

**Why `/api/v1/query` (instant queries), not `/api/v1/query_range`**: the
verified query is a PromQL instant-vector expression — it already embeds
its own trailing window via `rate(...[5m])`, so one instant query
evaluated at `time=T` already means "the rate over the 5 minutes ending at
T", not "the value at T with no history." `fetch_readings(machine_id,
start, end)` is built from repeated `/api/v1/query` calls across `[start,
end]` at `query_step_minutes` cadence (default 5, matching this project's
resample cadence), each with an explicit `time=` parameter — Prometheus
retains raw samples within its retention window, so a past `time=` still
works for recent history despite the endpoint's name. This is less
efficient than a single `query_range` call (N HTTP round-trips for an
N-point window instead of 1) but matches the exact endpoint this step was
told to use; see "Still open" for the trade-off this implies once Stage 4
starts calling it on a schedule.

`fetch_readings` filters each instant query's result vector down to the
one row whose mapped `instance_to_machine_id` matches the requested
`machine_id`; empty (not an error) if the node never appears in any
sampled response, matching the base `MetricsSource.fetch_readings`
contract's existing "empty is legitimate, not an error" rule.

`list_machine_ids(at=None)` — one instant query, every distinct mapped
node id, sorted. Not part of the `MetricsSource` ABC (a synthetic/offline
source has no notion of "what nodes exist right now") — an extension
specific to a live connector, for Stage 4's node-discovery.

**Tests (`tests/test_metrics_source.py`, +13 new, `tests/fixtures/prometheus_instant_query_response.json`
new)**: a mocked `requests.Session` (`unittest.mock.MagicMock`) returning
the fixture — no live cluster dependency, per this step's explicit
instruction. The fixture's own top-level `_fixture_note` field says
plainly what it is and isn't: authored to match Prometheus's documented
`/api/v1/query` response schema for this query shape exactly (`status` /
`data.resultType` / `data.result[].metric.instance` /
`data.result[].value: [ts, "stringified_float"]`), **not** a literal
packet capture from the user's actual cluster — this environment has no
network access to it. Covers: the query string matches the verified query
byte-for-byte; range-vector/idle-label parameterization; instance→machine_id
mapping (with and without a port suffix); `fetch_readings` issues one
`/api/v1/query` call per step across the requested range, filters to the
right instance, returns an empty-but-correctly-shaped frame for an unknown
machine; `end < start` raises; a non-`"success"` status or non-`"vector"`
resultType raises `PrometheusMetricsSourceError`; `list_machine_ids`
dedupes and sorts; **`fetch_readings`'s output feeds `resample_readings`
(Step 20, unchanged) without error** — the same "lands in the exact shape
`data._prepare_timeseries` already produces" claim Step 20 made for
`StaticMetricsSource`, now checked for the real connector's output shape
too. **103/103 project tests pass** (90 pre-existing + 13 new).

## Before → After

| | Before Step 21 | After Step 21 |
|---|---|---|
| Real monitoring system | Unnamed (Step 20's whole reason for stopping) | Named: kube-prometheus-stack Prometheus, in-cluster, node-exporter |
| Concrete `MetricsSource` connector | None | `PrometheusMetricsSource`, tested against a mocked HTTP response |
| Verified query | N/A | `100 * (1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])))`, parameterized via `build_cpu_util_query` |
| Live forecasting loop | None | Still none (Stage 4, next) |
| Test count | 90 | 103 |

## Impact

### The interface Step 20 built is now provably compatible with real (mocked) Prometheus output, not just synthetic data

Step 20's central claim was that `resample_readings`' output is
byte-identical to `data._prepare_timeseries`'s convention, checked against
`StaticMetricsSource`. This step extends that same check to
`PrometheusMetricsSource`'s output — the point of building the interface
first (Step 20) before naming a real system (this step) was exactly so
that this connection would be a drop-in, not a rewrite, and that's what
the new `test_fetch_readings_output_feeds_resample_readings_unchanged`
test confirms.

### The HTTP-call-volume trade-off is real and worth flagging now, before Stage 4 makes it matter

Because the verified query is instant-vector-shaped, every point of
history costs one HTTP round-trip. Fetching `config.LOOKBACK_STEPS` (6)
points via `fetch_lookback_window` means ~8-9 calls (the margin
`fetch_lookback_window` already adds); Stage 4's live loop will need a
larger window to fit ARIMA meaningfully, multiplying this by however many
points that window holds, times however many tracked nodes exist, every 5
minutes. This step doesn't fix that (it wasn't asked to, and doing so
would mean silently substituting `query_range` for the verified `query`
endpoint) — Stage 4's progress doc will need to state the actual call
volume once a concrete tracked-node count and fit-window size are chosen.

## Still open

- **No live forecasting loop yet.** Nothing calls
  `PrometheusMetricsSource.fetch_readings`/`fetch_lookback_window` on a
  schedule, runs ARIMA against the result, or feeds it into
  `shadow_store.run_shadow_cycle`. That's Stage 4, next.
- **The fixture is authored, not captured.** It matches Prometheus's
  documented response schema for an instant-vector query but was not
  pulled from the user's actual cluster (no network access to it from this
  environment) — worth a real end-to-end check against the live cluster
  before trusting this connector in production, even though the shape it
  assumes is Prometheus's standard, stable API contract.
- **HTTP call volume scales with window size × tracked-node count × tick
  frequency**, because the verified query is instant-vector-shaped
  (`/api/v1/query`, not `/api/v1/query_range`). Not a problem yet (nothing
  calls this on a schedule), but Stage 4 needs to pick a fit-window size
  deliberately with this in mind, not by accident.
- **No retry/backoff on transient Prometheus HTTP failures.** A
  `requests` exception (timeout, connection error, non-2xx status via
  `raise_for_status`) propagates as-is; Stage 4's scheduler will need to
  decide what "one tick's Prometheus call failed" should do (skip that
  node this tick and log it, most likely) rather than crash the whole
  loop — not decided here since this step is the connector only, not the
  loop that calls it repeatedly.
- **`list_machine_ids` reflects only nodes present in the single instant
  query it issues** — a node that was scraped a moment before or after
  won't appear. Fine for Stage 4's per-tick discovery use (it's meant to
  answer "what's live right now"), but not a complete historical roster.
- **No scale-up/scale-down call to real infrastructure exists anywhere in
  this codebase.** This step doesn't change that boundary and doesn't move
  it any closer without being asked to.
