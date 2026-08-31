# Step 19: durable shadow state (SQLite) — a storage change, not a logic change
Date: 2026-08-31
Status: done

Part 1 of the "make the shadow harness observe real infrastructure"
project. This step is persistence only. Live metrics ingestion and the
live forecasting loop are separate, later steps (20+) — see "Still open."
**No scale-up/scale-down call to real infrastructure exists anywhere in
this codebase, and none will be added without explicit confirmation first**
— this project's own evidence (m_2085, m_2380, m_2647) is exactly why that
boundary stays firm until the observe-only harness has proven itself
against real data.

## What changed

**`src/autoscaler/shadow.py`: zero changes.** Not a rhetorical claim —
verified with `git diff --stat src/autoscaler/shadow.py` showing an empty
diff, and all 60 pre-existing tests (25 of them `test_shadow.py`, exercising
that exact module) pass completely unmodified. Step 18's pure decision
logic (`decide_assignment`, `hybrid_wins_every_window`, `aggregate_verdict`,
`is_reevaluation_due`, `record_shadow_window`, `evaluate_and_maybe_reassign`,
`maybe_run_shadow_cycle`) has no notion of a database and never will —
persistence is a wrapper around it, not a rewrite of it.

**`src/autoscaler/shadow_store.py` created.** A `ShadowStore` class backed
by Python's stdlib `sqlite3` (no new dependency, as asked) with three
tables:
- `machines` — one row per tracked machine: `current_forecaster`,
  `last_evaluated_at` (the 30-day re-evaluation schedule state).
- `shadow_windows` — append-only, every window ever banked for every
  machine (NOT capped — the cap only applies to what the decision logic
  looks at, not to the audit log; `get_full_window_history` returns
  everything, `load_state` returns only the most recent `window_limit`).
- `assignment_changes` — append-only, every assignment change with its
  timestamp and evidence string (Step 18's logging requirement, now
  durable across restarts, not just in a Python log stream).

Two orchestration functions do the actual "storage change, not a logic
change" work:
- `record_and_evaluate(store, machine_id, window, now, ...)` — the
  persistent equivalent of calling `shadow.record_shadow_window` then
  `shadow.evaluate_and_maybe_reassign` directly: loads state from SQLite,
  calls those exact unmodified functions, persists whatever changed.
  Backs the `/shadow/{machine_id}/window` endpoint's "submit one
  already-scored window" flow.
- `run_shadow_cycle(store, machine_id, now, get_window_fn, ...)` — the
  persistent equivalent of `shadow.maybe_run_shadow_cycle`: loads state,
  calls that exact function through a tracking wrapper around
  `get_window_fn` (so this layer knows whether a new window was actually
  pulled, without re-deriving that from a before/after diff of the mutated
  state), persists whatever changed. This is the function Step 20's live
  forecasting loop will call once per tracked machine per tick.

Concurrency note found and fixed during testing: FastAPI's `TestClient`
(and any real ASGI server running sync `def` handlers) executes request
handlers in a worker thread, not the thread that constructed the app. A
naive single shared `:memory:` SQLite connection is thread-bound by
default and raised `sqlite3.ProgrammingError` under the test client. Fixed
with `check_same_thread=False` plus a `threading.Lock` around the shared
in-memory connection (file-backed paths don't need this — a fresh
`sqlite3.connect()` per operation reopens the same file safely from any
thread).

**`src/api/main.py`**: `_shadow_states` (the in-memory dict) replaced with
`_shadow_store = ShadowStore(_SHADOW_DB_PATH)`, `_SHADOW_DB_PATH` read from
`LSTM_AUTOSCALER_SHADOW_DB` (default: `shadow_state.db` at the project
root) so tests can point it at `:memory:` without touching the real file.
`submit_shadow_window` now calls `shadow_store.record_and_evaluate`;
`shadow_status` now calls `store.has_machine`/`load_state`/
`get_full_window_history`/`get_assignment_history`. Two behavior
refinements made while touching this code (both extending, not narrowing,
what's inspectable — see "Impact"):
- `GET /shadow/{machine_id}`'s `cumulative`/`n_banked_windows` now reflect
  ALL windows ever recorded, not just the decision-relevant capped subset
  — a more meaningful "cumulative" for a human checking in on a machine.
- **New endpoint `GET /shadow/{machine_id}/windows`** — the full per-window
  audit log, each row showing both forecasters' cost/SLA for that specific
  window. This is the "watch it working" view the task asked for:
  individual logged decisions, not just aggregates.

**Tests**: `tests/test_shadow_store.py` (13 tests) — schema init, restart
survival (a real temp SQLite file via `tmp_path`, a fresh `ShadowStore`
instance reading back what an earlier instance wrote), the assignment/
reversion flow reproduced against the store, the rolling-cap-vs-full-history
distinction, multi-machine isolation. `tests/test_api_shadow.py` (8 tests)
— the two `/shadow/*` endpoints exercised end-to-end via FastAPI's
`TestClient`, `LSTM_AUTOSCALER_SHADOW_DB=:memory:` so the real project-root
file is never touched. Neither file needs real infrastructure or network
access. **81/81 project tests pass** (60 pre-existing through Step 18 +
13 new store tests + 8 new API tests).

## Before → After

| | Before (Step 18) | After (Step 19) |
|---|---|---|
| State location | In-memory Python dict (`_shadow_states`) | SQLite file (`shadow_state.db`) via `ShadowStore` |
| Survives a restart | No | Yes (verified: a fresh `ShadowStore` instance over the same file reads back identical state) |
| Window history retained | Capped to `keep_last_n` (default 3), older windows silently dropped | Full history persisted forever (`shadow_windows` table); decision logic still only *looks at* the recent `keep_last_n` |
| Assignment-change audit trail | In-memory list + Python `logging` calls, lost on restart | Durable `assignment_changes` table, queryable independent of process lifetime |
| Inspectability | `GET /shadow/{machine_id}` (aggregate + capped count) | + `GET /shadow/{machine_id}/windows` (every individual window, not just the aggregate) |
| shadow.py logic | — | **Unchanged — 0-line diff, all pre-existing tests pass as-is** |
| Test count | 60 | 81 |

## Impact

### Verifiably a storage change, not a logic change

The empty `git diff` on `shadow.py` plus its 25 tests passing byte-for-byte
unmodified is the actual proof this step's own instructions asked for,
not just an assertion. Every new behavior lives in `shadow_store.py`
(new file) and thin call-site swaps in `main.py` (dict access →
store method calls) — the statistical rules that decide anything (three
consecutive consistent wins, the combined-±1σ significance bar, the
30-day cadence) are byte-identical to Step 18's, now just durable.

### Inspectability got strictly better, not just persistent

Two deliberate choices — cumulative stats now spanning the FULL window
history rather than the capped decision-relevant subset, and the new
`/shadow/{machine_id}/windows` endpoint — go beyond "restore what Step 18
had" to "make it something a human can actually audit." This matters
because Stage 3 (the live forecasting loop) will make this harness
produce windows on its own schedule with no human in the loop submitting
them by hand; having a real per-window log to check against is what turns
"trust it's running" into "watch it running," which is what this whole
multi-stage project was asked to deliver.

## Still open

- **Stage 2 (live metrics ingestion) and Stage 3 (live forecasting loop)
  are separate, not-yet-built steps.** This step only makes ALREADY-COMPUTED
  window data durable; nothing in this codebase yet watches real
  infrastructure or produces a forecast on its own schedule.
- **No scale-up/scale-down call to real infrastructure exists, and this
  step didn't add one.** The harness remains strictly observe-only, per
  this task's explicit instruction — that boundary needs explicit
  confirmation before it ever changes, not an assumption that persistence
  or live ingestion imply actuation is next.
- SQLite is a single-file, single-writer-friendly store appropriate for
  this step's "first pass, no new infra dependency" scope — it was not
  chosen with high-concurrency multi-process writes in mind. If this ever
  runs behind multiple API worker processes (not just multiple threads
  within one process, which the `threading.Lock` fix handles), a
  different store or `sqlite3`'s WAL mode would need revisiting.
- No schema migration tooling — `_SCHEMA` uses `CREATE TABLE IF NOT EXISTS`
  so it's safe to re-run, but there's no versioning story for a future
  schema change against an existing populated database.
- No retention/pruning policy on `shadow_windows`/`assignment_changes` —
  they grow forever. Fine at this step's scale; worth flagging before this
  runs unattended for months against many machines.
