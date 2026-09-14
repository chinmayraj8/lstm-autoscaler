"""
SQLite-backed persistence for the shadow-mode evaluation harness (Step 19).

Step 18's shadow.py is entirely pure, in-memory logic: `MachineShadowState`
is a plain dataclass, and `record_shadow_window`/`evaluate_and_maybe_reassign`/
`maybe_run_shadow_cycle` read and mutate it directly with no notion of
persistence. That module is deliberately left UNCHANGED here -- not one
line of shadow.py's decision logic is touched by this file. This module
adds a separate storage layer that:

  1. loads a `MachineShadowState` from SQLite (reconstructing the dataclass
     from rows, capped to the same `keep_last_n` window the in-memory
     version would have held),
  2. calls shadow.py's existing, unmodified functions against that state,
  3. writes back whatever changed (a new window row, an updated
     last-evaluated timestamp, a new assignment-change row) -- and nothing
     else, since those functions' return values and mutations already say
     exactly what changed.

This is why it's verifiable as a storage change and not a logic change:
`decide_assignment`, `hybrid_wins_every_window`, `aggregate_verdict`,
`is_reevaluation_due`, and `cumulative_summary` are never called through
anything but their existing, already-tested signatures, unmodified from
Step 18. The only new code here is schema + read/write plumbing.

Schema (three tables, no ORM -- Python's stdlib `sqlite3`, no new
dependency):
  - `machines`: one row per tracked machine -- current_forecaster,
    last_evaluated_at (the 30-day re-evaluation schedule state).
  - `shadow_windows`: append-only full history of every scored window ever
    banked for every machine (not capped -- the cap only applies to what
    `decide_assignment` looks at, not to the audit log).
  - `assignment_changes`: append-only full history of every assignment
    change, each with its timestamp and evidence string (Step 18's
    logging requirement, now durable across restarts).

Concurrency: a fresh `sqlite3.connect()` per operation (this harness is
explicitly observe-only, low-frequency by design -- see shadow.py's module
docstring -- so connection-pooling complexity isn't warranted at this
stage).

Step 22 (Stage 4, live forecasting loop) addition: a fourth table,
`observed_decisions`, and a `last_recommended_servers` column on
`machines`. This is a SEPARATE concern from the three tables above --
those back shadow.py's ARIMA-vs-hybrid comparison/gating logic (which
needs BOTH forecasters' numbers for a window to mean anything);
`observed_decisions` logs each individual tick's single-forecaster
decision (whichever forecaster is currently assigned) for every tracked
node, whether or not that node has a hybrid comparison running. See
`live_loop.py`'s module docstring for why the live loop treats these as
two distinct logging paths rather than forcing every tick through
shadow.py's two-forecaster data model.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Callable, List, Optional, Tuple

from . import instrumentation, shadow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS machines (
    machine_id TEXT PRIMARY KEY,
    current_forecaster TEXT NOT NULL DEFAULT 'arima',
    last_evaluated_at TEXT
);

CREATE TABLE IF NOT EXISTS shadow_windows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id TEXT NOT NULL,
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    arima_cost REAL NOT NULL,
    arima_sla_pct REAL NOT NULL,
    hybrid_cost REAL NOT NULL,
    hybrid_sla_pct REAL NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_shadow_windows_machine
    ON shadow_windows(machine_id, window_start, id);

CREATE TABLE IF NOT EXISTS assignment_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    old_forecaster TEXT NOT NULL,
    new_forecaster TEXT NOT NULL,
    verdict TEXT NOT NULL,
    n_windows INTEGER NOT NULL,
    evidence TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_assignment_changes_machine
    ON assignment_changes(machine_id, timestamp, id);

CREATE TABLE IF NOT EXISTS observed_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    forecaster TEXT NOT NULL,
    forecast_cpu_pct TEXT NOT NULL,
    planned_load_pct REAL NOT NULL,
    current_servers INTEGER NOT NULL,
    recommended_servers INTEGER NOT NULL,
    action TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_observed_decisions_machine
    ON observed_decisions(machine_id, observed_at, id);
"""

# Additive migration for pre-Step-22 database files: `machines` may already
# exist (via CREATE TABLE IF NOT EXISTS above) without this column. SQLite
# has no "ADD COLUMN IF NOT EXISTS"; the ShadowStore constructor below
# attempts the ALTER and ignores the "duplicate column" error it raises on
# a database that already has it.
_MACHINES_ADD_LAST_RECOMMENDED_SERVERS = (
    "ALTER TABLE machines ADD COLUMN last_recommended_servers INTEGER"
)


class ShadowStore:
    """SQLite-backed durable store for shadow-mode state. `db_path` may be
    a file path or ":memory:" (used by the test suite -- an in-memory
    connection is kept open for the store's lifetime so ":memory:" data
    survives between calls, unlike a fresh :memory: connection every time)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._persistent_conn = None
        self._lock = threading.Lock()
        if db_path == ":memory:":
            # A fresh :memory: connection per call would each get its own
            # empty database -- keep one open for the store's lifetime.
            # check_same_thread=False + a lock: a real ASGI server runs sync
            # `def` request handlers in a threadpool (this bit FastAPI's own
            # TestClient during testing), so a single shared connection must
            # tolerate being used from more than one thread, serialized.
            self._persistent_conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            try:
                conn.execute(_MACHINES_ADD_LAST_RECOMMENDED_SERVERS)
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

    @contextmanager
    def _connect(self):
        if self._persistent_conn is not None:
            with self._lock:
                conn = self._persistent_conn
                yield conn
                conn.commit()
        else:
            # File-backed: a fresh connection per call is thread-safe on its
            # own (each open() reopens the same file) and needs no lock.
            conn = sqlite3.connect(self.db_path)
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    # ── Reads ────────────────────────────────────────────────────────────

    def has_machine(self, machine_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM machines WHERE machine_id = ?", (machine_id,)
            ).fetchone()
        return row is not None

    def load_state(self, machine_id: str,
                   window_limit: int = shadow.DEFAULT_MIN_WINDOWS) -> shadow.MachineShadowState:
        """Reconstruct a `MachineShadowState` the way it would look had it
        been running continuously in memory: `current_forecaster` and
        `last_evaluated_at` from `machines` (defaults if the machine has
        never been seen), `window_results` capped to the most recent
        `window_limit` (matching `record_shadow_window`'s cap), and the
        FULL `assignment_history` (never capped, same as the in-memory
        version)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT current_forecaster, last_evaluated_at FROM machines WHERE machine_id = ?",
                (machine_id,),
            ).fetchone()
            current_forecaster = row[0] if row else "arima"
            last_evaluated_at = datetime.fromisoformat(row[1]) if row and row[1] else None

            window_rows = conn.execute(
                "SELECT window_start, window_end, arima_cost, arima_sla_pct, hybrid_cost, hybrid_sla_pct "
                "FROM shadow_windows WHERE machine_id = ? ORDER BY window_start DESC, id DESC LIMIT ?",
                (machine_id, window_limit),
            ).fetchall()
            # DESC-limited to get the N most recent, then re-ascend to
            # match append order (record_shadow_window appends chronologically).
            window_rows = list(reversed(window_rows))
            windows = [
                shadow.ShadowWindowResult(
                    machine_id=machine_id,
                    window_start=datetime.fromisoformat(r[0]), window_end=datetime.fromisoformat(r[1]),
                    arima_cost=r[2], arima_sla_pct=r[3], hybrid_cost=r[4], hybrid_sla_pct=r[5],
                )
                for r in window_rows
            ]

            change_rows = conn.execute(
                "SELECT timestamp, old_forecaster, new_forecaster, verdict, n_windows, evidence "
                "FROM assignment_changes WHERE machine_id = ? ORDER BY timestamp, id",
                (machine_id,),
            ).fetchall()
            history = [
                shadow.AssignmentChange(
                    timestamp=datetime.fromisoformat(r[0]), machine_id=machine_id,
                    old_forecaster=r[1], new_forecaster=r[2], verdict=r[3], n_windows=r[4], evidence=r[5],
                )
                for r in change_rows
            ]

        return shadow.MachineShadowState(
            machine_id=machine_id, current_forecaster=current_forecaster,
            last_evaluated_at=last_evaluated_at, window_results=windows, assignment_history=history,
        )

    def get_full_window_history(self, machine_id: str) -> List[shadow.ShadowWindowResult]:
        """Every window ever banked for this machine, not capped -- the
        full audit log this step's persistence requirement asks for,
        distinct from `load_state`'s decision-relevant capped subset."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT window_start, window_end, arima_cost, arima_sla_pct, hybrid_cost, hybrid_sla_pct "
                "FROM shadow_windows WHERE machine_id = ? ORDER BY window_start, id",
                (machine_id,),
            ).fetchall()
        return [
            shadow.ShadowWindowResult(
                machine_id=machine_id,
                window_start=datetime.fromisoformat(r[0]), window_end=datetime.fromisoformat(r[1]),
                arima_cost=r[2], arima_sla_pct=r[3], hybrid_cost=r[4], hybrid_sla_pct=r[5],
            )
            for r in rows
        ]

    def get_assignment_history(self, machine_id: str) -> List[shadow.AssignmentChange]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT timestamp, old_forecaster, new_forecaster, verdict, n_windows, evidence "
                "FROM assignment_changes WHERE machine_id = ? ORDER BY timestamp, id",
                (machine_id,),
            ).fetchall()
        return [
            shadow.AssignmentChange(
                timestamp=datetime.fromisoformat(r[0]), machine_id=machine_id,
                old_forecaster=r[1], new_forecaster=r[2], verdict=r[3], n_windows=r[4], evidence=r[5],
            )
            for r in rows
        ]

    def list_machine_ids(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT machine_id FROM machines ORDER BY machine_id").fetchall()
        return [r[0] for r in rows]

    # ── Observed decisions (Step 22, Stage 4 live loop) ─────────────────────
    # A separate, simpler log from the shadow_windows/assignment_changes
    # tables above: one row per tick per tracked node, whichever forecaster
    # is currently assigned -- not a two-forecaster comparison. See module
    # docstring.

    def get_last_recommended_servers(self, machine_id: str, default: int) -> int:
        """The live loop's running "shadow server count" for one node --
        purely a logging construct (nothing here ever calls a real scaling
        API), fed back as next tick's `current_servers` the same way
        `_build_lstm_targets` iterates `current_servers = new_target`
        internally. `default` is the caller's choice for a never-seen node
        (typically `config.SIM_INITIAL_SERVERS`)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_recommended_servers FROM machines WHERE machine_id = ?", (machine_id,),
            ).fetchone()
        if row is None or row[0] is None:
            return default
        return int(row[0])

    def set_last_recommended_servers(self, machine_id: str, value: int) -> None:
        """Directly corrects the running 'shadow server count' ledger for
        one node WITHOUT adding a new `observed_decisions` audit-log row --
        used only by `live_loop.run_tick`'s real-actuation read-before-write
        reconciliation (Step 26 follow-up) to keep this ledger truthful to
        the real cluster's replica count after a successful
        `actuator.set_replicas` call, even when that call's target ended up
        differing from what `observe_node_once` originally computed this
        tick (a real drift between this ledger and the live deployment --
        see live_loop.py's module docstring on why that drift can happen).
        `record_observed_decision` remains the only place that ever logs a
        new observed-decision row; this method only ever touches the
        mutable `machines.last_recommended_servers` pointer, same as that
        method's own last statement does."""
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO machines (machine_id) VALUES (?)", (machine_id,))
            conn.execute(
                "UPDATE machines SET last_recommended_servers = ? WHERE machine_id = ?",
                (value, machine_id),
            )

    def record_observed_decision(self, machine_id: str, observed_at: datetime, forecaster: str,
                                 forecast_cpu_pct: List[float], planned_load_pct: float,
                                 current_servers: int, recommended_servers: int, action: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO machines (machine_id) VALUES (?)", (machine_id,))
            conn.execute(
                "INSERT INTO observed_decisions "
                "(machine_id, observed_at, forecaster, forecast_cpu_pct, planned_load_pct, "
                " current_servers, recommended_servers, action) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (machine_id, observed_at.isoformat(), forecaster, json.dumps(forecast_cpu_pct),
                 planned_load_pct, current_servers, recommended_servers, action),
            )
            conn.execute(
                "UPDATE machines SET last_recommended_servers = ? WHERE machine_id = ?",
                (recommended_servers, machine_id),
            )

    def get_observed_decisions(self, machine_id: str, limit: int = 500) -> List[dict]:
        """Most recent `limit` observed decisions for this node, oldest
        first (log-tail semantics) -- the "watch real decisions accumulate"
        view Stage 4 was asked to expose."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT observed_at, forecaster, forecast_cpu_pct, planned_load_pct, "
                "current_servers, recommended_servers, action FROM observed_decisions "
                "WHERE machine_id = ? ORDER BY observed_at DESC, id DESC LIMIT ?",
                (machine_id, limit),
            ).fetchall()
        rows = list(reversed(rows))
        return [
            {
                "observed_at": r[0], "forecaster": r[1],
                "forecast_cpu_pct": json.loads(r[2]), "planned_load_pct": r[3],
                "current_servers": r[4], "recommended_servers": r[5], "action": r[6],
            }
            for r in rows
        ]

    # ── Writes (called only by the orchestration functions below) ──────────

    def _save_new_window(self, machine_id: str, window: shadow.ShadowWindowResult, recorded_at: datetime) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO machines (machine_id) VALUES (?)", (machine_id,)
            )
            conn.execute(
                "INSERT INTO shadow_windows "
                "(machine_id, window_start, window_end, arima_cost, arima_sla_pct, hybrid_cost, hybrid_sla_pct, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (machine_id, window.window_start.isoformat(), window.window_end.isoformat(),
                 window.arima_cost, window.arima_sla_pct, window.hybrid_cost, window.hybrid_sla_pct,
                 recorded_at.isoformat()),
            )

    def _save_last_evaluated(self, machine_id: str, last_evaluated_at: Optional[datetime]) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO machines (machine_id) VALUES (?)", (machine_id,))
            conn.execute(
                "UPDATE machines SET last_evaluated_at = ? WHERE machine_id = ?",
                (last_evaluated_at.isoformat() if last_evaluated_at else None, machine_id),
            )

    def _save_assignment_change(self, change: shadow.AssignmentChange) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO machines (machine_id) VALUES (?)", (change.machine_id,))
            conn.execute(
                "UPDATE machines SET current_forecaster = ? WHERE machine_id = ?",
                (change.new_forecaster, change.machine_id),
            )
            conn.execute(
                "INSERT INTO assignment_changes "
                "(machine_id, timestamp, old_forecaster, new_forecaster, verdict, n_windows, evidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (change.machine_id, change.timestamp.isoformat(), change.old_forecaster, change.new_forecaster,
                 change.verdict, change.n_windows, change.evidence),
            )


# ── Orchestration: load -> call shadow.py's UNCHANGED logic -> save what changed ──

def record_and_evaluate(store: ShadowStore, machine_id: str, window: shadow.ShadowWindowResult, now: datetime,
                        min_windows: int = shadow.DEFAULT_MIN_WINDOWS,
                        keep_last_n: int = shadow.DEFAULT_MIN_WINDOWS
                        ) -> Tuple[shadow.MachineShadowState, Optional[shadow.AssignmentChange]]:
    """Persistent equivalent of calling `shadow.record_shadow_window` then
    `shadow.evaluate_and_maybe_reassign` directly on an in-memory state --
    used by the `/shadow/{machine_id}/window` endpoint's "submit one
    already-scored window" path. Loads state, mutates it via shadow.py's
    own unchanged functions, persists exactly what changed."""
    state = store.load_state(machine_id, window_limit=keep_last_n)
    shadow.record_shadow_window(state, window, keep_last_n=keep_last_n)
    store._save_new_window(machine_id, window, recorded_at=now)

    change = shadow.evaluate_and_maybe_reassign(state, now, min_windows=min_windows)
    store._save_last_evaluated(machine_id, state.last_evaluated_at)
    if change is not None:
        store._save_assignment_change(change)
    return state, change


def run_shadow_cycle(store: ShadowStore, machine_id: str, now: datetime,
                     get_window_fn: Callable[[], shadow.ShadowWindowResult],
                     min_windows: int = shadow.DEFAULT_MIN_WINDOWS,
                     cadence_days: int = shadow.DEFAULT_REEVAL_CADENCE_DAYS,
                     keep_last_n: int = shadow.DEFAULT_MIN_WINDOWS) -> Optional[shadow.AssignmentChange]:
    """Persistent equivalent of `shadow.maybe_run_shadow_cycle` -- loads
    state, calls that exact unchanged function (via a tracking wrapper
    around `get_window_fn` so this layer knows whether a new window was
    actually pulled, without re-deriving that from before/after diffs of
    the mutated state), and persists whatever changed. This is the
    function a scheduled job (Step 20's live forecasting loop) calls once
    per tracked machine per tick."""
    state = store.load_state(machine_id, window_limit=keep_last_n)

    pulled: dict = {}

    def _tracking_get_window():
        window = get_window_fn()
        pulled["window"] = window
        return window

    old_last_evaluated = state.last_evaluated_at
    change = shadow.maybe_run_shadow_cycle(
        state, now, _tracking_get_window, min_windows=min_windows,
        cadence_days=cadence_days, keep_last_n=keep_last_n,
    )

    if "window" in pulled:
        store._save_new_window(machine_id, pulled["window"], recorded_at=now)
        # Cumulative-cost gauge (GET /metrics) -- the SAME totals
        # /shadow/{machine_id} already reports (shadow.cumulative_summary
        # over the FULL banked history, not just this state's capped
        # `keep_last_n` window_results), refreshed the instant a new
        # window actually banks rather than only on request.
        full_history = store.get_full_window_history(machine_id)
        cumulative = shadow.cumulative_summary(full_history)
        if cumulative["n_windows"] > 0:
            instrumentation.SHADOW_WINDOW_COST.labels(machine_id=machine_id, forecaster="arima").set(
                cumulative["arima_cost_total"]
            )
            instrumentation.SHADOW_WINDOW_COST.labels(machine_id=machine_id, forecaster="hybrid").set(
                cumulative["hybrid_cost_total"]
            )
    if state.last_evaluated_at != old_last_evaluated:
        store._save_last_evaluated(machine_id, state.last_evaluated_at)
    if change is not None:
        store._save_assignment_change(change)

    return change
