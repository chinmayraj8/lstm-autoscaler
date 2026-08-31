# Step 18: shadow-mode evaluation harness — empirical per-machine gating since no static predictor exists
Date: 2026-08-31
Status: done

## What changed

Step 17 found no static machine feature predicts which machines benefit
from the residual hybrid (Step 16) — the natural mechanistic hypothesis
(ARIMA residual autocorrelation) was actively inverted. This step builds
the empirical alternative that follows directly from that finding: measure
each machine's own behavior, in shadow, before ever trusting it with the
real fleet.

**`src/autoscaler/simulation.py`: `verdict()` promoted from
`experiments/run_arima_full13.py`'s private `_verdict`.** Byte-identical
combined-±1σ formula (confirmed cheaper / directional / tied / confirmed
more expensive), now a tested package function so shadow.py — and any
future caller — reuses the exact statistical bar this project has used
since Step 11 instead of a fresh reimplementation.
`run_arima_full13.py`'s own `_verdict` is left untouched (no reason to
risk an already-validated experiment script).

**`src/autoscaler/shadow.py` created** — the harness itself. Its module
docstring states plainly what this is and is not: *a shadow-mode
measurement-and-gating harness with repeat-window statistical gating, not
a continuously-exploring bandit, an online learner, or anything that
adapts within a window.* It makes one discrete decision (which forecaster
controls a machine) at discrete re-evaluation points, using a fixed
statistical rule against a fixed number of recently-banked windows. It
does not forecast anything itself — callers (an offline job or a future
online serving path) supply each window's `(y_pred_real, y_actual_real)`
arrays from whatever ARIMA/hybrid pipeline is in use; this module only
scores, banks, and gates on them, reusing `decision._build_lstm_targets`,
`simulation._run_simulation`, and `simulation._compute_cost_score`
unmodified — no new scoring logic anywhere in this step.

Core pieces:
- `ShadowWindowResult` / `AssignmentChange` / `MachineShadowState` —
  the data model. A machine with no prior assignment defaults to
  `"arima"` (Step 17's own recommendation) until the hybrid earns its way
  in.
- `score_forecaster_window` / `run_shadow_window` — score one forecaster's
  (or both forecasters', for one window) forecast against real demand
  through the existing simulator, unmodified.
- `hybrid_wins_every_window` — the *consistency* check: the hybrid's cost
  must be lower than ARIMA's in **every** banked window, not just on
  average. A machine that wins big in 2 windows and loses badly in a 3rd
  does not pass, even if the mean still favors the hybrid — this is
  specifically the single-bad-window fragility Steps 10 (m_2085) and 13/14
  (m_2380) already found in this project, now guarded against by
  construction rather than left to a mean-only comparison.
- `aggregate_verdict` — the *significance* check: `simulation.verdict`
  applied across shadow windows the same way this project applies it
  across LSTM training seeds everywhere else.
- `decide_assignment` — requires **both**: consistency alone or
  significance alone is not enough to switch a machine to the hybrid;
  below `min_windows` (default 3) it defaults to `"arima"` regardless.
- `record_shadow_window` — banks a window, keeping only the most recent
  `keep_last_n` (default 3, matching "start with 3") so a machine's
  assignment is judged on recent behavior, not an ever-growing history
  that would dilute drift.
- `is_reevaluation_due` / `maybe_run_shadow_cycle` — periodic
  re-evaluation (default 30-day cadence): once a machine has a stable
  assignment, no further shadow windows are pulled until the cadence
  elapses, avoiding wasted shadow compute; `maybe_run_shadow_cycle` has no
  opinion on wall-clock scheduling itself, only on whether *this* call
  should do anything.
- `evaluate_and_maybe_reassign` — runs the decision, updates
  `current_forecaster` if it changed, and appends a logged
  `AssignmentChange` (timestamp + old/new forecaster + verdict + evidence)
  — including the *reversion* case: a previously-hybrid-assigned machine
  whose recent windows now favor ARIMA switches back, logged the same way.
- Python `logging` calls (`logging.getLogger("autoscaler.shadow")`) at
  both window-banked and assignment-changed points, in addition to the
  structured `AssignmentChange`/`cumulative_summary` records — satisfies
  "log cumulative cost/SLA" and "log every assignment change" as actual
  operational log lines, not just in-memory state.

**`src/api/main.py`: two new endpoints, thin HTTP wiring only.**
`POST /shadow/{machine_id}/window` accepts one already-observed window's
real demand plus both forecasters' already-computed forecasts, calls
`shadow.run_shadow_window` + `record_shadow_window` +
`evaluate_and_maybe_reassign`, and returns the window's result and current
assignment. `GET /shadow/{machine_id}` returns current state (assignment,
banked-window count, cumulative cost/SLA, full assignment history).
State lives in a new module-level `_shadow_states: Dict[str,
MachineShadowState]` dict (same in-memory pattern as the existing
`_state` dict — not persisted across restarts). **Scope, stated plainly**:
these endpoints do not run ARIMA or the hybrid LSTM themselves — that
would require pulling statsmodels fitting and the hybrid LSTM inference
pipeline into the live API service, a materially larger rewrite of a
service that currently hardcodes one LSTM model for one machine with no
ARIMA support at all. The caller (an offline/scheduled job with access to
real fleet telemetry and both forecasting pipelines) is responsible for
producing each window's forecasts and submitting them already computed.
Verified end-to-end with FastAPI's `TestClient`: 3 submitted windows where
the hybrid wins every window and clears the statistical bar produce a
logged assignment change to `"hybrid"` with the correct verdict and
evidence string; the status endpoint reflects the full history; an
unknown machine_id correctly 404s.

**`tests/test_shadow.py` added — 25 tests**, synthetic data only, no
TensorFlow, no ARIMA fitting, no real dataset (same convention as
`test_arima_baseline.py`/`test_decision.py`). Covers: the sim-machinery
wiring (a perfect forecast costs near-zero, a wildly-wrong forecast costs
more, both forecasters score against the same actual demand); the
consistency check (all-windows-win vs. one-window-loses-even-with-a-better-mean,
constructed so the naive mean-only comparison would have wrongly passed
it); the significance check across all four verdict buckets; the combined
gate (each of the four combinations of consistent/inconsistent ×
confirmed/not); the rolling window cap; the re-evaluation cadence (never
evaluated / within cadence / cadence elapsed); the full reassignment
audit trail including reversion from hybrid back to arima; and the
scheduling no-op (confirms `get_window_fn` is not even called when a
stable machine isn't due for re-evaluation, so no shadow compute is
wasted). **60/60 tests pass project-wide** (35 pre-existing + 25 new).

## Before → After

| | Before Step 18 | After Step 18 |
|---|---|---|
| Machine → forecaster assignment | No mechanism at all — Step 17's recommendation was a manual default (plain ARIMA everywhere) | Empirically gated per machine: default ARIMA, switches to hybrid only after ≥3 consecutive shadow windows all favor it AND the aggregate margin clears the project's own combined-±1σ bar |
| Statistical bar for a "win" | Ad hoc per experiment script (`_verdict`, duplicated across `run_multimachine_v2.py`/`run_arima_full13.py`) | Promoted to `simulation.verdict`, one canonical, tested implementation |
| Protection against a fluky single window | None (the exact fragility that produced the m_2085 and m_2380 false positives) | `hybrid_wins_every_window` requires consistency across every banked window, not just the mean |
| Re-evaluation after assignment | N/A | Cadence-gated (default 30 days), machine's assignment can revert if its own recent shadow windows stop favoring the hybrid, every change logged with timestamp + evidence |
| API surface | `/health`, `/forecast` (single hardcoded model/machine) | + `/shadow/{machine_id}/window`, `/shadow/{machine_id}` |
| Test count | 35 | 60 |

## Impact

### This closes the loop Step 17 opened, without overclaiming what it is

Step 17's honest conclusion was "no predictor found, deploy ARIMA by
default, gate the hybrid empirically per machine." This step is that
gate, built to the same standard of rigor the rest of this project has
insisted on: reusing the validated simulation/scoring machinery rather
than writing new scoring logic, reusing the project's own statistical bar
rather than inventing a new threshold, and — the specific instruction this
step centered on — refusing to let a single 24-hour window make the
decision, because this project has already been burned by exactly that
fragility twice (m_2085's val-optimal Reactive threshold generalizing
catastrophically to test, Step 10; m_2380's apparent multistep win failing
2 of 3 cluster-mate counterfactual swaps, Step 14). The consistency
requirement (`hybrid_wins_every_window`) is a direct, mechanical answer to
that specific risk, not just a restatement of the significance test —
Step 16's own m_2104 evidence (5/5 individual seeds beating ARIMA, not
just the mean) is exactly the empirical pattern this harness now requires
before trusting any machine going forward.

### Validated against Step 16's own real evidence, not just synthetic data

Replaying Step 16's actual per-seed cost numbers through `decide_assignment`
directly (treating each of the 5 LSTM training seeds' results as one
shadow window — not literal 24h calendar windows, but the same underlying
statistical shape this harness is built to gate on) reproduces this
project's own manual audit judgment exactly, with no hand-tuning of the
harness to make it come out that way:

| Machine | Harness recommendation | Verdict | Matches Step 16's manual audit? |
|---|---|---|---|
| m_2104 | **hybrid** | confirmed cheaper, all 5 windows | ✓ ("robust — survives every check") |
| m_2380 | arima | inconsistent (1 of 5 seeds ties ARIMA exactly) | ✓ ("real but thin, partially fragile") |
| m_2647 | arima | inconsistent (2 of 5 seeds tie ARIMA exactly) | ✓ ("real but thin, partially fragile") |
| m_2189 | arima | tied, inconsistent | ✓ (no advantage) |
| m_2163 | arima | directional, inconsistent | ✓ (no advantage) |
| m_2056 | arima | tied, inconsistent | ✓ (no advantage) |

The strict `hybrid_wins_every_window` consistency check is what does the
real work here: m_2380 and m_2647's aggregate verdict alone would read
"confirmed cheaper" (their 5-window mean does clear the combined-σ bar),
which is exactly why Step 16's manual audit had to go further and find
the individual seeds where they tied — the harness now catches that
distinction automatically, without needing a human to notice it, because
consistency is checked before significance is even relevant to the
decision.

### What this is not, restated for the record

No exploration/exploitation tradeoff, no continuous online updating, no
mid-window adaptation, no probabilistic policy — one discrete,
auditable, evidence-logged decision per machine, made at discrete
re-evaluation points, using a fixed rule. A machine assigned to ARIMA
today because it hasn't cleared 3 consistent shadow wins yet is not "being
explored" in any bandit sense; it is simply unassigned-to-hybrid until
proven otherwise, by the same standard every other result in this project
has had to meet.

## Still open

- The harness has never been run against real *forecast arrays* end-to-end
  through `run_shadow_window` itself — only against synthetic test arrays
  (unit-level) and a synthetic TestClient smoke test (integration-level).
  `decide_assignment`'s decision LOGIC was validated against Step 16's
  real per-machine, per-seed cost numbers (see above) and reproduced this
  project's own manual audit judgment on all 6 machines checked — but that
  validation used already-computed cost scalars, not the full path from
  raw forecast arrays through `run_shadow_window`'s own call into
  `_build_lstm_targets`/`_run_simulation`. A true end-to-end replay (real
  forecast arrays for m_2104's 5 seeds, run through `run_shadow_window`
  itself) would close that last gap but wasn't done here.
- State is in-memory only (`_shadow_states`, module-level dict) — a
  service restart loses all banked windows and assignment history. A real
  deployment needs persistence (a database or equivalent); explicitly out
  of scope here.
- Nothing in `src/api/main.py` actually produces the ARIMA/hybrid
  forecasts fed to `/shadow/{machine_id}/window` — that pipeline (fitting
  ARIMA, running the hybrid's residual LSTM, producing per-window
  `(y_pred_real, y_actual_real)` arrays from real fleet telemetry) doesn't
  exist as a callable service component yet; today it only exists as the
  offline `experiments/run_hybrid_arima_lstm.py` script. Wiring that up
  end-to-end (a real scheduled job producing and submitting windows) is
  future work.
- The re-evaluation cadence (30 days) and window count (3) are the values
  named in this step's instructions ("start with 3", "e.g. monthly") —
  neither was tuned or validated against how quickly a machine's workload
  can plausibly drift; both are configurable parameters, not derived
  constants.
- Containerization (Docker) and a real Kubernetes/KEDA deployment: still
  not started.
