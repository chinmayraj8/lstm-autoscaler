"""
Shadow-mode empirical evaluation harness (Step 18).

Step 17 found no static machine feature (burstiness, demand_scale, cluster
membership, ARIMA residual autocorrelation) predicts which machines show a
real residual-hybrid advantage over plain ARIMA (Step 16). This module is
the empirical alternative that follows from that: for a given machine, run
BOTH forecasters' forecasts through this project's existing, unmodified
decision engine and simulator (`decision._build_lstm_targets`,
`simulation._run_simulation`, `simulation._compute_cost_score` -- no new
scoring logic is written here, everything is reused verbatim) against real
incoming data, in parallel, without either one actually controlling the
real fleet. Only after the hybrid has *consistently* beaten ARIMA across
several independent 24-hour windows, at this project's own statistical bar
(`simulation.verdict`, the same combined-±1σ rule used since Step 11), does
a machine get assigned to it. Otherwise it defaults to ARIMA. Assignments
are re-checked periodically (a machine's workload can drift), and every
assignment CHANGE is logged with a timestamp and the evidence that
triggered it.

What this deliberately is NOT
------------------------------
This is a shadow-mode measurement-and-gating harness with repeat-window
statistical gating -- not a continuously-exploring bandit, an online
learner, or anything that adapts within a window. It makes one discrete
decision (which forecaster controls a machine) at discrete re-evaluation
points, based on a fixed statistical rule applied to a fixed number of
recently-banked windows. It does not explore a distribution of policies,
does not use any exploration/exploitation tradeoff, and does not update
mid-window. Calling it "adaptive" beyond that -- one slow, auditable,
evidence-logged decision per machine per cadence -- would overclaim what it
does.

Why repeated windows, not one
-------------------------------
Steps 10 and 13/14 both found that a single train/val/test split (or a
single grid-search-chosen parameter) can look decisively good on one slice
of data and fail to generalize (m_2085's Reactive threshold; m_2380's
apparent multistep win) -- the exact fragility a one-shot 24-hour shadow
decision would reproduce in production. Requiring the hybrid to win EVERY
one of several independent windows, not just win on average, is this
module's guard against exactly that failure mode.

This module has no TensorFlow or ARIMA-fitting dependency -- like
decision.py and simulation.py, it only needs numpy, so it can be tested
(see tests/test_shadow.py) with plain synthetic arrays, no real dataset or
GPU required. It does not forecast anything itself: callers (the real
online serving path, or a scheduled offline job) are responsible for
producing each window's `(y_pred_real, y_actual_real)` arrays from
whatever ARIMA/hybrid pipeline is in use; this module only scores, banks,
and gates on them.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, List, Optional, Tuple

import numpy as np

from .decision import DecisionConfig, _build_lstm_targets
from .simulation import SimConfig, _compute_cost_score, _run_simulation, verdict

logger = logging.getLogger("autoscaler.shadow")

DEFAULT_MIN_WINDOWS = 3
DEFAULT_REEVAL_CADENCE_DAYS = 30


# ── Data model ──────────────────────────────────────────────────────────────

@dataclass
class ShadowWindowResult:
    """One 24-hour (or caller-defined) shadow window's scored comparison.
    Both costs come from the identical decision engine + simulator, applied
    to the same real demand series -- only the forecast feeding in differs.

    `hybrid_model_version` is a short content hash (see
    `live_loop._load_hybrid_residual_model`) of the exact `.keras` file the
    hybrid forecast for this window was produced from -- for auditability
    only (which weights produced this number), not a rollback mechanism.
    None for windows with no real hybrid model behind them at all (e.g.
    `/shadow/{id}/window`'s hand-submitted forecasts, which aren't tied to
    any specific loaded artifact)."""
    machine_id: str
    window_start: datetime
    window_end: datetime
    arima_cost: float
    arima_sla_pct: float
    hybrid_cost: float
    hybrid_sla_pct: float
    hybrid_model_version: Optional[str] = None


@dataclass
class AssignmentChange:
    """A logged record of one assignment decision that changed a machine's
    controlling forecaster -- the audit trail this step requires."""
    timestamp: datetime
    machine_id: str
    old_forecaster: str
    new_forecaster: str
    verdict: str
    n_windows: int
    evidence: str


@dataclass
class MachineShadowState:
    """Per-machine shadow-evaluation state. `current_forecaster` defaults to
    "arima" -- a machine with no confirmed assignment yet is, by
    definition, on ARIMA (this project's own established default, Step 17)
    until the hybrid earns its way in."""
    machine_id: str
    current_forecaster: str = "arima"
    last_evaluated_at: Optional[datetime] = None
    window_results: List[ShadowWindowResult] = field(default_factory=list)
    assignment_history: List[AssignmentChange] = field(default_factory=list)


# ── Scoring: reuses the existing decision engine + simulator, unmodified ────

def score_forecaster_window(y_pred_real: np.ndarray, y_actual_real: np.ndarray,
                            dec_cfg: DecisionConfig, sim_cfg: SimConfig,
                            demand_scale: float, safety_margin: float,
                            target_builder=_build_lstm_targets) -> Tuple[float, float]:
    """Run one forecaster's (y_pred_real, y_actual_real) pair for one window
    through the existing decision engine + simulator + cost scorer -- the
    same three functions every experiment in this project has used since
    Step 8. `target_builder` defaults to `_build_lstm_targets` (the
    original greedy engine, the one Step 16's hybrid evidence was produced
    under); pass `_build_multistep_targets` to score under the multistep
    engine instead if that's ever the deployed configuration.

    Returns (cost_score, sla_violation_rate_pct).
    """
    targets, demand = target_builder(y_pred_real, y_actual_real, dec_cfg, sim_cfg, demand_scale, safety_margin)
    metrics = _run_simulation(demand, targets, sim_cfg)
    cost = _compute_cost_score(metrics, dec_cfg)
    n = metrics.total_steps
    sla_pct = round(metrics.sla_violations / n * 100, 4)
    return cost, sla_pct


def run_shadow_window(machine_id: str, window_start: datetime, window_end: datetime,
                      y_actual_real: np.ndarray,
                      arima_pred_real: np.ndarray, arima_dec_cfg: DecisionConfig, arima_safety_margin: float,
                      hybrid_pred_real: np.ndarray, hybrid_dec_cfg: DecisionConfig, hybrid_safety_margin: float,
                      demand_scale: float, sim_cfg: Optional[SimConfig] = None,
                      target_builder=_build_lstm_targets,
                      hybrid_model_version: Optional[str] = None) -> ShadowWindowResult:
    """Score one shadow window for both forecasters against the SAME real
    demand data (`y_actual_real`) -- neither forecaster's targets are ever
    applied to the real fleet here; this only computes what each WOULD have
    cost, via the unmodified simulator. `arima_dec_cfg`/`hybrid_dec_cfg`
    are each forecaster's own already-validation-tuned decision params
    (produced elsewhere, e.g. `tune_on_validation`/`tune_arima_on_validation`
    -- this module does no tuning of its own). `hybrid_model_version`
    (see `ShadowWindowResult`) is passed straight through -- this function
    has no opinion on where it came from, only `live_loop.py`'s live path
    ever supplies one."""
    sim_cfg = sim_cfg or SimConfig()
    arima_cost, arima_sla = score_forecaster_window(
        arima_pred_real, y_actual_real, arima_dec_cfg, sim_cfg, demand_scale, arima_safety_margin, target_builder)
    hybrid_cost, hybrid_sla = score_forecaster_window(
        hybrid_pred_real, y_actual_real, hybrid_dec_cfg, sim_cfg, demand_scale, hybrid_safety_margin, target_builder)
    return ShadowWindowResult(
        machine_id=machine_id, window_start=window_start, window_end=window_end,
        arima_cost=arima_cost, arima_sla_pct=arima_sla,
        hybrid_cost=hybrid_cost, hybrid_sla_pct=hybrid_sla,
        hybrid_model_version=hybrid_model_version,
    )


# ── Cumulative logging (cost/SLA totals, independent of the pass/fail gate) ─

def cumulative_summary(window_results: List[ShadowWindowResult]) -> dict:
    """Cumulative cost/SLA for each forecaster across all banked windows --
    for logging/reporting. Deliberately separate from `decide_assignment`,
    which uses the same underlying numbers but run through the statistical
    gate; this is just "what happened," with no verdict attached."""
    if not window_results:
        return {"n_windows": 0}
    arima_costs = [w.arima_cost for w in window_results]
    hybrid_costs = [w.hybrid_cost for w in window_results]
    arima_slas = [w.arima_sla_pct for w in window_results]
    hybrid_slas = [w.hybrid_sla_pct for w in window_results]
    return {
        "n_windows": len(window_results),
        "arima_cost_mean": float(np.mean(arima_costs)),
        "arima_cost_total": float(np.sum(arima_costs)),
        "arima_cost_std": float(np.std(arima_costs)),
        "arima_sla_mean_pct": float(np.mean(arima_slas)),
        "hybrid_cost_mean": float(np.mean(hybrid_costs)),
        "hybrid_cost_total": float(np.sum(hybrid_costs)),
        "hybrid_cost_std": float(np.std(hybrid_costs)),
        "hybrid_sla_mean_pct": float(np.mean(hybrid_slas)),
    }


# ── Assignment gating: consistency AND the project's own statistical bar ───

def hybrid_wins_every_window(window_results: List[ShadowWindowResult]) -> bool:
    """"Consistently" (this step's word) is operationalized strictly: the
    hybrid's cost must be lower than ARIMA's in EVERY banked window, not
    just on average -- a machine that wins big in 2 windows and loses badly
    in a 3rd does not pass, even if the mean still favors the hybrid,
    because that's exactly the single-bad-window fragility this project
    has already been burned by twice (m_2085, m_2380)."""
    return len(window_results) > 0 and all(w.hybrid_cost < w.arima_cost for w in window_results)


def aggregate_verdict(window_results: List[ShadowWindowResult]) -> str:
    """The project's standard combined-±1σ classification (`simulation.verdict`),
    applied across shadow windows exactly the way this project has applied
    it across LSTM training seeds everywhere else. Returns one of
    "confirmed cheaper" / "directional (within ±1σ)" / "tied (within ±1σ)" /
    "confirmed more expensive" (all relative to the hybrid)."""
    hybrid_costs = np.array([w.hybrid_cost for w in window_results])
    arima_costs = np.array([w.arima_cost for w in window_results])
    return verdict(float(hybrid_costs.mean()), float(arima_costs.mean()),
                   float(hybrid_costs.std()), float(arima_costs.std()))


def decide_assignment(window_results: List[ShadowWindowResult],
                      min_windows: int = DEFAULT_MIN_WINDOWS) -> Tuple[str, str, str]:
    """The gate: returns (recommended_forecaster, verdict_label, evidence).

    Defaults to "arima" unless BOTH conditions hold:
      1. at least `min_windows` shadow windows have been banked,
      2. the hybrid wins every one of them (`hybrid_wins_every_window`), AND
      3. the aggregate verdict across those windows is "confirmed cheaper"
         (`aggregate_verdict`, the same statistical bar used everywhere
         else in this project).
    Consistency alone (2) without statistical significance (3), or
    significance without consistency, is not enough -- both are required,
    matching this step's explicit instruction.
    """
    if len(window_results) < min_windows:
        return ("arima", "insufficient_windows",
                f"only {len(window_results)}/{min_windows} shadow windows completed")

    consistent = hybrid_wins_every_window(window_results)
    v = aggregate_verdict(window_results)

    if consistent and v == "confirmed cheaper":
        return ("hybrid", v, f"hybrid confirmed cheaper across all {len(window_results)} windows")

    reason = "inconsistent across windows" if not consistent else v
    return ("arima", v, f"hybrid did not clear the bar ({reason}, {len(window_results)} windows)")


# ── State management: banking windows, periodic re-evaluation, change log ──

def record_shadow_window(state: MachineShadowState, window: ShadowWindowResult,
                         keep_last_n: int = DEFAULT_MIN_WINDOWS) -> None:
    """Bank a new shadow window, keeping only the most recent `keep_last_n`.
    A machine's assignment is judged on its RECENT shadow behavior, not an
    ever-growing history -- otherwise a machine that drifted after being
    correctly assigned to ARIMA six months ago could never accumulate
    enough fresh evidence to be re-evaluated fairly (old, possibly-stale
    windows would keep diluting new ones)."""
    state.window_results.append(window)
    if len(state.window_results) > keep_last_n:
        state.window_results = state.window_results[-keep_last_n:]
    summary = cumulative_summary(state.window_results)
    logger.info(
        "shadow window banked machine=%s window=[%s, %s] "
        "arima_cost=%.6f hybrid_cost=%.6f "
        "cumulative(n=%d): arima=%.6f±%.6f hybrid=%.6f±%.6f",
        state.machine_id, window.window_start, window.window_end,
        window.arima_cost, window.hybrid_cost,
        summary["n_windows"], summary["arima_cost_mean"], summary["arima_cost_std"],
        summary["hybrid_cost_mean"], summary["hybrid_cost_std"],
    )


def is_reevaluation_due(state: MachineShadowState, now: datetime,
                        cadence_days: int = DEFAULT_REEVAL_CADENCE_DAYS) -> bool:
    """True if this machine has never been evaluated, or its last
    evaluation was at least `cadence_days` ago."""
    if state.last_evaluated_at is None:
        return True
    return (now - state.last_evaluated_at) >= timedelta(days=cadence_days)


def evaluate_and_maybe_reassign(state: MachineShadowState, now: datetime,
                                min_windows: int = DEFAULT_MIN_WINDOWS) -> Optional[AssignmentChange]:
    """Run `decide_assignment` against the machine's currently-banked
    windows, stamp `last_evaluated_at`, and update `current_forecaster` if
    the recommendation differs from what's currently assigned -- logging an
    `AssignmentChange` (with timestamp and evidence) either way a change
    happens, whether that's the first-ever assignment to the hybrid or a
    later reversion back to ARIMA after workload drift. Returns the change
    record if one occurred, else None (including "no change because
    nothing was banked yet" or "no change because the existing assignment
    was reconfirmed")."""
    recommended, verdict_label, evidence = decide_assignment(state.window_results, min_windows)
    state.last_evaluated_at = now
    if recommended != state.current_forecaster:
        change = AssignmentChange(
            timestamp=now, machine_id=state.machine_id,
            old_forecaster=state.current_forecaster, new_forecaster=recommended,
            verdict=verdict_label, n_windows=len(state.window_results), evidence=evidence,
        )
        state.current_forecaster = recommended
        state.assignment_history.append(change)
        logger.info(
            "assignment change machine=%s %s -> %s at=%s verdict=%s evidence=%s",
            state.machine_id, change.old_forecaster, change.new_forecaster,
            change.timestamp, change.verdict, change.evidence,
        )
        return change
    logger.debug(
        "assignment reconfirmed machine=%s forecaster=%s verdict=%s evidence=%s",
        state.machine_id, state.current_forecaster, verdict_label, evidence,
    )
    return None


def maybe_run_shadow_cycle(state: MachineShadowState, now: datetime,
                           get_window_fn: Callable[[], ShadowWindowResult],
                           min_windows: int = DEFAULT_MIN_WINDOWS,
                           cadence_days: int = DEFAULT_REEVAL_CADENCE_DAYS,
                           keep_last_n: int = DEFAULT_MIN_WINDOWS) -> Optional[AssignmentChange]:
    """One measurement tick, meant to be invoked periodically by the caller
    (e.g. once a day while a machine is still accumulating its first
    `min_windows` shadow windows; once a month afterward). This module owns
    the windowing/statistics/state-tracking; it has no opinion on wall-clock
    scheduling -- how often to CALL this is the caller's decision, not
    enforced here.

    No-ops (returns None without calling `get_window_fn`, so no shadow
    compute is wasted) if the machine already has enough banked windows to
    have been evaluated at least once AND its re-evaluation cadence hasn't
    elapsed. Otherwise pulls one more window via `get_window_fn()` (the
    caller's job to actually score a real 24h window -- typically
    `run_shadow_window` fed by that machine's real forecasts/demand),
    banks it, and re-runs the assignment decision.
    """
    already_evaluated_and_stable = (
        state.last_evaluated_at is not None and len(state.window_results) >= min_windows
    )
    if already_evaluated_and_stable and not is_reevaluation_due(state, now, cadence_days):
        return None

    window = get_window_fn()
    record_shadow_window(state, window, keep_last_n=keep_last_n)
    return evaluate_and_maybe_reassign(state, now, min_windows=min_windows)
