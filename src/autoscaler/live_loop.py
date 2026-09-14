"""
Live forecasting loop (Step 22, Stage 4 of the "make the shadow harness
watch real infrastructure" project).

Step 21 (Stage 3) built `PrometheusMetricsSource`, a real, tested
`MetricsSource` connector, but nothing calls it yet. This module is the
scheduled loop that does: on a tick (`run_tick`, meant to be called every
`LiveLoopConfig.tick_seconds`, default 5 minutes), for every tracked node,
pull real history, run ARIMA through the existing, unmodified forecasting
and decision-engine code, and log the result. Through Step 25, no scaling
call existed anywhere in this module or anything it called -- every
function here either returned a value or wrote a log row via `ShadowStore`.
Step 26 adds exactly one real scaling call (`actuator.set_replicas`,
imported lazily inside `run_tick`, matching this module's existing
TensorFlow-isolation convention), gated behind
`LiveLoopConfig.enable_actuation` (default `False`) -- see "Path 3" below.
With actuation disabled (the default), this module's behavior is
byte-for-byte what it was through Step 25: observe and log only.

Two separate logging paths, not one
-------------------------------------
`shadow.py`'s data model (`ShadowWindowResult`) is fundamentally a
two-forecaster comparison -- ARIMA cost/SLA AND hybrid cost/SLA for the
same window, always both, because that comparison is what
`shadow.decide_assignment` gates on. A brand-new real node has never been
compared against the hybrid (no real trained hybrid model exists for it --
see "The residual hybrid: wired, not real" below) and defaults to ARIMA
(shadow.py's own established default, Step 17). Forcing every tick through
`shadow.py`'s two-forecaster shape would mean fabricating a hybrid number
that was never actually computed, just to satisfy the dataclass -- which
this module refuses to do. Instead:

1. **`observe_node_once`** -- ALWAYS runs, for every tracked node, every
   tick. Pulls this node's recent real history, fits+forecasts ARIMA
   (`arima_baseline._arima_forecast_once`), runs the SAME decision engine
   the `/forecast` endpoint uses (`decision._decide_scaling`, unmodified),
   and logs the single-forecaster decision via
   `ShadowStore.record_observed_decision` -- a new, simpler log, separate
   from the shadow-window tables. This is "watch real decisions
   accumulate," the thing this stage's own instructions asked to expose.
   Graceful degradation (Step 22 follow-up): if real history was fetched
   fine but the ARIMA fit/forecast step itself throws, this falls back to
   the reactive threshold policy (`simulation._reactive_decide_once`) on
   the current real reading rather than skipping the tick entirely --
   logged as `forecaster=REACTIVE_FALLBACK_FORECASTER`, still eligible
   for real actuation exactly like an ARIMA decision (same gate, same
   allow-list, same circuit breaker), per-tick rather than sticky. A
   Prometheus-fetch failure is a different, NOT-changed failure mode --
   see that function's own docstring.
2. **The hybrid shadow-window path** (`_build_hybrid_window` +
   `shadow_store.run_shadow_cycle`) -- runs ONLY for nodes ALREADY assigned
   to the hybrid (`state.current_forecaster == "hybrid"`), per this stage's
   explicit scoping. It re-validates (or reverts) that existing assignment
   against real data, using shadow.py's own unmodified re-evaluation
   cadence -- it does not, and structurally cannot, promote an ARIMA node
   to hybrid on its own. See below for why.

The residual hybrid: wired, not real
--------------------------------------
No production-ready hybrid-inference artifact exists anywhere in this
codebase. Step 16's residual-hybrid LSTM (predicting ARIMA's residuals)
only ever ran as an offline experiment script
(`experiments/run_hybrid_arima_lstm.py`) against the historical CSV,
trained fresh per run, and never saved a model file. On a brand-new real
cluster, every node starts on ARIMA (shadow.py's default) and there is
currently no path -- automatic or otherwise -- for the live loop itself to
produce a hybrid forecast for a node that hasn't already been assigned
one. This module's hybrid path is deliberately built and tested against a
STUB residual model (see `tests/test_live_loop.py`) to prove the plumbing
is correct, but has never run against a real pretrained artifact, because
none exists. `_load_hybrid_residual_model` raises `HybridModelUnavailable`
for exactly this reason -- caught one level up in `run_tick`, logged, and
skipped for that node that tick, not treated as a bug. Getting a node onto
the hybrid in the first place remains the existing, Step-18/19 manual path
(`/shadow/{id}/window`, e.g. fed from an offline analysis of enough
accumulated real Prometheus history) -- this module doesn't add a new one,
matching Step 17's caution against a static/automatic hybrid-assignment
rule.

No new scheduling dependency
-------------------------------
`run_scheduler_loop` is a plain blocking `while` loop with
`threading.Event.wait` as its sleep -- stdlib only, same "no new infra
dependency unless asked" preference this project applied to `shadow_store`
choosing `sqlite3` over an ORM (Step 19).
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from . import config, instrumentation, shadow
from .arima_baseline import ARIMA_ORDER, _arima_forecast_once, _arima_rolling_forecast, _inv_flat
from .decision import DecisionConfig, _decide_scaling
from .metrics_source import MetricsSource, resample_readings
from .shadow_store import ShadowStore, run_shadow_cycle
from .simulation import _reactive_decide_once

logger = logging.getLogger("autoscaler.live_loop")

DEFAULT_TICK_SECONDS = 300                 # 5 minutes, per this stage's instructions
DEFAULT_FIT_LOOKBACK_HOURS = 3.0           # history pulled each tick for the ARIMA-only observation
DEFAULT_SHADOW_WINDOW_HOURS = 24.0         # the "24h" a shadow window has meant since Step 18
DEFAULT_SHADOW_FIT_HOURS = 6.0             # history pulled to fit ARIMA before rolling through the shadow window
DEFAULT_HYBRID_MODEL_DIR = os.environ.get("LSTM_AUTOSCALER_HYBRID_MODEL_DIR", "models/hybrid_residual")
# Step 26 -- real actuation defaults. Namespace matches k8s/observer.yaml's
# own Namespace object; deployment name matches k8s/demo-workload.yaml's
# clearly-labeled demo target (see that manifest and actuator.py's module
# docstring for why a demo target, not a real service, is what gets scaled).
DEFAULT_ACTUATION_DEPLOYMENT = os.environ.get("LSTM_AUTOSCALER_ACTUATION_DEPLOYMENT", "demo-workload")
DEFAULT_ACTUATION_NAMESPACE = os.environ.get("LSTM_AUTOSCALER_ACTUATION_NAMESPACE", "lstm-autoscaler")
TRACKED_NODES_ENV_VAR = "LSTM_AUTOSCALER_TRACKED_NODES"

# ARIMA(2,0,1) needs more observations than its own parameter count to fit
# at all; this is a floor well above that, not a tight statistical bound.
MIN_FIT_POINTS = config.LOOKBACK_STEPS + 2

# Marks an observed decision produced by the reactive-threshold fallback
# (real CPU data was fetched, but the ARIMA fit/forecast step itself
# failed) instead of "arima" -- deliberately distinguishable in
# `/shadow/{id}/decisions` and the frontend rather than looking like a
# normal ARIMA tick. See `observe_node_once`'s docstring.
REACTIVE_FALLBACK_FORECASTER = "reactive_fallback"


class HybridModelUnavailable(Exception):
    """A node is assigned to the residual hybrid, but no pretrained
    per-machine residual model exists to serve it. Expected on every real
    node today -- see module docstring. Caught in `run_tick`; not a bug."""


@dataclass
class LiveLoopConfig:
    tick_seconds: int = DEFAULT_TICK_SECONDS
    fit_lookback_hours: float = DEFAULT_FIT_LOOKBACK_HOURS
    shadow_window_hours: float = DEFAULT_SHADOW_WINDOW_HOURS
    shadow_fit_hours: float = DEFAULT_SHADOW_FIT_HOURS
    demand_scale: float = config.DEMAND_SCALE
    safety_margin: float = config.SAFETY_MARGIN
    under_prov_weight: float = config.DEC_UNDER_WEIGHT
    order: Tuple[int, int, int] = ARIMA_ORDER
    hybrid_model_dir: str = DEFAULT_HYBRID_MODEL_DIR
    # Step 26 -- real actuation, off by default. `enable_actuation=False`
    # (the default) means run_tick's actuation branch never even imports
    # `actuator`, let alone calls it -- this project's safe, observe-only
    # behavior through Step 25 stays the default, not something that can
    # be silently switched on by forgetting a flag. `actuation_machine_id`
    # picks exactly ONE tracked node's recommendation to actuate from --
    # not an aggregate across nodes, deliberately, so "why did it scale"
    # always traces to one real forecast, never a blend.
    enable_actuation: bool = False
    actuation_machine_id: Optional[str] = None
    actuation_deployment: str = DEFAULT_ACTUATION_DEPLOYMENT
    actuation_namespace: str = DEFAULT_ACTUATION_NAMESPACE

    # Step 26 follow-up -- circuit breaker beyond the decision engine's own
    # +/-1-replica-per-tick step cap (that step's own "still open" list:
    # "no rate-limit beyond what the decision engine's own scale_step
    # already provides... worth revisiting if this ever points at
    # something real"). Without this, a real, persistent problem (RBAC
    # revoked, the deployment deleted, a real cluster outage) means
    # run_tick calls the Kubernetes API and fails EVERY tick forever,
    # indefinitely -- logged each time, but never backing off.
    # `actuation_circuit_breaker_threshold` consecutive real failures trips
    # it: real actuation attempts stop entirely for
    # `actuation_circuit_breaker_cooldown`, then exactly one probe attempt
    # is allowed -- success closes the breaker (resets the failure count),
    # another failure re-opens it for a fresh cooldown window. `run_tick`'s
    # OTHER work (observation, hybrid shadow-window scoring) is completely
    # unaffected either way -- this only ever gates the real `set_replicas`
    # call.
    #
    # Deliberately process-lifetime state, not persisted to `store` like
    # `last_recommended_servers` is: a pod restart clearing the breaker and
    # trying again is an acceptable, arguably correct reset point (many
    # real circuit-breaker implementations reset on process restart too),
    # and this project's own stated preference is "no new infra dependency
    # unless asked" (see shadow_store.py's sqlite choice, Step 19) --
    # nothing asked for cross-restart breaker persistence specifically.
    # The two leading-underscore fields below are mutable RUNTIME STATE
    # piggybacking on this otherwise-immutable-in-practice config object
    # (the same object `run_scheduler_loop` reuses across every tick, by
    # design -- see that function's docstring), not configuration; a test
    # or caller that wants an isolated breaker just constructs a fresh
    # `LiveLoopConfig`, exactly like every other test in this file already
    # does.
    actuation_circuit_breaker_threshold: int = 3
    actuation_circuit_breaker_cooldown: timedelta = timedelta(hours=1)
    _actuation_consecutive_failures: int = 0
    _actuation_circuit_opened_at: Optional[datetime] = None


# ── Path 1: always-on, single-forecaster observed decisions ─────────────────

def observe_node_once(machine_id: str, source: MetricsSource, store: ShadowStore, now: datetime,
                      cfg: LiveLoopConfig = LiveLoopConfig()) -> Optional[dict]:
    """One node's one-tick observation. Pulls `cfg.fit_lookback_hours` of
    real history via `source` (`fetch_readings` + `resample_readings`,
    both unmodified from Steps 20-21) -- a failure HERE (Prometheus
    unreachable, its own retry/backoff already exhausted) is NOT caught
    in this function; it propagates to `run_tick`'s own try/except
    exactly as before, since there is no real data at all for ANY policy
    to act on. Returns None (logging a warning, not raising) if there
    wasn't enough real data this tick to fit ARIMA at all -- a real
    scrape gap or a brand-new node are both legitimate reasons, same
    convention `MetricsSource.fetch_readings` already established for an
    empty result.

    With real data in hand, fits+forecasts ARIMA
    (`arima_baseline._arima_forecast_once`, built from the identical
    `ARIMA(...).fit()` call `_arima_rolling_forecast` already uses) and
    runs the SAME scaling decision the `/forecast` endpoint computes
    (`decision._decide_scaling`, unmodified). If THAT step itself raises
    (a numerical issue, insufficient variance, whatever -- distinct from
    the Prometheus-fetch failure above, which this function never even
    sees), falls back to the reactive threshold policy
    (`simulation._reactive_decide_once`) on the most recent real reading
    instead of producing no decision at all -- there IS real, current
    data, just no working forecast. Logged and recorded with
    `forecaster=REACTIVE_FALLBACK_FORECASTER`, clearly distinguishable
    from a real "arima" tick in `/shadow/{id}/decisions` and the
    frontend, with an empty `forecast_cpu_pct` -- there is no forecast to
    report, and fabricating one (e.g. repeating the current reading
    across the horizon) would be dishonest. This fallback is per-tick,
    not sticky: the very next tick tries ARIMA again from scratch.

    `current_servers` is a logging-only "shadow" count carried forward
    via `ShadowStore.get_last_recommended_servers`/`record_observed_
    decision`, never a real fleet size, regardless of which policy this
    tick used.
    """
    start = now - timedelta(hours=cfg.fit_lookback_hours)
    raw = source.fetch_readings(machine_id, start, now)
    resampled = resample_readings(raw)
    flat = resampled[config.FEATURE_COL].values.astype(np.float64)

    if len(flat) < MIN_FIT_POINTS:
        logger.warning(
            "live_loop: machine=%s only %d/%d real points in the last %.1fh -- skipping this tick",
            machine_id, len(flat), MIN_FIT_POINTS, cfg.fit_lookback_hours,
        )
        return None

    current_servers = store.get_last_recommended_servers(machine_id, default=config.SIM_INITIAL_SERVERS)

    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(flat.reshape(-1, 1)).flatten()

        forecast_scaled = _arima_forecast_once(scaled, config.HORIZON_STEPS, cfg.order)
        forecast_real = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

        planned_load = float(np.max(forecast_real)) * cfg.demand_scale * (1.0 + cfg.safety_margin)
        dec_cfg = DecisionConfig(under_prov_weight=cfg.under_prov_weight)
        action, recommended_servers = _decide_scaling(current_servers, planned_load, dec_cfg)

        forecaster = "arima"
        forecast_list = [round(float(v), 4) for v in forecast_real]
        planned_load_pct = round(planned_load, 4)
    except Exception:
        logger.exception(
            "live_loop: ARIMA fit/forecast failed for machine=%s on %d real points -- "
            "falling back to the reactive threshold policy",
            machine_id, len(flat),
        )
        current_cpu_pct = float(flat[-1])
        action, recommended_servers = _reactive_decide_once(
            current_cpu_pct, current_servers, demand_scale=cfg.demand_scale,
        )
        forecaster = REACTIVE_FALLBACK_FORECASTER
        forecast_list = []
        planned_load_pct = round(current_cpu_pct * cfg.demand_scale, 4)

    store.record_observed_decision(
        machine_id, now, forecaster=forecaster,
        forecast_cpu_pct=forecast_list, planned_load_pct=planned_load_pct,
        current_servers=current_servers, recommended_servers=recommended_servers, action=action,
    )
    instrumentation.DECISIONS_TOTAL.labels(
        machine_id=machine_id, forecaster=forecaster, action=action.split(" ", 1)[0],
    ).inc()
    instrumentation.RECOMMENDED_SERVERS.labels(machine_id=machine_id).set(recommended_servers)
    logger.info(
        "live_loop: observed machine=%s forecaster=%s forecast=%s planned_load=%.2f%% "
        "current=%d recommended=%d action=%s",
        machine_id, forecaster, forecast_list, planned_load_pct, current_servers, recommended_servers, action,
    )
    return {
        "machine_id": machine_id, "forecaster": forecaster, "forecast_cpu_pct": forecast_list,
        "planned_load_pct": planned_load_pct, "current_servers": current_servers,
        "recommended_servers": recommended_servers, "action": action,
    }


# ── Path 2: hybrid re-validation, only for already-hybrid-assigned nodes ───

def _load_hybrid_residual_model(machine_id: str, model_dir: str):
    """Lazily imports TensorFlow only on an actual load attempt -- keeps
    this module importable (and the ARIMA-only path testable/runnable)
    without a TensorFlow install, same isolation principle
    `forecasting.py`'s module docstring documents. Looks for
    `{model_dir}/{machine_id}.keras`; this per-machine convention has no
    real trained artifact behind it anywhere yet (see module docstring)."""
    path = os.path.join(model_dir, f"{machine_id}.keras")
    if not os.path.exists(path):
        raise HybridModelUnavailable(
            f"no pretrained residual model for machine_id={machine_id!r} at {path!r}"
        )
    from .forecasting import _build_lstm_model
    model = _build_lstm_model(config.LOOKBACK_STEPS, config.HORIZON_STEPS)
    model.load_weights(path)
    return model


def _build_hybrid_window(machine_id: str, source: MetricsSource, cfg: LiveLoopConfig,
                         now: datetime, model_loader: Callable = _load_hybrid_residual_model
                         ) -> shadow.ShadowWindowResult:
    """Build one live ARIMA-vs-hybrid shadow window over the last
    `cfg.shadow_window_hours`, for `shadow_store.run_shadow_cycle` to bank.
    Checks the residual model exists FIRST (cheap) before pulling
    `cfg.shadow_fit_hours + cfg.shadow_window_hours` of real Prometheus
    history (not cheap -- see Step 21's progress doc on instant-query call
    volume) -- raises `HybridModelUnavailable` immediately if not, same
    exception either way.

    Reuses, unmodified: `arima_baseline._arima_rolling_forecast` (the
    walk-forward rolling forecast, fit on the preceding `shadow_fit_hours`
    then rolled through the window), `arima_baseline._inv_flat`
    (inverse-scaling), and `shadow.run_shadow_window` (the same scoring
    path Steps 18-19's `/shadow/{id}/window` endpoint already uses for
    hand-submitted windows) -- this function's only new logic is producing
    the (arima_pred_real, hybrid_pred_real, y_actual_real) triple those
    reused functions need, from live data instead of a hand-submitted
    request body.

    Residual-hybrid mechanism (Step 16, unchanged): hybrid forecast =
    ARIMA's forecast + the residual model's prediction from the identical
    lookback window, in scaled space, clipped to [0, 1] before
    inverse-transforming -- same convention `_arima_rolling_forecast`
    itself uses for its own clip.
    """
    model = model_loader(machine_id, cfg.hybrid_model_dir)  # raises HybridModelUnavailable first, cheaply

    from .data import _make_sequences

    fit_start = now - timedelta(hours=cfg.shadow_fit_hours + cfg.shadow_window_hours)
    window_start = now - timedelta(hours=cfg.shadow_window_hours)

    raw = source.fetch_readings(machine_id, fit_start, now)
    resampled = resample_readings(raw)
    fit_flat = resampled.loc[resampled.index < window_start, config.FEATURE_COL].values.astype(np.float64)
    target_flat = resampled.loc[resampled.index >= window_start, config.FEATURE_COL].values.astype(np.float64)

    min_needed = config.LOOKBACK_STEPS + config.HORIZON_STEPS
    if len(fit_flat) < MIN_FIT_POINTS or len(target_flat) < min_needed:
        raise HybridModelUnavailable(
            f"not enough real history for a live shadow window on machine_id={machine_id!r} "
            f"(fit={len(fit_flat)} pts, target={len(target_flat)} pts, need >={min_needed} target pts)"
        )

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(fit_flat.reshape(-1, 1))
    fit_scaled = scaler.transform(fit_flat.reshape(-1, 1)).flatten()
    target_scaled = scaler.transform(target_flat.reshape(-1, 1)).flatten()

    arima_pred_scaled, y_true_scaled = _arima_rolling_forecast(
        fit_scaled, np.array([], dtype=fit_scaled.dtype), target_scaled,
        config.LOOKBACK_STEPS, config.HORIZON_STEPS, cfg.order,
    )

    X, _ = _make_sequences(target_scaled.reshape(-1, 1), config.LOOKBACK_STEPS, config.HORIZON_STEPS)
    residual_scaled = np.asarray(model.predict(X, verbose=0))
    hybrid_pred_scaled = np.clip(arima_pred_scaled + residual_scaled, 0.0, 1.0)

    arima_pred_real = _inv_flat(arima_pred_scaled, scaler)
    hybrid_pred_real = _inv_flat(hybrid_pred_scaled, scaler)
    y_actual_real = _inv_flat(y_true_scaled, scaler)

    dec_cfg = DecisionConfig(under_prov_weight=cfg.under_prov_weight)
    return shadow.run_shadow_window(
        machine_id, window_start, now, y_actual_real,
        arima_pred_real, dec_cfg, cfg.safety_margin,
        hybrid_pred_real, dec_cfg, cfg.safety_margin,
        demand_scale=cfg.demand_scale,
    )


# ── One full tick ────────────────────────────────────────────────────────────

def _record_actuation_failure(cfg: "LiveLoopConfig", machine_id: str, now: datetime) -> None:
    """Shared by both `except` branches in run_tick's actuation block --
    increments the consecutive-failure counter and (re)opens the circuit
    breaker's cooldown window from THIS failure once the threshold is
    reached, whether this is the failure that first trips it or a failed
    post-cooldown probe extending an already-open breaker. See
    LiveLoopConfig's own comment for the full design."""
    cfg._actuation_consecutive_failures += 1
    if cfg._actuation_consecutive_failures >= cfg.actuation_circuit_breaker_threshold:
        cfg._actuation_circuit_opened_at = now
        logger.warning(
            "live_loop: actuation circuit breaker TRIPPED for machine=%s after %d consecutive "
            "failures -- backing off real actuation attempts for %s",
            machine_id, cfg._actuation_consecutive_failures, cfg.actuation_circuit_breaker_cooldown,
        )


def run_tick(store: ShadowStore, source: MetricsSource, machine_ids: Sequence[str], now: datetime,
            cfg: LiveLoopConfig = LiveLoopConfig(),
            model_loader: Callable = _load_hybrid_residual_model) -> dict:
    """One full tick across every tracked node -- Stage 4's scheduled unit
    of work. For every node: always run `observe_node_once` (ARIMA-only,
    logged as an observed decision). Additionally, only for nodes ALREADY
    assigned to the hybrid, build a live shadow window and feed it into
    `shadow_store.run_shadow_cycle` (which itself no-ops unless this node
    is actually due for re-evaluation -- Step 19's existing cadence logic,
    unmodified). See module docstring for why these are two separate paths.

    Through Step 25, no scaling call existed anywhere in this function --
    logging only. Step 26 adds exactly one: if `cfg.enable_actuation` is
    True (default False) AND this tick's machine is `cfg.actuation_machine_id`
    AND `observe_node_once` produced a decision, the recommended replica
    count is applied for real via `actuator.set_replicas` -- see that
    module and `LiveLoopConfig`'s own comments. Every other node, and every
    node when actuation is disabled, is unaffected: still observe and log
    only. A per-node failure anywhere in this function (a Prometheus error,
    a missing hybrid model, a failed actuation call, anything else) is
    caught, logged, and does not stop the rest of the tick from running for
    other nodes -- actuation failures are handled with exactly the same
    "log it, move on" convention as every other real-infrastructure call
    this project makes.

    Returns a summary dict of machine_ids grouped by outcome, useful for
    tests and for a caller that wants to log/inspect a tick's shape without
    re-deriving it from individual log calls.
    """
    summary = {
        "observed": [], "skipped": [], "shadow_evaluated": [], "shadow_skipped": [],
        "actuated": [], "actuation_skipped": [], "reconciled": [],
    }

    for machine_id in machine_ids:
        try:
            decision = observe_node_once(machine_id, source, store, now, cfg)
        except Exception:
            logger.exception("live_loop: ARIMA observation failed for machine=%s", machine_id)
            decision = None
        summary["observed" if decision is not None else "skipped"].append(machine_id)

        if cfg.enable_actuation and decision is not None and machine_id == cfg.actuation_machine_id:
            from . import actuator  # lazy import -- see module docstring

            breaker_open = (
                cfg._actuation_circuit_opened_at is not None
                and now - cfg._actuation_circuit_opened_at < cfg.actuation_circuit_breaker_cooldown
            )
            if breaker_open:
                cooldown_until = cfg._actuation_circuit_opened_at + cfg.actuation_circuit_breaker_cooldown
                logger.warning(
                    "live_loop: actuation circuit breaker OPEN for machine=%s (%d consecutive "
                    "failures) -- skipping this tick's real call, next probe attempt after %s",
                    machine_id, cfg._actuation_consecutive_failures, cooldown_until,
                )
                summary["actuation_skipped"].append(machine_id)
            else:
                try:
                    recommended = decision["recommended_servers"]

                    # Read-before-write reconciliation (Step 26 follow-up): the
                    # real-cluster verification in that step found this write
                    # was always blind to the real deployment's actual replica
                    # count -- `recommended` above comes entirely from this
                    # loop's own internal ledger (`store.last_recommended_servers`,
                    # set by `observe_node_once`/`record_observed_decision`),
                    # which can drift from reality (a manual `kubectl scale`, a
                    # pod restart that lost state before Step 19's persistence,
                    # or simply the very first tick after actuation is enabled
                    # -- exactly what caused the real 2->5 single-tick jump this
                    # step's own progress doc documents). `get_current_replicas`
                    # failing here (no real cluster reachable, an RBAC problem,
                    # anything -- always `ActuationError`, see actuator.py) is
                    # treated as "couldn't verify" rather than a hard stop: fall
                    # back to the un-reconciled `recommended`, matching this
                    # function's existing per-tick "log it, move on" convention,
                    # rather than a new failure mode that blocks actuation
                    # entirely whenever a single read fails.
                    try:
                        real_current = actuator.get_current_replicas(
                            cfg.actuation_deployment, cfg.actuation_namespace,
                        )
                    except actuator.ActuationError as e:
                        logger.warning(
                            "live_loop: could not read the real replica count for machine=%s "
                            "before actuating (%s) -- proceeding unreconciled with this tick's "
                            "internally-tracked recommendation", machine_id, e,
                        )
                        real_current = None

                    if real_current is not None and real_current != decision["current_servers"]:
                        dec_cfg = DecisionConfig(under_prov_weight=cfg.under_prov_weight)
                        _, recommended = _decide_scaling(real_current, decision["planned_load_pct"], dec_cfg)
                        logger.warning(
                            "live_loop: reconciling machine=%s -- internal ledger said current=%d "
                            "but the real cluster reports %d replicas; recomputed recommended=%d "
                            "(this tick's un-reconciled value was %d)",
                            machine_id, decision["current_servers"], real_current,
                            recommended, decision["recommended_servers"],
                        )
                        summary["reconciled"].append(machine_id)

                    actuator.set_replicas(cfg.actuation_deployment, cfg.actuation_namespace, recommended)
                    # Keep the ledger truthful to what was actually just applied
                    # -- matters whether or not reconciliation changed anything
                    # above, since `observe_node_once` already wrote this tick's
                    # UN-reconciled `recommended_servers` into the same ledger
                    # column earlier in this loop iteration.
                    store.set_last_recommended_servers(machine_id, recommended)
                    summary["actuated"].append(machine_id)
                    instrumentation.ACTUATION_TOTAL.labels(machine_id=machine_id, result="success").inc()
                    # A real success closes the breaker outright, whether this
                    # was ordinary operation or a post-cooldown probe attempt.
                    cfg._actuation_consecutive_failures = 0
                    cfg._actuation_circuit_opened_at = None
                except actuator.ActuationError as e:
                    logger.warning("live_loop: %s -- actuation skipped this tick", e)
                    summary["actuation_skipped"].append(machine_id)
                    instrumentation.ACTUATION_TOTAL.labels(machine_id=machine_id, result="failure").inc()
                    _record_actuation_failure(cfg, machine_id, now)
                except Exception:
                    logger.exception("live_loop: actuation failed for machine=%s", machine_id)
                    summary["actuation_skipped"].append(machine_id)
                    instrumentation.ACTUATION_TOTAL.labels(machine_id=machine_id, result="failure").inc()
                    _record_actuation_failure(cfg, machine_id, now)

            breaker_open_now = (
                cfg._actuation_circuit_opened_at is not None
                and now - cfg._actuation_circuit_opened_at < cfg.actuation_circuit_breaker_cooldown
            )
            instrumentation.CIRCUIT_BREAKER_OPEN.labels(machine_id=machine_id).set(1.0 if breaker_open_now else 0.0)

        state = store.load_state(machine_id)
        if state.current_forecaster != "hybrid":
            continue

        def _get_window(mid=machine_id):
            return _build_hybrid_window(mid, source, cfg, now, model_loader)

        try:
            run_shadow_cycle(
                store, machine_id, now, _get_window,
                min_windows=shadow.DEFAULT_MIN_WINDOWS, cadence_days=shadow.DEFAULT_REEVAL_CADENCE_DAYS,
            )
            summary["shadow_evaluated"].append(machine_id)
        except HybridModelUnavailable as e:
            logger.warning("live_loop: %s -- skipping shadow-window scoring this tick", e)
            summary["shadow_skipped"].append(machine_id)
        except Exception:
            logger.exception("live_loop: shadow-window scoring failed for machine=%s", machine_id)
            summary["shadow_skipped"].append(machine_id)

    return summary


# ── Node discovery + the blocking scheduler (Stage 5 will invoke this) ─────

def resolve_tracked_machine_ids(source: MetricsSource,
                                env_var: str = TRACKED_NODES_ENV_VAR) -> List[str]:
    """Explicit config wins: a non-empty `$LSTM_AUTOSCALER_TRACKED_NODES`
    (comma-separated) is used as-is. Otherwise, if `source` supports
    discovery (`PrometheusMetricsSource.list_machine_ids`, an extension
    beyond the base `MetricsSource` contract -- see Step 21), ask it what's
    live right now. Raises if neither is available -- silently tracking
    zero nodes forever would be a much worse failure mode than an explicit
    startup error."""
    explicit = os.environ.get(env_var, "").strip()
    if explicit:
        return [m.strip() for m in explicit.split(",") if m.strip()]
    list_ids = getattr(source, "list_machine_ids", None)
    if callable(list_ids):
        return list_ids()
    raise ValueError(
        f"No tracked nodes configured (${env_var}) and {type(source).__name__} "
        "has no list_machine_ids() for discovery."
    )


def run_scheduler_loop(store: ShadowStore, source: MetricsSource,
                       get_tracked_machine_ids: Callable[[], Sequence[str]],
                       cfg: LiveLoopConfig = LiveLoopConfig(),
                       stop_event=None, max_ticks: Optional[int] = None) -> None:
    """Blocking loop: calls `run_tick` every `cfg.tick_seconds`, using
    `get_tracked_machine_ids()` freshly each tick (so newly-joined/departed
    nodes are picked up without a restart). Stdlib only -- a plain `while`
    loop with `threading.Event.wait` as its sleep, no new scheduling
    dependency (matching this project's established preference, e.g.
    `shadow_store.py` choosing `sqlite3` over an ORM in Step 19).

    `stop_event` (a `threading.Event`) and `max_ticks` are both optional
    and exist for tests/bounded runs -- production use (Stage 5) passes
    neither and this loops forever. A single tick's exception is caught and
    logged, not allowed to kill the loop -- an unattended process should
    keep trying next tick, not die on one bad Prometheus response.
    """
    import threading
    stop_event = stop_event or threading.Event()
    ticks = 0
    while not stop_event.is_set():
        now = datetime.now(timezone.utc)
        try:
            machine_ids = get_tracked_machine_ids()
            run_tick(store, source, machine_ids, now, cfg)
        except Exception:
            logger.exception("live_loop: tick failed at %s", now)
        ticks += 1
        if max_ticks is not None and ticks >= max_ticks:
            break
        stop_event.wait(cfg.tick_seconds)
