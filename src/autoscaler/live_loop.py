"""
Live forecasting loop (Step 22, Stage 4 of the "make the shadow harness
watch real infrastructure" project).

Step 21 (Stage 3) built `PrometheusMetricsSource`, a real, tested
`MetricsSource` connector, but nothing calls it yet. This module is the
scheduled loop that does: on a tick (`run_tick`, meant to be called every
`LiveLoopConfig.tick_seconds`, default 5 minutes), for every tracked node,
pull real history, run ARIMA through the existing, unmodified forecasting
and decision-engine code, and log the result. **No scaling call exists
anywhere in this module or anything it calls** -- every function here
either returns a value or writes a log row via `ShadowStore`; none of them
can reach a real fleet.

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

from . import config, shadow
from .arima_baseline import ARIMA_ORDER, _arima_forecast_once, _arima_rolling_forecast, _inv_flat
from .decision import DecisionConfig, _decide_scaling
from .metrics_source import MetricsSource, resample_readings
from .shadow_store import ShadowStore, run_shadow_cycle

logger = logging.getLogger("autoscaler.live_loop")

DEFAULT_TICK_SECONDS = 300                 # 5 minutes, per this stage's instructions
DEFAULT_FIT_LOOKBACK_HOURS = 3.0           # history pulled each tick for the ARIMA-only observation
DEFAULT_SHADOW_WINDOW_HOURS = 24.0         # the "24h" a shadow window has meant since Step 18
DEFAULT_SHADOW_FIT_HOURS = 6.0             # history pulled to fit ARIMA before rolling through the shadow window
DEFAULT_HYBRID_MODEL_DIR = os.environ.get("LSTM_AUTOSCALER_HYBRID_MODEL_DIR", "models/hybrid_residual")
TRACKED_NODES_ENV_VAR = "LSTM_AUTOSCALER_TRACKED_NODES"

# ARIMA(2,0,1) needs more observations than its own parameter count to fit
# at all; this is a floor well above that, not a tight statistical bound.
MIN_FIT_POINTS = config.LOOKBACK_STEPS + 2


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


# ── Path 1: always-on, single-forecaster observed decisions ─────────────────

def observe_node_once(machine_id: str, source: MetricsSource, store: ShadowStore, now: datetime,
                      cfg: LiveLoopConfig = LiveLoopConfig()) -> Optional[dict]:
    """One node's one-tick ARIMA observation. Pulls `cfg.fit_lookback_hours`
    of real history via `source` (`fetch_readings` + `resample_readings`,
    both unmodified from Steps 20-21), fits+forecasts ARIMA
    (`arima_baseline._arima_forecast_once`, built from the identical
    `ARIMA(...).fit()` call `_arima_rolling_forecast` already uses), and
    runs the SAME scaling decision the `/forecast` endpoint computes
    (`decision._decide_scaling`, unmodified) -- current_servers is a
    logging-only "shadow" count carried forward via
    `ShadowStore.get_last_recommended_servers`/`record_observed_decision`,
    never a real fleet size. Logs the result and returns it as a dict, or
    returns None (logging a warning, not raising) if there wasn't enough
    real data this tick to fit ARIMA at all -- a real scrape gap or a
    brand-new node are both legitimate reasons, same convention
    `MetricsSource.fetch_readings` already established for an empty result.
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

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(flat.reshape(-1, 1)).flatten()

    forecast_scaled = _arima_forecast_once(scaled, config.HORIZON_STEPS, cfg.order)
    forecast_real = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

    planned_load = float(np.max(forecast_real)) * cfg.demand_scale * (1.0 + cfg.safety_margin)
    dec_cfg = DecisionConfig(under_prov_weight=cfg.under_prov_weight)
    current_servers = store.get_last_recommended_servers(machine_id, default=config.SIM_INITIAL_SERVERS)
    action, recommended_servers = _decide_scaling(current_servers, planned_load, dec_cfg)

    forecast_list = [round(float(v), 4) for v in forecast_real]
    store.record_observed_decision(
        machine_id, now, forecaster="arima",
        forecast_cpu_pct=forecast_list, planned_load_pct=round(planned_load, 4),
        current_servers=current_servers, recommended_servers=recommended_servers, action=action,
    )
    logger.info(
        "live_loop: observed machine=%s forecast=%s planned_load=%.2f%% "
        "current=%d recommended=%d action=%s",
        machine_id, forecast_list, planned_load, current_servers, recommended_servers, action,
    )
    return {
        "machine_id": machine_id, "forecast_cpu_pct": forecast_list,
        "planned_load_pct": round(planned_load, 4), "current_servers": current_servers,
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

    No scaling call anywhere in this function or anything it calls -- still
    logging only, per this stage's explicit boundary. A per-node failure
    (a Prometheus error, a missing hybrid model, anything else) is caught,
    logged, and does not stop the rest of the tick from running for other
    nodes.

    Returns a summary dict of machine_ids grouped by outcome, useful for
    tests and for a caller that wants to log/inspect a tick's shape without
    re-deriving it from individual log calls.
    """
    summary = {"observed": [], "skipped": [], "shadow_evaluated": [], "shadow_skipped": []}

    for machine_id in machine_ids:
        try:
            decision = observe_node_once(machine_id, source, store, now, cfg)
        except Exception:
            logger.exception("live_loop: ARIMA observation failed for machine=%s", machine_id)
            decision = None
        summary["observed" if decision is not None else "skipped"].append(machine_id)

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
