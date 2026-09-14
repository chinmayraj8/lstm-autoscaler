"""
FastAPI autoscaler inference service.

Endpoints
---------
GET  /health              — liveness + model-info check
POST /forecast             — given the last 30 minutes of CPU utilisation (6
                             readings at 5-min intervals), returns the
                             15-minute-ahead forecast and a recommended
                             server scaling action.
POST /shadow/{machine_id}/window — Step 18: submit one already-observed
                             shadow window's real demand plus BOTH ARIMA's
                             and the residual hybrid's forecasts for it;
                             scores both through the existing decision
                             engine/simulator (src/autoscaler/shadow.py,
                             which reuses decision.py/simulation.py
                             unmodified), banks the result, and re-runs the
                             assignment decision. Neither forecaster
                             controls the real fleet through this endpoint
                             — this is pure shadow measurement. This
                             endpoint does NOT run ARIMA or the hybrid LSTM
                             itself; the caller (an offline/scheduled job
                             with access to the real fleet's telemetry and
                             both forecasting pipelines) submits each
                             window's forecasts already computed. See
                             shadow.py's module docstring for exactly what
                             the gating logic is and is not (a repeat-window
                             statistical gate, not a bandit).
GET  /shadow/{machine_id}  — current shadow-evaluation state for one
                             machine: which forecaster it's assigned to,
                             how many shadow windows are banked, cumulative
                             cost/SLA, and the full assignment-change
                             history.
GET  /shadow/{machine_id}/windows — the full per-window audit log: every
                             shadow window ever banked, each with both
                             forecasters' cost/SLA for that specific
                             window.
GET  /shadow/{machine_id}/decisions — Step 22: the live forecasting loop's
                             per-tick observed-decision log for one node --
                             every tick's forecast + would-be scaling
                             action, logged for every tracked node whether
                             or not it has an active hybrid comparison
                             running. Populated only once the live loop is
                             actually running (see below).

Step 22 (Stage 4): if `LSTM_AUTOSCALER_PROMETHEUS_URL` is set at startup,
a background thread polls that in-cluster Prometheus URL on a schedule
(`src/autoscaler/live_loop.py`) and logs both per-tick ARIMA decisions
(`/shadow/{machine_id}/decisions`) and, for nodes already assigned to the
residual hybrid, live shadow-window comparisons
(`/shadow/{machine_id}/windows`) through the SAME `_shadow_store` these
endpoints read. Unset (the default), no background thread starts. Set
`LSTM_AUTOSCALER_SKIP_LSTM_MODEL` to start this service without the LSTM
model/scaler (which needs the local Kaggle CSV, not present in a minimal
deployment) -- `/forecast` then returns 503; every `/shadow/*` endpoint and
the live loop are unaffected, since neither ever touches that model.
**No scaling call to real infrastructure exists anywhere in this service,
including in the live loop -- it only ever reads Prometheus and writes to
`_shadow_store`.**

Step 25: `LSTM_AUTOSCALER_SHADOW_FIT_HOURS`/`LSTM_AUTOSCALER_SHADOW_WINDOW_HOURS`
override `LiveLoopConfig`'s `shadow_fit_hours`/`shadow_window_hours`
(defaults 6/24 -- the real 30h-of-history methodology), the same way
`LSTM_AUTOSCALER_TICK_SECONDS` already overrode `tick_seconds`. Unset by
default; only ever meant to be set small for a deliberately-flagged demo
run (see progress/2026-09-03_step25-hybrid-path-mechanical-demo.md), never
left small in a real deployment. `LSTM_AUTOSCALER_HYBRID_MODEL_DIR`
(read directly by `live_loop.py`, not by this module) points
`_load_hybrid_residual_model` at a directory of per-machine `.keras`
residual models -- defaults to `models/hybrid_residual`, a path that has
never contained a real trained model (see Step 22's and Step 25's progress
docs).

Run from the project root:
    uvicorn src.api.main:app --reload

The service loads lstm_model.keras from the project root and fits the
MinMaxScaler used for normalisation from m_1933's training split at startup,
matching the split used throughout the experiment pipeline. The /shadow
endpoints are independent of that model/scaler state — they only need
already-computed forecast arrays. State is durable (Step 19): a SQLite
file (shadow_state.db, project root by default) via
src/autoscaler/shadow_store.py, which wraps shadow.py's unmodified
decision logic with load-before/save-after persistence -- shadow.py
itself has no notion of a database.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests as _requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.preprocessing import MinMaxScaler

# Step 25: without a configured handler, the `autoscaler.*` module loggers
# (shadow.py, live_loop.py -- every logger.info/.warning/.exception call
# they make, including the live loop's per-tick "observed" line and any
# exception it catches) are silently dropped: Python's logging module only
# auto-prints WARNING+ via a bare last-resort handler, and even that never
# fires for the module loggers here specifically once uvicorn's own logging
# setup runs (it configures the root logger, at which point the "no handler
# configured anywhere" condition no longer holds, but the *level* uvicorn
# sets doesn't include this project's INFO-level operational logs). This
# was invisible until Step 25 needed to debug the live loop's actual
# per-tick behavior on the real cluster via `kubectl logs` and found nothing
# -- a real, previously-unnoticed observability gap in a project whose
# whole point since Step 18 has been "watch it running," not just trust it.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Ensure the project root is on sys.path so experiments.pipeline is importable
# whether this module is run as `uvicorn src.api.main:app` from the root or
# from any other working directory.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.autoscaler import (  # noqa: E402
    DEMAND_SCALE,
    DEC_MAX_SERVERS,
    DEC_MIN_SERVERS,
    DEC_OVER_WEIGHT,
    DEC_SCALE_STEP,
    DEC_SERVER_CAPACITY,
    DEC_UNDER_WEIGHT,
    FEATURE_COL,
    HORIZON_STEPS,
    LOOKBACK_STEPS,
    SAFETY_MARGIN,
    DecisionConfig,
    _build_lstm_model,
    _decide_scaling,
    _load_and_prepare,
    _split_three_way,
    shadow,
)
from src.autoscaler.actuator import ActuationError, get_current_replicas  # noqa: E402
from src.autoscaler.arima_baseline import ARIMA_ORDER, _arima_forecast_with_ci  # noqa: E402
from src.autoscaler.live_loop import DEFAULT_FIT_LOOKBACK_HOURS, MIN_FIT_POINTS  # noqa: E402
from src.autoscaler.metrics_source import (  # noqa: E402
    DEFAULT_PROMETHEUS_URL,
    PrometheusMetricsSource,
    PrometheusMetricsSourceError,
    resample_readings,
)
from src.autoscaler.shadow_store import ShadowStore, record_and_evaluate  # noqa: E402

# ── Module-level state (populated at startup, read during requests) ───────────
_state: dict = {}

_MODEL_PATH = _ROOT / "lstm_model.keras"

# Step 19: durable shadow-evaluation state (src/autoscaler/shadow_store.py),
# a SQLite file at the project root. Overridable via env var so tests don't
# touch the real file (see tests/test_api_shadow.py).
_SHADOW_DB_PATH = os.environ.get("LSTM_AUTOSCALER_SHADOW_DB", str(_ROOT / "shadow_state.db"))
_shadow_store = ShadowStore(_SHADOW_DB_PATH)

# Step 22 (Stage 4): set to skip loading the LSTM model + fitting its scaler
# from the local Kaggle CSV at startup. The /shadow/* endpoints (and the live
# forecasting loop below) never touch either -- only /forecast does. A
# minimal observer deployment (Stage 5) has neither the CSV nor a reason to
# pay TensorFlow's model-load cost, so it sets this. Unset (the default),
# nothing about this service's existing behavior changes.
_SKIP_LSTM_MODEL = os.environ.get("LSTM_AUTOSCALER_SKIP_LSTM_MODEL", "").strip().lower() in {"1", "true", "yes"}

# Step 22 (Stage 4): if set, the live forecasting loop starts as a background
# thread at startup, polling this in-cluster Prometheus URL on a schedule
# (src/autoscaler/live_loop.py) and logging observed decisions / shadow
# windows through the same _shadow_store the /shadow/* endpoints read. Unset
# (the default, e.g. in tests and local dev without a real cluster), no
# background thread starts and nothing about this service's existing
# behavior changes. See live_loop.py's module docstring for exactly what
# this loop does and does not do -- it never calls a real scaling API.
_PROMETHEUS_URL = os.environ.get("LSTM_AUTOSCALER_PROMETHEUS_URL", "").strip() or None
_live_loop_stop_event = threading.Event()
_live_loop_thread: Optional[threading.Thread] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Keras model and fit the scaler on startup (unless
    LSTM_AUTOSCALER_SKIP_LSTM_MODEL is set), then optionally start the live
    forecasting loop (if LSTM_AUTOSCALER_PROMETHEUS_URL is set)."""
    warnings.filterwarnings("ignore")

    if _SKIP_LSTM_MODEL:
        print("[startup] LSTM_AUTOSCALER_SKIP_LSTM_MODEL set — skipping LSTM model/scaler load. "
              "/forecast will return 503; /shadow/* is unaffected.")
    else:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {_MODEL_PATH}\n"
                "Re-run the notebook to regenerate lstm_model.keras."
            )

        # Build architecture fresh (avoids Keras config-version deserialization errors)
        # then load weights from the saved file.
        print(f"[startup] Building model architecture and loading weights from {_MODEL_PATH} …")
        model = _build_lstm_model(LOOKBACK_STEPS, HORIZON_STEPS)
        model.load_weights(str(_MODEL_PATH))
        print(f"[startup] Model ready  "
              f"input={model.input_shape}  output={model.output_shape}")

        print("[startup] Fitting scaler on m_1933 training split …", flush=True)
        ts, machine_id = _load_and_prepare()          # picks the most-sampled machine
        _, _, _, scaler = _split_three_way(ts, FEATURE_COL)
        print(f"[startup] Scaler ready  machine={machine_id}  "
              f"CPU% range [{scaler.data_min_[0]:.3f}, {scaler.data_max_[0]:.3f}]")

        _state.update(model=model, scaler=scaler, machine_id=machine_id, ts_len=len(ts))

    _state["started_at"] = datetime.now(timezone.utc).isoformat()

    global _live_loop_thread
    live_loop_thread = None
    if _PROMETHEUS_URL:
        from src.autoscaler.live_loop import LiveLoopConfig, resolve_tracked_machine_ids, run_scheduler_loop
        from src.autoscaler.metrics_source import PrometheusMetricsSource

        prom_source = PrometheusMetricsSource(prometheus_url=_PROMETHEUS_URL)
        _default_cfg = LiveLoopConfig()
        tick_seconds = int(os.environ.get("LSTM_AUTOSCALER_TICK_SECONDS", _default_cfg.tick_seconds))
        # Step 25: shadow_fit_hours/shadow_window_hours are overridable the
        # same way tick_seconds already was. Production has no reason to set
        # either (the 6h/24h code defaults are the real methodology) -- these
        # exist so a deliberate, explicitly-flagged demo run (see
        # progress/2026-09-03_step25-*.md) can shrink them to minutes without
        # editing code, not so they get left small by default.
        shadow_fit_hours = float(os.environ.get("LSTM_AUTOSCALER_SHADOW_FIT_HOURS", _default_cfg.shadow_fit_hours))
        shadow_window_hours = float(
            os.environ.get("LSTM_AUTOSCALER_SHADOW_WINDOW_HOURS", _default_cfg.shadow_window_hours)
        )
        # Step 26: real actuation, off by default. LSTM_AUTOSCALER_ENABLE_ACTUATION
        # must be explicitly set truthy -- an unset or empty value (the default
        # in every deployment through Step 25) leaves this byte-for-byte the
        # observe-only behavior this project has had all along. See
        # actuator.py and LiveLoopConfig's own comments for the RBAC/scope
        # this assumes, and progress/2026-09-*-step26-*.md for why one named
        # machine, not an aggregate, is what drives it.
        enable_actuation = os.environ.get("LSTM_AUTOSCALER_ENABLE_ACTUATION", "").strip().lower() in {"1", "true", "yes"}
        actuation_machine_id = os.environ.get("LSTM_AUTOSCALER_ACTUATION_MACHINE_ID", "").strip() or None
        if enable_actuation and not actuation_machine_id:
            raise ValueError(
                "LSTM_AUTOSCALER_ENABLE_ACTUATION is set but "
                "LSTM_AUTOSCALER_ACTUATION_MACHINE_ID is not -- refusing to start "
                "with actuation enabled and no machine designated, rather than "
                "silently actuating nothing or guessing which node."
            )
        loop_cfg = LiveLoopConfig(
            tick_seconds=tick_seconds,
            shadow_fit_hours=shadow_fit_hours,
            shadow_window_hours=shadow_window_hours,
            enable_actuation=enable_actuation,
            actuation_machine_id=actuation_machine_id,
        )

        def _get_tracked_ids():
            return resolve_tracked_machine_ids(prom_source)

        _live_loop_stop_event.clear()
        live_loop_thread = threading.Thread(
            target=run_scheduler_loop,
            args=(_shadow_store, prom_source, _get_tracked_ids, loop_cfg),
            kwargs={"stop_event": _live_loop_stop_event},
            daemon=True, name="live-forecasting-loop",
        )
        live_loop_thread.start()
        # Kept as the SAME object `run_scheduler_loop` mutates every tick
        # (see LiveLoopConfig's own docstring) -- reading its two
        # leading-underscore runtime-state fields from a request handler
        # (see /actuation/status below) reflects the live circuit-breaker
        # state, not a snapshot taken at startup.
        _state["loop_cfg"] = loop_cfg
        if enable_actuation:
            print(f"[startup] Live forecasting loop started against {_PROMETHEUS_URL} "
                  f"(tick={tick_seconds}s). ACTUATION ENABLED for machine="
                  f"{actuation_machine_id!r} -> deployment={loop_cfg.actuation_deployment!r} "
                  f"namespace={loop_cfg.actuation_namespace!r}. Every other tracked node "
                  f"remains observe-only.")
        else:
            print(f"[startup] Live forecasting loop started against {_PROMETHEUS_URL} "
                  f"(tick={tick_seconds}s). Observe-only — no scaling call anywhere in this path.")

    _live_loop_thread = live_loop_thread

    yield

    if live_loop_thread is not None:
        _live_loop_stop_event.set()
        live_loop_thread.join(timeout=5)
    _live_loop_thread = None
    _state.clear()
    print("[shutdown] State cleared.")


app = FastAPI(
    title="LSTM Autoscaler API",
    description=(
        "Load-forecasting and scaling-decision service built on a 2-layer LSTM "
        "trained on Alibaba cluster-trace CPU data."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# The React frontend (Vite dev server, or a static build served from a
# different origin) calls this API cross-origin. No cookie/session auth
# exists anywhere in this service to protect (every route above is
# unauthenticated read/write already), so a wildcard origin doesn't widen
# this service's real attack surface -- it only stops the browser from
# blocking a request `curl`/`requests` could already make freely.
# LSTM_AUTOSCALER_CORS_ORIGINS overrides with a comma-separated allowlist
# for a deployment that wants one.
_cors_origins_env = os.environ.get("LSTM_AUTOSCALER_CORS_ORIGINS", "").strip()
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response schemas ────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    cpu_pct: List[float] = Field(
        ...,
        min_length=LOOKBACK_STEPS,
        max_length=LOOKBACK_STEPS,
        description=(
            f"Exactly {LOOKBACK_STEPS} CPU-utilisation readings "
            f"(one per 5-minute interval = last {LOOKBACK_STEPS * 5} min), "
            "in raw CPU-percent units (e.g. 5.2 means 5.2 %)."
        ),
    )
    current_servers: int = Field(
        2, ge=1, le=10,
        description="Number of servers currently active.",
    )
    demand_scale: float = Field(
        DEMAND_SCALE, gt=0,
        description=(
            "Multiplier converting CPU% to aggregate-demand%. "
            f"Default {DEMAND_SCALE} is calibrated for m_1933 (~5.7 % mean CPU → 114 % load)."
        ),
    )
    safety_margin: float = Field(
        SAFETY_MARGIN, ge=0.0, le=1.0,
        description="Fractional headroom added to the peak forecast before the scaling decision.",
    )
    under_prov_weight: float = Field(
        DEC_UNDER_WEIGHT, gt=0,
        description="Under-provisioning penalty weight in the cost function.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "cpu_pct": [4.8, 5.1, 5.3, 4.9, 5.0, 5.2],
                "current_servers": 2,
            }
        }
    }


class ForecastResponse(BaseModel):
    forecast_cpu_pct: List[float] = Field(
        description=(
            f"Predicted CPU% at t+5, t+10, t+15 minutes "
            f"({HORIZON_STEPS} horizon steps, each 5 min)."
        )
    )
    planned_load_pct: float = Field(
        description=(
            "peak_forecast_cpu × demand_scale × (1 + safety_margin), "
            "in aggregate-demand%. Used as input to the scaling decision."
        )
    )
    current_servers: int
    recommended_servers: int
    action: str = Field(
        description='"hold", "scale_up +N", or "scale_down -N".'
    )
    utc_timestamp: str


class HealthResponse(BaseModel):
    status: str
    lstm_model_loaded: bool = Field(
        description="False if LSTM_AUTOSCALER_SKIP_LSTM_MODEL was set at startup — "
                    "/forecast returns 503 in that case; /shadow/* is unaffected."
    )
    live_loop_running: bool = Field(
        description="True if LSTM_AUTOSCALER_PROMETHEUS_URL was set at startup and the "
                    "background live forecasting loop thread is alive."
    )
    machine_id: Optional[str] = None
    model_path: str
    lookback_steps: int
    lookback_minutes: int
    horizon_steps: int
    horizon_minutes: int
    started_at: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness check. Returns 503 only if startup hasn't finished at all
    (e.g. still loading, or a real startup failure). Does NOT require the
    LSTM model specifically — `LSTM_AUTOSCALER_SKIP_LSTM_MODEL` (Step 22)
    lets a minimal observer deployment start without it; check
    `lstm_model_loaded` for that, not the overall status."""
    if "started_at" not in _state:
        raise HTTPException(status_code=503, detail="Still starting up.")
    return HealthResponse(
        status="ok",
        lstm_model_loaded="model" in _state,
        live_loop_running=_live_loop_thread is not None and _live_loop_thread.is_alive(),
        machine_id=_state.get("machine_id"),
        model_path=str(_MODEL_PATH),
        lookback_steps=LOOKBACK_STEPS,
        lookback_minutes=LOOKBACK_STEPS * 5,
        horizon_steps=HORIZON_STEPS,
        horizon_minutes=HORIZON_STEPS * 5,
        started_at=_state["started_at"],
    )


@app.post("/forecast", response_model=ForecastResponse, tags=["inference"])
def forecast(req: ForecastRequest) -> ForecastResponse:
    """
    Given the last 30 minutes of CPU utilisation, return:
    - 15-minute-ahead forecast (3 values, one per 5-min step)
    - recommended scaling action using the same decision engine as the experiment pipeline
    """
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not yet loaded.")

    model  = _state["model"]
    scaler = _state["scaler"]

    # ── Scale input ───────────────────────────────────────────────────────────
    cpu_arr   = np.array(req.cpu_pct, dtype=np.float32).reshape(-1, 1)
    scaled_in = scaler.transform(cpu_arr)               # (LOOKBACK_STEPS, 1)
    X         = scaled_in.reshape(1, LOOKBACK_STEPS, 1) # (1, LOOKBACK_STEPS, 1)

    # ── Forward pass ──────────────────────────────────────────────────────────
    y_pred_scaled = model.predict(X, verbose=0)          # (1, HORIZON_STEPS)

    # Inverse-transform each horizon step independently (matches pipeline._inv)
    forecast_cpu = scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)                     # (HORIZON_STEPS, 1)
    ).flatten().tolist()

    # ── Scaling decision (mirrors _build_lstm_targets in pipeline.py) ─────────
    # planned_load = peak_forecast × demand_scale × (1 + safety_margin)
    planned_load = float(np.max(forecast_cpu)) * req.demand_scale * (1.0 + req.safety_margin)
    dec_cfg = DecisionConfig(
        under_prov_weight=req.under_prov_weight,
        over_prov_weight=DEC_OVER_WEIGHT,
    )
    action, recommended_servers = _decide_scaling(req.current_servers, planned_load, dec_cfg)

    return ForecastResponse(
        forecast_cpu_pct=[round(v, 4) for v in forecast_cpu],
        planned_load_pct=round(planned_load, 2),
        current_servers=req.current_servers,
        recommended_servers=recommended_servers,
        action=action,
        utc_timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ── Shadow-mode evaluation (Step 18) ────────────────────────────────────────
# See src/autoscaler/shadow.py's module docstring for the full design and
# for what this deliberately is NOT (a repeat-window statistical gate, not
# a continuously-exploring bandit). These endpoints are thin HTTP wiring
# around that module -- no scoring/gating logic lives here.

class ShadowWindowRequest(BaseModel):
    window_start: datetime
    window_end: datetime
    demand_scale: float = Field(..., gt=0)
    y_actual: List[List[float]] = Field(
        ..., description="Actual demand for this window, shape (n_sequences, horizon_steps), real CPU% units."
    )
    arima_forecast: List[List[float]] = Field(
        ..., description="ARIMA's forecast for this window, same shape as y_actual."
    )
    arima_under_prov_weight: float = Field(DEC_UNDER_WEIGHT, gt=0)
    arima_safety_margin: float = Field(SAFETY_MARGIN, ge=0.0, le=1.0)
    hybrid_forecast: List[List[float]] = Field(
        ..., description="Residual-hybrid forecast for this window, same shape as y_actual."
    )
    hybrid_under_prov_weight: float = Field(DEC_UNDER_WEIGHT, gt=0)
    hybrid_safety_margin: float = Field(SAFETY_MARGIN, ge=0.0, le=1.0)


class ShadowWindowResponse(BaseModel):
    machine_id: str
    window_result: dict = Field(description="arima_cost/arima_sla_pct/hybrid_cost/hybrid_sla_pct for this window.")
    current_forecaster: str = Field(description='"arima" or "hybrid" -- the forecaster now controlling this machine.')
    assignment_changed: bool
    verdict: Optional[str] = Field(None, description="Set only when assignment_changed is true.")
    n_banked_windows: int
    cumulative: dict


class ShadowStatusResponse(BaseModel):
    machine_id: str
    current_forecaster: str
    last_evaluated_at: Optional[str]
    n_banked_windows: int
    cumulative: dict
    assignment_history: List[dict]


class ShadowWindowListResponse(BaseModel):
    machine_id: str
    windows: List[dict] = Field(
        description="Every shadow window ever banked for this machine, oldest first -- "
                    "window_start/window_end, arima_cost/arima_sla_pct, hybrid_cost/hybrid_sla_pct."
    )


class ObservedDecisionListResponse(BaseModel):
    machine_id: str
    decisions: List[dict] = Field(
        description="Individual ticks of the live forecasting loop (Step 22), oldest first: "
                    "observed_at, forecaster, forecast_cpu_pct, planned_load_pct, current_servers "
                    "(a logging-only running count, never a real fleet size), recommended_servers, "
                    "action. Logged for EVERY tracked node every tick, regardless of which "
                    "forecaster it's assigned to -- unlike /shadow/{machine_id}/windows, which only "
                    "has rows for nodes with an active ARIMA-vs-hybrid comparison. Never the result "
                    "of a real scaling call -- this endpoint only ever reflects what the live loop "
                    "observed and would have recommended."
    )


@app.post("/shadow/{machine_id}/window", response_model=ShadowWindowResponse, tags=["shadow"])
def submit_shadow_window(machine_id: str, req: ShadowWindowRequest) -> ShadowWindowResponse:
    """
    Score one already-observed shadow window's real demand against BOTH
    ARIMA's and the residual hybrid's forecasts for it, bank the result, and
    re-run the assignment decision. Reuses `shadow.run_shadow_window` (which
    itself reuses `decision._build_lstm_targets`, `simulation._run_simulation`,
    `simulation._compute_cost_score` unmodified) and
    `shadow.evaluate_and_maybe_reassign`. A machine only switches to the
    hybrid once it has won every one of its last `shadow.DEFAULT_MIN_WINDOWS`
    banked windows AND the aggregate comparison clears this project's
    combined-±1σ statistical bar (`shadow.decide_assignment`) -- consistency
    alone or significance alone is not enough. Neither forecaster's targets
    are ever applied to the real fleet through this endpoint.
    """
    y_actual = np.array(req.y_actual, dtype=np.float32)
    arima_pred = np.array(req.arima_forecast, dtype=np.float32)
    hybrid_pred = np.array(req.hybrid_forecast, dtype=np.float32)
    if not (y_actual.shape == arima_pred.shape == hybrid_pred.shape):
        raise HTTPException(
            status_code=400,
            detail="y_actual, arima_forecast, and hybrid_forecast must all share the same "
                   "(n_sequences, horizon_steps) shape.",
        )

    arima_dec_cfg = DecisionConfig(under_prov_weight=req.arima_under_prov_weight)
    hybrid_dec_cfg = DecisionConfig(under_prov_weight=req.hybrid_under_prov_weight)
    window = shadow.run_shadow_window(
        machine_id, req.window_start, req.window_end, y_actual,
        arima_pred, arima_dec_cfg, req.arima_safety_margin,
        hybrid_pred, hybrid_dec_cfg, req.hybrid_safety_margin,
        demand_scale=req.demand_scale,
    )
    # Step 19: durable via ShadowStore -- loads state, calls shadow.py's
    # unmodified record_shadow_window/evaluate_and_maybe_reassign, persists
    # exactly what changed. See shadow_store.record_and_evaluate.
    state, change = record_and_evaluate(_shadow_store, machine_id, window, datetime.now(timezone.utc))

    return ShadowWindowResponse(
        machine_id=machine_id,
        window_result={
            "arima_cost": window.arima_cost, "arima_sla_pct": window.arima_sla_pct,
            "hybrid_cost": window.hybrid_cost, "hybrid_sla_pct": window.hybrid_sla_pct,
        },
        current_forecaster=state.current_forecaster,
        assignment_changed=change is not None,
        verdict=change.verdict if change else None,
        n_banked_windows=len(state.window_results),
        cumulative=shadow.cumulative_summary(state.window_results),
    )


@app.get("/shadow/{machine_id}", response_model=ShadowStatusResponse, tags=["shadow"])
def shadow_status(machine_id: str) -> ShadowStatusResponse:
    """Current shadow-evaluation state for one machine: which forecaster
    controls it right now, how many shadow windows are banked, cumulative
    cost/SLA for each forecaster (over ALL windows ever recorded, not just
    the decision-relevant recent subset), and the full history of
    assignment changes (each with its timestamp and the evidence that
    triggered it). Backed by ShadowStore (Step 19) -- survives a restart."""
    if not _shadow_store.has_machine(machine_id):
        raise HTTPException(status_code=404, detail=f"No shadow state for machine_id={machine_id!r} yet.")
    state = _shadow_store.load_state(machine_id)
    full_history = _shadow_store.get_full_window_history(machine_id)
    return ShadowStatusResponse(
        machine_id=machine_id,
        current_forecaster=state.current_forecaster,
        last_evaluated_at=state.last_evaluated_at.isoformat() if state.last_evaluated_at else None,
        n_banked_windows=len(full_history),
        cumulative=shadow.cumulative_summary(full_history),
        assignment_history=[
            {
                "timestamp": c.timestamp.isoformat(),
                "old_forecaster": c.old_forecaster,
                "new_forecaster": c.new_forecaster,
                "verdict": c.verdict,
                "evidence": c.evidence,
            }
            for c in _shadow_store.get_assignment_history(machine_id)
        ],
    )


@app.get("/shadow/{machine_id}/windows", response_model=ShadowWindowListResponse, tags=["shadow"])
def shadow_windows(machine_id: str) -> ShadowWindowListResponse:
    """The full per-window audit log for one machine -- every shadow
    window ever banked, each with both forecasters' cost/SLA for that
    specific window (not just the cumulative aggregate `/shadow/{id}`
    returns). This is the "watch it working" view: each row is one
    already-made forecast + would-be scaling comparison, logged and never
    mutated after the fact."""
    if not _shadow_store.has_machine(machine_id):
        raise HTTPException(status_code=404, detail=f"No shadow state for machine_id={machine_id!r} yet.")
    windows = _shadow_store.get_full_window_history(machine_id)
    return ShadowWindowListResponse(
        machine_id=machine_id,
        windows=[
            {
                "window_start": w.window_start.isoformat(), "window_end": w.window_end.isoformat(),
                "arima_cost": w.arima_cost, "arima_sla_pct": w.arima_sla_pct,
                "hybrid_cost": w.hybrid_cost, "hybrid_sla_pct": w.hybrid_sla_pct,
            }
            for w in windows
        ],
    )


@app.get("/shadow/{machine_id}/decisions", response_model=ObservedDecisionListResponse, tags=["shadow"])
def shadow_decisions(machine_id: str, limit: int = 500) -> ObservedDecisionListResponse:
    """The live forecasting loop's per-tick observed-decision log (Step 22)
    for one node: every tick's forecast + would-be scaling action, whether
    or not that node has an active hybrid comparison running. This is the
    primary "watch real decisions accumulate" view -- populated only once
    the live loop is actually running against a real cluster
    (`LSTM_AUTOSCALER_PROMETHEUS_URL` set at startup; see `/health`'s
    `live_loop_running` field). Empty (not 404) for a tracked-but-not-yet-
    observed machine, or one this service has never heard of -- both are
    ordinary states for a log endpoint, unlike `/shadow/{machine_id}`'s
    404-if-unseen (which reflects the shadow-comparison gate having an
    opinion, not just whether logging happened)."""
    return ObservedDecisionListResponse(
        machine_id=machine_id,
        decisions=_shadow_store.get_observed_decisions(machine_id, limit=limit),
    )


# ── New: thin REST wrappers around existing in-process-only functions ────────
# (React frontend build) `actuator.get_current_replicas` and
# `PrometheusMetricsSource.fetch_readings`/`list_machine_ids` were, until
# now, only ever called in-process by src/dashboard/live_app.py -- no route
# exposed them to an out-of-process caller. Each endpoint below is a thin,
# direct wrapper around an already-existing, already-tested function; no
# new scoring/decision/actuation logic lives here.

class ReplicasResponse(BaseModel):
    deployment: str
    namespace: str
    replicas: int
    checked_at: str


@app.get("/replicas/{deployment}", response_model=ReplicasResponse, tags=["cluster"])
def replicas(deployment: str, namespace: str = "lstm-autoscaler") -> ReplicasResponse:
    """Current replica count of a real Deployment, read live via the local/
    in-cluster kubeconfig (`actuator.get_current_replicas`, unmodified --
    reads the `deployments/scale` subresource only). Never a scaling call."""
    try:
        count = get_current_replicas(deployment, namespace)
    except ActuationError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return ReplicasResponse(
        deployment=deployment, namespace=namespace, replicas=count,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )


class CpuReading(BaseModel):
    timestamp: str
    cpu_pct: float


class CpuMetricsResponse(BaseModel):
    machine_id: str
    prometheus_url: str
    resample_minutes: int
    readings: List[CpuReading]


@app.get("/metrics/cpu", response_model=CpuMetricsResponse, tags=["cluster"])
def metrics_cpu(
    machine_id: str,
    prometheus_url: str = _PROMETHEUS_URL or DEFAULT_PROMETHEUS_URL,
    hours: float = 2.0,
) -> CpuMetricsResponse:
    """Real per-node CPU% history for `machine_id` over the last `hours`,
    via `PrometheusMetricsSource.fetch_readings` + `resample_readings`
    (both unmodified -- the same two calls `live_app.py` made in-process).
    Empty `readings` (not an error) if the node has no data in range --
    matches `MetricsSource.fetch_readings`'s own documented contract."""
    try:
        source = PrometheusMetricsSource(prometheus_url=prometheus_url)
        now = datetime.now(timezone.utc)
        raw = source.fetch_readings(machine_id, now - timedelta(hours=hours), now)
    except (_requests.exceptions.RequestException, PrometheusMetricsSourceError) as e:
        raise HTTPException(status_code=502, detail=f"Prometheus query failed: {e}") from e
    df = resample_readings(raw)
    readings = (
        [
            CpuReading(timestamp=ts.isoformat(), cpu_pct=round(float(val), 4))
            for ts, val in df[FEATURE_COL].items()
        ]
        if not df.empty else []
    )
    return CpuMetricsResponse(
        machine_id=machine_id, prometheus_url=prometheus_url,
        resample_minutes=5, readings=readings,
    )


class MachinesResponse(BaseModel):
    prometheus_url: str
    machines: List[str]


@app.get("/machines", response_model=MachinesResponse, tags=["cluster"])
def machines(prometheus_url: str = _PROMETHEUS_URL or DEFAULT_PROMETHEUS_URL) -> MachinesResponse:
    """Every node currently visible to Prometheus right now
    (`PrometheusMetricsSource.list_machine_ids`, unmodified) -- lets the
    frontend populate a machine selector from the real cluster instead of
    a hardcoded list."""
    try:
        source = PrometheusMetricsSource(prometheus_url=prometheus_url)
        ids = source.list_machine_ids()
    except (_requests.exceptions.RequestException, PrometheusMetricsSourceError) as e:
        raise HTTPException(status_code=502, detail=f"Prometheus query failed: {e}") from e
    return MachinesResponse(prometheus_url=prometheus_url, machines=ids)


class ScalingConfigResponse(BaseModel):
    server_capacity_pct: float
    min_servers: int
    max_servers: int
    scale_step: int
    over_prov_weight: float
    under_prov_weight: float
    safety_margin: float
    demand_scale: float
    tick_seconds: int


@app.get("/config", response_model=ScalingConfigResponse, tags=["ops"])
def scaling_config() -> ScalingConfigResponse:
    """The effective scaling-decision thresholds this service is actually
    running with right now -- the real values `_decide_scaling` and the
    live loop use, not a hardcoded guess in the frontend. Reflects env-var
    overrides when the live loop is running; falls back to this module's
    code defaults otherwise (e.g. local dev with no Prometheus configured).
    None of these are runtime-editable via this API -- they're Python
    constants / startup env vars, exactly as before this endpoint existed."""
    loop_cfg = _state.get("loop_cfg")
    return ScalingConfigResponse(
        server_capacity_pct=DEC_SERVER_CAPACITY,
        min_servers=DEC_MIN_SERVERS,
        max_servers=DEC_MAX_SERVERS,
        scale_step=DEC_SCALE_STEP,
        over_prov_weight=DEC_OVER_WEIGHT,
        under_prov_weight=loop_cfg.under_prov_weight if loop_cfg else DEC_UNDER_WEIGHT,
        safety_margin=loop_cfg.safety_margin if loop_cfg else SAFETY_MARGIN,
        demand_scale=loop_cfg.demand_scale if loop_cfg else DEMAND_SCALE,
        tick_seconds=loop_cfg.tick_seconds if loop_cfg else 300,
    )


class ActuationStatusResponse(BaseModel):
    enabled: bool
    machine_id: Optional[str] = None
    deployment: Optional[str] = None
    namespace: Optional[str] = None
    circuit_breaker_threshold: Optional[int] = None
    circuit_breaker_cooldown_seconds: Optional[float] = None
    consecutive_failures: int = 0
    circuit_open: bool = False
    circuit_opened_at: Optional[str] = None
    cooldown_remaining_seconds: Optional[float] = None


@app.get("/actuation/status", response_model=ActuationStatusResponse, tags=["ops"])
def actuation_status() -> ActuationStatusResponse:
    """Live real-actuation config + circuit-breaker state (Step 26 / Step
    26 follow-up), read directly off the SAME `LiveLoopConfig` object
    `run_scheduler_loop` mutates every tick -- not a snapshot taken at
    startup. `enabled=False` (every other field at its default) if the
    live loop isn't running, or actuation was never turned on -- both
    ordinary, non-error states."""
    loop_cfg = _state.get("loop_cfg")
    if loop_cfg is None or not loop_cfg.enable_actuation:
        return ActuationStatusResponse(enabled=False)

    failures = loop_cfg._actuation_consecutive_failures
    opened_at = loop_cfg._actuation_circuit_opened_at
    circuit_open = False
    cooldown_remaining = None
    if opened_at is not None:
        elapsed = datetime.now(timezone.utc) - opened_at
        if elapsed < loop_cfg.actuation_circuit_breaker_cooldown:
            circuit_open = True
            cooldown_remaining = (loop_cfg.actuation_circuit_breaker_cooldown - elapsed).total_seconds()

    return ActuationStatusResponse(
        enabled=True,
        machine_id=loop_cfg.actuation_machine_id,
        deployment=loop_cfg.actuation_deployment,
        namespace=loop_cfg.actuation_namespace,
        circuit_breaker_threshold=loop_cfg.actuation_circuit_breaker_threshold,
        circuit_breaker_cooldown_seconds=loop_cfg.actuation_circuit_breaker_cooldown.total_seconds(),
        consecutive_failures=failures,
        circuit_open=circuit_open,
        circuit_opened_at=opened_at.isoformat() if opened_at else None,
        cooldown_remaining_seconds=cooldown_remaining,
    )


# ── Real confidence-interval forecast ────────────────────────────────────────
# Ground truth (backend audit, React frontend build): neither this service's
# /forecast (the LSTM path) nor the live loop's ARIMA observation
# (`arima_baseline._arima_forecast_once`) has ever computed a prediction
# interval anywhere -- only a point forecast. This endpoint closes that gap
# for ARIMA specifically, using statsmodels' own `get_forecast().conf_int()`
# (`arima_baseline._arima_forecast_with_ci`, new) -- a real, principled
# interval derived from the fitted model's own forecast-error variance, not
# a fabricated band. Computed fresh per request (same fit-and-forecast
# primitive `observe_node_once` uses each tick, same real Prometheus
# history) -- nothing here is persisted to `_shadow_store`, and it never
# touches the live loop's own ticking, decisions, or actuation.

class ForecastPoint(BaseModel):
    step_minutes: int
    cpu_pct: float
    lower_pct: float
    upper_pct: float


class ForecastConfidenceResponse(BaseModel):
    machine_id: str
    observed_at: str
    arima_order: List[int]
    confidence_level: float
    fit_window_hours: float
    fit_points: int
    forecast: List[ForecastPoint]


@app.get("/forecast/confidence", response_model=ForecastConfidenceResponse, tags=["inference"])
def forecast_confidence(
    machine_id: str,
    prometheus_url: str = _PROMETHEUS_URL or DEFAULT_PROMETHEUS_URL,
    fit_hours: float = DEFAULT_FIT_LOOKBACK_HOURS,
    confidence_level: float = 0.95,
) -> ForecastConfidenceResponse:
    """A real `confidence_level` (default 95%) prediction interval around
    ARIMA's `horizon_steps`-ahead point forecast for `machine_id`, fit
    fresh on the last `fit_hours` of real Prometheus history -- the same
    data source and fit procedure as the live loop's own per-tick ARIMA
    observation (`live_loop.observe_node_once`), just with
    `get_forecast().conf_int()` instead of `.forecast()`. Not persisted;
    calling this repeatedly does not affect `/shadow/*` state.

    422 if there isn't enough real history in the window to fit ARIMA at
    all (a real scrape gap or a brand-new node) -- an empty/degraded
    response would silently look like a valid, narrow interval, which
    would be worse than a clear error here.
    """
    if not 0.0 < confidence_level < 1.0:
        raise HTTPException(status_code=400, detail="confidence_level must be strictly between 0 and 1.")

    try:
        source = PrometheusMetricsSource(prometheus_url=prometheus_url)
        now = datetime.now(timezone.utc)
        raw = source.fetch_readings(machine_id, now - timedelta(hours=fit_hours), now)
    except (_requests.exceptions.RequestException, PrometheusMetricsSourceError) as e:
        raise HTTPException(status_code=502, detail=f"Prometheus query failed: {e}") from e

    resampled = resample_readings(raw)
    flat = resampled[FEATURE_COL].values.astype(np.float64)
    if len(flat) < MIN_FIT_POINTS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"only {len(flat)}/{MIN_FIT_POINTS} real points in the last {fit_hours:.1f}h for "
                f"machine_id={machine_id!r} -- not enough real history to fit ARIMA."
            ),
        )

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(flat.reshape(-1, 1)).flatten()

    alpha = 1.0 - confidence_level
    point_sc, lower_sc, upper_sc = _arima_forecast_with_ci(scaled, HORIZON_STEPS, ARIMA_ORDER, alpha)

    point = scaler.inverse_transform(point_sc.reshape(-1, 1)).flatten()
    lower = scaler.inverse_transform(lower_sc.reshape(-1, 1)).flatten()
    upper = scaler.inverse_transform(upper_sc.reshape(-1, 1)).flatten()

    return ForecastConfidenceResponse(
        machine_id=machine_id,
        observed_at=now.isoformat(),
        arima_order=list(ARIMA_ORDER),
        confidence_level=confidence_level,
        fit_window_hours=fit_hours,
        fit_points=len(flat),
        forecast=[
            ForecastPoint(
                step_minutes=(i + 1) * 5,
                cpu_pct=round(float(point[i]), 4),
                lower_pct=round(float(lower[i]), 4),
                upper_pct=round(float(upper[i]), 4),
            )
            for i in range(HORIZON_STEPS)
        ],
    )
