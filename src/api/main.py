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

import os
import sys
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure the project root is on sys.path so experiments.pipeline is importable
# whether this module is run as `uvicorn src.api.main:app` from the root or
# from any other working directory.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.autoscaler import (  # noqa: E402
    DEMAND_SCALE,
    DEC_OVER_WEIGHT,
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
from src.autoscaler.shadow_store import ShadowStore, record_and_evaluate  # noqa: E402

# ── Module-level state (populated at startup, read during requests) ───────────
_state: dict = {}

_MODEL_PATH = _ROOT / "lstm_model.keras"

# Step 19: durable shadow-evaluation state (src/autoscaler/shadow_store.py),
# a SQLite file at the project root. Overridable via env var so tests don't
# touch the real file (see tests/test_api_shadow.py).
_SHADOW_DB_PATH = os.environ.get("LSTM_AUTOSCALER_SHADOW_DB", str(_ROOT / "shadow_state.db"))
_shadow_store = ShadowStore(_SHADOW_DB_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Keras model and fit the scaler on startup."""
    warnings.filterwarnings("ignore")
    import tensorflow as tf  # deferred — keeps import time fast when testing

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

    _state.update(
        model=model,
        scaler=scaler,
        machine_id=machine_id,
        ts_len=len(ts),
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    yield

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
    machine_id: str
    model_path: str
    lookback_steps: int
    lookback_minutes: int
    horizon_steps: int
    horizon_minutes: int
    started_at: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness check. Returns 503 if the model has not finished loading."""
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not yet loaded.")
    return HealthResponse(
        status="ok",
        machine_id=_state["machine_id"],
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
