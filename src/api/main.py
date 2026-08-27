"""
FastAPI autoscaler inference service.

Endpoints
---------
GET  /health     — liveness + model-info check
POST /forecast   — given the last 30 minutes of CPU utilisation (6 readings
                   at 5-min intervals), returns the 15-minute-ahead forecast
                   and a recommended server scaling action.

Run from the project root:
    uvicorn src.api.main:app --reload

The service loads lstm_model.keras from the project root and fits the
MinMaxScaler used for normalisation from m_1933's training split at startup,
matching the split used throughout the experiment pipeline.
"""

from __future__ import annotations

import sys
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure the project root is on sys.path so experiments.pipeline is importable
# whether this module is run as `uvicorn src.api.main:app` from the root or
# from any other working directory.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.pipeline import (  # noqa: E402
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
)

# ── Module-level state (populated at startup, read during requests) ───────────
_state: dict = {}

_MODEL_PATH = _ROOT / "lstm_model.keras"


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
