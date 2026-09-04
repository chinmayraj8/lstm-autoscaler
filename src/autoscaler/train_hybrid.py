"""
Step 28: train a REAL per-machine residual-hybrid model from real,
accumulated Prometheus history -- the piece every prior step (16, 22-26)
explicitly flagged as missing: "no pretrained residual-hybrid model
exists for any real node."

This deliberately reuses Step 16's already-audited residual-hybrid
mechanism unmodified:
  - arima_baseline.ARIMA_ORDER (the frozen (2,0,1) order used everywhere
    else in this project -- see arima_baseline.py's own module docstring
    for why Step 14's per-machine re-selection was rejected)
  - arima_baseline._arima_train_walkforward (honest walk-forward training
    residual labels from the deployed ARIMA's own fixed parameters -- no
    re-estimation leakage)
  - forecasting._build_lstm_model / _train_lstm (identical architecture
    and training procedure as every other LSTM in this project)

The only genuinely new logic in this module is *where the data comes
from*: real Prometheus history via a MetricsSource, instead of the
offline historical CSV experiments/run_hybrid_arima_lstm.py reads. This
mirrors live_loop._build_hybrid_window's existing real-data-sourcing
pattern for the live shadow-window path -- same idea, applied to training
instead of evaluation.

Output: {model_dir}/{machine_id}.keras, produced via model.save(path) (a
full architecture+weights save) -- the same convention the original
lstm_model.keras was built with. live_loop._load_hybrid_residual_model
loads it via model.load_weights(path) on a freshly-built architecture,
which works against a full-save .keras archive (it only reads the
weights back out), matching how main.py already loads the production
model.

NOT invoked by anything automatically. This is a deliberate,
human-triggered, offline action -- see scripts/train_real_hybrid_model.py
-- with the same posture as every other "actually change something real"
step in this project (Step 26's actuation is opt-in and explicit; this is
too). Training happens outside the live tick loop entirely; nothing here
runs on a schedule.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from . import config
from .arima_baseline import ARIMA_ORDER, _arima_train_walkforward
from .data import _make_sequences
from .metrics_source import MetricsSource, resample_readings

logger = logging.getLogger(__name__)

# Not just "enough points to avoid a crash" (that would be
# LOOKBACK_STEPS + HORIZON_STEPS, a handful of points) -- enough sequences
# for the residual LSTM to have something real to learn from. Deliberately
# conservative; see progress/2026-09-04_step28-real-hybrid-model.md for
# the reasoning (the offline historical dataset gave each machine roughly
# 1,300-2,300 points; this is a floor, not a target).
MIN_TRAIN_POINTS = config.LOOKBACK_STEPS + config.HORIZON_STEPS + 100


class InsufficientRealHistory(Exception):
    """Not enough real Prometheus history yet for a meaningful training
    run. Expected while the cluster is still accumulating real uptime --
    this is the normal state for days after Step 26/27, not a bug."""


@dataclass
class TrainingResult:
    machine_id: str
    model_path: str
    n_train_points: int
    n_sequences: int
    train_residual_std: float
    final_train_loss: float
    epochs_trained: int
    wall_clock_secs: float


def fetch_and_scale_training_series(machine_id: str, source: MetricsSource, now,
                                    fit_hours: float) -> tuple[np.ndarray, MinMaxScaler]:
    """Pull `fit_hours` of real history for `machine_id` ending at `now`,
    resample to the project's standard 5-minute cadence (matching
    `_build_hybrid_window`'s existing pattern), and scale it with a
    scaler fit on this data only -- the same train-only-fit convention
    used everywhere else in this project (see data.py's leakage-fix
    history). Raises InsufficientRealHistory rather than silently
    training on too little data.
    """
    start = now - timedelta(hours=fit_hours)
    raw = source.fetch_readings(machine_id, start, now)
    resampled = resample_readings(raw)
    flat = resampled[config.FEATURE_COL].values.astype(np.float64) if not resampled.empty else np.array([])

    if len(flat) < MIN_TRAIN_POINTS:
        raise InsufficientRealHistory(
            f"machine_id={machine_id!r}: only {len(flat)} real points over the last "
            f"{fit_hours}h (need >={MIN_TRAIN_POINTS} for a meaningful training run -- "
            "this is expected while the cluster is still accumulating real uptime)"
        )

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_flat_scaled = scaler.fit_transform(flat.reshape(-1, 1)).flatten()
    return train_flat_scaled, scaler


def train_residual_hybrid_model(machine_id: str, source: MetricsSource, now, model_dir: str,
                                fit_hours: float, order=ARIMA_ORDER,
                                seed: Optional[int] = 42) -> TrainingResult:
    """The real training run: real Prometheus history -> a real fitted
    ARIMA -> real walk-forward training residuals -> a real trained LSTM
    -> a real .keras file at {model_dir}/{machine_id}.keras.

    Deliberately mirrors experiments/run_hybrid_arima_lstm.py's
    `_train_residual_lstm` step for step, since that offline pipeline is
    already thoroughly audited (Step 16) -- this function's only real
    departure is where `train_flat_scaled` comes from.
    """
    # Fetched and length-checked BEFORE the TensorFlow import below, on
    # purpose: InsufficientRealHistory is the expected, routine outcome in
    # the days right after Step 26/27 while the cluster is still
    # accumulating real uptime, and this project's whole TensorFlow-isolation
    # convention (see forecasting.py's module docstring) exists so that
    # outcome doesn't require a TensorFlow install to observe -- callers
    # (and this module's own tests) can hit this path with plain
    # numpy/pandas/statsmodels only.
    train_flat_scaled, scaler = fetch_and_scale_training_series(machine_id, source, now, fit_hours)

    import tensorflow as tf  # lazy -- see forecasting.py's isolation convention
    from statsmodels.tsa.arima.model import ARIMA

    from .forecasting import _build_lstm_model, _train_lstm

    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_full = ARIMA(train_flat_scaled, order=order).fit()

    y_pred_sc, y_true_sc = _arima_train_walkforward(
        fit_full, train_flat_scaled, config.LOOKBACK_STEPS, config.HORIZON_STEPS
    )
    resid_train_sc = (y_true_sc - y_pred_sc).astype(np.float32)

    X_train, _ = _make_sequences(train_flat_scaled.reshape(-1, 1), config.LOOKBACK_STEPS, config.HORIZON_STEPS)
    if len(X_train) != len(resid_train_sc):
        # _arima_train_walkforward and _make_sequences are both aligned to
        # the same (lookback, horizon) windowing over the same array by
        # construction -- a mismatch here means a real bug, not bad luck,
        # so this fails loudly rather than silently truncating.
        raise AssertionError(
            f"machine_id={machine_id!r}: X/residual length mismatch "
            f"({len(X_train)} vs {len(resid_train_sc)}) -- this should be impossible"
        )

    model = _build_lstm_model(config.LOOKBACK_STEPS, config.HORIZON_STEPS)
    os.makedirs(model_dir, exist_ok=True)
    scratch_checkpoint = os.path.join(model_dir, f"_scratch_{machine_id}.keras")
    history, wall_clock_secs = _train_lstm(model, X_train, resid_train_sc, scratch_checkpoint)
    # _train_lstm deletes its own checkpoint after restore_best_weights=True
    # already synced the best weights into `model` in memory (see that
    # function's own body) -- this module's actual deliverable is written
    # explicitly, next.

    final_path = os.path.join(model_dir, f"{machine_id}.keras")
    model.save(final_path)  # full save -- same convention lstm_model.keras was built with
    logger.info(
        "train_hybrid: trained and saved machine_id=%s -> %s (%d train points, %d sequences, %d epochs, %.1fs)",
        machine_id, final_path, len(train_flat_scaled), len(X_train),
        len(history.history["loss"]), wall_clock_secs,
    )

    return TrainingResult(
        machine_id=machine_id,
        model_path=final_path,
        n_train_points=len(train_flat_scaled),
        n_sequences=len(X_train),
        train_residual_std=float(resid_train_sc.std()),
        final_train_loss=float(history.history["loss"][-1]),
        epochs_trained=len(history.history["loss"]),
        wall_clock_secs=wall_clock_secs,
    )
