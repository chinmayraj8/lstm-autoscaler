"""
LSTM model construction/training and forecast evaluation (LSTM + naive
persistence). This is the only module in the package that imports
TensorFlow — kept isolated so the rest of the package (decision engine,
simulator, calibration, data prep) can be imported and tested without a
TensorFlow install at all.

Logic copied unchanged from experiments/pipeline.py.
"""

import os
import time

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from . import config


def _build_lstm_model(lookback: int, horizon: int, n_features: int = 1) -> tf.keras.Model:
    """`n_features` (Step 14): 1 for the original univariate CPU% input,
    >1 for the multivariate input (data._prepare_multivariate) that adds
    time-of-day/day-of-week + rolling mean/std channels. Everything else
    about the architecture -- units, layers, dropout, optimizer -- is
    unchanged, so this is a clean test of additional input signal, not a
    confound of its own."""
    model = Sequential([
        LSTM(config.LSTM_UNITS, return_sequences=True, input_shape=(lookback, n_features)),
        Dropout(config.DROPOUT_RATE),
        LSTM(config.LSTM_UNITS, return_sequences=False),
        Dropout(config.DROPOUT_RATE),
        Dense(horizon),
    ])
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss="mse")
    return model


def _train_lstm(lstm_model, X_train, y_train, model_path: str):
    early_stop = EarlyStopping(
        monitor="val_loss", patience=config.ES_PATIENCE, restore_best_weights=True, verbose=0
    )
    checkpoint = ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=0)
    t0 = time.time()
    history = lstm_model.fit(
        X_train, y_train,
        epochs=config.MAX_EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VAL_SPLIT,
        callbacks=[early_stop, checkpoint],
        verbose=0,
    )
    elapsed = time.time() - t0
    if os.path.exists(model_path):
        os.remove(model_path)
    return history, elapsed


def _inv(arr: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    out = np.zeros_like(arr)
    for h in range(arr.shape[1]):
        out[:, h] = scaler.inverse_transform(arr[:, h].reshape(-1, 1)).flatten()
    return out


def _evaluate_lstm(model, X, y, scaler):
    y_pred_scaled = model.predict(X, verbose=0)
    y_pred_real   = _inv(y_pred_scaled, scaler)
    y_real        = _inv(y, scaler)
    rmse = float(np.sqrt(mean_squared_error(y_real.flatten(), y_pred_real.flatten())))
    mae  = float(mean_absolute_error(y_real.flatten(), y_pred_real.flatten()))
    return y_pred_real, y_real, rmse, mae


def _evaluate_naive(X, y, scaler, horizon):
    last_val       = X[:, -1, 0]
    y_naive_scaled = np.repeat(last_val[:, None], horizon, axis=1)
    y_naive_real   = _inv(y_naive_scaled, scaler)
    y_real         = _inv(y, scaler)
    rmse = float(np.sqrt(mean_squared_error(y_real.flatten(), y_naive_real.flatten())))
    mae  = float(mean_absolute_error(y_real.flatten(), y_naive_real.flatten()))
    return rmse, mae
