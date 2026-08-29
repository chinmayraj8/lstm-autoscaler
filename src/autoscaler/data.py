"""
Data loading and preprocessing: CSV -> one machine's resampled time series ->
chronological train/val/test split -> sliding-window sequences.

Logic copied unchanged from experiments/pipeline.py.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from . import config


def _pick_best_machine(df: pd.DataFrame) -> str:
    return df["machine_id"].value_counts().idxmax()


def _prepare_timeseries(df: pd.DataFrame, machine_id: str) -> pd.DataFrame:
    mdf = df[df["machine_id"] == machine_id].copy()
    mdf["time_stamp"] = pd.to_datetime(mdf["time_stamp"], unit="s")
    mdf = mdf.set_index("time_stamp").sort_index()
    mdf = mdf[["cpu_util_percent", "mem_util_percent"]]
    mdf = mdf.resample("5min").mean()
    mdf = mdf.ffill().dropna()
    return mdf


def _split_three_way(ts: pd.DataFrame, feature: str, val_ratio: float = config.VAL_RATIO,
                     test_ratio: float = config.TEST_RATIO):
    """Chronological 60/20/20 split. Scaler fits on train only."""
    values = ts[[feature]].values.astype(np.float32)
    n = len(values)
    train_end = int(n * (1.0 - val_ratio - test_ratio))
    val_end   = int(n * (1.0 - test_ratio))

    train_raw = values[:train_end]
    val_raw   = values[train_end:val_end]
    test_raw  = values[val_end:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_raw)
    return (
        scaler.transform(train_raw),
        scaler.transform(val_raw),
        scaler.transform(test_raw),
        scaler,
    )


def _make_sequences(data: np.ndarray, lookback: int, horizon: int):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback : i + lookback + horizon, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Multivariate features (Step 14) ────────────────────────────────────────
# Extra input channels the LSTM can use but a univariate ARIMA structurally
# can't: cyclically-encoded time-of-day/day-of-week (deterministic from the
# timestamp, no leakage risk by construction) and a short rolling mean/std
# of the target feature itself (uses only values already inside each row's
# own lookback window -- see _prepare_multivariate's docstring for why this
# isn't leakage). Column 0 is always the original target feature, so the
# existing `_evaluate_lstm`/`_inv` inverse-scaling path (which only ever
# handles the target channel) needs no changes.
MULTIVARIATE_ROLL_WINDOW = 6  # == LOOKBACK_STEPS; a "short" rolling window


def _prepare_multivariate(ts: pd.DataFrame, feature_col: str,
                          roll_window: int = MULTIVARIATE_ROLL_WINDOW) -> pd.DataFrame:
    """Build the multivariate feature frame from a machine's resampled time
    series. Returns a DataFrame with columns
    [feature_col, hour_sin, hour_cos, dow_sin, dow_cos, roll_mean, roll_std],
    same DatetimeIndex as `ts`.

    Leakage note: `roll_mean`/`roll_std` at row `t` are computed over
    `[t - roll_window + 1, t]` (pandas' default trailing, inclusive-of-current
    window) -- they never look past `t`. This does NOT leak information
    `_make_sequences`' lookback window doesn't already expose: X[i] already
    includes the raw feature values for every timestep up to and including
    the row immediately before the forecast horizon starts, so a rolling
    stat over that same span is just a different summary of data the model
    can already see, not new information from the future.
    """
    df = ts[[feature_col]].copy()
    idx = df.index
    hour_frac = idx.hour + idx.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24.0)
    dow = idx.dayofweek.values.astype(np.float64)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df["roll_mean"] = df[feature_col].rolling(roll_window, min_periods=1).mean()
    df["roll_std"] = df[feature_col].rolling(roll_window, min_periods=1).std().fillna(0.0)
    return df


# Channels already bounded in [-1, 1] by construction -- passed through
# unscaled rather than fit with a MinMaxScaler (fitting one on a feature
# that's already on a known, fixed scale would just be a no-op modulo the
# usual sin/cos range, and skipping it keeps the scaler list simpler).
_UNSCALED_MULTIVARIATE_COLS = {"hour_sin", "hour_cos", "dow_sin", "dow_cos"}


def _split_three_way_multivariate(df: pd.DataFrame, feature_col: str,
                                  val_ratio: float = config.VAL_RATIO,
                                  test_ratio: float = config.TEST_RATIO):
    """Multivariate analogue of `_split_three_way`. Every column is scaled
    independently, fit on TRAIN only (same convention as the univariate
    path) -- except the already-bounded cyclical columns, which pass
    through unscaled. `feature_col` (column 0) keeps its own MinMaxScaler,
    returned as `target_scaler` so the existing `_evaluate_lstm`/`_inv`
    inverse-transform path is unchanged: it only ever needs the target
    channel's scaler, and this scaler IS that scaler.

    Returns `(train_scaled, val_scaled, test_scaled, target_scaler)`, each
    array shape (n, n_features), column order == df.columns, column 0 ==
    feature_col.
    """
    values = df.values.astype(np.float32)
    n = len(values)
    train_end = int(n * (1.0 - val_ratio - test_ratio))
    val_end   = int(n * (1.0 - test_ratio))

    train_raw, val_raw, test_raw = values[:train_end], values[train_end:val_end], values[val_end:]
    train_scaled = np.zeros_like(train_raw)
    val_scaled   = np.zeros_like(val_raw)
    test_scaled  = np.zeros_like(test_raw)
    target_scaler = None

    for c, col in enumerate(df.columns):
        if col in _UNSCALED_MULTIVARIATE_COLS:
            train_scaled[:, c] = train_raw[:, c]
            val_scaled[:, c]   = val_raw[:, c]
            test_scaled[:, c]  = test_raw[:, c]
            continue
        sc = MinMaxScaler(feature_range=(0, 1))
        sc.fit(train_raw[:, c:c + 1])
        train_scaled[:, c] = sc.transform(train_raw[:, c:c + 1]).flatten()
        val_scaled[:, c]   = sc.transform(val_raw[:, c:c + 1]).flatten()
        test_scaled[:, c]  = sc.transform(test_raw[:, c:c + 1]).flatten()
        if col == feature_col:
            target_scaler = sc

    return train_scaled, val_scaled, test_scaled, target_scaler


def _make_multivariate_sequences(data: np.ndarray, lookback: int, horizon: int, target_col: int = 0):
    """Multivariate analogue of `_make_sequences`. X gets every channel
    (shape (n_sequences, lookback, n_features)); y stays single-channel,
    drawn only from `target_col` (shape (n_sequences, horizon)) -- unchanged
    from the univariate path's y shape, so `_evaluate_lstm`/`_inv` work
    without modification."""
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : i + lookback, :])
        y.append(data[i + lookback : i + lookback + horizon, target_col])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _load_and_prepare(machine_id=None, nrows=config.NROWS, df_raw=None):
    """Load data and prepare time-series for one machine.

    Accepts a pre-loaded df_raw to avoid redundant CSV reads when calling
    this for many machines in a loop.  Returns (ts, machine_id_used).
    """
    if df_raw is None:
        df_raw = pd.read_csv(
            config.DATA_PATH,
            nrows=nrows,
            usecols=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"],
        )
    if machine_id is None:
        machine_id = _pick_best_machine(df_raw)
    ts = _prepare_timeseries(df_raw, machine_id)
    return ts, machine_id
