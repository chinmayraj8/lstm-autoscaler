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
