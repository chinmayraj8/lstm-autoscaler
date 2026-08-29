"""Unit tests for data prep and splitting (src/autoscaler/data.py).

test_split_has_no_scaler_leakage is the one that matters most: it's a
regression test for the exact bug Step 1 of this project fixed (the
scaler must be fit on the train split only, never on val/test).
"""

import numpy as np
import pandas as pd

from src.autoscaler.data import _make_sequences, _pick_best_machine, _split_three_way


def test_split_three_way_sizes_are_chronological_60_20_20():
    n = 100
    ts = pd.DataFrame({"cpu_util_percent": np.arange(n, dtype=float)})

    train, val, test, scaler = _split_three_way(ts, "cpu_util_percent")

    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20


def test_split_has_no_scaler_leakage():
    # Values 0..99. Scaler must be fit on train (0..59) only, so it has
    # never seen the val/test range (60..99) -- transforming those values
    # through a train-only scaler should produce values outside [0, 1].
    n = 100
    ts = pd.DataFrame({"cpu_util_percent": np.arange(n, dtype=float)})

    train, val, test, scaler = _split_three_way(ts, "cpu_util_percent")

    # The scaler's own notion of the data range must match train only (0..59).
    assert scaler.data_min_[0] == 0.0
    assert scaler.data_max_[0] == 59.0

    # Values in val/test (60..99) exceed the train max, so they scale above 1.0 --
    # proof the scaler was never fit on them.
    assert (val > 1.0).any()
    assert (test > 1.0).any()
    assert train.max() <= 1.0 and train.min() >= 0.0


def test_make_sequences_window_count_and_content():
    data = np.arange(10, dtype=np.float32).reshape(-1, 1)  # [[0],[1],...,[9]]
    X, y = _make_sequences(data, lookback=3, horizon=2)

    # len(data) - lookback - horizon + 1 = 10 - 3 - 2 + 1 = 6
    assert X.shape == (6, 3, 1)
    assert y.shape == (6, 2)

    np.testing.assert_array_equal(X[0].flatten(), [0, 1, 2])
    np.testing.assert_array_equal(y[0], [3, 4])

    np.testing.assert_array_equal(X[-1].flatten(), [5, 6, 7])
    np.testing.assert_array_equal(y[-1], [8, 9])


def test_pick_best_machine_returns_highest_row_count():
    df = pd.DataFrame({
        "machine_id": ["a"] * 2 + ["b"] * 5 + ["c"] * 3,
    })
    assert _pick_best_machine(df) == "b"
