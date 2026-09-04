"""Unit tests for the ARIMA baseline wiring (src/autoscaler/arima_baseline.py).

Needs statsmodels (already a project dependency, used since Step 6) but no
TensorFlow -- imports the submodule directly, same convention as the other
test files (see conftest.py).

Uses a short synthetic AR(1)-like series and order=(1, 0, 0) throughout,
purely to keep each ARIMA fit fast; the real order (2, 0, 1) and real data
are exercised by experiments/run_arima_in_sim.py, not here.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.autoscaler.arima_baseline import _arima_rolling_forecast, _inv_flat
from src.autoscaler.data import _make_sequences

ORDER = (1, 0, 0)


def _synthetic_series(n=50, seed=7):
    rng = np.random.RandomState(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.6 * x[i - 1] + rng.normal(scale=0.1)
    # squash into [0, 1] so it looks like a scaled feature column
    x = (x - x.min()) / (x.max() - x.min())
    return x


def test_arima_rolling_forecast_n_sequences_matches_make_sequences_convention():
    series = _synthetic_series(n=60)
    train, target = series[:40], series[40:]
    lookback, horizon = 3, 2

    y_pred, y_true = _arima_rolling_forecast(train, np.array([]), target, lookback, horizon, ORDER)

    # Same window-count formula _make_sequences uses for an equivalent (lookback, horizon).
    expected_n = len(target) - lookback - horizon + 1
    X_check, y_check = _make_sequences(target.reshape(-1, 1), lookback, horizon)
    assert expected_n == len(X_check) == len(y_check)
    assert y_pred.shape == (expected_n, horizon)
    assert y_true.shape == (expected_n, horizon)


def test_arima_rolling_forecast_context_points_are_not_scored():
    series = _synthetic_series(n=60)
    train, target = series[:40], series[40:]
    lookback, horizon = 3, 2

    _, y_true = _arima_rolling_forecast(train, np.array([]), target, lookback, horizon, ORDER)

    # The first scored sequence's target is target[lookback:lookback+horizon] --
    # i.e. the first `lookback` points were used purely as state-advancing
    # context, exactly like the LSTM's lookback window (_make_sequences).
    np.testing.assert_array_almost_equal(y_true[0], target[lookback:lookback + horizon])
    np.testing.assert_array_almost_equal(y_true[-1], target[-horizon:])


def test_arima_rolling_forecast_is_deterministic():
    # Core claim of the module docstring: no seed-dependent randomness, so
    # running the identical (data, order) pair twice must be bit-identical.
    series = _synthetic_series(n=60)
    train, context, target = series[:30], series[30:40], series[40:]
    lookback, horizon = 3, 2

    y_pred_1, y_true_1 = _arima_rolling_forecast(train, context, target, lookback, horizon, ORDER)
    y_pred_2, y_true_2 = _arima_rolling_forecast(train, context, target, lookback, horizon, ORDER)

    np.testing.assert_array_equal(y_pred_1, y_pred_2)
    np.testing.assert_array_equal(y_true_1, y_true_2)


def test_arima_rolling_forecast_predictions_clipped_to_unit_interval():
    series = _synthetic_series(n=60)
    train, target = series[:40], series[40:]

    y_pred, _ = _arima_rolling_forecast(train, np.array([]), target, 3, 2, ORDER)

    assert (y_pred >= 0.0).all()
    assert (y_pred <= 1.0).all()


def test_inv_flat_matches_manual_scaler_inverse_transform():
    raw = np.array([[0.0], [10.0], [20.0], [30.0]])
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(raw)
    scaled = scaler.transform(raw).flatten()  # [0, 1/3, 2/3, 1]

    # shape (2, 2): two "sequences" of horizon=2
    arr = np.array([[scaled[0], scaled[1]], [scaled[2], scaled[3]]])
    inv = _inv_flat(arr, scaler)

    expected = np.array([[0.0, 10.0], [20.0, 30.0]])
    np.testing.assert_array_almost_equal(inv, expected)
