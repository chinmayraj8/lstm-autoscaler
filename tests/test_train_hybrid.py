"""Unit tests for src/autoscaler/train_hybrid.py (Step 28).

Split, on purpose, the same way live_loop's hybrid path is: everything
that does NOT require TensorFlow (data-fetch, scaling, the
InsufficientRealHistory guard) runs unconditionally, matching this
project's TensorFlow-free test-fast convention (see tests/conftest.py).
`train_residual_hybrid_model` itself does real ARIMA fitting and real
LSTM training, so it needs TensorFlow -- that one test uses
`pytest.importorskip("tensorflow")` so it's automatically skipped under
test-fast and actually runs under test-full, without needing a separate
--ignore entry in ci.yml.

No real Prometheus, no real cluster -- `StaticMetricsSource` (Step 20)
plays the same role here it already plays in tests/test_live_loop.py.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.autoscaler import config
from src.autoscaler.metrics_source import StaticMetricsSource, synthetic_readings_series
from src.autoscaler.train_hybrid import (
    MIN_TRAIN_POINTS,
    InsufficientRealHistory,
    fetch_and_scale_training_series,
)

T0 = datetime(2026, 1, 1)


def _source(n_hours, cadence_minutes=5, seed=1, machine_id="m_test", mean=40.0, amplitude=15.0):
    n_points = int(n_hours * 60 / cadence_minutes) + 1
    series = synthetic_readings_series(T0, n_points=n_points, cadence_minutes=cadence_minutes,
                                       mean=mean, amplitude=amplitude, noise_std=2.0, seed=seed)
    return StaticMetricsSource({machine_id: series})


# ── fetch_and_scale_training_series ──────────────────────────────────────

def test_raises_insufficient_real_history_when_too_little_data():
    # Only ~1h of 5-min-cadence data -> ~12 points, far under MIN_TRAIN_POINTS.
    source = _source(n_hours=1)
    now = T0 + timedelta(hours=1)

    with pytest.raises(InsufficientRealHistory):
        fetch_and_scale_training_series("m_test", source, now, fit_hours=1.0)


def test_raises_insufficient_real_history_for_unknown_machine():
    source = StaticMetricsSource({})
    now = T0 + timedelta(hours=24)

    with pytest.raises(InsufficientRealHistory):
        fetch_and_scale_training_series("does_not_exist", source, now, fit_hours=24.0)


def test_fetch_and_scale_returns_enough_points_scaled_to_unit_range():
    # Comfortably over MIN_TRAIN_POINTS (109) at 5-min cadence: 24h -> ~288 points.
    fit_hours = 24.0
    source = _source(n_hours=fit_hours + 1)
    now = T0 + timedelta(hours=fit_hours + 1)

    scaled, scaler = fetch_and_scale_training_series("m_test", source, now, fit_hours=fit_hours)

    assert len(scaled) >= MIN_TRAIN_POINTS
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0
    # The scaler was fit on exactly this series (train-only-fit convention,
    # same as the rest of this project -- see data.py's leakage-fix
    # history) -- inverse-transforming should round-trip back to the raw
    # readings, not to some other range.
    recovered = scaler.inverse_transform(scaled.reshape(-1, 1)).flatten()
    assert recovered.min() >= 0.0  # synthetic_readings_series clips to non-negative


def test_fetch_and_scale_only_uses_data_within_the_fit_window():
    # Two very different regimes back to back; only the second (recent)
    # `fit_hours` window should end up in the returned series.
    n_hours = 48
    cadence = 5
    n_points = int(n_hours * 60 / cadence) + 1
    low = synthetic_readings_series(T0, n_points=n_points // 2, cadence_minutes=cadence,
                                    mean=5.0, amplitude=1.0, noise_std=0.5, seed=1)
    high_start = low.index[-1] + timedelta(minutes=cadence)
    high = synthetic_readings_series(high_start, n_points=n_points // 2, cadence_minutes=cadence,
                                     mean=90.0, amplitude=1.0, noise_std=0.5, seed=2)
    combined = pd_concat_series(low, high)
    source = StaticMetricsSource({"m_test": combined})
    now = combined.index[-1]

    fit_hours = 12.0  # comfortably inside the `high` regime only
    scaled, scaler = fetch_and_scale_training_series("m_test", source, now, fit_hours=fit_hours)
    recovered = scaler.inverse_transform(scaled.reshape(-1, 1)).flatten()

    assert recovered.mean() > 50.0  # dominated by the `high` regime, not `low`


def pd_concat_series(a, b):
    import pandas as pd
    return pd.concat([a, b])


# ── train_residual_hybrid_model (needs TensorFlow -- skipped under test-fast) ──

def test_train_residual_hybrid_model_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("tensorflow")
    from src.autoscaler.train_hybrid import train_residual_hybrid_model

    # Keep this test fast: real ARIMA fit + real LSTM training, but bounded
    # to a couple of epochs on a small synthetic series -- this is a
    # wiring/smoke test (does the real pipeline produce a real, loadable
    # .keras file?), not a claim about forecast quality, which is what the
    # real multi-day cluster run (scripts/train_real_hybrid_model.py) is for.
    monkeypatch.setattr(config, "MAX_EPOCHS", 2)
    monkeypatch.setattr(config, "ES_PATIENCE", 1)

    fit_hours = 24.0
    source = _source(n_hours=fit_hours + 1, seed=3)
    now = T0 + timedelta(hours=fit_hours + 1)
    model_dir = str(tmp_path / "hybrid_residual")

    result = train_residual_hybrid_model(
        "m_test", source, now, model_dir, fit_hours=fit_hours, seed=42,
    )

    assert result.machine_id == "m_test"
    assert result.model_path == str(tmp_path / "hybrid_residual" / "m_test.keras")
    assert (tmp_path / "hybrid_residual" / "m_test.keras").exists()
    assert result.n_train_points >= MIN_TRAIN_POINTS
    assert result.n_sequences > 0
    assert result.epochs_trained >= 1
    assert result.wall_clock_secs >= 0.0
    assert np.isfinite(result.final_train_loss)
    assert np.isfinite(result.train_residual_std)

    # No scratch checkpoint left behind (train_hybrid.py's own comment on
    # this -- _train_lstm deletes it once restore_best_weights has already
    # synced the best weights into the in-memory model).
    assert not (tmp_path / "hybrid_residual" / "_scratch_m_test.keras").exists()


def test_train_residual_hybrid_model_raises_insufficient_history_before_touching_tensorflow(tmp_path):
    # Should fail fast on the data-sufficiency check, before ever reaching
    # the `import tensorflow` line -- so this test needs no TF install and
    # runs under test-fast too.
    from src.autoscaler.train_hybrid import train_residual_hybrid_model

    source = _source(n_hours=1)
    now = T0 + timedelta(hours=1)

    with pytest.raises(InsufficientRealHistory):
        train_residual_hybrid_model("m_test", source, now, str(tmp_path), fit_hours=1.0)
