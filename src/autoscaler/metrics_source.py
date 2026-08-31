"""
Swappable metrics-ingestion interface (Step 20).

This is interface-only, deliberately. The concrete real connector (which
monitoring system to actually pull from -- Prometheus, a cloud provider's
metrics API, something custom) is not yet known, and building a specific
connector before knowing that would mean guessing at auth, query shape,
and label/tag conventions that don't exist yet. Nothing here talks to any
real infrastructure. `StaticMetricsSource` below is a synthetic
reference implementation, used only for tests and to prove the interface
contract actually works end-to-end -- it is not a stand-in for a real
system and must not be treated as production-ready.

Why pull, not push
-------------------
Every existing consumer in this project already works on bounded
historical windows -- a lookback window for a forecast, a 24-hour shadow
window -- not a live event stream. `experiments/*.py` scripts read a CSV
slice; `shadow.py`'s harness scores an already-observed window. A pull
interface (`fetch_readings(machine_id, start, end)`) keeps a future live
connector a drop-in replacement for "read a bounded slice of history,"
nothing more, so the forecasting/decision/shadow code downstream never
needs to know or care whether that slice came from a CSV, a mock, or a
real monitoring API.

Resampling convention
-----------------------
`resample_readings` intentionally duplicates (does not import)
`data._prepare_timeseries`'s exact three-line resample convention
(`.resample("5min").mean()` -> `.ffill()` -> `.dropna()`) rather than
risking a signature change to that already-validated, widely-used
function just to make it reusable from here. The two must stay
byte-identical; `tests/test_metrics_source.py` checks that directly
against the same synthetic data through both paths.

Still open (Step 20's own scope, stated plainly): no concrete real
connector, and no live forecasting loop that calls this on a schedule --
both wait on knowing what real monitoring system this eventually points
at. See progress/2026-08-31_step20-metrics-source-interface.md.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from . import config


class MetricsSource(ABC):
    """Swappable source of raw per-machine CPU-utilization readings. A
    concrete implementation knows how to talk to one real system; nothing
    else in this project's pipeline needs to know which one is in use."""

    @abstractmethod
    def fetch_readings(self, machine_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Raw (not yet resampled) readings for `machine_id` in
        `[start, end]`, as a DataFrame with a DatetimeIndex and at least a
        `config.FEATURE_COL` column. Whatever native cadence or timestamp
        format the real source uses, converting it into this shape is the
        implementation's job -- everything downstream (`resample_readings`
        onward) assumes it's already done.

        May return an empty DataFrame (same columns, zero rows) if there's
        no data in range -- callers must handle that, not treat it as an
        error; a newly-tracked machine or a real monitoring gap are both
        legitimate reasons for an empty result.
        """
        raise NotImplementedError

    def fetch_lookback_window(self, machine_id: str, now: datetime,
                              lookback_steps: int = config.LOOKBACK_STEPS,
                              resample_minutes: int = 5) -> pd.DataFrame:
        """Convenience built on `fetch_readings` + `resample_readings`:
        exactly enough history ending at `now` to cover `lookback_steps`
        `resample_minutes`-cadence readings (matching config.LOOKBACK_STEPS
        by default), with a small margin so `ffill` has real data to work
        with right at the window's start rather than immediately dropping
        those rows. This is the call a live forecasting loop makes once
        per tracked machine per tick (not built yet -- see module
        docstring)."""
        margin = timedelta(minutes=resample_minutes * 2)
        start = now - timedelta(minutes=resample_minutes * lookback_steps) - margin
        raw = self.fetch_readings(machine_id, start, now)
        return resample_readings(raw, resample_rule=f"{resample_minutes}min")


def resample_readings(raw: pd.DataFrame, feature_col: str = config.FEATURE_COL,
                      resample_rule: str = "5min") -> pd.DataFrame:
    """Turn a MetricsSource's raw readings into the standard cadence shape
    the rest of this project's pipeline was built and tested against --
    see module docstring for why this duplicates rather than imports
    `data._prepare_timeseries`'s resample step."""
    if raw.empty:
        return raw[[feature_col]] if feature_col in raw.columns else raw
    out = raw[[feature_col]].resample(resample_rule).mean()
    out = out.ffill().dropna()
    return out


class StaticMetricsSource(MetricsSource):
    """Synthetic reference implementation for tests only -- NOT a real
    connector. Serves readings from an in-memory, machine_id -> pandas
    Series[DatetimeIndex] mapping supplied at construction, sliced to the
    requested [start, end] range. Exists to prove `MetricsSource`'s
    contract is implementable and to test `resample_readings`/
    `fetch_lookback_window` without any real infrastructure."""

    def __init__(self, series_by_machine: dict):
        self._series_by_machine = series_by_machine

    def fetch_readings(self, machine_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        series = self._series_by_machine.get(machine_id)
        if series is None or len(series) == 0:
            return pd.DataFrame({config.FEATURE_COL: []}, index=pd.DatetimeIndex([]))
        sliced = series[(series.index >= start) & (series.index <= end)]
        return sliced.to_frame(name=config.FEATURE_COL)


def synthetic_readings_series(start: datetime, n_points: int, cadence_minutes: int = 1,
                              mean: float = 40.0, amplitude: float = 10.0,
                              noise_std: float = 2.0, seed: int = 0) -> pd.Series:
    """Build one machine's synthetic raw reading series (finer-than-5min
    cadence, to exercise `resample_readings`' averaging/ffill behavior for
    real -- not just pass through 5-min data unchanged) for tests and for
    `StaticMetricsSource` fixtures. A smooth diurnal-ish sine plus noise,
    clipped to be non-negative -- shaped like a CPU% series, not
    meaningful beyond that."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range(start, periods=n_points, freq=f"{cadence_minutes}min")
    t = np.arange(n_points)
    values = mean + amplitude * np.sin(2 * np.pi * t / max(n_points, 1) * 3) + rng.normal(0, noise_std, n_points)
    values = np.clip(values, 0.0, None)
    return pd.Series(values, index=idx, name=config.FEATURE_COL)
