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

Step 20 left this interface-only, waiting on a named real system. Step 21
(Stage 3) names one: a Kubernetes cluster running kube-prometheus-stack,
Prometheus reachable in-cluster at
http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090,
scraping node-exporter, machine unit = node. `PrometheusMetricsSource`
below is the concrete connector for that system -- see its own docstring
for why it's built on `/api/v1/query` (instant queries), not
`/api/v1/query_range`, and for the HTTP-call-volume trade-off that choice
implies. See progress/2026-09-03_step21-prometheus-metrics-source.md.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

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


# ── PrometheusMetricsSource: the real connector (Step 21 / Stage 3) ────────

# In-cluster Prometheus URL (kube-prometheus-stack default Service DNS name),
# confirmed working against the real target cluster. Overridable per
# instance -- this default is just what a caller gets for free.
DEFAULT_PROMETHEUS_URL = "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090"

# The two pieces of the verified PromQL query that would otherwise be
# magic strings duplicated between the query-builder and anyone reading it:
#   100 * (1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])))
DEFAULT_RANGE_VECTOR = "5m"
CPU_IDLE_MODE_LABEL = "idle"


def build_cpu_util_query(range_vector: str = DEFAULT_RANGE_VECTOR,
                         idle_mode_label: str = CPU_IDLE_MODE_LABEL) -> str:
    """Build the per-node CPU% PromQL query, parameterized so the range
    vector and the idle-mode label are each written exactly once -- not
    hardcoded again inside `PrometheusMetricsSource`. With the defaults,
    reproduces byte-for-byte the query verified working against the real
    cluster:

        100 * (1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])))
    """
    return (
        f'100 * (1 - avg by (instance) '
        f'(rate(node_cpu_seconds_total{{mode="{idle_mode_label}"}}[{range_vector}])))'
    )


def instance_to_machine_id(instance_label: str) -> str:
    """Map a Prometheus `instance` label to this project's `machine_id`.

    node-exporter's `instance` label is conventionally `<host>:<port>`
    (e.g. "10.244.1.5:9100" -- kube-prometheus-stack's default node-exporter
    port). Every other `machine_id` in this project (the Alibaba trace CSV,
    `StaticMetricsSource` fixtures) is a bare identifier with no port, so
    the port suffix is stripped for consistency. A label with no ':' is
    returned unchanged (already bare, or a cluster whose instance labels
    don't include a port)."""
    return instance_label.rsplit(":", 1)[0] if ":" in instance_label else instance_label


class PrometheusMetricsSourceError(RuntimeError):
    """Raised when Prometheus responds but not the way this connector
    expects (a non-"success" status, or a resultType other than "vector"
    -- the instant-query endpoint should never return anything else for
    this query shape). Distinct from `requests` exceptions (network/HTTP
    failures), which are left to propagate as-is."""


class PrometheusMetricsSource(MetricsSource):
    """Real `MetricsSource` connector for an in-cluster Prometheus running
    kube-prometheus-stack, scraping node-exporter. Machine unit = node.

    Why `/api/v1/query` (instant queries), not `/api/v1/query_range`
    ----------------------------------------------------------------
    The verified query is a PromQL instant-vector expression -- it already
    embeds its own trailing window (`rate(...[5m])`), so a single instant
    query evaluated `time=T` reads as "the rate over the 5 minutes ending
    at T", not "the value at T with no history". Building history for
    `fetch_readings(machine_id, start, end)` therefore means calling
    `/api/v1/query` once per sample point across `[start, end]` at
    `query_step_minutes` cadence (default 5, matching this project's
    resample cadence -- see `resample_readings`), each with an explicit
    `time=` parameter, rather than one `/api/v1/query_range` call. This is
    less efficient (N HTTP round-trips instead of 1 for an N-point window)
    but matches the exact endpoint and query shape verified against the
    real cluster; see the Stage 3 progress doc for the trade-off and why
    `query_range` was not substituted in.

    Each call returns a vector (one sample per currently-scraped instance),
    filtered here to the one row matching `machine_id` via
    `instance_to_machine_id`.
    """

    def __init__(self, prometheus_url: str = DEFAULT_PROMETHEUS_URL,
                range_vector: str = DEFAULT_RANGE_VECTOR,
                idle_mode_label: str = CPU_IDLE_MODE_LABEL,
                query_step_minutes: int = 5,
                timeout_seconds: float = 10.0,
                session: Optional[requests.Session] = None):
        self.prometheus_url = prometheus_url.rstrip("/")
        self.range_vector = range_vector
        self.idle_mode_label = idle_mode_label
        self.query = build_cpu_util_query(range_vector, idle_mode_label)
        self.query_step_minutes = query_step_minutes
        self.timeout_seconds = timeout_seconds
        self._session = session or requests.Session()

    def _query_instant(self, at: Optional[datetime] = None) -> List[dict]:
        """One `/api/v1/query` call. `at=None` means "now" (Prometheus's own
        default when `time` is omitted); otherwise evaluates the query as of
        that timestamp -- Prometheus retains raw samples within its
        retention window, so a past `time=` still works for recent history,
        it isn't limited to "the current instant" despite the endpoint's name.
        Returns the raw `data.result` list of `{"metric": {...}, "value": [ts, "val"]}`.
        """
        params = {"query": self.query}
        if at is not None:
            params["time"] = at.timestamp()
        resp = self._session.get(
            f"{self.prometheus_url}/api/v1/query", params=params, timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != "success":
            raise PrometheusMetricsSourceError(f"Prometheus query failed: {payload}")
        data = payload.get("data", {})
        if data.get("resultType") != "vector":
            raise PrometheusMetricsSourceError(
                f"Expected an instant vector, got resultType={data.get('resultType')!r}"
            )
        return data.get("result", [])

    def fetch_readings(self, machine_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Raw per-node CPU% readings for `machine_id` in `[start, end]`,
        built from repeated `/api/v1/query` calls at `query_step_minutes`
        cadence (see class docstring). Empty (not an error) if the node
        never appears in any sampled response -- a newly-joined node, one
        that's since left the cluster, or a real scrape gap are all
        legitimate reasons, matching the base contract in `fetch_readings`'s
        docstring."""
        if end < start:
            raise ValueError(f"end ({end}) is before start ({start})")

        step = timedelta(minutes=self.query_step_minutes)
        timestamps: List[datetime] = []
        values: List[float] = []
        t = start
        while t <= end:
            for sample in self._query_instant(t):
                if instance_to_machine_id(sample.get("metric", {}).get("instance", "")) == machine_id:
                    _, val_str = sample["value"]
                    timestamps.append(t)
                    values.append(float(val_str))
                    break
            t += step

        if not timestamps:
            return pd.DataFrame({config.FEATURE_COL: []}, index=pd.DatetimeIndex([], name="time_stamp"))
        return pd.DataFrame(
            {config.FEATURE_COL: values},
            index=pd.DatetimeIndex(timestamps, name="time_stamp"),
        )

    def list_machine_ids(self, at: Optional[datetime] = None) -> List[str]:
        """Discover every node currently (or, with `at`, previously) visible
        to this query -- one instant query, mapped through
        `instance_to_machine_id` and deduplicated. NOT part of the
        `MetricsSource` ABC contract (a synthetic/offline source has no
        notion of "what nodes exist right now"); this is an extension
        specific to a live connector, used by the Stage 4 scheduler for
        node discovery when no explicit tracked-node list is configured."""
        ids = {
            instance_to_machine_id(sample.get("metric", {}).get("instance", ""))
            for sample in self._query_instant(at)
        }
        return sorted(i for i in ids if i)
