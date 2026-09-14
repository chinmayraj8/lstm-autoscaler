"""
Prometheus instrumentation -- metrics EXPOSED BY this service, scraped by
a real Prometheus (kube-prometheus-stack, via k8s/observer.yaml's
ServiceMonitor). The reverse direction of `metrics_source.py`, which PULLS
node CPU history FROM Prometheus; this module is what `GET /metrics`
(src/api/main.py) renders FOR Prometheus to pull.

Every metric here is set from real internal state at the exact point that
state already changes (`live_loop.py`'s `observe_node_once`/`run_tick`,
`shadow_store.py`'s `run_shadow_cycle`) -- nothing here is a separate,
parallel bookkeeping system that could drift from what actually happened.

A dedicated `CollectorRegistry`, not `prometheus_client`'s process-global
default -- keeps this service's metrics namespace self-contained (no risk
of a name collision with some other library that also happens to register
against the default registry) and means re-importing this module (e.g.
once from `live_loop.py`, once from a test) never risks a "duplicated
timeseries" registration error, since the Counter/Gauge objects below are
each created exactly once, at import time.
"""

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, generate_latest

REGISTRY = CollectorRegistry()

# One row per (machine_id, forecaster, action) combination ever observed.
# `action` is the "kind" only ("hold" | "scale_up" | "scale_down") -- the
# full action STRING from decision.py/simulation.py carries a "+N"/"-N"
# delta that would otherwise turn this into an unbounded-cardinality label.
DECISIONS_TOTAL = Counter(
    "autoscaler_decisions_total",
    "Observed scaling decisions logged by the live loop, one per tracked node per tick.",
    ["machine_id", "forecaster", "action"],
    registry=REGISTRY,
)

# The live loop's own ledger value (ShadowStore.get_last_recommended_servers),
# not necessarily the real cluster's current replica count -- same
# distinction observe_node_once's own docstring draws.
RECOMMENDED_SERVERS = Gauge(
    "autoscaler_recommended_servers",
    "Most recently recommended replica count per node, per the live loop's internal ledger.",
    ["machine_id"],
    registry=REGISTRY,
)

# Only incremented for the one machine actually designated by
# LiveLoopConfig.actuation_machine_id, and only when enable_actuation is
# True -- see live_loop.run_tick. "result" is "success" | "failure"; a
# circuit-breaker-open SKIP is not counted here (see CIRCUIT_BREAKER_OPEN
# below) since it isn't an actuation attempt.
ACTUATION_TOTAL = Counter(
    "autoscaler_actuation_total",
    "Real actuation attempts against the cluster, by outcome.",
    ["machine_id", "result"],
    registry=REGISTRY,
)

CIRCUIT_BREAKER_OPEN = Gauge(
    "autoscaler_circuit_breaker_open",
    "1 if the real-actuation circuit breaker is currently open for this machine, 0 otherwise.",
    ["machine_id"],
    registry=REGISTRY,
)

# Cumulative cost (shadow.cumulative_summary's arima_cost_total /
# hybrid_cost_total) across every shadow window ever banked for this
# machine -- the SAME numbers /shadow/{machine_id} already reports,
# exposed here as a gauge so Prometheus can chart the two forecasters'
# real cost-aware comparison over time, not just cost-per-window snapshots.
SHADOW_WINDOW_COST = Gauge(
    "autoscaler_shadow_window_cost",
    "Cumulative shadow-window cost score per forecaster, across all banked windows for this machine.",
    ["machine_id", "forecaster"],
    registry=REGISTRY,
)


def render_metrics() -> bytes:
    """Prometheus text-exposition-format bytes for GET /metrics."""
    return generate_latest(REGISTRY)
