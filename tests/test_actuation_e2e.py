"""One true end-to-end test for real actuation (Step 26) and its circuit
breaker (Step 26 follow-up): a REAL `kubernetes.client.rest.ApiException`,
raised from the actual Kubernetes client boundary this project's actuation
path ultimately calls (`AppsV1Api.patch_namespaced_deployment_scale`), run
all the way through `actuator.set_replicas`'s own translation into
`ActuationError`, into `live_loop.run_tick`'s circuit-breaker handling.

Every existing test elsewhere in this suite mocks at ONE layer's own
immediate boundary: test_actuator.py mocks the K8s client to prove
actuator.py's translation is correct in isolation; test_live_loop.py mocks
`actuator.set_replicas` itself (a hand-rolled `ActuationError`) to prove
run_tick's breaker logic is correct in isolation. Neither proves the two
translations actually compose -- that a real K8s failure, several layers
down, actually surfaces as a tripped breaker rather than, say, an
unhandled exception crashing the tick loop. This file is that proof, not
a replacement for either of those more targeted test files.

No TensorFlow needed -- this only exercises the ARIMA-only observation
path plus real actuation, never the hybrid shadow-window path.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client.rest import ApiException

from src.autoscaler import config
from src.autoscaler.actuator import ActuationError, set_replicas
from src.autoscaler.instrumentation import ACTUATION_TOTAL, CIRCUIT_BREAKER_OPEN
from src.autoscaler.live_loop import LiveLoopConfig, run_tick
from src.autoscaler.metrics_source import StaticMetricsSource, synthetic_readings_series
from src.autoscaler.shadow_store import ShadowStore

T0 = datetime(2026, 1, 1)


def _source(machine_id, n_hours=4, cadence_minutes=5, seed=1):
    n_points = int(n_hours * 60 / cadence_minutes) + 1
    series = synthetic_readings_series(T0, n_points=n_points, cadence_minutes=cadence_minutes,
                                       mean=40.0, amplitude=8.0, noise_std=1.5, seed=seed)
    return StaticMetricsSource({machine_id: series})


def _counter_value(counter, **labels):
    return counter.labels(**labels)._value.get()


def test_real_k8s_api_exception_trips_the_circuit_breaker_end_to_end():
    machine_id = "m_e2e_breaker"
    source = _source(machine_id)
    store = ShadowStore(":memory:")
    cfg = LiveLoopConfig(
        fit_lookback_hours=2.0, enable_actuation=True, actuation_machine_id=machine_id,
        actuation_deployment="demo-workload", actuation_namespace="lstm-autoscaler",
        actuation_circuit_breaker_threshold=2, actuation_circuit_breaker_cooldown=timedelta(hours=1),
    )

    # The actual K8s client boundary actuator.set_replicas calls --
    # `AppsV1Api.patch_namespaced_deployment_scale` -- raises a REAL
    # ApiException, not a hand-rolled ActuationError. `_load_config` is
    # mocked only so this test needs no real kubeconfig or in-cluster
    # token (same convention test_actuator.py already established);
    # `set_replicas` itself is left completely UNMOCKED, so its own real
    # `except ApiException as e: raise ActuationError(...)` translation
    # actually runs.
    fake_api = MagicMock()
    fake_api.patch_namespaced_deployment_scale.side_effect = ApiException(
        status=500, reason="Internal Server Error",
    )

    failures_before = _counter_value(ACTUATION_TOTAL, machine_id=machine_id, result="failure")

    # get_current_replicas is mocked at the HIGH level (matching the
    # ledger's default) purely to keep read-before-write reconciliation
    # out of this test's way -- that behavior is already covered
    # elsewhere (test_live_loop.py's reconciliation tests); it has no
    # bearing on whether the breaker trips, since set_replicas fails
    # regardless of which replica count it's called with.
    with patch("src.autoscaler.actuator.get_current_replicas", return_value=config.SIM_INITIAL_SERVERS), \
         patch("src.autoscaler.actuator._load_config"), \
         patch("src.autoscaler.actuator.k8s_client.AppsV1Api", return_value=fake_api):

        # First, prove the translation layer itself, directly: run_tick's
        # two except clauses (ActuationError vs. bare Exception) both
        # record the same failure/breaker bookkeeping, so a run_tick-only
        # assertion couldn't tell a real ActuationError translation apart
        # from a raw ApiException merely leaking up into the generic
        # fallback. Calling actuator.set_replicas directly, under the
        # SAME client-boundary mocks, pins down which one actually fires.
        with pytest.raises(ActuationError, match="could not set replicas"):
            set_replicas("demo-workload", "lstm-autoscaler", 3)
        assert fake_api.patch_namespaced_deployment_scale.call_count == 1
        fake_api.patch_namespaced_deployment_scale.reset_mock()  # call-count only; side_effect stays set

        # Now the actual end-to-end claim: two real, failed run_tick
        # attempts -- threshold=2 means the second trips the breaker.
        run_tick(store, source, [machine_id], T0 + timedelta(hours=3), cfg)
        run_tick(store, source, [machine_id], T0 + timedelta(hours=3, minutes=5), cfg)

        assert fake_api.patch_namespaced_deployment_scale.call_count == 2
        assert cfg._actuation_circuit_opened_at is not None

        # A third tick, still well inside the cooldown window: the breaker
        # is open, so the real K8s client must NOT be hit again this tick
        # -- proving the breaker actually stops hammering a genuinely
        # failing control plane, not just that it recorded the failures.
        summary = run_tick(store, source, [machine_id], T0 + timedelta(hours=3, minutes=10), cfg)

    assert fake_api.patch_namespaced_deployment_scale.call_count == 2  # unchanged -- no third attempt
    assert summary["actuation_skipped"] == [machine_id]
    assert summary["actuated"] == []

    # Nice-to-have: the same failure visible in GET /metrics, not just in
    # run_tick's return value.
    failures_after = _counter_value(ACTUATION_TOTAL, machine_id=machine_id, result="failure")
    assert failures_after == failures_before + 2
    assert CIRCUIT_BREAKER_OPEN.labels(machine_id=machine_id)._value.get() == 1.0
