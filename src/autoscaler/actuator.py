"""
Real Kubernetes actuation (Step 26).

Every module up to this point in the "make the shadow harness watch real
infrastructure" project (metrics_source.py, live_loop.py) was explicit that
NO scaling call existed anywhere -- observe and log only. This module is
the one exception, and it's deliberately narrow: it does exactly one
thing, read or patch a single target Deployment's `scale` subresource, and
nothing else. It never decides a target replica count itself (that
remains decision.py's job, completely unchanged) -- it only applies a
number it's handed.

Not wired to anything by default. `live_loop.run_tick` only ever calls
this module when `LiveLoopConfig.enable_actuation` is explicitly set,
which defaults to False -- see that module for the gating logic and why.
This module itself has no on/off opinion; it assumes that if it's being
called, actuation has already been deliberately enabled by the caller.

RBAC: this module needs (and the accompanying k8s/actuation-rbac.yaml
grants) get/patch on exactly one resource -- `deployments/scale` -- in one
namespace. Not cluster-admin, not access to any other object kind, not
access to any other namespace. See that manifest's own comments.

Target: since the real signal this project forecasts (node-level CPU on
the underlying cluster nodes) isn't tied to any actual variable-load
service in this cluster, the target this module scales in the current
deployment is a small, clearly-labeled demo workload
(k8s/demo-workload.yaml) created for exactly this purpose -- not a
real production service. That is a real, honestly-stated limitation:
this proves the actuation call itself works end-to-end against a real
Deployment, not that the forecast is driving a real service's real
traffic.
"""

import logging

from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)


class ActuationError(Exception):
    """Raised when a real scale call could not be completed -- a
    Kubernetes API error, an auth/RBAC problem, or an invalid replica
    count. Caught by the caller (live_loop.run_tick) exactly like every
    other per-tick real-infrastructure failure in this project: logged,
    the tick continues, the process never crashes over one bad call."""


def _load_config() -> None:
    """In-cluster config when running as a pod -- the real target for this
    module (k8s/observer.yaml's ServiceAccount). Falls back to the local
    kubeconfig only so this module is exercisable interactively/in a
    real-cluster integration test run from outside the cluster; the unit
    tests (tests/test_actuator.py) never reach this function at all --
    they patch `kubernetes.client.AppsV1Api` directly."""
    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()


def get_current_replicas(deployment: str, namespace: str) -> int:
    """Reads the target Deployment's CURRENT replica count via the scale
    subresource (not `spec.replicas` on the Deployment object itself --
    same underlying field, but reading/writing only the scale subresource
    keeps this module's RBAC need limited to exactly that subresource)."""
    _load_config()
    api = k8s_client.AppsV1Api()
    try:
        scale = api.read_namespaced_deployment_scale(deployment, namespace)
    except ApiException as e:
        raise ActuationError(
            f"could not read scale for deployment={deployment!r} namespace={namespace!r}: {e}"
        ) from e
    return int(scale.spec.replicas)


def set_replicas(deployment: str, namespace: str, replicas: int) -> None:
    """Patches the target Deployment's replica count via the scale
    subresource -- the actual real-world action this whole project has
    deliberately never taken until this step. Raises `ActuationError`
    (never a raw kubernetes-client exception) on any failure, so the
    caller's existing per-tick try/except pattern catches this exactly
    like every other real-infrastructure failure already handled in
    `live_loop.run_tick`."""
    if replicas < 0:
        raise ActuationError(f"refusing to set a negative replica count ({replicas})")
    _load_config()
    api = k8s_client.AppsV1Api()
    body = {"spec": {"replicas": replicas}}
    try:
        api.patch_namespaced_deployment_scale(deployment, namespace, body)
    except ApiException as e:
        raise ActuationError(
            f"could not set replicas={replicas} for deployment={deployment!r} "
            f"namespace={namespace!r}: {e}"
        ) from e
    logger.info(
        "actuator: set deployment=%s namespace=%s replicas=%d",
        deployment, namespace, replicas,
    )
