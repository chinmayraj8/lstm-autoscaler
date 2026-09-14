"""Integration tests for the bearer-token auth dependency (src/api/main.py).

`_API_TOKEN` is read from LSTM_AUTOSCALER_API_TOKEN once, at module import
time -- setting the env var after import wouldn't take effect, so these
tests poke `main_module._API_TOKEN` directly instead, the same way
test_api_shadow.py's `_fresh_store` fixture pokes `main_module._shadow_store`
rather than trying to re-import the module for a fresh :memory: DB.
"""

import inspect
import os

os.environ["LSTM_AUTOSCALER_SHADOW_DB"] = ":memory:"

import pytest
from fastapi.testclient import TestClient

import src.api.main as main_module

client = TestClient(main_module.app)

TEST_TOKEN = "test-token-abc123"


@pytest.fixture(autouse=True)
def _with_token():
    original = main_module._API_TOKEN
    main_module._API_TOKEN = TEST_TOKEN
    yield
    main_module._API_TOKEN = original


def test_protected_route_401_with_no_authorization_header():
    r = client.get("/config")
    assert r.status_code == 401
    assert "Authorization" in r.json()["detail"]


def test_protected_route_401_with_wrong_token():
    r = client.get("/config", headers={"Authorization": "Bearer wrong-token"})
    assert r.status_code == 401
    assert r.json()["detail"] == "Invalid API token."


def test_protected_route_200_with_correct_token():
    r = client.get("/config", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
    assert r.status_code == 200


def test_metrics_route_returns_prometheus_text_format_with_correct_token():
    r = client.get("/metrics", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "autoscaler_decisions_total" in r.text


def test_protected_route_401_for_malformed_authorization_header():
    # A header that isn't "Bearer <token>" at all (e.g. Basic auth, or the
    # raw token with no scheme) must not be treated as a bearer token.
    r = client.get("/config", headers={"Authorization": TEST_TOKEN})
    assert r.status_code == 401


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/machines"),
        ("get", "/actuation/status"),
        ("get", "/replicas/demo-workload"),
        ("get", "/shadow/m_never_seen"),
        ("get", "/shadow/m_never_seen/windows"),
        ("get", "/shadow/m_never_seen/decisions"),
        ("get", "/metrics/cpu?machine_id=m_test"),
        ("get", "/metrics"),
        ("get", "/forecast/confidence?machine_id=m_test"),
    ],
)
def test_every_protected_route_rejects_missing_auth(method, path):
    # A representative route from every tag (shadow, cluster, ops,
    # inference) -- not just /config -- so a missed `dependencies=`
    # on one specific route doesn't go unnoticed.
    r = getattr(client, method)(path)
    assert r.status_code == 401


def test_health_has_no_auth_dependency_even_when_token_is_set():
    r = client.get("/health")
    # This TestClient never runs `lifespan` (same convention
    # test_api_shadow.py already established -- see its own module
    # docstring), so /health 503s here for an unrelated, pre-existing
    # reason ("still starting up"), not for auth. The real assertion is
    # that a token being set never turns this specific route into a 401.
    assert r.status_code != 401


def test_auth_disabled_entirely_when_token_is_none():
    main_module._API_TOKEN = None
    r = client.get("/config")
    assert r.status_code == 200


def test_require_auth_uses_a_constant_time_comparison():
    # Inspects the real source, not just behavior -- a plain `==` and
    # `hmac.compare_digest` are behaviorally identical for correctness,
    # but only one is timing-safe.
    source = inspect.getsource(main_module.require_auth)
    assert "hmac.compare_digest" in source
    assert "provided ==" not in source
    assert "== _API_TOKEN" not in source
    assert "_API_TOKEN ==" not in source
