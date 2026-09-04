"""Unit tests for real Kubernetes actuation (src/autoscaler/actuator.py, Step 26).

No real cluster, no real kubeconfig: `kubernetes.client.AppsV1Api` is
mocked at the point actuator.py imports it, and `_load_config` (in-cluster
config, falling back to local kubeconfig) is mocked too so these tests
never touch any actual config file or cluster, matching this project's
existing "no real infrastructure in unit tests" convention
(test_metrics_source.py's PrometheusMetricsSource tests are the closest
precedent -- a mocked client, never a live call).
"""

from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client.rest import ApiException

from src.autoscaler.actuator import ActuationError, get_current_replicas, set_replicas


def test_get_current_replicas_reads_the_scale_subresource():
    fake_scale = MagicMock()
    fake_scale.spec.replicas = 3
    fake_api = MagicMock()
    fake_api.read_namespaced_deployment_scale.return_value = fake_scale

    with patch("src.autoscaler.actuator._load_config"), \
         patch("src.autoscaler.actuator.k8s_client.AppsV1Api", return_value=fake_api):
        replicas = get_current_replicas("demo-workload", "lstm-autoscaler")

    assert replicas == 3
    fake_api.read_namespaced_deployment_scale.assert_called_once_with("demo-workload", "lstm-autoscaler")


def test_get_current_replicas_wraps_api_errors():
    fake_api = MagicMock()
    fake_api.read_namespaced_deployment_scale.side_effect = ApiException(status=403, reason="Forbidden")

    with patch("src.autoscaler.actuator._load_config"), \
         patch("src.autoscaler.actuator.k8s_client.AppsV1Api", return_value=fake_api):
        with pytest.raises(ActuationError, match="could not read scale"):
            get_current_replicas("demo-workload", "lstm-autoscaler")


def test_set_replicas_patches_the_scale_subresource_with_the_given_value():
    fake_api = MagicMock()

    with patch("src.autoscaler.actuator._load_config"), \
         patch("src.autoscaler.actuator.k8s_client.AppsV1Api", return_value=fake_api):
        set_replicas("demo-workload", "lstm-autoscaler", 4)

    fake_api.patch_namespaced_deployment_scale.assert_called_once_with(
        "demo-workload", "lstm-autoscaler", {"spec": {"replicas": 4}},
    )


def test_set_replicas_wraps_api_errors():
    fake_api = MagicMock()
    fake_api.patch_namespaced_deployment_scale.side_effect = ApiException(status=404, reason="Not Found")

    with patch("src.autoscaler.actuator._load_config"), \
         patch("src.autoscaler.actuator.k8s_client.AppsV1Api", return_value=fake_api):
        with pytest.raises(ActuationError, match="could not set replicas"):
            set_replicas("demo-workload", "lstm-autoscaler", 2)


def test_set_replicas_refuses_a_negative_count_without_calling_the_api():
    fake_api = MagicMock()

    with patch("src.autoscaler.actuator._load_config"), \
         patch("src.autoscaler.actuator.k8s_client.AppsV1Api", return_value=fake_api):
        with pytest.raises(ActuationError, match="negative"):
            set_replicas("demo-workload", "lstm-autoscaler", -1)

    fake_api.patch_namespaced_deployment_scale.assert_not_called()


def test_load_config_falls_back_to_kube_config_outside_a_cluster():
    from kubernetes.config import ConfigException

    with patch("src.autoscaler.actuator.k8s_config.load_incluster_config", side_effect=ConfigException("no token")):
        with patch("src.autoscaler.actuator.k8s_config.load_kube_config") as fake_kube_config:
            from src.autoscaler.actuator import _load_config
            _load_config()
            fake_kube_config.assert_called_once()
