# Step 24: real-cluster verification of the Step 23 observer deployment
Date: 2026-09-03
Status: done

Closes the single most consequential "Still open" item from Step 23:
*"Not applied to the real target cluster... is the next action, and it's
the user's to take and verify."* The user has now built, deployed, and
verified the observer service against a real cluster (Docker Desktop
Kubernetes, 2 nodes) with kube-prometheus-stack installed.

## What was verified (against the real cluster, not a mock)

- Docker Desktop Kubernetes running, 2 nodes; `kubectl config
  current-context` = `docker-desktop` (not `kind` — `kind` isn't even
  installed on this machine).
- kube-prometheus-stack healthy in the `monitoring` namespace.
- `lstm-autoscaler-observer` pod: `1/1 Running` in the `lstm-autoscaler`
  namespace.
- Confirmed reachable at
  `http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090`
  (the in-cluster DNS URL Step 23's manifest hard-coded) — from inside the
  pod, `/api/v1/query` returned real CPU values for both node-exporter
  instances. This is the first time this project's Prometheus connector
  (Step 21) has been exercised against real infrastructure rather than a
  port-forwarded laptop or a mock.
- `/data/shadow_state.db` (the PVC-backed SQLite file, Step 19/23) exists
  and is being written to: 2 machines discovered and tracked, 6
  `observed_decisions` logged.
- `LSTM_AUTOSCALER_SKIP_LSTM_MODEL=1` confirmed intentional for this run —
  this is an observer-only shadow test, not exercising `/forecast` or the
  residual hybrid model (none exists, per every prior step's "still
  open").

## Deployment path actually used (deviates from Step 23's assumption)

Step 23 assumed the target cluster would need a real registry push and
left `image:` as a placeholder. In practice: the target is `docker-desktop`,
and Docker Desktop's Kubernetes does **not** share `docker build`'s image
store with the cluster the way it might seem to — kubelet reads images via
containerd's `k8s.io` namespace, while `docker build` writes to the
separate `moby` namespace. Setting `imagePullPolicy: Never` against a bare
local build tag reproducibly hangs the pod in `ErrImageNeverPull` on this
machine, even after a successful `docker build` and a rollout restart.

Resolved with a throwaway local registry, which Docker Desktop's
Kubernetes node can reach directly over `localhost`:

```
docker run -d --restart=always -p 5000:5000 --name local-registry registry:2
docker tag lstm-autoscaler-observer:latest localhost:5000/lstm-autoscaler-observer:latest
docker push localhost:5000/lstm-autoscaler-observer:latest
```

`k8s/observer.yaml` was updated in place to reference
`localhost:5000/lstm-autoscaler-observer:latest` with `imagePullPolicy:
IfNotPresent`, and its comments rewritten to document this as the actual
path for a `docker-desktop` context, not "still needs a registry, TBD."

## 0 observed shadow_windows, 0 assignment_changes: expected, not a gap

Worth stating explicitly since it could otherwise look broken.
`run_scheduler_loop` (Step 22, `live_loop.py`) needs `shadow_fit_hours`
(6h) + `shadow_window_hours` (24h) = **30h** of real Prometheus history for
a machine before `run_tick` can score even the first shadow window
(`shadow.run_shadow_window`) — `evaluate_and_maybe_reassign` never runs,
so no `AssignmentChange` can exist, until at least one window is banked.

6 `observed_decisions` at the 5-minute tick cadence
(`LSTM_AUTOSCALER_TICK_SECONDS=300`) is roughly 30 minutes of uptime —
nowhere near 30 hours. The always-on single-forecaster ARIMA path
(`observe_node_once`, populating `observed_decisions`) and the
shadow-comparison path (`maybe_run_shadow_cycle`, populating
`shadow_windows`/`assignment_changes`) run on two different cadences by
design (Step 22); only the first has had time to produce anything yet.

## Before → After

| | Before Step 24 | After Step 24 |
|---|---|---|
| Deployed where | Nowhere (manifest + Dockerfile only, unapplied) | Real Docker Desktop Kubernetes cluster (2 nodes) |
| Image delivery | Placeholder tag, registry TBD | Local registry (`localhost:5000`), documented in `k8s/observer.yaml` |
| Prometheus connectivity | Validated against mocks / local port-forward only | Confirmed live: real CPU values for 2 node-exporter instances |
| Shadow state persistence | Validated locally only (Step 19) | Confirmed PVC-backed and actively written to on the real cluster |
| observed_decisions | 0 | 6 |
| shadow_windows / assignment_changes | 0 / 0 (untestable — no cluster access) | 0 / 0 (expected — needs ~30h more uptime, not a defect) |

## Still open

Unchanged from Step 23 except where noted:

- **shadow_windows / assignment_changes still empty** — needs real
  wall-clock time (~30h minimum per machine) before the first one can even
  be attempted. Worth checking again once that much uptime has
  accumulated, not before.
- **No scale-up/scale-down call to real infrastructure exists anywhere in
  this codebase.** Unchanged, out of scope by design — this step doesn't
  move that boundary.
- **No pretrained residual-hybrid model exists.** Unchanged; once shadow
  windows do start banking, `decide_assignment` can in principle recommend
  the hybrid, but nothing serves it yet (`HybridModelUnavailable`, by
  design).
- **PVC `storageClassName`, backup/snapshot policy, and server-side
  manifest validation** — still exactly as Step 23 left them. This
  verification did not test a pod reschedule/eviction against the real
  PVC, so "does shadow state actually survive a real reschedule on this
  cluster" remains unverified, not just theoretically fine.
- **HTTP call volume, no retry/backoff on a transient Prometheus failure,
  ARIMA refit-from-scratch-every-tick** — every Step 21-22 "still open"
  item, unchanged by this step.
