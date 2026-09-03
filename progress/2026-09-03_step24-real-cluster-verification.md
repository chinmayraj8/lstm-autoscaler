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

## 0 observed shadow_windows, 0 assignment_changes: NOT just "needs more uptime"

**Correction (Step 25):** this section originally said the reason
`shadow_windows`/`assignment_changes` were still empty was that a machine
needs 30h of real Prometheus history before the first shadow window can be
scored, and that this was "expected, not a gap" — just a matter of waiting.
**That explanation was incomplete to the point of being misleading.**
Step 25 traced the actual code path and found two structural blockers that
mean `shadow_windows`/`assignment_changes` can never appear on their own,
at ANY wall-clock time, no matter how long this deployment runs
unattended:

1. `run_tick` (`live_loop.py`) only attempts shadow-window scoring
   `if state.current_forecaster == "hybrid"`. But the ONLY thing that ever
   sets `current_forecaster` to `"hybrid"` is `evaluate_and_maybe_reassign`
   — which is itself only ever reached from INSIDE that same
   hybrid-gated path (`run_tick` → `run_shadow_cycle` →
   `maybe_run_shadow_cycle` → `evaluate_and_maybe_reassign`). A machine
   starting on the database default (`"arima"`) can never organically
   become eligible to be evaluated for hybrid assignment — this is a
   closed loop with no entry point, not a slow ramp.
2. Even a machine manually pushed to `"hybrid"` would immediately hit
   `HybridModelUnavailable`: no `.keras` residual-hybrid model exists
   anywhere in this repository for any machine (only the unrelated
   original `lstm_model.keras` from the early single-machine steps).

So the true reason 0/0 held at Step 24's writing wasn't "needs ~30h more
uptime" — it was that nothing in this deployment, run for any amount of
time, could ever produce a `shadow_window` without deliberate manual
intervention (seeding an assignment directly in the database, and
providing a residual model file). Step 25 does exactly that — as an
explicit, flagged mechanical demo, not a real forecasting result — to
prove the scoring pipeline itself works once those two blockers are
worked around by hand. See that step's progress doc for what was actually
done and verified.

The now-corrected but still true parts of the original explanation: 6
`observed_decisions` at the 5-minute tick cadence
(`LSTM_AUTOSCALER_TICK_SECONDS=300`) reflected roughly 30 minutes of
uptime at the time of writing. The always-on single-forecaster ARIMA path
(`observe_node_once`, populating `observed_decisions`) and the
shadow-comparison path (`maybe_run_shadow_cycle`, populating
`shadow_windows`/`assignment_changes`) do run on two different cadences by
design (Step 22) — that part was accurate. What was wrong was implying the
second path would eventually produce something on its own given enough
elapsed time; it structurally cannot, without the intervention Step 25
performed.

## Before → After

| | Before Step 24 | After Step 24 |
|---|---|---|
| Deployed where | Nowhere (manifest + Dockerfile only, unapplied) | Real Docker Desktop Kubernetes cluster (2 nodes) |
| Image delivery | Placeholder tag, registry TBD | Local registry (`localhost:5000`), documented in `k8s/observer.yaml` |
| Prometheus connectivity | Validated against mocks / local port-forward only | Confirmed live: real CPU values for 2 node-exporter instances |
| Shadow state persistence | Validated locally only (Step 19) | Confirmed PVC-backed and actively written to on the real cluster |
| observed_decisions | 0 | 6 |
| shadow_windows / assignment_changes | 0 / 0 (untestable — no cluster access) | 0 / 0 (see correction above — a structural gating issue, not just elapsed time; see Step 25) |

## Still open

Unchanged from Step 23 except where noted:

- ~~shadow_windows / assignment_changes still empty — needs real wall-clock
  time (~30h minimum per machine) before the first one can even be
  attempted~~ — **corrected by Step 25**: this was never just a matter of
  elapsed time. Two structural blockers (the hybrid-assignment closed loop,
  and no residual-hybrid model existing anywhere) mean these tables cannot
  populate on their own at all, regardless of uptime. See Step 25's
  progress doc for the full trace and a mechanical demo proving the
  scoring pipeline works once those blockers are worked around by hand.
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
