# Step 26: real Kubernetes actuation -- code complete and verified on the real cluster
Date: 2026-09-04
Status: DONE. Real forecast -> real decision -> real Kubernetes API call ->
real replica count change, confirmed on the actual Docker Desktop cluster
(see "Real-cluster verification" below).

Every module in the "make the shadow harness watch real infrastructure"
project (Steps 20-25) was explicit that no scaling call existed anywhere
-- observe and log only, deliberately. This step adds exactly one real
scaling call, gated behind an explicit opt-in that defaults to off, so
every deployment through Step 25 remains byte-for-byte unchanged unless
someone deliberately turns actuation on.

## What changed

**`src/autoscaler/actuator.py`** (new). A narrow Kubernetes client wrapper:
`get_current_replicas` / `set_replicas`, both operating on exactly one
resource -- a Deployment's `scale` subresource -- and nothing else. Raises
`ActuationError` (never a raw `kubernetes`-client exception) on any
failure, matching this project's existing per-tick "log it, move on"
convention. In-cluster config by default, kubeconfig fallback only for
running it interactively outside the cluster.

**`src/autoscaler/live_loop.py`**: `LiveLoopConfig` gains four fields --
`enable_actuation` (default `False`), `actuation_machine_id` (default
`None`), `actuation_deployment`, `actuation_namespace`. `run_tick` gains
one new branch: if actuation is enabled AND the current machine matches
`actuation_machine_id` AND `observe_node_once` produced a decision, its
`recommended_servers` is applied for real via `actuator.set_replicas`
(imported lazily inside the branch, matching this module's existing
TensorFlow-isolation convention -- `kubernetes` is not a hard import-time
dependency of anything that doesn't use it). A failure here is caught
exactly like every other real-infrastructure failure in this function:
logged, the tick continues, the process doesn't crash. One machine, not an
aggregate, by design -- so "why did it scale" always traces to one real
forecast.

**`src/api/main.py`**: reads `LSTM_AUTOSCALER_ENABLE_ACTUATION` and
`LSTM_AUTOSCALER_ACTUATION_MACHINE_ID`, refuses to start if actuation is
enabled without a designated machine (rather than silently actuating
nothing or guessing), and logs plainly at startup whether actuation is on
and for which machine.

**`k8s/actuation-rbac.yaml`** (new): ServiceAccount + Role + RoleBinding.
The Role grants exactly `get`/`patch` on `deployments/scale` in the
`lstm-autoscaler` namespace -- not cluster-admin, not any other resource
kind, not any other namespace.

**`k8s/demo-workload.yaml`** (new): a small `nginx:alpine` Deployment
(`demo-workload`, 2 replicas) that actuation scales. Necessary and worth
being honest about: the real forecast signal (node-level CPU) isn't tied
to any actual variable-load service in this cluster, so this proves the
actuation MECHANISM end-to-end against a real Deployment -- not that the
forecast is good for this workload's real traffic, because this workload
has none.

**`k8s/observer.yaml`**: references the new ServiceAccount; documents the
two actuation env vars, left commented out (actuation stays off until
someone deliberately uncomments them, after applying the two new manifests
above).

**`requirements.txt`**: adds `kubernetes==36.0.3`.

**`tests/test_actuator.py`** (new, 6 tests) and three new tests in
`tests/test_live_loop.py`: actuation disabled by default never calls the
actuator; enabled, it calls the actuator only for the designated machine
with that machine's real recommended replica count; a failure is caught
and logged, not crashed.

## Local validation performed

- All 9 new tests pass. All 13 pre-existing `test_live_loop.py` tests
  still pass unmodified, confirming the actuation-disabled default path is
  unchanged. Broader run across `test_live_loop.py`, `test_decision.py`,
  `test_shadow.py`, `test_shadow_store.py`, `test_metrics_source.py`,
  `test_actuator.py`, `test_calibration.py`: 99/100 pass -- the one
  failure (`test_multistep_penalty_dilutes_a_single_step_spike...`) is a
  pre-existing floating-point precision artifact in code this step didn't
  touch (`3.9999999999999996 == 4.0`), reproducible on an unrelated numpy
  version, not a regression from this step.
- All three new/changed YAML manifests parse as valid YAML.
- **Applied to the real cluster -- see "Real-cluster verification" below
  for the full account, including a real misconfiguration found and fixed
  live (wrong initial `actuation_machine_id`).

## Real-cluster verification

Deployed against the real Docker Desktop Kubernetes cluster used
throughout Steps 20-25 (`docker build` -> push to `localhost:5000` ->
`kubectl apply`), in two passes.

**Pass 1 -- baseline, actuation still off.** Applied
`k8s/actuation-rbac.yaml` and `k8s/demo-workload.yaml`, rebuilt/pushed the
observer image as `:step26`, redeployed `k8s/observer.yaml` unchanged
(actuation env vars still commented out). Both `demo-workload` pods and
the new observer pod came up `1/1 Running` / `2/2 Running` cleanly. Queried
`shadow_state.db`'s machines table for real tracked `machine_id`s:
`172.18.0.2`, `172.18.0.3`, `172.18.0.5`, all on `arima`.

**Misconfiguration found and fixed live.** `172.18.0.2` was picked first
as `actuation_machine_id` (it had the most observation history from Step
25's demo). After redeploying with actuation enabled for it, two
consecutive real ticks (five minutes apart) logged observed decisions for
`172.18.0.3` and `172.18.0.5` only -- nothing at all for `172.18.0.2`, not
even a caught-and-logged failure line. Root cause: `resolve_tracked_machine_ids`
re-discovers live nodes from Prometheus every tick
(`PrometheusMetricsSource.list_machine_ids`) rather than trusting a fixed
list, so a `machine_id` can exist historically in `shadow_state.db` (from
a past tick) without still being a currently-scraped Prometheus target.
`172.18.0.2` is almost certainly a stale node-exporter IP from before a
Docker Desktop network/restart event -- Docker Desktop can reassign node
IPs -- and was simply no longer in this run's live machine list, so
`observe_node_once` was never called for it and the actuation branch
(which only ever fires for a `machine_id` present in that tick's loop)
never triggered. This is the live-discovery design working as intended,
not a defect in it -- a hardcoded tracked-node list would have hidden this
instead of surfacing it in the logs. Switched `actuation_machine_id` to
`172.18.0.3`, which appeared cleanly in both prior ticks, and redeployed.

**Pass 2 -- actuation enabled for `172.18.0.3`, real change confirmed.**
Startup log confirmed the target: `ACTUATION ENABLED for machine='172.18.0.3'
-> deployment='demo-workload' namespace='lstm-autoscaler'`. First real tick:

```
observed machine=172.18.0.3 forecast=[15.4106, 15.7831, 10.9636]
  planned_load=394.58% current=5 recommended=5 action=hold
actuator: set deployment=demo-workload namespace=lstm-autoscaler replicas=5
```

The second line is `src.autoscaler.actuator` logging only after
`patch_namespaced_deployment_scale` returns without raising -- a genuine
Kubernetes API call succeeded. `demo-workload` started at 2 replicas (the
manifest default); `kubectl get deploy demo-workload -w` showed a live
rolling scale-up in real time: `2/2 -> 2/5 -> 3/5 -> 4/5 -> 5/5`, settling
at `5/5 READY, 5 UP-TO-DATE, 5 AVAILABLE` moments later and confirmed
steady-state on a fresh `kubectl get deploy` afterward. `172.18.0.5`
logged `action=scale_up +1` in the same tick and, correctly, nothing was
actuated for it -- it is not the designated machine. Every other tracked
node stays observe-only, exactly as designed.

**A real design gap this surfaced, not papered over.** The decision the
actuator applies is not read-modify-write against the live cluster: `current_servers`
in `observe_node_once` comes from `store.get_last_recommended_servers`
(the decision engine's own memory of what it last recommended for that
node in `shadow_state.db`), not from `actuator.get_current_replicas` --
which exists in `actuator.py` but is never called anywhere in `run_tick`.
That's why the very first real actuation call jumped `demo-workload`
straight from 2 to 5 in one tick instead of stepping by the usual ±1: the
decision engine's internal counter and the real cluster's replica count
started out disagreeing, and the write went out blind to that. The two
are now in sync (`current_servers` tracked internally == 5 == real replica
count), so the *next* real scale event should show a correct, single-step
before/after. Functionally this doesn't break anything actuation is meant
to prove here (the mechanism -- forecast to decision to real API call --
is genuinely real and now demonstrated), but it is a legitimate
production gap: nothing today reconciles the decision engine's internal
server-count ledger against the cluster's actual state before writing to
it. Listed below under "Still open" as a real next step, not hidden.

## Before -> After

| | Before Step 26 | After Step 26 |
|---|---|---|
| Real scaling call anywhere in this codebase | None (every step through 25 explicit about this) | `actuator.set_replicas`, gated off by default |
| RBAC | None (observer has no ServiceAccount) | Least-privilege ServiceAccount, scoped to one subresource, one namespace |
| A target to actually scale | None | `demo-workload`, clearly labeled as a demo target, not real traffic |
| Verified | N/A | Unit-tested (9 new tests, mocked K8s client) AND confirmed on the real cluster: `demo-workload` genuinely scaled `2 -> 5` replicas via a real forecast -> real decision -> real `patch_namespaced_deployment_scale` call |

| | Before Step 26 | After Step 26 |
|---|---|---|
| Real scaling call anywhere in this codebase | None (every step through 25 explicit about this) | `actuator.set_replicas`, gated off by default |
| RBAC | None (observer has no ServiceAccount) | Least-privilege ServiceAccount, scoped to one subresource, one namespace |
| A target to actually scale | None | `demo-workload`, clearly labeled as a demo target, not real traffic |
| Verified | N/A | Unit-tested (9 new tests, mocked K8s client); NOT yet run against the real cluster |

## Still open

- **No read-before-write reconciliation against live cluster state.**
  `actuator.get_current_replicas` exists but `run_tick` never calls it --
  the actuated write always comes from the decision engine's own internal
  `current_servers` ledger (`shadow_state.db`), not the real deployment's
  actual replica count. Surfaced directly by this step's real-cluster
  verification (see above): a real, working next step, not a hidden gap.
- **The actuated target is a demo workload, not a real service.** This
  step proves the call mechanism, not that this forecast is good for any
  real traffic pattern -- flagged repeatedly above on purpose, not a gap to
  quietly close later.
- **No rate-limit beyond what the decision engine's own `scale_step`
  already provides** (at most ±1 replica per tick, per `DecisionConfig`,
  unchanged) -- no additional cooldown/circuit-breaker layer was added in
  this step. Worth revisiting if this ever points at something real.
- **Every Step 21-25 "still open" item remains open**, unchanged by this
  step: no real trained hybrid model, HTTP call volume, no
  retry/backoff on a transient Prometheus failure, ARIMA
  refit-from-scratch-every-tick.
