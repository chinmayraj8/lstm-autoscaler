# Step 23 (Stage 5 of 3+2): minimal in-cluster observer runner
Date: 2026-09-03
Status: done (scoped — see "Still open")

Part 5 of the "make the shadow harness watch real infrastructure" project,
and its final stage as scoped. Steps 21-22 built a real Prometheus
connector and a live forecasting loop, both only ever run locally (a
laptop, port-forwarded to the cluster, staying on). This step packages
just enough to run that loop continuously **inside** the cluster instead:
a Dockerfile for the observer service (Stage 4's scheduler + the FastAPI
`/shadow/*` endpoints) and a minimal Kubernetes manifest (Namespace + PVC +
Deployment + Service) that runs it talking to Prometheus over in-cluster
DNS directly.

This is explicitly **not** the broader "Part 2" Docker/Kubernetes
deployment work this project's index has listed as not-started since
before Step 18 — no HPA, no `/forecast` model serving from this image, no
Ingress, no CI/CD, no multi-environment config. Scoped to exactly what
Stage 4 needs to run unattended: stay up, poll Prometheus, log decisions,
survive a restart.

**No scale-up/scale-down call to real infrastructure exists anywhere in
this codebase, and this step doesn't change that boundary.** This step
doesn't even add a way to reach one — the image contains no code path that
calls anything but Prometheus's read-only query endpoint and this
service's own SQLite file.

## What changed

**`Dockerfile`** (new, project root): `python:3.11-slim` base,
`requirements.txt` installed, `src/` copied in, runs
`uvicorn src.api.main:app`. Deliberately excludes (via `.dockerignore`,
new) the training CSV, `lstm_model.keras`, `experiments/`, `docs/`,
notebooks, and this project's own dev tooling — none of it is needed to
run the observer path. `LSTM_AUTOSCALER_SKIP_LSTM_MODEL` and
`LSTM_AUTOSCALER_PROMETHEUS_URL` are set in the K8s manifest, not baked
into the image, so the same image can be pointed at a different
Prometheus URL (or run with the LSTM model available) without a rebuild.

**`requirements.txt`** (new, project root): the runtime dependency pins
for this image, matched to what's installed in this project's own dev
venv — **with one deliberate substitution**: `tensorflow` (the standard
PyPI package) instead of the `tensorflow-macos`/`tensorflow-metal` pair
README.md's local setup instructions use, since those are Apple
Silicon-only and won't install in a Linux container.
`main.py` imports `_build_lstm_model` at module level (unconditionally,
regardless of `LSTM_AUTOSCALER_SKIP_LSTM_MODEL` — see Step 22's progress
doc and `tests/test_api_shadow.py`'s own docstring on this), so
TensorFlow remains a real import-time dependency of this service even
though the observer deployment never actually builds or loads a model.

**`k8s/observer.yaml`** (new): a single multi-document manifest —
- `Namespace lstm-autoscaler` — separate from `monitoring` on purpose;
  this is a new, independent workload, not part of the kube-prometheus-stack
  release.
- `PersistentVolumeClaim observer-shadow-state` (1Gi, default
  StorageClass, `ReadWriteOnce`) — see "The PVC decision" below.
- `Deployment lstm-autoscaler-observer` — 1 replica (see PVC note on why
  not more), the PVC mounted at `/data`,
  `LSTM_AUTOSCALER_SHADOW_DB=/data/shadow_state.db`,
  `LSTM_AUTOSCALER_PROMETHEUS_URL` set to the confirmed working in-cluster
  URL, `LSTM_AUTOSCALER_SKIP_LSTM_MODEL=1`, readiness/liveness probes on
  `/health`, conservative resource requests/limits. No ServiceAccount/RBAC
  — this service only ever makes outbound HTTP GETs to Prometheus, never
  talks to the Kubernetes API.
- `Service lstm-autoscaler-observer` — ClusterIP, `:80` → `:8000`, so
  `/shadow/*` and `/health` are reachable in-cluster (or via
  `kubectl port-forward svc/lstm-autoscaler-observer`) without needing a
  laptop kept on and port-forwarded to Prometheus directly, which is the
  actual problem this stage was asked to solve.

## The PVC decision

This stage's own instructions asked for an explicit choice, not a default:
add a PVC now, or accept and flag the data-loss limitation. **A PVC was
added.** Reasoning: it's genuinely minimal (one `PersistentVolumeClaim`
object plus a `volumeMounts`/`volumes` pair, not a new subsystem), and the
alternative — losing every banked shadow window and observed decision on
every pod reschedule — would undermine the specific thing this multi-stage
project has been building toward ("watch real decisions accumulate").
`replicas: 1` is deliberate and load-bearing alongside it: SQLite (Step
19) is single-writer-friendly, not built for concurrent multi-pod writes
to the same file — raising replica count against this same PVC path would
need a different store first, not just a bigger claim. That trade-off is
flagged directly in the manifest's own comments, not just here.

## Local validation performed (no real cluster access)

- **`docker build` on this Dockerfile succeeds**, run in this environment
  (a local Docker install) — `pip install -r requirements.txt` resolves
  end-to-end (all pins compatible together, including `tensorflow==2.16.2`
  against `numpy==1.26.4`), image built and tagged
  (`lstm-autoscaler-observer:test`, ~2.4GB). Not just "the Dockerfile looks
  right."
- **The built image was actually run** (`docker run`, this environment,
  not the target cluster) and exercised two ways, both torn down after:
  - `LSTM_AUTOSCALER_SKIP_LSTM_MODEL=1` alone: container starts, logs the
    expected skip message, `GET /health` returns 200 with
    `lstm_model_loaded: false`, `live_loop_running: false`,
    `GET /shadow/{id}/decisions` returns the expected empty-list shape.
  - Same, plus `LSTM_AUTOSCALER_PROMETHEUS_URL` pointed at a deliberately
    unresolvable host: the live-loop background thread starts
    (`live_loop_running: true`), its first tick's DNS failure is caught
    and logged (a full traceback appears in the container log, exactly as
    `run_scheduler_loop`'s docstring says it should), and — the actual
    point of this check — **the service stays up and `/health` keeps
    returning 200** despite the loop failing every tick. This is the
    closest this environment can get to proving the "one bad tick doesn't
    kill the process" claim without a real Prometheus to point at.
  - This is NOT the same as running the container against the real target
    cluster — no network access to it from this environment, so the
    ACTUAL query against real node-exporter data was never exercised
    end-to-end, only the failure path and the skip-model path.
- **`kubectl apply --dry-run=client -f k8s/observer.yaml` succeeds** — all
  four objects parse and pass client-side structural validation against
  this environment's installed `kubectl`. Server-side dry-run against a
  real API server (which would catch schema issues client-side validation
  can't, e.g. immutable-field or admission-webhook problems) was not run
  against the actual target cluster — no access to it — and wasn't run
  against a local test cluster either, to avoid creating real (if
  ultimately harmless) objects on infrastructure outside this project's
  scope without asking first.

## Before → After

| | Before Step 23 | After Step 23 |
|---|---|---|
| Runs where | A laptop, port-forwarded, staying on | Can run in-cluster: `Dockerfile` + `k8s/observer.yaml` |
| Shadow state across a restart | Survives a laptop restart (SQLite file on local disk, Step 19) | Survives a pod restart/reschedule too (PVC-backed, this step) |
| Deployment artifacts | None | `Dockerfile`, `requirements.txt`, `.dockerignore`, `k8s/observer.yaml` |
| Verified locally | N/A | `docker build` succeeds; the built image actually runs and serves `/health`/`/shadow/*` correctly, including surviving a live-loop tick failure without crashing; manifest passes `kubectl` client-side validation |
| Verified against the real target cluster | N/A | Not done — no access to it from this environment |

## Impact

### This closes the loop this 5-stage project set out to close — with one real caveat

Stage 3 named the real system; Stage 4 built the loop that watches it;
this stage is what lets that loop actually run unattended against it,
instead of depending on this conversation's own local environment staying
up. The one honest caveat: everything here has been validated as far as
this environment can validate it (a Dockerfile that builds, a manifest
that parses and passes client-side checks) — nobody has run
`kubectl apply -f k8s/observer.yaml` against the actual cluster with
kube-prometheus-stack running, because this environment has no access to
it. The first real application to the real cluster is still a step the
user needs to take and watch.

## Still open

- **Not applied to the real target cluster.** No access to it from this
  environment — `kubectl apply -f k8s/observer.yaml` (after building and
  pushing the image, and editing the manifest's `image:` field) is the
  next action, and it's the user's to take and verify.
- **`image:` is a placeholder** (`lstm-autoscaler-observer:latest`) — needs
  building and pushing to a registry the target cluster can actually pull
  from, then editing the manifest to match.
- **PVC's `storageClassName` is left unset** (cluster default) — untested
  against whatever the real cluster's default StorageClass actually
  provisions; may need setting explicitly depending on that cluster's
  setup.
- **No backup/snapshot policy on the PVC.** A lost/corrupted volume still
  loses shadow history — a PVC solves "pod reschedule," not "storage
  failure."
- **Server-side manifest validation against a real API server wasn't
  run** (see "Local validation performed" above) — only client-side
  structural validation.
- **Every "still open" item from Steps 21-22 remains open**, unchanged by
  this step: HTTP call volume from the instant-query-only Prometheus
  connector, no retry/backoff on a transient Prometheus failure, ARIMA
  refit-from-scratch-every-tick on a short window not validated against
  real data, and — the largest one — no pretrained residual-hybrid model
  exists anywhere in this codebase, so the hybrid re-validation path stays
  wired-but-inert once this actually runs for real.
- **No scale-up/scale-down call to real infrastructure exists anywhere in
  this codebase.** This step doesn't change that boundary and doesn't move
  it any closer without being asked to. Actuation remains a separate,
  not-yet-approved step, exactly as this whole 5-stage project was scoped
  from the start.
