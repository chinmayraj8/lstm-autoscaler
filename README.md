# LSTM Autoscaler

Cost-aware predictive autoscaling for Kubernetes. An ARIMA baseline and a
residual-hybrid (ARIMA + LSTM) forecaster are evaluated head-to-head in
**shadow mode** against a real 2-node Kubernetes cluster's real Prometheus
telemetry, with an explicit statistical gate deciding which forecaster
actually drives scaling. Real actuation (patching a Deployment's replica
count) is wired and can be enabled, but is off by default.

This started as offline research — ARIMA and LSTM forecasting tuned and
validated against the Alibaba Cluster Trace 2018 — and has since grown
into a live system: a FastAPI service running in the cluster, a durable
shadow-evaluation store, and a React ops console. The forecasting and
decision-engine logic in `src/autoscaler/` is that same validated
methodology, carried forward unchanged into the live path.

## Current status

- **Live**: a 2-node Docker Desktop Kubernetes cluster, `kube-prometheus-stack`
  for real node CPU%, and an observer service ticking every 5 minutes.
- **Forecasters**: ARIMA is the current live default. A residual-hybrid
  LSTM model is trained for both tracked nodes (`models/hybrid_residual/`)
  and is being scored against ARIMA in shadow mode — it has not yet won
  enough consecutive shadow windows to be promoted.
- **Actuation**: real (patches `demo-workload`'s replica count via the
  Kubernetes API), gated behind an explicit opt-in, currently enabled for
  one designated node. See `src/autoscaler/actuator.py` and
  `k8s/observer.yaml`'s comments for the exact scope and RBAC.
- **Frontend**: a React ops console (`frontend/`) is the production UI.
  The previous Streamlit dashboard (`src/dashboard/live_app.py`) has been
  retired now that the new frontend is verified against the real cluster.
- **CI**: GitHub Actions runs lint + the test suite on every push (see
  `.github/workflows/ci.yml`).

## Architecture

```
src/autoscaler/   Core pipeline: data prep, ARIMA baseline, the residual-
                  hybrid mechanism, the scaling-decision engine, the
                  shadow-evaluation harness + its durable SQLite store,
                  the live 5-minute tick loop, and the real Kubernetes
                  actuation client.
src/api/          FastAPI service (main.py) exposing /health, /forecast,
                  and the /shadow/* + /replicas + /metrics + /machines +
                  /config + /actuation endpoints. Runs the live loop as a
                  background thread when deployed.
src/dashboard/    app.py — a replay of the original offline Phase 1/2
                  Reactive-vs-LSTM simulation on the historical Alibaba
                  trace, kept for reference. (live_app.py, the Streamlit
                  live-cluster dashboard, has been retired — see below.)
frontend/         React + TypeScript + Vite + Tailwind + shadcn/ui +
                  ECharts + TanStack Query ops console. The production
                  dashboard: system status, forecasting, scaling
                  decisions, model comparison, events, settings.
k8s/              Manifests for the observer Deployment/Service, the
                  actuation RBAC, the demo workload it scales, and a
                  synthetic load generator.
tests/            pytest suite covering the autoscaler package and the
                  API's /shadow endpoints.
scripts/          One-off/offline tools: bootstrapping a shadow window
                  from real history, training a real hybrid model per node.
models/hybrid_residual/
                  The real trained per-node residual-hybrid LSTM models
                  (.keras) the live loop actually loads.
```

## Getting started

### Backend

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

pytest tests/ -v
```

Run the API locally, skipping the optional legacy `/forecast` endpoint
(its LSTM point-forecast model isn't part of this repository —
`/shadow/*`, the live loop, and everything else are unaffected):

```bash
LSTM_AUTOSCALER_SKIP_LSTM_MODEL=1 uvicorn src.api.main:app --reload
```

Or reach the real deployed observer in the cluster:

```bash
kubectl port-forward svc/lstm-autoscaler-observer -n lstm-autoscaler 8000:80
kubectl port-forward svc/kube-prometheus-stack-prometheus -n monitoring 9090:9090
```

### Frontend

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173, configure the observer URL in Settings
```

### Deploying to the cluster

```bash
docker build -t lstm-autoscaler-observer:<tag> .
docker tag lstm-autoscaler-observer:<tag> localhost:5000/lstm-autoscaler-observer:<tag>
docker push localhost:5000/lstm-autoscaler-observer:<tag>
# update the image tag in k8s/observer.yaml, then apply it first --
# it creates the lstm-autoscaler namespace the other manifests depend on:
kubectl apply -f k8s/observer.yaml
kubectl apply -f k8s/actuation-rbac.yaml -f k8s/demo-workload.yaml -f k8s/load-generator.yaml
```

See `k8s/observer.yaml`'s own comments for the Docker-Desktop-specific
local-registry step and every environment variable it reads.
