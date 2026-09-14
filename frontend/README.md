# LSTM Autoscaler — Ops Console

React + TypeScript + Vite frontend for the LSTM autoscaler dashboard.
Replaces `src/dashboard/live_app.py` as the production UI; the Python
backend (`src/api/main.py` and the `src/autoscaler` package) is unchanged.

## Stack

React, TypeScript, Vite, Tailwind CSS, shadcn/ui (Base UI), Apache
ECharts, TanStack Query.

## Running locally

```bash
npm install
npm run dev
```

Requires the observer API reachable (default `http://localhost:8000`,
configurable in Settings) — see the project root README for how to
port-forward it from the cluster.

## Scripts

- `npm run dev` — dev server (default port 5173)
- `npm run build` — typecheck (`tsc -b`) + production build
- `npm run lint` — oxlint
