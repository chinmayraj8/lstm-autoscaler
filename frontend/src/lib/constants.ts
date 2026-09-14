// Matches src/dashboard/live_app.py's DEMO_DEPLOYMENT/DEMO_NAMESPACE and
// k8s/demo-workload.yaml -- the one real Deployment actuation ever targets.
export const DEMO_DEPLOYMENT = "demo-workload"
export const DEMO_NAMESPACE = "lstm-autoscaler"

// Matches live_loop.py's DEFAULT_TICK_SECONDS -- also returned live by
// GET /config, which is the source of truth; this is only the pre-fetch
// fallback used before that first response lands.
export const DEFAULT_TICK_SECONDS = 300
export const STALE_AFTER_SECONDS = DEFAULT_TICK_SECONDS * 3
