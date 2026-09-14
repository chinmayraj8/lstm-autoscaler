/**
 * Types mirror src/api/main.py's Pydantic response models field-for-field.
 * Nothing here is invented -- if a field isn't in this list, the backend
 * doesn't expose it yet.
 */

export interface HealthResponse {
  status: string
  lstm_model_loaded: boolean
  live_loop_running: boolean
  machine_id: string | null
  model_path: string
  lookback_steps: number
  lookback_minutes: number
  horizon_steps: number
  horizon_minutes: number
  started_at: string
}

export type Forecaster = "arima" | "hybrid"

export interface CumulativeSummary {
  n_windows: number
  arima_cost_mean?: number
  arima_cost_total?: number
  arima_cost_std?: number
  arima_sla_mean_pct?: number
  hybrid_cost_mean?: number
  hybrid_cost_total?: number
  hybrid_cost_std?: number
  hybrid_sla_mean_pct?: number
}

export interface AssignmentChange {
  timestamp: string
  old_forecaster: Forecaster
  new_forecaster: Forecaster
  verdict: string
  evidence?: string
}

export interface ShadowStatusResponse {
  machine_id: string
  current_forecaster: Forecaster
  last_evaluated_at: string | null
  n_banked_windows: number
  cumulative: CumulativeSummary
  assignment_history: AssignmentChange[]
}

export interface ShadowWindow {
  window_start: string
  window_end: string
  arima_cost: number
  arima_sla_pct: number
  hybrid_cost: number
  hybrid_sla_pct: number
}

export interface ShadowWindowListResponse {
  machine_id: string
  windows: ShadowWindow[]
}

export interface ObservedDecision {
  observed_at: string
  forecaster: Forecaster
  forecast_cpu_pct: number[]
  planned_load_pct: number
  current_servers: number
  recommended_servers: number
  action: string // "hold" | "scale_up +N" | "scale_down -N"
}

export interface ObservedDecisionListResponse {
  machine_id: string
  decisions: ObservedDecision[]
}

export interface ReplicasResponse {
  deployment: string
  namespace: string
  replicas: number
  checked_at: string
}

export interface CpuReading {
  timestamp: string
  cpu_pct: number
}

export interface CpuMetricsResponse {
  machine_id: string
  prometheus_url: string
  resample_minutes: number
  readings: CpuReading[]
}

export interface MachinesResponse {
  prometheus_url: string
  machines: string[]
}

export interface ScalingConfigResponse {
  server_capacity_pct: number
  min_servers: number
  max_servers: number
  scale_step: number
  over_prov_weight: number
  under_prov_weight: number
  safety_margin: number
  demand_scale: number
  tick_seconds: number
}

export interface ActuationStatusResponse {
  enabled: boolean
  machine_id: string | null
  deployment: string | null
  namespace: string | null
  circuit_breaker_threshold: number | null
  circuit_breaker_cooldown_seconds: number | null
  consecutive_failures: number
  circuit_open: boolean
  circuit_opened_at: string | null
  cooldown_remaining_seconds: number | null
}

/** Parsed form of ObservedDecision.action -- backend sends "hold" |
 * "scale_up +N" | "scale_down -N" as one string; this is the only place
 * that string gets parsed. */
export interface ParsedAction {
  kind: "hold" | "scale_up" | "scale_down"
  delta: number
}

export function parseAction(action: string): ParsedAction {
  if (action.startsWith("scale_up")) {
    const n = Number(action.split("+")[1] ?? 1)
    return { kind: "scale_up", delta: Number.isFinite(n) ? n : 1 }
  }
  if (action.startsWith("scale_down")) {
    const n = Number(action.split("-")[1] ?? 1)
    return { kind: "scale_down", delta: Number.isFinite(n) ? -n : -1 }
  }
  return { kind: "hold", delta: 0 }
}
