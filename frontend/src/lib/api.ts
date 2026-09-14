import type {
  ActuationStatusResponse,
  CpuMetricsResponse,
  ForecastConfidenceResponse,
  HealthResponse,
  MachinesResponse,
  ObservedDecisionListResponse,
  ReplicasResponse,
  ScalingConfigResponse,
  ShadowStatusResponse,
  ShadowWindowListResponse,
} from "./types"

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

/** A 404 from /shadow/{id} and /shadow/{id}/windows means "no shadow state
 * for this machine yet" -- an ordinary state (main.py's own documented
 * convention), not a failure. A 422 from /forecast/confidence means "not
 * enough real history to fit ARIMA yet" -- also ordinary. Callers that
 * pass `treat404AsNull`/`treat422AsNull` get `null` back instead of a
 * thrown ApiError for those specific statuses. */
async function getJson<T>(
  url: string,
  opts: { treat404AsNull?: boolean; treat422AsNull?: boolean } = {},
): Promise<T | null> {
  const res = await fetch(url)
  if (res.status === 404 && opts.treat404AsNull) return null
  if (res.status === 422 && opts.treat422AsNull) return null
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body?.detail ?? detail
    } catch {
      // non-JSON error body -- keep statusText
    }
    throw new ApiError(res.status, detail)
  }
  return (await res.json()) as T
}

function qs(params: Record<string, string | number | boolean | undefined>): string {
  const usp = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== "") usp.set(k, String(v))
  }
  const s = usp.toString()
  return s ? `?${s}` : ""
}

export function createApiClient(observerUrl: string) {
  const base = observerUrl.replace(/\/+$/, "")

  return {
    health: () => getJson<HealthResponse>(`${base}/health`),

    shadowStatus: (machineId: string) =>
      getJson<ShadowStatusResponse>(`${base}/shadow/${encodeURIComponent(machineId)}`, {
        treat404AsNull: true,
      }),

    shadowWindows: (machineId: string) =>
      getJson<ShadowWindowListResponse>(`${base}/shadow/${encodeURIComponent(machineId)}/windows`, {
        treat404AsNull: true,
      }),

    shadowDecisions: (machineId: string, limit = 200) =>
      getJson<ObservedDecisionListResponse>(
        `${base}/shadow/${encodeURIComponent(machineId)}/decisions${qs({ limit })}`,
      ),

    replicas: (deployment: string, namespace: string) =>
      getJson<ReplicasResponse>(`${base}/replicas/${encodeURIComponent(deployment)}${qs({ namespace })}`),

    cpuMetrics: (machineId: string, prometheusUrl: string, hours = 2) =>
      getJson<CpuMetricsResponse>(
        `${base}/metrics/cpu${qs({ machine_id: machineId, prometheus_url: prometheusUrl, hours })}`,
      ),

    machines: (prometheusUrl: string) =>
      getJson<MachinesResponse>(`${base}/machines${qs({ prometheus_url: prometheusUrl })}`),

    scalingConfig: () => getJson<ScalingConfigResponse>(`${base}/config`),

    forecastConfidence: (machineId: string, prometheusUrl: string, fitHours = 3, confidenceLevel = 0.95) =>
      getJson<ForecastConfidenceResponse>(
        `${base}/forecast/confidence${qs({
          machine_id: machineId,
          prometheus_url: prometheusUrl,
          fit_hours: fitHours,
          confidence_level: confidenceLevel,
        })}`,
        { treat422AsNull: true },
      ),

    actuationStatus: () => getJson<ActuationStatusResponse>(`${base}/actuation/status`),
  }
}

export type ApiClient = ReturnType<typeof createApiClient>
