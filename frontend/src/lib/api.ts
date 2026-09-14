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

/**
 * Whether the most recently completed request got a real 401 back.
 * Tracked independently of TanStack Query's own cache/retry state (not
 * read off `query.state.error`) -- deliberately a tiny, direct pub-sub
 * updated the instant `getJson` sees a 401/non-401 response, so the "not
 * authenticated" indicator (`useUnauthorized`) reflects the real server
 * answer immediately regardless of any particular query's retry timing.
 */
let _unauthorized = false
const _unauthorizedListeners = new Set<() => void>()

function _setUnauthorized(value: boolean) {
  if (_unauthorized === value) return
  _unauthorized = value
  for (const listener of _unauthorizedListeners) listener()
}

export function getUnauthorized(): boolean {
  return _unauthorized
}

export function subscribeUnauthorized(listener: () => void): () => void {
  _unauthorizedListeners.add(listener)
  return () => _unauthorizedListeners.delete(listener)
}

/** A 404 from /shadow/{id} and /shadow/{id}/windows means "no shadow state
 * for this machine yet" -- an ordinary state (main.py's own documented
 * convention), not a failure. A 422 from /forecast/confidence means "not
 * enough real history to fit ARIMA yet" -- also ordinary. Callers that
 * pass `treat404AsNull`/`treat422AsNull` get `null` back instead of a
 * thrown ApiError for those specific statuses. A 401 (missing/wrong
 * bearer token) is never treated as null -- it always throws, AND flips
 * `_unauthorized` so a wrong/missing token fails visibly instead of just
 * showing empty charts.
 *
 * `trackAuth` (default true) gates whether THIS call's status is allowed
 * to touch the shared `_unauthorized` flag at all. /health is the one
 * caller that passes `trackAuth: false` -- it's deliberately
 * unauthenticated server-side (see main.py), so it always returns 200
 * regardless of whether the token is valid. Without this exclusion, a
 * /health poll settling AFTER a real 401 from a protected endpoint would
 * flip `_unauthorized` back to false and hide an actual bad/missing
 * token -- confirmed as a real bug, not hypothetical: the "NOT
 * AUTHENTICATED" badge disappeared on screen while /machines, /config,
 * and /shadow/* kept 401ing in the server's own logs the whole time. */
async function getJson<T>(
  url: string,
  token: string,
  opts: { treat404AsNull?: boolean; treat422AsNull?: boolean; trackAuth?: boolean } = {},
): Promise<T | null> {
  const headers: HeadersInit = token ? { Authorization: `Bearer ${token}` } : {}
  const res = await fetch(url, { headers })
  if (opts.trackAuth !== false) _setUnauthorized(res.status === 401)
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

export function createApiClient(observerUrl: string, token: string) {
  const base = observerUrl.replace(/\/+$/, "")

  return {
    // /health is deliberately unauthenticated server-side too (see
    // main.py) but sending the header anyway is harmless. trackAuth:
    // false so this call's always-200 response never masks a real 401
    // from a protected endpoint (see getJson's own comment).
    health: () => getJson<HealthResponse>(`${base}/health`, token, { trackAuth: false }),

    shadowStatus: (machineId: string) =>
      getJson<ShadowStatusResponse>(`${base}/shadow/${encodeURIComponent(machineId)}`, token, {
        treat404AsNull: true,
      }),

    shadowWindows: (machineId: string) =>
      getJson<ShadowWindowListResponse>(`${base}/shadow/${encodeURIComponent(machineId)}/windows`, token, {
        treat404AsNull: true,
      }),

    shadowDecisions: (machineId: string, limit = 200) =>
      getJson<ObservedDecisionListResponse>(
        `${base}/shadow/${encodeURIComponent(machineId)}/decisions${qs({ limit })}`,
        token,
      ),

    replicas: (deployment: string, namespace: string) =>
      getJson<ReplicasResponse>(`${base}/replicas/${encodeURIComponent(deployment)}${qs({ namespace })}`, token),

    cpuMetrics: (machineId: string, prometheusUrl: string, hours = 2) =>
      getJson<CpuMetricsResponse>(
        `${base}/metrics/cpu${qs({ machine_id: machineId, prometheus_url: prometheusUrl, hours })}`,
        token,
      ),

    machines: (prometheusUrl: string) =>
      getJson<MachinesResponse>(`${base}/machines${qs({ prometheus_url: prometheusUrl })}`, token),

    scalingConfig: () => getJson<ScalingConfigResponse>(`${base}/config`, token),

    forecastConfidence: (machineId: string, prometheusUrl: string, fitHours = 3, confidenceLevel = 0.95) =>
      getJson<ForecastConfidenceResponse>(
        `${base}/forecast/confidence${qs({
          machine_id: machineId,
          prometheus_url: prometheusUrl,
          fit_hours: fitHours,
          confidence_level: confidenceLevel,
        })}`,
        token,
        { treat422AsNull: true },
      ),

    actuationStatus: () => getJson<ActuationStatusResponse>(`${base}/actuation/status`, token),
  }
}

export type ApiClient = ReturnType<typeof createApiClient>
