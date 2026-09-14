import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react"

/**
 * Client-side settings only -- these parameterize which API calls the
 * frontend makes (which machine, which base URLs, how often to poll).
 * They are NOT backend configuration; the actual scaling thresholds live
 * in ScalingConfigResponse (/config) and are read-only. Persisted to
 * localStorage the same way the old Streamlit sidebar's inputs lived in
 * session state -- per-browser, not shared.
 */
export interface Settings {
  observerUrl: string
  apiToken: string
  prometheusUrl: string
  machineId: string
  refreshIntervalSeconds: number
  autoRefresh: boolean
  showCpuTrend: boolean
}

const DEFAULT_SETTINGS: Settings = {
  observerUrl: "http://localhost:8000",
  // Empty by default -- an unauthenticated local observer (the default,
  // LSTM_AUTOSCALER_API_TOKEN unset) works with no header at all. Sent as
  // `Authorization: Bearer <token>` only when non-empty (see api.ts).
  apiToken: "",
  // Unlike observerUrl (which the BROWSER calls directly, over the
  // port-forward), this URL is used server-side -- the observer pod
  // itself makes the Prometheus call from inside the cluster (see
  // /metrics/cpu, /machines in src/api/main.py), so it needs Prometheus's
  // in-cluster Service DNS name, not a laptop-side port-forward address.
  prometheusUrl: "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090",
  machineId: "172.18.0.3",
  refreshIntervalSeconds: 30,
  autoRefresh: true,
  showCpuTrend: true,
}

const STORAGE_KEY = "lstm-autoscaler.settings.v1"

function loadSettings(): Settings {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return DEFAULT_SETTINGS
    const parsed = JSON.parse(raw)
    return { ...DEFAULT_SETTINGS, ...parsed }
  } catch {
    return DEFAULT_SETTINGS
  }
}

interface SettingsContextValue {
  settings: Settings
  setSettings: (patch: Partial<Settings>) => void
  resetSettings: () => void
}

const SettingsContext = createContext<SettingsContextValue | null>(null)

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettingsState] = useState<Settings>(loadSettings)

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings))
    } catch {
      // Private-mode/blocked storage -- settings just won't persist across reloads.
    }
  }, [settings])

  const value = useMemo<SettingsContextValue>(
    () => ({
      settings,
      setSettings: (patch) => setSettingsState((prev) => ({ ...prev, ...patch })),
      resetSettings: () => setSettingsState(DEFAULT_SETTINGS),
    }),
    [settings],
  )

  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>
}

export function useSettings(): SettingsContextValue {
  const ctx = useContext(SettingsContext)
  if (!ctx) throw new Error("useSettings must be used within SettingsProvider")
  return ctx
}
