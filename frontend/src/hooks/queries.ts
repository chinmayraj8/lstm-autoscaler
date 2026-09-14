import { useQuery } from "@tanstack/react-query"
import { useApi } from "@/hooks/use-api"
import { DEMO_DEPLOYMENT, DEMO_NAMESPACE } from "@/lib/constants"
import { useSettings } from "@/lib/settings"

/** Shared polling cadence -- mirrors the old dashboard's "Auto-refresh" +
 * "Refresh interval (s)" sidebar controls exactly. `false` disables
 * TanStack Query's own interval polling (a manual refetch still works). */
function useRefetchInterval() {
  const { settings } = useSettings()
  return settings.autoRefresh ? settings.refreshIntervalSeconds * 1000 : false
}

export function useHealth() {
  const api = useApi()
  const refetchInterval = useRefetchInterval()
  return useQuery({
    queryKey: ["health", api],
    queryFn: () => api.health(),
    refetchInterval,
  })
}

export function useShadowStatus(machineId: string) {
  const api = useApi()
  const refetchInterval = useRefetchInterval()
  return useQuery({
    queryKey: ["shadow-status", api, machineId],
    queryFn: () => api.shadowStatus(machineId),
    refetchInterval,
  })
}

export function useShadowWindows(machineId: string) {
  const api = useApi()
  const refetchInterval = useRefetchInterval()
  return useQuery({
    queryKey: ["shadow-windows", api, machineId],
    queryFn: () => api.shadowWindows(machineId),
    refetchInterval,
  })
}

export function useShadowDecisions(machineId: string, limit = 200) {
  const api = useApi()
  const refetchInterval = useRefetchInterval()
  return useQuery({
    queryKey: ["shadow-decisions", api, machineId, limit],
    queryFn: () => api.shadowDecisions(machineId, limit),
    refetchInterval,
  })
}

export function useReplicas(deployment = DEMO_DEPLOYMENT, namespace = DEMO_NAMESPACE) {
  const api = useApi()
  const refetchInterval = useRefetchInterval()
  return useQuery({
    queryKey: ["replicas", api, deployment, namespace],
    queryFn: () => api.replicas(deployment, namespace),
    refetchInterval,
    retry: 1,
  })
}

export function useCpuMetrics(machineId: string, hours = 2) {
  const api = useApi()
  const { settings } = useSettings()
  const refetchInterval = useRefetchInterval()
  return useQuery({
    queryKey: ["cpu-metrics", api, machineId, settings.prometheusUrl, hours],
    queryFn: () => api.cpuMetrics(machineId, settings.prometheusUrl, hours),
    refetchInterval,
    enabled: settings.showCpuTrend,
    retry: 1,
  })
}

export function useMachines() {
  const api = useApi()
  const { settings } = useSettings()
  return useQuery({
    queryKey: ["machines", api, settings.prometheusUrl],
    queryFn: () => api.machines(settings.prometheusUrl),
    retry: 1,
    staleTime: 60_000,
  })
}

export function useScalingConfig() {
  const api = useApi()
  return useQuery({
    queryKey: ["scaling-config", api],
    queryFn: () => api.scalingConfig(),
    staleTime: 60_000,
  })
}

export function useActuationStatus() {
  const api = useApi()
  const refetchInterval = useRefetchInterval()
  return useQuery({
    queryKey: ["actuation-status", api],
    queryFn: () => api.actuationStatus(),
    refetchInterval,
  })
}
