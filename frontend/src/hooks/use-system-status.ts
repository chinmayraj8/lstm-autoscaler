import { useHealth, useShadowDecisions } from "@/hooks/queries"
import { STALE_AFTER_SECONDS } from "@/lib/constants"

export interface SystemStatus {
  isLoading: boolean
  isStale: boolean
  liveLoopRunning: boolean
  note: string
  lastTickAt: string | null
}

/** Byte-for-byte port of live_app.py's staleness logic (lines ~300-321):
 * live_loop_running (from /health) + how recent the most recent decision
 * actually is -- the decisions feed is the genuine "is data flowing right
 * now" signal, not last_evaluated_at (a ~24h/30d shadow-window
 * re-evaluation stamp). */
export function useSystemStatus(machineId: string): SystemStatus {
  const health = useHealth()
  const decisions = useShadowDecisions(machineId, 1)

  if (health.isLoading || decisions.isLoading) {
    return { isLoading: true, isStale: true, liveLoopRunning: false, note: "", lastTickAt: null }
  }

  const liveLoopRunning = health.data?.live_loop_running ?? false
  const lastDecision = decisions.data?.decisions.at(-1)
  const lastTickAt = lastDecision?.observed_at ?? null

  let isStale = true
  let note = "no ticks observed yet"

  if (liveLoopRunning && lastTickAt) {
    const ageSeconds = (Date.now() - new Date(lastTickAt).getTime()) / 1000
    isStale = ageSeconds > STALE_AFTER_SECONDS
    note = `last tick ${Math.floor(ageSeconds / 60)}m ago`
  } else if (!liveLoopRunning) {
    note = "live loop not running on the observer"
  }

  return { isLoading: false, isStale, liveLoopRunning, note, lastTickAt }
}
