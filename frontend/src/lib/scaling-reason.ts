import { parseAction } from "@/lib/types"

/**
 * A human-readable reason string, computed ONLY from real fields the
 * backend already returns (planned_load_pct, current_servers, the
 * server_capacity_pct threshold from /config) -- decision.py's actual
 * rule (_decide_scaling) picks whichever of hold/scale_up/scale_down
 * minimizes a capacity-vs-load penalty, so "forecasted load vs current
 * capacity" is the real mechanism, not a gloss invented for the UI.
 */
export function buildScalingReason(
  action: string,
  plannedLoadPct: number,
  currentServers: number,
  capacityPct: number,
): string {
  const currentCapacity = currentServers * capacityPct
  const kind = parseAction(action).kind
  const load = plannedLoadPct.toFixed(1)
  const cap = currentCapacity.toFixed(0)
  if (kind === "scale_up") {
    return `Forecasted load (${load}%) exceeds current capacity (${cap}% at ${currentServers} replicas).`
  }
  if (kind === "scale_down") {
    return `Forecasted load (${load}%) is well under current capacity (${cap}% at ${currentServers} replicas).`
  }
  return `Forecasted load (${load}%) fits within current capacity (${cap}% at ${currentServers} replicas).`
}
