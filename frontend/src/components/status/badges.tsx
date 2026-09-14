import { cn } from "@/lib/utils"
import { parseAction, type ObservedForecaster } from "@/lib/types"

export function ForecasterBadge({ forecaster }: { forecaster: ObservedForecaster }) {
  if (forecaster === "reactive_fallback") {
    return (
      <span
        className={cn(
          "inline-flex items-center rounded-sm border px-1.5 py-0.5 font-mono text-[11px] font-medium",
          "border-status-warning/30 bg-status-warning/10 text-status-warning",
        )}
      >
        reactive fallback
      </span>
    )
  }
  const isHybrid = forecaster === "hybrid"
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-sm border px-1.5 py-0.5 font-mono text-[11px] font-medium",
        isHybrid
          ? "border-series-hybrid/30 bg-series-hybrid/10 text-series-hybrid"
          : "border-series-arima/30 bg-series-arima/10 text-series-arima",
      )}
    >
      {forecaster}
    </span>
  )
}

export function ActionBadge({ action }: { action: string }) {
  const parsed = parseAction(action)
  const styles: Record<typeof parsed.kind, string> = {
    scale_up: "border-status-healthy/30 bg-status-healthy/10 text-status-healthy",
    scale_down: "border-primary/30 bg-primary/10 text-primary",
    hold: "border-border bg-secondary/40 text-muted-foreground",
  }
  const labels: Record<typeof parsed.kind, string> = {
    scale_up: "Scale up",
    scale_down: "Scale down",
    hold: "Hold",
  }
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-sm border px-1.5 py-0.5 font-mono text-[11px] font-medium",
        styles[parsed.kind],
      )}
    >
      {labels[parsed.kind]}
    </span>
  )
}

export function VerdictBadge({ verdict }: { verdict: string }) {
  const lower = verdict.toLowerCase()
  const tone = lower.includes("confirmed cheaper")
    ? "border-status-healthy/30 bg-status-healthy/10 text-status-healthy"
    : lower.includes("more expensive")
      ? "border-status-critical/30 bg-status-critical/10 text-status-critical"
      : "border-status-warning/30 bg-status-warning/10 text-status-warning"
  return (
    <span className={cn("inline-flex items-center rounded-sm border px-1.5 py-0.5 font-mono text-[11px] font-medium", tone)}>
      {verdict}
    </span>
  )
}
