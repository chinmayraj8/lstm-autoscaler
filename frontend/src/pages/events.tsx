import { ArrowRight, GitBranch, TrendingDown, TrendingUp } from "lucide-react"
import { Panel, PanelHeader, Section } from "@/components/section"
import { ForecasterBadge, VerdictBadge } from "@/components/status/badges"
import { useShadowDecisions, useShadowStatus } from "@/hooks/queries"
import { formatDateTime } from "@/lib/format"
import { parseAction } from "@/lib/types"
import { useSettings } from "@/lib/settings"

type Event =
  | { kind: "assignment"; timestamp: string; oldForecaster: string; newForecaster: string; verdict: string; evidence?: string }
  | { kind: "scaling"; timestamp: string; forecaster: string; from: number; to: number; action: string }

export default function EventsPage() {
  const { settings } = useSettings()
  const shadowStatus = useShadowStatus(settings.machineId)
  const decisions = useShadowDecisions(settings.machineId, 200)

  const assignmentEvents: Event[] = (shadowStatus.data?.assignment_history ?? []).map((c) => ({
    kind: "assignment",
    timestamp: c.timestamp,
    oldForecaster: c.old_forecaster,
    newForecaster: c.new_forecaster,
    verdict: c.verdict,
    evidence: c.evidence,
  }))

  const scalingEvents: Event[] = (decisions.data?.decisions ?? [])
    .filter((d) => parseAction(d.action).kind !== "hold")
    .map((d) => ({
      kind: "scaling",
      timestamp: d.observed_at,
      forecaster: d.forecaster,
      from: d.current_servers,
      to: d.recommended_servers,
      action: d.action,
    }))

  const events = [...assignmentEvents, ...scalingEvents].sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
  )

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-semibold">Events</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Forecaster re-assignments and non-hold scaling decisions, merged and time-ordered.
        </p>
      </div>

      <Section title="Timeline">
        <Panel>
          <PanelHeader title="Event feed" meta={`${events.length} events`} />
          <div className="divide-y divide-border">
            {events.length === 0 && (
              <div className="px-4 py-8 text-center text-sm text-muted-foreground">No events recorded yet.</div>
            )}
            {events.map((e, i) => (
              <div key={i} className="flex items-start gap-3 px-4 py-3">
                <div className="mt-0.5">
                  {e.kind === "assignment" ? (
                    <GitBranch className="h-4 w-4 text-primary" />
                  ) : parseAction(e.action).kind === "scale_up" ? (
                    <TrendingUp className="h-4 w-4 text-status-healthy" />
                  ) : (
                    <TrendingDown className="h-4 w-4 text-primary" />
                  )}
                </div>
                <div className="flex-1">
                  {e.kind === "assignment" ? (
                    <div className="flex items-center gap-2 text-sm">
                      <span className="text-muted-foreground">Forecaster re-assigned:</span>
                      <ForecasterBadge forecaster={e.oldForecaster as "arima" | "hybrid"} />
                      <ArrowRight className="h-3 w-3 text-muted-foreground" />
                      <ForecasterBadge forecaster={e.newForecaster as "arima" | "hybrid"} />
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 text-sm">
                      <span className="text-muted-foreground">Scaling:</span>
                      <ForecasterBadge forecaster={e.forecaster as "arima" | "hybrid"} />
                      <span className="font-mono text-xs">
                        {e.from} <ArrowRight className="inline h-3 w-3" /> {e.to} replicas
                      </span>
                    </div>
                  )}
                  {e.kind === "assignment" && (
                    <div className="mt-1 flex items-center gap-2">
                      <VerdictBadge verdict={e.verdict} />
                      {e.evidence && <span className="text-xs text-muted-foreground">{e.evidence}</span>}
                    </div>
                  )}
                  <div className="mt-1 font-mono text-[11px] text-muted-foreground">{formatDateTime(e.timestamp)}</div>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </Section>
    </div>
  )
}
