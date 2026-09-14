import { ArrowRight } from "lucide-react"
import { EChart } from "@/components/charts/echart"
import { KpiRow, type Kpi } from "@/components/kpi-row"
import { Panel, PanelHeader, Section } from "@/components/section"
import { ActionBadge, ForecasterBadge } from "@/components/status/badges"
import { StatusPill } from "@/components/status/status-dot"
import { Skeleton } from "@/components/ui/skeleton"
import {
  useCpuMetrics,
  useReplicas,
  useScalingConfig,
  useShadowDecisions,
  useShadowStatus,
} from "@/hooks/queries"
import { useSystemStatus } from "@/hooks/use-system-status"
import { buildForecastOption } from "@/lib/build-forecast-option"
import { formatNumber, formatRelative, formatSigned } from "@/lib/format"
import { buildScalingReason } from "@/lib/scaling-reason"
import { useSettings } from "@/lib/settings"

export default function OverviewPage() {
  const { settings } = useSettings()
  const machineId = settings.machineId
  const status = useSystemStatus(machineId)
  const shadowStatus = useShadowStatus(machineId)
  const decisions = useShadowDecisions(machineId, 6)
  const cpu = useCpuMetrics(machineId, 2)
  const replicas = useReplicas()
  const config = useScalingConfig()

  const latestDecision = decisions.data?.decisions.at(-1)
  const cumulative = shadowStatus.data?.cumulative

  const costDelta =
    cumulative?.arima_cost_mean !== undefined && cumulative?.hybrid_cost_mean !== undefined
      ? ((cumulative.hybrid_cost_mean - cumulative.arima_cost_mean) / cumulative.arima_cost_mean) * 100
      : undefined

  const kpis: Kpi[] = [
    {
      label: "Current CPU",
      value: cpu.data?.readings.at(-1) ? `${cpu.data.readings.at(-1)!.cpu_pct.toFixed(1)}%` : "—",
      caption: `node ${machineId}`,
    },
    {
      label: "Replicas",
      value: replicas.data?.replicas ?? (replicas.isError ? "—" : <Skeleton className="h-7 w-8" />),
      caption: replicas.isError ? "unreachable" : "demo-workload, read-only",
    },
    {
      label: "Assigned forecaster",
      value: shadowStatus.data ? shadowStatus.data.current_forecaster.toUpperCase() : "—",
      caption: shadowStatus.data ? `${shadowStatus.data.n_banked_windows} windows banked` : "no shadow state yet",
    },
    {
      label: "Hybrid vs ARIMA cost",
      value: costDelta !== undefined ? formatSigned(costDelta) : "—",
      tone: costDelta !== undefined ? (costDelta < 0 ? "healthy" : "neutral") : "neutral",
      caption: costDelta !== undefined ? "negative = hybrid cheaper" : "not enough data",
    },
  ]

  return (
    <div className="space-y-6">
      <div>
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-semibold">System Overview</h1>
          <StatusPill tone={status.isStale ? "neutral" : "healthy"} pulse={!status.isStale}>
            {status.isStale ? "Stale" : "Live"}
          </StatusPill>
        </div>
        <p className="mt-1 text-sm text-muted-foreground">
          LSTM Autoscaler shadow-forecasting service &middot; node {machineId} &middot; {status.note}
        </p>
      </div>

      <KpiRow items={kpis} />

      <div className="grid gap-4 lg:grid-cols-3">
        <Section title="Actual vs. forecast" className="lg:col-span-2" description="Last 2 hours, next 15 minutes">
          <Panel>
            {cpu.data ? (
              <EChart option={buildForecastOption(cpu.data.readings, decisions.data?.decisions ?? [], { compact: true })} height={220} />
            ) : (
              <div className="flex h-[220px] items-center justify-center text-sm text-muted-foreground">
                {cpu.isError ? "Prometheus unreachable" : "Loading…"}
              </div>
            )}
          </Panel>
        </Section>

        <Section title="Will autoscaling react?">
          <Panel className="flex h-full flex-col justify-between p-4">
            {latestDecision && config.data ? (
              <>
                <div>
                  <div className="flex items-center gap-2">
                    <ForecasterBadge forecaster={latestDecision.forecaster} />
                    <ActionBadge action={latestDecision.action} />
                  </div>
                  <div className="mt-3 flex items-baseline gap-2">
                    <span className="font-mono text-2xl font-semibold">{latestDecision.current_servers}</span>
                    <ArrowRight className="h-4 w-4 text-muted-foreground" />
                    <span className="font-mono text-2xl font-semibold text-primary">
                      {latestDecision.recommended_servers}
                    </span>
                    <span className="text-xs text-muted-foreground">replicas</span>
                  </div>
                  <p className="mt-2 text-xs text-muted-foreground">
                    {buildScalingReason(
                      latestDecision.action,
                      latestDecision.planned_load_pct,
                      latestDecision.current_servers,
                      config.data.server_capacity_pct,
                    )}
                  </p>
                </div>
                <div className="mt-3 text-[11px] font-mono text-muted-foreground">
                  observed {formatRelative(latestDecision.observed_at)}
                </div>
              </>
            ) : (
              <div className="text-sm text-muted-foreground">No decisions observed yet.</div>
            )}
          </Panel>
        </Section>
      </div>

      <Section title="Recently" description="Latest scaling decisions for this node">
        <Panel>
          <PanelHeader title="Recent decisions" meta={`showing ${decisions.data?.decisions.length ?? 0}`} />
          <div className="divide-y divide-border">
            {decisions.data?.decisions
              .slice()
              .reverse()
              .map((d, i) => (
                <div key={i} className="flex items-center gap-3 px-4 py-2 text-sm">
                  <span className="w-40 font-mono text-xs text-muted-foreground">{formatRelative(d.observed_at)}</span>
                  <ForecasterBadge forecaster={d.forecaster} />
                  <ActionBadge action={d.action} />
                  <span className="font-mono text-xs text-muted-foreground">
                    {d.current_servers} <ArrowRight className="inline h-3 w-3" /> {d.recommended_servers}
                  </span>
                  <span className="ml-auto font-mono text-xs text-muted-foreground">
                    {formatNumber(d.planned_load_pct, 1)}% planned load
                  </span>
                </div>
              )) ?? <div className="px-4 py-6 text-center text-sm text-muted-foreground">No ticks logged yet.</div>}
          </div>
        </Panel>
      </Section>
    </div>
  )
}
