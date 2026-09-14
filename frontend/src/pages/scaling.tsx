import { ArrowRight, ShieldAlert } from "lucide-react"
import { EChart } from "@/components/charts/echart"
import { Panel, PanelHeader, Section } from "@/components/section"
import { ActionBadge, ForecasterBadge, VerdictBadge } from "@/components/status/badges"
import { StatusPill } from "@/components/status/status-dot"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import {
  useActuationStatus,
  useScalingConfig,
  useShadowDecisions,
  useShadowStatus,
  useShadowWindows,
} from "@/hooks/queries"
import { buildCostOption } from "@/lib/build-cost-option"
import { formatDateTime, formatNumber, formatPct, formatRelative } from "@/lib/format"
import { buildScalingReason } from "@/lib/scaling-reason"
import { useSettings } from "@/lib/settings"

export default function ScalingPage() {
  const { settings } = useSettings()
  const machineId = settings.machineId
  const decisions = useShadowDecisions(machineId, 1)
  const config = useScalingConfig()
  const actuation = useActuationStatus()
  const shadowStatus = useShadowStatus(machineId)
  const shadowWindows = useShadowWindows(machineId)

  const latest = decisions.data?.decisions.at(0)
  const cumulative = shadowStatus.data?.cumulative

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-semibold">Scaling</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          What the live loop decided this tick, why, and which forecaster is currently trusted to drive it.
        </p>
      </div>

      <Section title="Current scaling decision">
        <Panel className="p-5">
          {latest && config.data ? (
            <div className="grid gap-6 sm:grid-cols-[auto_1fr]">
              <div className="flex items-center gap-4">
                <div className="text-center">
                  <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Current</div>
                  <div className="font-mono text-4xl font-semibold">{latest.current_servers}</div>
                </div>
                <ArrowRight className="h-5 w-5 text-muted-foreground" />
                <div className="text-center">
                  <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Predicted</div>
                  <div className="font-mono text-4xl font-semibold text-primary">{latest.recommended_servers}</div>
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] uppercase tracking-wide text-muted-foreground">Decision</span>
                  <ActionBadge action={latest.action} />
                  <span className="text-[11px] uppercase tracking-wide text-muted-foreground">via</span>
                  <ForecasterBadge forecaster={latest.forecaster} />
                </div>
                <div>
                  <span className="text-[11px] uppercase tracking-wide text-muted-foreground">Reason</span>
                  <p className="text-sm">
                    {buildScalingReason(latest.action, latest.planned_load_pct, latest.current_servers, config.data.server_capacity_pct)}
                  </p>
                </div>
                <div className="font-mono text-[11px] text-muted-foreground">observed {formatRelative(latest.observed_at)}</div>
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No decision observed yet for this node.</p>
          )}
        </Panel>
      </Section>

      <Section title="Forecast vs. threshold vs. capacity" description="Per-replica capacity threshold from the live decision-engine config">
        <Panel className="p-4">
          {latest && config.data ? (
            <div className="space-y-3">
              {[
                { label: "Current capacity", value: latest.current_servers * config.data.server_capacity_pct, tone: "bg-muted-foreground" },
                { label: "Planned load (forecast)", value: latest.planned_load_pct, tone: "bg-primary" },
                { label: "Recommended capacity", value: latest.recommended_servers * config.data.server_capacity_pct, tone: "bg-status-healthy" },
              ].map((row) => {
                const max = Math.max(
                  latest.current_servers * config.data!.server_capacity_pct,
                  latest.planned_load_pct,
                  latest.recommended_servers * config.data!.server_capacity_pct,
                  1,
                )
                const pct = Math.min(100, (row.value / max) * 100)
                return (
                  <div key={row.label}>
                    <div className="mb-1 flex justify-between font-mono text-xs">
                      <span className="text-muted-foreground">{row.label}</span>
                      <span>{row.value.toFixed(1)}%</span>
                    </div>
                    <div className="h-2 w-full bg-secondary">
                      <div className={`h-2 ${row.tone}`} style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                )
              })}
              <p className="pt-1 text-xs text-muted-foreground">
                Capacity per replica: {config.data.server_capacity_pct}% &middot; bounds: {config.data.min_servers}&ndash;
                {config.data.max_servers} replicas &middot; step: &plusmn;{config.data.scale_step}/tick
              </p>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No data yet.</p>
          )}
        </Panel>
      </Section>

      <Section title="Real actuation">
        <Panel className="p-4">
          {actuation.data?.enabled ? (
            <div className="flex flex-wrap items-center gap-4">
              <StatusPill tone={actuation.data.circuit_open ? "critical" : "healthy"}>
                {actuation.data.circuit_open ? "Circuit open" : "Enabled"}
              </StatusPill>
              <span className="font-mono text-xs text-muted-foreground">
                machine={actuation.data.machine_id} target={actuation.data.deployment}/{actuation.data.namespace}
              </span>
              <span className="font-mono text-xs text-muted-foreground">
                failures: {actuation.data.consecutive_failures}/{actuation.data.circuit_breaker_threshold}
              </span>
              {actuation.data.circuit_open && (
                <span className="flex items-center gap-1 font-mono text-xs text-status-critical">
                  <ShieldAlert className="h-3.5 w-3.5" />
                  cooldown {Math.ceil((actuation.data.cooldown_remaining_seconds ?? 0) / 60)}m remaining
                </span>
              )}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              Real actuation is disabled or the live loop isn&rsquo;t running &mdash; every decision above is observe-only.
            </p>
          )}
        </Panel>
      </Section>

      <Section
        title="Model comparison"
        description="Why the current forecaster is trusted &mdash; real shadow-evaluation metrics only"
      >
        <div className="grid gap-4 lg:grid-cols-3">
          <Panel className="lg:col-span-2">
            <PanelHeader title="Cost per shadow window" meta={`${shadowWindows.data?.windows.length ?? 0} windows`} />
            {shadowWindows.data && shadowWindows.data.windows.length >= 2 ? (
              <EChart option={buildCostOption(shadowWindows.data.windows)} height={260} />
            ) : (
              <div className="flex h-[260px] items-center justify-center text-sm text-muted-foreground">
                Needs at least 2 banked shadow windows to plot a trend.
              </div>
            )}
          </Panel>

          <Panel className="p-4">
            <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Currently assigned</div>
            <div className="mt-1">
              {shadowStatus.data ? <ForecasterBadge forecaster={shadowStatus.data.current_forecaster} /> : "—"}
            </div>
            <dl className="mt-4 space-y-2 font-mono text-xs">
              <div className="flex justify-between">
                <dt className="text-muted-foreground">ARIMA mean cost</dt>
                <dd>{formatNumber(cumulative?.arima_cost_mean)}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Hybrid mean cost</dt>
                <dd>{formatNumber(cumulative?.hybrid_cost_mean)}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">ARIMA SLA violations</dt>
                <dd>{formatPct(cumulative?.arima_sla_mean_pct)}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Hybrid SLA violations</dt>
                <dd>{formatPct(cumulative?.hybrid_sla_mean_pct)}</dd>
              </div>
            </dl>
            <p className="mt-3 text-[11px] text-muted-foreground">
              MAE/RMSE are computed only in offline experiments (experiments/*.py), not exposed by the live API &mdash;
              not shown here to avoid implying a live metric that doesn&rsquo;t exist.
            </p>
          </Panel>
        </div>

        {shadowStatus.data && shadowStatus.data.assignment_history.length > 0 && (
          <Panel>
            <PanelHeader title="Assignment history" />
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>When</TableHead>
                  <TableHead>Change</TableHead>
                  <TableHead>Verdict</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {shadowStatus.data.assignment_history
                  .slice()
                  .reverse()
                  .map((c, i) => (
                    <TableRow key={i}>
                      <TableCell className="font-mono text-xs text-muted-foreground">{formatDateTime(c.timestamp)}</TableCell>
                      <TableCell className="flex items-center gap-1.5">
                        <ForecasterBadge forecaster={c.old_forecaster} /> <ArrowRight className="h-3 w-3" />{" "}
                        <ForecasterBadge forecaster={c.new_forecaster} />
                      </TableCell>
                      <TableCell><VerdictBadge verdict={c.verdict} /></TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
          </Panel>
        )}
      </Section>
    </div>
  )
}
