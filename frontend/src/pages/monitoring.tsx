import { EChart } from "@/components/charts/echart"
import { KpiRow, type Kpi } from "@/components/kpi-row"
import { Panel, PanelHeader, Section } from "@/components/section"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useCpuMetrics, useReplicas } from "@/hooks/queries"
import { buildCpuOption } from "@/lib/build-cpu-option"
import { useSettings } from "@/lib/settings"
import { useState } from "react"

const HOUR_OPTIONS = [
  { value: "1", label: "1 hour" },
  { value: "2", label: "2 hours" },
  { value: "6", label: "6 hours" },
  { value: "24", label: "24 hours" },
]

export default function MonitoringPage() {
  const { settings } = useSettings()
  const [hours, setHours] = useState("2")
  const cpu = useCpuMetrics(settings.machineId, Number(hours))
  const replicas = useReplicas()

  const latest = cpu.data?.readings.at(-1)
  const readings = cpu.data?.readings ?? []
  const mean = readings.length ? readings.reduce((s, r) => s + r.cpu_pct, 0) / readings.length : undefined
  const peak = readings.length ? Math.max(...readings.map((r) => r.cpu_pct)) : undefined

  const kpis: Kpi[] = [
    { label: "Current CPU", value: latest ? `${latest.cpu_pct.toFixed(1)}%` : "—" },
    { label: "Mean (window)", value: mean !== undefined ? `${mean.toFixed(1)}%` : "—" },
    { label: "Peak (window)", value: peak !== undefined ? `${peak.toFixed(1)}%` : "—" },
    { label: "Replicas", value: replicas.data?.replicas ?? "—", caption: "demo-workload" },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-semibold">Monitoring</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Real per-node CPU utilization, scraped from Prometheus (node-exporter) &middot; node {settings.machineId}
        </p>
      </div>

      <KpiRow items={kpis} />

      <Section
        title="Node CPU%"
        description={`resample: ${cpu.data?.resample_minutes ?? 5} min`}
        action={
          <Select value={hours} onValueChange={(v) => v && setHours(v)}>
            <SelectTrigger size="sm" className="w-[120px] font-mono text-xs">
              <SelectValue>{(v: string | null) => HOUR_OPTIONS.find((o) => o.value === v)?.label ?? "Select"}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {HOUR_OPTIONS.map((o) => (
                <SelectItem key={o.value} value={o.value} className="font-mono text-xs">
                  {o.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        }
      >
        <Panel>
          {cpu.isLoading ? (
            <div className="flex h-[320px] items-center justify-center text-sm text-muted-foreground">Loading…</div>
          ) : cpu.isError || !cpu.data?.readings.length ? (
            <div className="flex h-[320px] items-center justify-center text-sm text-muted-foreground">
              {cpu.isError ? "Prometheus unreachable at the configured URL." : "No CPU data in this window."}
            </div>
          ) : (
            <EChart option={buildCpuOption(cpu.data.readings)} height={320} />
          )}
        </Panel>
      </Section>

      <Section title="Workload replicas" description="demo-workload deployment, read via local/in-cluster kubeconfig">
        <Panel>
          <PanelHeader title="demo-workload" meta={replicas.data ? `checked ${new Date(replicas.data.checked_at).toLocaleTimeString()}` : undefined} />
          <div className="p-4">
            {replicas.isError ? (
              <p className="text-sm text-muted-foreground">Could not read replica count (cluster/kubeconfig unreachable).</p>
            ) : (
              <div className="font-mono text-3xl font-semibold">{replicas.data?.replicas ?? "—"}</div>
            )}
          </div>
        </Panel>
      </Section>
    </div>
  )
}
