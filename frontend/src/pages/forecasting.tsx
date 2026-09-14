import { Info } from "lucide-react"
import { EChart } from "@/components/charts/echart"
import { Panel, PanelHeader, Section } from "@/components/section"
import { useCpuMetrics, useShadowDecisions } from "@/hooks/queries"
import { buildForecastOption } from "@/lib/build-forecast-option"
import { formatRelative } from "@/lib/format"
import { useSettings } from "@/lib/settings"

export default function ForecastingPage() {
  const { settings } = useSettings()
  const cpu = useCpuMetrics(settings.machineId, 2)
  const decisions = useShadowDecisions(settings.machineId, 50)
  const latestDecision = decisions.data?.decisions.at(-1)

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-semibold">Forecasting</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Real CPU history vs. the live loop's most recent 15-minute-ahead point forecast &middot; node {settings.machineId}
        </p>
      </div>

      <Section
        title="Actual vs. forecast"
        description={latestDecision ? `forecast issued ${formatRelative(latestDecision.observed_at)} by ${latestDecision.forecaster}` : undefined}
      >
        <Panel>
          <PanelHeader
            title={
              <div className="flex items-center gap-4 font-mono text-xs">
                <span className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-3 bg-[#c7cbd1]" /> Actual</span>
                <span className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-3 border-t-2 border-dashed border-[#4c8df0]" /> Forecast</span>
              </div>
            }
          />
          {cpu.data ? (
            <EChart option={buildForecastOption(cpu.data.readings, decisions.data?.decisions ?? [])} height={420} />
          ) : (
            <div className="flex h-[420px] items-center justify-center text-sm text-muted-foreground">
              {cpu.isError ? "Prometheus unreachable" : "Loading…"}
            </div>
          )}
        </Panel>
      </Section>

      <div className="flex items-start gap-2 border border-status-warning/30 bg-status-warning/5 px-4 py-3 text-xs text-muted-foreground">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-status-warning" />
        <p>
          No confidence/uncertainty band is drawn. Neither ARIMA nor the residual-hybrid forecaster in this project's
          live path calls a prediction-interval method (e.g. <code className="font-mono">get_forecast().conf_int()</code>)
          anywhere &mdash; both produce a single point forecast. Adding a real confidence band would require that
          backend change first; this chart intentionally does not fabricate one.
        </p>
      </div>

      {latestDecision && (
        <Section title="Latest forecast detail">
          <Panel className="p-4">
            <div className="grid grid-cols-3 gap-4 font-mono text-sm sm:grid-cols-6">
              {latestDecision.forecast_cpu_pct.map((v, i) => (
                <div key={i}>
                  <div className="text-[10px] uppercase text-muted-foreground">t+{(i + 1) * 5}min</div>
                  <div className="text-lg font-semibold">{v.toFixed(2)}%</div>
                </div>
              ))}
            </div>
          </Panel>
        </Section>
      )}
    </div>
  )
}
