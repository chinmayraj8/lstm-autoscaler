import { Info } from "lucide-react"
import { EChart } from "@/components/charts/echart"
import { Panel, PanelHeader, Section } from "@/components/section"
import { useCpuMetrics, useForecastConfidence, useShadowDecisions } from "@/hooks/queries"
import { buildForecastOption } from "@/lib/build-forecast-option"
import { formatRelative } from "@/lib/format"
import { useSettings } from "@/lib/settings"

export default function ForecastingPage() {
  const { settings } = useSettings()
  const cpu = useCpuMetrics(settings.machineId, 2)
  const decisions = useShadowDecisions(settings.machineId, 50)
  const forecastCi = useForecastConfidence(settings.machineId)
  const latestDecision = decisions.data?.decisions.at(-1)

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-semibold">Forecasting</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Real CPU history vs. ARIMA's 15-minute-ahead forecast, with a real {(
            (forecastCi.data?.confidence_level ?? 0.95) * 100
          ).toFixed(0)}% confidence interval &middot; node {settings.machineId}
        </p>
      </div>

      <Section
        title="Actual vs. forecast"
        description={
          forecastCi.data
            ? `fit on ${forecastCi.data.fit_points} real points over the last ${forecastCi.data.fit_window_hours}h · ARIMA${JSON.stringify(forecastCi.data.arima_order)}`
            : latestDecision
              ? `forecast issued ${formatRelative(latestDecision.observed_at)} by ${latestDecision.forecaster}`
              : undefined
        }
      >
        <Panel>
          <PanelHeader
            title={
              <div className="flex items-center gap-4 font-mono text-xs">
                <span className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-3 bg-[#c7cbd1]" /> Actual</span>
                <span className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-3 border-t-2 border-dashed border-[#4c8df0]" /> Forecast</span>
                {forecastCi.data && (
                  <span className="flex items-center gap-1.5">
                    <span className="inline-block h-2.5 w-3 bg-[#4c8df0]/20" /> Confidence interval
                  </span>
                )}
              </div>
            }
          />
          {cpu.data ? (
            <EChart
              option={buildForecastOption(cpu.data.readings, decisions.data?.decisions ?? [], forecastCi.data ?? null)}
              height={420}
            />
          ) : (
            <div className="flex h-[420px] items-center justify-center text-sm text-muted-foreground">
              {cpu.isError ? "Prometheus unreachable" : "Loading…"}
            </div>
          )}
        </Panel>
      </Section>

      {!forecastCi.isLoading && !forecastCi.data && (
        <div className="flex items-start gap-2 border border-status-warning/30 bg-status-warning/5 px-4 py-3 text-xs text-muted-foreground">
          <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-status-warning" />
          <p>
            No confidence band available right now &mdash; <code className="font-mono">/forecast/confidence</code>{" "}
            reported not enough real history in the fit window for this node. Falling back to the point-only forecast
            from the live loop's decision log, with no fabricated band.
          </p>
        </div>
      )}

      {forecastCi.data ? (
        <Section title="Latest forecast detail">
          <Panel className="p-4">
            <div className="grid grid-cols-3 gap-4 font-mono text-sm sm:grid-cols-6">
              {forecastCi.data.forecast.map((p) => (
                <div key={p.step_minutes}>
                  <div className="text-[10px] uppercase text-muted-foreground">t+{p.step_minutes}min</div>
                  <div className="text-lg font-semibold">{p.cpu_pct.toFixed(2)}%</div>
                  <div className="text-[11px] text-muted-foreground">
                    [{p.lower_pct.toFixed(2)}, {p.upper_pct.toFixed(2)}]
                  </div>
                </div>
              ))}
            </div>
          </Panel>
        </Section>
      ) : latestDecision ? (
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
      ) : null}
    </div>
  )
}
