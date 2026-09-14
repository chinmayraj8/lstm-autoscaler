import type { EChartsCoreOption } from "echarts/core"
import { baseAxisStyle, baseTooltip, chartColors } from "@/lib/chart-theme"
import type { CpuReading, ObservedDecision } from "@/lib/types"
import { parseAction } from "@/lib/types"

/**
 * Builds the ACTUAL vs FORECAST chart option. Ground truth (see backend
 * audit): no confidence-interval forecast exists anywhere in this
 * codebase -- ARIMA/hybrid produce a single point forecast
 * (`ObservedDecision.forecast_cpu_pct`, 3 steps @ 5min), never a
 * `get_forecast().conf_int()` band. So this chart plots:
 *   - a SOLID line: real Prometheus CPU% history (`readings`)
 *   - a DASHED line: the most recent tick's 3-step-ahead point forecast,
 *     starting from the last actual point so it visually continues it
 *   - a NOW markLine at the boundary between the two
 *   - markPoints on the actual line for any non-hold scaling decision
 *     observed in the plotted window
 * No confidence band is drawn -- there is no real data for one.
 */
export function buildForecastOption(
  readings: CpuReading[],
  decisions: ObservedDecision[],
  opts: { compact?: boolean } = {},
): EChartsCoreOption {
  const actual = readings.map((r) => [new Date(r.timestamp).getTime(), r.cpu_pct] as [number, number])
  const lastActual = actual.at(-1)

  const latestDecision = decisions.at(-1)
  const forecast: [number, number][] = []
  if (lastActual) forecast.push(lastActual)
  if (latestDecision) {
    const baseTs = new Date(latestDecision.observed_at).getTime()
    latestDecision.forecast_cpu_pct.forEach((v, i) => {
      forecast.push([baseTs + (i + 1) * 5 * 60_000, v])
    })
  }

  const nowTs = lastActual?.[0] ?? Date.now()

  // Scaling-action markers: nearest actual reading to each non-hold
  // decision's timestamp, so the marker sits on the real observed line
  // rather than inventing a y-value.
  const scalingEvents = decisions
    .filter((d) => parseAction(d.action).kind !== "hold")
    .map((d) => {
      const ts = new Date(d.observed_at).getTime()
      let nearest = actual[0]
      let bestDiff = Infinity
      for (const point of actual) {
        const diff = Math.abs(point[0] - ts)
        if (diff < bestDiff) {
          bestDiff = diff
          nearest = point
        }
      }
      return nearest ? { coord: nearest, action: parseAction(d.action) } : null
    })
    .filter((x): x is NonNullable<typeof x> => x !== null)

  return {
    animation: false,
    grid: { left: 48, right: 16, top: opts.compact ? 16 : 36, bottom: 32 },
    tooltip: { trigger: "axis", ...baseTooltip },
    legend: opts.compact
      ? undefined
      : {
          top: 0,
          right: 0,
          textStyle: { color: chartColors.muted, fontFamily: "IBM Plex Mono, monospace", fontSize: 11 },
          itemWidth: 16,
          itemHeight: 2,
        },
    xAxis: { type: "time", ...baseAxisStyle },
    yAxis: { type: "value", name: "CPU %", nameTextStyle: { color: chartColors.muted, fontSize: 11 }, ...baseAxisStyle },
    series: [
      {
        name: "Actual",
        type: "line",
        showSymbol: false,
        data: actual,
        lineStyle: { color: chartColors.actual, width: 2 },
        itemStyle: { color: chartColors.actual },
        markLine: {
          symbol: "none",
          silent: true,
          lineStyle: { color: chartColors.muted, type: "solid", width: 1 },
          label: { formatter: "NOW", color: chartColors.muted, fontFamily: "IBM Plex Mono, monospace", fontSize: 10 },
          data: [{ xAxis: nowTs }],
        },
        markPoint: {
          symbolSize: 8,
          data: scalingEvents.map((e) => ({
            coord: e.coord,
            itemStyle: {
              color: e.action.kind === "scale_up" ? chartColors.healthy : chartColors.arima,
            },
          })),
        },
      },
      {
        name: "Forecast",
        type: "line",
        showSymbol: false,
        data: forecast,
        lineStyle: { color: chartColors.arima, width: 2, type: "dashed" },
        itemStyle: { color: chartColors.arima },
      },
    ],
  }
}
