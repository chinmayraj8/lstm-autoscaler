import type { LineSeriesOption } from "echarts/charts"
import type { EChartsCoreOption } from "echarts/core"
import { baseAxisStyle, baseTooltip, chartColors } from "@/lib/chart-theme"
import type { CpuReading, ForecastConfidenceResponse, ObservedDecision } from "@/lib/types"
import { parseAction } from "@/lib/types"

/**
 * Builds the ACTUAL vs FORECAST chart option. Plots:
 *   - a SOLID line: real Prometheus CPU% history (`readings`)
 *   - a DASHED line: ARIMA's point forecast, starting from the last
 *     actual point so it visually continues it
 *   - a shaded confidence band around the forecast -- REAL, from
 *     GET /forecast/confidence (statsmodels' get_forecast().conf_int()),
 *     when that endpoint has enough real history to answer (`forecastCi`
 *     non-null). When it doesn't (a brand-new node, a real scrape gap),
 *     falls back to the point-only forecast already logged in the live
 *     loop's `decisions` -- never fabricates a band for that case.
 *   - a NOW markLine at the boundary between actual and forecast
 *   - markPoints on the actual line for any non-hold scaling decision
 *     observed in the plotted window
 */
export function buildForecastOption(
  readings: CpuReading[],
  decisions: ObservedDecision[],
  forecastCi: ForecastConfidenceResponse | null,
  opts: { compact?: boolean } = {},
): EChartsCoreOption {
  const actual = readings.map((r) => [new Date(r.timestamp).getTime(), r.cpu_pct] as [number, number])
  const lastActual = actual.at(-1)
  const nowTs = lastActual?.[0] ?? Date.now()

  const forecast: [number, number][] = []
  const lower: [number, number][] = []
  const band: [number, number][] = [] // upper - lower, for ECharts' stack-to-build-a-band trick
  if (lastActual) {
    forecast.push(lastActual)
    lower.push([lastActual[0], lastActual[1]])
    band.push([lastActual[0], 0])
  }

  if (forecastCi) {
    const baseTs = new Date(forecastCi.observed_at).getTime()
    forecastCi.forecast.forEach((p) => {
      const ts = baseTs + p.step_minutes * 60_000
      forecast.push([ts, p.cpu_pct])
      lower.push([ts, p.lower_pct])
      band.push([ts, p.upper_pct - p.lower_pct])
    })
  } else {
    const latestDecision = decisions.at(-1)
    if (latestDecision) {
      const baseTs = new Date(latestDecision.observed_at).getTime()
      latestDecision.forecast_cpu_pct.forEach((v, i) => {
        forecast.push([baseTs + (i + 1) * 5 * 60_000, v])
      })
    }
  }

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

  const series: LineSeriesOption[] = [
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
          name: e.action.kind,
          coord: e.coord,
          itemStyle: {
            color: e.action.kind === "scale_up" ? chartColors.healthy : chartColors.arima,
          },
        })),
      },
    },
  ]

  if (forecastCi) {
    // Standard ECharts "band" trick: an invisible line at the lower
    // bound, then the (upper - lower) delta stacked on top of it with a
    // fill -- the visible top edge lands exactly at the real upper
    // bound. Excluded from the tooltip/legend since the raw delta value
    // is meaningless on its own.
    series.push(
      {
        name: "__ci_lower",
        type: "line",
        stack: "ci-band",
        data: lower,
        showSymbol: false,
        lineStyle: { opacity: 0 },
        tooltip: { show: false },
      },
      {
        name: "__ci_band",
        type: "line",
        stack: "ci-band",
        data: band,
        showSymbol: false,
        lineStyle: { opacity: 0 },
        areaStyle: { color: chartColors.arima, opacity: 0.14 },
        tooltip: { show: false },
      },
    )
  }

  series.push({
    name: "Forecast",
    type: "line",
    showSymbol: false,
    data: forecast,
    lineStyle: { color: chartColors.arima, width: 2, type: "dashed" },
    itemStyle: { color: chartColors.arima },
  })

  return {
    animation: false,
    grid: { left: 48, right: 16, top: opts.compact ? 16 : 36, bottom: 32 },
    tooltip: { trigger: "axis", ...baseTooltip },
    legend: opts.compact
      ? undefined
      : {
          top: 0,
          right: 0,
          data: ["Actual", "Forecast"],
          textStyle: { color: chartColors.muted, fontFamily: "IBM Plex Mono, monospace", fontSize: 11 },
          itemWidth: 16,
          itemHeight: 2,
        },
    xAxis: { type: "time", ...baseAxisStyle },
    yAxis: { type: "value", name: "CPU %", nameTextStyle: { color: chartColors.muted, fontSize: 11 }, ...baseAxisStyle },
    series,
  }
}
