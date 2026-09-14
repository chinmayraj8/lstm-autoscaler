import type { EChartsCoreOption } from "echarts/core"
import { baseAxisStyle, baseTooltip, chartColors } from "@/lib/chart-theme"
import type { ShadowWindow } from "@/lib/types"

/** Cost per shadow window -- ARIMA vs Hybrid, the real backward-looking
 * comparison metric this project actually has (see shadow_windows table).
 * Not a forecast; each point is an already-elapsed 24h window's cost
 * score for both forecasters against the same real outcome. */
export function buildCostOption(windows: ShadowWindow[]): EChartsCoreOption {
  const x = windows.map((_, i) => `W${i + 1}`)
  return {
    animation: false,
    grid: { left: 44, right: 16, top: 32, bottom: 28 },
    tooltip: { trigger: "axis", ...baseTooltip },
    legend: {
      top: 0,
      right: 0,
      textStyle: { color: chartColors.muted, fontFamily: "IBM Plex Mono, monospace", fontSize: 11 },
      itemWidth: 16,
      itemHeight: 2,
    },
    xAxis: { type: "category", data: x, name: "Window #", nameTextStyle: { color: chartColors.muted, fontSize: 11 }, ...baseAxisStyle },
    yAxis: { type: "value", name: "Cost score", nameTextStyle: { color: chartColors.muted, fontSize: 11 }, ...baseAxisStyle },
    series: [
      {
        name: "ARIMA",
        type: "line",
        data: windows.map((w) => w.arima_cost),
        lineStyle: { color: chartColors.arima, width: 2 },
        itemStyle: { color: chartColors.arima },
        symbolSize: 5,
      },
      {
        name: "Hybrid",
        type: "line",
        data: windows.map((w) => w.hybrid_cost),
        lineStyle: { color: chartColors.hybrid, width: 2 },
        itemStyle: { color: chartColors.hybrid },
        symbolSize: 5,
      },
    ],
  }
}
