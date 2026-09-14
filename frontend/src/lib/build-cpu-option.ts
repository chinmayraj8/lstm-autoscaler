import type { EChartsCoreOption } from "echarts/core"
import { baseAxisStyle, baseTooltip, chartColors } from "@/lib/chart-theme"
import type { CpuReading } from "@/lib/types"

export function buildCpuOption(readings: CpuReading[]): EChartsCoreOption {
  const data = readings.map((r) => [new Date(r.timestamp).getTime(), r.cpu_pct] as [number, number])
  return {
    animation: false,
    grid: { left: 48, right: 16, top: 16, bottom: 32 },
    tooltip: { trigger: "axis", ...baseTooltip },
    xAxis: { type: "time", ...baseAxisStyle },
    yAxis: { type: "value", name: "CPU %", nameTextStyle: { color: chartColors.muted, fontSize: 11 }, ...baseAxisStyle },
    series: [
      {
        type: "line",
        showSymbol: false,
        data,
        lineStyle: { color: chartColors.actual, width: 2 },
        areaStyle: { color: chartColors.actual, opacity: 0.08 },
      },
    ],
  }
}
