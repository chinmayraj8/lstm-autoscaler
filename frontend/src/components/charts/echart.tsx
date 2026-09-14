import * as echarts from "echarts/core"
import { LineChart } from "echarts/charts"
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  MarkPointComponent,
  TooltipComponent,
} from "echarts/components"
import { CanvasRenderer } from "echarts/renderers"
import { useEffect, useRef } from "react"
import type { EChartsCoreOption } from "echarts/core"

echarts.use([
  LineChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  MarkLineComponent,
  MarkPointComponent,
  MarkAreaComponent,
  DataZoomComponent,
  CanvasRenderer,
])

/** Thin ECharts wrapper -- no charting library indirection beyond what's
 * needed to mount/resize/dispose an instance from React. Chart option
 * objects are built by each page/component, not here. */
export function EChart({
  option,
  height = 280,
  className,
}: {
  option: EChartsCoreOption
  height?: number
  className?: string
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<echarts.ECharts | null>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const chart = echarts.init(containerRef.current, undefined, { renderer: "canvas" })
    chartRef.current = chart

    const resizeObserver = new ResizeObserver(() => chart.resize())
    resizeObserver.observe(containerRef.current)

    return () => {
      resizeObserver.disconnect()
      chart.dispose()
      chartRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    chartRef.current?.setOption(option, { notMerge: true })
  }, [option])

  return <div ref={containerRef} className={className} style={{ height, width: "100%" }} />
}
