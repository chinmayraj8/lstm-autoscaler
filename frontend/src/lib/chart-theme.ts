/**
 * ECharts renders to canvas, so it can't read the CSS custom properties in
 * index.css directly -- these mirror that file's token values exactly so
 * charts and the rest of the UI never drift apart. If a token in
 * index.css changes, update it here too.
 */
export const chartColors = {
  background: "transparent",
  foreground: "#e8eaed",
  muted: "#8b929e",
  border: "#22262e",
  grid: "#1c2027",
  card: "#111318",
  healthy: "#22c55e",
  warning: "#f5a524",
  critical: "#ef4444",
  arima: "#4c8df0",
  hybrid: "#22d3ee",
  actual: "#c7cbd1",
}

export const baseAxisStyle = {
  axisLine: { lineStyle: { color: chartColors.border } },
  axisLabel: { color: chartColors.muted, fontFamily: "IBM Plex Mono, monospace", fontSize: 11 },
  splitLine: { lineStyle: { color: chartColors.grid, type: "dashed" as const } },
  axisTick: { show: false },
}

export const baseTooltip = {
  backgroundColor: chartColors.card,
  borderColor: chartColors.border,
  borderWidth: 1,
  textStyle: { color: chartColors.foreground, fontFamily: "IBM Plex Mono, monospace", fontSize: 12 },
  padding: 8,
}
