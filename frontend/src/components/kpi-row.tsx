import { cn } from "@/lib/utils"

export interface Kpi {
  label: string
  value: React.ReactNode
  caption?: React.ReactNode
  tone?: "healthy" | "warning" | "critical" | "neutral"
}

const TONE_TEXT: Record<NonNullable<Kpi["tone"]>, string> = {
  healthy: "text-status-healthy",
  warning: "text-status-warning",
  critical: "text-status-critical",
  neutral: "text-foreground",
}

/**
 * A dense inline KPI strip -- deliberately NOT a row of cards. Spacing,
 * a hairline divider between items, and typography carry the hierarchy
 * instead of a box around every number.
 */
export function KpiRow({ items }: { items: Kpi[] }) {
  return (
    <div className="grid grid-cols-2 divide-x divide-border border border-border sm:grid-cols-3 lg:grid-cols-4">
      {items.map((item, i) => (
        <div key={i} className="px-4 py-3">
          <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
            {item.label}
          </div>
          <div className={cn("mt-1 font-mono text-2xl font-semibold tabular-nums", TONE_TEXT[item.tone ?? "neutral"])}>
            {item.value}
          </div>
          {item.caption && <div className="mt-0.5 text-xs text-muted-foreground">{item.caption}</div>}
        </div>
      ))}
    </div>
  )
}
