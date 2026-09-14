import { cn } from "@/lib/utils"

type Tone = "healthy" | "warning" | "critical" | "neutral"

const TONE_CLASSES: Record<Tone, string> = {
  healthy: "bg-status-healthy",
  warning: "bg-status-warning",
  critical: "bg-status-critical",
  neutral: "bg-muted-foreground",
}

/** A status dot with an optional pulse -- reserved for genuine liveness
 * signals (is data flowing right now), never decoration. */
export function StatusDot({ tone, pulse = false, className }: { tone: Tone; pulse?: boolean; className?: string }) {
  return (
    <span className={cn("relative inline-flex h-2 w-2", className)}>
      {pulse && (
        <span className={cn("absolute inline-flex h-full w-full animate-ping rounded-full opacity-60", TONE_CLASSES[tone])} />
      )}
      <span className={cn("relative inline-flex h-2 w-2 rounded-full", TONE_CLASSES[tone])} />
    </span>
  )
}

export function StatusPill({
  tone,
  pulse = false,
  children,
}: {
  tone: Tone
  pulse?: boolean
  children: React.ReactNode
}) {
  const textTone: Record<Tone, string> = {
    healthy: "text-status-healthy",
    warning: "text-status-warning",
    critical: "text-status-critical",
    neutral: "text-muted-foreground",
  }
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-sm border border-border/80 bg-secondary/40 px-2 py-0.5 font-mono text-[11px] font-medium uppercase tracking-wide",
        textTone[tone],
      )}
    >
      <StatusDot tone={tone} pulse={pulse} />
      {children}
    </span>
  )
}
