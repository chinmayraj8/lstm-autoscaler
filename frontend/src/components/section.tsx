import { cn } from "@/lib/utils"

export function Section({
  title,
  description,
  action,
  children,
  className,
}: {
  title: string
  description?: React.ReactNode
  action?: React.ReactNode
  children: React.ReactNode
  className?: string
}) {
  return (
    <section className={cn("space-y-3", className)}>
      <div className="flex items-baseline justify-between gap-4">
        <div>
          <h2 className="text-sm font-semibold text-foreground">{title}</h2>
          {description && <p className="mt-0.5 text-xs text-muted-foreground">{description}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  )
}

export function Panel({ className, children }: { className?: string; children: React.ReactNode }) {
  return <div className={cn("border border-border bg-card", className)}>{children}</div>
}

export function PanelHeader({
  title,
  meta,
}: {
  title: React.ReactNode
  meta?: React.ReactNode
}) {
  return (
    <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
      <div className="text-[13px] font-medium text-foreground">{title}</div>
      {meta && <div className="font-mono text-[11px] text-muted-foreground">{meta}</div>}
    </div>
  )
}
