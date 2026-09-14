import { Menu, RefreshCw } from "lucide-react"
import { useState } from "react"
import { Outlet } from "react-router-dom"
import { SidebarNav } from "@/components/layout/sidebar"
import { StatusPill } from "@/components/status/status-dot"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet"
import { useQueryClient } from "@tanstack/react-query"
import { useSystemStatus } from "@/hooks/use-system-status"
import { useSettings } from "@/lib/settings"

export function AppShell() {
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const { settings, setSettings } = useSettings()
  const status = useSystemStatus(settings.machineId)
  const queryClient = useQueryClient()

  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <aside className="hidden w-56 shrink-0 border-r border-sidebar-border bg-sidebar md:block">
        <SidebarNav />
      </aside>

      <Sheet open={mobileNavOpen} onOpenChange={setMobileNavOpen}>
        <SheetContent side="left" className="w-56 bg-sidebar p-0 [&>button]:text-sidebar-foreground">
          <SheetTitle className="sr-only">Navigation</SheetTitle>
          <SidebarNav onNavigate={() => setMobileNavOpen(false)} />
        </SheetContent>
      </Sheet>

      <div className="flex min-h-screen flex-1 flex-col">
        <header className="flex h-14 items-center gap-3 border-b border-border bg-card px-4">
          <Button variant="ghost" size="icon" className="md:hidden" onClick={() => setMobileNavOpen(true)}>
            <Menu className="h-5 w-5" />
          </Button>

          <StatusPill tone={status.isStale ? "neutral" : "healthy"} pulse={!status.isStale}>
            {status.isStale ? "Stale" : "Live"}
          </StatusPill>

          <span className="hidden font-mono text-xs text-muted-foreground sm:inline">{status.note}</span>

          <div className="flex-1" />

          <Select value={settings.machineId} onValueChange={(v) => v && setSettings({ machineId: v })}>
            <SelectTrigger size="sm" className="w-[150px] font-mono text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="172.18.0.3" className="font-mono text-xs">172.18.0.3</SelectItem>
              <SelectItem value="172.18.0.5" className="font-mono text-xs">172.18.0.5</SelectItem>
            </SelectContent>
          </Select>

          <Button
            variant="outline"
            size="sm"
            className="gap-1.5"
            onClick={() => queryClient.invalidateQueries()}
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </Button>
        </header>

        <main className="flex-1 overflow-y-auto p-4 md:p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
