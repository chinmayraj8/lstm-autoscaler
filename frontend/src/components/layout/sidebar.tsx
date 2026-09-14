import {
  Activity,
  AlertTriangle,
  Gauge,
  LayoutDashboard,
  LineChart,
  Server,
  Settings as SettingsIcon,
} from "lucide-react"
import { NavLink } from "react-router-dom"
import { cn } from "@/lib/utils"

const NAV_ITEMS = [
  { to: "/", label: "Overview", icon: LayoutDashboard },
  { to: "/monitoring", label: "Monitoring", icon: Activity },
  { to: "/forecasting", label: "Forecasting", icon: LineChart },
  { to: "/scaling", label: "Scaling", icon: Gauge },
  { to: "/workloads", label: "Workloads", icon: Server },
  { to: "/events", label: "Events", icon: AlertTriangle },
] as const

export function SidebarNav({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <div className="flex h-full flex-col">
      <div className="flex h-14 items-center gap-2 border-b border-sidebar-border px-4">
        <div className="flex h-6 w-6 items-center justify-center rounded-sm bg-primary/15 text-primary">
          <Gauge className="h-3.5 w-3.5" />
        </div>
        <div className="leading-tight">
          <div className="text-[13px] font-semibold text-sidebar-foreground">LSTM Autoscaler</div>
          <div className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
            Ops Console
          </div>
        </div>
      </div>

      <nav className="flex-1 space-y-0.5 overflow-y-auto px-2 py-3">
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            onClick={onNavigate}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-2.5 rounded-sm px-3 py-2 text-[13px] font-medium transition-colors",
                isActive
                  ? "bg-sidebar-accent text-sidebar-accent-foreground"
                  : "text-sidebar-foreground/70 hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground",
              )
            }
          >
            <Icon className="h-4 w-4 shrink-0" />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-sidebar-border p-2">
        <NavLink
          to="/settings"
          onClick={onNavigate}
          className={({ isActive }) =>
            cn(
              "flex items-center gap-2.5 rounded-sm px-3 py-2 text-[13px] font-medium transition-colors",
              isActive
                ? "bg-sidebar-accent text-sidebar-accent-foreground"
                : "text-sidebar-foreground/70 hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground",
            )
          }
        >
          <SettingsIcon className="h-4 w-4 shrink-0" />
          Settings
        </NavLink>
      </div>
    </div>
  )
}
