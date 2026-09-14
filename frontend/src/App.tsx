import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { lazy, Suspense } from "react"
import { BrowserRouter, Route, Routes } from "react-router-dom"
import { AppShell } from "@/components/layout/app-shell"
import { TooltipProvider } from "@/components/ui/tooltip"
import { SettingsProvider } from "@/lib/settings"

// Route-level code splitting -- each page (and its ECharts usage) only
// loads when actually navigated to, instead of one monolithic bundle.
const OverviewPage = lazy(() => import("@/pages/overview"))
const MonitoringPage = lazy(() => import("@/pages/monitoring"))
const ForecastingPage = lazy(() => import("@/pages/forecasting"))
const ScalingPage = lazy(() => import("@/pages/scaling"))
const WorkloadsPage = lazy(() => import("@/pages/workloads"))
const EventsPage = lazy(() => import("@/pages/events"))
const SettingsPage = lazy(() => import("@/pages/settings"))

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
})

function PageFallback() {
  return <div className="p-6 text-sm text-muted-foreground">Loading…</div>
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <SettingsProvider>
        <TooltipProvider delay={150}>
          <BrowserRouter>
            <Suspense fallback={<PageFallback />}>
              <Routes>
                <Route element={<AppShell />}>
                  <Route index element={<OverviewPage />} />
                  <Route path="monitoring" element={<MonitoringPage />} />
                  <Route path="forecasting" element={<ForecastingPage />} />
                  <Route path="scaling" element={<ScalingPage />} />
                  <Route path="workloads" element={<WorkloadsPage />} />
                  <Route path="events" element={<EventsPage />} />
                  <Route path="settings" element={<SettingsPage />} />
                </Route>
              </Routes>
            </Suspense>
          </BrowserRouter>
        </TooltipProvider>
      </SettingsProvider>
    </QueryClientProvider>
  )
}
