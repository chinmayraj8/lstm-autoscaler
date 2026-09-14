import { Eye, EyeOff } from "lucide-react"
import { useEffect, useState } from "react"
import { Panel, PanelHeader, Section } from "@/components/section"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { useHealth, useMachines, useScalingConfig } from "@/hooks/queries"
import { useApi } from "@/hooks/use-api"
import { ApiError } from "@/lib/api"
import { useSettings } from "@/lib/settings"

type TokenStatus = "empty" | "checking" | "ok" | "rejected" | "unknown"

export default function SettingsPage() {
  const { settings, setSettings, resetSettings } = useSettings()
  const machines = useMachines()
  const health = useHealth()
  const config = useScalingConfig()
  const api = useApi()
  const [showToken, setShowToken] = useState(false)
  // "empty" is derived at render time below, never via setState -- it's
  // knowable synchronously from settings.apiToken alone, no need to
  // synchronize with anything external for that case. This state only
  // ever holds the outcome of the actual async check.
  const [checkResult, setCheckResult] = useState<Exclude<TokenStatus, "empty">>("checking")

  // A plain, direct fetch via `api.scalingConfig()` -- deliberately NOT
  // `useScalingConfig()`'s React Query state (tried that first; got it
  // wrong). A query against a bad token can get stuck in TanStack Query's
  // own "pending"/paused limbo and never actually settle to `isError` --
  // the exact reason `_unauthorized` in api.ts already exists as an
  // independent pub-sub rather than something read off `query.state`.
  // This effect is that same pattern applied here: call the endpoint
  // directly, read the real resolved/rejected outcome, done -- this is
  // the check that would have caught a bad/garbage token immediately, at
  // the point of typing it in, rather than only surfacing minutes later
  // as a confusing "stale"/"unreachable" dashboard with no obvious cause
  // (see incident this was added after: the token field had gotten a
  // stray paste of unrelated text into it, which nothing on this page
  // said out loud until the header's own "not authenticated" badge
  // happened to catch it).
  useEffect(() => {
    if (settings.apiToken === "") return
    setCheckResult("checking")
    let cancelled = false
    api
      .scalingConfig()
      .then(() => {
        if (!cancelled) setCheckResult("ok")
      })
      .catch((e: unknown) => {
        if (cancelled) return
        setCheckResult(e instanceof ApiError && e.status === 401 ? "rejected" : "unknown")
      })
    return () => {
      cancelled = true
    }
  }, [api, settings.apiToken])

  const tokenStatus: TokenStatus = settings.apiToken === "" ? "empty" : checkResult

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-lg font-semibold">Settings</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Connection and polling settings for this browser only &mdash; not shared, not backend configuration.
        </p>
      </div>

      <Section title="Connection">
        <Panel className="space-y-4 p-4">
          <div className="space-y-1.5">
            <Label htmlFor="observer-url">Observer service URL</Label>
            <Input
              id="observer-url"
              value={settings.observerUrl}
              onChange={(e) => setSettings({ observerUrl: e.target.value })}
              className="font-mono text-sm"
              placeholder="http://localhost:8000"
            />
            <p className="text-xs text-muted-foreground">
              Reached via <code className="font-mono">kubectl port-forward svc/lstm-autoscaler-observer -n lstm-autoscaler 8000:80</code>
            </p>
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="api-token">API token</Label>
            <div className="relative">
              <Input
                id="api-token"
                type={showToken ? "text" : "password"}
                autoComplete="off"
                value={settings.apiToken}
                onChange={(e) => setSettings({ apiToken: e.target.value })}
                // Trimmed on BLUR, not on every keystroke -- trimming on
                // change looked right but wasn't: since this is a
                // controlled input, trimming on every change strips any
                // space typed while it's still trailing, which is true of
                // basically every space typed sequentially, silently
                // mangling anything with spaces in it as you type it (a
                // real paste is unaffected either way, since it fires one
                // change with the whole string). Trimming once on blur
                // still quietly absorbs a stray leading/trailing space or
                // newline from a paste, without that side effect.
                onBlur={(e) => setSettings({ apiToken: e.target.value.trim() })}
                className="pr-8 font-mono text-sm"
                placeholder="leave blank if the observer has no LSTM_AUTOSCALER_API_TOKEN set"
              />
              <button
                type="button"
                onClick={() => setShowToken((v) => !v)}
                className="absolute inset-y-0 right-1.5 flex items-center text-muted-foreground hover:text-foreground"
                aria-label={showToken ? "Hide token" : "Show token"}
              >
                {showToken ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
              </button>
            </div>
            <p className="text-xs text-muted-foreground">
              Sent as <code className="font-mono">Authorization: Bearer &lt;token&gt;</code> on every request when
              set. Leave blank against an unauthenticated observer &mdash; no header is sent at all, rather than a
              blank one.
            </p>
            {/* A masked password field looks identical whether it holds a
                real 64-char hex token or several sentences of pasted
                prose -- this is the actual, direct fix for that failure
                mode: say plainly, right here, whether the server just
                accepted or rejected what's currently in the field. */}
            {tokenStatus === "checking" && <p className="text-xs text-muted-foreground">Checking token&hellip;</p>}
            {tokenStatus === "ok" && <p className="text-xs text-status-healthy">&#10003; Token accepted by the observer.</p>}
            {tokenStatus === "rejected" && (
              <p className="text-xs text-status-critical">
                &#10007; Rejected by the observer (401) &mdash; this doesn&rsquo;t match its
                LSTM_AUTOSCALER_API_TOKEN. Re-copy just the token value, nothing else.
              </p>
            )}
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="prometheus-url">Prometheus URL</Label>
            <Input
              id="prometheus-url"
              value={settings.prometheusUrl}
              onChange={(e) => setSettings({ prometheusUrl: e.target.value })}
              className="font-mono text-sm"
              placeholder="http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090"
            />
            <p className="text-xs text-muted-foreground">
              Called server-side by the observer pod (<code className="font-mono">/metrics/cpu</code>,{" "}
              <code className="font-mono">/machines</code>), not by your browser &mdash; use Prometheus&rsquo;s
              in-cluster Service DNS name here, not a localhost port-forward address.
            </p>
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="machine-id">Machine</Label>
            <div className="flex gap-2">
              {machines.data?.machines.length ? (
                <Select value={settings.machineId} onValueChange={(v) => v && setSettings({ machineId: v })}>
                  <SelectTrigger id="machine-id" className="w-full font-mono text-sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {machines.data.machines.map((id) => (
                      <SelectItem key={id} value={id} className="font-mono text-sm">
                        {id}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              ) : (
                <Input
                  id="machine-id"
                  value={settings.machineId}
                  onChange={(e) => setSettings({ machineId: e.target.value })}
                  className="font-mono text-sm"
                />
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              {machines.isError
                ? "Could not discover nodes from Prometheus — enter a machine ID manually."
                : "Discovered live from Prometheus; falls back to manual entry if unreachable."}
            </p>
          </div>
        </Panel>
      </Section>

      <Section title="Monitoring">
        <Panel className="space-y-4 p-4">
          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="auto-refresh">Auto-refresh</Label>
              <p className="text-xs text-muted-foreground">Poll the API on the interval below.</p>
            </div>
            <Switch
              id="auto-refresh"
              checked={settings.autoRefresh}
              onCheckedChange={(checked) => setSettings({ autoRefresh: checked })}
            />
          </div>

          <div className="space-y-2">
            <div className="flex justify-between">
              <Label>Refresh interval</Label>
              <span className="font-mono text-sm text-primary">{settings.refreshIntervalSeconds}s</span>
            </div>
            <Slider
              value={[settings.refreshIntervalSeconds]}
              min={15}
              max={120}
              step={5}
              onValueChange={(v) => {
                const next = Array.isArray(v) ? v[0] : v
                if (typeof next === "number") setSettings({ refreshIntervalSeconds: next })
              }}
              disabled={!settings.autoRefresh}
            />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="cpu-trend">Show real CPU trend</Label>
              <p className="text-xs text-muted-foreground">Needs the observer to reach Prometheus at the URL above.</p>
            </div>
            <Switch
              id="cpu-trend"
              checked={settings.showCpuTrend}
              onCheckedChange={(checked) => setSettings({ showCpuTrend: checked })}
            />
          </div>
        </Panel>
      </Section>

      <Section title="Effective backend configuration" description="Read-only &mdash; these are Python constants / startup env vars on the observer, not editable from here">
        <Panel>
          <PanelHeader title="/config + /health" />
          <dl className="grid grid-cols-2 gap-3 p-4 font-mono text-xs sm:grid-cols-3">
            <Row label="Server capacity" value={config.data ? `${config.data.server_capacity_pct}%/replica` : "—"} />
            <Row label="Replica bounds" value={config.data ? `${config.data.min_servers}–${config.data.max_servers}` : "—"} />
            <Row label="Scale step" value={config.data ? `±${config.data.scale_step}/tick` : "—"} />
            <Row label="Safety margin" value={config.data ? `${(config.data.safety_margin * 100).toFixed(0)}%` : "—"} />
            <Row label="Tick interval" value={config.data ? `${config.data.tick_seconds}s` : "—"} />
            <Row label="Under-prov weight" value={config.data ? String(config.data.under_prov_weight) : "—"} />
            <Row label="LSTM model loaded" value={health.data ? String(health.data.lstm_model_loaded) : "—"} />
            <Row label="Live loop running" value={health.data ? String(health.data.live_loop_running) : "—"} />
            <Row label="Observer started" value={health.data ? new Date(health.data.started_at).toLocaleString() : "—"} />
          </dl>
        </Panel>
      </Section>

      <Button variant="outline" size="sm" onClick={resetSettings}>
        Reset to defaults
      </Button>
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase text-muted-foreground">{label}</div>
      <div>{value}</div>
    </div>
  )
}
