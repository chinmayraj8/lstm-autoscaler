import { Panel, PanelHeader, Section } from "@/components/section"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { DEMO_DEPLOYMENT, DEMO_NAMESPACE } from "@/lib/constants"
import { useMachines, useReplicas } from "@/hooks/queries"
import { useSettings } from "@/lib/settings"

export default function WorkloadsPage() {
  const { settings } = useSettings()
  const replicas = useReplicas()
  const machines = useMachines()

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-semibold">Workloads</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          The real Deployment actuation targets, and every node currently visible to Prometheus.
        </p>
      </div>

      <Section title="Actuation target">
        <Panel>
          <PanelHeader title={`${DEMO_DEPLOYMENT}.${DEMO_NAMESPACE}`} />
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Deployment</TableHead>
                <TableHead>Namespace</TableHead>
                <TableHead>Replicas</TableHead>
                <TableHead>Checked</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              <TableRow>
                <TableCell className="font-mono text-xs">{DEMO_DEPLOYMENT}</TableCell>
                <TableCell className="font-mono text-xs">{DEMO_NAMESPACE}</TableCell>
                <TableCell className="font-mono text-sm font-semibold">
                  {replicas.isError ? "unreachable" : replicas.data?.replicas ?? "…"}
                </TableCell>
                <TableCell className="font-mono text-xs text-muted-foreground">
                  {replicas.data ? new Date(replicas.data.checked_at).toLocaleTimeString() : "—"}
                </TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </Panel>
      </Section>

      <Section title="Tracked nodes" description="Discovered live via Prometheus (node-exporter instance labels)">
        <Panel>
          <PanelHeader
            title="Nodes"
            meta={machines.isError ? "prometheus unreachable" : `${machines.data?.machines.length ?? 0} nodes`}
          />
          {machines.isError ? (
            <p className="px-4 py-6 text-sm text-muted-foreground">
              Could not reach Prometheus at {settings.prometheusUrl}. Check the URL in Settings.
            </p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Machine ID</TableHead>
                  <TableHead>Tracked for scaling</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {machines.data?.machines.map((id) => (
                  <TableRow key={id}>
                    <TableCell className="font-mono text-xs">{id}</TableCell>
                    <TableCell className="font-mono text-xs">
                      {id === settings.machineId ? "selected" : "—"}
                    </TableCell>
                  </TableRow>
                )) ?? (
                  <TableRow>
                    <TableCell colSpan={2} className="text-center text-sm text-muted-foreground">
                      Loading…
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          )}
        </Panel>
      </Section>
    </div>
  )
}
