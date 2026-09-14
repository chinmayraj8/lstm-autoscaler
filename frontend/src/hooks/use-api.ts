import { useMemo } from "react"
import { createApiClient } from "@/lib/api"
import { useSettings } from "@/lib/settings"

export function useApi() {
  const { settings } = useSettings()
  return useMemo(
    () => createApiClient(settings.observerUrl, settings.apiToken),
    [settings.observerUrl, settings.apiToken],
  )
}
