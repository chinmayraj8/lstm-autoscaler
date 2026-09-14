import { useSyncExternalStore } from "react"
import { getUnauthorized, subscribeUnauthorized } from "@/lib/api"

/**
 * True whenever the most recent completed API request got a real 401
 * back -- a wrong or missing bearer token. Subscribes directly to the
 * tiny pub-sub in api.ts (updated the instant `getJson` sees a response),
 * rather than reading it off TanStack Query's cache: a query's own
 * retry/networkMode state is about *when* to refetch, not a reliable
 * signal of "the last real answer was a 401". Lets the UI show a visible
 * "not authenticated" indicator instead of a wrong/missing token just
 * quietly rendering empty charts.
 */
export function useUnauthorized(): boolean {
  return useSyncExternalStore(subscribeUnauthorized, getUnauthorized)
}
