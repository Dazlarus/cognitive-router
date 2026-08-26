// src/decision_source.ts — Decision-source taxonomy + observability
// (Switchyard easy win #2, docs/switchyard-research-aug25.md §4)

/** Fixed taxonomy of decision sources. Every routing decision is tagged with exactly one. */
export const DECISION_SOURCES = [
  "scored_pick",
  "provider_fallback",
  "circuit_breaker_skip",
  "context_window_skip",
  "tool_policy",
  "last_resort_local",
  "all_exhausted",
] as const;

export type DecisionSource = (typeof DECISION_SOURCES)[number];

/** Response header carrying the upstream model that actually served the request. */
export const SELECTED_MODEL_HEADER = "x-model-router-selected-model";

/** Runtime facts from the proxy request loop used to classify the FINAL decision
 *  source — the source of truth is how the request was actually served, not what
 *  the router predicted before the first attempt. */
export interface ProxyDecisionSourceFacts {
  /** Whether any candidate actually served the request. */
  served: boolean;
  /** Provisional source tagged by the RoutingEngine (scored_pick / context_window_skip / last_resort_local). */
  routerSource?: DecisionSource | null;
  /** Tool-bearing request: candidate list was rebuilt by tool policy (decision winner deferred). */
  usesTools: boolean;
  /** Proxy context-window guard skipped at least one candidate (effective input limit). */
  contextGuardSkipped: boolean;
  /** A candidate was skipped by an open circuit breaker or provider backoff. */
  circuitOrBackoffSkipped: boolean;
  /** A candidate was attempted and failed before the server succeeded. */
  attemptFailed: boolean;
  /** Serving candidate is the appended local emergency model (Ollama last resort at tail). */
  servedByAppendedLastResort: boolean;
}

/** Classify how the serving model was chosen, from proxy runtime facts.
 *  Pure function — unit tested. Precedence (first match wins):
 *    1. all_exhausted        (served=false — nothing served the request)
 *    2. tool_policy          (tools reshaped the candidate list)
 *    3. context_window_skip  (router pre-filter degraded OR proxy guard skipped candidates)
 *    4. last_resort_local    (appended local emergency model served)
 *    5. circuit_breaker_skip (circuit/backoff skipped higher-priority candidates)
 *    6. provider_fallback    (earlier candidates were tried and failed)
 *    7. scored_pick          (default — first choice served)
 */
export function computeProxyDecisionSource(facts: ProxyDecisionSourceFacts): DecisionSource {
  if (!facts.served) return "all_exhausted";
  if (facts.usesTools) return "tool_policy";
  if (facts.routerSource === "context_window_skip" || facts.contextGuardSkipped) {
    return "context_window_skip";
  }
  if (
    facts.servedByAppendedLastResort ||
    facts.routerSource === "last_resort_local"
  ) {
    return "last_resort_local";
  }
  if (facts.circuitOrBackoffSkipped) return "circuit_breaker_skip";
  if (facts.attemptFailed) return "provider_fallback";
  return "scored_pick";
}

/** In-memory per-source decision counters (since process start / last reset). */
class DecisionSourceCounters {
  private counts = new Map<DecisionSource, number>();
  private startedAt = new Date();

  increment(source: DecisionSource): void {
    this.counts.set(source, (this.counts.get(source) ?? 0) + 1);
  }

  get(source: DecisionSource): number {
    return this.counts.get(source) ?? 0;
  }

  /** Snapshot with every taxonomy key present (zeros included) plus a total. */
  snapshot(): Record<string, number> {
    const out: Record<string, number> = {};
    let total = 0;
    for (const s of DECISION_SOURCES) {
      const v = this.get(s);
      out[s] = v;
      total += v;
    }
    out.total = total;
    return out;
  }

  sinceIso(): string {
    return this.startedAt.toISOString();
  }

  reset(): void {
    this.counts.clear();
    this.startedAt = new Date();
  }
}

export const decisionSourceCounters = new DecisionSourceCounters();
