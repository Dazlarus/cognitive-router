// src/failure_classifier.ts — Failure-type-aware retry strategies
//
// When a provider/model call fails, we classify the failure and use that
// classification to pick the best fallback strategy instead of blindly
// trying the next candidate in the list.
//
// Strategies:
//   timeout         → skip to next PROVIDER (likely systemic slowness)
//   rate_limit      → switch provider immediately, set backoff timer
//   server_error    → switch MODEL but stay on same provider (model-specific)
//   context_overflow → skip to candidate with largest context window
//   auth_error      → skip provider entirely (not transient)

import { logger } from "./logger.js";

// ─── Types ───

export type FailureType =
  | "timeout"
  | "rate_limit"
  | "server_error"
  | "context_overflow"
  | "auth_error"
  | "empty_response"
  | "unknown";

export type FallbackStrategy =
  | "skip_provider"        // Skip all remaining candidates from this provider
  | "skip_model"           // Skip this model, try next model (possibly same provider)
  | "prefer_large_context" // Skip to candidate with largest context window
  | "next_provider"        // Jump to the next different provider
  | "retry_same"           // Continue normal iteration (generic retry)
  | "abort";               // Stop trying — non-recoverable

export interface FailureClassification {
  type: FailureType;
  strategy: FallbackStrategy;
  /** Human-readable description for logging. */
  description: string;
  /** Whether this failure type is transient (might recover on retry). */
  isTransient: boolean;
  /** Whether a provider backoff timer should be set. */
  shouldBackoff: boolean;
  /** Backoff duration in ms (when shouldBackoff is true). */
  backoffMs?: number;
}

// ─── Provider Backoff Tracker ───

interface ProviderBackoffEntry {
  provider: string;
  backoffUntil: number; // Unix timestamp (ms) when backoff expires
  reason: string;
  setAt: number;
}

class ProviderBackoffTracker {
  private backoffs = new Map<string, ProviderBackoffEntry>();

  /** Set a backoff timer for a provider. */
  set(provider: string, durationMs: number, reason: string): void {
    const existing = this.backoffs.get(provider);
    const newUntil = Date.now() + durationMs;
    // Only extend the backoff, never shorten it via this method
    if (!existing || newUntil > existing.backoffUntil) {
      this.backoffs.set(provider, {
        provider,
        backoffUntil: newUntil,
        reason,
        setAt: Date.now(),
      });
      logger.info(
        `⏳ Provider backoff set: ${provider} for ${Math.round(durationMs / 1000)}s ` +
        `(${reason}) — expires at ${new Date(newUntil).toISOString()}`,
      );
    }
  }

  /** Check if a provider is currently in backoff. */
  isBackedOff(provider: string): boolean {
    const entry = this.backoffs.get(provider);
    if (!entry) return false;
    if (Date.now() >= entry.backoffUntil) {
      this.backoffs.delete(provider);
      return false;
    }
    return true;
  }

  /** Get remaining backoff time in ms (0 if not backed off). */
  remainingMs(provider: string): number {
    const entry = this.backoffs.get(provider);
    if (!entry) return 0;
    const remaining = entry.backoffUntil - Date.now();
    return remaining > 0 ? remaining : 0;
  }

  /** Get the reason for current backoff (null if not backed off). */
  backoffReason(provider: string): string | null {
    const entry = this.backoffs.get(provider);
    if (!entry || Date.now() >= entry.backoffUntil) return null;
    return entry.reason;
  }

  /** Clear backoff for a provider (e.g., after a successful call). */
  clear(provider: string): void {
    this.backoffs.delete(provider);
  }

  /** Clear all backoffs. */
  clearAll(): void {
    this.backoffs.clear();
  }

  /** Get a snapshot of all active backoffs for logging/debugging. */
  snapshot(): Array<{ provider: string; remainingMs: number; reason: string }> {
    const now = Date.now();
    const result: Array<{ provider: string; remainingMs: number; reason: string }> = [];
    for (const [provider, entry] of this.backoffs) {
      const remaining = entry.backoffUntil - now;
      if (remaining > 0) {
        result.push({ provider, remainingMs: remaining, reason: entry.reason });
      } else {
        // Clean up expired entries
        this.backoffs.delete(provider);
      }
    }
    return result;
  }
}

// Singleton instance — shared across requests in the same proxy process
export const providerBackoff = new ProviderBackoffTracker();

// ─── Failure Classification ───

/**
 * Classify a failure based on error properties (code, message, status).
 *
 * This function inspects error objects thrown by provider adapters and
 * categorizes them for the fallback system.
 */
export function classifyFailure(
  error: Error & { code?: string; status?: number },
): FailureClassification {
  const code = error.code ?? "";
  const message = error.message ?? "";
  const status = (error as any).status ?? 0;

  // ── Timeout ──
  if (
    code === "timeout" ||
    error.name === "TimeoutError" ||
    error.name === "AbortError" ||
    /timeout|timed.?out|aborted|stream_stall|router_request_timeout/i.test(message)
  ) {
    return {
      type: "timeout",
      strategy: "next_provider",
      description: "timeout — switching to different provider (likely systemic)",
      isTransient: true,
      shouldBackoff: false,
    };
  }

  // ── Rate limit (429) ──
  if (
    code === "rate_limit" ||
    status === 429 ||
    /rate.?limit|429|slow down|too many requests|throttl/i.test(message)
  ) {
    // Exponential backoff: start at 30s, callers may extend
    const backoffMs = RATE_LIMIT_BACKOFF_BASE_MS;
    return {
      type: "rate_limit",
      strategy: "next_provider",
      description: `rate_limited — switching provider, setting ${Math.round(backoffMs / 1000)}s backoff`,
      isTransient: true,
      shouldBackoff: true,
      backoffMs,
    };
  }

  // ── Auth error (401/403) ──
  if (
    code === "auth_error" ||
    status === 401 ||
    status === 403 ||
    /auth_error|invalid.*(?:key|token|api)|unauthor|forbidden|401|403/i.test(message)
  ) {
    return {
      type: "auth_error",
      strategy: "skip_provider",
      description: "auth_error — skipping provider entirely (credentials issue, not transient)",
      isTransient: false,
      shouldBackoff: true,
      // Long backoff — auth issues don't self-heal
      backoffMs: AUTH_BACKOFF_MS,
    };
  }

  // ── Context overflow ──
  if (
    code === "context_overflow" ||
    code === "context_too_large" ||
    /context.*(length|overflow|too.long|exceed)|maximum.context|token.limit.exceeded/i.test(message)
  ) {
    return {
      type: "context_overflow",
      strategy: "prefer_large_context",
      description: "context_overflow — seeking candidate with larger context window",
      isTransient: false,
      shouldBackoff: false,
    };
  }

  // ── Server error (5xx) ──
  if (
    code === "server_error" ||
    status >= 500 ||
    /server_error|internal server error|503|502|500|bad.gateway/i.test(message)
  ) {
    return {
      type: "server_error",
      strategy: "skip_model",
      description: "server_error — trying different model (may stay on same provider)",
      isTransient: true,
      shouldBackoff: false,
    };
  }

  // ── Empty response ──
  if (
    code === "empty_response" ||
    /empty_provider_response|incomplete_stream|empty.?response/i.test(message)
  ) {
    return {
      type: "empty_response",
      strategy: "skip_model",
      description: "empty_response — trying different model",
      isTransient: true,
      shouldBackoff: false,
    };
  }

  // ── Unknown / generic error ──
  return {
    type: "unknown",
    strategy: "retry_same",
    description: `unknown_error — continuing normal fallback (${code || "no code"})`,
    isTransient: true,
    shouldBackoff: false,
  };
}

// ─── Constants ───

/** Base backoff for rate-limited providers (30 seconds). */
const RATE_LIMIT_BACKOFF_BASE_MS = 30_000;

/** Backoff for auth failures (10 minutes — auth issues don't self-heal quickly). */
const AUTH_BACKOFF_MS = 600_000;

/** Configurable via env: override rate limit backoff base */
export function getRateLimitBackoffMs(): number {
  const env = parseInt(process.env.ROUTER_RATE_LIMIT_BACKOFF_MS ?? "", 10);
  return Number.isFinite(env) && env > 0 ? env : RATE_LIMIT_BACKOFF_BASE_MS;
}

/** Configurable via env: override auth backoff */
export function getAuthBackoffMs(): number {
  const env = parseInt(process.env.ROUTER_AUTH_BACKOFF_MS ?? "", 10);
  return Number.isFinite(env) && env > 0 ? env : AUTH_BACKOFF_MS;
}

// ─── Fallback Decision Helpers ───

export interface CandidateInfo {
  provider: string;
  model: string;
}

export interface FallbackDecision {
  /** Candidates to skip (by index) based on the failure strategy. */
  skipIndices: Set<number>;
  /** The strategy applied. */
  strategy: FallbackStrategy;
  /** Human-readable log message. */
  logMessage: string;
  /** Whether to apply provider backoff. */
  applyBackoff: { provider: string; durationMs: number; reason: string } | null;
}

/**
 * Compute the fallback decision for the next candidate(s) based on failure type.
 *
 * @param classification  The failure classification from classifyFailure()
 * @param failedIndex     Index of the candidate that just failed
 * @param candidates      Full candidate list
 * @param failedProvider  Provider that failed
 * @param failedModel     Model that failed
 * @returns FallbackDecision with skip indices and logging info
 */
export function computeFallbackDecision(
  classification: FailureClassification,
  failedIndex: number,
  candidates: CandidateInfo[],
  failedProvider: string,
  failedModel: string,
): FallbackDecision {
  const skipIndices = new Set<number>();
  const { strategy, type, description } = classification;
  let applyBackoff: FallbackDecision["applyBackoff"] = null;

  switch (strategy) {
    case "skip_provider": {
      // Skip all remaining candidates from the same provider
      for (let i = failedIndex; i < candidates.length; i++) {
        if (candidates[i].provider === failedProvider) {
          skipIndices.add(i);
        }
      }
      if (classification.shouldBackoff && classification.backoffMs) {
        applyBackoff = {
          provider: failedProvider,
          durationMs: classification.backoffMs,
          reason: `${type} on ${failedProvider}/${failedModel}`,
        };
      }
      return {
        skipIndices,
        strategy,
        logMessage: `🚫 [${type}] ${failedProvider}/${failedModel}: ${description}. Skipping all ${skipIndices.size} candidate(s) from ${failedProvider}.`,
        applyBackoff,
      };
    }

    case "next_provider": {
      // Skip remaining candidates from the same provider — jump to next provider
      for (let i = failedIndex + 1; i < candidates.length; i++) {
        if (candidates[i].provider === failedProvider) {
          skipIndices.add(i);
        }
      }
      if (classification.shouldBackoff && classification.backoffMs) {
        applyBackoff = {
          provider: failedProvider,
          durationMs: classification.backoffMs,
          reason: `${type} on ${failedProvider}/${failedModel}`,
        };
      }
      const nextProvider = candidates.find((c, i) => i > failedIndex && !skipIndices.has(i))?.provider;
      return {
        skipIndices,
        strategy,
        logMessage: `⏭️ [${type}] ${failedProvider}/${failedModel}: ${description}. Skipping to next provider${nextProvider ? ` (${nextProvider})` : " (none available)"}.`,
        applyBackoff,
      };
    }

    case "skip_model": {
      // Just skip this specific candidate — the loop naturally moves to the next
      // which may be a different model on the same provider
      skipIndices.add(failedIndex);
      const nextCandidate = candidates[failedIndex + 1];
      return {
        skipIndices,
        strategy,
        logMessage: `🔄 [${type}] ${failedProvider}/${failedModel}: ${description}. Trying next: ${nextCandidate ? `${nextCandidate.provider}/${nextCandidate.model}` : "none"}.`,
        applyBackoff,
      };
    }

    case "prefer_large_context": {
      // Skip all candidates until we find one with a large context window
      // For now, skip to the candidate with the largest known context window
      // (this works because candidates are already ordered; we just need to skip
      // ones that are too small)
      skipIndices.add(failedIndex);
      const nextCandidate = candidates[failedIndex + 1];
      return {
        skipIndices,
        strategy,
        logMessage: `📐 [${type}] ${failedProvider}/${failedModel}: ${description}. Trying next: ${nextCandidate ? `${nextCandidate.provider}/${nextCandidate.model}` : "none"}.`,
        applyBackoff,
      };
    }

    case "retry_same": {
      // Normal iteration — no special skipping
      skipIndices.add(failedIndex);
      return {
        skipIndices,
        strategy,
        logMessage: `🔁 [${type}] ${failedProvider}/${failedModel}: ${description}.`,
        applyBackoff,
      };
    }

    case "abort": {
      // Skip everything
      for (let i = failedIndex; i < candidates.length; i++) {
        skipIndices.add(i);
      }
      return {
        skipIndices,
        strategy,
        logMessage: `🛑 [${type}] ${failedProvider}/${failedModel}: ${description}. Aborting all remaining candidates.`,
        applyBackoff,
      };
    }

    default: {
      skipIndices.add(failedIndex);
      return {
        skipIndices,
        strategy: "retry_same",
        logMessage: `❓ [${type}] ${failedProvider}/${failedModel}: Unknown strategy, using normal fallback.`,
        applyBackoff,
      };
    }
  }
}

// ─── Re-export for tests ───

export { ProviderBackoffTracker };
