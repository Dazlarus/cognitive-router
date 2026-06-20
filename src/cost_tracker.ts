// src/cost_tracker.ts — Provider cost/budget/health tracking + circuit breaker
//
// Key design decisions:
// - Subscription providers (Z.AI) are PREFERRED in cost scoring — we want to
//   get our money's worth from the prepaid subscription.
// - Subscription providers get HARSH graduated backoff on failures because rate
//   limits can be 5min, 30min, 5hr, or even weekly. We escalate fast.
// - Local providers (Ollama) are treated as fallbacks — they're free but tie up
//   local GPU resources, so they score lower on cost preference.
// - Pay-per-token providers (Requesty) scale cost by remaining budget.

import { logger } from "./logger.js";
import type { DBService } from "./db_service.js";
import type { CognitiveRouterConfig, ProviderBudget } from "./config.js";

export interface ProviderState {
  name: string;
  budget: ProviderBudget;
  status: "healthy" | "throttled" | "circuit_open";
  consecutiveFailures: number;
  recentLatencies: number[]; // rolling window
  recentCalls: number; // calls in current window
  monthlySpendUsd: number;
  lastFailureTime: number;
  /** Escalation tier for graduated backoff (0=none, 1=30s, 2=1min, 3=5min, 4=30min) */
  backoffTier: number;
}

// Unified backoff schedule for all providers (subscription or not)
// Each tier is exponentially harsher. We probe after the tier's cooldown.
const BACKOFF_MS = [
  0,            // tier 0: healthy, no backoff
  30_000,       // tier 1: 30 seconds
  60_000,       // tier 2: 1 minute
  300_000,      // tier 3: 5 minutes
  1_800_000,    // tier 4: 30 minutes (maximum for all providers)
];

const LATENCY_WINDOW_SIZE = 20;

export class CostTracker {
  private states = new Map<string, ProviderState>();
  private modelStates = new Map<string, ProviderState>();

  constructor(
    private db: DBService,
    private config: CognitiveRouterConfig,
  ) {}

  async refreshProviderStatus(): Promise<void> {
    for (const [name, budget] of Object.entries(this.config.providers)) {
      this.states.set(name, {
        name,
        budget,
        status: "healthy",
        consecutiveFailures: 0,
        recentLatencies: [],
        recentCalls: 0,
        monthlySpendUsd: 0,
        lastFailureTime: 0,
        backoffTier: 0,
      });
    }
    logger.info(`Tracking ${this.states.size} providers.`);
  }

  private getBackoffMs(state: ProviderState): number {
    return BACKOFF_MS[state.backoffTier] ?? BACKOFF_MS[4];
  }

  private applyBackoffProbe(state: ProviderState, label: string): ProviderState {
    // Check if backoff period has elapsed — probe (half-open)
    if (state.status === "circuit_open" && state.backoffTier > 0) {
      const elapsed = Date.now() - state.lastFailureTime;
      const backoff = this.getBackoffMs(state);
      if (elapsed > backoff) {
        state.status = "throttled"; // half-open: allow a probe request
        logger.info(
          `${label}: backoff tier ${state.backoffTier} elapsed (${Math.round(backoff / 60_000)}min) — probing.`,
        );
      }
    }

    return state;
  }

  private modelStateKey(providerName: string, modelName: string): string {
    return `${providerName}/${modelName}`;
  }

  private usesModelCircuit(providerName: string, modelName?: string): modelName is string {
    return providerName === "zai" && Boolean(modelName);
  }

  private getOrCreateModelState(providerName: string, modelName: string): ProviderState | undefined {
    const providerState = this.states.get(providerName);
    if (!providerState) return undefined;

    const key = this.modelStateKey(providerName, modelName);
    let state = this.modelStates.get(key);
    if (!state) {
      state = {
        name: key,
        budget: providerState.budget,
        status: "healthy",
        consecutiveFailures: 0,
        recentLatencies: [],
        recentCalls: 0,
        monthlySpendUsd: 0,
        lastFailureTime: 0,
        backoffTier: 0,
      };
      this.modelStates.set(key, state);
    }

    return state;
  }

  getProviderState(name: string): ProviderState | undefined {
    const state = this.states.get(name);
    if (!state) return undefined;
    return this.applyBackoffProbe(state, `Provider ${name}`);
  }

  getModelState(providerName: string, modelName: string): ProviderState | undefined {
    const state = this.getOrCreateModelState(providerName, modelName);
    if (!state) return undefined;
    return this.applyBackoffProbe(state, `Model ${providerName}/${modelName}`);
  }

  isAvailable(providerName: string, modelName?: string): boolean {
    const providerState = this.getProviderState(providerName);
    if (!providerState || providerState.status === "circuit_open") return false;

    if (this.usesModelCircuit(providerName, modelName)) {
      const modelState = this.getModelState(providerName, modelName);
      if (!modelState) return false;
      return modelState.status !== "circuit_open";
    }

    return true;
  }

  /** Compute cost score: higher = more preferred (0-1)
   *
   * Priority order:
   * 1. High priority free remote (OpenRouter) — highest priority among free tiers
   * 2. High priority local (Ollama) — good but ties up local resources
   * 3. Subscription (we already paid — use it) — score boosted
   * 4. Medium priority credits (Gemini) — finite but cheap
   * 5. Pay-per-token (Requesty) — scale by remaining budget
   * 6. Low priority free remote — lowest priority, rate-limited
   */
  getCostScore(providerName: string): number {
    const state = this.getProviderState(providerName);
    if (!state) return 0;

    // If throttled, penalize cost score so alternatives look better
    const throttlePenalty = state.status === "throttled" ? 0.15 : 0;

    // Use priority field if available, otherwise fall back to budget type scoring
    const priority = state.budget.priority ?? "medium";

    switch (priority) {
      case "high":
        // High priority providers (OpenRouter free, Z.AI, Ollama)
        // OpenRouter gets slightly higher score for free remote vs local
        if (state.budget.budgetType === "free" && providerName === "openrouter") {
          return 0.80 - throttlePenalty; // OpenRouter free = highest priority
        }
        if (state.budget.budgetType === "free") {
          return 0.60 - throttlePenalty; // Local Ollama = good but ties up GPU
        }
        return 0.95 - throttlePenalty; // Subscription = prioritize

      case "medium":
        return 0.45 - throttlePenalty;

      case "low":
        return 0.20 - throttlePenalty;

      default:
        return 0.50 - throttlePenalty;
    }
  }

  /** Compute reliability score based on recent failures + backoff tier */
  getReliabilityScore(providerName: string, modelName?: string): number {
    const providerState = this.getProviderState(providerName);
    if (!providerState || providerState.status === "circuit_open") return 0;

    const state = this.usesModelCircuit(providerName, modelName)
      ? this.getModelState(providerName, modelName)
      : providerState;
    if (!state) return 0;
    if (state.status === "circuit_open") return 0;

    // Base reliability on consecutive failures
    const failurePenalty = state.consecutiveFailures * 0.15;

    // Subscription providers get harsher reliability penalties
    // because rate limits are unpredictable and can last hours/days
    let tierPenalty = 0;
    if (state.budget.budgetType === "subscription") {
      tierPenalty = state.backoffTier * 0.20; // each tier = -20%
    } else {
      tierPenalty = state.backoffTier * 0.10;
    }

    return Math.max(0, 1.0 - failurePenalty - tierPenalty);
  }

  /** Compute latency score from rolling average */
  getLatencyScore(providerName: string): number {
    const state = this.getProviderState(providerName);
    if (!state || state.recentLatencies.length === 0) return 0.5; // unknown

    const avg =
      state.recentLatencies.reduce((a, b) => a + b, 0) /
      state.recentLatencies.length;

    // Normalize: <500ms = 1.0, >10s = 0.0, linear between
    if (avg < 500) return 1.0;
    if (avg > 10_000) return 0.0;
    return 1.0 - (avg - 500) / 9_500;
  }

  /** Get the effective cost of a model accounting for usage multiplier
   * This is used for relative cost comparison between models within a provider.
   * Returns a multiplier where 1.0 = standard cost, 2.0 = double quota usage.
   */
  getModelUsageMultiplier(providerName: string, modelName: string): number {
    // This will be queried from ModelRegistry in the router
    // For now, default to 1.0 — the router reads it from registry directly
    return 1.0;
  }

  async recordCall(
    providerName: string,
    result: { durationMs: number; outcome: string },
    modelName?: string,
  ): Promise<void> {
    const providerState = this.states.get(providerName);
    if (!providerState) return;

    // Update latency window
    providerState.recentLatencies.push(result.durationMs);
    if (providerState.recentLatencies.length > LATENCY_WINDOW_SIZE) {
      providerState.recentLatencies.shift();
    }
    providerState.recentCalls++;

    const state = this.usesModelCircuit(providerName, modelName)
      ? this.getOrCreateModelState(providerName, modelName)
      : providerState;
    if (!state) return;

    if (state !== providerState) {
      state.recentLatencies.push(result.durationMs);
      if (state.recentLatencies.length > LATENCY_WINDOW_SIZE) {
        state.recentLatencies.shift();
      }
      state.recentCalls++;
    }

    const label = state === providerState
      ? `Provider ${providerName}`
      : `Model ${providerName}/${modelName}`;

    // Classify the failure type
    const isRateLimit = result.outcome === "rate_limit";
    const isFailure =
      result.outcome === "error" ||
      isRateLimit ||
      result.outcome === "timeout";

    if (isFailure) {
      state.consecutiveFailures++;
      state.lastFailureTime = Date.now();

      // Escalate backoff tier on rate limits
      if (isRateLimit) {
        // Rate limits are unpredictable — escalate fast
        const newTier = Math.min(state.backoffTier + 1, 4);
        if (newTier !== state.backoffTier) {
          const backoffMs = BACKOFF_MS[newTier];
          logger.warn(
            `${label}: RATE LIMIT — escalating to backoff tier ${newTier} ` +
            `(${Math.round(backoffMs / 60_000)}min cooldown).`,
          );
          state.backoffTier = newTier;
        }
        state.status = "circuit_open";
      } else if (state.consecutiveFailures >= 3) {
        // Generic errors — open circuit after 3 consecutive
        state.status = "circuit_open";
        const newTier = Math.min(state.backoffTier + 1, 4);
        state.backoffTier = newTier;
        logger.warn(
          `${label}: circuit OPENED (tier ${newTier}) after ${state.consecutiveFailures} failures.`,
        );
      } else if (state.status === "healthy") {
        state.status = "throttled";
        logger.info(
          `${label}: marked throttled (failure ${state.consecutiveFailures}).`,
        );
      }
    } else {
      // Success — reset failure counter and de-escalate backoff
      if (state.consecutiveFailures > 0 || state.backoffTier > 0) {
        logger.info(
          `${label}: recovered — clearing ${state.consecutiveFailures} failures, ` +
          `backoff tier ${state.backoffTier} → 0.`,
        );
        state.consecutiveFailures = 0;
        state.backoffTier = 0;
        state.status = "healthy";
      }
    }
  }

  async recordUsage(
    providerName: string,
    _model: string,
    usage: any,
  ): Promise<void> {
    const state = this.states.get(providerName);
    if (!state) return;

    if (usage.costUsd) {
      state.monthlySpendUsd += usage.costUsd;
    }
  }

  getAllStates(): ProviderState[] {
    return Array.from(this.states.values());
  }

  getAllModelStates(): ProviderState[] {
    return Array.from(this.modelStates.values());
  }
}
