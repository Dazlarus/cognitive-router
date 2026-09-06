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
import { ageWeightedEpisodic } from "./learning_guards.js";

/** Failure type classification for pattern detection. */
export type FailureType = "timeout" | "rate_limit" | "error" | "empty";

/** Pattern flags — aggressive deprioritization signals.
 *
 * These are set when the cost tracker detects repeated failure patterns and
 * cause the model/provider to be skipped or heavily penalized in routing.
 * Flags clear after a successful response + cooldown period. */
export interface PatternFlags {
  /** Model consistently times out on large contexts — skip for large requests. */
  contextLimited?: {
    reason: string;
    setAt: number;
    /** Token threshold above which this model should be skipped. */
    skipAboveTokens: number;
  };
  /** Provider is rate-limited — skip entirely for a cooldown period. */
  throttled?: {
    reason: string;
    setAt: number;
    /** Timestamp when the throttle cooldown expires. */
    expiresAt: number;
  };
  /** Model crashes/500s — skip entirely until manual reset or cooldown. */
  unstable?: {
    reason: string;
    setAt: number;
    /** Timestamp when the unstable flag can be probed. */
    expiresAt: number;
  };
}

export interface ProviderState {
  name: string;
  budget: ProviderBudget;
  status: "healthy" | "throttled" | "circuit_open";
  consecutiveFailures: number;
  /** Type of the current consecutive failure streak. */
  consecutiveFailureType: FailureType | null;
  recentLatencies: number[]; // rolling window
  recentCalls: number; // calls in current window
  monthlySpendUsd: number;
  dailySpendUsd: number;
  lastFailureTime: number;
  /** Escalation tier for graduated backoff (0=none, 1=30s, 2=1min, 3=5min, 4=30min) */
  backoffTier: number;
  /** Token usage tracking for subscription providers (e.g., Z.AI) */
  totalTokensUsed: number;
  /** Current quota consumption as percentage (0-100) */
  quotaPercent: number;
  /** Whether quota warning has been logged for current period */
  quotaWarned: boolean;
  /** Negative signal amplification: pattern flags for aggressive deprioritization. */
  patternFlags: PatternFlags;
  /** History of recent failure types (rolling window) for pattern detection. */
  recentFailureTypes: FailureType[];
  /** Timestamp of last successful response (for recovery cooldown). */
  lastSuccessTime: number;
  abortPenalty?: number;
}

// ─── Request-size latency profiling ───

/** Size buckets for latency profiling. */
export type SizeBucket = "small" | "medium" | "large";

export const SIZE_BUCKETS: Record<SizeBucket, { min: number; max: number }> = {
  small:  { min: 0,      max: 5_000 },
  medium: { min: 5_000,  max: 50_000 },
  large:  { min: 50_000, max: Infinity },
};

/** Classify an estimated token count into a size bucket. */
export function bucketForTokenCount(tokens: number): SizeBucket {
  if (tokens < SIZE_BUCKETS.small.max) return "small";
  if (tokens < SIZE_BUCKETS.medium.max) return "medium";
  return "large";
}

interface BucketLatencyEntry {
  samples: number[];   // rolling window of response times (ms)
  totalCalls: number;
}

/** Static provider latency-tier defaults for when no observed data exists yet.
 *  These encode rough expectations: flash/low-param models are faster on small
 *  inputs, while providers with huge context windows handle large inputs better. */
const PROVIDER_SIZE_LATENCY_DEFAULTS: Record<string, Partial<Record<SizeBucket, number>>> = {
  // Z.AI flash models are optimized for fast small-request turnaround
  zai:        { small: 800,  medium: 2500, large: 8000 },
  // Gemini has 1M context — excellent for large requests
  gemini:     { small: 1000, medium: 3000, large: 5000 },
  openrouter: { small: 1500, medium: 4000, large: 10000 },
  ollama:     { small: 3000, medium: 8000, large: 20000 },
};

const BUCKET_WINDOW_SIZE = 15;

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

// ─── Negative signal amplification constants ───

/** Number of consecutive same-type failures before aggressive deprioritization kicks in. */
const PATTERN_FAILURE_THRESHOLD = 3;

/** Reliability floor when pattern flags are active (near-dead but not zero). */
const PATTERN_DEPRIORITIZED_RELIABILITY = 0.1;

/** Token count above which a context-limited model should be skipped. */
const CONTEXT_LIMITED_THRESHOLD_TOKENS = 50_000;

/** Cooldown after a success before pattern flags clear (prevents flapping). */
const PATTERN_RECOVERY_COOLDOWN_MS = 60_000; // 1 minute

/** How long the unstable flag blocks a model before allowing a probe. */
const UNSTABLE_COOLDOWN_MS = 5 * 60_000; // 5 minutes

/** How long the throttled flag blocks a provider. */
const THROTTLE_COOLDOWN_MS = 2 * 60_000; // 2 minutes

/** Rolling window size for failure type history. */
const FAILURE_TYPE_WINDOW = 10;

/** Threshold for timeout+large-context pattern detection: if 2+ timeouts
 *  occurred on requests above this token count, mark model as context-limited. */
const CONTEXT_TIMEOUT_MIN_OCCURRENCES = 2;

/** Min token count for a request to be considered "large" for timeout correlation. */
const LARGE_REQUEST_THRESHOLD_TOKENS = 30_000;

/** Default daily spend budget (USD) for paid providers. */
const DEFAULT_DAILY_BUDGET_USD = 5.0;
/** Default monthly spend budget (USD) for paid providers. */
const DEFAULT_MONTHLY_BUDGET_USD = 50.0;

export interface HedgeStats {
  total: number;
  primaryWins: number;
  fallbackWins: number;
  bothFailed: number;
  lastOutcome: string | null;
}

export class CostTracker {
  private states = new Map<string, ProviderState>();
  private modelStates = new Map<string, ProviderState>();
  private zombieStallCounts = new Map<string, number>();
  private dailyBudgetUsd: number;
  private monthlyBudgetUsd: number;
  private budgetWarnedProviders = new Set<string>();
  /** Per-provider latency data keyed by size bucket: provider → bucket → entry. */
  private sizeLatency = new Map<string, Partial<Record<SizeBucket, BucketLatencyEntry>>>();
  /** Hedged request outcome tracking for analytics. */
  private hedgeOutcomes: Array<{ result: string; winnerProvider: string; winnerModel: string; loserCancelled: boolean; timestamp: string }> = [];

  /** Z.AI quota multipliers based on time of day */
  private readonly ZAI_QUOTA_MULTIPLIERS = {
    peak: 3,      // 3× during peak hours (Z.AI business hours, e.g., 9-18 UTC+8)
    offPeak: 2,   // 2× during off-peak
    promo: 1,     // 1× during promo period (through Sep 2026)
  };

  constructor(
    private db: DBService,
    private config: CognitiveRouterConfig,
  ) {
    this.dailyBudgetUsd = parseFloat(process.env.ROUTER_DAILY_BUDGET_USD ?? "") || DEFAULT_DAILY_BUDGET_USD;
    this.monthlyBudgetUsd = parseFloat(process.env.ROUTER_MONTHLY_BUDGET_USD ?? "") || DEFAULT_MONTHLY_BUDGET_USD;
  }

  /** Get current quota multiplier for Z.AI based on time of day */
  private getZaiQuotaMultiplier(): number {
    // Promo period through Sep 2026 = 1× all day
    const now = new Date();
    const promoEnd = new Date('2026-09-30T23:59:59Z');
    if (now <= promoEnd) {
      return this.ZAI_QUOTA_MULTIPLIERS.promo;
    }

    // Check if we're in Z.AI business hours (China Standard Time, UTC+8)
    // Peak hours: 09:00-18:00 CST = 01:00-10:00 UTC
    const utcHour = now.getUTCHours();
    if (utcHour >= 1 && utcHour < 10) {
      return this.ZAI_QUOTA_MULTIPLIERS.peak;
    }

    return this.ZAI_QUOTA_MULTIPLIERS.offPeak;
  }

  async refreshProviderStatus(): Promise<void> {
    for (const [name, budget] of Object.entries(this.config.providers)) {
      // Load persisted spend from SQLite
      const monthlySpend = this.db.getSpend(name, "monthly");
      const dailySpend = this.db.getSpend(name, "daily");

      // Restore persisted circuit state from previous session — age-weighted
      // per §4.6: episodic counters (consecutive failures / backoff tier) decay
      // with a 1h half-life since last_check; drift toward healthy. Cold-start = healthy.
      const savedCircuit = this.db.loadCircuitState ? this.db.loadCircuitState(name) : null;
      let restoredFailures = 0;
      let restoredTier = 0;
      let restoredStatus: ProviderState["status"] = "healthy";
      if (savedCircuit) {
        const ageMs = savedCircuit.lastCheck
          ? Math.max(0, Date.now() - new Date(savedCircuit.lastCheck).getTime())
          : 0;
        const decayed = ageWeightedEpisodic(savedCircuit.consecutiveFailures, savedCircuit.backoffTier, ageMs);
        restoredFailures = decayed.consecutiveFailures;
        restoredTier = decayed.backoffTier;
        restoredStatus = (savedCircuit.status as ProviderState["status"]) ?? "healthy";
        // A fully-decayed circuit is no longer evidence of anything.
        if (restoredFailures === 0 && restoredTier === 0) restoredStatus = "healthy";
        logger.info(
          `Restored circuit state for ${name}: ${savedCircuit.status} → ${restoredStatus} ` +
          `(failures=${savedCircuit.consecutiveFailures}→${restoredFailures}, tier=${savedCircuit.backoffTier}→${restoredTier}, ` +
          `age=${(ageMs / 3_600_000).toFixed(1)}h, episodic half-life 1h)`,
        );
      }
      this.states.set(name, {
        name,
        budget,
        status: restoredStatus,
        consecutiveFailures: restoredFailures,
        consecutiveFailureType: null,
        recentLatencies: [],
        recentCalls: 0,
        monthlySpendUsd: monthlySpend,
        dailySpendUsd: dailySpend,
        lastFailureTime: savedCircuit ? Date.now() : 0,
        backoffTier: restoredTier,
        totalTokensUsed: 0,
        quotaPercent: 0,
        quotaWarned: false,
        patternFlags: {},
        recentFailureTypes: [],
        lastSuccessTime: 0,
      });
    }
    logger.info(`Tracking ${this.states.size} providers. Budgets: daily=$${this.dailyBudgetUsd}, monthly=$${this.monthlyBudgetUsd}`);
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
        consecutiveFailureType: null,
        recentLatencies: [],
        recentCalls: 0,
        monthlySpendUsd: 0,
        dailySpendUsd: 0,
        lastFailureTime: 0,
        backoffTier: 0,
        totalTokensUsed: 0,
        quotaPercent: 0,
        quotaWarned: false,
        patternFlags: {},
        recentFailureTypes: [],
        lastSuccessTime: 0,
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

  /** Reset circuit breaker for a specific provider or model.
   * Used by external liveness checks to recover from transient failures. */
  resetCircuit(providerName: string, modelName?: string): void {
    if (modelName) {
      const ms = this.states.get(`${providerName}/${modelName}`);
      if (ms) {
        ms.status = "healthy";
        ms.consecutiveFailures = 0;
        ms.consecutiveFailureType = null;
        ms.backoffTier = 0;
        ms.patternFlags = {};
        ms.recentFailureTypes = [];
      }
    }
    const ps = this.states.get(providerName);
    if (ps) {
      ps.status = "healthy";
      ps.consecutiveFailures = 0;
      ps.consecutiveFailureType = null;
      ps.backoffTier = 0;
      ps.patternFlags = {};
      ps.recentFailureTypes = [];
    }
    if (this.db.saveCircuitState) {
      this.db.saveCircuitState(providerName, "healthy", 0, 0);
    }
  }

  resetAllCircuits(): void {
    for (const state of this.states.values()) {
      state.status = "healthy";
      state.consecutiveFailures = 0;
      state.consecutiveFailureType = null;
      state.backoffTier = 0;
      state.patternFlags = {};
      state.recentFailureTypes = [];
    }
    for (const state of this.modelStates.values()) {
      state.status = "healthy";
      state.consecutiveFailures = 0;
      state.consecutiveFailureType = null;
      state.backoffTier = 0;
      state.patternFlags = {};
      state.recentFailureTypes = [];
    }
  }

  isAvailable(providerName: string, modelName?: string): boolean {
    const providerState = this.getProviderState(providerName);
    if (!providerState || providerState.status === "circuit_open") return false;

    // Check provider-level throttle flag
    if (this.isThrottled(providerName)) return false;

    if (this.usesModelCircuit(providerName, modelName)) {
      const modelState = this.getModelState(providerName, modelName);
      if (!modelState) return false;
      if (modelState.status === "circuit_open") return false;
      // Check model-level unstable flag
      if (this.isUnstable(providerName, modelName)) return false;
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

  /** Compute reliability score based on recent failures + backoff tier + pattern flags */
  getReliabilityScore(providerName: string, modelName?: string): number {
    const providerState = this.getProviderState(providerName);
    if (!providerState || providerState.status === "circuit_open") return 0;

    const state = this.usesModelCircuit(providerName, modelName)
      ? this.getModelState(providerName, modelName)
      : providerState;
    if (!state) return 0;
    if (state.status === "circuit_open") return 0;

    // ─── Negative signal amplification ───
    // If any pattern flags are active and not expired, aggressively deprioritize.
    const now = Date.now();

    // Model-level unstable flag can exist even if usesModelCircuit is false
    const modelState = modelName ? this.getOrCreateModelState(providerName, modelName) : undefined;
    if (modelState && modelState.patternFlags.unstable) {
      if (now < modelState.patternFlags.unstable.expiresAt) {
        return PATTERN_DEPRIORITIZED_RELIABILITY;
      }
      logger.info(`${modelState.name}: unstable flag expired — clearing.`);
      delete modelState.patternFlags.unstable;
    }

    // Unstable flag → near-zero reliability
    if (state.patternFlags.unstable) {
      if (now < state.patternFlags.unstable.expiresAt) {
        return PATTERN_DEPRIORITIZED_RELIABILITY;
      }
      // Expired — clear it
      logger.info(`${state.name}: unstable flag expired — clearing.`);
      delete state.patternFlags.unstable;
    }

    // Throttled flag → near-zero reliability for provider-level
    if (state.patternFlags.throttled) {
      if (now < state.patternFlags.throttled.expiresAt) {
        return PATTERN_DEPRIORITIZED_RELIABILITY;
      }
      // Expired — clear it
      logger.info(`${state.name}: throttled flag expired — clearing.`);
      delete state.patternFlags.throttled;
    }

    // Context-limited flag doesn't affect reliability globally —
    // it only causes the model to be skipped for large requests.
    // The router handles that filtering.

    // ─── Aggressive consecutive failure deprioritization ───
    // 3+ consecutive same-type failures → reliability to 0.1 immediately
    if (
      state.consecutiveFailures >= PATTERN_FAILURE_THRESHOLD &&
      state.consecutiveFailureType
    ) {
      logger.debug(
        `${state.name}: aggressive deprioritization — ${state.consecutiveFailures} consecutive ` +
        `${state.consecutiveFailureType} failures → reliability=${PATTERN_DEPRIORITIZED_RELIABILITY}`,
      );
      return PATTERN_DEPRIORITIZED_RELIABILITY;
    }

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

    const abortPenalty = (state.abortPenalty ?? 0) + (modelState && modelState !== state ? (modelState.abortPenalty ?? 0) : 0);

    return Math.max(0, 1.0 - failurePenalty - tierPenalty - abortPenalty);
  }

  applyAbortPenalty(provider: string, model: string, reason: string = "abort"): void {
    const penaltyAmount = this.config.reliabilityAbortPenalty ?? 0.2;

    const modelState = this.getOrCreateModelState(provider, model);
    if (modelState) {
      const currentPenalty = modelState.abortPenalty ?? 0;
      modelState.abortPenalty = Math.min(1.0, currentPenalty + penaltyAmount);
      logger.info(
        `Applied abort penalty of ${penaltyAmount} to model ${provider}/${model}. ` +
        `New model abort penalty: ${modelState.abortPenalty}`
      );

      if (reason === "stall") {
        const key = `${provider}/${model}`;
        const currentZombies = (this.zombieStallCounts.get(key) ?? 0) + 1;
        this.zombieStallCounts.set(key, currentZombies);
        logger.info(`Stall abort detected. Model ${key} zombie stall count: ${currentZombies}`);

        if (currentZombies >= 3) {
          modelState.patternFlags.unstable = {
            reason: "zombie_stalls",
            setAt: Date.now(),
            expiresAt: Date.now() + UNSTABLE_COOLDOWN_MS,
          };
          logger.warn(`Model ${key} reached 3 zombie stalls — flagged as unstable.`);
        }
      }
    } else {
      const providerState = this.states.get(provider);
      if (providerState) {
        const currentPenalty = providerState.abortPenalty ?? 0;
        providerState.abortPenalty = Math.min(1.0, currentPenalty + penaltyAmount);
        logger.info(
          `Applied abort penalty of ${penaltyAmount} to provider ${provider}. ` +
          `New provider abort penalty: ${providerState.abortPenalty}`
        );
      } else {
        logger.warn(`Could not apply abort penalty: state not found for ${provider}/${model}`);
      }
    }

  }

  // ─── Per-model speed telemetry (TTFT + tokens/sec) ───
  // In-memory EWMA per provider/model; rebuilt from live traffic. Advisory
  // scoring only — losing it on restart just means a few minutes of fallback.
  private modelSpeed = new Map<string, { ttftEwma: number; tpsEwma: number; samples: number }>();

  /** Record one observed speed sample for a model (streamed requests). */
  recordSpeedSample(provider: string, model: string, ttftMs: number, tokensPerSec: number): void {
    const key = `${provider}/${model}`;
    let s = this.modelSpeed.get(key);
    if (!s) {
      this.modelSpeed.set(key, { ttftEwma: ttftMs, tpsEwma: tokensPerSec, samples: 1 });
      return;
    }
    const a = 0.25; // EWMA smoothing
    s.ttftEwma += a * (ttftMs - s.ttftEwma);
    s.tpsEwma += a * (tokensPerSec - s.tpsEwma);
    s.samples++;
  }

  /** Per-model speed score (0–1) from TTFT + throughput. null = not enough samples yet. */
  getModelSpeedScore(provider: string, model: string): number | null {
    const s = this.modelSpeed.get(`${provider}/${model}`);
    if (!s || s.samples < 3) return null;
    // TTFT: ≤400ms = 1.0, ≥4s = 0.0, linear between
    const ttftScore =
      s.ttftEwma <= 400 ? 1 : s.ttftEwma >= 4000 ? 0 : 1 - (s.ttftEwma - 400) / 3600;
    // Throughput: ≥80 tok/s = 1.0, ≤10 tok/s = 0.0, linear between
    const tpsScore =
      s.tpsEwma >= 80 ? 1 : s.tpsEwma <= 10 ? 0 : (s.tpsEwma - 10) / 70;
    return 0.6 * ttftScore + 0.4 * tpsScore;
  }

  /** Compute latency score — per-model measured speed (TTFT/TPS) when
   *  available, otherwise the provider-level rolling average. */
  getLatencyScore(providerName: string, modelName?: string): number {
    if (modelName) {
      const measured = this.getModelSpeedScore(providerName, modelName);
      if (measured !== null) return measured;
    }
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

  // ─── Size-bucket latency profiling ───

  /** Record a latency sample for a provider keyed by request-size bucket. */
  recordSizeLatency(providerName: string, estimatedTokens: number, durationMs: number): void {
    const bucket = bucketForTokenCount(estimatedTokens);
    let profile = this.sizeLatency.get(providerName);
    if (!profile) {
      profile = {};
      this.sizeLatency.set(providerName, profile);
    }
    let entry = profile[bucket];
    if (!entry) {
      entry = { samples: [], totalCalls: 0 };
      profile[bucket] = entry;
    }
    entry.samples.push(durationMs);
    if (entry.samples.length > BUCKET_WINDOW_SIZE) entry.samples.shift();
    entry.totalCalls++;
  }

  /** Get the average observed latency for a provider in a size bucket, or the
   *  static default if no observed data exists yet. Returns undefined if no
   *  data or default is available. */
  getSizeLatencyMs(providerName: string, bucket: SizeBucket): number | undefined {
    const profile = this.sizeLatency.get(providerName);
    const entry = profile?.[bucket];
    if (entry && entry.samples.length > 0) {
      return entry.samples.reduce((a, b) => a + b, 0) / entry.samples.length;
    }
    // Fall back to static defaults
    return PROVIDER_SIZE_LATENCY_DEFAULTS[providerName]?.[bucket];
  }

  /** Compute a size-aware latency score (0–1) for a provider given an estimated
   *  token count. Uses observed data when available, static defaults otherwise. */
  getSizeLatencyScore(providerName: string, estimatedTokens: number): number {
    const bucket = bucketForTokenCount(estimatedTokens);
    const avg = this.getSizeLatencyMs(providerName, bucket);
    if (avg === undefined) return 0.5; // completely unknown — neutral
    // Normalize: <500ms = 1.0, >15s = 0.0
    if (avg < 500) return 1.0;
    if (avg > 15_000) return 0.0;
    return 1.0 - (avg - 500) / 14_500;
  }

  /** Get a human-readable latency profile summary for logging. */
  getSizeLatencySummary(providerName: string): string {
    const parts: string[] = [];
    for (const bucket of ["small", "medium", "large"] as SizeBucket[]) {
      const avg = this.getSizeLatencyMs(providerName, bucket);
      if (avg !== undefined) {
        parts.push(`${bucket}=${Math.round(avg)}ms`);
      }
    }
    return parts.join(" ");
  }

  async recordCall(
    providerName: string,
    result: { durationMs: number; outcome: string },
    modelName?: string,
    estimatedTokens?: number,
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

    if (result.outcome === "success" && modelName) {
      const key = `${providerName}/${modelName}`;
      this.zombieStallCounts.delete(key);
      // Decay abort penalty on success: reduce by 0.1 per successful completion
      const modelState = this.getOrCreateModelState(providerName, modelName);
      if (modelState && modelState.abortPenalty && modelState.abortPenalty > 0) {
        modelState.abortPenalty = Math.max(0, modelState.abortPenalty - 0.1);
        if (modelState.abortPenalty === 0) {
          delete modelState.abortPenalty;
        }
        logger.info(`Abort penalty decayed for ${key}: now ${(modelState.abortPenalty ?? 0).toFixed(1)}`);
      }
    }

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

      // Classify failure type
      const failureType: FailureType =
        result.outcome === "timeout" ? "timeout" :
        isRateLimit ? "rate_limit" :
        result.outcome === "empty" ? "empty" :
        "error";

      // Track consecutive same-type failures
      if (state.consecutiveFailureType !== failureType) {
        state.consecutiveFailureType = failureType;
      }

      // Record failure type in rolling window
      state.recentFailureTypes.push(failureType);
      if (state.recentFailureTypes.length > FAILURE_TYPE_WINDOW) {
        state.recentFailureTypes.shift();
      }

      // ─── Pattern Detection ───
      this.detectPatterns(state, label, failureType, estimatedTokens);

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
        this.db.saveCircuitState(providerName, state.status, state.consecutiveFailures, state.backoffTier);
      } else if (state.consecutiveFailures >= 3) {
        // Generic errors — open circuit after 3 consecutive
        state.status = "circuit_open";
        const newTier = Math.min(state.backoffTier + 1, 4);
        state.backoffTier = newTier;
        logger.warn(
          `${label}: circuit OPENED (tier ${newTier}) after ${state.consecutiveFailures} failures.`,
        );
        this.db.saveCircuitState(providerName, state.status, state.consecutiveFailures, state.backoffTier);
      } else if (state.status === "healthy") {
        state.status = "throttled";
        logger.info(
          `${label}: marked throttled (failure ${state.consecutiveFailures}).`,
        );
      }
    } else {
      // Success — reset failure counter and de-escalate backoff
      const wasFlagged =
        state.patternFlags.unstable ||
        state.patternFlags.throttled ||
        state.patternFlags.contextLimited;

      if (state.consecutiveFailures > 0 || state.backoffTier > 0 || wasFlagged) {
        const now = Date.now();
        const sinceSuccess = now - state.lastSuccessTime;

        // Recovery cooldown: only clear pattern flags after cooldown period
        // to avoid flapping between healthy/unhealthy states
        if (wasFlagged && sinceSuccess < PATTERN_RECOVERY_COOLDOWN_MS && state.lastSuccessTime > 0) {
          logger.info(
            `${label}: recovered but in cooldown (${Math.round((PATTERN_RECOVERY_COOLDOWN_MS - sinceSuccess) / 1000)}s remaining) ` +
            `— keeping pattern flags.`,
          );
          // Still reset failure counters so the model gets a chance
          state.consecutiveFailures = 0;
          state.consecutiveFailureType = null;
        } else {
          logger.info(
            `${label}: recovered — clearing ${state.consecutiveFailures} failures, ` +
            `backoff tier ${state.backoffTier} → 0, clearing pattern flags.`,
          );
          state.consecutiveFailures = 0;
          state.consecutiveFailureType = null;
          state.backoffTier = 0;
          state.status = "healthy";
          state.patternFlags = {};
          state.recentFailureTypes = [];
          state.lastSuccessTime = now;
        }
      } else {
        state.lastSuccessTime = Date.now();
      }
    }
  }

  // ─── Negative Signal Amplification: Pattern Detection ───

  /** Detect failure patterns and set aggressive deprioritization flags.
   *
   * Pattern types:
   * - timeout + large context → context-limited (skip for large requests)
   * - rate_limit → throttled (skip provider for cooldown period)
   * - error/crash (500) → unstable (skip model entirely)
   */
  private detectPatterns(
    state: ProviderState,
    label: string,
    failureType: FailureType,
    estimatedTokens?: number,
  ): void {
    const now = Date.now();

    // ─── Pattern: 3+ consecutive same-type failures → aggressive action ───
    if (state.consecutiveFailures >= PATTERN_FAILURE_THRESHOLD) {
      switch (failureType) {
        case "rate_limit": {
          // Provider is rate-limited — set throttled flag with cooldown
          if (!state.patternFlags.throttled) {
            const expiresAt = now + THROTTLE_COOLDOWN_MS;
            state.patternFlags.throttled = {
              reason: `${state.consecutiveFailures} consecutive rate_limit failures`,
              setAt: now,
              expiresAt,
            };
            logger.warn(
              `${label}: 🚫 THROTTLED — ${state.consecutiveFailures} consecutive rate limits. ` +
              `Provider will be skipped until ${new Date(expiresAt).toLocaleTimeString()}.`,
            );
          } else {
            // Extend cooldown
            state.patternFlags.throttled.expiresAt = now + THROTTLE_COOLDOWN_MS;
          }
          break;
        }
        case "error": {
          // Model is crashing/500ing — mark as unstable
          if (!state.patternFlags.unstable) {
            const expiresAt = now + UNSTABLE_COOLDOWN_MS;
            state.patternFlags.unstable = {
              reason: `${state.consecutiveFailures} consecutive error/crash failures`,
              setAt: now,
              expiresAt,
            };
            logger.warn(
              `${label}: ⚠️ UNSTABLE — ${state.consecutiveFailures} consecutive errors. ` +
              `Model will be skipped until ${new Date(expiresAt).toLocaleTimeString()}.`,
            );
          } else {
            // Extend cooldown
            state.patternFlags.unstable.expiresAt = now + UNSTABLE_COOLDOWN_MS;
          }
          break;
        }
        case "timeout": {
          // Timeout + large context → context-limited
          if (
            estimatedTokens &&
            estimatedTokens >= LARGE_REQUEST_THRESHOLD_TOKENS &&
            !state.patternFlags.contextLimited
          ) {
            state.patternFlags.contextLimited = {
              reason: `Timeouts on large requests (~${estimatedTokens} tokens)`,
              setAt: now,
              skipAboveTokens: Math.max(estimatedTokens - 5_000, LARGE_REQUEST_THRESHOLD_TOKENS),
            };
            logger.warn(
              `${label}: 📏 CONTEXT-LIMITED — timeouts on large requests (~${estimatedTokens} tokens). ` +
              `Model will be skipped for requests > ${state.patternFlags.contextLimited.skipAboveTokens} tokens.`,
            );
          }
          break;
        }
      }
    }

    // ─── Pattern: timeout + large context correlation (even without 3 consecutive) ───
    // If we see 2+ timeouts in the recent window on large requests, mark as context-limited
    if (failureType === "timeout" && estimatedTokens && estimatedTokens >= LARGE_REQUEST_THRESHOLD_TOKENS) {
      const recentLargeTimeouts = state.recentFailureTypes.filter(t => t === "timeout").length;
      if (recentLargeTimeouts >= CONTEXT_TIMEOUT_MIN_OCCURRENCES && !state.patternFlags.contextLimited) {
        const skipAbove = Math.max(estimatedTokens - 5_000, LARGE_REQUEST_THRESHOLD_TOKENS);
        state.patternFlags.contextLimited = {
          reason: `${recentLargeTimeouts} timeouts on requests ~${estimatedTokens} tokens`,
          setAt: now,
          skipAboveTokens: skipAbove,
        };
        logger.warn(
          `${label}: 📏 CONTEXT-LIMITED — ${recentLargeTimeouts} timeouts on large requests. ` +
          `Model will be skipped for requests > ${skipAbove} tokens.`,
        );
      }
    }
  }

  /** Check if a model is flagged as context-limited for the given request size. */
  isContextLimited(providerName: string, modelName: string, estimatedTokens: number): boolean {
    const state = this.usesModelCircuit(providerName, modelName)
      ? this.getModelState(providerName, modelName)
      : this.getProviderState(providerName);
    if (!state?.patternFlags.contextLimited) return false;
    return estimatedTokens >= state.patternFlags.contextLimited.skipAboveTokens;
  }

  /** Check if a provider is currently in throttle cooldown. */
  isThrottled(providerName: string): boolean {
    const state = this.states.get(providerName);
    if (!state?.patternFlags.throttled) return false;
    if (Date.now() < state.patternFlags.throttled.expiresAt) return true;
    // Expired — clean up
    delete state.patternFlags.throttled;
    return false;
  }

  /** Check if a model is flagged as unstable. */
  isUnstable(providerName: string, modelName: string): boolean {
    const state = this.usesModelCircuit(providerName, modelName)
      ? this.getModelState(providerName, modelName)
      : this.getProviderState(providerName);
    if (!state?.patternFlags.unstable) return false;
    if (Date.now() < state.patternFlags.unstable.expiresAt) return true;
    // Expired — clean up
    delete state.patternFlags.unstable;
    return false;
  }

  /** Get active pattern flags for a provider/model (for stats and decision transparency). */
  getPatternFlags(providerName: string, modelName?: string): PatternFlags {
    const state = modelName && this.usesModelCircuit(providerName, modelName)
      ? this.getModelState(providerName, modelName)
      : this.getProviderState(providerName);
    if (!state) return {};
    const now = Date.now();
    const active: PatternFlags = {};

    if (state.patternFlags.contextLimited) {
      active.contextLimited = state.patternFlags.contextLimited;
    }
    if (state.patternFlags.throttled && now < state.patternFlags.throttled.expiresAt) {
      active.throttled = state.patternFlags.throttled;
    } else if (state.patternFlags.throttled) {
      delete state.patternFlags.throttled;
    }
    if (state.patternFlags.unstable && now < state.patternFlags.unstable.expiresAt) {
      active.unstable = state.patternFlags.unstable;
    } else if (state.patternFlags.unstable) {
      delete state.patternFlags.unstable;
    }

    return active;
  }

  /** Get a human-readable summary of all pattern flags across all providers/models. */
  getPatternFlagSummary(): Array<{ entity: string; flag: string; reason: string; expiresAt?: string }> {
    const summaries: Array<{ entity: string; flag: string; reason: string; expiresAt?: string }> = [];
    const now = Date.now();

    for (const state of this.states.values()) {
      if (state.patternFlags.throttled && now < state.patternFlags.throttled.expiresAt) {
        summaries.push({
          entity: state.name,
          flag: "throttled",
          reason: state.patternFlags.throttled.reason,
          expiresAt: new Date(state.patternFlags.throttled.expiresAt).toISOString(),
        });
      }
      if (state.patternFlags.contextLimited) {
        summaries.push({
          entity: state.name,
          flag: "contextLimited",
          reason: state.patternFlags.contextLimited.reason,
        });
      }
      if (state.patternFlags.unstable && now < state.patternFlags.unstable.expiresAt) {
        summaries.push({
          entity: state.name,
          flag: "unstable",
          reason: state.patternFlags.unstable.reason,
          expiresAt: new Date(state.patternFlags.unstable.expiresAt).toISOString(),
        });
      }
    }

    for (const state of this.modelStates.values()) {
      if (state.patternFlags.contextLimited) {
        summaries.push({
          entity: state.name,
          flag: "contextLimited",
          reason: state.patternFlags.contextLimited.reason,
        });
      }
      if (state.patternFlags.unstable && now < state.patternFlags.unstable.expiresAt) {
        summaries.push({
          entity: state.name,
          flag: "unstable",
          reason: state.patternFlags.unstable.reason,
          expiresAt: new Date(state.patternFlags.unstable.expiresAt).toISOString(),
        });
      }
    }

    return summaries;
  }

  /** Get recent failure types for a provider/model (for decision transparency). */
  getRecentFailureTypes(providerName: string, modelName?: string): FailureType[] {
    const state = modelName && this.usesModelCircuit(providerName, modelName)
      ? this.getModelState(providerName, modelName)
      : this.getProviderState(providerName);
    return state?.recentFailureTypes ?? [];
  }

  /**
   * Record dollar spend for a provider. Persists to SQLite for cross-restart tracking.
   * Amount is calculated from token usage × model pricing by the caller.
   */
  async recordUsage(
    providerName: string,
    _model: string,
    usage: any,
  ): Promise<void> {
    const state = this.states.get(providerName);
    if (!state) return;

    if (usage.costUsd) {
      const amount = usage.costUsd;
      state.monthlySpendUsd += amount;
      state.dailySpendUsd += amount;

      // Persist to SQLite
      this.db.recordSpend(providerName, amount, "daily");
      this.db.recordSpend(providerName, amount, "monthly");

      // Check if budget threshold was newly crossed
      this.checkBudgetWarning(providerName);
    }
  }

  /**
   * Record spend directly with a known dollar amount.
   * Used by the proxy when it calculates cost from token counts × model pricing.
   */
  recordSpend(providerName: string, amountUsd: number): void {
    const state = this.states.get(providerName);
    if (!state) return;

    state.monthlySpendUsd += amountUsd;
    state.dailySpendUsd += amountUsd;

    // Persist to SQLite
    this.db.recordSpend(providerName, amountUsd, "daily");
    this.db.recordSpend(providerName, amountUsd, "monthly");

    this.checkBudgetWarning(providerName);
  }

  /** Log a warning when a provider crosses the budget threshold. */
  private checkBudgetWarning(providerName: string): void {
    const state = this.states.get(providerName);
    if (!state) return;

    const dailyExceeded = state.dailySpendUsd >= this.dailyBudgetUsd;
    const monthlyExceeded = state.monthlySpendUsd >= this.monthlyBudgetUsd;
    const warnKey = `${providerName}`;

    if ((dailyExceeded || monthlyExceeded) && !this.budgetWarnedProviders.has(warnKey)) {
      this.budgetWarnedProviders.add(warnKey);
      if (dailyExceeded) {
        logger.warn(
          `Budget: ${providerName} daily spend $${state.dailySpendUsd.toFixed(2)} ` +
          `exceeded daily budget $${this.dailyBudgetUsd.toFixed(2)} — paid models will be skipped.`,
        );
      }
      if (monthlyExceeded) {
        logger.warn(
          `Budget: ${providerName} monthly spend $${state.monthlySpendUsd.toFixed(2)} ` +
          `exceeded monthly budget $${this.monthlyBudgetUsd.toFixed(2)} — paid models will be skipped.`,
        );
      }
    } else if (!dailyExceeded && !monthlyExceeded && this.budgetWarnedProviders.has(warnKey)) {
      // Reset warning when spend drops below threshold (e.g. new day/month)
      this.budgetWarnedProviders.delete(warnKey);
    }
  }

  /** Check if a provider has exceeded its daily or monthly budget. */
  isBudgetExceeded(providerName: string): boolean {
    const state = this.states.get(providerName);
    if (!state) return false;

    return state.dailySpendUsd >= this.dailyBudgetUsd ||
           state.monthlySpendUsd >= this.monthlyBudgetUsd;
  }

  /** Get current daily spend for a provider. */
  getDailySpend(providerName: string): number {
    return this.states.get(providerName)?.dailySpendUsd ?? 0;
  }

  /** Get current monthly spend for a provider. */
  getMonthlySpend(providerName: string): number {
    return this.states.get(providerName)?.monthlySpendUsd ?? 0;
  }

  /** Get the configured daily budget. */
  get dailyBudget(): number {
    return this.dailyBudgetUsd;
  }

  /** Get the configured monthly budget. */
  get monthlyBudget(): number {
    return this.monthlyBudgetUsd;
  }

  getAllStates(): ProviderState[] {
    return Array.from(this.states.values());
  }

  getAllModelStates(): ProviderState[] {
    return Array.from(this.modelStates.values());
  }

  // ─── Hedge Outcome Tracking ───

  /** Record a hedged request outcome for analytics. */
  recordHedgeOutcome(outcome: {
    result: string;
    winnerProvider: string;
    winnerModel: string;
    loserCancelled: boolean;
  }): void {
    this.hedgeOutcomes.push({
      result: outcome.result,
      winnerProvider: outcome.winnerProvider,
      winnerModel: outcome.winnerModel,
      loserCancelled: outcome.loserCancelled,
      timestamp: new Date().toISOString(),
    });

    const tag =
      outcome.result === "primary_win" ? "🏆 Primary retry won" :
      outcome.result === "fallback_win" ? "🏆 Fallback won" :
      "❌ Both failed";
    logger.info(
      `Hedge outcome: ${tag} — winner=${outcome.winnerProvider}/${outcome.winnerModel}, loser_cancelled=${outcome.loserCancelled}`,
    );
  }

  /** Get aggregate hedge statistics. */
  getHedgeStats(): HedgeStats {
    const total = this.hedgeOutcomes.length;
    return {
      total,
      primaryWins: this.hedgeOutcomes.filter((h) => h.result === "primary_win").length,
      fallbackWins: this.hedgeOutcomes.filter((h) => h.result === "fallback_win").length,
      bothFailed: this.hedgeOutcomes.filter((h) => h.result === "both_fail").length,
      lastOutcome: total > 0 ? this.hedgeOutcomes[total - 1].result : null,
    };
  }

  /** Get raw hedge outcome log (for dashboards/API). */
  getHedgeOutcomes(): Array<{ result: string; winnerProvider: string; winnerModel: string; loserCancelled: boolean; timestamp: string }> {
    return [...this.hedgeOutcomes];
  }

  // ─── Z.AI Quota Tracking ───

  /** Record token usage for a subscription provider (e.g., Z.AI) */
  recordTokenUsage(providerName: string, inputTokens: number, outputTokens: number): void {
    const state = this.states.get(providerName);
    if (!state || state.budget.budgetType !== "subscription") {
      return; // Only track for subscription providers
    }

    const totalTokens = inputTokens + outputTokens;
    state.totalTokensUsed += totalTokens;

    // Update quota % based on current multiplier
    this.updateQuotaPercent(providerName);
  }

  /** Update quota percentage for a subscription provider */
  private updateQuotaPercent(providerName: string): void {
    const state = this.states.get(providerName);
    if (!state || state.budget.budgetType !== "subscription") {
      return;
    }

    // Get current quota multiplier (peak/off-peak/promo)
    const multiplier = this.getZaiQuotaMultiplier();

    // Assume 1M base quota (this would come from provider API in production)
    const baseQuota = 1_000_000; // 1M tokens base quota
    const effectiveQuota = baseQuota * multiplier;

    state.quotaPercent = Math.min((state.totalTokensUsed / effectiveQuota) * 100, 100);

    // Log warning if quota > 75% and not already warned
    if (state.quotaPercent > 75 && !state.quotaWarned) {
      state.quotaWarned = true;
      logger.warn(
        `Quota warning: ${providerName} at ${state.quotaPercent.toFixed(1)}% of ${effectiveQuota.toLocaleString()} tokens ` +
        `(multiplier=${multiplier}x, used=${state.totalTokensUsed.toLocaleString()})`,
      );
    }

    // Reset warning if quota drops below 50% (e.g., new period)
    if (state.quotaPercent < 50 && state.quotaWarned) {
      state.quotaWarned = false;
    }
  }

  /** Get current quota percentage for a provider */
  getQuotaPercent(providerName: string): number {
    const state = this.states.get(providerName);
    if (!state || state.budget.budgetType !== "subscription") {
      return 0;
    }
    return state.quotaPercent;
  }

  /** Get total tokens used for a provider */
  getTotalTokensUsed(providerName: string): number {
    const state = this.states.get(providerName);
    if (!state) {
      return 0;
    }
    return state.totalTokensUsed;
  }

  /** Get current quota multiplier for Z.AI */
  getCurrentQuotaMultiplier(): number {
    return this.getZaiQuotaMultiplier();
  }

  /** Reset quota tracking (call at start of new period) */
  resetQuotaTracking(providerName: string): void {
    const state = this.states.get(providerName);
    if (!state) {
      return;
    }
    state.totalTokensUsed = 0;
    state.quotaPercent = 0;
    state.quotaWarned = false;
    logger.info(`Quota tracking reset for ${providerName}`);
  }
}
