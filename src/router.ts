// src/router.ts — The routing decision engine
// Combines intent classification + model capabilities + cost/health state
// to select the optimal model for each turn.

import { logger } from "./logger.js";
import type { ModelRegistry } from "./model_registry.js";
import type { CostTracker } from "./cost_tracker.js";
import type { DBService } from "./db_service.js";
import type { CognitiveRouterConfig } from "./config.js";
import type { Classification } from "./classifier.js";
import { isGenerationModel } from "./model_policy.js";
import { bucketForTokenCount, type SizeBucket } from "./cost_tracker.js";

/** Context window filter result — attached to every decision for observability. */
export interface ContextFilterInfo {
  estimatedTokens: number;
  /** Safety factor applied (0.8 = 80% of context window). */
  safetyFactor: number;
  /** Models that were excluded because their context window was too small. */
  filteredOut: Array<{
    provider: string;
    model: string;
    contextWindow: number;
    effectiveLimit: number;
  }>;
  /** Largest context window among all known models (for error messaging). */
  maxContextWindow: number;
}

export interface RoutingDecision {
  provider: string;
  model: string;
  scores: {
    capability: number;
    reliability: number;
    cost: number;
    latency: number;
    sizeAdjust?: number;
  };
  overallScore: number;
  rationale: string;
  candidates?: Array<{
    provider: string;
    model: string;
    scores: RoutingDecision["scores"];
    overallScore: number;
  }>;
  /** Context window filter details — present when estimatedTokens > 0. */
  contextFilter?: ContextFilterInfo;
  /** Set when the request exceeds all available context windows. */
  error?: {
    code: "CONTEXT_TOO_LARGE";
    message: string;
    estimatedTokens: number;
    maxContextWindow: number;
  };
}

/** Context window safety factor — models must have contextWindow * this >= estimatedTokens. */
const CONTEXT_SAFETY_FACTOR = 0.8;

/** Size-aware scoring configuration.
 *
 *  SMALL requests (<5K tokens) get a boost toward low-latency providers.
 *  LARGE requests (>50K tokens) get a boost toward high-throughput providers.
 *  MEDIUM requests get no adjustment (neutral).
 *
 *  The adjustment magnitude is intentionally small (±5–10%) — this nudges
 *  the ranking without overriding strong capability/reliability signals. */
const SIZE_SCORE_BOOST = 0.08;   // max boost for best size-fit provider
const SIZE_SCORE_PENALTY = 0.05; // max penalty for worst size-fit provider

/** Providers that are known to be fast for small requests (<5K tokens).
 *  These get boosted on small requests. */
const LOW_LATENCY_PROVIDERS = new Set(["zai", "gemini"]);
/** Providers with large context windows / high throughput for big requests. */
const HIGH_THROUGHPUT_PROVIDERS = new Set(["gemini", "zai"]);

/** Compute a size-aware score adjustment in range [-SIZE_SCORE_PENALTY, +SIZE_SCORE_BOOST].
 *  Returns 0 for medium-sized requests (no adjustment). */
function sizeScoreAdjust(
  provider: string,
  estimatedTokens: number,
): number {
  const bucket = bucketForTokenCount(estimatedTokens);

  if (bucket === "small") {
    // Boost low-latency providers, penalize those known to be slow on small inputs
    if (LOW_LATENCY_PROVIDERS.has(provider)) return SIZE_SCORE_BOOST;
    // Local models are slower for quick requests due to GPU scheduling overhead
    if (provider === "ollama") return -SIZE_SCORE_PENALTY;
    return 0;
  }

  if (bucket === "large") {
    // Boost providers with large context windows and high throughput
    if (HIGH_THROUGHPUT_PROVIDERS.has(provider)) return SIZE_SCORE_BOOST;
    // Providers with small effective limits are penalized for large requests
    if (provider === "ollama") return -SIZE_SCORE_PENALTY;
    if (provider === "openrouter") return -SIZE_SCORE_PENALTY * 0.5; // free-tier 66K cap
    return 0;
  }

  return 0; // medium — no adjustment
}

export class RoutingEngine {
  constructor(
    private registry: ModelRegistry,
    private costTracker: CostTracker,
    private db: DBService,
    private config: CognitiveRouterConfig,
  ) {}

  async decide(
    classification: Classification,
    sessionKey: string,
    context: any = {},
  ): Promise<RoutingDecision | null> {
    const { intent, confidence } = classification;

    // Extract estimated token count from context (passed by proxy-stream)
    const estimatedTokens: number = context?.estimatedTokens ?? 0;
    const sizeBucket: SizeBucket | null = estimatedTokens > 0
      ? bucketForTokenCount(estimatedTokens)
      : null;

    // Check for manual overrides first
    const override = this.config.overrides.find((o) => o.intent === intent);
    if (override) {
      return {
        provider: override.provider,
        model: override.model,
        scores: { capability: 1, reliability: 1, cost: 1, latency: 1 },
        overallScore: 1,
        rationale: `Manual override: ${override.reason ?? "user-configured"}`,
      };
    }

    // Get ALL providers from priority list (not just healthy ones)
    // The router will handle degradation by retrying with next-best providers
    const priorityProviders = [...this.config.providerPriority];

    // Get all models from registry for these providers
    let candidates = this.registry.getAvailableModels(priorityProviders);

    // Filter out local models that exceed GPU VRAM limit
    const vramLimit = this.config.localVramLimitGb ?? 11;
    candidates = candidates.filter((m) => {
      if (m.isLocal && m.vramRequiredGb && m.vramRequiredGb > vramLimit) {
        logger.info(
          `Skipping ${m.provider}/${m.model} — needs ${m.vramRequiredGb}GB VRAM, limit is ${vramLimit}GB.`,
        );
        return false;
      }
      if (!isGenerationModel(m.provider, m.model)) {
        logger.debug(`Skipping ${m.provider}/${m.model} — embedding-only model.`);
        return false;
      }
      // Filter out ZAI models not covered by the coding plan
      if (m.provider === "zai" && m.planEligible === false) {
        logger.debug(`Skipping ${m.provider}/${m.model} — not in coding plan (requires credits).`);
        return false;
      }
      // ─── Negative signal amplification: skip pattern-flagged models ───
      if (this.costTracker.isUnstable(m.provider, m.model)) {
        const flags = this.costTracker.getPatternFlags(m.provider, m.model);
        logger.info(
          `Skipping ${m.provider}/${m.model} — UNSTABLE: ${flags.unstable?.reason ?? "pattern detected"}.`,
        );
        return false;
      }
      if (this.costTracker.isThrottled(m.provider)) {
        const flags = this.costTracker.getPatternFlags(m.provider);
        logger.info(
          `Skipping ${m.provider}/${m.model} — THROTTLED: ${flags.throttled?.reason ?? "rate limited"}.`,
        );
        return false;
      }
      if (
        estimatedTokens > 0 &&
        this.costTracker.isContextLimited(m.provider, m.model, estimatedTokens)
      ) {
        const flags = this.costTracker.getPatternFlags(m.provider, m.model);
        logger.info(
          `Skipping ${m.provider}/${m.model} — CONTEXT-LIMITED: ${flags.contextLimited?.reason ?? "timeouts on large requests"} ` +
          `(skip above ${flags.contextLimited?.skipAboveTokens ?? "?"} tokens, request ~${estimatedTokens}).`,
        );
        return false;
      }
      return true;
    });

    if (candidates.length === 0) {
      // No models available at all — return a safe default
      logger.warn("No models available — defaulting to ollama/gemma4.");
      return {
        provider: "ollama",
        model: "gemma4:latest",
        scores: { capability: 0, reliability: 0, cost: 1, latency: 0.5 },
        overallScore: 0,
        rationale: "Fallback — no models available",
      };
    }

    // ─── Context window enforcement ───
    // Filter out models whose effective context window (80% of nominal) is
    // smaller than the estimated token count. This prevents the router from
    // selecting a model that will reject or truncate the request.
    let contextFilter: ContextFilterInfo | undefined;
    if (estimatedTokens > 0) {
      const filteredOut: ContextFilterInfo["filteredOut"] = [];
      const passingContext: typeof candidates = [];

      for (const m of candidates) {
        const effectiveLimit = Math.floor(m.contextWindow * CONTEXT_SAFETY_FACTOR);
        if (estimatedTokens > effectiveLimit) {
          filteredOut.push({
            provider: m.provider,
            model: m.model,
            contextWindow: m.contextWindow,
            effectiveLimit,
          });
        } else {
          passingContext.push(m);
        }
      }

      // Log each filtered model for observability
      for (const f of filteredOut) {
        logger.info(
          `Context filter: excluded ${f.provider}/${f.model} — ` +
          `~${estimatedTokens} tokens > ${f.effectiveLimit} limit ` +
          `(80% of ${f.contextWindow.toLocaleString()}).`,
        );
      }

      const maxContextWindow = candidates.reduce(
        (max, m) => Math.max(max, m.contextWindow), 0,
      );

      contextFilter = {
        estimatedTokens,
        safetyFactor: CONTEXT_SAFETY_FACTOR,
        filteredOut,
        maxContextWindow,
      };

      if (passingContext.length === 0) {
        // No model can handle this request size
        const maxModel = candidates.find(
          (m) => m.contextWindow === maxContextWindow,
        );
        const maxEffective = Math.floor(maxContextWindow * CONTEXT_SAFETY_FACTOR);

        logger.warn(
          `Context filter: ALL ${candidates.length} candidates excluded — ` +
          `~${estimatedTokens} tokens exceeds max effective ${maxEffective} ` +
          `(largest: ${maxModel?.provider}/${maxModel?.model} at ${maxContextWindow.toLocaleString()}).`,
        );

        return {
          provider: "",
          model: "",
          scores: { capability: 0, reliability: 0, cost: 0, latency: 0 },
          overallScore: 0,
          rationale: `CONTEXT_TOO_LARGE: ~${estimatedTokens} tokens exceeds ` +
            `max effective context ${maxEffective.toLocaleString()} ` +
            `(80% of ${maxContextWindow.toLocaleString()} from ` +
            `${maxModel?.provider}/${maxModel?.model}).`,
          contextFilter,
          error: {
            code: "CONTEXT_TOO_LARGE",
            message: `Request size ~${estimatedTokens} tokens exceeds all available ` +
              `context windows. Largest: ${maxModel?.provider}/${maxModel?.model} ` +
              `at ${maxContextWindow.toLocaleString()} tokens ` +
              `(effective limit: ${maxEffective.toLocaleString()} at 80% safety margin).`,
            estimatedTokens,
            maxContextWindow,
          },
        };
      }

      candidates = passingContext;
    }

    if (sizeBucket) {
      logger.debug(
        `Size-aware routing: ~${estimatedTokens} tokens (${sizeBucket} bucket) — ` +
        `adjustments: ${this.config.providerPriority
          .map((p) => `${p}=${sizeScoreAdjust(p, estimatedTokens) >= 0 ? "+" : ""}${sizeScoreAdjust(p, estimatedTokens).toFixed(3)}`)
          .join(", ")}`,
      );
    }

    // Score every candidate
    const scored: RoutingDecision[] = candidates.map((modelEntry) => {
      const capabilityScore =
        this.registry.getCapabilityScore(modelEntry.provider, modelEntry.model, intent) *
        confidence;

      const reliabilityScore =
        this.costTracker.getReliabilityScore(modelEntry.provider, modelEntry.model);
      let costScore = this.costTracker.getCostScore(modelEntry.provider);
      const latencyScore = this.costTracker.getLatencyScore(modelEntry.provider);

      // Apply usage multiplier to cost score
      const usageMultiplier = modelEntry.usageMultiplier ?? 1;
      if (usageMultiplier > 1) {
        costScore = costScore / usageMultiplier;
      }

      // Penalize local models slightly — they tie up GPU resources
      if (modelEntry.isLocal) {
        costScore *= 0.85;
      }

      // ─── Size-aware scoring adjustment ───
      // For small requests, boost low-latency providers.
      // For large requests, boost high-throughput providers.
      // This is a gentle nudge, not a hard filter.
      let sizeAdjust = 0;
      if (sizeBucket) {
        sizeAdjust = sizeScoreAdjust(modelEntry.provider, estimatedTokens);

        // If we have observed latency data for this provider+bucket, refine the
        // adjustment based on actual performance rather than static defaults.
        if (sizeAdjust !== 0) {
          const observedLatency = this.costTracker.getSizeLatencyMs(modelEntry.provider, sizeBucket);
          if (observedLatency !== undefined) {
            // Provider is faster than average for this bucket → extra boost
            // Provider is slower than average → reduce boost / increase penalty
            const bucketAvg = this.computeBucketAverageLatency(sizeBucket);
            if (bucketAvg !== undefined && bucketAvg > 0) {
              const ratio = observedLatency / bucketAvg;
              // ratio < 1 means faster than average → scale boost up
              // ratio > 1 means slower than average → scale boost down
              sizeAdjust = sizeAdjust * (2 - Math.min(ratio, 2));
              sizeAdjust = Math.max(-SIZE_SCORE_PENALTY, Math.min(SIZE_SCORE_BOOST, sizeAdjust));
            }
          }
        }
      }

      const w = this.config.weights;
      const overall =
        w.capability * capabilityScore +
        w.reliability * reliabilityScore +
        w.cost * costScore +
        w.latency * latencyScore +
        sizeAdjust; // additive adjustment (not weighted)

      const rationaleParts = [
        `cap=${capabilityScore.toFixed(2)}`,
        `rel=${reliabilityScore.toFixed(2)}`,
        `cost=${costScore.toFixed(2)}`,
        `lat=${latencyScore.toFixed(2)}`,
        `mult=${usageMultiplier}`,
      ];
      if (sizeAdjust !== 0) {
        rationaleParts.push(`size=${sizeAdjust >= 0 ? "+" : ""}${sizeAdjust.toFixed(3)}`);
      }
      // Include pattern flags in rationale for decision transparency
      const patternFlags = this.costTracker.getPatternFlags(modelEntry.provider, modelEntry.model);
      if (patternFlags.contextLimited) {
        rationaleParts.push(`⚠ctx_limited>${patternFlags.contextLimited.skipAboveTokens}`);
      }
      if (patternFlags.unstable) {
        rationaleParts.push(`⚠unstable`);
      }
      if (patternFlags.throttled) {
        rationaleParts.push(`⚠throttled`);
      }

      return {
        provider: modelEntry.provider,
        model: modelEntry.model,
        scores: {
          capability: capabilityScore,
          reliability: reliabilityScore,
          cost: costScore,
          latency: latencyScore,
          ...(sizeAdjust !== 0 ? { sizeAdjust } : {}),
        },
        overallScore: overall,
        rationale: rationaleParts.join(" "),
      };
    });

    // Sort by overall score descending
    scored.sort((a, b) => b.overallScore - a.overallScore);

    // Filter out entries that are unavailable. Z.AI health is tracked per model
    // so one unavailable GLM route does not poison the whole provider.
    const healthyScored = scored.filter((s) => s.scores.reliability > 0);

    if (healthyScored.length === 0) {
      logger.warn(
        `All providers are unhealthy — returning null (no override).`,
      );
      return null; // Signal: don't override, let OpenClaw's native fallback work
    }

    const best = healthyScored[0];
    best.candidates = healthyScored.slice(1, 6);
    const runnerUp = healthyScored[1];

    // If top 2 are close, pick the cheaper one
    if (
      runnerUp &&
      best.overallScore - runnerUp.overallScore < 0.03 // epsilon
    ) {
      const bestCost = this.costTracker.getCostScore(best.provider);
      const runnerCost = this.costTracker.getCostScore(runnerUp.provider);
      if (runnerCost > bestCost) {
        logger.debug(
          `Close call — picking cheaper: ${runnerUp.provider}/${runnerUp.model} over ${best.provider}/${best.model}`,
        );
        if (contextFilter) runnerUp.contextFilter = contextFilter;
        return runnerUp;
      }
    }

    if (contextFilter) best.contextFilter = contextFilter;
    logger.debug(
      `Winner: ${best.provider}/${best.model} (${best.overallScore.toFixed(3)}) — ${best.rationale}`,
    );
    return best;
  }

  /** Compute the average observed latency across all providers for a size bucket.
   *  Used as a baseline for relative size-score refinement. */
  private computeBucketAverageLatency(bucket: SizeBucket): number | undefined {
    const latencies: number[] = [];
    for (const provider of this.config.providerPriority) {
      const ms = this.costTracker.getSizeLatencyMs(provider, bucket);
      if (ms !== undefined) latencies.push(ms);
    }
    if (latencies.length === 0) return undefined;
    return latencies.reduce((a, b) => a + b, 0) / latencies.length;
  }
}
