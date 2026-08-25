// src/router.ts — The routing decision engine
// Combines intent classification + model capabilities + cost/health state
// to select the optimal model for each turn.

import { logger } from "./logger.js";
import type { ModelRegistry } from "./model_registry.js";
import type { CostTracker } from "./cost_tracker.js";
import type { DBService } from "./db_service.js";
import type { CognitiveRouterConfig } from "./config.js";
import type { Classification } from "./classifier.js";
import { isGenerationModel, generationRoutingExclusionReason } from "./model_policy.js";
import { bucketForTokenCount, type SizeBucket } from "./cost_tracker.js";
import type { BudgetTracker, CostEfficiency } from "./budget_tracker.js";
import { OllamaWarmthChecker, type WarmthInfo } from "./ollama_warmth.js";
import type { Modality } from "./modality.js";

/** Machine-readable reason a candidate was excluded from scoring before
 *  the ranking stage. Clean, reusable vocabulary for decision logs — the
 *  planned decision-source taxonomy (observability work) should extend
 *  these codes, not replace them. */
export type CandidateSkipReason = "context_window_exceeded";

/** Context window filter result — attached to every decision for observability. */
export interface ContextFilterInfo {
  estimatedTokens: number;
  /** Safety factor applied (0.8 = 80% of context window). */
  safetyFactor: number;
  /** Models that were excluded because their context window was too small.
   *  Each entry carries a machine-readable skipReason for decision logs. */
  filteredOut: Array<{
    provider: string;
    model: string;
    contextWindow: number;
    effectiveLimit: number;
    skipReason: CandidateSkipReason;
  }>;
  /** Largest context window among all known models (for error messaging). */
  maxContextWindow: number;
  /** True when EVERY candidate was filtered out and the router degraded
   *  gracefully: the pre-filter candidate set is kept and the best-scoring
   *  model wins (with a warn). Never hard-fail on a heuristic estimate. */
  allFilteredDegraded?: boolean;
}

/** Modality filter result — attached to decisions for observability. */
export interface ModalityFilterInfo {
  /** Modalities required by the request. */
  requiredModalities: Modality[];
  /** Models that were excluded because they lack a required modality. */
  filteredOut: Array<{
    provider: string;
    model: string;
    modalities: string[];
  }>;
  /** Whether multimodal routing was active. */
  wasMultimodal: boolean;
  /** Vision quality scores for all vision-capable candidates (if vision is required) */
  visionQualityScores?: Array<{
    provider: string;
    model: string;
    visionQuality: number;
  }>;
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
  /** Modality filter details — present when request is multimodal. */
  modalityFilter?: ModalityFilterInfo;
  /** Set when the request requires a modality no model supports.
   *  Context-window overruns no longer produce an error: the filter degrades
   *  gracefully (contextFilter.allFilteredDegraded=true) and the best-scoring
   *  model still serves the request. */
  error?: {
    code: "MODALITY_UNSUPPORTED";
    message: string;
    estimatedTokens?: number;
    maxContextWindow?: number;
    requiredModalities?: Modality[];
    availableModalityModels?: string[];
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
  private budgetTracker: BudgetTracker | null = null;
  private warmthChecker: OllamaWarmthChecker | null = null;

  constructor(
    private registry: ModelRegistry,
    private costTracker: CostTracker,
    private db: DBService,
    private config: CognitiveRouterConfig,
  ) {}

  /** Inject a BudgetTracker instance for budget-aware routing. */
  setBudgetTracker(bt: BudgetTracker): void {
    this.budgetTracker = bt;
  }

  /** Inject an OllamaWarmthChecker for warm model awareness. */
  setWarmthChecker(wc: OllamaWarmthChecker): void {
    this.warmthChecker = wc;
  }

  async decide(
    classification: Classification,
    sessionKey: string,
    context: any = {},
    routingProfile?: string,
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

    // Filter out the excluded provider/model if specified in context
    if (context?.excludeProvider && context?.excludeModel) {
      candidates = candidates.filter(
        (m) => !(m.provider === context.excludeProvider && m.model === context.excludeModel)
      );
    }

    const requiredMods: string[] = context?.requiredModalities ?? [];
    const nonTextMods = requiredMods.filter((r) => r !== "text");

    // Filter out local models that exceed GPU VRAM limit
    const vramLimit = this.config.localVramLimitGb ?? 11;
    candidates = candidates.filter((m) => {
      if (m.isLocal && m.vramRequiredGb && m.vramRequiredGb > vramLimit) {
        logger.info(
          `Skipping ${m.provider}/${m.model} — needs ${m.vramRequiredGb}GB VRAM, limit is ${vramLimit}GB.`,
        );
        return false;
      }
      if (nonTextMods.length === 0) {
        const exclusionReason = generationRoutingExclusionReason(m.provider, m.model);
        if (exclusionReason) {
          logger.debug(`Skipping ${m.provider}/${m.model} — excluded: ${exclusionReason}`);
          return false;
        }
      }
      if (nonTextMods.length === 0 && !isGenerationModel(m.provider, m.model)) {
        logger.debug(`Skipping ${m.provider}/${m.model} — not a generation model.`);
        return false;
      }
      // Filter out ZAI models not covered by the coding plan — UNLESS the
      // request requires a modality (vision/audio) that this model supports.
      // Vision models like glm-4.6v are not in the coding plan but are the
      // only option for multimodal requests.
      if (m.provider === "zai" && m.planEligible === false) {
        const requiredMods: string[] = context?.requiredModalities ?? [];
        const needsVision = requiredMods.includes("vision");
        const needsAudio = requiredMods.includes("audio");
        const supportsNeeded = (needsVision && m.modalities.includes("vision")) ||
                              (needsAudio && m.modalities.includes("audio"));
        if (!supportsNeeded) {
          logger.debug(`Skipping ${m.provider}/${m.model} — not in coding plan (requires credits).`);
          return false;
        }
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

    // ─── Modality filtering ───
    // If the request contains images or audio, filter candidates to only
    // those models that support the required modality.
    let modalityFilter: ModalityFilterInfo | undefined;
    const requiredModalities: Modality[] = context?.requiredModalities ?? [];
    const nonTextModalities = requiredModalities.filter((m) => m !== "text");

    if (nonTextModalities.length > 0) {
      const modalityLabel = nonTextModalities.join(", ");
      const modalityFilteredOut: ModalityFilterInfo["filteredOut"] = [];
      const passingModality: typeof candidates = [];

      for (const m of candidates) {
        const supportsAll = nonTextModalities.every((mod) => m.modalities.includes(mod));
        if (!supportsAll) {
          modalityFilteredOut.push({
            provider: m.provider,
            model: m.model,
            modalities: [...m.modalities],
          });
        } else {
          passingModality.push(m);
        }
      }

      // Log each filtered model
      for (const f of modalityFilteredOut) {
        logger.info(
          `Modality filter: excluded ${f.provider}/${f.model} — ` +
          `needs [${nonTextModalities.join(", ")}] but model only supports [${f.modalities.join(", ")}].`,
        );
      }

      let visionQualityScores: Array<{ provider: string; model: string; visionQuality: number }> | undefined;
      if (requiredModalities.includes("vision")) {
        visionQualityScores = [];
        for (const m of candidates) {
          if (m.modalities.includes("vision") && m.visionQuality !== undefined) {
            visionQualityScores.push({
              provider: m.provider,
              model: m.model,
              visionQuality: m.visionQuality,
            });
          }
        }
      }

      modalityFilter = {
        requiredModalities,
        filteredOut: modalityFilteredOut,
        wasMultimodal: true,
        ...(visionQualityScores ? { visionQualityScores } : {}),
      };

      if (passingModality.length === 0) {
        // No model supports the required modality
        const availableModalityModels = this.registry
          .getAvailableModels(priorityProviders)
          .filter((m) => nonTextModalities.some((mod) => m.modalities.includes(mod)))
          .map((m) => `${m.provider}/${m.model}`);

        logger.warn(
          `Modality filter: ALL ${candidates.length} candidates excluded — ` +
          `no model supports [${modalityLabel}]. ` +
          `Known modality-capable: [${availableModalityModels.slice(0, 5).join(", ")}${availableModalityModels.length > 5 ? "..." : ""}].`,
        );

        return {
          provider: "",
          model: "",
          scores: { capability: 0, reliability: 0, cost: 0, latency: 0 },
          overallScore: 0,
          rationale: `MODALITY_UNSUPPORTED: request requires [${modalityLabel}] ` +
            `but no available model supports it.`,
          modalityFilter,
          error: {
            code: "MODALITY_UNSUPPORTED",
            message: `Request requires modality [${modalityLabel}] but no available ` +
              `model supports it. ${availableModalityModels.length > 0
                ? `Models with some required modalities exist but are unavailable: ${availableModalityModels.slice(0, 3).join(", ")}`
                : "No modality-capable models configured."}`,
            requiredModalities,
            availableModalityModels,
          },
        };
      }

      logger.info(
        `Modality routing: [${modalityLabel}] required — ` +
        `${passingModality.length}/${candidates.length} candidates support it ` +
        `(${passingModality.map((m) => `${m.provider}/${m.model}`).slice(0, 3).join(", ")}${passingModality.length > 3 ? "..." : ""}).`,
      );

      candidates = passingModality;
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
            skipReason: "context_window_exceeded",
          });
        } else {
          passingContext.push(m);
        }
      }

      // Log each filtered model for observability (structured skip reason)
      for (const f of filteredOut) {
        logger.info(
          `Context filter: excluded ${f.provider}/${f.model} (skip_reason=${f.skipReason}) — ` +
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
        // ALL candidates filtered out — degrade gracefully, never hard-fail.
        // Keep the pre-filter candidate set: the best-scoring model wins and
        // the request still gets a chance upstream (the estimate is a
        // heuristic; a wrong estimate must not kill the request). The proxy's
        // per-provider effective-limit guard still skips hopeless candidates.
        const maxModel = candidates.find(
          (m) => m.contextWindow === maxContextWindow,
        );
        const maxEffective = Math.floor(maxContextWindow * CONTEXT_SAFETY_FACTOR);

        contextFilter.allFilteredDegraded = true;

        logger.warn(
          `Context filter: ALL ${candidates.length} candidates excluded ` +
          `(skip_reason=context_window_exceeded) — ~${estimatedTokens} tokens exceeds ` +
          `max effective ${maxEffective} (largest: ${maxModel?.provider}/${maxModel?.model} ` +
          `at ${maxContextWindow.toLocaleString()}). Degrading gracefully: falling back to ` +
          `best-scoring model from the unfiltered candidate set.`,
        );
      } else {
        candidates = passingContext;
      }
    }

    if (sizeBucket) {
      logger.debug(
        `Size-aware routing: ~${estimatedTokens} tokens (${sizeBucket} bucket) — ` +
        `adjustments: ${this.config.providerPriority
          .map((p) => `${p}=${sizeScoreAdjust(p, estimatedTokens) >= 0 ? "+" : ""}${sizeScoreAdjust(p, estimatedTokens).toFixed(3)}`)
          .join(", ")}`,
      );
    }

    // ─── Budget-aware routing ───
    // Check if auto-downgrade should be active (computed once per decision)
    let budgetCostWeightBoost = 1.0;
    let budgetDowngradeActive = false;
    let budgetDowngradeReason = "";
    if (this.budgetTracker) {
      const projection = this.budgetTracker.projectBudget("daily", this.costTracker.dailyBudget);
      if (projection.autoDowngradeActive) {
        budgetCostWeightBoost = projection.costWeightBoost;
        budgetDowngradeActive = true;
        budgetDowngradeReason = projection.downgradeReason;
        logger.info(
          `Budget auto-downgrade ACTIVE: ${budgetDowngradeReason} — ` +
          `cost weight ×${budgetCostWeightBoost.toFixed(1)}, paid providers penalized.`,
        );
      }
    }

    // ─── Cost efficiency scoring ───
    // Compute cost efficiency for all candidates relative to each other
    let costEfficiencyMap: Map<string, CostEfficiency> | null = null;
    if (this.budgetTracker) {
      const efficiencies = this.budgetTracker.computeCostEfficiency(candidates, intent, this.registry);
      costEfficiencyMap = new Map();
      for (const ce of efficiencies) {
        costEfficiencyMap.set(`${ce.provider}/${ce.model}`, ce);
      }
    }

    // ─── Ollama warmth pre-check ───
    // Batch-check all Ollama candidates for GPU residency before scoring.
    const ollamaModels = candidates
      .filter((m) => m.isLocal)
      .map((m) => m.model);
    const warmthMap = new Map<string, WarmthInfo>();
    if (this.warmthChecker && ollamaModels.length > 0) {
      try {
        const batch = await this.warmthChecker.checkWarmthBatch(ollamaModels);
        for (const [name, info] of batch) {
          warmthMap.set(name, info);
        }
      } catch (err) {
        logger.debug(`Warmth check failed (non-fatal): ${err instanceof Error ? err.message : err}`);
      }
    }

    // Score every candidate
    let w = { ...this.config.weights };
    if (routingProfile === "cron") {
      const reliabilityIncrease = w.reliability;
      w.reliability = w.reliability * 2.0;
      w.capability = Math.max(0, w.capability - reliabilityIncrease);
    }

    const scored: RoutingDecision[] = candidates.map((modelEntry) => {
      // Phase-1: UCB exploration read path (learned + k/√n; §4.2). While the
      // learning shadow window is ON this returns the plain learned score —
      // live routing is untouched until gates pass and a human flips it.
      const capabilityScore =
        (typeof this.registry.getExplorationScore === "function"
          ? this.registry.getExplorationScore(modelEntry.provider, modelEntry.model, intent)
          : this.registry.getCapabilityScore(modelEntry.provider, modelEntry.model, intent)) *
        confidence;

      const reliabilityScore =
        this.costTracker.getReliabilityScore(modelEntry.provider, modelEntry.model);
      let costScore = this.costTracker.getCostScore(modelEntry.provider);
      let latencyScore = this.costTracker.getLatencyScore(modelEntry.provider);

      // ─── Ollama warm/cold scoring adjustments ───
      let warmthAdjust = 0;
      let warmthLabel = "";
      const warmth = warmthMap.get(modelEntry.model);
      if (modelEntry.isLocal && warmth) {
        if (warmth.isWarm) {
          // Model is already in GPU memory — free hit, no cold start
          warmthLabel = "warm";
          warmthAdjust = 0.10;
          latencyScore = Math.min(1.0, latencyScore + 0.15);
        } else {
          // Model is cold — estimate cold-start penalty
          warmthLabel = warmth.ollamaReachable ? "cold" : "cold(ollama_unreachable)";
          const coldStartSeconds = warmth.estimatedColdStartMs / 1000;
          warmthAdjust = -0.05 * Math.min(coldStartSeconds / 5, 2);
          latencyScore = Math.max(0.0, latencyScore - 0.20);
        }
      }

      // ─── Budget auto-downgrade: penalize paid providers ───
      if (budgetDowngradeActive) {
        const providerBudget = this.config.providers[modelEntry.provider];
        const budgetType = providerBudget?.budgetType ?? "free";
        const isPaid = budgetType === "pay_per_token" || budgetType === "credits";
        const hasTokenCost = (modelEntry.costPer1kInput ?? 0) > 0 || (modelEntry.costPer1kOutput ?? 0) > 0;
        if (isPaid || hasTokenCost) {
          const penalty = this.budgetTracker!.getPaidProviderPenalty();
          costScore = Math.max(0, costScore - penalty);
        }
      }

      // Apply usage multiplier to cost score
      const usageMultiplier = modelEntry.usageMultiplier ?? 1;
      if (usageMultiplier > 1) {
        costScore = costScore / usageMultiplier;
      }

      // Penalize local models for GPU resource usage — UNLESS already warm
      if (modelEntry.isLocal) {
        const isWarm = warmthMap.get(modelEntry.model)?.isWarm ?? false;
        if (!isWarm) {
          costScore *= 0.85;
        }
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

      // ─── Cost efficiency adjustment ───
      // Models with poor cost efficiency (expensive but not much better) get penalized
      let efficiencyAdjust = 0;
      if (costEfficiencyMap) {
        const ce = costEfficiencyMap.get(`${modelEntry.provider}/${modelEntry.model}`);
        if (ce && !ce.recommended && ce.costRatio > 1.5) {
          // Model costs >1.5x the cheapest but isn't in the top efficiency tier
          // Scale penalty by how overpriced it is relative to capability gain
          const capabilityDelta = ce.capabilityScore - 0.5; // relative to mediocre baseline
          const expectedValue = capabilityDelta / ce.costRatio;
          if (expectedValue < 0.2) {
            // Poor value: penalize up to -0.05
            efficiencyAdjust = Math.max(-0.05, -0.02 * ce.costRatio);
          }
        }
      }

      // Apply budget cost weight boost: when auto-downgrade is active,
      // cost scoring carries more influence in the overall score
      const effectiveCostWeight = w.cost * budgetCostWeightBoost;
      const overall =
        w.capability * capabilityScore +
        w.reliability * reliabilityScore +
        effectiveCostWeight * costScore +
        w.latency * latencyScore +
        sizeAdjust + // additive adjustment (not weighted)
        efficiencyAdjust +
        warmthAdjust; // additive warmth adjustment

      const rationaleParts = [
        `cap=${capabilityScore.toFixed(2)}`,
        `rel=${reliabilityScore.toFixed(2)}`,
        `cost=${costScore.toFixed(2)}`,
        `lat=${latencyScore.toFixed(2)}`,
        `mult=${usageMultiplier}`,
      ];
      if (requiredModalities.includes("vision") && modelEntry.visionQuality !== undefined) {
        rationaleParts.push(`visionQ=${modelEntry.visionQuality.toFixed(2)}`);
      }
      if (sizeAdjust !== 0) {
        rationaleParts.push(`size=${sizeAdjust >= 0 ? "+" : ""}${sizeAdjust.toFixed(3)}`);
      }
      if (efficiencyAdjust !== 0) {
        rationaleParts.push(`eff=${efficiencyAdjust >= 0 ? "+" : ""}${efficiencyAdjust.toFixed(3)}`);
      }
      // Log warm/cold status for Ollama models in decision transparency output
      if (warmthLabel) {
        rationaleParts.push(`ollama=${warmthLabel}`);
      }
      if (warmthAdjust !== 0) {
        rationaleParts.push(`warmth=${warmthAdjust >= 0 ? "+" : ""}${warmthAdjust.toFixed(3)}`);
      }
      if (budgetDowngradeActive) {
        rationaleParts.push(`⚠budget_downgrade`);
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
        if (modalityFilter) runnerUp.modalityFilter = modalityFilter;
        return runnerUp;
      }
    }

    if (contextFilter) best.contextFilter = contextFilter;
    if (modalityFilter) best.modalityFilter = modalityFilter;
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

  async getFallback(
    excludeProvider: string,
    excludeModel: string,
    reason?: string
  ): Promise<RoutingDecision | null> {
    const classification: Classification = {
      intent: "conversation",
      confidence: 1.0,
    };
    const context = {
      excludeProvider,
      excludeModel,
      reason,
    };
    return this.decide(classification, "fallback-session", context);
  }
}
