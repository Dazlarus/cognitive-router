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

export interface RoutingDecision {
  provider: string;
  model: string;
  scores: {
    capability: number;
    reliability: number;
    cost: number;
    latency: number;
  };
  overallScore: number;
  rationale: string;
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
    _context: any,
  ): Promise<RoutingDecision | null> {
    const { intent, confidence } = classification;

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

    // Score every candidate
    const scored = candidates.map((modelEntry) => {
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

      const w = this.config.weights;
      const overall =
        w.capability * capabilityScore +
        w.reliability * reliabilityScore +
        w.cost * costScore +
        w.latency * latencyScore;

      return {
        provider: modelEntry.provider,
        model: modelEntry.model,
        scores: {
          capability: capabilityScore,
          reliability: reliabilityScore,
          cost: costScore,
          latency: latencyScore,
        },
        overallScore: overall,
        rationale: `cap=${capabilityScore.toFixed(2)} rel=${reliabilityScore.toFixed(2)} cost=${costScore.toFixed(2)} lat=${latencyScore.toFixed(2)} mult=${usageMultiplier}`,
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
        return runnerUp;
      }
    }

    logger.debug(
      `Winner: ${best.provider}/${best.model} (${best.overallScore.toFixed(3)}) — ${best.rationale}`,
    );
    return best;
  }
}
