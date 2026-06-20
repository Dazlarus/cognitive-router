import type { CognitiveRouterConfig } from "./config.js";
import type { CostTracker } from "./cost_tracker.js";
import type { ModelRegistry, ModelCapability } from "./model_registry.js";
import {
  generationRoutingExclusionReason,
  isGenerationModel,
  modelSupportsTools,
  modelTaskTypes,
} from "./model_policy.js";

const ROUTING_INTENTS = [
  "coding",
  "research",
  "creative",
  "conversation",
  "summary",
  "retrieval",
  "science",
  "business",
  "math",
  "analysis",
];

function averageLatency(latencies: number[]): number | null {
  if (latencies.length === 0) return null;
  return Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length);
}

function modelCostScore(model: ModelCapability, providerCostScore: number): number {
  let score = providerCostScore;
  const usageMultiplier = model.usageMultiplier ?? 1;
  if (usageMultiplier > 1) {
    score = score / usageMultiplier;
  }
  if (model.isLocal) {
    score *= 0.85;
  }
  return score;
}

export function buildStatsPayload(
  modelRegistry: ModelRegistry,
  costTracker: CostTracker,
  config: CognitiveRouterConfig,
): Record<string, unknown> {
  const providerPriority = config.providerPriority;
  const weights = config.weights;
  const localVramLimitGb = config.localVramLimitGb ?? 11;

  const providerStates = costTracker.getAllStates().map((s) => ({
    name: s.name,
    status: s.status,
    priorityIndex: providerPriority.indexOf(s.name) === -1 ? null : providerPriority.indexOf(s.name),
    inProviderPriority: providerPriority.includes(s.name),
    budgetType: s.budget.budgetType,
    budgetPriority: s.budget.priority,
    consecutiveFailures: s.consecutiveFailures,
    backoffTier: s.backoffTier,
    monthlySpendUsd: s.monthlySpendUsd,
    recentCalls: s.recentCalls,
    avgLatencyMs: averageLatency(s.recentLatencies),
    scores: {
      reliability: costTracker.getReliabilityScore(s.name),
      cost: costTracker.getCostScore(s.name),
      latency: costTracker.getLatencyScore(s.name),
    },
    available: costTracker.isAvailable(s.name),
  }));

  const modelHealthStates = costTracker.getAllModelStates().map((s) => ({
    name: s.name,
    status: s.status,
    consecutiveFailures: s.consecutiveFailures,
    backoffTier: s.backoffTier,
    recentCalls: s.recentCalls,
    avgLatencyMs: averageLatency(s.recentLatencies),
    available: s.status !== "circuit_open",
  }));

  const models = modelRegistry.getAllModels()
    .map((model) => {
      const providerPriorityIndex = providerPriority.indexOf(model.provider);
      const inProviderPriority = providerPriorityIndex !== -1;
      const providerAvailable = costTracker.isAvailable(model.provider);
      const modelAvailable = costTracker.isAvailable(model.provider, model.model);
      const vramEligible = !model.isLocal || !model.vramRequiredGb || model.vramRequiredGb <= localVramLimitGb;
      const generationCapable = isGenerationModel(model.provider, model.model);
      const generationRoutingExclusion = generationRoutingExclusionReason(model.provider, model.model);
      const reliabilityScore = costTracker.getReliabilityScore(model.provider, model.model);
      const latencyScore = costTracker.getLatencyScore(model.provider);
      const providerCostScore = costTracker.getCostScore(model.provider);
      const costScore = modelCostScore(model, providerCostScore);
      const toolCapable = modelSupportsTools(model.provider, model.model);

      const intentScores = Object.fromEntries(ROUTING_INTENTS.map((intent) => {
        const capability = modelRegistry.getCapabilityScore(model.provider, model.model, intent);
        const overall =
          weights.capability * capability +
          weights.reliability * reliabilityScore +
          weights.cost * costScore +
          weights.latency * latencyScore;

        return [intent, {
          capability,
          reliability: reliabilityScore,
          cost: costScore,
          latency: latencyScore,
          overall,
        }];
      }));

      return {
        id: `${model.provider}/${model.model}`,
        provider: model.provider,
        model: model.model,
        source: model.source,
        contextWindow: model.contextWindow,
        modalities: model.modalities,
        taskTypes: modelTaskTypes(model.provider, model.model),
        toolCapable,
        generationCapable,
        isLocal: model.isLocal,
        vramRequiredGb: model.vramRequiredGb ?? null,
        localVramLimitGb: model.isLocal ? localVramLimitGb : null,
        costsPer1kTokensUsd: {
          input: model.costPer1kInput ?? null,
          output: model.costPer1kOutput ?? null,
        },
        usageMultiplier: model.usageMultiplier ?? 1,
        capabilities: model.capabilities,
        decisionData: {
          providerPriorityIndex: providerPriorityIndex === -1 ? null : providerPriorityIndex,
          inProviderPriority,
          providerAvailable,
          modelAvailable,
          vramEligible,
          generationRoutingExclusion,
          eligibleForRouting: inProviderPriority && providerAvailable && modelAvailable && vramEligible && generationCapable,
          weights,
          providerScores: {
            reliability: reliabilityScore,
            cost: providerCostScore,
            latency: latencyScore,
          },
          effectiveScores: {
            reliability: reliabilityScore,
            cost: costScore,
            latency: latencyScore,
          },
          intentScores,
        },
      };
    })
    .sort((a, b) => {
      const aPriority = a.decisionData.providerPriorityIndex === null
        ? Number.MAX_SAFE_INTEGER
        : a.decisionData.providerPriorityIndex;
      const bPriority = b.decisionData.providerPriorityIndex === null
        ? Number.MAX_SAFE_INTEGER
        : b.decisionData.providerPriorityIndex;
      return aPriority - bPriority ||
        a.provider.localeCompare(b.provider) ||
        a.model.localeCompare(b.model);
    });

  return {
    providers: providerStates,
    modelHealth: modelHealthStates,
    modelCount: models.length,
    models,
    routing: {
      providerPriority,
      weights,
      localVramLimitGb,
      intents: ROUTING_INTENTS,
      scoreFormula: "overall = capability*0.50 + reliability*0.25 + cost*0.15 + latency*0.10",
      fallbackModels: {
        openrouter: {
          tool: process.env.ROUTER_OPENROUTER_TOOL_MODEL ?? "qwen/qwen3-coder:free",
          chat: process.env.ROUTER_OPENROUTER_FALLBACK_MODEL ?? "openrouter/owl-alpha",
        },
        gemini: {
          tool: process.env.ROUTER_GEMINI_TOOL_MODEL ?? "gemini-2.5-flash",
          chat: process.env.ROUTER_GEMINI_FALLBACK_MODEL ?? "gemini-2.5-flash",
        },
      },
      pools: {
        agentGeneration: "chat/generation task types only; excludes embedding-only, vision-only local models, and non-whitelisted Ollama experiments",
        embeddings: "embedding task type only; served through /v1/embeddings, not chat routing",
        vision: "vision task type; reserved for future vision-specific routing",
        localEmergency: "Ollama models with local-emergency task type, used as local fallback/offline pool",
      },
      notes: [
        "intentScores use confidence=1.0; runtime requests multiply capability by classifier confidence",
        "local models receive the same cost penalty used by the router before scoring",
        "tool-capable Ollama models come from ROUTER_OLLAMA_TOOL_MODELS",
        "embedding-only models are listed for inventory and embeddings routing but are not eligible for chat/generation routing",
        "Ollama chat/generation routing is intentionally limited to the local-emergency whitelist",
        "vision models are listed for inventory and future vision routing; local vision models are excluded from normal agent generation",
        "fallbackModels are used when a provider has no seeded registry models, such as Gemini",
      ],
    },
  };
}
