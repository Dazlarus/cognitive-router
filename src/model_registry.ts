// src/model_registry.ts - Model discovery + capability profiles

import { logger } from "./logger.js";
import { registryReadiness } from "./readiness.js";
import type { DBService } from "./db_service.js";
import type { CognitiveRouterConfig } from "./config.js";
import { ChatBenchmark } from "./benchmark_chat.js";
import {
  clampCapability,
  classifyAttribution,
  decayTowardSeed,
  decayIdleDays,
  learningShadowMode,
  normalizeNote,
  noteHash,
  noteCanaryThreshold,
  noteHoldThreshold,
  rocMax24h,
  rocMax7d,
  ucbBonus,
  type GateArm,
} from "./learning_guards.js";

export interface ModelCapability {
  provider: string;
  model: string;
  contextWindow: number;
  modalities: string[]; // text, vision, audio, etc.
  /** Convenience flag: true when modalities includes "vision". */
  supportsVision?: boolean;
  /** Convenience flag: true when modalities includes "audio". */
  supportsAudio?: boolean;
  /** Vision quality score (0.0–1.0). Higher = better vision understanding.
   *  0 or undefined = no vision support. */
  visionQuality?: number;
  // Capability dimensions (0-1), seeded from benchmarks, refined by observation
  capabilities: {
    coding: number;
    reasoning: number;
    creative: number;
    math: number;
    analysis: number;
    conversation: number;
    retrieval: number;
    science: number;
    business: number;
    summary: number;
  };
  // Cost info (per 1M tokens, USD)
  costPer1kInput?: number;
  costPer1kOutput?: number;
  /** Oversubscription multiplier (e.g. x2 means model counts double against quota) */
  usageMultiplier?: number;
  /** VRAM required in GB (for local GPU models). 0 = unknown/N/A */
  vramRequiredGb?: number;
  isLocal: boolean;
  source: "benchmark" | "observed" | "blended" | "auto-bench" | "inferred"; // how current the data is
  /** Whether this model is included in the provider's subscription plan (vs pay-per-credit) */
  planEligible?: boolean;
}

type Caps = ModelCapability["capabilities"];

// ─── Helper ───
function makeModel(
  provider: string,
  id: string,
  ctx: number,
  caps: Caps,
  opts: { input?: number; output?: number; local?: boolean; vision?: boolean; visionQuality?: number; audio?: boolean; usageMultiplier?: number; vram?: number; planEligible?: boolean } = {},
): ModelCapability {
  const modalities: string[] = ["text"];
  const vq = opts.visionQuality;
  if (opts.vision || (vq !== undefined && vq > 0)) modalities.push("vision");
  if (opts.audio) modalities.push("audio");
  return {
    provider,
    model: id,
    contextWindow: ctx,
    modalities,
    supportsVision: modalities.includes("vision"),
    supportsAudio: modalities.includes("audio"),
    visionQuality: modalities.includes("vision") ? (vq ?? 0.5) : undefined,
    capabilities: caps,
    costPer1kInput: opts.input,
    costPer1kOutput: opts.output,
    usageMultiplier: opts.usageMultiplier ?? 1,
    vramRequiredGb: opts.vram,
    isLocal: opts.local ?? false,
    source: "benchmark",
    planEligible: opts.planEligible ?? true,
  };
}

// ─── Seed Data ───
// Z.AI costs are per 1M tokens (API lists per 1M); we store per-1k internally
// So divide by 1000: $1.2/1M → 0.0012/1k

// Z.AI models NOT included in the Coding subscription - exclude from auto-discovery
const ZAI_EXCLUDED_MODELS = new Set([
  "glm-4.5", "glm-4.5-air", "glm-4.5-flash", "glm-4.5v",
  "glm-4.6", "glm-4.6v", // glm-4.6v seeded explicitly as vision-only
  "glm-5", // base glm-5 (not turbo) - not in coding plan
]);

const SEED_MODELS: ModelCapability[] = [
  // ═══════════════════════════════════════════════════════════════
  // Z.AI Flagship Lineup (13 models from OpenClaw config)
  // ═══════════════════════════════════════════════════════════════

  // GLM-5.x - top tier reasoning
  // ✅ Coding Plan eligible (Opus-tier, 3× peak / 2× off-peak quota, 1× promo through Sep 2026)
  makeModel("zai", "glm-5.2", 202_800,
    { coding: 0.88, reasoning: 0.92, creative: 0.84, math: 0.90, analysis: 0.91, conversation: 0.87, retrieval: 0.85, science: 0.93, business: 0.88, summary: 0.87 },
    { input: 0, output: 0, usageMultiplier: 1, planEligible: true },
  ),
  makeModel("zai", "glm-5.1", 202_800,
    { coding: 0.85, reasoning: 0.88, creative: 0.80, math: 0.84, analysis: 0.86, conversation: 0.87, retrieval: 0.83, science: 0.86, business: 0.84, summary: 0.86 },
    { input: 0, output: 0, planEligible: true },
  ),
  // ✅ Coding Plan eligible (Opus-tier)
  makeModel("zai", "glm-5-turbo", 202_800,
    { coding: 0.80, reasoning: 0.82, creative: 0.76, math: 0.78, analysis: 0.80, conversation: 0.84, retrieval: 0.78, science: 0.78, business: 0.80, summary: 0.82 },
    { input: 0, output: 0, usageMultiplier: 1, planEligible: true },
  ),

  // ✅ Coding Plan eligible (Sonnet-tier, standard 1× quota)
  makeModel("zai", "glm-4.7", 204_800,
    { coding: 0.78, reasoning: 0.80, creative: 0.74, math: 0.76, analysis: 0.78, conversation: 0.82, retrieval: 0.78, science: 0.77, business: 0.78, summary: 0.80 },
    { input: 0, output: 0, planEligible: true },
  ),

  // Z.AI Vision+Audio model — used for multimodal routing (not in coding plan)
  // GLM-4.6V supports image and audio input, making it the primary multimodal option.
  makeModel("zai", "glm-4.6v", 64_000,
    { coding: 0.60, reasoning: 0.68, creative: 0.70, math: 0.58, analysis: 0.66, conversation: 0.78, retrieval: 0.64, science: 0.62, business: 0.64, summary: 0.72 },
    { input: 0, output: 0, vision: true, visionQuality: 0.85, audio: true, planEligible: false },
  ),

  // ═══════════════════════════════════════════════════════════════
  // OpenRouter free agent-generation candidates only. Avoid seeding
  // openrouter/free here because it is a random free-model router, not a
  // deterministic model row suitable for agent routing.
  // ═══════════════════════════════════════════════════════════════

  // NOTE: OpenRouter free model availability changes frequently.
  // Last verified: 2026-08-01. Removed dead models:
  //   qwen3-coder:free (moved to paid), owl-alpha (deleted),
  //   qwen3.6-plus:free (deprecated), nemotron-3-super-120b:free (resource exhausted),
  //   poolside/laguna-m.1:free (429 rate-limited, removed 2026-07-28),
  //   nvidia/nemotron-3-ultra-550b-a55b:free (502 resource exhausted, removed 2026-08-01).
  makeModel("openrouter", "cohere/north-mini-code:free", 256_000,
    { coding: 0.74, reasoning: 0.62, creative: 0.50, math: 0.58, analysis: 0.64, conversation: 0.56, retrieval: 0.58, science: 0.56, business: 0.56, summary: 0.58 },
    { input: 0, output: 0 },
  ),
  makeModel("openrouter", "qwen/qwen3-30b-a3b-instruct-2507", 262_000,
    { coding: 0.80, reasoning: 0.82, creative: 0.64, math: 0.74, analysis: 0.80, conversation: 0.72, retrieval: 0.72, science: 0.76, business: 0.70, summary: 0.74 },
    { input: 0, output: 0 },
  ),

  // ═══════════════════════════════════════════════════════════════
  // OpenRouter Paid Fallback (used when free tier can't handle the request)
  // ═══════════════════════════════════════════════════════════════

  makeModel("openrouter", "deepseek/deepseek-v4-flash", 1_000_000,
    { coding: 0.84, reasoning: 0.82, creative: 0.72, math: 0.80, analysis: 0.82, conversation: 0.76, retrieval: 0.74, science: 0.80, business: 0.74, summary: 0.76 },
    { input: 0.09, output: 0.18 }, // per 1K tokens (paid)
  ),

  // ═══════════════════════════════════════════════════════════════
  // Ollama Local Models
  // ═══════════════════════════════════════════════════════════════

  // General purpose
  makeModel("ollama", "gemma4:latest", 128_000,
    { coding: 0.55, reasoning: 0.58, creative: 0.60, math: 0.50, analysis: 0.55, conversation: 0.70, retrieval: 0.65, science: 0.50, business: 0.52, summary: 0.68 },
    { local: true, vram: 9.6 },
  ),
  makeModel("ollama", "gemma3:12b", 8_000,
    { coding: 0.52, reasoning: 0.55, creative: 0.58, math: 0.48, analysis: 0.52, conversation: 0.68, retrieval: 0.62, science: 0.48, business: 0.50, summary: 0.65 },
    { local: true, vision: true, visionQuality: 0.75, vram: 8.1 },
  ),
  makeModel("ollama", "gemma3:4b", 8_000,
    { coding: 0.42, reasoning: 0.44, creative: 0.50, math: 0.38, analysis: 0.42, conversation: 0.60, retrieval: 0.55, science: 0.38, business: 0.40, summary: 0.55 },
    { local: true, vision: true, visionQuality: 0.60, vram: 3.3 },
  ),
  makeModel("ollama", "gemma3:1b", 8_000,
    { coding: 0.30, reasoning: 0.32, creative: 0.40, math: 0.25, analysis: 0.30, conversation: 0.50, retrieval: 0.42, science: 0.25, business: 0.28, summary: 0.45 },
    { local: true, vision: true, visionQuality: 0.40, vram: 0.8 },
  ),
  makeModel("ollama", "mistral:7b", 32_000,
    { coding: 0.50, reasoning: 0.52, creative: 0.62, math: 0.45, analysis: 0.50, conversation: 0.65, retrieval: 0.58, science: 0.45, business: 0.48, summary: 0.60 },
    { local: true, vram: 4.4 },
  ),
  makeModel("ollama", "llama3.1:8b", 128_000,
    { coding: 0.48, reasoning: 0.50, creative: 0.55, math: 0.42, analysis: 0.48, conversation: 0.62, retrieval: 0.55, science: 0.45, business: 0.46, summary: 0.58 },
    { local: true, vram: 4.9 },
  ),
  makeModel("ollama", "deepseek-coder-v2:latest", 128_000,
    { coding: 0.78, reasoning: 0.68, creative: 0.45, math: 0.65, analysis: 0.60, conversation: 0.40, retrieval: 0.50, science: 0.55, business: 0.42, summary: 0.48 },
    { local: true, vram: 8.9 },
  ),
  makeModel("ollama", "qwen2.5-coder:3b", 32_000,
    { coding: 0.62, reasoning: 0.48, creative: 0.35, math: 0.42, analysis: 0.45, conversation: 0.38, retrieval: 0.40, science: 0.38, business: 0.35, summary: 0.42 },
    { local: true, vram: 1.9 },
  ),
  makeModel("ollama", "qwen2.5-coder:1.5b", 32_000,
    { coding: 0.50, reasoning: 0.38, creative: 0.28, math: 0.32, analysis: 0.35, conversation: 0.30, retrieval: 0.32, science: 0.28, business: 0.28, summary: 0.35 },
    { local: true, vram: 1.0 },
  ),
  makeModel("ollama", "deepcoder:1.5b", 16_000,
    { coding: 0.45, reasoning: 0.35, creative: 0.25, math: 0.28, analysis: 0.32, conversation: 0.28, retrieval: 0.30, science: 0.25, business: 0.25, summary: 0.32 },
    { local: true, vram: 1.1 },
  ),
  makeModel("ollama", "starcoder2:instruct", 16_000,
    { coding: 0.55, reasoning: 0.40, creative: 0.30, math: 0.35, analysis: 0.38, conversation: 0.32, retrieval: 0.35, science: 0.30, business: 0.30, summary: 0.35 },
    { local: true, vram: 9.1 },
  ),

  // Reasoning specialists (DeepSeek R1 family) - 32B removed (19GB VRAM - too large for RTX 4070 Ti)
  makeModel("ollama", "deepseek-r1:14b", 128_000,
    { coding: 0.62, reasoning: 0.76, creative: 0.45, math: 0.72, analysis: 0.68, conversation: 0.42, retrieval: 0.50, science: 0.70, business: 0.50, summary: 0.45 },
    { local: true, vram: 9.0 },
  ),
  makeModel("ollama", "deepseek-r1:8b", 128_000,
    { coding: 0.55, reasoning: 0.68, creative: 0.40, math: 0.64, analysis: 0.60, conversation: 0.40, retrieval: 0.45, science: 0.60, business: 0.45, summary: 0.42 },
    { local: true, vram: 5.2 },
  ),
  makeModel("ollama", "deepseek-r1:1.5b", 128_000,
    { coding: 0.38, reasoning: 0.48, creative: 0.30, math: 0.42, analysis: 0.40, conversation: 0.32, retrieval: 0.35, science: 0.38, business: 0.32, summary: 0.35 },
    { local: true, vram: 1.1 },
  ),

  // Vision-capable
  makeModel("ollama", "llama3.2-vision:11b", 128_000,
    { coding: 0.45, reasoning: 0.48, creative: 0.50, math: 0.40, analysis: 0.46, conversation: 0.58, retrieval: 0.52, science: 0.42, business: 0.44, summary: 0.52 },
    { local: true, vision: true, visionQuality: 0.70, vram: 7.8 },
  ),
  makeModel("ollama", "moondream:latest", 8_000,
    { coding: 0.15, reasoning: 0.20, creative: 0.35, math: 0.12, analysis: 0.25, conversation: 0.40, retrieval: 0.30, science: 0.15, business: 0.18, summary: 0.35 },
    { local: true, vision: true, visionQuality: 0.50, vram: 1.7 },
  ),

  // New additions - high-quality models that fit 11GB limit
  makeModel("ollama", "qwen2.5-coder:7b", 128_000,
    { coding: 0.78, reasoning: 0.55, creative: 0.38, math: 0.70, analysis: 0.60, conversation: 0.42, retrieval: 0.50, science: 0.62, business: 0.45, summary: 0.48 },
    { local: true, vram: 4.7 },
  ),
  makeModel("ollama", "phi4:14b", 16_000,
    { coding: 0.55, reasoning: 0.82, creative: 0.42, math: 0.80, analysis: 0.75, conversation: 0.45, retrieval: 0.48, science: 0.85, business: 0.50, summary: 0.48 },
    { local: true, vram: 9.1 },
  ),
  makeModel("ollama", "mistral-nemo:12b", 128_000,
    { coding: 0.60, reasoning: 0.65, creative: 0.52, math: 0.60, analysis: 0.62, conversation: 0.58, retrieval: 0.62, science: 0.65, business: 0.60, summary: 0.58 },
    { local: true, vram: 7.1 },
  ),
];

export class ModelRegistry {
  private models = new Map<string, ModelCapability>();
  /** Learned sample counts per provider/model/intent (UCB n). */
  private sampleCounts = new Map<string, number>();
  /** Manually pinned cells the learner refuses to overwrite. */
  private pinnedCells = new Set<string>();
  private static readonly INTENT_MAP: Record<string, keyof Caps> = {
    coding: "coding",
    research: "retrieval",
    creative: "creative",
    conversation: "conversation",
    summary: "summary",
    "doc-summary": "summary",
    retrieval: "retrieval",
    science: "science",
    business: "business",
    math: "math",
    analysis: "analysis",
  };

  constructor(
    private db: DBService,
    private config: CognitiveRouterConfig,
  ) {}

  async loadCachedState(): Promise<void> {
    // Seed with known models
    for (const m of SEED_MODELS) {
      this.models.set(`${m.provider}/${m.model}`, m);
    }

    // Live-discover models from provider APIs
    await this.discoverModels();

    // Auto-benchmark discovered Ollama chat models (non-blocking — runs after server is ready)
    this.benchmarkDiscoveredModels().catch(err =>
      logger.warn("Auto-benchmark failed: " + (err instanceof Error ? err.message : err))
    );

    // Apply any saved capability overrides from the judge feedback loop
    this.applySavedOverrides();

    registryReadiness.setSynced();

    logger.info(`Model registry loaded - ${this.models.size} models tracked.`);
  }

  /** Auto-benchmark discovered and inferred Ollama chat models using the
   *  persistent multi-probe ChatBenchmark system.
   *  Replaces the old shallow single-probe heuristic.
   *  Skips models that have fresh cached results with matching version hash. */
  private async benchmarkDiscoveredModels(): Promise<void> {
    const EMBEDDING_PREFIXES = ["nomic-embed", "bge", "qwen3-embedding", "embeddinggemma"];
    const toBenchmark: Array<{ provider: string; model: string }> = [];

    // Collect all models that need benchmarking
    for (const [key, cap] of this.models) {
      // Skip embedding models
      if (cap.model.split(":")[0].includes("embed")) continue;
      if (EMBEDDING_PREFIXES.some(p => cap.model.toLowerCase().startsWith(p))) continue;

      // For inferred models: always benchmark
      // For seeded models: benchmark only if no cached results exist
      if (cap.source === "inferred") {
        toBenchmark.push({ provider: cap.provider, model: cap.model });
      } else if (cap.source === "benchmark" || cap.source === "auto-bench") {
        // Check if we have fresh cached results
        const modelId = `${cap.provider}/${cap.model}`;
        const cached = this.db.getAllLatestChatBenchmarks(modelId);
        const probeTypes = ["coding", "reasoning", "conversation"];
        const hasAllFresh = probeTypes.every(pt => {
          const entry = cached.get(pt);
          if (!entry) return false;
          const age = (Date.now() - new Date(entry.timestamp).getTime()) / 86_400_000;
          return age < 7;
        });
        if (!hasAllFresh) {
          toBenchmark.push({ provider: cap.provider, model: cap.model });
        }
      }
    }

    if (toBenchmark.length === 0) {
      logger.info("Chat benchmark: all models have fresh cached results, skipping.");
      return;
    }

    logger.info(`Chat benchmark: ${toBenchmark.length} model(s) need probing.`);

    const bench = new ChatBenchmark(this.db, this);
    await bench.benchmarkModels(toBenchmark);
  }

  /** Discover available models from Z.AI and Ollama APIs at startup */
  private async discoverModels(): Promise<void> {
    // Discover Z.AI models
    try {
      const zaiApiKey = process.env.ZAI_API_KEY;
      if (!zaiApiKey) {
        logger.debug("Skipping Z.AI model discovery: ZAI_API_KEY is not set.");
      } else {
        const resp = await fetch("https://api.z.ai/api/paas/v4/models", {
          headers: { Authorization: `Bearer ${zaiApiKey}` },
          signal: AbortSignal.timeout(8000),
        });
        if (resp.ok) {
          const data = await resp.json() as any;
          const remoteIds = new Set<string>();
          for (const m of data.data ?? []) {
            if (m.id) remoteIds.add(m.id);
          }
          for (const id of remoteIds) {
            if (!this.models.has(`zai/${id}`)) {
              if (ZAI_EXCLUDED_MODELS.has(id)) {
                logger.info(`Skipping non-subscription Z.AI model: zai/${id}`);
                continue;
              }
              logger.info(`Discovered unseeded Z.AI model: zai/${id} - using neutral defaults`);
              this.models.set(`zai/${id}`, makeModel("zai", id, 128_000,
                { coding: 0.65, reasoning: 0.65, creative: 0.60, math: 0.60, analysis: 0.65, conversation: 0.68, retrieval: 0.62, science: 0.62, business: 0.63, summary: 0.65 },
                {},
              ));
            }
          }

          // Auto-disable deprecated models (those in local registry but not in API response)
          let deprecatedCount = 0;
          for (const [key, m] of this.models) {
            if (m.provider === "zai" && !remoteIds.has(m.model)) {
              logger.warn(`Z.AI model ${key} not in API response - disabling as deprecated`);
              this.models.delete(key);
              deprecatedCount++;
            }
          }
          if (deprecatedCount > 0) {
            logger.info(`Auto-disabled ${deprecatedCount} deprecated Z.AI models`);
          }
        }
      }
    } catch (e) {
      logger.warn(`Failed to discover Z.AI models: ${e instanceof Error ? e.message : e}`);
    }

    // Discover Ollama models
    try {
      const resp = await fetch("http://localhost:11434/api/tags", {
        signal: AbortSignal.timeout(5000),
      });
      if (resp.ok) {
        const data = await resp.json() as any;
        for (const m of data.models ?? []) {
          if (m.name && !this.models.has(`ollama/${m.name}`)) {
            const caps = this.inferOllamaCapabilities(m.name, m.size);
            logger.info(`Discovered unseeded Ollama model: ollama/${m.name} (inferred from name/size)`);
            const discovered = makeModel("ollama", m.name, 8_000, caps, { local: true });
            discovered.source = "inferred";
            this.models.set(`ollama/${m.name}`, discovered);
          }
        }
      }
    } catch (e) {
      logger.warn(`Failed to discover Ollama models: ${e instanceof Error ? e.message : e}`);
    }
  }

  getCapability(provider: string, model: string): ModelCapability | undefined {
    return this.models.get(`${provider}/${model}`);
  }

  /** Remove a model from the registry (used by curator pruning). */
  removeModel(provider: string, model: string): boolean {
    const key = `${provider}/${model}`;
    const existed = this.models.delete(key);
    if (existed) {
      logger.info(`Removed model from registry: ${key}`);
    }
    return existed;
  }

  /** Add a dynamically discovered model to the registry (used by curator). */
  addDiscoveredModel(cap: ModelCapability): void {
    const key = `${cap.provider}/${cap.model}`;
    if (this.models.has(key)) {
      // Update existing entry
      this.models.set(key, cap);
      logger.debug(`Updated discovered model: ${key}`);
      return;
    }
    this.models.set(key, cap);
    logger.info(`Added discovered model to registry: ${key} (source: ${cap.source})`);
  }

  getAllModels(): ModelCapability[] {
    return Array.from(this.models.values());
  }

  getAvailableModels(availableProviders: string[]): ModelCapability[] {
    return this.getAllModels().filter((m) =>
      availableProviders.includes(m.provider),
    );
  }

  getCapabilityScore(provider: string, model: string, intent: string): number {
    const cap = this.getCapability(provider, model);
    if (!cap) return 0.5;
    const dim = ModelRegistry.INTENT_MAP[intent];
    if (!dim) return 0.5;
    return cap.capabilities[dim];
  }

  /** Read-path capability with UCB exploration bonus: learned + k/√n (n=0→1).
   *  While shadow mode is ON this returns the plain learned score — live
   *  routing stays untouched until the shadow gates pass and a human flips it. */
  getExplorationScore(provider: string, model: string, intent: string): number {
    const learned = this.getCapabilityScore(provider, model, intent);
    if (learningShadowMode()) return learned;
    const n = this.sampleCounts.get(`${provider}/${model}/${intent}`) ?? 0;
    return Math.min(1, learned + ucbBonus(n));
  }

  /** Manual pin accessor: pinned cells refuse learner updates (§4.2). */
  setCapabilityPin(provider: string, model: string, intent: string, pinned: boolean): void {
    const key = `${provider}/${model}/${intent}`;
    if (pinned) this.pinnedCells.add(key);
    else this.pinnedCells.delete(key);
    if (typeof this.db.setCapabilityPin === "function") {
      this.db.setCapabilityPin(provider, model, intent, pinned);
    }
    logger.info(`Capability cell ${key} ${pinned ? "pinned" : "unpinned"}.`);
  }

  isCapabilityPinned(provider: string, model: string, intent: string): boolean {
    return this.pinnedCells.has(`${provider}/${model}/${intent}`);
  }

  /** Check if a model supports a given modality (e.g. "vision", "audio"). */
  supportsModality(provider: string, model: string, modality: string): boolean {
    const cap = this.getCapability(provider, model);
    if (!cap) return false;
    return cap.modalities.includes(modality);
  }

  /** Get all models that support a given modality, from the available providers. */
  getModelsByModality(modality: string, availableProviders: string[]): ModelCapability[] {
    return this.getAvailableModels(availableProviders).filter(
      (m) => m.modalities.includes(modality),
    );
  }

  /**
   * Blend an observed judge score into the existing capability score via EMA.
   * Persists the result to SQLite so it survives restarts.
   *
   * EMA weight: alpha = 0.25 (25% new observation, 75% historical)
   * After ~10 samples the score converges to mostly observed data.
   */
  updateCapability(
    provider: string,
    model: string,
    intent: string,
    observedScore: number,
  ): void {
    const key = `${provider}/${model}`;
    const cap = this.models.get(key);
    if (!cap) return;
    const dim = ModelRegistry.INTENT_MAP[intent];
    if (!dim) return;

    const current = cap.capabilities[dim];
    const alpha = 0.25; // EMA smoothing - new observations weighted 25%
    const updated = clampCapability(current * (1 - alpha) + observedScore * alpha);
    const cellKey = `${provider}/${model}/${intent}`;

    if (typeof this.db.applyGuardedCapabilityUpdate === "function") {
      // Hardened path: atomic pin re-check + RoC window sums inside a
      // BEGIN IMMEDIATE transaction; reject-and-log over Δmax (§4.2).
      const existing = this.db.getCapabilityOverride(provider, model, intent);
      const baseline = existing?.score ?? current;
      const deltaAbs = Math.abs(updated - baseline);
      const result = this.db.applyGuardedCapabilityUpdate(
        provider, model, intent, updated, deltaAbs,
        new Date().toISOString(), rocMax24h(), rocMax7d(),
      );
      if (!result.applied) {
        logger.info(
          `Capability update rejected (${result.reason}): ${key} [${intent}] ` +
          `stays ${current.toFixed(3)} (judge=${observedScore.toFixed(2)})`,
        );
        return;
      }
      cap.capabilities[dim] = updated;
      cap.source = "blended";
      this.sampleCounts.set(cellKey, result.sampleCount);
      logger.info(
        `Capability evolved: ${key} [${intent}] ${current.toFixed(3)} → ${updated.toFixed(3)} ` +
        `(sample #${result.sampleCount}, judge=${observedScore.toFixed(2)})`,
      );
      return;
    }

    // Legacy path (mock DBs / pre-v4 databases): EMA + clamp only.
    cap.capabilities[dim] = updated;
    cap.source = "blended";
    this.sampleCounts.set(cellKey, (this.sampleCounts.get(cellKey) ?? 0) + 1);

    // Persist to DB
    const existing = this.db.getCapabilityOverride(provider, model, intent);
    const newCount = (existing?.sampleCount ?? 0) + 1;
    this.db.upsertCapabilityOverride(provider, model, intent, updated, newCount);

    logger.info(
      `Capability evolved: ${key} [${intent}] ${current.toFixed(3)} → ${updated.toFixed(3)} ` +
      `(sample #${newCount}, judge=${observedScore.toFixed(2)})`,
    );
  }

  /** Load judge-adjusted scores from the database on startup. */

  /**
   * Full Phase-1 judged-score pipeline (§4.1): attribution gate → quarantine
   * → guardrails → apply, with shadow-mode logging. Returns the outcome.
   * Replaces the old updateCapability+recordJudgeEvaluation call pairs.
   */
  applyJudgedScore(
    provider: string,
    model: string,
    intent: string,
    normalizedScore: number,
    opts: {
      rawScore: number;
      judgeNote: string;
      judgeModelId: string;
      confidence?: number | null;
      truncated?: boolean;
      malformed?: boolean;
    },
  ): { applied: boolean; arm: GateArm; reason: string; quarantined: boolean; wouldBe: number | null } {
    const shadow = learningShadowMode();
    const v = classifyAttribution({
      truncated: opts.truncated,
      malformed: opts.malformed,
      confidence: opts.confidence ?? null,
    });
    const norm = normalizeNote(opts.judgeNote);
    const hash = noteHash(norm);
    const canary = noteCanaryThreshold();

    const logEval = (noApply: boolean, reason: string, arm: string) => {
      if (typeof this.db.recordJudgeEvaluationEx === "function") {
        this.db.recordJudgeEvaluationEx({
          provider, model, intent,
          judgeScore: opts.rawScore,
          judgeNote: opts.judgeNote,
          judgeModel: opts.judgeModelId,
          noApply, gateArm: arm, gateReason: reason, noteHash: hash,
        });
      } else {
        this.db.recordJudgeEvaluation(provider, model, intent, opts.rawScore, opts.judgeNote, opts.judgeModelId);
      }
    };
    const logShadow = (reason: string | null, arm: string, preClamp: number | null, wouldBe: number | null, n: number | null) => {
      if (shadow && typeof this.db.recordShadowDecision === "function") {
        this.db.recordShadowDecision({
          provider, model, intent,
          confidence: opts.confidence ?? null,
          gateArm: arm,
          rejectionReason: reason,
          preClamp,
          wouldBeValue: wouldBe,
          sampleN: n,
        });
      }
    };

    // Arm A — provider-attributable fault: reliability trackers only, never capability.
    if (v.arm === "A") {
      logEval(true, v.reason, "A");
      logShadow(v.reason, "A", null, null, null);
      logger.info(`Judge gate Arm A (${v.reason}): ${provider}/${model} [${intent}] — reliability only, no capability write.`);
      return { applied: false, arm: "A", reason: v.reason, quarantined: false, wouldBe: null };
    }

    // Arm B — low/missing classifier confidence or malformed verdict: no_apply.
    if (v.arm === "B") {
      logEval(true, v.reason, "B");
      logShadow(v.reason, "B", null, null, null);
      logger.info(`Judge gate Arm B (${v.reason}): ${provider}/${model} [${intent}] — no_apply.`);
      return { applied: false, arm: "B", reason: v.reason, quarantined: false, wouldBe: null };
    }

    // Arm C — model-attributable: quarantine checks, then guardrails.
    const cap = this.models.get(`${provider}/${model}`);
    const dim = ModelRegistry.INTENT_MAP[intent];
    if (!cap || !dim) {
      logEval(true, "unknown_model_or_intent", "C");
      logShadow("unknown_model_or_intent", "C", null, null, null);
      return { applied: false, arm: "C", reason: "unknown_model_or_intent", quarantined: false, wouldBe: null };
    }

    const existing = typeof this.db.getCapabilityOverride === "function"
      ? this.db.getCapabilityOverride(provider, model, intent)
      : null;
    const baseline = existing?.score ?? cap.capabilities[dim];
    const n = existing?.sampleCount ?? this.sampleCounts.get(`${provider}/${model}/${intent}`) ?? 0;
    const alpha = 0.25;
    const preClamp = baseline * (1 - alpha) + normalizedScore * alpha;
    const wouldBe = clampCapability(preClamp);

    // Quarantined pair already? Never apply.
    if (typeof this.db.isPairQuarantined === "function" && this.db.isPairQuarantined(provider, model, intent)) {
      logEval(true, "pair_quarantined", "C");
      logShadow("pair_quarantined", "C", preClamp, wouldBe, n);
      return { applied: false, arm: "C", reason: "pair_quarantined", quarantined: true, wouldBe };
    }

    // Pre/post application note-similarity quarantine (§4.1).
    let priorCount = 0;
    if (typeof this.db.countJudgeNoteHash === "function") {
      priorCount = this.db.countJudgeNoteHash(provider, model, intent, hash);
    }
    if (priorCount >= canary) {
      // Post-application canary: ≥5 same note historically → quarantine + alert.
      if (typeof this.db.quarantinePair === "function") {
        this.db.quarantinePair(provider, model, intent, hash, norm, priorCount + 1);
      }
      logger.warn(
        `⚠ QUARANTINE: ${provider}/${model} [${intent}] — identical judge note ×${priorCount + 1} ` +
        `(task mix suspect, not model). Pair quarantined.`
      );
      logEval(true, "quarantine_canary", "C");
      logShadow("quarantine_canary", "C", preClamp, wouldBe, n);
      return { applied: false, arm: "C", reason: "quarantine_canary", quarantined: true, wouldBe };
    }
    if (priorCount >= noteHoldThreshold()) {
      // Pre-application hold: near-identical verdict seen recently — hold, inspect.
      logEval(true, "note_hold", "C");
      logShadow("note_hold", "C", preClamp, wouldBe, n);
      logger.info(`Judge gate hold (note seen ×${priorCount}): ${provider}/${model} [${intent}] — not applied.`);
      return { applied: false, arm: "C", reason: "note_hold", quarantined: false, wouldBe };
    }

    if (shadow) {
      // Shadow window: log the would-be value; never touch capability_overrides.
      logEval(false, "shadow_mode", "C");
      logShadow(null, "C", preClamp, wouldBe, n);
      logger.info(
        `Shadow: would evolve ${provider}/${model} [${intent}] ${baseline.toFixed(3)} → ${wouldBe.toFixed(3)} (n=${n})`
      );
      return { applied: false, arm: "C", reason: "shadow_mode", quarantined: false, wouldBe };
    }

    // Live apply through the guarded path (pin re-check + RoC inside txn).
    this.updateCapability(provider, model, intent, normalizedScore);
    logEval(false, "applied", "C");
    logShadow(null, "C", preClamp, wouldBe, n + 1);
    return { applied: true, arm: "C", reason: "applied", quarantined: false, wouldBe };
  }

  /** Infer capability scores for an unseeded Ollama model based on name and size.
   *  Uses family detection (deepseek, qwen, gemma, etc.) and size scaling. */
  private inferOllamaCapabilities(modelName: string, sizeBytes?: number): { coding: number; reasoning: number; creative: number; math: number; analysis: number; conversation: number; retrieval: number; science: number; business: number; summary: number } {
    const lower = modelName.toLowerCase();
    const familyScores: Record<string, Record<string, number>> = {
      deepseek:  { coding: 0.82, reasoning: 0.80, creative: 0.60, math: 0.78, analysis: 0.80, conversation: 0.64, retrieval: 0.70, science: 0.72, business: 0.62, summary: 0.68 },
      qwen:      { coding: 0.78, reasoning: 0.76, creative: 0.62, math: 0.72, analysis: 0.76, conversation: 0.66, retrieval: 0.70, science: 0.68, business: 0.64, summary: 0.66 },
      starcoder: { coding: 0.85, reasoning: 0.60, creative: 0.35, math: 0.55, analysis: 0.58, conversation: 0.30, retrieval: 0.40, science: 0.40, business: 0.30, summary: 0.50 },
      gemma:     { coding: 0.72, reasoning: 0.74, creative: 0.65, math: 0.68, analysis: 0.72, conversation: 0.72, retrieval: 0.68, science: 0.68, business: 0.64, summary: 0.70 },
      llama:     { coding: 0.70, reasoning: 0.72, creative: 0.64, math: 0.66, analysis: 0.70, conversation: 0.68, retrieval: 0.66, science: 0.66, business: 0.62, summary: 0.66 },
      phi:       { coding: 0.68, reasoning: 0.70, creative: 0.58, math: 0.66, analysis: 0.66, conversation: 0.62, retrieval: 0.62, science: 0.64, business: 0.60, summary: 0.64 },
      mistral:   { coding: 0.66, reasoning: 0.68, creative: 0.60, math: 0.62, analysis: 0.68, conversation: 0.66, retrieval: 0.62, science: 0.60, business: 0.58, summary: 0.64 },
      nomic:     { coding: 0.30, reasoning: 0.30, creative: 0.35, math: 0.25, analysis: 0.32, conversation: 0.40, retrieval: 0.80, science: 0.30, business: 0.28, summary: 0.45 },
      bge:       { coding: 0.20, reasoning: 0.20, creative: 0.20, math: 0.20, analysis: 0.25, conversation: 0.20, retrieval: 0.85, science: 0.20, business: 0.20, summary: 0.30 },
      minicpm:   { coding: 0.60, reasoning: 0.62, creative: 0.55, math: 0.60, analysis: 0.62, conversation: 0.60, retrieval: 0.58, science: 0.58, business: 0.54, summary: 0.60 },
    };
    const defaultCaps = { coding: 0.40, reasoning: 0.40, creative: 0.45, math: 0.35, analysis: 0.40, conversation: 0.50, retrieval: 0.45, science: 0.35, business: 0.38, summary: 0.48 };

    let caps = { ...defaultCaps };
    for (const [family, scores] of Object.entries(familyScores)) {
      if (lower.includes(family)) { caps = { ...scores } as typeof caps; break; }
    }

    // Size-based scaling: rough bytes→params (1.5 bytes/param for quantized)
    if (sizeBytes) {
      const params = sizeBytes / 1.5e9;
      let scale = 1;
      if (params >= 70) scale = 1.18;
      else if (params >= 30) scale = 1.12;
      else if (params < 3) scale = 0.85;
      for (const k of Object.keys(caps) as Array<keyof typeof caps>) {
        caps[k] = Math.min(0.95, Math.max(0.15, caps[k] * scale));
      }
    }

    return caps;
  }

  private applySavedOverrides(): void {
    const overrides = typeof this.db.loadCapabilityOverrideRows === "function"
      ? this.db.loadCapabilityOverrideRows()
      : this.db.loadCapabilityOverrides().map(o => ({ ...o, lastJudged: new Date().toISOString(), pinned: false }));
    if (overrides.length === 0) return;

    let applied = 0;
    let decayed = 0;
    const idleDays = decayIdleDays();
    const now = Date.now();
    for (const o of overrides) {
      const cap = this.models.get(`${o.provider}/${o.model}`);
      if (!cap) continue;
      const dim = ModelRegistry.INTENT_MAP[o.intent];
      if (!dim) continue;

      const cellKey = `${o.provider}/${o.model}/${o.intent}`;
      this.sampleCounts.set(cellKey, o.sampleCount);
      if (o.pinned) this.pinnedCells.add(cellKey);

      const seedScore = cap.capabilities[dim];
      let score = o.score;

      // Decay-to-prior on idle (§4.2): pinned cells exempt.
      if (!o.pinned) {
        const lastJudgedMs = o.lastJudged ? new Date(o.lastJudged).getTime() : now;
        const idle = (now - lastJudgedMs) / 86_400_000;
        if (idle > idleDays) {
          const decayedScore = clampCapability(decayTowardSeed(score, seedScore, idle - idleDays));
          if (Math.abs(decayedScore - score) > 0.0005) {
            logger.info(
              `Decay-to-prior: ${o.provider}/${o.model} [${o.intent}] ` +
              `${score.toFixed(3)} → ${decayedScore.toFixed(3)} (idle ${idle.toFixed(1)}d, seed=${seedScore.toFixed(3)})`
            );
            score = decayedScore;
            decayed++;
            if (typeof this.db.updateCapabilityScoreOnly === "function") {
              this.db.updateCapabilityScoreOnly(o.provider, o.model, o.intent, decayedScore);
            }
          }
        }
      }

      cap.capabilities[dim] = score;
      cap.source = "blended";
      applied++;

      logger.debug(
        `Override applied: ${o.provider}/${o.model} [${o.intent}] ` +
        `seed=${seedScore.toFixed(3)} → learned=${score.toFixed(3)} (${o.sampleCount} samples)`,
      );
    }

    if (applied > 0) {
      logger.info(`Applied ${applied} capability overrides from judge feedback (${decayed} decayed toward seed).`);
    }
  }
}
