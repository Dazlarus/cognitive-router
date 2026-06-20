// src/model_registry.ts — Model discovery + capability profiles

import { logger } from "./logger.js";
import type { DBService } from "./db_service.js";
import type { CognitiveRouterConfig } from "./config.js";

export interface ModelCapability {
  provider: string;
  model: string;
  contextWindow: number;
  modalities: string[]; // text, vision, audio, etc.
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
  source: "benchmark" | "observed" | "blended"; // how current the data is
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
  opts: { input?: number; output?: number; local?: boolean; vision?: boolean; usageMultiplier?: number; vram?: number; planEligible?: boolean } = {},
): ModelCapability {
  return {
    provider,
    model: id,
    contextWindow: ctx,
    modalities: opts.vision ? ["text", "vision"] : ["text"],
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

const SEED_MODELS: ModelCapability[] = [
  // ═══════════════════════════════════════════════════════════════
  // Z.AI Flagship Lineup (13 models from OpenClaw config)
  // ═══════════════════════════════════════════════════════════════

  // GLM-5.x — top tier reasoning
  // ✅ Coding Plan eligible (Opus-tier, 3× peak / 2× off-peak quota, 1× promo through Sep 2026)
  makeModel("zai", "glm-5.2", 202_800,
    { coding: 0.88, reasoning: 0.92, creative: 0.84, math: 0.90, analysis: 0.91, conversation: 0.87, retrieval: 0.85, science: 0.93, business: 0.88, summary: 0.87 },
    { input: 0, output: 0, usageMultiplier: 1, planEligible: true },
  ),
  makeModel("zai", "glm-5.1", 202_800,
    { coding: 0.85, reasoning: 0.88, creative: 0.80, math: 0.84, analysis: 0.86, conversation: 0.87, retrieval: 0.83, science: 0.86, business: 0.84, summary: 0.86 },
    { input: 0, output: 0, planEligible: false }, // Not in coding plan — requires credits
  ),
  makeModel("zai", "glm-5", 202_800,
    { coding: 0.82, reasoning: 0.85, creative: 0.78, math: 0.80, analysis: 0.82, conversation: 0.85, retrieval: 0.80, science: 0.82, business: 0.82, summary: 0.83 },
    { input: 0, output: 0, planEligible: false },
  ),
  // ✅ Coding Plan eligible (Opus-tier)
  makeModel("zai", "glm-5-turbo", 202_800,
    { coding: 0.80, reasoning: 0.82, creative: 0.76, math: 0.78, analysis: 0.80, conversation: 0.84, retrieval: 0.78, science: 0.78, business: 0.80, summary: 0.82 },
    { input: 0, output: 0, usageMultiplier: 1, planEligible: true },
  ),
  makeModel("zai", "glm-5v-turbo", 202_800,
    { coding: 0.78, reasoning: 0.80, creative: 0.75, math: 0.76, analysis: 0.78, conversation: 0.82, retrieval: 0.76, science: 0.76, business: 0.78, summary: 0.80 },
    { input: 0, output: 0, vision: true, planEligible: false },
  ),

  // ✅ Coding Plan eligible (Sonnet-tier, standard 1× quota)
  makeModel("zai", "glm-4.7", 204_800,
    { coding: 0.78, reasoning: 0.80, creative: 0.74, math: 0.76, analysis: 0.78, conversation: 0.82, retrieval: 0.78, science: 0.77, business: 0.78, summary: 0.80 },
    { input: 0, output: 0, planEligible: true },
  ),
  // Flash models — not in coding plan, require credits
  makeModel("zai", "glm-4.7-flash", 200_000,
    { coding: 0.65, reasoning: 0.66, creative: 0.68, math: 0.60, analysis: 0.64, conversation: 0.90, retrieval: 0.88, science: 0.62, business: 0.72, summary: 0.87 },
    { input: 0, output: 0, planEligible: false },
  ),
  makeModel("zai", "glm-4.7-flashx", 200_000,
    { coding: 0.58, reasoning: 0.58, creative: 0.65, math: 0.52, analysis: 0.56, conversation: 0.88, retrieval: 0.85, science: 0.54, business: 0.68, summary: 0.84 },
    { input: 0, output: 0, planEligible: false },
  ),

  // GLM-4.6 — not in coding plan
  makeModel("zai", "glm-4.6", 204_800,
    { coding: 0.72, reasoning: 0.74, creative: 0.70, math: 0.70, analysis: 0.72, conversation: 0.78, retrieval: 0.74, science: 0.71, business: 0.72, summary: 0.76 },
    { input: 0, output: 0, planEligible: false },
  ),
  makeModel("zai", "glm-4.6v", 128_000,
    { coding: 0.68, reasoning: 0.70, creative: 0.68, math: 0.66, analysis: 0.68, conversation: 0.75, retrieval: 0.70, science: 0.67, business: 0.68, summary: 0.72 },
    { input: 0, output: 0, vision: true, planEligible: false },
  ),

  // GLM-4.5 — not in coding plan
  makeModel("zai", "glm-4.5", 131_072,
    { coding: 0.68, reasoning: 0.70, creative: 0.68, math: 0.66, analysis: 0.68, conversation: 0.76, retrieval: 0.70, science: 0.67, business: 0.68, summary: 0.72 },
    { input: 0, output: 0, planEligible: false },
  ),
  makeModel("zai", "glm-4.5-air", 131_072,
    { coding: 0.55, reasoning: 0.56, creative: 0.60, math: 0.52, analysis: 0.55, conversation: 0.72, retrieval: 0.60, science: 0.52, business: 0.54, summary: 0.66 },
    { input: 0, output: 0, planEligible: false },
  ),
  makeModel("zai", "glm-4.5-flash", 131_072,
    { coding: 0.48, reasoning: 0.48, creative: 0.62, math: 0.42, analysis: 0.48, conversation: 0.85, retrieval: 0.80, science: 0.45, business: 0.60, summary: 0.80 },
    { input: 0, output: 0, planEligible: false },
  ),
  makeModel("zai", "glm-4.5v", 64_000,
    { coding: 0.58, reasoning: 0.60, creative: 0.58, math: 0.55, analysis: 0.58, conversation: 0.70, retrieval: 0.62, science: 0.56, business: 0.58, summary: 0.66 },
    { input: 0, output: 0, vision: true, planEligible: false },
  ),

  // ═══════════════════════════════════════════════════════════════
  // OpenRouter free agent-generation candidates only. Avoid seeding
  // openrouter/free here because it is a random free-model router, not a
  // deterministic model row suitable for agent routing.
  // ═══════════════════════════════════════════════════════════════

  makeModel("openrouter", "qwen/qwen3-coder:free", 1_000_000,
    { coding: 0.84, reasoning: 0.76, creative: 0.60, math: 0.72, analysis: 0.78, conversation: 0.62, retrieval: 0.68, science: 0.68, business: 0.64, summary: 0.66 },
    { input: 0, output: 0 },
  ),
  makeModel("openrouter", "poolside/laguna-m.1:free", 262_000,
    { coding: 0.82, reasoning: 0.74, creative: 0.56, math: 0.68, analysis: 0.76, conversation: 0.58, retrieval: 0.62, science: 0.62, business: 0.60, summary: 0.62 },
    { input: 0, output: 0 },
  ),
  makeModel("openrouter", "openrouter/owl-alpha", 1_000_000,
    { coding: 0.76, reasoning: 0.78, creative: 0.66, math: 0.70, analysis: 0.80, conversation: 0.72, retrieval: 0.76, science: 0.72, business: 0.72, summary: 0.74 },
    { input: 0, output: 0 },
  ),
  makeModel("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free", 1_000_000,
    { coding: 0.72, reasoning: 0.82, creative: 0.62, math: 0.76, analysis: 0.82, conversation: 0.68, retrieval: 0.78, science: 0.78, business: 0.76, summary: 0.76 },
    { input: 0, output: 0 },
  ),
  makeModel("openrouter", "nvidia/nemotron-3-super-120b-a12b:free", 1_000_000,
    { coding: 0.70, reasoning: 0.80, creative: 0.60, math: 0.76, analysis: 0.80, conversation: 0.66, retrieval: 0.76, science: 0.76, business: 0.74, summary: 0.74 },
    { input: 0, output: 0 },
  ),
  makeModel("openrouter", "qwen/qwen3.6-plus:free", 1_000_000,
    { coding: 0.80, reasoning: 0.78, creative: 0.70, math: 0.72, analysis: 0.78, conversation: 0.72, retrieval: 0.74, science: 0.72, business: 0.72, summary: 0.74 },
    { input: 0, output: 0 },
  ),
  makeModel("openrouter", "cohere/north-mini-code:free", 256_000,
    { coding: 0.74, reasoning: 0.62, creative: 0.50, math: 0.58, analysis: 0.64, conversation: 0.56, retrieval: 0.58, science: 0.56, business: 0.56, summary: 0.58 },
    { input: 0, output: 0 },
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
    { local: true, vision: true, vram: 8.1 },
  ),
  makeModel("ollama", "gemma3:4b", 8_000,
    { coding: 0.42, reasoning: 0.44, creative: 0.50, math: 0.38, analysis: 0.42, conversation: 0.60, retrieval: 0.55, science: 0.38, business: 0.40, summary: 0.55 },
    { local: true, vision: true, vram: 3.3 },
  ),
  makeModel("ollama", "gemma3:1b", 8_000,
    { coding: 0.30, reasoning: 0.32, creative: 0.40, math: 0.25, analysis: 0.30, conversation: 0.50, retrieval: 0.42, science: 0.25, business: 0.28, summary: 0.45 },
    { local: true, vision: true, vram: 0.8 },
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
    { local: true, vision: true, vram: 7.8 },
  ),
  makeModel("ollama", "moondream:latest", 8_000,
    { coding: 0.15, reasoning: 0.20, creative: 0.35, math: 0.12, analysis: 0.25, conversation: 0.40, retrieval: 0.30, science: 0.15, business: 0.18, summary: 0.35 },
    { local: true, vision: true, vram: 1.7 },
  ),

  // New additions — high-quality models that fit 11GB limit
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
  private static readonly INTENT_MAP: Record<string, keyof Caps> = {
    coding: "coding",
    research: "retrieval",
    creative: "creative",
    conversation: "conversation",
    summary: "summary",
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

    // Apply any saved capability overrides from the judge feedback loop
    this.applySavedOverrides();

    logger.info(`Model registry loaded — ${this.models.size} models tracked.`);
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
              logger.info(`Discovered unseeded Z.AI model: zai/${id} — using neutral defaults`);
              this.models.set(`zai/${id}`, makeModel("zai", id, 128_000,
                { coding: 0.65, reasoning: 0.65, creative: 0.60, math: 0.60, analysis: 0.65, conversation: 0.68, retrieval: 0.62, science: 0.62, business: 0.63, summary: 0.65 },
                {},
              ));
            }
          }
          for (const [key, m] of this.models) {
            if (m.provider === "zai" && !remoteIds.has(m.model)) {
              logger.warn(`Z.AI model ${key} not in API response — may be deprecated`);
            }
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
            logger.info(`Discovered unseeded Ollama model: ollama/${m.name}`);
            this.models.set(`ollama/${m.name}`, makeModel("ollama", m.name, 8_000,
              { coding: 0.40, reasoning: 0.40, creative: 0.45, math: 0.35, analysis: 0.40, conversation: 0.50, retrieval: 0.45, science: 0.35, business: 0.38, summary: 0.48 },
              { local: true },
            ));
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
    const alpha = 0.25; // EMA smoothing — new observations weighted 25%
    const updated = current * (1 - alpha) + observedScore * alpha;
    cap.capabilities[dim] = updated;
    cap.source = "blended";

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
  private applySavedOverrides(): void {
    const overrides = this.db.loadCapabilityOverrides();
    if (overrides.length === 0) return;

    let applied = 0;
    for (const o of overrides) {
      const cap = this.models.get(`${o.provider}/${o.model}`);
      if (!cap) continue;
      const dim = ModelRegistry.INTENT_MAP[o.intent];
      if (!dim) continue;

      const seedScore = cap.capabilities[dim];
      cap.capabilities[dim] = o.score;
      cap.source = "blended";
      applied++;

      logger.debug(
        `Override applied: ${o.provider}/${o.model} [${o.intent}] ` +
        `seed=${seedScore.toFixed(3)} → learned=${o.score.toFixed(3)} (${o.sampleCount} samples)`,
      );
    }

    if (applied > 0) {
      logger.info(`Applied ${applied} capability overrides from judge feedback.`);
    }
  }
}
