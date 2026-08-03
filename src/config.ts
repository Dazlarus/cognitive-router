// src/config.ts — Configuration loading + defaults

import { setLevel } from "./logger.js";

export interface Weights {
  capability: number;
  reliability: number;
  cost: number;
  latency: number;
}

export interface ProviderBudget {
  budgetType: "free" | "subscription" | "pay_per_token" | "credits";
  priority: "high" | "medium" | "low";
  monthlyBudgetUsd?: number;
  /** For subscription providers: track quota consumption. Values: 0-1 (1=exhausted) */
  quotaConsumed?: number;
  /** For subscription providers: display quota consumption % */
  quotaPercent?: number;
}

export interface EmbeddingConfig {
  provider: "ollama" | "gemini" | "openrouter";
  model: string;
  fallback: "ollama" | "gemini" | "openrouter" | "none";
  fallbackModel?: string;
  timeoutMs: number;
}

export interface CognitiveRouterConfig {
  enabled: boolean;
  weights: Weights;
  probeRate: number;
  tiebreakerThreshold: number;
  benchmarkSyncIntervalHours: number;
  logLevel: string;
  dbPath: string;
  /** Max VRAM in GB for local (Ollama) models — models exceeding this are filtered out */
  localVramLimitGb: number;
  providers: Record<string, ProviderBudget>;
  overrides: RoutingOverride[];
  /** Provider priority order — lower index = higher priority. Models will be routed to
   *  higher priority providers first. Only models from providers in this list are eligible. */
  providerPriority: string[];

  /** Port for standalone proxy mode (default 3456) */
  proxyPort?: number;

  reliabilityAbortPenalty: number;
}

export interface RoutingOverride {
  intent: string;
  provider: string;
  model: string;
  reason?: string;
}

const DEFAULT_WEIGHTS: Weights = {
  capability: 0.50,
  reliability: 0.25,
  cost: 0.15,
  latency: 0.10,
};

const DEFAULT_PROVIDERS: Record<string, ProviderBudget> = {
  openrouter: { budgetType: "free", priority: "high" },
  zai: { budgetType: "subscription", priority: "high", quotaConsumed: 0, quotaPercent: 0 },
  gemini: { budgetType: "credits", priority: "medium" },
  requesty: { budgetType: "pay_per_token", priority: "medium", monthlyBudgetUsd: 20 },
  ollama: { budgetType: "free", priority: "low" },
};

const DEFAULT_EMBEDDING: EmbeddingConfig = {
  provider: "ollama",
  model: "qwen3-embedding:latest", // Updated 2026-06-23: 2x faster (214ms vs 389ms) per embedding benchmark
  fallback: "gemini",
  fallbackModel: "gemini-embedding-001",
  timeoutMs: 2000,
};

const DEFAULT_PROVIDER_PRIORITY: string[] = ["zai", "openrouter", "gemini", "ollama"];

export function loadConfig(pluginConfig: Record<string, any>): CognitiveRouterConfig {
  const config: CognitiveRouterConfig = {
    enabled: pluginConfig.enabled ?? false,
    weights: { ...DEFAULT_WEIGHTS, ...(pluginConfig.weights ?? {}) },
    probeRate: pluginConfig.probeRate ?? 0.05,
    tiebreakerThreshold: pluginConfig.tiebreakerThreshold ?? 0.70,
    benchmarkSyncIntervalHours: pluginConfig.benchmarkSyncIntervalHours ?? 168,
    logLevel: pluginConfig.logLevel ?? "info",
    dbPath: pluginConfig.dbPath ?? "data/cognitive-router.db",
    localVramLimitGb: pluginConfig.localVramLimitGb ?? 11,
    providers: { ...DEFAULT_PROVIDERS, ...(pluginConfig.providers ?? {}) },
    overrides: pluginConfig.overrides ?? [],
    proxyPort: pluginConfig.proxyPort ?? 3456,
    providerPriority: pluginConfig.providerPriority ?? DEFAULT_PROVIDER_PRIORITY,
    reliabilityAbortPenalty: pluginConfig.reliabilityAbortPenalty ?? 0.2,
  };

  setLevel(config.logLevel as any);
  return config;
}
