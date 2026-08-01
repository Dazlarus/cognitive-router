// src/ollama_warmth.ts — Ollama warm model detection
// Polls Ollama's /api/ps endpoint to check which models are currently
// loaded in GPU memory. Caches results with a short TTL to avoid
// hammering Ollama on every routing decision.

import { logger } from "./logger.js";

/** Default Ollama base URL. */
const DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434";

/** Cache TTL in milliseconds. */
const CACHE_TTL_MS = 5_000;

/** Cold-start latency estimates (ms) keyed by approximate parameter count.
 *  These are rough heuristics for model load time from disk → GPU. */
const COLD_START_BASE_MS = 5_000; // 7B-class models: ~5s
const COLD_START_PER_VRAM_GB = 800; // ~800ms per additional GB of VRAM beyond 7B baseline
const COLD_START_BASELINE_VRAM_GB = 5; // 7B ~ 5GB VRAM

/** Response shape from Ollama /api/ps. */
interface OllamaPsResponse {
  models?: Array<{
    name?: string;
    model?: string;
    size?: number;
    digest?: string;
    details?: {
      parameter_size?: string;
      quantization_level?: string;
      family?: string;
    };
  }>;
}

/** Result of a warmth check for a single model. */
export interface WarmthInfo {
  /** Whether the model is currently loaded in GPU memory. */
  isWarm: boolean;
  /** Estimated cold-start latency in ms (0 if warm). */
  estimatedColdStartMs: number;
  /** Whether Ollama was reachable for this check. */
  ollamaReachable: boolean;
  /** All models currently loaded (empty if unreachable). */
  loadedModels: string[];
}

export class OllamaWarmthChecker {
  private baseUrl: string;
  private cacheTtlMs: number;

  /** Cached /api/ps response. */
  private cachedModels: Set<string> = new Set();
  private cacheTimestamp = 0;
  private cacheValid = false;
  private ollamaReachable = true;

  constructor(opts: {
    baseUrl?: string;
    cacheTtlMs?: number;
  } = {}) {
    this.baseUrl = (opts.baseUrl ?? process.env.OLLAMA_BASE_URL ?? DEFAULT_OLLAMA_BASE_URL).replace(/\/$/, "");
    this.cacheTtlMs = opts.cacheTtlMs ?? CACHE_TTL_MS;
  }

  /** Fetch the set of loaded model names from Ollama, using cache if fresh.
   *  Returns a Set of model name strings (lowercased for matching). */
  private async getLoadedModels(): Promise<{ models: Set<string>; reachable: boolean }> {
    const now = Date.now();

    // Return cached result if still valid
    if (this.cacheValid && (now - this.cacheTimestamp) < this.cacheTtlMs) {
      return { models: this.cachedModels, reachable: this.ollamaReachable };
    }

    // Cache expired — fetch fresh data
    try {
      const response = await fetch(`${this.baseUrl}/api/ps`, {
        method: "GET",
        headers: { "Accept": "application/json" },
        signal: AbortSignal.timeout(2_000),
      });

      if (!response.ok) {
        logger.warn(`Ollama /api/ps returned HTTP ${response.status} — treating all models as cold.`);
        this.updateCache(new Set(), false);
        return { models: this.cachedModels, reachable: false };
      }

      const data = (await response.json()) as OllamaPsResponse;
      const models = new Set<string>();

      if (data && Array.isArray(data.models)) {
        for (const entry of data.models) {
          // /api/ps returns both "name" and "model" fields — prefer "name", fall back to "model"
          const name = entry.name ?? entry.model;
          if (name) {
            models.add(name.toLowerCase());
          }
        }
      }

      this.updateCache(models, true);
      return { models, reachable: true };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // Don't spam logs — only warn if we transitioned from reachable to unreachable
      if (this.ollamaReachable) {
        logger.warn(`Ollama /api/ps unreachable (${msg}) — treating all models as cold.`);
      }
      this.updateCache(new Set(), false);
      return { models: this.cachedModels, reachable: false };
    }
  }

  private updateCache(models: Set<string>, reachable: boolean): void {
    this.cachedModels = models;
    this.cacheTimestamp = Date.now();
    this.cacheValid = true;
    this.ollamaReachable = reachable;
  }

  /** Force-invalidate the cache. Useful for testing. */
  invalidateCache(): void {
    this.cacheValid = false;
    this.cacheTimestamp = 0;
  }

  /** Check if a specific Ollama model is warm (loaded in GPU memory).
   *  Model name matching is case-insensitive. */
  async checkWarmth(modelName: string): Promise<WarmthInfo> {
    const { models, reachable } = await this.getLoadedModels();
    const isWarm = reachable && models.has(modelName.toLowerCase());

    return {
      isWarm,
      estimatedColdStartMs: isWarm ? 0 : this.estimateColdStartMs(modelName),
      ollamaReachable: reachable,
      loadedModels: [...models],
    };
  }

  /** Batch-check warmth for multiple models in a single /api/ps call. */
  async checkWarmthBatch(modelNames: string[]): Promise<Map<string, WarmthInfo>> {
    const { models, reachable } = await this.getLoadedModels();
    const result = new Map<string, WarmthInfo>();

    for (const name of modelNames) {
      const isWarm = reachable && models.has(name.toLowerCase());
      result.set(name, {
        isWarm,
        estimatedColdStartMs: isWarm ? 0 : this.estimateColdStartMs(name),
        ollamaReachable: reachable,
        loadedModels: [...models],
      });
    }

    return result;
  }

  /** Estimate cold-start latency based on model name heuristics.
   *  This is intentionally rough — real cold-start depends on disk speed,
   *  GPU bandwidth, and model format (GGUF quantization). */
  private estimateColdStartMs(modelName: string): number {
    // Try to extract parameter count from model name (e.g., "qwen2.5:7b", "llama3:70b")
    const paramMatch = modelName.match(/(\d+(?:\.\d+)?)\s*b/i);
    if (paramMatch) {
      const paramsInBillions = parseFloat(paramMatch[1]);
      // Rough VRAM estimate: ~1.5GB per billion params at Q4 quantization
      const estimatedVramGb = paramsInBillions * 1.5;
      if (estimatedVramGb <= COLD_START_BASELINE_VRAM_GB) {
        return COLD_START_BASE_MS;
      }
      const extra = estimatedVramGb - COLD_START_BASELINE_VRAM_GB;
      return COLD_START_BASE_MS + Math.round(extra * COLD_START_PER_VRAM_GB);
    }

    // If we can't parse parameter count, use a conservative default
    return COLD_START_BASE_MS;
  }
}
