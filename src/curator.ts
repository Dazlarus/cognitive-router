// src/curator.ts — Active model curator
// Periodically scans Ollama library + OpenRouter API for new models,
// pulls eligible Ollama models, auto-prunes dead OpenRouter models,
// and evaluates candidates against current capability gaps.

import { logger } from "./logger.js";
import type { DBService } from "./db_service.js";
import type { ModelRegistry, ModelCapability } from "./model_registry.js";
import type { CognitiveRouterConfig } from "./config.js";
import { exec } from "node:child_process";
import { promisify } from "node:util";

const execAsync = promisify(exec);

// ─── Types ───

export interface CuratorResult {
  timestamp: string;
  ollamaScanned: number;
  ollamaPulled: string[];
  ollamaSkipped: string[];
  openrouterScanned: number;
  openrouterAdded: string[];
  openrouterPruned: string[];
  errors: string[];
}

export interface DiscoveredOllamaModel {
  name: string;
  description: string;
  pulls: number;
  tags: string[];
  sizeBytes?: number;
}

export interface DiscoveredOpenRouterModel {
  id: string;
  name: string;
  contextLength: number;
  pricing: { prompt: string; completion: string };
  isFree: boolean;
  architecture?: string;
  topProvider?: {
    contextLength?: number;
    maxCompletionTokens?: number;
  };
}

export interface TrustSignals {
  downloadCount: number;
  maintainer: string;
  hasModeration: boolean;
  isOfficial: boolean;
  trustScore: number; // 0-1
}

// ─── Constants ───

const OPENROUTER_API = "https://openrouter.ai/api/v1/models";
const OLLAMA_LIBRARY = "https://ollama.com/library";
const OLLAMA_API = "http://localhost:11434";

/** How often the curator runs automatically (6 hours). */
const CURATOR_INTERVAL_MS = 6 * 60 * 60 * 1000;

/** Number of consecutive 404 checks before pruning an OpenRouter model. */
const PRUNE_THRESHOLD = 3;

/** VRAM safety margin — only pull models that fit in 80% of the VRAM limit. */
const VRAM_SAFETY_FACTOR = 0.8;

/** Maximum models to pull per curator run (to avoid flooding disk). */
const MAX_PULLS_PER_RUN = 2;

/** Known Ollama model families and their capability profiles. */
const FAMILY_PROFILES: Record<string, { caps: Partial<ModelCapability["capabilities"]>; vramGb: number }> = {
  deepseek: { caps: { coding: 0.82, reasoning: 0.80, math: 0.78 }, vramGb: 0 },
  qwen: { caps: { coding: 0.78, reasoning: 0.76, math: 0.72 }, vramGb: 0 },
  llama: { caps: { coding: 0.70, reasoning: 0.72, math: 0.66 }, vramGb: 0 },
  gemma: { caps: { coding: 0.72, reasoning: 0.74, math: 0.68 }, vramGb: 0 },
  mistral: { caps: { coding: 0.66, reasoning: 0.68, math: 0.62 }, vramGb: 0 },
  phi: { caps: { coding: 0.68, reasoning: 0.70, math: 0.66 }, vramGb: 0 },
  starcoder: { caps: { coding: 0.85, reasoning: 0.60, math: 0.55 }, vramGb: 0 },
  codeqwen: { caps: { coding: 0.80, reasoning: 0.65, math: 0.60 }, vramGb: 0 },
  nomic: { caps: { coding: 0.30, reasoning: 0.30, retrieval: 0.80 }, vramGb: 0 },
};

/** Intent dimensions we track for gap analysis. */
const CAPABILITY_DIMENSIONS = [
  "coding", "reasoning", "creative", "math", "analysis",
  "conversation", "retrieval", "science", "business", "summary",
] as const;

// ─── Curator ───

export class ModelCurator {
  private timer: ReturnType<typeof setInterval> | null = null;
  private running = false;

  constructor(
    private db: DBService,
    private registry: ModelRegistry,
    private config: CognitiveRouterConfig,
  ) {}

  /** Start the periodic curator. */
  startPeriodic(): void {
    if (this.timer) return;
    logger.info(`Curator started — runs every ${CURATOR_INTERVAL_MS / 3600_000}h.`);
    this.timer = setInterval(() => {
      this.run().catch((err) => {
        logger.error(`Curator run failed: ${err instanceof Error ? err.message : err}`);
      });
    }, CURATOR_INTERVAL_MS);
    // Don't keep the process alive just for the curator
    if (this.timer.unref) this.timer.unref();
  }

  /** Stop the periodic curator. */
  stopPeriodic(): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
      logger.info("Curator stopped.");
    }
  }

  /** Execute one full curation cycle. */
  async run(): Promise<CuratorResult> {
    if (this.running) {
      logger.warn("Curator already running — skipping.");
      return {
        timestamp: new Date().toISOString(),
        ollamaScanned: 0, ollamaPulled: [], ollamaSkipped: [],
        openrouterScanned: 0, openrouterAdded: [], openrouterPruned: [],
        errors: ["Curator already running"],
      };
    }

    this.running = true;
    const result: CuratorResult = {
      timestamp: new Date().toISOString(),
      ollamaScanned: 0, ollamaPulled: [], ollamaSkipped: [],
      openrouterScanned: 0, openrouterAdded: [], openrouterPruned: [],
      errors: [],
    };

    logger.info("Curator cycle started.");

    try {
      await this.scanOllamaLibrary(result);
    } catch (err) {
      const msg = `Ollama scan failed: ${err instanceof Error ? err.message : err}`;
      logger.warn(msg);
      result.errors.push(msg);
    }

    try {
      await this.scanOpenRouter(result);
    } catch (err) {
      const msg = `OpenRouter scan failed: ${err instanceof Error ? err.message : err}`;
      logger.warn(msg);
      result.errors.push(msg);
    }

    logger.info(
      `Curator cycle complete: OR scanned=${result.openrouterScanned}, ` +
      `added=${result.openrouterAdded.length}, pruned=${result.openrouterPruned.length}, ` +
      `Ollama scanned=${result.ollamaScanned}, pulled=${result.ollamaPulled.length}.`,
    );

    this.db.recordCuratorRun(result);
    this.running = false;
    return result;
  }

  // ─── Ollama Library Scanner ───

  private async scanOllamaLibrary(result: CuratorResult): Promise<void> {
    // 1. Fetch installed models from local Ollama
    const installed = await this.getInstalledOllamaModels();
    const installedSet = new Set(installed.map((m) => m.name));

    // 2. Fetch library page for available models
    const libraryModels = await this.fetchOllamaLibrary();
    result.ollamaScanned = libraryModels.length;

    if (libraryModels.length === 0) {
      logger.debug("Ollama library returned no models — skipping.");
      return;
    }

    // 3. Evaluate capability gaps
    const gaps = this.identifyCapabilityGaps();

    // 4. Score and rank candidates
    const candidates = libraryModels
      .filter((m) => !installedSet.has(m.name) && !installedSet.has(m.name + ":latest"))
      .map((m) => ({
        model: m,
        gapScore: this.scoreOllamaCandidate(m, gaps),
        trust: this.evaluateTrustSignals(m),
      }))
      .filter((c) => c.gapScore > 0.3 && c.trust.trustScore > 0.4)
      .sort((a, b) => b.gapScore * b.trust.trustScore - a.gapScore * a.trust.trustScore);

    logger.info(`Ollama: ${candidates.length} viable candidates (of ${libraryModels.length} scanned).`);

    // 5. Check VRAM and pull top candidates
    const vramLimit = this.config.localVramLimitGb ?? 11;
    const effectiveVram = vramLimit * VRAM_SAFETY_FACTOR;
    let pulled = 0;

    for (const candidate of candidates) {
      if (pulled >= MAX_PULLS_PER_RUN) break;

      const modelName = candidate.model.name;
      const estimatedVram = this.estimateVramForModel(modelName, candidate.model.sizeBytes);

      if (estimatedVram > effectiveVram) {
        logger.info(
          `Skipping ollama/${modelName} — estimated ${estimatedVram.toFixed(1)}GB VRAM ` +
          `exceeds limit ${effectiveVram.toFixed(1)}GB.`,
        );
        result.ollamaSkipped.push(`${modelName} (VRAM: ${estimatedVram.toFixed(1)}GB)`);
        continue;
      }

      // Check if we already know about this model in the registry
      const existing = this.registry.getCapability("ollama", modelName + ":latest");
      if (existing) {
        continue;
      }

      // Check if we've recently attempted this model (avoid retry loops)
      const lastAttempt = this.db.getCuratorModelAttempt("ollama", modelName);
      if (lastAttempt && lastAttempt.consecutiveFailures > 0 && lastAttempt.consecutiveFailures < 3) {
        const hoursSince = (Date.now() - new Date(lastAttempt.lastAttempt).getTime()) / 3600_000;
        if (hoursSince < 12) {
          logger.debug(`Skipping ollama/${modelName} — last attempt ${hoursSince.toFixed(1)}h ago failed.`);
          result.ollamaSkipped.push(`${modelName} (recent fail)`);
          continue;
        }
      }

      logger.info(
        `Pulling ollama/${modelName} — gapScore=${candidate.gapScore.toFixed(2)}, ` +
        `trust=${candidate.trust.trustScore.toFixed(2)}, estVRAM=${estimatedVram.toFixed(1)}GB.`,
      );

      try {
        await this.pullOllamaModel(modelName);
        result.ollamaPulled.push(modelName);
        this.db.recordCuratorModelAttempt("ollama", modelName, true);
        pulled++;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.warn(`Failed to pull ollama/${modelName}: ${msg}`);
        result.errors.push(`Pull failed: ${modelName} — ${msg}`);
        this.db.recordCuratorModelAttempt("ollama", modelName, false);
      }
    }
  }

  /** Fetch models from local Ollama instance. */
  private async getInstalledOllamaModels(): Promise<Array<{ name: string; size: number }>> {
    try {
      const resp = await fetch(`${OLLAMA_API}/api/tags`, {
        signal: AbortSignal.timeout(5000),
      });
      if (!resp.ok) return [];
      const data = await resp.json() as any;
      return (data.models ?? []).map((m: any) => ({
        name: m.name,
        size: m.size ?? 0,
      }));
    } catch {
      return [];
    }
  }

  /** Fetch the Ollama library page and parse model entries. */
  private async fetchOllamaLibrary(): Promise<DiscoveredOllamaModel[]> {
    try {
      const resp = await fetch(OLLAMA_LIBRARY, {
        signal: AbortSignal.timeout(15000),
        headers: { "User-Agent": "CognitiveRouter/Curator" },
      });
      if (!resp.ok) {
        logger.warn(`Ollama library returned HTTP ${resp.status}`);
        return [];
      }
      const html = await resp.text();
      return this.parseOllamaLibraryHtml(html);
    } catch (err) {
      logger.warn(`Failed to fetch Ollama library: ${err instanceof Error ? err.message : err}`);
      return [];
    }
  }

  /** Parse the Ollama library HTML to extract model metadata.
   *  The library page uses server-rendered cards with model info. */
  private parseOllamaLibraryHtml(html: string): DiscoveredOllamaModel[] {
    const models: DiscoveredOllamaModel[] = [];

    // Parse model entries from the library page.
    // Ollama's library page has entries like: <a href="/library/modelname">...</a>
    // with descriptions and pull counts.
    const modelRegex = /href="\/library\/([a-z0-9][a-z0-9._-]*[a-z0-9])"/gi;
    const seen = new Set<string>();
    let match: RegExpExecArray | null;

    while ((match = modelRegex.exec(html)) !== null) {
      const name = match[1];
      if (seen.has(name)) continue;
      seen.add(name);

      // Skip utility/non-model entries
      if (["search", "featured", "popular"].includes(name)) continue;

      // Extract description from nearby text (best-effort)
      const descMatch = html.slice(match.index, match.index + 500).match(/<p[^>]*>([^<]+)<\/p>/i);
      const description = descMatch?.[1]?.trim() ?? "";

      // Extract pull count (formatted like "1.2M Pulls" or "1234 Pulls")
      const pullsMatch = html.slice(match.index, match.index + 1000).match(/([\d.]+)\s*([KMB]?)\s*Pulls/i);
      let pulls = 0;
      if (pullsMatch) {
        const num = parseFloat(pullsMatch[1]);
        const suffix = pullsMatch[2]?.toUpperCase() ?? "";
        const mult = suffix === "K" ? 1e3 : suffix === "M" ? 1e6 : suffix === "B" ? 1e9 : 1;
        pulls = Math.floor(num * mult);
      }

      models.push({ name, description, pulls, tags: [] });
    }

    return models;
  }

  /** Score an Ollama candidate based on capability gaps and trust signals. */
  private scoreOllamaCandidate(
    model: DiscoveredOllamaModel,
    gaps: Map<string, number>,
  ): number {
    const lower = model.name.toLowerCase();

    // Match model family
    let familyMatch: string | null = null;
    for (const family of Object.keys(FAMILY_PROFILES)) {
      if (lower.includes(family)) {
        familyMatch = family;
        break;
      }
    }

    if (!familyMatch) return 0.3; // unknown family — low default

    const profile = FAMILY_PROFILES[familyMatch];
    let gapScore = 0;
    let dimCount = 0;

    for (const dim of CAPABILITY_DIMENSIONS) {
      const capValue = (profile.caps as any)[dim];
      if (capValue !== undefined) {
        const currentBest = gaps.get(dim) ?? 1.0;
        // If our best model in this dimension is < 0.8, a new model at capValue helps
        if (currentBest < 0.8 && capValue > currentBest) {
          gapScore += (capValue - currentBest) * 0.5;
        }
        dimCount++;
      }
    }

    if (dimCount === 0) return 0.3;
    return Math.min(1, gapScore / dimCount + 0.2);
  }

  /** Evaluate trust signals for an Ollama model. */
  private evaluateTrustSignals(model: DiscoveredOllamaModel): TrustSignals {
    const downloadCount = model.pulls;
    const maintainer = model.name.split("/")[0] ?? "unknown";
    const isOfficial = !model.name.includes("/");
    const hasModeration = true; // Ollama library models are curated

    // Trust score: heavily weighted by download count (log scale)
    const logPulls = Math.log10(Math.max(downloadCount, 1));
    const downloadTrust = Math.min(1, logPulls / 7); // 10M pulls = 1.0
    const officialBoost = isOfficial ? 0.15 : 0;
    const moderationBoost = hasModeration ? 0.1 : 0;

    const trustScore = Math.min(1, downloadTrust + officialBoost + moderationBoost);

    return { downloadCount, maintainer, hasModeration, isOfficial, trustScore };
  }

  /** Estimate VRAM requirement for an Ollama model based on name/size. */
  private estimateVramForModel(modelName: string, sizeBytes?: number): number {
    if (sizeBytes && sizeBytes > 0) {
      // Quantized models: roughly bytes / 1.5 for param count, then ~2 bytes per param for VRAM
      const params = sizeBytes / 1.5e9;
      // VRAM = params * ~1.3GB (quantized, with overhead)
      return params * 1.3;
    }

    // Name-based heuristic: look for parameter count in name
    const lower = modelName.toLowerCase();
    const paramMatch = lower.match(/(\d+(?:\.\d+)?)\s*(b|m)/);
    if (paramMatch) {
      const num = parseFloat(paramMatch[1]);
      const unit = paramMatch[2];
      const params = unit === "b" ? num : num / 1000;
      // Quantized (Q4): ~0.7GB per billion params + 20% overhead
      return params * 0.7 * 1.2;
    }

    // Default conservative estimate
    return 8;
  }

  /** Pull an Ollama model via CLI. */
  private async pullOllamaModel(modelName: string): Promise<void> {
    logger.info(`ollama pull ${modelName}...`);
    const { stdout, stderr } = await execAsync(`ollama pull ${modelName}`, {
      timeout: 300_000, // 5 minute timeout for large pulls
      maxBuffer: 10 * 1024 * 1024,
    });
    if (stderr) {
      logger.debug(`ollama pull ${modelName} stderr: ${stderr.slice(0, 200)}`);
    }
    logger.info(`ollama pull ${modelName} complete.`);
  }

  // ─── OpenRouter Scanner ───

  private async scanOpenRouter(result: CuratorResult): Promise<void> {
    const apiModels = await this.fetchOpenRouterModels();
    result.openrouterScanned = apiModels.length;

    if (apiModels.length === 0) {
      logger.debug("OpenRouter API returned no models — skipping.");
      return;
    }

    const apiModelIds = new Set(apiModels.map((m) => m.id));

    // 1. Check for new models to add
    const existingORModels = this.registry
      .getAllModels()
      .filter((m) => m.provider === "openrouter");

    const existingIds = new Set(existingORModels.map((m) => m.model));

    // Find new free models that fill gaps
    const gaps = this.identifyCapabilityGaps();
    const newFreeModels = apiModels
      .filter((m) => m.isFree && !existingIds.has(m.id))
      .map((m) => ({ model: m, gapScore: this.scoreOpenRouterCandidate(m, gaps) }))
      .filter((c) => c.gapScore > 0.2)
      .sort((a, b) => b.gapScore - a.gapScore)
      .slice(0, 5); // max 5 new models per run

    for (const candidate of newFreeModels) {
      const added = this.tryAddOpenRouterModel(candidate.model);
      if (added) {
        result.openrouterAdded.push(candidate.model.id);
        logger.info(
          `Discovered new OpenRouter model: ${candidate.model.id} ` +
          `(gapScore=${candidate.gapScore.toFixed(2)}, ctx=${candidate.model.contextLength.toLocaleString()}).`,
        );
      }
    }

    // 2. Check for dead models to prune
    for (const existing of existingORModels) {
      if (!apiModelIds.has(existing.model)) {
        // Model not in API response — increment prune counter
        const count = this.db.incrementPruneCounter("openrouter", existing.model);
        logger.info(
          `OpenRouter model ${existing.model} not in API response ` +
          `(miss #${count}/${PRUNE_THRESHOLD}).`,
        );

        if (count >= PRUNE_THRESHOLD) {
          logger.warn(`Pruning dead OpenRouter model: ${existing.model} (${count} consecutive misses).`);
          this.registry.removeModel("openrouter", existing.model);
          this.db.resetPruneCounter("openrouter", existing.model);
          result.openrouterPruned.push(existing.model);
        }
      } else {
        // Model exists — reset prune counter
        const counter = this.db.getPruneCounter("openrouter", existing.model);
        if (counter && counter > 0) {
          this.db.resetPruneCounter("openrouter", existing.model);
        }
      }
    }
  }

  /** Fetch models from OpenRouter API. */
  private async fetchOpenRouterModels(): Promise<DiscoveredOpenRouterModel[]> {
    try {
      const resp = await fetch(OPENROUTER_API, {
        signal: AbortSignal.timeout(15000),
        headers: { "User-Agent": "CognitiveRouter/Curator" },
      });
      if (!resp.ok) {
        logger.warn(`OpenRouter API returned HTTP ${resp.status}`);
        return [];
      }
      const data = await resp.json() as any;
      return (data.data ?? []).map((m: any) => ({
        id: m.id ?? "",
        name: m.name ?? m.id ?? "",
        contextLength: m.context_length ?? m.top_provider?.context_length ?? 8_000,
        pricing: {
          prompt: m.pricing?.prompt ?? "0",
          completion: m.pricing?.completion ?? "0",
        },
        isFree: parseFloat(m.pricing?.prompt ?? "0") === 0 && parseFloat(m.pricing?.completion ?? "0") === 0,
        architecture: m.architecture?.modality,
        topProvider: {
          contextLength: m.top_provider?.context_length,
          maxCompletionTokens: m.top_provider?.max_completion_tokens,
        },
      }));
    } catch (err) {
      logger.warn(`Failed to fetch OpenRouter models: ${err instanceof Error ? err.message : err}`);
      return [];
    }
  }

  /** Score an OpenRouter candidate based on capability gaps. */
  private scoreOpenRouterCandidate(
    model: DiscoveredOpenRouterModel,
    gaps: Map<string, number>,
  ): number {
    let score = 0.3; // base score

    // Free models get a boost
    if (model.isFree) score += 0.3;

    // Large context window is valuable
    if (model.contextLength >= 100_000) score += 0.2;
    else if (model.contextLength >= 32_000) score += 0.1;

    // Name-based capability hints
    const lower = model.id.toLowerCase();
    if (lower.includes("coder") || lower.includes("code")) {
      const gap = gaps.get("coding") ?? 1.0;
      if (gap < 0.85) score += 0.2;
    }
    if (lower.includes("reason") || lower.includes("think")) {
      const gap = gaps.get("reasoning") ?? 1.0;
      if (gap < 0.85) score += 0.15;
    }
    if (lower.includes("vision") || lower.includes("vl")) {
      // Vision capability is always useful
      score += 0.1;
    }

    return Math.min(1, score);
  }

  /** Try to add an OpenRouter model to the registry. Returns true if added. */
  private tryAddOpenRouterModel(model: DiscoveredOpenRouterModel): boolean {
    const existing = this.registry.getCapability("openrouter", model.id);
    if (existing) return false;

    // Infer capabilities from model name/family
    const caps = this.inferOpenRouterCapabilities(model.id);
    const contextWindow = model.contextLength || model.topProvider?.contextLength || 32_000;

    this.registry.addDiscoveredModel({
      provider: "openrouter",
      model: model.id,
      contextWindow,
      modalities: ["text"],
      capabilities: caps,
      costPer1kInput: model.isFree ? 0 : parseFloat(model.pricing.prompt) / 1000,
      costPer1kOutput: model.isFree ? 0 : parseFloat(model.pricing.completion) / 1000,
      usageMultiplier: 1,
      isLocal: false,
      source: "inferred",
      planEligible: true,
    });

    return true;
  }

  /** Infer capability scores for an OpenRouter model from its name. */
  private inferOpenRouterCapabilities(modelId: string): ModelCapability["capabilities"] {
    const lower = modelId.toLowerCase();
    const defaults: ModelCapability["capabilities"] = {
      coding: 0.50, reasoning: 0.50, creative: 0.50, math: 0.45,
      analysis: 0.50, conversation: 0.55, retrieval: 0.50,
      science: 0.48, business: 0.48, summary: 0.52,
    };

    // Family-based adjustments
    for (const [family, profile] of Object.entries(FAMILY_PROFILES)) {
      if (lower.includes(family)) {
        Object.assign(defaults, profile.caps);
        break;
      }
    }

    // Special hints
    if (lower.includes("coder") || lower.includes("code")) {
      defaults.coding = Math.max(defaults.coding, 0.75);
    }
    if (lower.includes("reason") || lower.includes("think")) {
      defaults.reasoning = Math.max(defaults.reasoning, 0.78);
    }
    if (lower.includes("instruct")) {
      defaults.conversation = Math.max(defaults.conversation, 0.68);
    }

    return defaults;
  }

  // ─── Gap Analysis ───

  /** Identify capability dimensions where the current model pool is weak.
   *  Returns a map of dimension → current best score (0-1). */
  private identifyCapabilityGaps(): Map<string, number> {
    const bestPerDim = new Map<string, number>();

    for (const dim of CAPABILITY_DIMENSIONS) {
      bestPerDim.set(dim, 0);
    }

    const allModels = this.registry.getAllModels();
    const eligibleProviders = new Set(this.config.providerPriority);

    for (const model of allModels) {
      if (!eligibleProviders.has(model.provider)) continue;
      // Skip embedding-only models
      if (model.provider === "ollama" && model.model.toLowerCase().includes("embed")) continue;

      for (const dim of CAPABILITY_DIMENSIONS) {
        const score = (model.capabilities as any)[dim] ?? 0;
        if (score > (bestPerDim.get(dim) ?? 0)) {
          bestPerDim.set(dim, score);
        }
      }
    }

    return bestPerDim;
  }
}
