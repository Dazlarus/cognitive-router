// src/discovery.ts - Model discovery modes (Daz-spec, 2026-09-06)
//
// Three modes (ROUTER_DISCOVERY_MODE):
//   auto   - query every configured provider's model-list endpoint on
//            startup AND hourly (ROUTER_DISCOVERY_INTERVAL_MINUTES) AND on
//            admin hook. Firehose on initial setup; afterwards only NEW
//            models register (results persist in DB; infra restarts re-sync
//            from cached state without re-benching).
//   safe   - same as auto on startup + admin hook only; no periodic refresh.
//            (Default — matches pre-existing behavior.)
//   manual - no endpoint discovery at all. Only ROUTER_MANUAL_MODELS slugs
//            ("provider/model" comma-separated) register. Seeds still load.
//
// Extended providers beyond the registry's built-ins (zai/ollama):
// openrouter, openai, anthropic, gemini. A provider participates only when its API
// key is configured — keys gate inclusion.
//
// Cost safety: this module REGISTERS models (registry rows, neutral caps).
// It never spends tokens. Benchmarking newly discovered identities is a
// separate, deliberate step (benchmark ladder) owned by its own triggers.

import { logger } from "./logger.js";
import { heuristicCardFor, type ConfigCard } from "./effort_profiles.js";

/** Extended-provider discovery row (model identity + pricing + config card). */
type DiscoveredModel = {
  provider: string;
  model: string;
  costPer1kInput?: number;
  costPer1kOutput?: number;
  configCard?: ConfigCard;
};
import type { ModelRegistry } from "./model_registry.js";
import type { DBService } from "./db_service.js";

export type DiscoveryMode = "auto" | "safe" | "manual";

export interface DiscoverySummary {
  mode: DiscoveryMode;
  triggeredBy: "startup" | "interval" | "admin";
  registered: string[];
  alreadyKnown: number;
  /** Remote models currently quarantined for unknown pricing (open gaps). */
  pricingGapsOpen: number;
  /** Change in open gaps this pass (negative = gaps resolved). */
  pricingGapsDelta: number;
  errors: string[];
  durationMs: number;
}

/** Gemini listing excludes non-chat experiment families. */
const GEMINI_EXCLUDE = /embedding|aqa|imagen|tts|veo|native-audio|lyria/i;

/** OpenClaw's public model catalog — maintained multi-provider pricing
 *  (13.7k+ entries, refreshed upstream ~daily). Read-only GET, no user data
 *  leaves the machine. Override for mirrors/offline via env. */
/** Catalog key candidates: provider/model, provider.model (anthropic dot
 *  notation), vendor aliases (gemini -> google), bare model id. Exported
 *  standalone so tests + the dry-run harness exercise the real lookup. */
export function catalogKeyCandidates(provider: string, model: string): string[] {
  const aliases: Record<string, string[]> = { gemini: ["google"] };
  const prefixes = [provider, ...(aliases[provider] ?? [])];
  const out: string[] = [];
  for (const p of prefixes) {
    out.push(`${p}/${model}`, `${p}.${model}`);
  }
  out.push(model);
  return out;
}

const PRICING_CATALOG_URL =
  process.env.ROUTER_PRICING_CATALOG_URL ??
  "https://catalog.openclaw.ai/models/v1/catalog.json";

export class ModelDiscovery {
  private mode: DiscoveryMode;
  private manualSlugs: Array<{ provider: string; model: string }>;
  private intervalMin: number;
  private timer?: NodeJS.Timeout;
  private running = false;
  private catalogMap?: Map<string, { inputPerM: number; outputPerM: number }>;
  /** Fired (not awaited) after every completed pass. The proxy wires this
   *  to the bench trigger: new identities -> one deliberate ladder insert
   *  per pass (ROUTER_BENCH_TRIGGER=1 gates spending). */
  onPassComplete?: (summary: DiscoverySummary) => void;

  constructor(
    private registry: ModelRegistry,
    private db: DBService,
  ) {
    const raw = (process.env.ROUTER_DISCOVERY_MODE ?? "safe").toLowerCase();
    this.mode = raw === "auto" || raw === "manual" ? raw : "safe";
    this.manualSlugs = (process.env.ROUTER_MANUAL_MODELS ?? "")
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0 && s.includes("/"))
      .map((s) => {
        const idx = s.indexOf("/");
        return { provider: s.slice(0, idx), model: s.slice(idx + 1) };
      });
    this.intervalMin = Math.max(
      5,
      parseInt(process.env.ROUTER_DISCOVERY_INTERVAL_MINUTES ?? "60", 10) || 60,
    );

    // The hook routes startup discovery through mode logic in all modes:
    // manual -> slugs only; auto/safe -> built-ins + extended providers.
    registry.discoveryHook = async () => {
      await this.run("startup");
    };

    logger.info(
      `Model discovery: mode=${this.mode}` +
        (this.mode === "manual" ? ` (${this.manualSlugs.length} manual slugs)` : "") +
        (this.mode === "auto" ? `, refresh every ${this.intervalMin}m` : ""),
    );
  }

  get discoveryMode(): DiscoveryMode {
    return this.mode;
  }

  /** One discovery pass. Safe to call concurrently — serializes on `running`. */
  async run(triggeredBy: "startup" | "interval" | "admin"): Promise<DiscoverySummary> {
    const started = Date.now();
    const summary: DiscoverySummary = {
      mode: this.mode,
      triggeredBy,
      registered: [],
      alreadyKnown: 0,
      pricingGapsOpen: 0,
      pricingGapsDelta: 0,
      errors: [],
      durationMs: 0,
    };
    const gapsBefore = this.db.getPricingGaps().length;

    if (this.running) {
      summary.errors.push("discovery already in progress");
      summary.durationMs = Date.now() - started;
      return summary;
    }
    this.running = true;

    try {
      // Pricing catalog first (read-only GET; cached table survives failures).
      await this.refreshCatalogPricing();
      if (this.mode === "manual") {
        for (const { provider, model } of this.manualSlugs) {
          try {
            const cat = this.lookupCatalogPricing(provider, model);
            if (this.registry.registerExternalModel(provider, model, {
              costPer1kInput: cat?.costPer1kInput,
              costPer1kOutput: cat?.costPer1kOutput,
            })) {
              summary.registered.push(`${provider}/${model}`);
            } else {
              summary.alreadyKnown++;
            }
          } catch (err) {
            summary.errors.push(`${provider}/${model}: ${err instanceof Error ? err.message : err}`);
          }
        }
      } else {
        // auto/safe: registry built-ins (zai + ollama) + extended providers.
        try {
          await this.registry.discoverModels();
        } catch (err) {
          summary.errors.push(`zai/ollama: ${err instanceof Error ? err.message : err}`);
        }
        for (const fetcher of [
          this.fetchOpenRouter,
          this.fetchOpenAI,
          this.fetchAnthropic,
          this.fetchGemini,
        ]) {
          try {
            const found = await fetcher.call(this);
            for (const f of found) {
              const cat = this.lookupCatalogPricing(f.provider, f.model);
              if (this.registry.registerExternalModel(f.provider, f.model, {
                costPer1kInput: f.costPer1kInput ?? cat?.costPer1kInput,
                costPer1kOutput: f.costPer1kOutput ?? cat?.costPer1kOutput,
                configCard: f.configCard,
              })) {
                summary.registered.push(`${f.provider}/${f.model}`);
              } else {
                summary.alreadyKnown++;
              }
            }
          } catch (err) {
            summary.errors.push(`${fetcher.name}: ${err instanceof Error ? err.message : err}`);
          }
        }
      }

      // Pricing-gap sweep (uniform truth across all registration sources —
      // extended providers, manual slugs, zai/ollama built-ins, seeds):
      // remote models with undefined cost are quarantined by the router and
      // recorded here as the actionable gap list; priced models clear gaps.
      const openAfter = this.sweepPricingGaps();
      summary.pricingGapsOpen = openAfter;
      summary.pricingGapsDelta = openAfter - gapsBefore;

      summary.durationMs = Date.now() - started;
      logger.info(
        `Discovery pass (${triggeredBy}): +${summary.registered.length} new, ` +
          `${summary.alreadyKnown} known, ${openAfter} pricing gaps` +
          `${summary.errors.length ? `, ${summary.errors.length} errors` : ""} ` +
          `in ${summary.durationMs}ms`,
      );
      try {
        this.onPassComplete?.(summary);
      } catch (err) {
        logger.warn(`Discovery onPassComplete hook failed: ${err}`);
      }
      return summary;
    } finally {
      this.running = false;
    }
  }

  /** Upsert/resolve pricing gaps from current registry state.
   *  Returns the number of open gaps after the sweep. */
  private sweepPricingGaps(): number {
    const known = new Set(
      this.registry.getAllModels().map((m) => `${m.provider}/${m.model}`),
    );
    for (const m of this.registry.getAllModels()) {
      if (m.isLocal) continue;
      if (m.costPer1kInput === undefined) {
        this.db.upsertPricingGap(m.provider, m.model);
      } else {
        this.db.resolvePricingGap(m.provider, m.model);
      }
    }
    // Close gaps for models no longer in the registry (excluded/removed):
    // they cannot route at all, so an open gap would be misleading.
    for (const g of this.db.getPricingGaps()) {
      if (!known.has(`${g.provider}/${g.model}`)) {
        this.db.resolvePricingGap(g.provider, g.model);
        logger.info(
          `[pricing] gap closed for vanished model ${g.provider}/${g.model}`,
        );
      }
    }
    return this.db.getPricingGaps().length;
  }

  startPeriodic(): void {
    if (this.mode !== "auto") return;
    this.timer = setInterval(
      () => {
        this.run("interval").catch((err) =>
          logger.error(`Periodic discovery failed: ${err}`),
        );
      },
      this.intervalMin * 60_000,
    );
    this.timer.unref?.();
    logger.info(`Periodic discovery armed (every ${this.intervalMin} minutes)`);
  }

  stopPeriodic(): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = undefined;
    }
  }

  // ---------- pricing catalog ----------

  /** Fetch + persist the pricing catalog; keep the in-memory map fresh.
   *  On failure, fall back to the cached sqlite table (survives restarts). */
  private async refreshCatalogPricing(): Promise<void> {
    try {
      const resp = await fetch(PRICING_CATALOG_URL, {
        signal: AbortSignal.timeout(30_000),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = (await resp.json()) as any;
      const pricing = data?.pricing;
      if (!pricing || typeof pricing !== "object") {
        throw new Error("catalog has no pricing record");
      }
      const generatedAt = new Date(
        typeof data.generatedAt === "number" ? data.generatedAt : Date.now(),
      ).toISOString();
      const entries: Array<{ key: string; inputPerM: number; outputPerM: number; cacheReadPerM?: number }> = [];
      for (const [key, v] of Object.entries(pricing) as Array<[string, any]>) {
        if (typeof v?.input !== "number" || typeof v?.output !== "number") continue;
        entries.push({
          key,
          inputPerM: v.input,
          outputPerM: v.output,
          cacheReadPerM: typeof v?.cacheRead === "number" ? v.cacheRead : undefined,
        });
      }
      this.db.replaceCatalogPricing(entries, generatedAt);
      this.catalogMap = new Map(
        entries.map((e) => [e.key, { inputPerM: e.inputPerM, outputPerM: e.outputPerM }]),
      );
      logger.info(
        `Pricing catalog refreshed: ${entries.length} entries (upstream ${generatedAt})`,
      );
    } catch (err) {
      logger.warn(
        `Pricing catalog fetch failed (${err instanceof Error ? err.message : err}) - using cached table`,
      );
      if (!this.catalogMap) {
        this.catalogMap = this.db.getCatalogPricingMap();
        logger.info(`Using cached pricing catalog: ${this.catalogMap.size} entries`);
      }
    }
  }

  /** Catalog key candidates (standalone, tested). */
  private catalogKeyCandidates(provider: string, model: string): string[] {
    return catalogKeyCandidates(provider, model);
  }

  /** Price lookup with exact + variant-suffix matching (:0, :batch, :v2 …).
   *  Shortest suffix wins (closest to base on-demand price). Returns per-1k
   *  costs, or undefined when the catalog does not know the model. */
  private lookupCatalogPricing(
    provider: string,
    model: string,
  ): { costPer1kInput: number; costPer1kOutput: number } | undefined {
    const map = this.catalogMap;
    if (!map || map.size === 0) return undefined;
    for (const k of this.catalogKeyCandidates(provider, model)) {
      const hit = map.get(k);
      if (hit) {
        return { costPer1kInput: hit.inputPerM / 1000, costPer1kOutput: hit.outputPerM / 1000 };
      }
    }
    let best: { key: string; v: { inputPerM: number; outputPerM: number } } | undefined;
    for (const base of this.catalogKeyCandidates(provider, model)) {
      const prefix = `${base}:`;
      for (const [k, v] of map) {
        if (k.startsWith(prefix) && (!best || k.length < best.key.length)) {
          best = { key: k, v };
        }
      }
    }
    if (best) {
      return {
        costPer1kInput: best.v.inputPerM / 1000,
        costPer1kOutput: best.v.outputPerM / 1000,
      };
    }
    return undefined;
  }

  // ---------- extended providers ----------

  private async fetchOpenRouter(): Promise<DiscoveredModel[]> {
    const apiKey = process.env.OPENROUTER_API_KEY;
    if (!apiKey) return [];
    const resp = await fetch("https://openrouter.ai/api/v1/models", {
      headers: { Authorization: `Bearer ${apiKey}` },
      signal: AbortSignal.timeout(15_000),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = (await resp.json()) as any;
    const out: DiscoveredModel[] = [];
    for (const m of data.data ?? []) {
      if (typeof m?.id !== "string" || !m.id) continue;
      // Chat-capable only: output must include text.
      const outputs: string[] = m?.architecture?.output_modalities ?? ["text"];
      if (!outputs.includes("text")) continue;
      // Pricing ships per-million tokens as strings; store per-1k.
      const inPerM = parseFloat(m?.pricing?.prompt);
      const outPerM = parseFloat(m?.pricing?.completion);
      const cacheReadPerM = parseFloat(m?.pricing?.cache_read);
      const params: string[] = Array.isArray(m?.supported_parameters) ? m.supported_parameters : [];
      out.push({
        provider: "openrouter",
        model: m.id,
        costPer1kInput: Number.isFinite(inPerM) ? inPerM / 1000 : undefined,
        costPer1kOutput: Number.isFinite(outPerM) ? outPerM / 1000 : undefined,
        // Config card from live capability data (seed cards still win on merge).
        configCard: heuristicCardFor("openrouter", m.id, {
          cacheReadPrice: Number.isFinite(cacheReadPerM) ? cacheReadPerM : undefined,
          supportsReasoningEffort: params.includes("reasoning") || params.includes("reasoning_effort"),
        }),
      });
    }
    return out;
  }

  private async fetchOpenAI(): Promise<DiscoveredModel[]> {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) return [];
    const base =
      process.env.OPENAI_BASE_URL?.replace(/\/+$/, "") ??
      "https://api.openai.com/v1";
    const resp = await fetch(`${base}/models`, {
      headers: { Authorization: `Bearer ${apiKey}` },
      signal: AbortSignal.timeout(15_000),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = (await resp.json()) as any;
    // /v1/models lists every model family (embeddings, tts, whisper,
    // moderation, image …). Keep chat-capable families only; pricing comes
    // from the catalog lookup in run(), same as anthropic/gemini.
    const CHAT_FAMILY = /^(gpt-|o\d|chatgpt-|codex)/;
    const NON_CHAT = /embed|whisper|tts|audio|realtime|moderation|image|transcribe|search|video/i;
    const out: DiscoveredModel[] = [];
    for (const m of data.data ?? []) {
      if (typeof m?.id !== "string" || !m.id) continue;
      if (!CHAT_FAMILY.test(m.id) || NON_CHAT.test(m.id)) continue;
      out.push({ provider: "openai", model: m.id, configCard: heuristicCardFor("openai", m.id, { supportsReasoningEffort: true }) });
    }
    return out;
  }

  private async fetchAnthropic(): Promise<DiscoveredModel[]> {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) return [];
    const resp = await fetch("https://api.anthropic.com/v1/models?limit=1000", {
      headers: {
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      signal: AbortSignal.timeout(15_000),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = (await resp.json()) as any;
    const out: DiscoveredModel[] = [];
    for (const m of data.data ?? []) {
      if (typeof m?.id === "string" && m.id) {
        out.push({ provider: "anthropic", model: m.id, configCard: heuristicCardFor("anthropic", m.id, { supportsReasoningEffort: true }) });
      }
    }
    return out;
  }

  private async fetchGemini(): Promise<DiscoveredModel[]> {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) return [];
    const base =
      process.env.GEMINI_BASE_URL?.replace(/\/$/, "") ??
      "https://generativelanguage.googleapis.com/v1beta";
    const resp = await fetch(`${base}/models`, {
      headers: { "x-goog-api-key": apiKey },
      signal: AbortSignal.timeout(15_000),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = (await resp.json()) as any;
    const out: DiscoveredModel[] = [];
    for (const m of data.models ?? []) {
      const name: string | undefined = m?.name;
      if (!name) continue;
      const model = name.replace(/^models\//, "");
      if (GEMINI_EXCLUDE.test(model)) continue;
      // Gemini thinking = thinkingBudget ladder (0|1024|8192|24576) -> normalized low/medium/high + off.
      out.push({ provider: "gemini", model, configCard: heuristicCardFor("gemini", model, { supportsReasoningEffort: true }) });
    }
    return out;
  }
}
