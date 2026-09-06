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
// openrouter, anthropic, gemini. A provider participates only when its API
// key is configured — keys gate inclusion.
//
// Cost safety: this module REGISTERS models (registry rows, neutral caps).
// It never spends tokens. Benchmarking newly discovered identities is a
// separate, deliberate step (benchmark ladder) owned by its own triggers.

import { logger } from "./logger.js";
import type { ModelRegistry } from "./model_registry.js";

export type DiscoveryMode = "auto" | "safe" | "manual";

export interface DiscoverySummary {
  mode: DiscoveryMode;
  triggeredBy: "startup" | "interval" | "admin";
  registered: string[];
  alreadyKnown: number;
  errors: string[];
  durationMs: number;
}

/** Gemini listing excludes non-chat experiment families. */
const GEMINI_EXCLUDE = /embedding|aqa|imagen|tts|veo|native-audio/i;

export class ModelDiscovery {
  private mode: DiscoveryMode;
  private manualSlugs: Array<{ provider: string; model: string }>;
  private intervalMin: number;
  private timer?: NodeJS.Timeout;
  private running = false;

  constructor(private registry: ModelRegistry) {
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
      errors: [],
      durationMs: 0,
    };

    if (this.running) {
      summary.errors.push("discovery already in progress");
      summary.durationMs = Date.now() - started;
      return summary;
    }
    this.running = true;

    try {
      if (this.mode === "manual") {
        for (const { provider, model } of this.manualSlugs) {
          try {
            if (this.registry.registerExternalModel(provider, model)) {
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
          this.fetchAnthropic,
          this.fetchGemini,
        ]) {
          try {
            const found = await fetcher.call(this);
            for (const f of found) {
              if (this.registry.registerExternalModel(f.provider, f.model, {
                costPer1kInput: f.costPer1kInput,
                costPer1kOutput: f.costPer1kOutput,
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

      summary.durationMs = Date.now() - started;
      logger.info(
        `Discovery pass (${triggeredBy}): +${summary.registered.length} new, ` +
          `${summary.alreadyKnown} known${summary.errors.length ? `, ${summary.errors.length} errors` : ""} ` +
          `in ${summary.durationMs}ms`,
      );
      return summary;
    } finally {
      this.running = false;
    }
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

  // ---------- extended providers ----------

  private async fetchOpenRouter(): Promise<Array<{ provider: string; model: string; costPer1kInput?: number; costPer1kOutput?: number }>> {
    const apiKey = process.env.OPENROUTER_API_KEY;
    if (!apiKey) return [];
    const resp = await fetch("https://openrouter.ai/api/v1/models", {
      headers: { Authorization: `Bearer ${apiKey}` },
      signal: AbortSignal.timeout(15_000),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = (await resp.json()) as any;
    const out: Array<{ provider: string; model: string; costPer1kInput?: number; costPer1kOutput?: number }> = [];
    for (const m of data.data ?? []) {
      if (typeof m?.id !== "string" || !m.id) continue;
      // Chat-capable only: output must include text.
      const outputs: string[] = m?.architecture?.output_modalities ?? ["text"];
      if (!outputs.includes("text")) continue;
      // Pricing ships per-million tokens as strings; store per-1k.
      const inPerM = parseFloat(m?.pricing?.prompt);
      const outPerM = parseFloat(m?.pricing?.completion);
      out.push({
        provider: "openrouter",
        model: m.id,
        costPer1kInput: Number.isFinite(inPerM) ? inPerM / 1000 : undefined,
        costPer1kOutput: Number.isFinite(outPerM) ? outPerM / 1000 : undefined,
      });
    }
    return out;
  }

  private async fetchAnthropic(): Promise<Array<{ provider: string; model: string; costPer1kInput?: number; costPer1kOutput?: number }>> {
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
    const out: Array<{ provider: string; model: string }> = [];
    for (const m of data.data ?? []) {
      if (typeof m?.id === "string" && m.id) {
        out.push({ provider: "anthropic", model: m.id });
      }
    }
    return out;
  }

  private async fetchGemini(): Promise<Array<{ provider: string; model: string; costPer1kInput?: number; costPer1kOutput?: number }>> {
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
    const out: Array<{ provider: string; model: string }> = [];
    for (const m of data.models ?? []) {
      const name: string | undefined = m?.name;
      if (!name) continue;
      const model = name.replace(/^models\//, "");
      if (GEMINI_EXCLUDE.test(model)) continue;
      out.push({ provider: "gemini", model });
    }
    return out;
  }
}
