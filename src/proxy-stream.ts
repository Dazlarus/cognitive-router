// src/proxy-stream.ts - Streaming + self-healing proxy for Cognitive Router

import http from "node:http";
import crypto from "node:crypto";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { logger } from "./logger.js";
import { exec } from "node:child_process";
import { promisify } from "node:util";

const execAsync = promisify(exec);

/** Constant-time string compare for secrets (keys/tokens). */
function timingSafeEqualStr(a: string, b: string): boolean {
  const ab = Buffer.from(a, "utf8");
  const bb = Buffer.from(b, "utf8");
  if (ab.length !== bb.length) return false;
  return crypto.timingSafeEqual(ab, bb);
}

// Exploration probe guardrails: never probe large requests (a bad pick on a
// big context wastes real money) and stop probing a cell once it has enough
// judge observations to be trusted.
const PROBE_MAX_INPUT_TOKENS = 8_000;
const PROBE_MIN_SAMPLES = 10;
import { IntentClassifier } from "./classifier.js";
import { RoutingEngine } from "./router.js";
import { DBService } from "./db_service.js";
import { CostTracker } from "./cost_tracker.js";
import { ModelRegistry } from "./model_registry.js";
import { loadConfig, type CognitiveRouterConfig } from "./config.js";
import { getProvider, extractThinkingLevel, extractSpeedMode } from "./providers.js";
import { loadProjectEnv } from "./env.js";
import { buildStatsPayload } from "./stats.js";
import { isGenerationModel, modelSupportsTools } from "./model_policy.js";
import { BudgetTracker } from "./budget_tracker.js";
import { ModelCurator } from "./curator.js";
import { ModelDiscovery } from "./discovery.js";
import { PROMPT_GENERATION, rebuildLadder } from "./benchmark_ladder.js";
import { withUpstreamTimeoutOverride } from "./upstream_timeout_scope.js";
import { JudgeEvaluator } from "./judge.js";
import { decideOutboundEffort, effortPolicyMode } from "./effort_policy.js";
import { getZaiQuotaProbe } from "./quota_probe.js";
import { EmbeddingBenchmark, initializeBenchmarkTables } from "./benchmark_embeddings.js";
import type { EmbeddingModelInfo } from "./benchmark_embeddings.js";
import { raceHedgedRequests, hedgeRetryDelayMs, type HedgeCandidate } from "./hedged_request.js";
import { classifyFailure, computeFallbackDecision, providerBackoff } from "./failure_classifier.js";
import { registryReadiness } from "./readiness.js";
import { detectModalities, type Modality } from "./modality.js";
import { computeProxyDecisionSource, decisionSourceCounters, SELECTED_MODEL_HEADER, UPSTREAM_MODEL_HEADER } from "./decision_source.js";

const CHAT_ALIAS_MODEL = "CognitiveRouter:latest";
const LEGACY_CHAT_ALIAS_MODEL = "CogRouter:latest";
const EMBEDDING_ALIAS_MODEL = "Embeddings:latest";
const CHAT_RESPONSE_MODEL = CHAT_ALIAS_MODEL;
const EMBEDDING_RESPONSE_MODEL = EMBEDDING_ALIAS_MODEL;
const DEFAULT_OPENROUTER_TOOL_MODEL = "qwen/qwen3-30b-a3b-instruct-2507";
const DEFAULT_OPENROUTER_FALLBACK_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free";
const DEFAULT_GEMINI_FALLBACK_MODEL = "gemini-2.5-flash";
const DEFAULT_GEMINI_TOOL_MODEL = "gemini-2.5-flash";
const DEFAULT_OPENROUTER_PAID_FALLBACK_MODEL = "deepseek/deepseek-v4-flash";
const ALLOWED_OPENROUTER_FALLBACK_MODELS = new Set([
  "cohere/north-mini-code:free",
  "openrouter/free",
  "nvidia/nemotron-3-ultra-550b-a55b:free",
  "poolside/laguna-m.1:free",
  "qwen/qwen3-30b-a3b-instruct-2507",
]);

// Paid OpenRouter models allowed as mid-tier fallback (after free, before local)
const ALLOWED_OPENROUTER_PAID_MODELS = new Set([
  "deepseek/deepseek-v4-flash",
  "qwen/qwen3-235b-a22b-2507",
  "qwen/qwen3-coder-30b-a3b-instruct",
]);

// Load .env before standalone proxy code reads provider keys from process.env.
loadProjectEnv(import.meta.url);

// ─── Types ───

export interface ChatMessage {
  role: string;
  content: string;
}

export interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  thinking?: any;
  reasoning?: any;
  reasoning_effort?: string;
  speed?: string;
  [key: string]: any;
}

// ─── Token estimation & provider context limits ───

/**
 * Rough token estimate: ~4 chars per token for typical mixed text/JSON.
 * This is intentionally conservative (overestimates slightly) so we err
 * on the side of skipping providers that would reject the request.
 */
// Lazy-loaded tiktoken encoder for accurate token counting.
// cl100k_base is used by GPT-4, Qwen, and most modern models.
let _tiktoken: { encode(text: string): number[] } | null = null;
let _tiktokenLoadFailed = false;

// ESM-safe loader: this package compiles to ESM ("type": "module"), so require()
// is undefined in the bundle. Load via dynamic import at module init so
// getTiktoken() stays synchronous for callers.
void (async () => {
  try {
    const { getEncoding } = await import("js-tiktoken");
    _tiktoken = getEncoding("cl100k_base");
  } catch (e) {
    _tiktokenLoadFailed = true;
    logger.warn(`tiktoken load failed, falling back to char estimation: ${e instanceof Error ? e.message : e}`);
  }
})();

function getTiktoken(): { encode(text: string): number[] } | null {
  if (_tiktokenLoadFailed) return null;
  return _tiktoken;
}

export function estimateTokenCount(request: ChatCompletionRequest): number {
  const encoder = getTiktoken();
  if (encoder) {
    let tokenCount = 0;
    for (const msg of request.messages ?? []) {
      if (typeof msg.content === "string") {
        tokenCount += encoder.encode(msg.content).length;
      } else if (Array.isArray(msg.content)) {
        tokenCount += encoder.encode(JSON.stringify(msg.content)).length;
      }
      tokenCount += 4; // role + formatting overhead
    }
    if (Array.isArray(request.tools) && request.tools.length > 0) {
      tokenCount += encoder.encode(JSON.stringify(request.tools)).length;
    }
    if (Array.isArray(request.functions) && request.functions.length > 0) {
      tokenCount += encoder.encode(JSON.stringify(request.functions)).length;
    }
    return tokenCount;
  }

  // Fallback: char-based heuristic
  let chars = 0;
  for (const msg of request.messages ?? []) {
    if (typeof msg.content === "string") {
      chars += msg.content.length;
    } else if (Array.isArray(msg.content)) {
      chars += JSON.stringify(msg.content).length;
    }
    chars += (msg.role?.length ?? 0) + 5;
  }
  if (Array.isArray(request.tools)) {
    chars += JSON.stringify(request.tools).length;
  }
  if (Array.isArray(request.functions)) {
    chars += JSON.stringify(request.functions).length;
  }
  return Math.ceil(chars / 3.5);
}

const PROVIDER_EFFECTIVE_INPUT_LIMITS: Record<string, number> = {
  zai: 200_000,
  openrouter: 66_327,
  gemini: 1_000_000,
  ollama: 32_768,
};

const MIN_USEFUL_CONTEXT_TOKENS = Number.parseInt(
  process.env.ROUTER_MIN_CONTEXT_TOKENS ?? "86000",
  10,
) || 86_000;

const CONTEXT_SAFETY_MARGIN = 1 - Number.parseFloat(
  process.env.ROUTER_CONTEXT_SAFETY_MARGIN ?? "0.20",
);

function getEffectiveInputLimit(provider: string, model: string, modelWindow?: number): number {
  const raw = (() => {
    if (provider === "openrouter") {
      const isFreeModel = model.endsWith(":free");
      if (isFreeModel) {
        return PROVIDER_EFFECTIVE_INPUT_LIMITS.openrouter;
      }
      return 256_000;
    }
    const providerDefault = PROVIDER_EFFECTIVE_INPUT_LIMITS[provider] ?? 128_000;
    // Context gate (2026-09-03): a registry-known per-model window LARGER than
    // the provider default lifts this model's ceiling (glm-5.2 1M); smaller or
    // unknown windows keep the provider default. Provider map stays the
    // conservative baseline for every other model.
    return Math.max(providerDefault, modelWindow ?? 0);
  })();
  return Math.floor(raw * CONTEXT_SAFETY_MARGIN);
}

function maxAttemptsPerProvider(): number {
  const raw = Number.parseInt(process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER ?? "1", 10);
  return Number.isFinite(raw) && raw > 0 ? raw : 1;
}

function routerRequestTimeoutMs(): number {
  const raw = Number.parseInt(process.env.ROUTER_REQUEST_TIMEOUT_MS ?? "55000", 10);
  return Number.isFinite(raw) && raw > 0 ? raw : 55_000;
}

function streamStallTimeoutMs(): number {
  const raw = Number.parseInt(process.env.ROUTER_STREAM_STALL_TIMEOUT_MS ?? "45000", 10);
  return Number.isFinite(raw) && raw > 0 ? raw : 45_000;
}

/** Per-request deadline override from a trusted caller (bench sidecar).
 *  Honored ONLY for loopback connections and capped at 10 minutes: the
 *  default 55s failover budget exists so production requests fail over
 *  before OpenClaw's outer LLM timeout, but the bench sidecar's
 *  non-streaming generations legitimately run minutes (CONTRACT.md §2,
 *  max_tokens 6144 reasoning gens). Cap keeps a buggy caller from parking
 *  the router forever. */
function requestTimeoutOverrideMs(req?: http.IncomingMessage): number | null {
  if (!req) return null;
  const raw = req.headers["x-router-timeout-ms"];
  if (typeof raw !== "string" || raw.length === 0) return null;
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n) || n <= 0) return null;
  const remote = req.socket.remoteAddress ?? "";
  const isLoopback = remote === "127.0.0.1" || remote === "::1" || remote === "::ffff:127.0.0.1";
  if (!isLoopback) return null;
  // 900s cap (Daz 2026-09-09 00:15): bench-pinned as-served generations for
  // slow thinkers legitimately run 5-8 min; loopback-only callers may ask for
  // up to 15 min. Normal (non-pinned) traffic is unaffected — this override
  // only applies to requests that carry the header from localhost.
  return Math.min(n, 900_000);
}

// ─── Buffered streaming types ───

interface BufferedChunk {
  // Deep-cloned SSE chunk from provider, with model scrubbed
  chunk: any;
}

interface BufferedStreamResult {
  chunks: BufferedChunk[];
  accumulatedContent: string;
  finishReason: string | null;
  hasPayload: boolean;
}

function requestUsesTools(request: ChatCompletionRequest): boolean {
  return Boolean(
    (Array.isArray(request.tools) && request.tools.length > 0) ||
    request.tool_choice !== undefined ||
    (Array.isArray(request.functions) && request.functions.length > 0) ||
    request.function_call !== undefined,
  );
}

function providerSupportsTools(provider: string): boolean {
  return provider === "zai" || provider === "openrouter" || provider === "gemini" || provider === "ollama";
}

export function fallbackModelForProvider(provider: string, usesTools: boolean): string | null {
  if (provider === "openrouter") {
    return usesTools
      ? cheapOpenRouterModel(process.env.ROUTER_OPENROUTER_TOOL_MODEL, DEFAULT_OPENROUTER_TOOL_MODEL)
      : cheapOpenRouterModel(process.env.ROUTER_OPENROUTER_FALLBACK_MODEL, DEFAULT_OPENROUTER_FALLBACK_MODEL);
  }
  if (provider === "gemini") {
    return usesTools
      ? process.env.ROUTER_GEMINI_TOOL_MODEL ?? DEFAULT_GEMINI_TOOL_MODEL
      : process.env.ROUTER_GEMINI_FALLBACK_MODEL ?? DEFAULT_GEMINI_FALLBACK_MODEL;
  }
  return null;
}

/**
 * Returns the paid OpenRouter fallback model for when free models are exhausted
 * or when the request is too large for free-tier context limits.
 * Configurable via ROUTER_OPENROUTER_PAID_MODEL env var.
 */
export function paidOpenRouterFallbackModel(): string {
  const configured = process.env.ROUTER_OPENROUTER_PAID_MODEL?.trim();
  if (configured && ALLOWED_OPENROUTER_PAID_MODELS.has(configured.toLowerCase())) {
    return configured;
  }
  return DEFAULT_OPENROUTER_PAID_FALLBACK_MODEL;
}

function cheapOpenRouterModel(value: string | undefined, fallback: string): string {
  const model = value?.trim();
  if (!model) return fallback;

  if (!model.endsWith(":free") && !ALLOWED_OPENROUTER_FALLBACK_MODELS.has(model.toLowerCase())) {
    logger.warn(`Ignoring non-cheap OpenRouter fallback model ${model}; using ${fallback}.`);
    return fallback;
  }

  return model;
}

function sanitizeErrorForClient(message: string): string {
  return message
    .replace(/https:\/\/openrouter\.ai\/workspaces\/[^"'\s)\]]+/gi, "[openrouter-key-settings]")
    .replace(/user_[A-Za-z0-9]+/g, "[provider-user]")
    .replace(/[A-Fa-f0-9]{32,}/g, "[redacted]");
}

/**
 * Calculate the dollar cost of a request based on token usage and model pricing.
 * Returns 0 for free models (no cost data).
 */
function calculateRequestCost(
  provider: string,
  model: string,
  modelRegistry: ModelRegistry,
  usage: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number } | undefined,
): number {
  if (!usage) return 0;
  const cap = modelRegistry.getCapability(provider, model);
  if (!cap || (!cap.costPer1kInput && !cap.costPer1kOutput)) return 0;

  const inputTokens = usage.prompt_tokens ?? 0;
  const outputTokens = usage.completion_tokens ?? 0;
  const inputCost = (cap.costPer1kInput ?? 0) * (inputTokens / 1000);
  const outputCost = (cap.costPer1kOutput ?? 0) * (outputTokens / 1000);

  return inputCost + outputCost;
}

/**
 * Check if a model is a paid model (has non-zero token costs).
 */
function isPaidModel(provider: string, model: string, modelRegistry: ModelRegistry): boolean {
  const cap = modelRegistry.getCapability(provider, model);
  return Boolean(cap && ((cap.costPer1kInput ?? 0) > 0 || (cap.costPer1kOutput ?? 0) > 0));
}

export class ProxyServerStreaming {
  private server: http.Server;
  private classifier: IntentClassifier;
  private router: RoutingEngine;
  private db: DBService;
  private costTracker: CostTracker;
  private budgetTracker: BudgetTracker;
  private modelRegistry: ModelRegistry;
  private config: CognitiveRouterConfig;
  private embedFn: (text: string) => Promise<number[]>;
  private judge: JudgeEvaluator;
  private curator: ModelCurator;
  private discovery: ModelDiscovery;
  private initialized = false;

  constructor(config: CognitiveRouterConfig) {
    this.config = config;

    const dbPath = resolve(config.dbPath);
    mkdirSync(resolve(dbPath, ".."), { recursive: true });

    this.db = new DBService(dbPath);
    this.db.initializeSchema();

    // Initialize benchmark tables
    initializeBenchmarkTables(this.db);

    this.modelRegistry = new ModelRegistry(this.db, config);
    this.costTracker = new CostTracker(this.db, config);
    this.budgetTracker = new BudgetTracker(this.db, this.costTracker);
    this.classifier = new IntentClassifier({ tiebreakerThreshold: config.tiebreakerThreshold });
    this.router = new RoutingEngine(this.modelRegistry, this.costTracker, this.db, config);
    this.router.setBudgetTracker(this.budgetTracker);

    this.embedFn = this.createEmbedder();
    this.judge = new JudgeEvaluator();
    this.curator = new ModelCurator(this.db, this.modelRegistry, config);
    this.discovery = new ModelDiscovery(this.modelRegistry, this.db);
    // Bench subsystem EXTRACTED (2026-09-07): the cogrouter-bench sidecar now
    // owns passes/judging; verdicts arrive via POST /admin/bench/sync and
    // routing consumes them after restart (restart-to-apply, CONTRACT.md §6).
    // At startup: rebuild the coding ladder from synced verdict rows so the
    // projection is consistent with the verdict table (idempotent).
    try {
      const rebuilt = rebuildLadder(this.db, "coding", PROMPT_GENERATION);
      if (rebuilt.length > 0) this.db.replaceLadder("coding", PROMPT_GENERATION, rebuilt);
      logger.info(`Ladder projection rebuilt from synced verdicts: ${rebuilt.length} identities`);
    } catch (err) {
      logger.warn(`Ladder rebuild skipped: ${err instanceof Error ? err.message : String(err)}`);
    }
    this.server = http.createServer((req, res) => {
      this.handleRequest(req, res).catch((err) => {
        logger.error(`Unhandled error: ${err}`);
        if (!res.headersSent) {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: { message: "Internal error" } }));
        }
      });
    });
  }

  async start(): Promise<void> {
    // Kill any lingering process on the proxy port to avoid EADDRINUSE
    const port = this.config.proxyPort ?? 3456;
    await this.cleanupPort(port);

    // Load model registry
    await this.modelRegistry.loadCachedState();
    await this.costTracker.refreshProviderStatus();

    const host = this.config.bindHost ?? "127.0.0.1";
    const loopback = host === "127.0.0.1" || host === "localhost" || host === "::1";
    const apiKeys = this.config.apiKeys ?? [];
    if (!loopback && apiKeys.length === 0) {
      logger.error(
        `FATAL: bind host ${host} is non-loopback but no API keys are configured. ` +
          `Central-server mode requires ROUTER_API_KEYS (comma-separated Bearer keys). ` +
          `Refusing to start. Bind 127.0.0.1 for open loopback-only mode.`
      );
      process.exit(1);
    }
    this.server.listen(this.config.proxyPort, host, () => {
      logger.info(
        `Cognitive Router proxy listening on http://${host}:${this.config.proxyPort}` +
          ` (auth: ${apiKeys.length > 0 ? `${apiKeys.length} key(s) required` : "open, loopback-only"})`
      );
      logger.info(`   POST /v1/chat/completions    - Chat (streaming + non-streaming)`);
      logger.info(`   POST /v1/embeddings           - Embeddings (caller-specified model or default)`);
      logger.info(`   GET  /v1/embeddings/models    - List all available embedding models`);
      logger.info(`   GET  /v1/embeddings/benchmarks - Embedding benchmark results`);
      logger.info(`   GET  /v1/models               - List available models`);
      logger.info(`   GET  /health                  - Health check`);
      logger.info(`   GET  /stats                   - Provider health + stats`);
      logger.info(`   GET  /v1/dashboard?range=24h   - Aggregated routing dashboard`);
      logger.info(`   POST /v1/benchmark/embeddings  - Run embedding model benchmark`);
      logger.info(`   POST /v1/curate                - Trigger model curator manually`);
    });

    // Start periodic curator (every 6h)
    this.curator.startPeriodic();

    // Start periodic model discovery (hourly, auto mode only)
    this.discovery.startPeriodic();

    // Check Ollama health before starting classifier embeddings
    const ollamaHealthy = await this.checkOllamaHealth();
    if (!ollamaHealthy) {
      logger.warn("Ollama not reachable - embedding classifier will use keyword fallback");
    }

    // Initialize classifier embeddings in the background. Chat can use keyword
    // fallback until prototypes are ready.
    this.initialized = true;
    logger.info("Proxy ready");
    this.classifier.initialize(this.embedFn)
      .then(() => { logger.info("Embedding classifier ready"); })
      .catch((err) => {
        logger.warn(`Classifier init failed (keyword fallback active): ${err}`);
      });
  }

  async stop(): Promise<void> {
    this.curator.stopPeriodic();
    this.discovery.stopPeriodic();
    this.server.close();
    this.db.close();
  }

  // ─── Request Handler ───

  private async handleRequest(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    await registryReadiness.wait();
    // CORS
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Headers", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    if (req.method === "OPTIONS") {
      res.writeHead(204);
      res.end();
      return;
    }

    try {
      const url = req.url ?? "";
      // URL query parsing helper - avoids new URL() issues with proxied requests
      const urlQuery = url.includes("?") ? new URLSearchParams(url.split("?")[1]) : new URLSearchParams();

      if (url === "/health" && req.method === "GET") {
        // Include budget status in health response
        const dailyProjection = this.budgetTracker.projectBudget("daily", this.costTracker.dailyBudget);
        const monthlyProjection = this.budgetTracker.projectBudget("monthly", this.costTracker.monthlyBudget);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({
          status: "ok",
          initialized: this.initialized,
          budget: {
            daily: {
              spentUsd: dailyProjection.spentUsd,
              budgetUsd: dailyProjection.budgetUsd,
              remainingUsd: dailyProjection.remainingUsd,
              consumedFraction: dailyProjection.consumedFraction,
              burnRatePerHour: dailyProjection.burnRatePerHour,
              autoDowngradeActive: dailyProjection.autoDowngradeActive,
              downgradeReason: dailyProjection.downgradeReason,
            },
            monthly: {
              spentUsd: monthlyProjection.spentUsd,
              budgetUsd: monthlyProjection.budgetUsd,
              remainingUsd: monthlyProjection.remainingUsd,
              consumedFraction: monthlyProjection.consumedFraction,
              burnRatePerHour: monthlyProjection.burnRatePerHour,
              autoDowngradeActive: monthlyProjection.autoDowngradeActive,
              downgradeReason: monthlyProjection.downgradeReason,
            },
          },
        }));
        return;
      }

      // GET /v1/budget — detailed budget status with burn rate projection
      // Inbound auth (central-server mode): active only when apiKeys configured.
      // /health stays open for load-balancer probes; /admin/restart carries its
      // own admin token on top of this check.
      if ((this.config.apiKeys?.length ?? 0) > 0) {
        const auth = req.headers["authorization"];
        const bearer = typeof auth === "string" && auth.startsWith("Bearer ") ? auth.slice(7) : "";
        const ok = !!bearer && (this.config.apiKeys ?? []).some((k) => timingSafeEqualStr(bearer, k));
        if (!ok) {
          res.writeHead(401, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "unauthorized", message: "Authorization: Bearer <key> required" }));
          return;
        }
      }

      if (url === "/v1/budget" && req.method === "GET") {
        const status = this.budgetTracker.getBudgetStatus();
        // Enrich with cost efficiency data from model registry
        const allModels = this.modelRegistry.getAllModels();
        const costEfficiencies = this.budgetTracker.getCostEfficiencyForModels(
          allModels,
          "conversation",
          this.modelRegistry,
        );
        status.costEfficiency = costEfficiencies;
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(status, null, 2));
        return;
      }

      if (url === "/v1/models" && req.method === "GET") {
        const aliases = [
          {
            id: CHAT_ALIAS_MODEL,
            object: "model",
            owned_by: "CognitiveRouter",
            context_window: 200_000,
          },
          {
            id: LEGACY_CHAT_ALIAS_MODEL,
            object: "model",
            owned_by: "CognitiveRouter",
            context_window: 200_000,
          },
          {
            id: EMBEDDING_ALIAS_MODEL,
            object: "model",
            owned_by: "CognitiveRouter",
            context_window: 8_000,
          },
        ];
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ object: "list", data: aliases }));
        return;
      }

      if (url === "/stats" && req.method === "GET") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(buildStatsPayload(this.modelRegistry, this.costTracker, this.config), null, 2));
        return;
      }

      if (url.startsWith("/judge/effort") && req.method === "GET") {
        const days = Math.max(1, Math.min(90, parseInt(urlQuery.get("days") ?? "7", 10) || 7));
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ windowDays: days, buckets: this.modelRegistry.getJudgeQualityByEffort(days) }, null, 2));
        return;
      }

      if (url.startsWith("/v1/fallback") && req.method === "GET") {
        const providerParam = urlQuery.get("provider");
        const modelParam = urlQuery.get("model");
        const turnParam = urlQuery.get("turn");
        const reasonParam = urlQuery.get("reason");

        if (!providerParam || typeof providerParam !== "string" || !providerParam.trim()) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "provider query param is required and must be non-empty" }));
          return;
        }

        // Must match configured provider
        if (!this.config.providers[providerParam]) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: `provider '${providerParam}' is not a configured provider` }));
          return;
        }

        if (!modelParam || typeof modelParam !== "string" || !modelParam.trim()) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "model query param is required and must be non-empty" }));
          return;
        }

        if (turnParam !== null) {
          const turn = parseInt(turnParam, 10);
          if (isNaN(turn) || turn < 0 || turn > 1000) {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: "turn must be an integer between 0 and 1000" }));
            return;
          }
        }

        if (reasonParam !== null) {
          const validReasons = ["abort", "timeout", "stall", "context_overflow"];
          if (!validReasons.includes(reasonParam)) {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: `reason must be one of: ${validReasons.join(", ")}` }));
            return;
          }
        }

        try {
          const decision = await this.router.getFallback(providerParam, modelParam, reasonParam ?? undefined);
          if (!decision) {
            res.writeHead(404, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: "No fallback candidate available" }));
            return;
          }

          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({
            provider: decision.provider,
            model: decision.model,
            rationale: decision.rationale,
            score: decision.overallScore,
          }));
        } catch (err) {
          logger.error(`Fallback calculation failed: ${err}`);
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Fallback calculation failed", message: err instanceof Error ? err.message : String(err) }));
        }
        return;
      }

      // GET /v1/dashboard — aggregated routing stats and trends
      if (url.startsWith("/v1/dashboard") && req.method === "GET") {
        const range = urlQuery.get("range") ?? "24h";
        try {
          const data = this.db.getDashboardData(range);
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify(data, null, 2));
        } catch (err) {
          logger.error(`Dashboard query failed: ${err}`);
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Dashboard query failed", message: err instanceof Error ? err.message : String(err) }));
        }
        return;
      }

      if (url.startsWith("/last-decision") && req.method === "GET") {
        try {
          const rawDecisions = this.db.getRecentDecisions(parseInt(urlQuery.get("limit") ?? "5", 10) || 5);
          const decisions = rawDecisions.map((d: any) => {
            const out: any = { ...d };
            if (out.candidates_json) {
              try { out.candidates = JSON.parse(out.candidates_json); } catch { out.candidates = out.candidates_json; }
              delete out.candidates_json;
            }
            if (out.context_filter_json) {
              try { out.contextFilter = JSON.parse(out.context_filter_json); } catch { out.contextFilter = out.context_filter_json; }
              delete out.context_filter_json;
            }
            if (out.modality_filter_json) {
              try { out.modalityFilter = JSON.parse(out.modality_filter_json); } catch { out.modalityFilter = out.modality_filter_json; }
              delete out.modality_filter_json;
            }
            if (out.routing_scores) {
              try { out.winner_scores = JSON.parse(out.routing_scores); } catch { out.winner_scores = out.routing_scores; }
              delete out.routing_scores;
            }
            return out;
          });
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify(decisions, null, 2));
        } catch (err) {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Failed to fetch decisions" }));
        }
        return;
      }

      if (url.startsWith("/reset-circuit") && req.method === "POST") {
        const provider = urlQuery.get("provider");
        if (provider) {
          this.costTracker.resetCircuit(provider);
          logger.info(`Circuit breaker reset for ${provider}`);
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ status: "ok", provider, message: "circuit reset" }));
        } else {
          this.costTracker.resetAllCircuits();
          logger.info("All circuit breakers reset");
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ status: "ok", message: "all circuits reset" }));
        }
        return;
      }

      // Admin bench sync — the bench sidecar's ONLY write path into router
      // state (CONTRACT.md §4). BENCH_SYNC_TOKEN gate (separate secret from
      // ROUTER_ADMIN_TOKEN; >=32 bytes, timing-safe compare). Whole-payload
      // validation, single-transaction upsert, per-row generated_at >=
      // watermark with equal = idempotent no-op, strictly older = reject.
      if (url === "/admin/bench/sync" && req.method === "POST") {
        const syncToken = process.env.BENCH_SYNC_TOKEN;
        if (!syncToken) {
          logger.warn("Bench sync denied: BENCH_SYNC_TOKEN not configured");
          res.writeHead(503, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: { code: "sync_disabled", message: "BENCH_SYNC_TOKEN not configured" } }));
          return;
        }
        const auth = req.headers["authorization"];
        const provided = typeof auth === "string" && auth.startsWith("Bearer ") ? auth.slice(7) : undefined;
        if (!provided || !timingSafeEqualStr(provided, syncToken)) {
          logger.warn(`Bench sync denied: bad or missing token (from ${req.socket.remoteAddress ?? "?"})`);
          res.writeHead(401, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: { code: "unauthorized", message: "invalid credentials" } }));
          return;
        }
        // Body size cap BEFORE parse (2 MB)
        const contentLength = parseInt(String(req.headers["content-length"] ?? "0"), 10);
        if (contentLength > 2_000_000) {
          res.writeHead(413, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: { code: "payload_too_large", message: "body exceeds cap" } }));
          return;
        }
        let raw = "";
        req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
        req.on("end", () => {
          try {
            const parsed = JSON.parse(raw);
            if (parsed?.schema_version !== 1) {
              res.writeHead(400, { "Content-Type": "application/json" });
              res.end(JSON.stringify({ error: { code: "unsupported_schema", message: "unknown schema_version" } }));
              return;
            }
            const verdicts = Array.isArray(parsed?.verdicts) ? parsed.verdicts : null;
            if (!verdicts || verdicts.length === 0 || verdicts.length > 500) {
              res.writeHead(400, { "Content-Type": "application/json" });
              res.end(JSON.stringify({ error: { code: "bad_row_count", message: "verdicts must be 1..500 rows" } }));
              return;
            }
            // Field validation: one bad row rejects the whole payload
            const MODEL_ID = /^[a-z0-9._:/|-]+$/i;
            const TS = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/;
            for (const v of verdicts) {
              const bad =
                !v ||
                typeof v.model_a !== "string" || !MODEL_ID.test(v.model_a) || v.model_a.length > 128 ||
                typeof v.model_b !== "string" || !MODEL_ID.test(v.model_b) || v.model_b.length > 128 ||
                !["coding", "reasoning", "conversation"].includes(v.intent) ||
                typeof v.prompt_generation !== "string" || v.prompt_generation.length > 64 ||
                !Number.isInteger(v.round) || v.round < 0 || v.round > 100 ||
                (v.swap_order !== 0 && v.swap_order !== 1) ||
                !["a", "b", "tie"].includes(v.verdict) ||
                (v.judge_provider !== null && (typeof v.judge_provider !== "string" || v.judge_provider.length > 64)) ||
                (v.judge_model !== null && (typeof v.judge_model !== "string" || v.judge_model.length > 128)) ||
                typeof v.timestamp !== "string" || !TS.test(v.timestamp) ||
                typeof v.generated_at !== "string" || !TS.test(v.generated_at);
              if (bad) {
                logger.warn(`Bench sync rejected: invalid row (schema violation)`);
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ error: { code: "invalid_row", message: "one or more rows failed validation" } }));
                return;
              }
            }
            // Per-row staleness: all rows must be >= persisted watermark
            const watermark = this.db.getBenchSyncWatermark();
            if (watermark) {
              const stale = verdicts.filter((v: any) => v.generated_at < watermark);
              if (stale.length === verdicts.length) {
                // entire payload strictly older — reject as stale
                logger.warn(`Bench sync rejected: payload older than watermark`);
                res.writeHead(409, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ error: { code: "stale_payload", message: "generated_at below high-water mark" } }));
                return;
              }
            }
            const result = this.db.applyBenchSync({ verdicts });
            // advance watermark to max generated_at in payload
            const maxGen = verdicts.reduce((m: string, v: any) => (v.generated_at > m ? v.generated_at : m), watermark ?? "");
            if (maxGen) this.db.setBenchSyncWatermark(maxGen);
            logger.info(`Bench sync ok: ${result.applied} rows applied, watermark -> ${maxGen}`);
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ status: "ok", applied: result.applied }));
          } catch (err) {
            logger.error(`Bench sync failed: ${err instanceof Error ? err.message : String(err)}`);
            res.writeHead(500, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: { code: "internal", message: "sync failed" } }));
          }
        });
        return;
      }

      // Admin restart — Layer 1 of the no-elevation restart bridge (2026-09-06).
      // Gated by ROUTER_ADMIN_TOKEN (Bearer or x-admin-token header). Clean exit;
      // NSSM's default restart-on-exit respawns the process (~1.5s), loading any
      // freshly compiled dist/. In-flight streams are cut — restart during idle.
      if (url === "/admin/restart" && req.method === "POST") {
        const token = process.env.ROUTER_ADMIN_TOKEN;
        if (!token) {
          res.writeHead(503, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "admin restart disabled", message: "ROUTER_ADMIN_TOKEN not configured" }));
          return;
        }
        const auth = req.headers["authorization"];
        const provided =
          (typeof auth === "string" && auth.startsWith("Bearer ") ? auth.slice(7) : undefined) ??
          (req.headers["x-admin-token"] as string | undefined);
        if (!provided || !timingSafeEqualStr(provided, token)) {
          logger.warn("Admin restart denied: bad or missing token");
          res.writeHead(403, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "forbidden" }));
          return;
        }
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ status: "restarting", message: "clean exit; service manager will restart" }));
        logger.info("ADMIN RESTART requested — exiting cleanly for service restart");
        setTimeout(() => process.exit(0), 500);
        return;
      }

      // Admin discovery - forced model-discovery refresh in ANY mode.
      // Same token gate as /admin/restart. Returns the discovery summary.
      if (url === "/admin/discover" && req.method === "POST") {
        const token = process.env.ROUTER_ADMIN_TOKEN;
        if (!token) {
          res.writeHead(503, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "admin discovery disabled", message: "ROUTER_ADMIN_TOKEN not configured" }));
          return;
        }
        const auth = req.headers["authorization"];
        const provided =
          (typeof auth === "string" && auth.startsWith("Bearer ") ? auth.slice(7) : undefined) ??
          (req.headers["x-admin-token"] as string | undefined);
        if (!provided || !timingSafeEqualStr(provided, token)) {
          logger.warn("Admin discovery denied: bad or missing token");
          res.writeHead(403, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "forbidden" }));
          return;
        }
        try {
          const summary = await this.discovery.run("admin");
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify(summary, null, 2));
        } catch (err) {
          logger.error(`Discovery failed: ${err}`);
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "discovery failed", message: err instanceof Error ? err.message : String(err) }));
        }
        return;
      }

      if (url === "/v1/benchmark/embeddings" && req.method === "POST") {
        const benchmark = new EmbeddingBenchmark(this.db);
        try {
          const summary = await benchmark.runFullBenchmark();
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify(summary, null, 2));
        } catch (err) {
          logger.error(`Benchmark failed: ${err}`);
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Benchmark failed", message: err instanceof Error ? err.message : String(err) }));
        }
        return;
      }

      if (url === "/v1/chat/completions" && req.method === "POST") {
        const body = await this.readBody(req);
        const request = JSON.parse(body) as ChatCompletionRequest;
        // Trusted loopback callers (bench sidecar) may lift BOTH the total
        // request deadline and the per-attempt upstream fetch timeout via
        // x-router-timeout-ms (CONTRACT.md §2). The AsyncLocalStorage scope
        // propagates to adapters without signature changes.
        const overrideMs = requestTimeoutOverrideMs(req);
        const chat = () => this.handleChat(request, res, this.extractSessionKey(req, request), req);
        if (overrideMs === null) await chat();
        else await withUpstreamTimeoutOverride(overrideMs, chat);
        return;
      }

      if (url === "/v1/report/abort" && req.method === "POST") {
        try {
          const body = await this.readBody(req);
          const data = JSON.parse(body);

          if (!data || typeof data.provider !== "string" || !data.provider.trim()) {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: "provider must be a non-empty string" }));
            return;
          }
          if (typeof data.model !== "string" || !data.model.trim()) {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: "model must be a non-empty string" }));
            return;
          }

          if (data.turnsCompleted !== undefined && (typeof data.turnsCompleted !== "number" || !Number.isInteger(data.turnsCompleted) || data.turnsCompleted < 0)) {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: "turnsCompleted must be a non-negative integer" }));
            return;
          }

          if (data.durationMs !== undefined && (typeof data.durationMs !== "number" || !Number.isInteger(data.durationMs) || data.durationMs < 0)) {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: "durationMs must be a non-negative integer" }));
            return;
          }

          let dbWriteSuccess = true;
          try {
            this.db.recordAbortEvent({
              provider: data.provider,
              model: data.model,
              turnsCompleted: data.turnsCompleted,
              durationMs: data.durationMs,
              sessionKey: data.sessionKey,
            });
          } catch (dbErr) {
            logger.warn(`Failed to record abort event in database: ${dbErr}`);
            dbWriteSuccess = false;
          }

          // Apply in-memory penalty
          this.costTracker.applyAbortPenalty(data.provider, data.model);

          if (!dbWriteSuccess) {
            res.writeHead(500, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: "Database write failure" }));
            return;
          }

          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ status: "success" }));
        } catch (err) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Invalid JSON or request processing error", message: err instanceof Error ? err.message : String(err) }));
        }
        return;
      }

      // GET /v1/embeddings/models — list all available embedding models
      if (url === "/v1/embeddings/models" && req.method === "GET") {
        try {
          const benchmark = new EmbeddingBenchmark(this.db);
          const models = await benchmark.listEmbeddingModels();
          // Also include remote embedding providers
          const remoteModels = this.listRemoteEmbeddingModels();
          const allModels = [...models, ...remoteModels];
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({
            object: "list",
            data: allModels.map(m => ({
              id: m.name,
              provider: m.provider,
              object: "embedding_model",
              context_window: m.contextWindow,
              is_local: m.isLocal,
              dimensions: m.dimensions,
            })),
          }, null, 2));
        } catch (err) {
          logger.error(`Failed to list embedding models: ${err}`);
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Failed to list embedding models" }));
        }
        return;
      }

      // GET /v1/embeddings/benchmarks — return persisted benchmark results
      if (url === "/v1/embeddings/benchmarks" && req.method === "GET") {
        try {
          const benchmark = new EmbeddingBenchmark(this.db);
          const runs = benchmark.getLatestResults(parseInt(urlQuery.get("limit") ?? "5", 10) || 5);
          const modelScores = benchmark.getLatestModelScores();
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({
            runs,
            latest_model_scores: modelScores,
          }, null, 2));
        } catch (err) {
          logger.error(`Failed to fetch benchmark results: ${err}`);
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Failed to fetch benchmark results" }));
        }
        return;
      }

      // POST /v1/embeddings — pure proxy, caller chooses model
      if (url === "/v1/embeddings" && req.method === "POST") {
        const body = await this.readBody(req);
        const request = JSON.parse(body) as { model?: string; input: string | string[] };
        await this.handleEmbeddings(request, res);
        return;
      }

      if (url === "/v1/curate" && req.method === "POST") {
        logger.info("Manual curator trigger received.");
        try {
          const result = await this.curator.run();
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify(result, null, 2));
        } catch (err) {
          logger.error(`Curator manual trigger failed: ${err}`);
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({
            error: "Curator failed",
            message: err instanceof Error ? err.message : String(err),
          }));
        }
        return;
      }

      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Not found", path: url }));
    } catch (err) {
      logger.error(`Request error: ${err}`);
      const isAllDead = err instanceof AllProvidersDeadError;
      const status = isAllDead ? 503 : 500;
      if (!res.headersSent) {
        res.writeHead(status, { "Content-Type": "application/json" });
        res.end(JSON.stringify({
          error: {
            message: err instanceof Error ? err.message : String(err),
            type: isAllDead ? "all_providers_exhausted" : "internal_error",
          },
        }));
      }
    }
  }

  // ─── Unified Chat Handler (streaming + non-streaming) ───

  private async handleChat(
    request: ChatCompletionRequest,
    res: http.ServerResponse,
    sessionKey: string,
    req?: http.IncomingMessage,
  ): Promise<void> {
    if (!this.initialized) throw new Error("Proxy not initialized");
    const isStreaming = request.stream === true;

    // For streaming, SSE response headers are DEFERRED until the first write.
    // Buffered streaming flushes nothing to the client before the checkpoint
    // anyway, so this is not observable — and it lets the
    // x-model-router-selected-model header name the model that actually served
    // the request. OpenClaw still never sees an empty assistant delta: the
    // first flushed chunk always carries real content. (This also fixes a
    // latent ERR_HTTP_HEADERS_SENT on streaming MODALITY_UNSUPPORTED rejects.)

    // Classify intent
    const lastMessage = request.messages[request.messages.length - 1];
    const prompt = lastMessage?.content ?? "";

    let classification;
    try {
      classification = this.classifier.isInitialized()
        ? await this.classifier.classify(prompt, this.embedFn)
        : this.classifier.classifyByKeyword(prompt);
      logger.debug(`Intent: ${classification.intent} (${classification.confidence.toFixed(2)})`);
    } catch {
      classification = { intent: "conversation", confidence: 0.5 };
    }

    // Ensure confidence is always a valid number (fallback for DB constraint)
    if (typeof classification.confidence !== 'number' || isNaN(classification.confidence)) {
      classification.confidence = 0.5;
    }

    // Estimate prompt token count for routing + context window guard
    const estimatedTokens = estimateTokenCount(request);

    // Hardening 2026-09-03: requests that exceed EVERY provider's effective
    // input limit get an immediate 413 instead of falling through the
    // pipeline (previously: skip-cascade with hang/crash potential).
    const maxAnyProviderLimit = Math.max(
      ...Object.values(PROVIDER_EFFECTIVE_INPUT_LIMITS),
    );
    if (estimatedTokens > maxAnyProviderLimit) {
      const message =
        `request ~${estimatedTokens} tokens exceeds the largest provider input limit ` +
        `(${maxAnyProviderLimit}); trim the conversation or reset the session`;
      logger.warn(message);
      if (!res.headersSent) {
        res.writeHead(413, { "Content-Type": "application/json" });
        res.end(JSON.stringify({
          error: {
            message,
            type: "request_too_large",
            estimatedTokens,
            maxProviderLimit: maxAnyProviderLimit,
          },
        }));
      }
      return;
    }

    // ─── Anomaly detection: flag token spikes (>2x rolling average) ───
    const anomaly = this.budgetTracker.detectAnomaly(estimatedTokens);
    if (anomaly.isAnomalous) {
      logger.warn(`Token anomaly detected: ${anomaly.reason}`);
    }

    // ─── Modality detection: scan for images, audio, and other non-text content ───
    const modalityResult = detectModalities(request.messages ?? []);
    if (modalityResult.isMultimodal) {
      logger.info(
        `Modality detected: ${modalityResult.summary} — ` +
        `routing to ${modalityResult.modalities.filter((m: Modality) => m !== "text").join(", ")}-capable models only.`,
      );
    }

    // Build candidate list - pass estimated tokens and required modalities for routing
    // Tier aliases (Daz, 2026-09-07): ":fast" = speed tier; ":lite" =
    // subscription-covered + local only (glm-5-turbo/4.7-class) - light work
    // (crons, mechanical subagents) must not spend premium window or meters.
    const modelAlias = typeof request.model === "string" ? request.model.toLowerCase() : "";
    const requestTier =
      modelAlias.endsWith(":fast") ? "fast"
      : modelAlias.endsWith(":lite") ? "lite"
      : undefined;
    const reqEffort = extractThinkingLevel(request);
    const reqSpeedMode = extractSpeedMode(request);
    // Outbound effort is the router's purchase decision (effort = tokens =
    // cost vs depth), not a pass-through. The client lever stays a routing
    // weight and an optional clamp inside the policy. Adapters translate the
    // chosen level to per-provider formats (zai budget_tokens, Gemini
    // thinkingBudget, OR reasoning_effort); ollama strips it.
    const effortDecision = effortPolicyMode() === "auto"
      ? decideOutboundEffort({
          intent: classification.intent,
          clientHint: reqEffort,
          speedMode: reqSpeedMode,
          quotaPressure: getZaiQuotaProbe().pressure(),
          budgetExceeded: this.costTracker.isBudgetExceeded("zai"),
        })
      : null;
    if (effortDecision) {
      logger.debug(`Effort policy: ${effortDecision.level} [${effortDecision.trace.join(" ")}]`);
    }
    const decision = await this.router.decide(classification, sessionKey, {
      estimatedTokens,
      requiredModalities: modalityResult.modalities,
      effortLevel: reqEffort,
      speedMode: reqSpeedMode,
      modelTier: requestTier,
      requestModelString: modelAlias,
    }, req?.headers?.["x-routing-profile"] as string | undefined);
    const builtCandidates = this.buildCandidateList(decision, request, req);
    const candidates = builtCandidates.list;
    const lastResortCandidate = builtCandidates.lastResortAppended
      ? candidates[candidates.length - 1]
      : null;
    const requestId = this.extractRequestId(request);

    if (decision) {
      this.db.recordDecision({
        timestamp: new Date().toISOString(),
        sessionKey,
        messageHash: this.hashPrompt(prompt),
        intent: classification.intent,
        confidence: classification.confidence ?? 0.5, // Default if missing
        provider: decision.provider,
        model: decision.model,
        scores: {
          ...decision.scores,
          levers: {
            effort: reqEffort,
            speed: reqSpeedMode,
            tier: requestTier ?? null,
            effortOut: effortDecision ? { level: effortDecision.level, why: effortDecision.trace.join(" ") } : "passthrough",
          },
        },
        overallScore: decision.overallScore,
        outcome: decision.error ? decision.error.code : "PENDING",
        requestId,
        candidatesJson: decision.candidates ? JSON.stringify(decision.candidates) : null,
        contextFilterJson: decision.contextFilter ? JSON.stringify(decision.contextFilter) : null,
        modalityFilterJson: decision.modalityFilter ? JSON.stringify(decision.modalityFilter) : null,
      });
    }

    // Context window guard: if no model can handle the request, return error immediately
    if (decision?.error) {
      const errorCode = decision.error.code;
      const isModalityError = errorCode === "MODALITY_UNSUPPORTED";
      const httpStatus = isModalityError ? 422 : 413;
      const errorType = isModalityError ? "modality_unsupported" : "context_too_large";

      logger.error(`${errorCode} — rejecting request: ${decision.error.message}`);
      res.writeHead(httpStatus, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        error: {
          message: decision.error.message,
          type: errorType,
          code: decision.error.code,
          estimated_tokens: decision.error.estimatedTokens,
          max_context_window: decision.error.maxContextWindow,
          required_modalities: decision.error.requiredModalities,
          available_modality_models: decision.error.availableModalityModels,
        },
      }));
      return;
    }

    logger.info(
      `Chat request - session=${sessionKey} intent=${classification.intent} stream=${isStreaming} - ` +
      `${candidates.length} candidates across ${new Set(candidates.map(c => c.provider)).size} providers` +
      `${requestUsesTools(request) ? " tools=yes" : ""}`,
    );

    const maxProviderAttempts = maxAttemptsPerProvider();
    const maxAttemptsForProvider = (provider: string): number => {
      if (provider === "zai") {
        return candidates.filter((candidate) => candidate.provider === provider).length;
      }
      return maxProviderAttempts;
    };

    if (estimatedTokens > MIN_USEFUL_CONTEXT_TOKENS) {
      logger.info(`Large request: ~${estimatedTokens} input tokens - context guard active`);
    }

    // Retry loop. Keep the default conservative so CogRouter can fail over
    // before OpenClaw's outer LLM timeout fires.
    let lastError: Error | null = null;
    const modelStrikes = new Map<string, number>();
    const effectiveRequestTimeoutMs =
      requestTimeoutOverrideMs(req) ?? routerRequestTimeoutMs();
    const requestDeadlineMs = Date.now() + effectiveRequestTimeoutMs;
    let hedgeAttempted = false;

    // ─── Decision-source observability (Switchyard §4) ───
    // Runtime facts that determine how the serving model was chosen. The
    // final source is classified exactly once per request, at its terminal
    // outcome, and counted in /stats (stats.decisionSources).
    const usesTools = requestUsesTools(request);

    // Exploration probe (judge-loop cold start, 2026-09-06):
    // probeRate of eligible requests promote an under-sampled candidate ahead
    // of the scored winner, so the judge can start evolving capability cells
    // that would otherwise never serve traffic. Graceful by construction: the
    // scored winner remains the next candidate in the retry chain.
    let exploredProbe: { provider: string; model: string } | null = null;
    const probeRate = this.config.probeRate ?? 0.05;
    if (
      decision && !decision.error && probeRate > 0 &&
      !usesTools &&
      !requestTier && // fast-tier requests keep their tier promise
      estimatedTokens <= PROBE_MAX_INPUT_TOKENS &&
      candidates.length >= 2 &&
      Math.random() < probeRate
    ) {
      // Probe targets: any candidate except the winner and the appended last resort.
      const nonResort = lastResortCandidate
        ? candidates.slice(0, candidates.length - 1)
        : candidates;
      const cold = nonResort.slice(1).filter((c) =>
        this.modelRegistry.isUnderSampled(c.provider, c.model, classification.intent, PROBE_MIN_SAMPLES));
      if (cold.length > 0) {
        const pick = cold[Math.floor(Math.random() * cold.length)];
        exploredProbe = pick;
        candidates.splice(0, candidates.length, pick, ...candidates.filter((c) => c !== pick));
        logger.info(
          `Exploration probe: ${pick.provider}/${pick.model} promoted over winner ` +
            `${decision.provider}/${decision.model} [${classification.intent}] ` +
            `(p=${probeRate}, ~${estimatedTokens} tok, ${cold.length} cold candidate(s))`,
        );
      }
    }

    let circuitOrBackoffSkipped = false;
    let contextGuardSkipped = false;
    let attemptFailed = false;
    const recordSource = (
      served: boolean,
      servedCandidate?: { provider: string; model: string } | null,
    ): void => {
      const source = computeProxyDecisionSource({
        served,
        routerSource: decision?.decisionSource ?? null,
        usesTools,
        contextGuardSkipped,
        circuitOrBackoffSkipped,
        attemptFailed,
        servedByAppendedLastResort: Boolean(
          servedCandidate &&
          lastResortCandidate &&
          servedCandidate.provider === lastResortCandidate.provider &&
          servedCandidate.model === lastResortCandidate.model,
        ),
        exploredServed: Boolean(
          exploredProbe &&
          servedCandidate &&
          servedCandidate.provider === exploredProbe.provider &&
          servedCandidate.model === exploredProbe.model,
        ),
      });
      decisionSourceCounters.increment(source);
      logger.debug(
        `decision_source=${source} served_by=${servedCandidate ? `${servedCandidate.provider}/${servedCandidate.model}` : "none"}`,
      );
    };

    for (let ci = 0; ci < candidates.length; ci++) {
      const candidate = candidates[ci];
      if (Date.now() >= requestDeadlineMs) {
        lastError = new Error(`router_request_timeout: exceeded ${effectiveRequestTimeoutMs}ms before trying ${candidate.provider}/${candidate.model}`);
        logger.warn(lastError.message);
        break;
      }

      const providerAttemptLimit = maxAttemptsForProvider(candidate.provider);
      const candidateKey = `${candidate.provider}/${candidate.model}`;
      const strikes = modelStrikes.get(candidateKey) ?? 0;
      if (strikes >= providerAttemptLimit) {
        logger.debug(`Skipping ${candidateKey} - ${strikes} strikes`);
        continue;
      }

      const adapter = getProvider(candidate.provider);
      if (!adapter) continue;

      // Check circuit breaker
      if (!this.costTracker.isAvailable(candidate.provider)) {
        logger.debug(`Skipping ${candidate.provider} - circuit open`);
        circuitOrBackoffSkipped = true;
        modelStrikes.set(candidateKey, providerAttemptLimit);
        continue;
      }

      // Check provider backoff timer (failure-type-aware)
      if (providerBackoff.isBackedOff(candidate.provider)) {
        const reason = providerBackoff.backoffReason(candidate.provider);
        const remaining = Math.round(providerBackoff.remainingMs(candidate.provider) / 1000);
        logger.debug(`Skipping ${candidate.provider} - backoff active (${remaining}s remaining: ${reason})`);
        circuitOrBackoffSkipped = true;
        modelStrikes.set(candidateKey, providerAttemptLimit);
        continue;
      }

      if (!this.costTracker.isAvailable(candidate.provider, candidate.model)) {
        logger.debug(`Skipping ${candidateKey} - model circuit open`);
        circuitOrBackoffSkipped = true;
        modelStrikes.set(candidateKey, strikes + 1);
        continue;
      }

      // Context window guard: skip providers whose effective input limit
      // is too small for this request. This saves a wasted API call + timeout.
      const candidateWindow = this.modelRegistry
        .getAvailableModels([candidate.provider])
        .find((m) => m.model === candidate.model)?.contextWindow;
      const effectiveLimit = getEffectiveInputLimit(candidate.provider, candidate.model, candidateWindow);
      if (estimatedTokens > effectiveLimit) {
        logger.info(
          `Skipping ${candidate.provider}/${candidate.model} - request ~${estimatedTokens} tokens exceeds limit ${effectiveLimit}`,
        );
        contextGuardSkipped = true;
        modelStrikes.set(candidate.provider, strikes + 1);
        continue;
      }

      const apiKey = process.env[candidate.provider.toUpperCase() + "_API_KEY"] ?? "";
      const startTime = Date.now();

      try {
        logger.info(`Trying ${candidate.provider}/${candidate.model}...`);

        if (isStreaming) {
          // ── Buffered streaming ──
          // Tokens are buffered internally and only flushed to the client when
          // a checkpoint is reached (finish_reason received). If the upstream
          // dies before that, the buffer is discarded and we silently retry
          // on the next candidate — the client never sees a truncated response.
          const providerRequest = {
            ...request,
            model: candidate.model,
            stream: true,
            ...(effortDecision ? { thinking: effortDecision.level } : {}),
          };
          let accumulatedContent = "";
          let hasPayload = false;
          const bufferedChunks: any[] = [];
          let reachedCheckpoint = false;
          let firstChunkAt: number | undefined;
          let firstContentAt: number | undefined;
          let upstreamModel: string | undefined;

          try {
            const stallTimeout = streamStallTimeoutMs();
            let lastChunkTime = Date.now();

            for await (const chunk of adapter.chatCompletionStream(candidate.model, providerRequest, apiKey)) {
              const now = Date.now();
              if (firstChunkAt === undefined) firstChunkAt = now;
              if (now - lastChunkTime > stallTimeout) {
                this.costTracker.applyAbortPenalty(candidate.provider, candidate.model, "stall");
                throw new Error(`stream_stall: no data for ${stallTimeout}ms from ${candidate.provider}/${candidate.model}`);
              }
              lastChunkTime = now;

              // Capture the provider's own reported model before scrubbing (bench seam: provider truth)
              if (upstreamModel === undefined && typeof chunk.model === "string" && chunk.model) {
                upstreamModel = chunk.model;
              }

              // Scrub provider model name
              chunk.model = CHAT_RESPONSE_MODEL;

              // Track content for judge evaluation + empty-response detection
              const delta = chunk.choices?.[0]?.delta;
              if (delta?.content) {
                if (firstContentAt === undefined) firstContentAt = now;
                accumulatedContent += delta.content;
                hasPayload = true;
              }
              if (delta?.tool_calls?.length) {
                hasPayload = true;
              }

              // Buffer the chunk — do NOT forward yet
              bufferedChunks.push(chunk);

              // Check for finish_reason — this is our checkpoint
              const finishReason = chunk.choices?.[0]?.finish_reason;
              if (finishReason) {
                reachedCheckpoint = true;
                break;
              }
            }
          } catch (streamErr) {
            // Any stream error before checkpoint → discard buffer, retry
            // The outer catch handles strikes, hedging, and next-candidate logic
            logger.debug(`Buffered stream failed on ${candidate.provider}/${candidate.model} (${streamErr instanceof Error ? streamErr.message.substring(0, 80) : streamErr}) — discarding ${bufferedChunks.length} buffered chunks for retry`);
            throw streamErr;
          }

          if (!reachedCheckpoint) {
            // Stream ended without finish_reason — treat as incomplete, retry
            const err = new Error(`incomplete_stream: ${candidate.provider}/${candidate.model} ended without finish_reason`);
            (err as any).code = "empty_response";
            throw err;
          }

          if (!hasPayload) {
            const err = new Error(`empty_provider_response: stream from ${candidate.provider}/${candidate.model} produced no content`);
            (err as any).code = "empty_response";
            throw err;
          }

          // ── Checkpoint reached: flush all buffered chunks to client ──
          const durationMs = Date.now() - startTime;
          // Speed telemetry: TTFT + throughput sample (content-bearing streams only)
          const ttftMs = (firstContentAt ?? firstChunkAt ?? startTime) - startTime;
          if (accumulatedContent.length > 0) {
            const genMs = Math.max(1, durationMs - ttftMs);
            const estTokensOut = Math.max(1, Math.round(accumulatedContent.length / 4));
            const tokensPerSec = estTokensOut / (genMs / 1000);
            this.costTracker.recordSpeedSample(candidate.provider, candidate.model, ttftMs, tokensPerSec, reqEffort);
          }
          recordSource(true, candidate);
          // Open SSE headers now (deferred from request start) so the
          // selected-model header names the model that actually served.
          this.openSSE(res, candidate.model, upstreamModel);
          for (const bufferedChunk of bufferedChunks) {
            this.writeSSE(res, bufferedChunk);
          }
          res.write("data: [DONE]\n\n");
          res.end();

          await this.costTracker.recordCall(candidate.provider, { durationMs, outcome: "success" }, candidate.model, estimatedTokens);
          this.costTracker.recordSizeLatency(candidate.provider, estimatedTokens, durationMs);
          this.db.recordCallOutcome({
            provider: candidate.provider, model: candidate.model,
            durationMs, outcome: "success", timestamp: new Date().toISOString(),
          });
          this.db.updateDecisionOutcome(requestId, "success", durationMs);

          logger.info(`✅ ${candidate.provider}/${candidate.model} streamed (buffered, ${bufferedChunks.length} chunks) in ${durationMs}ms`);

          // Clear any provider backoff on success
          providerBackoff.clear(candidate.provider);

          // ─── Async LLM-as-judge feedback ───
          if (
            // Probed requests are always judged - the whole point of the probe
            // is the observation; the sample roll only governs normal traffic.
            this.judge.shouldJudge(Boolean(
              exploredProbe &&
              candidate.provider === exploredProbe.provider &&
              candidate.model === exploredProbe.model,
            )) &&
            !this.judge.isSameModel(candidate.provider, candidate.model) &&
            accumulatedContent // Skip empty/tool-only responses
          ) {
            const evalProvider = candidate.provider;
            const evalModel = candidate.model;
            const evalIntent = classification.intent;
            const evalPrompt = prompt;

            // Fire and forget - don't await, don't block
            // Phase-1 hardened path: attribution gate → quarantine → guardrails →
            // shadow-aware apply (§4.1). Confidence + truncation feed the gate.
            this.judge.evaluate(evalPrompt, accumulatedContent, evalIntent)
              .then((result) => {
                if (result) {
                  this.modelRegistry.applyJudgedScore(
                    evalProvider, evalModel, evalIntent, result.score,
                    {
                      rawScore: result.rawScore,
                      judgeNote: result.note,
                      judgeModelId: `${this.judge.judgeModelId}`,
                      confidence: classification.confidence,
                      truncated: result.truncated,
                      effortLevel: reqEffort ?? null,
                    },
                  );
                }
              })
              .catch((err) => {
                logger.debug(`Judge async error (non-fatal): ${err instanceof Error ? err.message : err}`);
              });
          }

          return;
        }

        // ── Non-streaming path ──
        const providerRequest = {
          ...request,
          model: candidate.model,
          stream: false,
          ...(effortDecision ? { thinking: effortDecision.level } : {}),
        };
        const response = await this.withRequestDeadline(
          adapter.chatCompletion(candidate.model, providerRequest, apiKey),
          requestDeadlineMs,
          `${candidate.provider}/${candidate.model}`,
        );
        const durationMs = Date.now() - startTime;
        const choice = response.choices?.[0] as any;
        const message = choice?.message ?? {};
        const content = message.content ?? "";
        const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : undefined;
        const finishReason = choice?.finish_reason ?? (toolCalls ? "tool_calls" : "stop");
        const hasVisiblePayload = Boolean(content) || Boolean(toolCalls?.length);

        if (!hasVisiblePayload) {
          const err = new Error(`empty_provider_response: finish_reason=${finishReason ?? "unknown"}`);
          (err as any).code = "empty_response";
          throw err;
        }

        await this.costTracker.recordCall(candidate.provider, { durationMs, outcome: "success" }, candidate.model, estimatedTokens);
        this.costTracker.recordSizeLatency(candidate.provider, estimatedTokens, durationMs);
        this.db.recordCallOutcome({
          provider: candidate.provider, model: candidate.model,
          durationMs, outcome: "success", timestamp: new Date().toISOString(),
        });
        this.db.updateDecisionOutcome(requestId, "success", durationMs);

        // Track token usage for subscription providers (e.g., Z.AI quota)
        if (response.usage && candidate.provider === "zai") {
          this.costTracker.recordTokenUsage(
            candidate.provider,
            response.usage.prompt_tokens || 0,
            response.usage.completion_tokens || 0,
          );
        }

        // Track dollar spend for paid models
        const requestCost = calculateRequestCost(candidate.provider, candidate.model, this.modelRegistry, response.usage);
        if (requestCost > 0) {
          this.costTracker.recordSpend(candidate.provider, requestCost);
          logger.debug(`Spend: ${candidate.provider} +$${requestCost.toFixed(4)} (daily=$${this.costTracker.getDailySpend(candidate.provider).toFixed(2)}, monthly=$${this.costTracker.getMonthlySpend(candidate.provider).toFixed(2)})`);
        }

        logger.info(`✅ ${candidate.provider}/${candidate.model} succeeded in ${durationMs}ms`);

        // Clear any provider backoff on success
        providerBackoff.clear(candidate.provider);

        const upstreamModel = typeof response.model === "string" ? response.model : undefined;
        response.model = CHAT_RESPONSE_MODEL;
        recordSource(true, candidate);
        res.writeHead(200, {
          "Content-Type": "application/json",
          [SELECTED_MODEL_HEADER]: candidate.model,
          ...(upstreamModel ? { [UPSTREAM_MODEL_HEADER]: upstreamModel } : {}),
        });
        res.end(JSON.stringify(response));

        // ─── Async LLM-as-judge feedback ───
        // After the response is sent, maybe evaluate quality and evolve capability scores.
        // This runs in the background and never affects the response path.
        if (
          this.judge.shouldJudge() &&
          !this.judge.isSameModel(candidate.provider, candidate.model) &&
          !toolCalls // Skip tool-call responses - judge evaluates text quality only
        ) {
          const evalProvider = candidate.provider;
          const evalModel = candidate.model;
          const evalIntent = classification.intent;
          const evalPrompt = prompt;
          const evalResponse = content;

          // Fire and forget - don't await, don't block
          // Phase-1 hardened path: attribution gate → quarantine → guardrails →
          // shadow-aware apply (§4.1). Confidence + truncation feed the gate.
          this.judge.evaluate(evalPrompt, evalResponse, evalIntent)
            .then((result) => {
              if (result) {
                this.modelRegistry.applyJudgedScore(
                  evalProvider, evalModel, evalIntent, result.score,
                  {
                    rawScore: result.rawScore,
                    judgeNote: result.note,
                    judgeModelId: `${this.judge.judgeModelId}`,
                    confidence: classification.confidence,
                    truncated: result.truncated,
                    effortLevel: reqEffort ?? null,
                  },
                );
              }
            })
            .catch((err) => {
              logger.debug(`Judge async error (non-fatal): ${err instanceof Error ? err.message : err}`);
            });
        }

        return;
      } catch (err) {
        const durationMs = Date.now() - startTime;
        const error = err as Error;

        // ── Classify failure type for failure-aware fallback ──
        const failure = classifyFailure(error);
        const isRateLimit = failure.type === "rate_limit";
        const isQuota = (error as any).code === "quota_exceeded" || /quota|monthly limit|prompt tokens limit exceeded|402/i.test(error.message);
        const isTimeout = failure.type === "timeout";
        const isEmpty = failure.type === "empty_response";
        const outcome = isRateLimit || isQuota ? "rate_limit" : isTimeout ? "timeout" : isEmpty ? "empty" : "error";

        lastError = error;
        attemptFailed = true;
        const newStrikes = strikes + 1;
        modelStrikes.set(candidateKey, newStrikes);

        logger.warn(
          `❌ ${candidate.provider}/${candidate.model} failed (${outcome}/${failure.type}) in ${durationMs}ms ` +
          `[strike ${newStrikes}/${providerAttemptLimit}]: ${sanitizeErrorForClient(error.message).substring(0, 100)}`,
        );

        await this.costTracker.recordCall(candidate.provider, { durationMs, outcome }, candidate.model, estimatedTokens);
        this.db.recordCallOutcome({
          provider: candidate.provider, model: candidate.model,
          durationMs, outcome, timestamp: new Date().toISOString(),
        });
        this.db.updateDecisionOutcome(requestId, outcome, durationMs);

        // ── Apply failure-type-aware fallback strategy ──
        const fallbackDecision = computeFallbackDecision(
          failure,
          ci,
          candidates.map(c => ({ provider: c.provider, model: c.model })),
          candidate.provider,
          candidate.model,
        );

        logger.info(fallbackDecision.logMessage);

        // Apply provider backoff if the strategy calls for it
        if (fallbackDecision.applyBackoff) {
          providerBackoff.set(
            fallbackDecision.applyBackoff.provider,
            fallbackDecision.applyBackoff.durationMs,
            fallbackDecision.applyBackoff.reason,
          );
        }

        // Mark candidates to skip based on strategy
        for (const skipIdx of fallbackDecision.skipIndices) {
          if (skipIdx > ci) {
            const skipCand = candidates[skipIdx];
            const skipKey = `${skipCand.provider}/${skipCand.model}`;
            modelStrikes.set(skipKey, maxAttemptsForProvider(skipCand.provider));
          }
        }

        if (newStrikes >= providerAttemptLimit) {
          logger.info(`⏭️ ${candidate.provider} exhausted (${newStrikes} strikes) - moving to next provider`);
        }

        // ─── Hedged request: race fallback vs delayed primary retry ───
        // When the primary returns 429/503, immediately try the fallback AND
        // queue a delayed retry of the primary. First response wins.
        const isServerErr = (error as any).code === "server_error" || /server_error|503|internal server error/i.test(error.message);
        const shouldHedge =
          !hedgeAttempted &&
          (isRateLimit || isServerErr) &&
          !isStreaming &&
          !res.headersSent &&
          // Need at least one candidate from a DIFFERENT provider after this one
          candidates.some((c, idx) => idx > ci && c.provider !== candidate.provider) &&
          Date.now() < requestDeadlineMs;

        if (shouldHedge) {
          hedgeAttempted = true;
          // Find the next candidate from a DIFFERENT provider (not another model from same provider)
          let fallbackCand = candidates[ci + 1];
          for (let fi = ci + 1; fi < candidates.length; fi++) {
            if (candidates[fi].provider !== candidate.provider) {
              fallbackCand = candidates[fi];
              break;
            }
          }
          const fallbackAdapter = getProvider(fallbackCand.provider);
          const fallbackApiKey = process.env[fallbackCand.provider.toUpperCase() + "_API_KEY"] ?? "";

          if (fallbackAdapter && this.costTracker.isAvailable(fallbackCand.provider)) {
            const retryDelay = hedgeRetryDelayMs();
            logger.info(
              `🂺 Hedging: racing ${fallbackCand.provider}/${fallbackCand.model} (immediate) vs ` +
              `${candidate.provider}/${candidate.model} (retry in ${retryDelay}ms)`,
            );

            try {
              const hedgeResult = await raceHedgedRequests(
                {
                  provider: candidate.provider,
                  model: candidate.model,
                  adapter: adapter,
                  apiKey: apiKey,
                },
                {
                  provider: fallbackCand.provider,
                  model: fallbackCand.model,
                  adapter: fallbackAdapter,
                  apiKey: fallbackApiKey,
                },
                request,
                retryDelay,
              );

              // Winner determined - record stats and send response
              const { response: winResponse, outcome: hedgeOutcome } = hedgeResult;
              const winDurationMs = hedgeOutcome.winnerDurationMs;

              // Record success for winner
              await this.costTracker.recordCall(
                hedgeOutcome.winnerProvider,
                { durationMs: winDurationMs, outcome: "success" },
                hedgeOutcome.winnerModel,
                estimatedTokens,
              );
              this.costTracker.recordSizeLatency(hedgeOutcome.winnerProvider, estimatedTokens, winDurationMs);
              this.db.recordCallOutcome({
                provider: hedgeOutcome.winnerProvider, model: hedgeOutcome.winnerModel,
                durationMs: winDurationMs, outcome: "success", timestamp: new Date().toISOString(),
              });
              this.db.updateDecisionOutcome(requestId, "success", winDurationMs);

              // Track token usage for subscription providers (e.g., Z.AI quota)
              if (winResponse.usage && hedgeOutcome.winnerProvider === "zai") {
                this.costTracker.recordTokenUsage(
                  hedgeOutcome.winnerProvider,
                  winResponse.usage.prompt_tokens || 0,
                  winResponse.usage.completion_tokens || 0,
                );
              }

              // Track dollar spend for paid models
              const hedgeCost = calculateRequestCost(
                hedgeOutcome.winnerProvider, hedgeOutcome.winnerModel,
                this.modelRegistry, winResponse.usage,
              );
              if (hedgeCost > 0) {
                this.costTracker.recordSpend(hedgeOutcome.winnerProvider, hedgeCost);
              }

              // Record hedge outcome for analytics
              this.costTracker.recordHedgeOutcome({
                result: hedgeOutcome.result,
                winnerProvider: hedgeOutcome.winnerProvider,
                winnerModel: hedgeOutcome.winnerModel,
                loserCancelled: hedgeOutcome.loserCancelled,
              });

              logger.info(`✅ Hedge winner: ${hedgeOutcome.winnerProvider}/${hedgeOutcome.winnerModel} in ${winDurationMs}ms`);

              // Clear any provider backoff on success
              providerBackoff.clear(hedgeOutcome.winnerProvider);

              // Send the winning response to client
              const hedgeUpstreamModel = typeof winResponse.model === "string" ? winResponse.model : undefined;
              winResponse.model = CHAT_RESPONSE_MODEL;
              recordSource(true, { provider: hedgeOutcome.winnerProvider, model: hedgeOutcome.winnerModel });
              res.writeHead(200, {
                "Content-Type": "application/json",
                [SELECTED_MODEL_HEADER]: hedgeOutcome.winnerModel,
                ...(hedgeUpstreamModel ? { [UPSTREAM_MODEL_HEADER]: hedgeUpstreamModel } : {}),
              });
              res.end(JSON.stringify(winResponse));
              return;
            } catch (hedgeErr) {
              logger.warn(
                `Hedge failed: ${hedgeErr instanceof Error ? hedgeErr.message : hedgeErr} - continuing fallback loop`,
              );
              this.costTracker.recordHedgeOutcome({
                result: "both_fail",
                winnerProvider: "none",
                winnerModel: "none",
                loserCancelled: false,
              });
              // Skip the fallback candidate since hedge already tried it
              ci++;
              lastError = hedgeErr instanceof Error ? hedgeErr : new Error(String(hedgeErr));
            }
          }
        }
      }
    }

    // All providers exhausted
    recordSource(false, null);
    const errMsg = sanitizeErrorForClient(lastError?.message ?? "unknown");
    if (isStreaming) {
      this.writeSSE(res, {
        error: { message: `All providers exhausted. Last error: ${errMsg}`, type: "all_providers_exhausted" },
      });
      res.write("data: [DONE]\n\n");
      res.end();
    } else {
      res.writeHead(503, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        error: { message: `All providers exhausted. Last error: ${errMsg}`, type: "all_providers_exhausted" },
      }));
    }
  }

  // ─── Embeddings Handler (pure proxy — caller chooses model) ───

  private async handleEmbeddings(
    request: { model?: string; input: string | string[] },
    res: http.ServerResponse,
  ): Promise<void> {
    const inputs = Array.isArray(request.input) ? request.input : [request.input];
    const requestedModel = request.model ?? "";

    // If caller specified a model, use it directly (pure proxy — no routing)
    if (requestedModel) {
      const result = await this.proxyEmbedding(requestedModel, inputs);
      if (result.error) {
        res.writeHead(503, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: { message: result.error } }));
        return;
      }
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        object: "list",
        data: result.embeddings!.map((vec, i) => ({ object: "embedding", index: i, embedding: vec })),
        model: requestedModel,
        usage: { prompt_tokens: 0, total_tokens: 0 },
      }));
      return;
    }

    // No model specified — use default embedder (backward compat)
    const embeddings: Array<{ object: string; index: number; embedding: number[] }> = [];
    for (let i = 0; i < inputs.length; i++) {
      try {
        const vector = await this.embedFn(inputs[i]);
        embeddings.push({ object: "embedding", index: i, embedding: vector });
      } catch (err) {
        logger.error(`Embedding failed for input ${i}: ${err}`);
        res.writeHead(503, { "Content-Type": "application/json" });
        res.end(JSON.stringify({
          error: { message: `Embedding failed: ${err instanceof Error ? err.message : String(err)}` },
        }));
        return;
      }
    }

    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({
      object: "list",
      data: embeddings,
      model: EMBEDDING_RESPONSE_MODEL,
      usage: { prompt_tokens: 0, total_tokens: 0 },
    }));
  }

  /** Proxy an embedding request to the appropriate provider based on model name */
  private async proxyEmbedding(
    model: string,
    inputs: string[],
  ): Promise<{ embeddings?: number[][]; error?: string }> {
    // Determine provider from model name
    const { provider, endpoint } = this.resolveEmbeddingProvider(model);

    try {
      if (provider === "ollama") {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model, input: inputs }),
          signal: AbortSignal.timeout(120000),
        });
        if (!response.ok) throw new Error(`Ollama embed failed: ${response.status}`);
        const data = await response.json() as any;
        return { embeddings: data.embeddings as number[][] };
      }

      if (provider === "gemini") {
        const apiKey = process.env.GEMINI_API_KEY ?? "";
        const results: number[][] = [];
        for (const text of inputs) {
          const response = await fetch(
            `${endpoint}?key=${apiKey}`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ content: { parts: [{ text }] } }),
              signal: AbortSignal.timeout(30000),
            },
          );
          if (!response.ok) throw new Error(`Gemini embed failed: ${response.status}`);
          const data = await response.json() as any;
          results.push(data.embedding.values as number[]);
        }
        return { embeddings: results };
      }

      if (provider === "openai") {
        const apiKey = process.env.OPENAI_API_KEY ?? "";
        const response = await fetch(endpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${apiKey}`,
          },
          body: JSON.stringify({ model, input: inputs }),
          signal: AbortSignal.timeout(60000),
        });
        if (!response.ok) throw new Error(`OpenAI embed failed: ${response.status}`);
        const data = await response.json() as any;
        return { embeddings: data.data.map((d: any) => d.embedding) };
      }

      return { error: `Unknown embedding provider for model: ${model}` };
    } catch (err) {
      return { error: err instanceof Error ? err.message : String(err) };
    }
  }

  /** Resolve which provider handles a given embedding model */
  private resolveEmbeddingProvider(model: string): { provider: string; endpoint: string } {
    // Ollama local models
    if (
      model.includes("embed") ||
      model.includes("bge") ||
      model.includes("nomic") ||
      model === "embeddinggemma:latest"
    ) {
      return {
        provider: "ollama",
        endpoint: "http://localhost:11434/api/embed",
      };
    }

    // Gemini embedding models
    if (model.startsWith("gemini-embedding") || model.startsWith("text-embedding-004")) {
      return {
        provider: "gemini",
        endpoint: `https://generativelanguage.googleapis.com/v1beta/models/${model}:embedContent`,
      };
    }

    // OpenAI-compatible embedding models
    if (model.startsWith("text-embedding-") || model.startsWith("text-embedding-3")) {
      return {
        provider: "openai",
        endpoint: "https://api.openai.com/v1/embeddings",
      };
    }

    // Default: try Ollama
    return {
      provider: "ollama",
      endpoint: "http://localhost:11434/api/embed",
    };
  }

  /** List remote embedding models that are configured but not local */
  private listRemoteEmbeddingModels(): EmbeddingModelInfo[] {
    const remote: EmbeddingModelInfo[] = [];

    // Gemini embedding models (if API key configured)
    if (process.env.GEMINI_API_KEY) {
      remote.push({
        name: "gemini-embedding-001",
        provider: "gemini",
        contextWindow: 2048,
        isLocal: false,
        dimensions: 768,
      });
      remote.push({
        name: "text-embedding-004",
        provider: "gemini",
        contextWindow: 2048,
        isLocal: false,
        dimensions: 768,
      });
    }

    // OpenAI embedding models (if API key configured)
    if (process.env.OPENAI_API_KEY) {
      remote.push({
        name: "text-embedding-3-small",
        provider: "openai",
        contextWindow: 8192,
        isLocal: false,
        dimensions: 1536,
      });
      remote.push({
        name: "text-embedding-3-large",
        provider: "openai",
        contextWindow: 8192,
        isLocal: false,
        dimensions: 3072,
      });
    }

    return remote;
  }

  // ─── Port Cleanup ───

  private async cleanupPort(port: number): Promise<void> {
    try {
      const { stdout } = await execAsync(`netstat -ano | findstr :${port}`);
      if (stdout.trim()) {
        const lines = stdout.trim().split('\n');
        for (const line of lines) {
          const parts = line.trim().split(/\s+/);
          const pid = parts[parts.length - 1];
          const localAddress = parts[1];
          // Only kill processes in LISTENING state on 127.0.0.1
          if (localAddress.includes(`127.0.0.1:${port}`) || localAddress.includes(`0.0.0.0:${port}`)) {
            try {
              await execAsync(`taskkill /F /PID ${pid}`);
              logger.info(`Killed lingering process ${pid} on port ${port}`);
            } catch (err) {
              // Process might have already exited
              logger.debug(`Process ${pid} already exited: ${err}`);
            }
          }
        }
      }
    } catch (err) {
      // netstat might not find anything, or there's a permission issue - not fatal
      logger.debug(`Port cleanup check for ${port}: ${err}`);
    }
  }

  // ─── Ollama Health Check ───

  private async checkOllamaHealth(): Promise<boolean> {
    try {
      const response = await fetch('http://localhost:11434/api/tags', {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });
      return response.ok;
    } catch (err) {
      logger.debug(`Ollama health check failed: ${err}`);
      return false;
    }
  }

  // ─── Helpers ───

  /** Open SSE response headers. Deferred until the first write so the
   *  x-model-router-selected-model header can name the upstream model that
   *  actually served the request (set before any stream data is flushed). */
  private openSSE(res: http.ServerResponse, selectedModel?: string, upstreamModel?: string): void {
    if (res.headersSent || res.writableEnded) return;
    if (selectedModel) res.setHeader(SELECTED_MODEL_HEADER, selectedModel);
    if (upstreamModel) res.setHeader(UPSTREAM_MODEL_HEADER, upstreamModel);
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
    });
  }

  private writeSSE(res: http.ServerResponse, data: any): void {
    this.openSSE(res);
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  }

  private withRequestDeadline<T>(
    promise: Promise<T>,
    deadlineMs: number,
    label: string,
  ): Promise<T> {
    const remainingMs = deadlineMs - Date.now();
    if (remainingMs <= 0) {
      return Promise.reject(new Error(`router_request_timeout: no time remaining before ${label}`));
    }

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`router_request_timeout: exceeded total router deadline while waiting for ${label}`));
      }, remainingMs);

      promise.then(
        (value) => {
          clearTimeout(timeout);
          resolve(value);
        },
        (err) => {
          clearTimeout(timeout);
          reject(err);
        },
      );
    });
  }

  private buildCandidateList(
    decision: Awaited<ReturnType<RoutingEngine["decide"]>>,
    request: ChatCompletionRequest,
    req?: http.IncomingMessage,
  ): { list: Array<{ provider: string; model: string }>; lastResortAppended: boolean } {
    const candidates: Array<{ provider: string; model: string }> = [];
    let lastResortAppended = false;
    const usesTools = requestUsesTools(request);
    const estimatedTokens = estimateTokenCount(request);

    // Pin means pin (bench-extraction, 2026-09-07): when the decision came
    // from an explicit registry id, the candidate list is THAT MODEL ALONE.
    // Walking the failover chain on a pinned failure silently substitutes a
    // different model — the exact disease pin-mode exists to kill (live:
    // anchor gen returned empty → router served glm-5.1 → ollama last-resort
    // → 5-minute chain walk → client fetch died at undici's 300s
    // headersTimeout). Failing loudly lets the bench retry/backoff itself.
    if (decision?.decisionSource === "pinned_model") {
      candidates.push({ provider: decision.provider, model: decision.model });
      // Cross-provider failover for bench pins (Daz 2026-09-07): same model,
      // different door. Opt-in via x-bench-failover header (bench-only seam,
      // CONTRACT.md §2 rule 7). The pin stays first — free/subscription
      // endpoint serves when healthy; alternates append in cost order, so a
      // dead or zombie coding endpoint degrades to metered serving instead
      // of a 503. Pin semantics hold: every alternate is the SAME base model
      // (walking to a different model is the disease pin-mode kills).
      if (req?.headers["x-bench-failover"] === "true") {
        const base = (m: string) => m.split("/").pop()?.toLowerCase() ?? "";
        const pinnedBase = base(decision.model);
        const equivalents = this.modelRegistry
          .getAvailableModels(this.config.providerPriority)
          .filter(
            (m) =>
              m.provider !== decision.provider &&
              pinnedBase !== "" &&
              base(m.model) === pinnedBase &&
              this.costTracker.isAvailable(m.provider) &&
              estimatedTokens <= (m.contextWindow ?? 128_000),
          )
          .sort(
            (a, b) =>
              (a.costPer1kInput ?? 9e9) + (a.costPer1kOutput ?? 9e9) -
              ((b.costPer1kInput ?? 9e9) + (b.costPer1kOutput ?? 9e9)),
          )
          .slice(0, 3);
        for (const eq of equivalents) {
          candidates.push({ provider: eq.provider, model: eq.model });
        }
        if (equivalents.length > 0) {
          logger.info(
            `bench pin failover armed: ${decision.provider}/${decision.model} → ${equivalents
              .map((e) => `${e.provider}/${e.model}`)
              .join(", ")}`,
          );
        }
      }
      return { list: candidates, lastResortAppended: false };
    }

    // Router's top pick first for plain chat. For tool turns, keep provider
    // priority strict so remote tool-capable providers stay ahead of local ones.
    // Only include the decision winner if its provider is in the active priority list.
    const prioritySet = new Set(this.config.providerPriority);
    if (!usesTools && decision && prioritySet.has(decision.provider)) {
      candidates.push({ provider: decision.provider, model: decision.model });
    } else if (decision && usesTools) {
      logger.debug(`Deferring ${decision.provider}/${decision.model} - tool request uses provider priority`);
    }

    const maxProviderAttempts = maxAttemptsPerProvider();

    // Then top models per provider in priority order
    for (const provider of this.config.providerPriority) {
      if (usesTools && !providerSupportsTools(provider)) {
        logger.debug(`Skipping ${provider} - request uses tools`);
        continue;
      }

      if (!this.costTracker.isAvailable(provider)) {
        logger.debug(`Skipping ${provider} - circuit open`);
        continue;
      }

      // Context gate (2026-09-03): skip entire provider only when NO model in
      // it fits — a big-window model (glm-5.2 1M) keeps the provider alive
      // for oversized requests instead of skipping it wholesale.
      const providerDefault = PROVIDER_EFFECTIVE_INPUT_LIMITS[provider] ?? 128_000;
      const providerModels = this.modelRegistry.getAvailableModels([provider]);
      const bestWindow = Math.max(
        providerDefault,
        ...providerModels.map((m) => m.contextWindow ?? 0),
      );
      if (estimatedTokens > bestWindow) {
        logger.info(
          `Skipping ${provider} - request ~${estimatedTokens} tokens exceeds provider limit ${bestWindow}`,
        );
        continue;
      }

      const fallbackModel = usesTools ? fallbackModelForProvider(provider, true) : null;
      const budgetExceeded = this.costTracker.isBudgetExceeded(provider);
      let models = this.modelRegistry
        .getAvailableModels([provider])
        .filter((m) => {
          if (m.isLocal && m.vramRequiredGb && m.vramRequiredGb > (this.config.localVramLimitGb ?? 11)) {
            return false;
          }
          // Context gate (2026-09-03): only context-eligible models become
          // candidates — oversized requests fall through to big-window models
          // (glm-5.2 1M) instead of dying on the default pick.
          const mWindow = m.contextWindow ?? 0;
          if (mWindow > 0 && estimatedTokens > Math.floor(mWindow * CONTEXT_SAFETY_MARGIN)) {
            return false;
          }
          if (!isGenerationModel(m.provider, m.model)) {
            return false;
          }
          if (!this.costTracker.isAvailable(m.provider, m.model)) {
            return false;
          }
          if (usesTools && !modelSupportsTools(m.provider, m.model)) {
            return false;
          }
          // Skip paid models when budget is exceeded
          if (budgetExceeded && ((m.costPer1kInput ?? 0) > 0 || (m.costPer1kOutput ?? 0) > 0)) {
            logger.info(
              `Skipping ${m.provider}/${m.model} - budget exceeded ` +
              `(daily=$${this.costTracker.getDailySpend(provider).toFixed(2)}/${this.costTracker.dailyBudget.toFixed(2)}, ` +
              `monthly=$${this.costTracker.getMonthlySpend(provider).toFixed(2)}/${this.costTracker.monthlyBudget.toFixed(2)})`,
            );
            return false;
          }
          return true;
        })
        .sort((a, b) => {
          const capA = this.modelRegistry.getCapabilityScore(a.provider, a.model, "conversation");
          const capB = this.modelRegistry.getCapabilityScore(b.provider, b.model, "conversation");
          return capB - capA;
        })
        .slice(0, maxProviderAttempts);

      if (fallbackModel) {
        models = [{
          provider,
          model: fallbackModel,
          contextWindow: 128_000,
          modalities: ["text"],
          capabilities: {
            coding: 0.65,
            reasoning: 0.65,
            creative: 0.60,
            math: 0.60,
            analysis: 0.65,
            conversation: 0.68,
            retrieval: 0.62,
            science: 0.62,
            business: 0.63,
            summary: 0.65,
          },
          isLocal: false,
          source: "observed" as const,
        }, ...models.filter((m) => m.model !== fallbackModel)].slice(0, maxProviderAttempts);
      }

      if (models.length === 0) {
        const fallbackModel = fallbackModelForProvider(provider, usesTools);
        if (fallbackModel) {
          models = [{
            provider,
            model: fallbackModel,
            contextWindow: 128_000,
            modalities: ["text"],
            capabilities: {
              coding: 0.65,
              reasoning: 0.65,
              creative: 0.60,
              math: 0.60,
              analysis: 0.65,
              conversation: 0.68,
              retrieval: 0.62,
              science: 0.62,
              business: 0.63,
              summary: 0.65,
            },
            isLocal: false,
            source: "observed",
          }];
        }
      }

      for (const model of models) {
        const entry = { provider: model.provider, model: model.model };
        if (!candidates.some((c) => c.provider === entry.provider && c.model === entry.model)) {
          candidates.push(entry);
        }
      }
    }

    // Add paid OpenRouter fallback (DeepSeek V4 Flash) after free providers are exhausted.
    // This costs real money (~$0.01/request) but only fires when ZAI + Gemini are both down
    // and the request is too large for free OpenRouter models.
    // Skip if daily/monthly budget is exceeded.
    if (this.config.providerPriority.includes("openrouter")) {
      const paidModel = paidOpenRouterFallbackModel();
      const paidLimit = getEffectiveInputLimit("openrouter", paidModel);
      const alreadyHaveIt = candidates.some((c) => c.provider === "openrouter" && c.model === paidModel);
      const orStillAvailable = this.costTracker.isAvailable("openrouter");
      const orBudgetExceeded = this.costTracker.isBudgetExceeded("openrouter");
      if (!alreadyHaveIt && orStillAvailable && estimatedTokens <= paidLimit && !orBudgetExceeded) {
        candidates.push({ provider: "openrouter", model: paidModel });
      } else if (alreadyHaveIt && orBudgetExceeded) {
        // Remove the paid model if budget was exceeded after it was added
        const idx = candidates.findIndex((c) => c.provider === "openrouter" && c.model === paidModel);
        if (idx !== -1) candidates.splice(idx, 1);
      }
    }

    // Ensure Ollama is always in the list as last resort for plain chat.
    // Tool turns only use explicitly allowlisted local models above.
    const hasOllama = candidates.some((c) => c.provider === "ollama");
    if (!usesTools && !hasOllama) {
      candidates.push({ provider: "ollama", model: "gemma4:latest" });
      lastResortAppended = true;
    }

    return { list: candidates, lastResortAppended };
  }

  private readBody(req: http.IncomingMessage): Promise<string> {
    return new Promise((resolve, reject) => {
      let body = "";
      req.on("data", (chunk) => {
        body += chunk;
        if (body.length > 10_000_000) {
          req.destroy();
          reject(new Error("Body too large"));
        }
      });
      req.on("end", () => resolve(body));
      req.on("error", reject);
    });
  }

  private extractSessionKey(req: http.IncomingMessage, request: ChatCompletionRequest): string {
    const headerValue =
      req.headers["x-openclaw-session-key"] ??
      req.headers["x-session-key"] ??
      req.headers["x-session-id"];
    const headerSession = Array.isArray(headerValue) ? headerValue[0] : headerValue;
    const metadataSession =
      request.metadata?.sessionKey ??
      request.metadata?.session_key ??
      request.metadata?.sessionId ??
      request.metadata?.session_id ??
      request.user;

    return String(headerSession ?? metadataSession ?? "proxy");
  }

  private extractRequestId(request: ChatCompletionRequest): string {
    const metadataRequest =
      request.metadata?.requestId ??
      request.metadata?.request_id;

    return String(metadataRequest ?? `req_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`);
  }

  private hashPrompt(prompt: string): string {
    // Defensive: OpenClaw 2026.8.x can deliver message content as structured
    // blocks (arrays/objects), not always a plain string. Coerce before hashing
    // - a 500 here kills the whole request (charCodeAt crash, Aug/Sep 2026 incident).
    const text = typeof prompt === "string" ? prompt : JSON.stringify(prompt ?? "");
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      hash = Math.imul(31, hash) + text.charCodeAt(i) | 0;
    }
    return hash.toString(16);
  }

  private createEmbedder(): (text: string) => Promise<number[]> {
    return async (text: string): Promise<number[]> => {
      // Ollama primary
      try {
        const response = await fetch("http://localhost:11434/api/embeddings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: "nomic-embed-text", prompt: text }),
          signal: AbortSignal.timeout(10000),
        });
        if (!response.ok) throw new Error(`Ollama embedding failed: ${response.status}`);
        const data = await response.json() as { embedding: number[] };
        return data.embedding;
      } catch (err) {
        // Gemini fallback
        try {
          const apiKey = process.env.GEMINI_API_KEY;
          if (!apiKey) throw new Error("No Gemini key");
          const response = await fetch(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key=" + apiKey,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ content: { parts: [{ text }] } }),
              signal: AbortSignal.timeout(10000),
            },
          );
          if (!response.ok) throw new Error(`Gemini embedding failed: ${response.status}`);
          const data = await response.json() as { embedding: { values: number[] } };
          return data.embedding.values;
        } catch (geminiErr) {
          throw new Error(`Embedding failed: ${err} / ${geminiErr}`);
        }
      }
    };
  }
}

class AllProvidersDeadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AllProvidersDeadError";
  }
}

export async function startProxyStreaming(): Promise<void> {
  getZaiQuotaProbe().start();
  const pluginConfig = {
    enabled: true,
    logLevel: process.env.ROUTER_LOG_LEVEL ?? "info",
    dbPath: process.env.ROUTER_DB_PATH ?? "data/cognitive-router.db",
    providerPriority: (process.env.ROUTER_PRIORITY ?? "zai,openrouter,gemini,ollama").split(","),
    localVramLimitGb: parseInt(process.env.ROUTER_VRAM_LIMIT ?? "11", 10),
    providers: {
      openrouter: { budgetType: "free", priority: "high" },
      zai: { budgetType: "subscription", priority: "high" },
      gemini: { budgetType: "credits", priority: "medium" },
      ollama: { budgetType: "free", priority: "low" },
    },
    overrides: [],
    weights: { capability: 0.50, reliability: 0.25, cost: 0.15, latency: 0.10 },
    probeRate: parseFloat(process.env.ROUTER_PROBE_RATE ?? "0.05"),
    tiebreakerThreshold: 0.70,
    proxyPort: parseInt(process.env.ROUTER_PORT ?? "3456", 10),
    bindHost: process.env.ROUTER_BIND_HOST ?? "127.0.0.1",
    apiKeys: (process.env.ROUTER_API_KEYS ?? "")
      .split(",")
      .map((k) => k.trim())
      .filter(Boolean),
  };

  const config = loadConfig(pluginConfig);

  const proxy = new ProxyServerStreaming(config);
  await proxy.start();

  const shutdown = async (signal: string) => {
    logger.info(`${signal} received - shutting down...`);
    await proxy.stop();
    process.exit(0);
  };

  process.on("SIGINT", () => shutdown("SIGINT"));
  process.on("SIGTERM", () => shutdown("SIGTERM"));
}

