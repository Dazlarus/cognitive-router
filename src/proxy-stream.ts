// src/proxy-stream.ts — Streaming + self-healing proxy for Cognitive Router

import http from "node:http";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { logger } from "./logger.js";
import { IntentClassifier } from "./classifier.js";
import { RoutingEngine } from "./router.js";
import { DBService } from "./db_service.js";
import { CostTracker } from "./cost_tracker.js";
import { ModelRegistry } from "./model_registry.js";
import { loadConfig, type CognitiveRouterConfig } from "./config.js";
import { getProvider } from "./providers.js";
import { loadProjectEnv } from "./env.js";
import { buildStatsPayload } from "./stats.js";
import { isGenerationModel, modelSupportsTools } from "./model_policy.js";
import { JudgeEvaluator } from "./judge.js";

const CHAT_ALIAS_MODEL = "CognitiveRouter:latest";
const LEGACY_CHAT_ALIAS_MODEL = "CogRouter:latest";
const EMBEDDING_ALIAS_MODEL = "Embeddings:latest";
const CHAT_RESPONSE_MODEL = CHAT_ALIAS_MODEL;
const EMBEDDING_RESPONSE_MODEL = EMBEDDING_ALIAS_MODEL;
const DEFAULT_OPENROUTER_TOOL_MODEL = "qwen/qwen3-coder:free";
const DEFAULT_OPENROUTER_FALLBACK_MODEL = "openrouter/owl-alpha";
const DEFAULT_GEMINI_FALLBACK_MODEL = "gemini-2.5-flash";
const DEFAULT_GEMINI_TOOL_MODEL = "gemini-2.5-flash";
const ALLOWED_OPENROUTER_FALLBACK_MODELS = new Set([
  "cohere/north-mini-code:free",
  "openrouter/free",
  "nvidia/nemotron-3-ultra-550b-a55b:free",
  "nvidia/nemotron-3-super-120b-a12b:free",
  "openrouter/owl-alpha",
  "poolside/laguna-m.1:free",
  "qwen/qwen3-coder:free",
  "qwen/qwen3.6-plus:free",
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
  [key: string]: any;
}

function maxAttemptsPerProvider(): number {
  const raw = Number.parseInt(process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER ?? "1", 10);
  return Number.isFinite(raw) && raw > 0 ? raw : 1;
}

function routerRequestTimeoutMs(): number {
  const raw = Number.parseInt(process.env.ROUTER_REQUEST_TIMEOUT_MS ?? "55000", 10);
  return Number.isFinite(raw) && raw > 0 ? raw : 55_000;
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

export class ProxyServerStreaming {
  private server: http.Server;
  private classifier: IntentClassifier;
  private router: RoutingEngine;
  private db: DBService;
  private costTracker: CostTracker;
  private modelRegistry: ModelRegistry;
  private config: CognitiveRouterConfig;
  private embedFn: (text: string) => Promise<number[]>;
  private judge: JudgeEvaluator;
  private initialized = false;

  constructor(config: CognitiveRouterConfig) {
    this.config = config;

    const dbPath = resolve(config.dbPath);
    mkdirSync(resolve(dbPath, ".."), { recursive: true });

    this.db = new DBService(dbPath);
    this.db.initializeSchema();

    this.modelRegistry = new ModelRegistry(this.db, config);
    this.costTracker = new CostTracker(this.db, config);
    this.classifier = new IntentClassifier({ tiebreakerThreshold: config.tiebreakerThreshold });
    this.router = new RoutingEngine(this.modelRegistry, this.costTracker, this.db, config);

    this.embedFn = this.createEmbedder();
    this.judge = new JudgeEvaluator();
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
    // Load model registry
    await this.modelRegistry.loadCachedState();
    await this.costTracker.refreshProviderStatus();

    this.server.listen(this.config.proxyPort, "127.0.0.1", () => {
      logger.info(`Cognitive Router proxy listening on http://127.0.0.1:${this.config.proxyPort}`);
      logger.info(`   POST /v1/chat/completions    — Chat (streaming + non-streaming)`);
      logger.info(`   POST /v1/embeddings           — Embeddings (Ollama + Gemini fallback)`);
      logger.info(`   GET  /v1/models               — List available models`);
      logger.info(`   GET  /health                  — Health check`);
      logger.info(`   GET  /stats                   — Provider health + stats`);
    });

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
    this.server.close();
    this.db.close();
  }

  // ─── Request Handler ───

  private async handleRequest(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
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

      if (url === "/health" && req.method === "GET") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ status: "ok", initialized: this.initialized }));
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

      if (url === "/v1/chat/completions" && req.method === "POST") {
        const body = await this.readBody(req);
        const request = JSON.parse(body) as ChatCompletionRequest;
        await this.handleChat(request, res, this.extractSessionKey(req, request));
        return;
      }

      if (url === "/v1/embeddings" && req.method === "POST") {
        const body = await this.readBody(req);
        const request = JSON.parse(body) as { model: string; input: string | string[] };
        await this.handleEmbeddings(request, res);
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
  ): Promise<void> {
    if (!this.initialized) throw new Error("Proxy not initialized");
    const isStreaming = request.stream === true;

    // For streaming, open SSE immediately but do not send an empty assistant
    // chunk. OpenClaw treats an empty assistant delta as an empty model answer.
    if (isStreaming) {
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
      });
    }

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

    // Build candidate list
    const decision = await this.router.decide(classification, sessionKey, {});
    const candidates = this.buildCandidateList(decision, request);
    const requestId = this.extractRequestId(request);

    if (decision) {
      this.db.recordDecision({
        timestamp: new Date().toISOString(),
        sessionKey,
        messageHash: this.hashPrompt(prompt),
        intent: classification.intent,
        confidence: classification.confidence,
        provider: decision.provider,
        model: decision.model,
        scores: decision.scores,
        overallScore: decision.overallScore,
        outcome: "PENDING",
        requestId,
      });
    }

    logger.info(
      `Chat request — session=${sessionKey} intent=${classification.intent} stream=${isStreaming} — ` +
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

    // Retry loop. Keep the default conservative so CogRouter can fail over
    // before OpenClaw's outer LLM timeout fires.
    let lastError: Error | null = null;
    const providerStrikes = new Map<string, number>();
    const requestDeadlineMs = Date.now() + routerRequestTimeoutMs();

    for (const candidate of candidates) {
      if (Date.now() >= requestDeadlineMs) {
        lastError = new Error(`router_request_timeout: exceeded ${routerRequestTimeoutMs()}ms before trying ${candidate.provider}/${candidate.model}`);
        logger.warn(lastError.message);
        break;
      }

      const providerAttemptLimit = maxAttemptsForProvider(candidate.provider);
      const strikes = providerStrikes.get(candidate.provider) ?? 0;
      if (strikes >= providerAttemptLimit) {
        logger.debug(`Skipping ${candidate.provider}/${candidate.model} — ${strikes} strikes`);
        continue;
      }

      const adapter = getProvider(candidate.provider);
      if (!adapter) continue;

      // Check circuit breaker
      if (!this.costTracker.isAvailable(candidate.provider)) {
        logger.debug(`Skipping ${candidate.provider} — circuit open`);
        providerStrikes.set(candidate.provider, providerAttemptLimit);
        continue;
      }

      if (!this.costTracker.isAvailable(candidate.provider, candidate.model)) {
        logger.debug(`Skipping ${candidate.provider}/${candidate.model} — model circuit open`);
        providerStrikes.set(candidate.provider, strikes + 1);
        continue;
      }

      const apiKey = process.env[candidate.provider.toUpperCase() + "_API_KEY"] ?? "";
      const startTime = Date.now();

      try {
        logger.info(`Trying ${candidate.provider}/${candidate.model}...`);

        // Always use non-streaming from provider (we handle SSE to client)
        const providerRequest = { ...request, model: candidate.model, stream: false };
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

        await this.costTracker.recordCall(candidate.provider, { durationMs, outcome: "success" }, candidate.model);
        this.db.recordCallOutcome({
          provider: candidate.provider, model: candidate.model,
          durationMs, outcome: "success", timestamp: new Date().toISOString(),
        });

        logger.info(`✅ ${candidate.provider}/${candidate.model} succeeded in ${durationMs}ms`);

        if (isStreaming) {
          const delta: Record<string, any> = { role: "assistant" };
          if (content) {
            delta.content = content;
          }
          if (toolCalls) {
            delta.tool_calls = toolCalls;
          }

          // Send the provider's assistant payload as SSE. This may be text
          // content or tool_calls; both are valid OpenAI assistant payloads.
          this.writeSSE(res, {
            id: response.id ?? `chatcmpl-${Date.now()}`,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model: CHAT_RESPONSE_MODEL,
            choices: [{ index: 0, delta, finish_reason: null }],
          });
          this.writeSSE(res, {
            id: response.id ?? `chatcmpl-${Date.now()}`,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model: CHAT_RESPONSE_MODEL,
            choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
          });
          res.write("data: [DONE]\n\n");
          res.end();
        } else {
          response.model = CHAT_RESPONSE_MODEL;
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify(response));
        }

        // ─── Async LLM-as-judge feedback ───
        // After the response is sent, maybe evaluate quality and evolve capability scores.
        // This runs in the background and never affects the response path.
        if (
          this.judge.shouldJudge() &&
          !this.judge.isSameModel(candidate.provider, candidate.model) &&
          !toolCalls // Skip tool-call responses — judge evaluates text quality only
        ) {
          const evalProvider = candidate.provider;
          const evalModel = candidate.model;
          const evalIntent = classification.intent;
          const evalPrompt = prompt;
          const evalResponse = content;

          // Fire and forget — don't await, don't block
          this.judge.evaluate(evalPrompt, evalResponse, evalIntent)
            .then((result) => {
              if (result) {
                this.modelRegistry.updateCapability(evalProvider, evalModel, evalIntent, result.score);
                this.db.recordJudgeEvaluation(
                  evalProvider, evalModel, evalIntent,
                  result.rawScore, result.note,
                  `${this.judge.judgeModelId}`,
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
        const isRateLimit = (error as any).code === "rate_limit" || /rate.?limit|429|slow down/i.test(error.message);
        const isQuota = (error as any).code === "quota_exceeded" || /quota|monthly limit|prompt tokens limit exceeded|402/i.test(error.message);
        const isTimeout = error.name === "TimeoutError" || error.name === "AbortError" || /timeout|aborted/i.test(error.message);
        const isEmpty = (error as any).code === "empty_response";
        const outcome = isRateLimit || isQuota ? "rate_limit" : isTimeout ? "timeout" : isEmpty ? "empty" : "error";

        lastError = error;
        const newStrikes = strikes + 1;
        providerStrikes.set(candidate.provider, newStrikes);

        logger.warn(
          `❌ ${candidate.provider}/${candidate.model} failed (${outcome}) in ${durationMs}ms ` +
          `[strike ${newStrikes}/${providerAttemptLimit}]: ${sanitizeErrorForClient(error.message).substring(0, 100)}`,
        );

        await this.costTracker.recordCall(candidate.provider, { durationMs, outcome }, candidate.model);
        this.db.recordCallOutcome({
          provider: candidate.provider, model: candidate.model,
          durationMs, outcome, timestamp: new Date().toISOString(),
        });

        if (newStrikes >= providerAttemptLimit) {
          logger.info(`⏭️ ${candidate.provider} exhausted (${newStrikes} strikes) — moving to next provider`);
        }
      }
    }

    // All providers exhausted
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

  // ─── Embeddings Handler ───

  private async handleEmbeddings(
    request: { model: string; input: string | string[] },
    res: http.ServerResponse,
  ): Promise<void> {
    const inputs = Array.isArray(request.input) ? request.input : [request.input];
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

  // ─── Helpers ───

  private writeSSE(res: http.ServerResponse, data: any): void {
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
  ): Array<{ provider: string; model: string }> {
    const candidates: Array<{ provider: string; model: string }> = [];
    const usesTools = requestUsesTools(request);

    // Router's top pick first for plain chat. For tool turns, keep provider
    // priority strict so remote tool-capable providers stay ahead of local ones.
    if (!usesTools && decision) {
      candidates.push({ provider: decision.provider, model: decision.model });
    } else if (decision && usesTools) {
      logger.debug(`Deferring ${decision.provider}/${decision.model} — tool request uses provider priority`);
    }

    const maxProviderAttempts = maxAttemptsPerProvider();

    // Then top models per provider in priority order
    for (const provider of this.config.providerPriority) {
      if (usesTools && !providerSupportsTools(provider)) {
        logger.debug(`Skipping ${provider} — request uses tools`);
        continue;
      }

      if (!this.costTracker.isAvailable(provider)) {
        logger.debug(`Skipping ${provider} — circuit open`);
        continue;
      }

      const fallbackModel = usesTools ? fallbackModelForProvider(provider, true) : null;
      let models = this.modelRegistry
        .getAvailableModels([provider])
        .filter((m) => {
          if (m.isLocal && m.vramRequiredGb && m.vramRequiredGb > (this.config.localVramLimitGb ?? 11)) {
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

    // Ensure Ollama is always in the list as last resort for plain chat.
    // Tool turns only use explicitly allowlisted local models above.
    const hasOllama = candidates.some((c) => c.provider === "ollama");
    if (!usesTools && !hasOllama) {
      candidates.push({ provider: "ollama", model: "gemma4:latest" });
    }

    return candidates;
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
    let hash = 0;
    for (let i = 0; i < prompt.length; i++) {
      hash = Math.imul(31, hash) + prompt.charCodeAt(i) | 0;
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
    probeRate: 0.05,
    tiebreakerThreshold: 0.70,
    proxyPort: parseInt(process.env.ROUTER_PORT ?? "3456", 10),
  };

  const config = loadConfig(pluginConfig);

  const proxy = new ProxyServerStreaming(config);
  await proxy.start();

  const shutdown = async (signal: string) => {
    logger.info(`${signal} received — shutting down...`);
    await proxy.stop();
    process.exit(0);
  };

  process.on("SIGINT", () => shutdown("SIGINT"));
  process.on("SIGTERM", () => shutdown("SIGTERM"));
}
