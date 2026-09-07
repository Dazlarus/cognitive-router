// src/providers.ts — Provider adapters
// Each adapter translates an OpenAI-compatible request to the provider's native API format
// and returns an OpenAI-compatible response. Supports both non-streaming and streaming.

import { logger } from "./logger.js";
import { maybeGeminiCache, observeZaIPrefix, PrefixCache } from "./prefix_cache.js";

// ─── Types ───

export interface ChatMessage {
  role: string;
  content: string;
  [key: string]: any;
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

// Thinking/Reasoning helpers — translate OpenClaw's thinking param to each provider's format

export function extractThinkingLevel(request: ChatCompletionRequest): "none" | "low" | "medium" | "high" {
  if (request.thinking) {
    if (typeof request.thinking === "string") {
      const v = request.thinking.toLowerCase();
      if (v === "none" || v === "off") return "none";
      if (v === "low" || v === "light") return "low";
      if (v === "high" || v === "deep" || v === "enabled") return "high";
      return "medium";
    }
    if (typeof request.thinking === "object") {
      const type = request.thinking.type ?? request.thinking.level ?? "medium";
      if (type === "disabled" || type === "none") return "none";
      const budget = request.thinking.budget_tokens;
      if (budget !== undefined) {
        if (budget <= 1024) return "low";
        if (budget <= 8192) return "medium";
        return "high";
      }
      return "medium";
    }
  }
  if (request.reasoning_effort) {
    const v = request.reasoning_effort.toLowerCase();
    if (v === "none" || v === "minimal") return "none";
    if (v === "low") return "low";
    if (v === "high") return "high";
    return "medium";
  }
  if (request.reasoning) {
    if (typeof request.reasoning === "string") return request.reasoning.toLowerCase() as any;
    if (typeof request.reasoning === "object") return (request.reasoning.effort ?? "medium").toLowerCase() as any;
  }
  return "none";
}

export type SpeedMode = "normal" | "fast";

/** Read OpenClaw's request-level speed hint (wire format: speed: "fast"). */
export function extractSpeedMode(request: ChatCompletionRequest): SpeedMode {
  const v = typeof request.speed === "string" ? request.speed.toLowerCase() : "";
  if (v === "fast" || v === "priority" || v === "turbo") return "fast";
  return "normal";
}

function envTimeoutMs(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;

  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function remoteTimeoutMs(): number {
  return envTimeoutMs("ROUTER_REMOTE_TIMEOUT_MS", envTimeoutMs("ROUTER_PROVIDER_TIMEOUT_MS", 25_000));
}

function remoteStreamTimeoutMs(): number {
  return envTimeoutMs("ROUTER_REMOTE_STREAM_TIMEOUT_MS", envTimeoutMs("ROUTER_PROVIDER_STREAM_TIMEOUT_MS", 120_000));
}

function localTimeoutMs(): number {
  return envTimeoutMs("ROUTER_LOCAL_TIMEOUT_MS", envTimeoutMs("ROUTER_PROVIDER_TIMEOUT_MS", 45_000));
}

function localStreamTimeoutMs(): number {
  return envTimeoutMs("ROUTER_LOCAL_STREAM_TIMEOUT_MS", envTimeoutMs("ROUTER_PROVIDER_STREAM_TIMEOUT_MS", 120_000));
}

/**
 * Combine an external AbortSignal with a timeout signal.
 * If no external signal is provided, returns just the timeout signal.
 */
function withExternalSignal(timeoutMs: number, external?: AbortSignal): AbortSignal {
  const timeout = AbortSignal.timeout(timeoutMs);
  if (!external) return timeout;
  return AbortSignal.any([timeout, external]);
}

// ─── Provider Base URLs (env-configurable) ───

/** ZAI billing truth follows the effective base URL (2026-09-06, Daz):
 *  coding-plan endpoints (…/api/coding/paas/v4, /api/anthropic, /api/v1)
 *  are subscription (sunk cost); the platform endpoint (…/api/paas/v4)
 *  is pay-per-token. Pricing decisions for discovered zai models use this. */
export function zaiBillingMode(): "coding_plan" | "platform" {
  const base = providerBaseUrl("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4");
  return /\/coding\//.test(base) ? "coding_plan" : "platform";
}

function providerBaseUrl(envVar: string, fallback: string): string {
  const raw = process.env[envVar];
  if (!raw) return fallback;
  // Strip trailing slash so callers can append "/chat/completions"
  return raw.replace(/\/+$/, "");
}

const ZAI_BASE = providerBaseUrl("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4");
const OPENAI_BASE = providerBaseUrl("OPENAI_BASE_URL", "https://api.openai.com/v1");
const OPENROUTER_BASE = providerBaseUrl("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1");
const ANTHROPIC_BASE = providerBaseUrl("ANTHROPIC_BASE_URL", "https://api.anthropic.com");
const GEMINI_BASE = providerBaseUrl("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta");
const OLLAMA_BASE = providerBaseUrl("OLLAMA_BASE_URL", "http://localhost:11434/v1");

logger.info(`Provider endpoints: zai=${ZAI_BASE} openai=${OPENAI_BASE} openrouter=${OPENROUTER_BASE} anthropic=${ANTHROPIC_BASE} gemini=${GEMINI_BASE} ollama=${OLLAMA_BASE}`);

function buildZaiThinking(level: string): any {
  if (level === "none") return undefined;
  const budgets: Record<string, number> = { low: 2048, medium: 8192, high: 32768 };
  return { type: "enabled", budget_tokens: budgets[level] ?? 8192 };
}

/** Model-aware zai thinking translation. GLM-5.3+ is reasoning-always-on:
 *  it accepts ONLY `thinking.type: "enabled"` + `reasoning_effort`
 *  low|high|max — the disabled type and budget_tokens style are rejected
 *  (docs.z.ai/guides/llm/glm-5.3, migration notice). Older models keep the
 *  budget_tokens ladder. Effort mapping preserves ordering:
 *  low→low, medium→high, high→max, none→low (5.3 cannot disable). */
function buildZaiThinkingFields(model: string, level: string): { thinking?: unknown; reasoning_effort?: string } {
  if (/glm-5\.[3-9]|glm-[6-9]/i.test(model)) {
    const effort = level === "high" ? "max" : level === "medium" ? "high" : "low";
    return { thinking: { type: "enabled" }, reasoning_effort: effort };
  }
  if (/glm-5\.2/i.test(model)) {
    // GLM-5.2 supports the full effort ladder AND a genuine disabled state
    // (docs concept-param: low/medium internally map to high, xhigh→max;
    // reasoning_effort requires thinking.type=enabled, GLM-5.2 and above).
    if (level === "none") return { thinking: { type: "disabled" } };
    const effort = level === "high" ? "max" : level === "medium" ? "medium" : "low";
    return { thinking: { type: "enabled" }, reasoning_effort: effort };
  }
  const thinking = buildZaiThinking(level);
  return thinking ? { thinking } : {};
}

function buildOpenAIThinking(level: string): string | undefined {
  if (level === "none") return undefined;
  return level; // reasoning_effort: none|minimal|low|medium|high|xhigh|max (per-model support)
}

function buildAnthropicThinking(level: string): { thinking?: { type: "adaptive" }; effort?: string } {
  if (level === "none") return {};
  // Claude 4.7+ rejects thinking.type "enabled" (400); the current control
  // surface is thinking.type "adaptive" + top-level effort (docs: Thinking /
  // Thinking and effort, platform.claude.com). "none" means "model default"
  // here — some models (Claude 5 family) cannot disable thinking at all.
  return { thinking: { type: "adaptive" }, effort: level };
}

function buildOpenRouterThinking(level: string): any {
  if (level === "none") return undefined;
  return level;
}

function buildGeminiThinking(level: string): any {
  if (level === "none") return { thinkingBudget: 0 };
  const budgets: Record<string, number> = { low: 1024, medium: 8192, high: 24576 };
  return { thinkingBudget: budgets[level] ?? 8192 };
}

function stripThinking(request: ChatCompletionRequest): ChatCompletionRequest {
  const { thinking, reasoning, reasoning_effort, speed, ...rest } = request;
  return rest as ChatCompletionRequest;
}

function parseFunctionArguments(value: unknown): Record<string, unknown> {
  if (!value) return {};
  if (typeof value === "object") return value as Record<string, unknown>;
  if (typeof value !== "string") return {};
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

export interface ChatCompletionResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: ChatMessage;
    finish_reason: string;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

// Streaming chunk — OpenAI SSE format
export interface ChatCompletionChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: { role?: string; content?: string; tool_calls?: any[] };
    finish_reason: string | null;
  }>;
}

export interface ProviderAdapter {
  name: string;
  /** Send a non-streaming chat completion request. Throws on failure. */
  chatCompletion(
    model: string,
    request: ChatCompletionRequest,
    apiKey: string,
    signal?: AbortSignal,
  ): Promise<ChatCompletionResponse>;

  /** Send a streaming chat completion request. Yields chunks. Throws on failure. */
  chatCompletionStream(
    model: string,
    request: ChatCompletionRequest,
    apiKey: string,
    signal?: AbortSignal,
  ): AsyncIterable<ChatCompletionChunk>;
}

// ─── Helpers ───

function generateId(): string {
  return `chatcmpl-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
}

function classifyError(status: number, body: string): Error {
  const cleanBody = sanitizeProviderErrorBody(body);
  if (status === 429 || /rate.?limit|too many requests|throttl|slow down/i.test(body)) {
    const err = new Error(`rate_limit: ${cleanBody}`);
    (err as any).code = "rate_limit";
    return err;
  }
  if (status === 402 || /prompt tokens limit exceeded|monthly limit|quota|insufficient credit/i.test(body)) {
    const err = new Error(`quota_exceeded (${status}): ${cleanBody}`);
    (err as any).code = "quota_exceeded";
    return err;
  }
  if (status >= 500) {
    const err = new Error(`server_error (${status}): ${cleanBody}`);
    (err as any).code = "server_error";
    return err;
  }
  if (status === 401 || status === 403) {
    const err = new Error(`auth_error (${status}): ${cleanBody}`);
    (err as any).code = "auth_error";
    return err;
  }
  const err = new Error(`http_error (${status}): ${cleanBody}`);
  (err as any).code = "http_error";
  return err;
}

function sanitizeProviderErrorBody(body: string): string {
  return body
    .replace(/https:\/\/openrouter\.ai\/workspaces\/[^"'\s)\]]+/gi, "[openrouter-key-settings]")
    .replace(/user_[A-Za-z0-9]+/g, "[provider-user]")
    .replace(/[A-Fa-f0-9]{32,}/g, "[redacted]");
}

// ─── SSE Stream Parser ───
// Parses an SSE stream from a fetch Response and yields ChatCompletionChunks.

async function* parseSSEStream(
  response: Response,
  model: string,
): AsyncIterable<ChatCompletionChunk> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE events are separated by double newlines
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith(":")) continue; // comment or empty
      if (!trimmed.startsWith("data: ")) continue;

      const data = trimmed.substring(6);
      if (data === "[DONE]") return;

      try {
        const chunk = JSON.parse(data) as ChatCompletionChunk;
        if (chunk.model) chunk.model = model; // normalize model name
        yield chunk;
      } catch {
        // Skip malformed chunks
      }
    }
  }
}

// ─── ZAI Adapter (OpenAI-compatible, supports streaming) ───

export const ZAIAdapter: ProviderAdapter = {
  name: "zai",

  async chatCompletion(
    model: string,
    request: ChatCompletionRequest,
    apiKey: string,
    externalSignal?: AbortSignal,
  ): Promise<ChatCompletionResponse> {
    const level = extractThinkingLevel(request);
    const cleaned = stripThinking(request);
    const thinkingFields = buildZaiThinkingFields(model, level);

    // Observe system prompt prefix for ZAI automatic prefix caching
    // (GLM-4+ caches identical prefixes ≥1024 tokens transparently)
    observeZaIPrefix(request.messages);

    const body = { ...cleaned, model, stream: false, ...thinkingFields };
    const resp = await fetch(`${ZAI_BASE}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(body),
      signal: withExternalSignal(remoteTimeoutMs(), externalSignal),
    });

    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw classifyError(resp.status, text);
    }

    return (await resp.json()) as ChatCompletionResponse;
  },

  async* chatCompletionStream(
    model: string,
    request: ChatCompletionRequest,
    apiKey: string,
    externalSignal?: AbortSignal,
  ): AsyncIterable<ChatCompletionChunk> {
    const level = extractThinkingLevel(request);
    const cleaned = stripThinking(request);
    const thinkingFields = buildZaiThinkingFields(model, level);

    // Observe system prompt prefix for ZAI automatic prefix caching
    observeZaIPrefix(request.messages);

    const body = { ...cleaned, model, stream: true, ...thinkingFields };
    const resp = await fetch(`${ZAI_BASE}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(body),
      signal: withExternalSignal(remoteStreamTimeoutMs(), externalSignal),
    });

    if (!resp.ok || !resp.body) {
      const text = await resp.text().catch(() => "");
      throw classifyError(resp.status, text);
    }

    yield* parseSSEStream(resp, model);
  },
};

// ─── OpenAI Adapter (native /v1/chat/completions, supports streaming) ───
// The inbound format is already OpenAI-compatible; translation is limited to
// the reasoning-era deltas: max_tokens is deprecated (rejected by o-series —
// max_completion_tokens replaces it), and thinking maps to reasoning_effort
// (docs.openai.com: supported values none|minimal|low|medium|high|xhigh|max).

export const OpenAIAdapter: ProviderAdapter = {
  name: "openai",

  async chatCompletion(
    model: string,
    request: ChatCompletionRequest,
    apiKey: string,
    externalSignal?: AbortSignal,
  ): Promise<ChatCompletionResponse> {
    const level = extractThinkingLevel(request);
    const cleaned = stripThinking(request);
    const reasoningEffort = buildOpenAIThinking(level);
    const body = {
      ...maxTokensToCompletionTokens(cleaned),
      model,
      stream: false,
      ...(reasoningEffort ? { reasoning_effort: reasoningEffort } : {}),
    };
    const resp = await fetch(`${OPENAI_BASE}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(body),
      signal: withExternalSignal(remoteTimeoutMs(), externalSignal),
    });

    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw classifyError(resp.status, text);
    }

    return (await resp.json()) as ChatCompletionResponse;
  },

  async* chatCompletionStream(
    model: string,
    request: ChatCompletionRequest,
    apiKey: string,
    externalSignal?: AbortSignal,
  ): AsyncIterable<ChatCompletionChunk> {
    const level = extractThinkingLevel(request);
    const cleaned = stripThinking(request);
    const reasoningEffort = buildOpenAIThinking(level);
    const body = {
      ...maxTokensToCompletionTokens(cleaned),
      model,
      stream: true,
      ...(reasoningEffort ? { reasoning_effort: reasoningEffort } : {}),
    };
    const resp = await fetch(`${OPENAI_BASE}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(body),
      signal: withExternalSignal(remoteStreamTimeoutMs(), externalSignal),
    });

    if (!resp.ok || !resp.body) {
      const text = await resp.text().catch(() => "");
      throw classifyError(resp.status, text);
    }

    yield* parseSSEStream(resp, model);
  },
};

// ─── OpenRouter Adapter (OpenAI-compatible, supports streaming) ───

export const OpenRouterAdapter: ProviderAdapter = {
  name: "openrouter",

  async chatCompletion(
    model: string,
    request: ChatCompletionRequest,
    apiKey: string,
    externalSignal?: AbortSignal,
  ): Promise<ChatCompletionResponse> {
    const level = extractThinkingLevel(request);
    const cleaned = stripThinking(request);
    const reasoning = buildOpenRouterThinking(level);
    const body = { ...cleaned, model, stream: false, ...(reasoning ? { reasoning_effort: reasoning } : {}) };
    const resp = await fetch(`${OPENROUTER_BASE}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
        "HTTP-Referer": "https://github.com/dazwritescode/cognitive-router",
        "X-Title": "Cognitive Router",
      },
      body: JSON.stringify(body),
      signal: withExternalSignal(remoteTimeoutMs(), externalSignal),
    });

    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw classifyError(resp.status, text);
    }

    return (await resp.json()) as ChatCompletionResponse;
  },

  async* chatCompletionStream(
    model: string,
    request: ChatCompletionRequest,
    apiKey: string,
    externalSignal?: AbortSignal,
  ): AsyncIterable<ChatCompletionChunk> {
    const level = extractThinkingLevel(request);
    const cleaned = stripThinking(request);
    const reasoning = buildOpenRouterThinking(level);
    const body = { ...cleaned, model, stream: true, ...(reasoning ? { reasoning_effort: reasoning } : {}) };
    const resp = await fetch(`${OPENROUTER_BASE}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
        "HTTP-Referer": "https://github.com/dazwritescode/cognitive-router",
        "X-Title": "Cognitive Router",
      },
      body: JSON.stringify(body),
      signal: withExternalSignal(remoteStreamTimeoutMs(), externalSignal),
    });

    if (!resp.ok || !resp.body) {
      const text = await resp.text().catch(() => "");
      throw classifyError(resp.status, text);
    }

    yield* parseSSEStream(resp, model);
  },
};

// ─── Anthropic Adapter (native Messages API, supports streaming) ───
// Wire-format differences vs OpenAI (docs: platform.claude.com/en/api/messages):
//   - system prompt is a top-level `system` param, NOT a message role
//   - messages carry only user/assistant roles; tool results are user-role
//     content blocks (`tool_result`) keyed by tool_use_id
//   - tools use {name, description, input_schema} + tool_choice {auto|any|tool}
//   - max_tokens is REQUIRED (no server default)
//   - stop_reason: end_turn|stop_sequence→stop, max_tokens→length,
//     tool_use→tool_calls, refusal→content_filter
//   - thinking: {type:"adaptive"} + top-level effort (Claude 4.7+ rejects the
//     legacy `enabled`+budget_tokens form)
//   - streaming uses named SSE events (message_start, content_block_start,
//     content_block_delta, content_block_stop, message_delta, message_stop)

export const AnthropicAdapter: ProviderAdapter = {
  name: "anthropic",

  async chatCompletion(
    model: string,
    request: ChatCompletionRequest,
    apiKey: string,
    externalSignal?: AbortSignal,
  ): Promise<ChatCompletionResponse> {
    const { body, url } = buildAnthropicRequest(model, request, false);
    const resp = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify(body),
      signal: withExternalSignal(remoteTimeoutMs(), externalSignal),
    });

    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw classifyError(resp.status, text);
    }

    const data = await resp.json() as any;
    return anthropicToOpenAI(data, model);
  },

  async* chatCompletionStream(
    model: string,
    request: ChatCompletionRequest,
    apiKey: string,
    externalSignal?: AbortSignal,
  ): AsyncIterable<ChatCompletionChunk> {
    const { body, url } = buildAnthropicRequest(model, request, true);
    const resp = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify(body),
      signal: withExternalSignal(remoteStreamTimeoutMs(), externalSignal),
    });

    if (!resp.ok || !resp.body) {
      const text = await resp.text().catch(() => "");
      throw classifyError(resp.status, text);
    }

    yield* parseAnthropicSSEStream(resp, model);
  },
};

// ─── Gemini Adapter (Google Generative AI, stream via SSE) ───

export const GeminiAdapter: ProviderAdapter = {
  name: "gemini",

  async chatCompletion(
    model: string,
    request: ChatCompletionRequest,
    apiKey: string,
    externalSignal?: AbortSignal,
  ): Promise<ChatCompletionResponse> {
    // Check/create prefix cache for the system prompt
    await maybeGeminiCache(model, request.messages, apiKey);
    const { geminiBody, url } = buildGeminiRequest(model, request, apiKey, false);

    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(geminiBody),
      signal: withExternalSignal(remoteTimeoutMs(), externalSignal),
    });

    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw classifyError(resp.status, text);
    }

    const data = await resp.json() as any;
    return geminiToOpenAI(data, model);
  },

  async* chatCompletionStream(
    model: string,
    request: ChatCompletionRequest,
    apiKey: string,
    externalSignal?: AbortSignal,
  ): AsyncIterable<ChatCompletionChunk> {
    // Check/create prefix cache for the system prompt
    await maybeGeminiCache(model, request.messages, apiKey);
    const { geminiBody, url } = buildGeminiRequest(model, request, apiKey, true);

    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(geminiBody),
      signal: withExternalSignal(remoteStreamTimeoutMs(), externalSignal),
    });

    if (!resp.ok || !resp.body) {
      const text = await resp.text().catch(() => "");
      throw classifyError(resp.status, text);
    }

    // Gemini streaming returns JSON objects separated by newlines
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const id = generateId();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data: ") && !trimmed.startsWith("{")) continue;

        const jsonStr = trimmed.startsWith("data: ") ? trimmed.substring(6) : trimmed;
        if (jsonStr === "[DONE]") return;

        try {
          const data = JSON.parse(jsonStr);
          const parts = data?.candidates?.[0]?.content?.parts ?? [];
          const text = parts.map((p: any) => p.text ?? "").join("");
          const toolCalls = parts
            .filter((p: any) => p.functionCall?.name)
            .map((p: any, index: number) => ({
              id: `call_${Date.now()}_${index}`,
              type: "function",
              function: {
                name: p.functionCall.name,
                arguments: JSON.stringify(p.functionCall.args ?? {}),
              },
            }));

          if (text || toolCalls.length > 0) {
            yield {
              id,
              object: "chat.completion.chunk",
              created: Date.now(),
              model: `gemini/${model}`,
              choices: [{
                index: 0,
                delta: { ...(text ? { content: text } : {}), ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
                finish_reason: null,
              }],
            };
          }
        } catch {
          // Skip malformed
        }
      }
    }

    // Final chunk with finish_reason
    yield {
      id,
      object: "chat.completion.chunk",
      created: Date.now(),
      model: `gemini/${model}`,
      choices: [{
        index: 0,
        delta: {},
        finish_reason: "stop",
      }],
    };
  },
};

// ─── Ollama Adapter (local, OpenAI-compatible endpoint, supports streaming) ───

export const OllamaAdapter: ProviderAdapter = {
  name: "ollama",

  async chatCompletion(
    model: string,
    request: ChatCompletionRequest,
    _apiKey: string,
    externalSignal?: AbortSignal,
  ): Promise<ChatCompletionResponse> {
    const level = extractThinkingLevel(request);
    const cleaned = stripThinking(request);
    const resp = await fetch(`${OLLAMA_BASE}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...cleaned, model, stream: false, think: level !== "none" }),
      signal: withExternalSignal(localTimeoutMs(), externalSignal),
    });

    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw classifyError(resp.status, text);
    }

    return (await resp.json()) as ChatCompletionResponse;
  },

  async* chatCompletionStream(
    model: string,
    request: ChatCompletionRequest,
    _apiKey: string,
    externalSignal?: AbortSignal,
  ): AsyncIterable<ChatCompletionChunk> {
    const level = extractThinkingLevel(request);
    const cleaned = stripThinking(request);
    const resp = await fetch(`${OLLAMA_BASE}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...cleaned, model, stream: true, think: level !== "none" }),
      signal: withExternalSignal(localStreamTimeoutMs(), externalSignal),
    });

    if (!resp.ok || !resp.body) {
      const text = await resp.text().catch(() => "");
      throw classifyError(resp.status, text);
    }

    yield* parseSSEStream(resp, model);
  },
};

// ─── OpenAI Helpers ───

/** Translate max_tokens → max_completion_tokens. OpenAI deprecated max_tokens
 *  and rejects it on o-series reasoning models; max_completion_tokens is
 *  accepted across the current lineup. An explicit max_completion_tokens
 *  on the inbound request wins untouched. */
function maxTokensToCompletionTokens(request: ChatCompletionRequest): ChatCompletionRequest {
  if (request.max_tokens === undefined || request.max_completion_tokens !== undefined) return request;
  const { max_tokens, ...rest } = request;
  return { ...rest, max_completion_tokens: max_tokens } as ChatCompletionRequest;
}

// ─── Anthropic Helpers ───

function extractTextContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((b: any) => (typeof b === "string" ? b : b?.type === "text" ? b.text ?? "" : ""))
      .filter(Boolean)
      .join("");
  }
  return "";
}

function mapAnthropicStopReason(stopReason: string | undefined): string {
  switch (stopReason) {
    case "tool_use": return "tool_calls";
    case "max_tokens":
    case "model_context_window_exceeded": return "length";
    case "refusal": return "content_filter";
    default: return "stop"; // end_turn | stop_sequence | unknown
  }
}

/** Convert OpenAI-style content (string or parts array) to Anthropic content
 *  blocks. Text parts become text blocks; image_url parts become image blocks
 *  (base64 data URIs decode to the base64 source, plain URLs use the URL
 *  source — both documented ImageBlockParam forms). */
function openAIContentToAnthropicBlocks(content: any): any[] {
  if (typeof content === "string") {
    return [{ type: "text", text: content }];
  }
  if (!Array.isArray(content)) {
    return [{ type: "text", text: String(content ?? "") }];
  }
  const blocks: any[] = [];
  for (const part of content) {
    if (typeof part === "string") {
      blocks.push({ type: "text", text: part });
    } else if (part?.type === "text" && part.text) {
      blocks.push({ type: "text", text: part.text });
    } else if (part?.type === "image_url" && part.image_url?.url) {
      const url: string = part.image_url.url;
      const dataUri = url.match(/^data:(image\/[a-z+]+);base64,(.*)$/s);
      if (dataUri) {
        blocks.push({ type: "image", source: { type: "base64", media_type: dataUri[1], data: dataUri[2] } });
      } else {
        blocks.push({ type: "image", source: { type: "url", url } });
      }
    }
  }
  return blocks.length > 0 ? blocks : [{ type: "text", text: "" }];
}

/** Convert OpenAI chat messages (minus system) to Anthropic message params. */
function openAIToAnthropicMessages(messages: ChatMessage[]): any[] {
  const out: any[] = [];
  for (const m of messages) {
    if (m.role === "tool" || m.role === "function") {
      // Tool results ride on user-role tool_result blocks keyed by tool_use_id
      // (OpenAI's equivalent field is tool_call_id).
      out.push({
        role: "user",
        content: [{
          type: "tool_result",
          tool_use_id: m.tool_call_id ?? m.name ?? "tool",
          content: extractTextContent(m.content),
        }],
      });
      continue;
    }
    if (m.role === "assistant") {
      const blocks: any[] = [];
      const text = extractTextContent(m.content);
      if (text) blocks.push({ type: "text", text });
      for (const toolCall of m.tool_calls ?? []) {
        const fn = toolCall?.function;
        if (!fn?.name) continue;
        blocks.push({
          type: "tool_use",
          id: toolCall.id ?? `call_${Date.now()}_${blocks.length}`,
          name: fn.name,
          input: parseFunctionArguments(fn.arguments),
        });
      }
      out.push({ role: "assistant", content: blocks.length > 0 ? blocks : [{ type: "text", text: "" }] });
      continue;
    }
    out.push({ role: "user", content: openAIContentToAnthropicBlocks(m.content) });
  }
  return out;
}

/** Convert OpenAI tools/functions to Anthropic tool definitions. */
function openAIToolsToAnthropic(request: ChatCompletionRequest): any[] {
  const tools: any[] = [];
  for (const tool of request.tools ?? []) {
    const fn = tool?.function;
    if (!fn?.name) continue;
    tools.push({ name: fn.name, description: fn.description ?? "", input_schema: fn.parameters ?? { type: "object", properties: {} } });
  }
  for (const fn of request.functions ?? []) {
    if (!fn?.name) continue;
    tools.push({ name: fn.name, description: fn.description ?? "", input_schema: fn.parameters ?? { type: "object", properties: {} } });
  }
  return tools;
}

function buildAnthropicRequest(
  model: string,
  request: ChatCompletionRequest,
  stream: boolean,
): { body: any; url: string } {
  const level = extractThinkingLevel(request);
  const cleaned = stripThinking(request);
  const thinkingFields = buildAnthropicThinking(level);

  const systemPrompt = request.messages
    .filter((m) => m.role === "system")
    .map((m) => extractTextContent(m.content))
    .filter(Boolean)
    .join("\n\n");
  const messages = openAIToAnthropicMessages(
    request.messages.filter((m) => m.role !== "system"),
  );

  const body: any = {
    model,
    // Anthropic requires max_tokens; OpenAI-compatible requests may omit it.
    max_tokens: request.max_tokens ?? 8192,
    messages,
    ...(systemPrompt ? { system: systemPrompt } : {}),
    ...(stream ? { stream: true } : {}),
    // Anthropic temperature range is 0-1 (OpenAI allows up to 2) — clamp.
    ...(request.temperature !== undefined ? { temperature: Math.min(Math.max(request.temperature, 0), 1) } : {}),
    ...(request.top_p !== undefined ? { top_p: request.top_p } : {}),
    ...(cleaned.stop !== undefined
      ? { stop_sequences: Array.isArray(cleaned.stop) ? cleaned.stop : [cleaned.stop] }
      : {}),
    ...thinkingFields,
  };

  const tools = openAIToolsToAnthropic(request);
  if (tools.length > 0) {
    body.tools = tools;
    // Anthropic tool_choice: {auto|any|tool{name}} — no "none" (omit tools
    // entirely for that semantic instead).
    if (request.tool_choice === "auto") body.tool_choice = { type: "auto" };
    else if (request.tool_choice === "required") body.tool_choice = { type: "any" };
    else if (typeof request.tool_choice === "object" && (request.tool_choice as any)?.function?.name) {
      body.tool_choice = { type: "tool", name: (request.tool_choice as any).function.name };
    }
  }

  return { body, url: `${ANTHROPIC_BASE}/v1/messages` };
}

function anthropicToOpenAI(data: any, model: string): ChatCompletionResponse {
  const blocks: any[] = Array.isArray(data?.content) ? data.content : [];
  const text = blocks.filter((b) => b?.type === "text").map((b) => b.text ?? "").join("");
  const thinking = blocks.filter((b) => b?.type === "thinking").map((b) => b.thinking ?? "").join("");
  const toolCalls = blocks
    .filter((b) => b?.type === "tool_use")
    .map((b, index) => ({
      id: b.id ?? `call_${Date.now()}_${index}`,
      type: "function",
      function: { name: b.name, arguments: JSON.stringify(b.input ?? {}) },
    }));

  const usage = data?.usage;
  const promptTokens = (usage?.input_tokens ?? 0) + (usage?.cache_read_input_tokens ?? 0) + (usage?.cache_creation_input_tokens ?? 0);

  return {
    id: data?.id ?? generateId(),
    object: "chat.completion",
    created: Date.now(),
    model: `anthropic/${model}`,
    choices: [{
      index: 0,
      message: {
        role: "assistant",
        content: text,
        ...(thinking ? { reasoning_content: thinking } : {}),
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      },
      finish_reason: mapAnthropicStopReason(data?.stop_reason),
    }],
    usage: usage ? {
      prompt_tokens: promptTokens,
      completion_tokens: usage.output_tokens ?? 0,
      total_tokens: promptTokens + (usage.output_tokens ?? 0),
    } : undefined,
  };
}

/** Parse Anthropic's named-event SSE stream into OpenAI-style chunks.
 *  Event flow (docs: Streaming messages): message_start → per-block
 *  content_block_start/content_block_delta/content_block_stop →
 *  message_delta (stop_reason + CUMULATIVE usage) → message_stop. */
async function* parseAnthropicSSEStream(
  response: Response,
  model: string,
): AsyncIterable<ChatCompletionChunk> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";
  const id = generateId();
  // content-block index → tool_calls array index (text blocks skip indexing)
  const toolCallIndexByBlock = new Map<number, number>();
  let toolCallCount = 0;
  let finishReason = "stop";
  let usage: any;

  const makeChunk = (
    delta: Record<string, unknown>,
    finish: string | null,
  ): ChatCompletionChunk => ({
    id,
    object: "chat.completion.chunk",
    created: Date.now(),
    model: `anthropic/${model}`,
    choices: [{ index: 0, delta: delta as any, finish_reason: finish }],
  });

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith(":") || trimmed.startsWith("event:")) continue;
      if (!trimmed.startsWith("data: ")) continue;

      const data = trimmed.substring(6);
      if (data === "[DONE]") return;

      let ev: any;
      try {
        ev = JSON.parse(data);
      } catch {
        continue; // skip malformed events
      }
      const blockIndex: number = ev.index ?? 0;

      switch (ev.type) {
        case "message_start":
          yield makeChunk({ role: "assistant", content: "" }, null);
          break;

        case "content_block_start":
          if (ev.content_block?.type === "tool_use") {
            const tcIndex = toolCallCount++;
            toolCallIndexByBlock.set(blockIndex, tcIndex);
            yield makeChunk({
              tool_calls: [{
                index: tcIndex,
                id: ev.content_block.id,
                type: "function",
                function: { name: ev.content_block.name, arguments: "" },
              }],
            }, null);
          }
          break;

        case "content_block_delta": {
          const delta = ev.delta ?? {};
          if (delta.type === "text_delta") {
            yield makeChunk({ content: delta.text ?? "" }, null);
          } else if (delta.type === "thinking_delta") {
            yield makeChunk({ reasoning_content: delta.thinking ?? "" }, null);
          } else if (delta.type === "input_json_delta") {
            yield makeChunk({
              tool_calls: [{
                index: toolCallIndexByBlock.get(blockIndex) ?? 0,
                function: { arguments: delta.partial_json ?? "" },
              }],
            }, null);
          }
          break;
        }

        case "message_delta":
          if (ev.delta?.stop_reason) finishReason = mapAnthropicStopReason(ev.delta.stop_reason);
          if (ev.usage) usage = ev.usage; // cumulative
          break;

        case "message_stop": {
          const chunk = makeChunk({}, finishReason);
          if (usage) {
            const promptTokens = (usage.input_tokens ?? 0) + (usage.cache_read_input_tokens ?? 0) + (usage.cache_creation_input_tokens ?? 0);
            (chunk as any).usage = {
              prompt_tokens: promptTokens,
              completion_tokens: usage.output_tokens ?? 0,
              total_tokens: promptTokens + (usage.output_tokens ?? 0),
            };
          }
          yield chunk;
          return;
        }

        case "error":
          throw classifyError(502, JSON.stringify(ev.error ?? ev));
      }
    }
  }

  // Stream ended without message_stop — still emit a terminal chunk.
  yield makeChunk({}, finishReason);
}

// ─── Gemini Helpers ───

function buildGeminiRequest(
  model: string,
  request: ChatCompletionRequest,
  apiKey: string,
  stream: boolean,
): { geminiBody: any; url: string } {
  const level = extractThinkingLevel(request);
  const thinkingConfig = buildGeminiThinking(level);

  const contents = request.messages
    .filter((m) => m.role !== "system")
    .map((m) => openAIMessageToGeminiContent(m));

  const systemPrompt = request.messages
    .filter((m) => m.role === "system")
    .map((m) => m.content)
    .join("\n\n");

  const geminiBody: any = {
    contents,
    generationConfig: {
      temperature: request.temperature ?? 0.7,
      maxOutputTokens: request.max_tokens ?? 8192,
      thinkingConfig,
    },
  };

  // Check for an existing Gemini cachedContent for this system prompt.
  // If one exists, reference it via cachedContent field and skip systemInstruction
  // (the cached content already includes it).
  const existingCacheName = PrefixCache.instance.getGeminiCacheName(systemPrompt);
  if (existingCacheName) {
    geminiBody.cachedContent = existingCacheName;
    PrefixCache.instance.recordGeminiCacheHit();
    logger.debug(`Gemini: using cachedContent ${existingCacheName} for system prompt`);
    // Don't include systemInstruction — it's in the cached content
  } else if (systemPrompt) {
    geminiBody.systemInstruction = { parts: [{ text: systemPrompt }] };
  }

  const functionDeclarations = openAIToolsToGeminiFunctionDeclarations(request);
  if (functionDeclarations.length > 0) {
    geminiBody.tools = [{ functionDeclarations }];
    geminiBody.toolConfig = {
      functionCallingConfig: {
        mode: request.tool_choice === "none" ? "NONE" : request.tool_choice === "required" ? "ANY" : "AUTO",
      },
    };
  }

  const endpoint = stream ? "streamGenerateContent" : "generateContent";
  const url = `${GEMINI_BASE}/models/${model}:${endpoint}?key=${apiKey}${stream ? "&alt=sse" : ""}`;

  return { geminiBody, url };
}

function openAIMessageToGeminiContent(message: ChatMessage): any {
  if (message.role === "tool" || message.role === "function") {
    return {
      role: "user",
      parts: [{
        functionResponse: {
          name: message.name ?? "tool_response",
          response: { result: message.content ?? "" },
        },
      }],
    };
  }

  const parts: any[] = [];
  const content = message.content as any;
  if (content) {
    // Handle both string content and array content (OpenAI vision format)
    if (typeof content === "string") {
      parts.push({ text: content });
    } else if (Array.isArray(content)) {
      // Extract text from content blocks
      const textParts = content
        .filter((block: any) => typeof block === "string" || block?.type === "text")
        .map((block: any) => typeof block === "string" ? block : block.text ?? "");
      if (textParts.length > 0) {
        parts.push({ text: textParts.join("\n") });
      }
      // Note: image content blocks are not handled here yet
    } else if (typeof content === "object") {
      parts.push({ text: String(content) });
    }
  }

  if (Array.isArray(message.tool_calls)) {
    for (const toolCall of message.tool_calls) {
      const fn = toolCall?.function;
      if (!fn?.name) continue;
      parts.push({
        functionCall: {
          name: fn.name,
          args: parseFunctionArguments(fn.arguments),
        },
      });
    }
  }

  if (parts.length === 0) {
    parts.push({ text: "" });
  }

  return {
    role: message.role === "assistant" ? "model" : "user",
    parts,
  };
}

/**
 * Recursively sanitize a JSON Schema object for Gemini's function declaration API.
 *
 * Gemini's GenerateContent API accepts a subset of OpenAPI 3.0 schema fields.
 * It rejects standard JSON Schema draft-07 fields like `$schema`, `$ref`, `$defs`,
 * and `additionalProperties`, returning a 400 "Invalid JSON payload" error.
 *
 * Supported fields (per Google's documentation + empirical testing):
 *   - type, description, properties, required, enum, items, format
 *
 * Stripped fields:
 *   - $schema, $ref, $defs, $id, $comment, additionalProperties,
 *     default, examples, readOnly, writeOnly, deprecated, contentEncoding,
 *     contentMediaType, pattern (Gemini ignores regex constraints)
 */
const GEMINI_UNSUPPORTED_KEYS = new Set([
  "$schema", "$ref", "$defs", "$id", "$comment", "additionalProperties",
  "default", "examples", "readOnly", "writeOnly", "deprecated",
  "contentEncoding", "contentMediaType", "pattern",
  // JSON Schema composition keywords — Gemini doesn't support these
  "anyOf", "oneOf", "allOf", "not", "const",
  "if", "then", "else",
  // Other unsupported validation keywords
  "minProperties", "maxProperties", "minItems", "maxItems",
  "uniqueItems", "multipleOf", "exclusiveMinimum", "exclusiveMaximum",
]);

/**
 * Convert anyOf/oneOf/allOf schemas to a flattened form Gemini can understand.
 * Gemini only supports: type, description, properties, required, enum, items, format.
 * When we encounter anyOf/oneOf, we flatten to the first type option (best effort).
 */
function flattenForGemini(schema: any): any {
  if (!schema || typeof schema !== "object") return schema;

  // Handle anyOf/oneOf: pick the first non-null option as the type
  const union = schema.anyOf ?? schema.oneOf;
  if (Array.isArray(union)) {
    const nonNull = union.filter((o: any) => o && o.type !== "null");
    const picked = nonNull[0] ?? union[0];
    if (picked) {
      // Merge description from parent
      const merged = { ...picked };
      if (schema.description && !merged.description) {
        merged.description = schema.description;
      }
      return flattenForGemini(merged);
    }
    return { type: "string" }; // safe fallback
  }

  // Handle allOf: merge all schemas together
  if (Array.isArray(schema.allOf)) {
    const merged: any = { type: "object", properties: {} };
    for (const sub of schema.allOf) {
      const flat = flattenForGemini(sub);
      if (flat.properties) Object.assign(merged.properties, flat.properties);
      if (flat.required) {
        merged.required = [...(merged.required ?? []), ...flat.required];
      }
      if (flat.type && merged.type === "object" && flat.type !== "object") {
        merged.type = flat.type;
      }
    }
    if (schema.description) merged.description = schema.description;
    return merged;
  }

  return schema;
}

function sanitizeSchemaForGemini(schema: any): any {
  if (!schema || typeof schema !== "object") return schema;
  if (Array.isArray(schema)) return schema.map(sanitizeSchemaForGemini);

  // First, flatten any anyOf/oneOf/allOf into Gemini-compatible types
  const flattened = flattenForGemini(schema);
  if (flattened !== schema) {
    return sanitizeSchemaForGemini(flattened);
  }

  const cleaned: Record<string, any> = {};
  for (const [key, value] of Object.entries(schema)) {
    if (GEMINI_UNSUPPORTED_KEYS.has(key)) continue;
    if (key === "properties" && value && typeof value === "object") {
      const cleanedProps: Record<string, any> = {};
      for (const [propName, propSchema] of Object.entries(value)) {
        cleanedProps[propName] = sanitizeSchemaForGemini(propSchema);
      }
      cleaned[key] = cleanedProps;
    } else if (key === "items") {
      cleaned[key] = sanitizeSchemaForGemini(value);
    } else if (value && typeof value === "object") {
      cleaned[key] = sanitizeSchemaForGemini(value);
    } else {
      cleaned[key] = value;
    }
  }
  return cleaned;
}

function openAIToolsToGeminiFunctionDeclarations(request: ChatCompletionRequest): any[] {
  const declarations: any[] = [];

  for (const tool of request.tools ?? []) {
    const fn = tool?.function;
    if (!fn?.name) continue;
    declarations.push({
      name: fn.name,
      description: fn.description ?? "",
      parameters: sanitizeSchemaForGemini(fn.parameters ?? { type: "object", properties: {} }),
    });
  }

  for (const fn of request.functions ?? []) {
    if (!fn?.name) continue;
    declarations.push({
      name: fn.name,
      description: fn.description ?? "",
      parameters: sanitizeSchemaForGemini(fn.parameters ?? { type: "object", properties: {} }),
    });
  }

  return declarations;
}

function geminiToOpenAI(data: any, model: string): ChatCompletionResponse {
  const candidate = data.candidates?.[0];
  const parts = candidate?.content?.parts ?? [];
  const text = parts.map((p: any) => p.text ?? "").join("");
  const toolCalls = parts
    .filter((p: any) => p.functionCall?.name)
    .map((p: any, index: number) => ({
      id: `call_${Date.now()}_${index}`,
      type: "function",
      function: {
        name: p.functionCall.name,
        arguments: JSON.stringify(p.functionCall.args ?? {}),
      },
    }));
  const finishReason = toolCalls.length > 0
    ? "tool_calls"
    : candidate?.finishReason === "STOP" ? "stop" : candidate?.finishReason?.toLowerCase() ?? "stop";

  return {
    id: generateId(),
    object: "chat.completion",
    created: Date.now(),
    model: `gemini/${model}`,
    choices: [{
      index: 0,
      message: { role: "assistant", content: text, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
      finish_reason: finishReason,
    }],
    usage: data.usageMetadata ? {
      prompt_tokens: data.usageMetadata.promptTokenCount ?? 0,
      completion_tokens: data.usageMetadata.candidatesTokenCount ?? 0,
      total_tokens: data.usageMetadata.totalTokenCount ?? 0,
    } : undefined,
  };
}

// ─── Registry ───

export const PROVIDERS: Record<string, ProviderAdapter> = {
  zai: ZAIAdapter,
  openai: OpenAIAdapter,
  openrouter: OpenRouterAdapter,
  anthropic: AnthropicAdapter,
  gemini: GeminiAdapter,
  ollama: OllamaAdapter,
};

export function getProvider(name: string): ProviderAdapter | undefined {
  return PROVIDERS[name];
}
