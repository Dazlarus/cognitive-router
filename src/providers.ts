// src/providers.ts — Provider adapters
// Each adapter translates an OpenAI-compatible request to the provider's native API format
// and returns an OpenAI-compatible response. Supports both non-streaming and streaming.

import { logger } from "./logger.js";

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
  [key: string]: any;
}

// Thinking/Reasoning helpers — translate OpenClaw's thinking param to each provider's format

function extractThinkingLevel(request: ChatCompletionRequest): "none" | "low" | "medium" | "high" {
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

// ─── Provider Base URLs (env-configurable) ───

function providerBaseUrl(envVar: string, fallback: string): string {
  const raw = process.env[envVar];
  if (!raw) return fallback;
  // Strip trailing slash so callers can append "/chat/completions"
  return raw.replace(/\/+$/, "");
}

const ZAI_BASE = providerBaseUrl("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4");
const OPENROUTER_BASE = providerBaseUrl("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1");
const GEMINI_BASE = providerBaseUrl("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta");
const OLLAMA_BASE = providerBaseUrl("OLLAMA_BASE_URL", "http://localhost:11434/v1");

logger.info(`Provider endpoints: zai=${ZAI_BASE} openrouter=${OPENROUTER_BASE} gemini=${GEMINI_BASE} ollama=${OLLAMA_BASE}`);

function buildZaiThinking(level: string): any {
  if (level === "none") return undefined;
  const budgets: Record<string, number> = { low: 2048, medium: 8192, high: 32768 };
  return { type: "enabled", budget_tokens: budgets[level] ?? 8192 };
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
  const { thinking, reasoning, reasoning_effort, ...rest } = request;
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
  ): Promise<ChatCompletionResponse>;

  /** Send a streaming chat completion request. Yields chunks. Throws on failure. */
  chatCompletionStream(
    model: string,
    request: ChatCompletionRequest,
    apiKey: string,
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
  ): Promise<ChatCompletionResponse> {
    const level = extractThinkingLevel(request);
    const cleaned = stripThinking(request);
    const thinking = buildZaiThinking(level);
    const body = { ...cleaned, model, stream: false, ...(thinking ? { thinking } : {}) };
    const resp = await fetch(`${ZAI_BASE}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(remoteTimeoutMs()),
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
  ): AsyncIterable<ChatCompletionChunk> {
    const level = extractThinkingLevel(request);
    const cleaned = stripThinking(request);
    const thinking = buildZaiThinking(level);
    const body = { ...cleaned, model, stream: true, ...(thinking ? { thinking } : {}) };
    const resp = await fetch(`${ZAI_BASE}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(remoteStreamTimeoutMs()),
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
      signal: AbortSignal.timeout(remoteTimeoutMs()),
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
      signal: AbortSignal.timeout(remoteStreamTimeoutMs()),
    });

    if (!resp.ok || !resp.body) {
      const text = await resp.text().catch(() => "");
      throw classifyError(resp.status, text);
    }

    yield* parseSSEStream(resp, model);
  },
};

// ─── Gemini Adapter (Google Generative AI, stream via SSE) ───

export const GeminiAdapter: ProviderAdapter = {
  name: "gemini",

  async chatCompletion(
    model: string,
    request: ChatCompletionRequest,
    apiKey: string,
  ): Promise<ChatCompletionResponse> {
    const { geminiBody, url } = buildGeminiRequest(model, request, apiKey, false);

    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(geminiBody),
      signal: AbortSignal.timeout(remoteTimeoutMs()),
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
  ): AsyncIterable<ChatCompletionChunk> {
    const { geminiBody, url } = buildGeminiRequest(model, request, apiKey, true);

    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(geminiBody),
      signal: AbortSignal.timeout(remoteStreamTimeoutMs()),
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
  ): Promise<ChatCompletionResponse> {
    const level = extractThinkingLevel(request);
    const cleaned = stripThinking(request);
    const resp = await fetch(`${OLLAMA_BASE}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...cleaned, model, stream: false, think: level !== "none" }),
      signal: AbortSignal.timeout(localTimeoutMs()),
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
  ): AsyncIterable<ChatCompletionChunk> {
    const level = extractThinkingLevel(request);
    const cleaned = stripThinking(request);
    const resp = await fetch(`${OLLAMA_BASE}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...cleaned, model, stream: true, think: level !== "none" }),
      signal: AbortSignal.timeout(localStreamTimeoutMs()),
    });

    if (!resp.ok || !resp.body) {
      const text = await resp.text().catch(() => "");
      throw classifyError(resp.status, text);
    }

    yield* parseSSEStream(resp, model);
  },
};

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

  if (systemPrompt) {
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
  if (message.content) {
    parts.push({ text: message.content });
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

function openAIToolsToGeminiFunctionDeclarations(request: ChatCompletionRequest): any[] {
  const declarations: any[] = [];

  for (const tool of request.tools ?? []) {
    const fn = tool?.function;
    if (!fn?.name) continue;
    declarations.push({
      name: fn.name,
      description: fn.description ?? "",
      parameters: fn.parameters ?? { type: "object", properties: {} },
    });
  }

  for (const fn of request.functions ?? []) {
    if (!fn?.name) continue;
    declarations.push({
      name: fn.name,
      description: fn.description ?? "",
      parameters: fn.parameters ?? { type: "object", properties: {} },
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
  openrouter: OpenRouterAdapter,
  gemini: GeminiAdapter,
  ollama: OllamaAdapter,
};

export function getProvider(name: string): ProviderAdapter | undefined {
  return PROVIDERS[name];
}
