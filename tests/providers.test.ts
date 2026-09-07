// tests/providers.test.ts — Unit tests for providers.ts
// Run with: npx tsx --test tests/providers.test.ts

import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import {
  PROVIDERS,
  getProvider,
  ZAIAdapter,
  OpenAIAdapter,
  AnthropicAdapter,
  OpenRouterAdapter,
  GeminiAdapter,
  OllamaAdapter,
  type ChatCompletionRequest,
} from "../src/providers.ts";

// ─── Helpers ────────────────────────────────────────────────

/** Capture the fetch call's URL and body for assertion. */
function makeFetchCapture(): {
  fetchMock: typeof fetch;
  calls: Array<{ url: string; init?: RequestInit; body: any }>;
} {
  const calls: Array<{ url: string; init?: RequestInit; body: any }> = [];
  const fetchMock = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = String(input);
    let parsedBody: any = null;
    if (init?.body) {
      try {
        parsedBody = JSON.parse(String(init.body));
      } catch {
        parsedBody = String(init.body);
      }
    }
    calls.push({ url, init, body: parsedBody });

    // Default success response
    return new Response(JSON.stringify({
      id: "chatcmpl-test",
      object: "chat.completion",
      created: Date.now(),
      model: parsedBody?.model ?? "test-model",
      choices: [{
        index: 0,
        message: { role: "assistant", content: "test response" },
        finish_reason: "stop",
      }],
    }), { status: 200 });
  }) as typeof fetch;

  return { fetchMock, calls };
}

/** Create a mock SSE stream response. */
function makeSSEResponse(chunks: any[]): Response {
  const sseBody = chunks
    .map((chunk) => `data: ${JSON.stringify(chunk)}\n`)
    .join("\n") + "data: [DONE]\n\n";

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(sseBody));
      controller.close();
    },
  });

  return new Response(stream, { status: 200 });
}

/** Create a mock Gemini streaming response (JSON objects separated by newlines with data: prefix). */
function makeGeminiSSEResponse(parts: any[][]): Response {
  const chunks = parts.map((p) => ({
    candidates: [{
      content: { parts: p },
      finishReason: "STOP",
    }],
  }));

  const sseBody = chunks
    .map((chunk) => `data: ${JSON.stringify(chunk)}\n`)
    .join("\n") + "data: [DONE]\n\n";

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(sseBody));
      controller.close();
    },
  });

  return new Response(stream, { status: 200 });
}

/** Create a mock Anthropic SSE stream response (named events). */
function makeAnthropicSSEResponse(events: any[]): Response {
  const sseBody = events
    .map((ev) => `event: ${ev.type}\ndata: ${JSON.stringify(ev)}\n\n`)
    .join("") + "data: [DONE]\n\n";

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(sseBody));
      controller.close();
    },
  });

  return new Response(stream, { status: 200 });
}

// ─── Tests ──────────────────────────────────────────────────

describe("Provider Registry", () => {
  it("should export all six providers", () => {
    assert.ok(PROVIDERS.zai, "ZAI provider should exist");
    assert.ok(PROVIDERS.openai, "OpenAI provider should exist");
    assert.ok(PROVIDERS.openrouter, "OpenRouter provider should exist");
    assert.ok(PROVIDERS.anthropic, "Anthropic provider should exist");
    assert.ok(PROVIDERS.gemini, "Gemini provider should exist");
    assert.ok(PROVIDERS.ollama, "Ollama provider should exist");
  });

  it("getProvider should return the correct adapter by name", () => {
    assert.equal(getProvider("zai"), ZAIAdapter);
    assert.equal(getProvider("openai"), OpenAIAdapter);
    assert.equal(getProvider("openrouter"), OpenRouterAdapter);
    assert.equal(getProvider("anthropic"), AnthropicAdapter);
    assert.equal(getProvider("gemini"), GeminiAdapter);
    assert.equal(getProvider("ollama"), OllamaAdapter);
  });

  it("getProvider should return undefined for unknown provider", () => {
    assert.equal(getProvider("nonexistent"), undefined);
  });

  it("each adapter should have a name property", () => {
    assert.equal(ZAIAdapter.name, "zai");
    assert.equal(OpenAIAdapter.name, "openai");
    assert.equal(OpenRouterAdapter.name, "openrouter");
    assert.equal(AnthropicAdapter.name, "anthropic");
    assert.equal(GeminiAdapter.name, "gemini");
    assert.equal(OllamaAdapter.name, "ollama");
  });
});

// ─── Gemini Schema Sanitizer Tests ──────────────────────────
// sanitizeSchemaForGemini is not exported, so we test it indirectly
// through the Gemini adapter's request building (tools → functionDeclarations).

describe("Gemini Schema Sanitizer (indirect via adapter)", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("should strip unsupported JSON Schema keys from tool parameters", async () => {
    const { fetchMock, calls } = makeFetchCapture();
    globalThis.fetch = fetchMock;

    const request: ChatCompletionRequest = {
      model: "gemini-2.5-flash",
      messages: [{ role: "user", content: "test" }],
      tools: [{
        type: "function",
        function: {
          name: "test_tool",
          description: "A test tool",
          parameters: {
            type: "object",
            properties: {
              foo: { type: "string", description: "foo field" },
            },
            // Unsupported keys that should be stripped:
            $schema: "https://json-schema.org/draft-07/schema#",
            $ref: "#/$defs/foo",
            $defs: { foo: { type: "string" } },
            $id: "urn:test",
            $comment: "this is a comment",
            additionalProperties: false,
            default: {},
            examples: [{ foo: "bar" }],
            readOnly: ["foo"],
            writeOnly: [],
            deprecated: true,
            contentEncoding: "utf-8",
            contentMediaType: "application/json",
            pattern: "^[a-z]+$",
          },
        },
      }],
    };

    await GeminiAdapter.chatCompletion("gemini-2.5-flash", request, "test-key");

    const call = calls[0];
    assert.ok(call.body.tools, "Gemini request should include tools");
    assert.ok(call.body.tools[0].functionDeclarations, "Should have functionDeclarations");

    const params = call.body.tools[0].functionDeclarations[0].parameters;

    // Stripped keys should NOT exist
    assert.equal(params.$schema, undefined, "$schema should be stripped");
    assert.equal(params.$ref, undefined, "$ref should be stripped");
    assert.equal(params.$defs, undefined, "$defs should be stripped");
    assert.equal(params.$id, undefined, "$id should be stripped");
    assert.equal(params.$comment, undefined, "$comment should be stripped");
    assert.equal(params.additionalProperties, undefined, "additionalProperties should be stripped");
    assert.equal(params.default, undefined, "default should be stripped");
    assert.equal(params.examples, undefined, "examples should be stripped");
    assert.equal(params.readOnly, undefined, "readOnly should be stripped");
    assert.equal(params.writeOnly, undefined, "writeOnly should be stripped");
    assert.equal(params.deprecated, undefined, "deprecated should be stripped");
    assert.equal(params.contentEncoding, undefined, "contentEncoding should be stripped");
    assert.equal(params.contentMediaType, undefined, "contentMediaType should be stripped");
    assert.equal(params.pattern, undefined, "pattern should be stripped");
  });

  it("should preserve supported schema keys", async () => {
    const { fetchMock, calls } = makeFetchCapture();
    globalThis.fetch = fetchMock;

    const request: ChatCompletionRequest = {
      model: "gemini-2.5-flash",
      messages: [{ role: "user", content: "test" }],
      tools: [{
        type: "function",
        function: {
          name: "test_tool",
          description: "A test tool",
          parameters: {
            type: "object",
            description: "Test parameters",
            properties: {
              color: {
                type: "string",
                description: "Color value",
                enum: ["red", "green", "blue"],
              },
              count: {
                type: "integer",
                format: "int32",
                description: "Number of items",
              },
              items: {
                type: "array",
                items: { type: "string", description: "An item" },
              },
            },
            required: ["color"],
          },
        },
      }],
    };

    await GeminiAdapter.chatCompletion("gemini-2.5-flash", request, "test-key");

    const params = calls[0].body.tools[0].functionDeclarations[0].parameters;

    // Preserved keys
    assert.equal(params.type, "object", "type should be preserved");
    assert.equal(params.description, "Test parameters", "description should be preserved");
    assert.ok(params.properties, "properties should be preserved");
    assert.ok(params.required, "required should be preserved");
    assert.deepEqual(params.required, ["color"]);

    // Nested properties should also be preserved
    assert.equal(params.properties.color.type, "string");
    assert.equal(params.properties.color.description, "Color value");
    assert.deepEqual(params.properties.color.enum, ["red", "green", "blue"]);

    assert.equal(params.properties.count.type, "integer");
    assert.equal(params.properties.count.format, "int32");

    // items recursion
    assert.equal(params.properties.items.type, "array");
    assert.equal(params.properties.items.items.type, "string");
    assert.equal(params.properties.items.items.description, "An item");
  });

  it("should sanitize deeply nested structures", async () => {
    const { fetchMock, calls } = makeFetchCapture();
    globalThis.fetch = fetchMock;

    const request: ChatCompletionRequest = {
      model: "gemini-2.5-flash",
      messages: [{ role: "user", content: "test" }],
      tools: [{
        type: "function",
        function: {
          name: "nested_tool",
          description: "Deeply nested",
          parameters: {
            type: "object",
            properties: {
              level1: {
                type: "object",
                properties: {
                  level2: {
                    type: "object",
                    properties: {
                      level3: {
                        type: "string",
                        description: "deep field",
                        additionalProperties: false,
                        pattern: "^[a-z]+$",
                        default: "hello",
                      },
                    },
                    additionalProperties: true,
                  },
                },
                additionalProperties: false,
              },
            },
          },
        },
      }],
    };

    await GeminiAdapter.chatCompletion("gemini-2.5-flash", request, "test-key");

    const params = calls[0].body.tools[0].functionDeclarations[0].parameters;
    const l1 = params.properties.level1;
    const l2 = l1.properties.level2;
    const l3 = l2.properties.level3;

    // Supported keys preserved at all levels
    assert.equal(l1.type, "object");
    assert.equal(l2.type, "object");
    assert.equal(l3.type, "string");
    assert.equal(l3.description, "deep field");

    // Unsupported keys stripped at all levels
    assert.equal(l1.additionalProperties, undefined, "level1 additionalProperties stripped");
    assert.equal(l2.additionalProperties, undefined, "level2 additionalProperties stripped");
    assert.equal(l3.additionalProperties, undefined, "level3 additionalProperties stripped");
    assert.equal(l3.pattern, undefined, "level3 pattern stripped");
    assert.equal(l3.default, undefined, "level3 default stripped");
  });
});

// ─── Gemini Adapter Request Building ────────────────────────

describe("Gemini Adapter — request building", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("should build a properly formatted Gemini request with contents, systemInstruction, and generationConfig", async () => {
    const { fetchMock, calls } = makeFetchCapture();
    globalThis.fetch = fetchMock;

    const request: ChatCompletionRequest = {
      model: "gemini-2.5-flash",
      messages: [
        { role: "system", content: "You are helpful." },
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi there" },
        { role: "user", content: "What's up?" },
      ],
      temperature: 0.5,
      max_tokens: 4096,
    };

    await GeminiAdapter.chatCompletion("gemini-2.5-flash", request, "test-api-key");

    const call = calls[0];
    const body = call.body;

    // Contents should exclude system messages and map roles correctly
    assert.ok(body.contents, "Should have contents");
    assert.equal(body.contents.length, 3, "Should have 3 non-system messages");
    assert.equal(body.contents[0].role, "user");
    assert.equal(body.contents[0].parts[0].text, "Hello");
    assert.equal(body.contents[1].role, "model", "Assistant should map to model");
    assert.equal(body.contents[1].parts[0].text, "Hi there");
    assert.equal(body.contents[2].role, "user");
    assert.equal(body.contents[2].parts[0].text, "What's up?");

    // System instruction
    assert.ok(body.systemInstruction, "Should have systemInstruction");
    assert.equal(body.systemInstruction.parts[0].text, "You are helpful.");

    // Generation config
    assert.ok(body.generationConfig, "Should have generationConfig");
    assert.equal(body.generationConfig.temperature, 0.5);
    assert.equal(body.generationConfig.maxOutputTokens, 4096);
    assert.ok(body.generationConfig.thinkingConfig, "Should have thinkingConfig");

    // URL should include the model and endpoint
    assert.ok(call.url.includes("models/gemini-2.5-flash:generateContent"), "URL should target generateContent endpoint");
    assert.ok(call.url.includes("key=test-api-key"), "URL should include API key");
  });
});

// ─── Gemini Response Conversion ─────────────────────────────

describe("Gemini response conversion (geminiToOpenAI)", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("should convert Gemini text response to OpenAI format", async () => {
    globalThis.fetch = (async (): Promise<Response> => {
      return new Response(JSON.stringify({
        candidates: [{
          content: { parts: [{ text: "Hello from Gemini" }] },
          finishReason: "STOP",
        }],
        usageMetadata: {
          promptTokenCount: 10,
          candidatesTokenCount: 5,
          totalTokenCount: 15,
        },
      }), { status: 200 });
    }) as typeof fetch;

    const result = await GeminiAdapter.chatCompletion(
      "gemini-2.5-flash",
      { model: "gemini-2.5-flash", messages: [{ role: "user", content: "Hi" }] },
      "test-key",
    );

    assert.equal(result.object, "chat.completion");
    assert.ok(result.id.startsWith("chatcmpl-"));
    assert.equal(result.choices.length, 1);
    assert.equal(result.choices[0].message.role, "assistant");
    assert.equal(result.choices[0].message.content, "Hello from Gemini");
    assert.equal(result.choices[0].finish_reason, "stop");

    // Usage mapping
    assert.ok(result.usage);
    assert.equal(result.usage!.prompt_tokens, 10);
    assert.equal(result.usage!.completion_tokens, 5);
    assert.equal(result.usage!.total_tokens, 15);
  });

  it("should convert Gemini functionCall parts to OpenAI tool_calls", async () => {
    globalThis.fetch = (async (): Promise<Response> => {
      return new Response(JSON.stringify({
        candidates: [{
          content: {
            parts: [{
              functionCall: {
                name: "get_weather",
                args: { city: "Tokyo", unit: "celsius" },
              },
            }],
          },
          finishReason: "STOP",
        }],
      }), { status: 200 });
    }) as typeof fetch;

    const result = await GeminiAdapter.chatCompletion(
      "gemini-2.5-flash",
      { model: "gemini-2.5-flash", messages: [{ role: "user", content: "Weather?" }] },
      "test-key",
    );

    const toolCalls = result.choices[0].message.tool_calls;
    assert.ok(toolCalls, "Should have tool_calls");
    assert.equal(toolCalls.length, 1);
    assert.equal(toolCalls[0].type, "function");
    assert.equal(toolCalls[0].function.name, "get_weather");

    const args = JSON.parse(toolCalls[0].function.arguments);
    assert.equal(args.city, "Tokyo");
    assert.equal(args.unit, "celsius");

    // Finish reason should be tool_calls when functionCall present
    assert.equal(result.choices[0].finish_reason, "tool_calls");
  });

  it("should handle mixed text and functionCall parts", async () => {
    globalThis.fetch = (async (): Promise<Response> => {
      return new Response(JSON.stringify({
        candidates: [{
          content: {
            parts: [
              { text: "Let me check that for you." },
              { functionCall: { name: "search", args: { q: "test" } } },
            ],
          },
          finishReason: "STOP",
        }],
      }), { status: 200 });
    }) as typeof fetch;

    const result = await GeminiAdapter.chatCompletion(
      "gemini-2.5-flash",
      { model: "gemini-2.5-flash", messages: [{ role: "user", content: "Search" }] },
      "test-key",
    );

    // Both text and tool_calls should be present
    assert.ok(result.choices[0].message.content.includes("Let me check"));
    assert.ok(result.choices[0].message.tool_calls);
    assert.equal(result.choices[0].message.tool_calls.length, 1);
  });

  it("should handle empty candidates gracefully", async () => {
    globalThis.fetch = (async (): Promise<Response> => {
      return new Response(JSON.stringify({
        candidates: [],
      }), { status: 200 });
    }) as typeof fetch;

    const result = await GeminiAdapter.chatCompletion(
      "gemini-2.5-flash",
      { model: "gemini-2.5-flash", messages: [{ role: "user", content: "Hi" }] },
      "test-key",
    );

    assert.equal(result.choices.length, 1);
    assert.equal(result.choices[0].message.content, "");
    assert.equal(result.choices[0].finish_reason, "stop");
  });
});

// ─── ZAI Adapter — Thinking Parameters ──────────────────────

describe("ZAI Adapter — thinking parameters", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("should inject thinking with correct budget for 'medium' level", async () => {
    const { fetchMock, calls } = makeFetchCapture();
    globalThis.fetch = fetchMock;

    await ZAIAdapter.chatCompletion(
      "glm-5.1",
      {
        model: "glm-5.1",
        messages: [{ role: "user", content: "test" }],
        thinking: { type: "enabled", budget_tokens: 8192 },
      },
      "test-key",
    );

    const body = calls[0].body;
    assert.ok(body.thinking, "Thinking should be injected");
    assert.equal(body.thinking.type, "enabled");
    assert.equal(body.thinking.budget_tokens, 8192);
  });

  it("should inject thinking for 'high' level", async () => {
    const { fetchMock, calls } = makeFetchCapture();
    globalThis.fetch = fetchMock;

    await ZAIAdapter.chatCompletion(
      "glm-5.1",
      {
        model: "glm-5.1",
        messages: [{ role: "user", content: "test" }],
        thinking: { type: "enabled", budget_tokens: 32768 },
      },
      "test-key",
    );

    const body = calls[0].body;
    assert.ok(body.thinking);
    assert.equal(body.thinking.budget_tokens, 32768);
  });

  it("should NOT inject thinking when level is 'none'", async () => {
    const { fetchMock, calls } = makeFetchCapture();
    globalThis.fetch = fetchMock;

    await ZAIAdapter.chatCompletion(
      "glm-5.1",
      {
        model: "glm-5.1",
        messages: [{ role: "user", content: "test" }],
        thinking: "none",
      },
      "test-key",
    );

    const body = calls[0].body;
    assert.equal(body.thinking, undefined, "Thinking should not be present for 'none'");
  });

  it("should strip thinking/reasoning/reasoning_effort from request body", async () => {
    const { fetchMock, calls } = makeFetchCapture();
    globalThis.fetch = fetchMock;

    await ZAIAdapter.chatCompletion(
      "glm-5.1",
      {
        model: "glm-5.1",
        messages: [{ role: "user", content: "test" }],
        thinking: "high",
        reasoning: "high",
        reasoning_effort: "high",
      } as any,
      "test-key",
    );

    const body = calls[0].body;
    // The original thinking/reasoning/reasoning_effort should be stripped and replaced with computed thinking
    assert.equal(body.reasoning, undefined, "reasoning should be stripped");
    assert.equal(body.reasoning_effort, undefined, "reasoning_effort should be stripped");
    // thinking should be the computed value
    assert.ok(body.thinking, "Computed thinking should be present");
  });

  it("should send Bearer token authorization", async () => {
    const { fetchMock, calls } = makeFetchCapture();
    globalThis.fetch = fetchMock;

    await ZAIAdapter.chatCompletion(
      "glm-5.1",
      { model: "glm-5.1", messages: [{ role: "user", content: "test" }] },
      "my-secret-key",
    );

    const init = calls[0].init!;
    const headers = init.headers as Record<string, string>;
    assert.equal(headers.Authorization, "Bearer my-secret-key");
    assert.equal(headers["Content-Type"], "application/json");
  });
});

// ─── OpenRouter Adapter ─────────────────────────────────────

describe("OpenRouter Adapter", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("should inject reasoning_effort for medium thinking level", async () => {
    const { fetchMock, calls } = makeFetchCapture();
    globalThis.fetch = fetchMock;

    await OpenRouterAdapter.chatCompletion(
      "qwen/qwen3-coder:free",
      {
        model: "qwen/qwen3-coder:free",
        messages: [{ role: "user", content: "test" }],
        thinking: { type: "enabled", budget_tokens: 8192 },
      },
      "or-key",
    );

    const body = calls[0].body;
    assert.equal(body.reasoning_effort, "medium");
  });

  it("should inject reasoning_effort for high thinking level", async () => {
    const { fetchMock, calls } = makeFetchCapture();
    globalThis.fetch = fetchMock;

    await OpenRouterAdapter.chatCompletion(
      "qwen/qwen3-coder:free",
      {
        model: "qwen/qwen3-coder:free",
        messages: [{ role: "user", content: "test" }],
        thinking: { type: "enabled", budget_tokens: 32768 },
      },
      "or-key",
    );

    assert.equal(calls[0].body.reasoning_effort, "high");
  });

  it("should NOT inject reasoning_effort for none level", async () => {
    const { fetchMock, calls } = makeFetchCapture();
    globalThis.fetch = fetchMock;

    await OpenRouterAdapter.chatCompletion(
      "qwen/qwen3-coder:free",
      {
        model: "qwen/qwen3-coder:free",
        messages: [{ role: "user", content: "test" }],
        thinking: "none",
      },
      "or-key",
    );

    assert.equal(calls[0].body.reasoning_effort, undefined);
  });

  it("should include HTTP-Referer and X-Title headers", async () => {
    const { fetchMock, calls } = makeFetchCapture();
    globalThis.fetch = fetchMock;

    await OpenRouterAdapter.chatCompletion(
      "qwen/qwen3-coder:free",
      { model: "qwen/qwen3-coder:free", messages: [{ role: "user", content: "test" }] },
      "or-key",
    );

    const headers = calls[0].init!.headers as Record<string, string>;
    assert.ok(headers["HTTP-Referer"], "HTTP-Referer header should be present");
    assert.ok(headers["HTTP-Referer"].includes("github.com"), "HTTP-Referer should contain github.com");
    assert.equal(headers["X-Title"], "Cognitive Router");
    assert.equal(headers.Authorization, "Bearer or-key");
  });
});

// ─── Error Classification ───────────────────────────────────
// classifyError is not exported; we test it indirectly by checking that
// adapters throw errors with the right .code property.

describe("Error classification (indirect via adapter)", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("should classify HTTP 429 as rate_limit", async () => {
    globalThis.fetch = (async (): Promise<Response> => {
      return new Response("Too many requests", { status: 429 });
    }) as typeof fetch;

    await assert.rejects(
      () => ZAIAdapter.chatCompletion("glm-5.1", { model: "glm-5.1", messages: [{ role: "user", content: "x" }] }, "key"),
      (err: any) => {
        assert.equal(err.code, "rate_limit");
        assert.ok(err.message.includes("rate_limit"));
        return true;
      },
    );
  });

  it("should classify HTTP 402 as quota_exceeded", async () => {
    globalThis.fetch = (async (): Promise<Response> => {
      return new Response("monthly limit exceeded", { status: 402 });
    }) as typeof fetch;

    await assert.rejects(
      () => ZAIAdapter.chatCompletion("glm-5.1", { model: "glm-5.1", messages: [{ role: "user", content: "x" }] }, "key"),
      (err: any) => {
        assert.equal(err.code, "quota_exceeded");
        return true;
      },
    );
  });

  it("should classify HTTP 500 as server_error", async () => {
    globalThis.fetch = (async (): Promise<Response> => {
      return new Response("Internal server error", { status: 500 });
    }) as typeof fetch;

    await assert.rejects(
      () => ZAIAdapter.chatCompletion("glm-5.1", { model: "glm-5.1", messages: [{ role: "user", content: "x" }] }, "key"),
      (err: any) => {
        assert.equal(err.code, "server_error");
        return true;
      },
    );
  });

  it("should classify HTTP 503 as server_error", async () => {
    globalThis.fetch = (async (): Promise<Response> => {
      return new Response("Service unavailable", { status: 503 });
    }) as typeof fetch;

    await assert.rejects(
      () => ZAIAdapter.chatCompletion("glm-5.1", { model: "glm-5.1", messages: [{ role: "user", content: "x" }] }, "key"),
      (err: any) => {
        assert.equal(err.code, "server_error");
        return true;
      },
    );
  });

  it("should classify HTTP 401 as auth_error", async () => {
    globalThis.fetch = (async (): Promise<Response> => {
      return new Response("Unauthorized", { status: 401 });
    }) as typeof fetch;

    await assert.rejects(
      () => ZAIAdapter.chatCompletion("glm-5.1", { model: "glm-5.1", messages: [{ role: "user", content: "x" }] }, "key"),
      (err: any) => {
        assert.equal(err.code, "auth_error");
        return true;
      },
    );
  });

  it("should classify HTTP 403 as auth_error", async () => {
    globalThis.fetch = (async (): Promise<Response> => {
      return new Response("Forbidden", { status: 403 });
    }) as typeof fetch;

    await assert.rejects(
      () => ZAIAdapter.chatCompletion("glm-5.1", { model: "glm-5.1", messages: [{ role: "user", content: "x" }] }, "key"),
      (err: any) => {
        assert.equal(err.code, "auth_error");
        return true;
      },
    );
  });

  it("should classify HTTP 400 as http_error", async () => {
    globalThis.fetch = (async (): Promise<Response> => {
      return new Response("Bad request", { status: 400 });
    }) as typeof fetch;

    await assert.rejects(
      () => ZAIAdapter.chatCompletion("glm-5.1", { model: "glm-5.1", messages: [{ role: "user", content: "x" }] }, "key"),
      (err: any) => {
        assert.equal(err.code, "http_error");
        return true;
      },
    );
  });

  it("should classify rate limit from body text even with non-429 status", async () => {
    globalThis.fetch = (async (): Promise<Response> => {
      return new Response(JSON.stringify({ error: { message: "Rate limit exceeded. Please slow down." } }), { status: 400 });
    }) as typeof fetch;

    await assert.rejects(
      () => ZAIAdapter.chatCompletion("glm-5.1", { model: "glm-5.1", messages: [{ role: "user", content: "x" }] }, "key"),
      (err: any) => {
        assert.equal(err.code, "rate_limit");
        return true;
      },
    );
  });
});

// ─── SSE Stream Parsing ─────────────────────────────────────

describe("SSE stream parsing", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("should parse SSE chunks from a streaming response", async () => {
    const chunks = [
      {
        id: "chatcmpl-1",
        object: "chat.completion.chunk",
        created: Date.now(),
        model: "test-model",
        choices: [{ index: 0, delta: { content: "Hello" }, finish_reason: null }],
      },
      {
        id: "chatcmpl-1",
        object: "chat.completion.chunk",
        created: Date.now(),
        model: "test-model",
        choices: [{ index: 0, delta: { content: " world" }, finish_reason: null }],
      },
      {
        id: "chatcmpl-1",
        object: "chat.completion.chunk",
        created: Date.now(),
        model: "test-model",
        choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
      },
    ];

    globalThis.fetch = (async (): Promise<Response> => {
      return makeSSEResponse(chunks);
    }) as typeof fetch;

    const collected: any[] = [];
    for await (const chunk of ZAIAdapter.chatCompletionStream(
      "test-model",
      { model: "test-model", messages: [{ role: "user", content: "x" }] },
      "key",
    )) {
      collected.push(chunk);
    }

    assert.equal(collected.length, 3, "Should yield 3 chunks before [DONE]");
    assert.equal(collected[0].choices[0].delta.content, "Hello");
    assert.equal(collected[1].choices[0].delta.content, " world");
    assert.equal(collected[2].choices[0].finish_reason, "stop");
  });

  it("should normalize model name in parsed chunks", async () => {
    const chunks = [{
      id: "x",
      object: "chat.completion.chunk",
      created: Date.now(),
      model: "original-model",
      choices: [{ index: 0, delta: { content: "hi" }, finish_reason: null }],
    }];

    globalThis.fetch = (async (): Promise<Response> => {
      return makeSSEResponse(chunks);
    }) as typeof fetch;

    const collected: any[] = [];
    for await (const chunk of ZAIAdapter.chatCompletionStream(
      "normalized-model",
      { model: "normalized-model", messages: [{ role: "user", content: "x" }] },
      "key",
    )) {
      collected.push(chunk);
    }

    // parseSSEStream normalizes the model field to the passed-in model
    assert.equal(collected[0].model, "normalized-model", "Model should be normalized");
  });

  it("should handle empty lines and comments in SSE stream", async () => {
    const sseBody = [
      ": this is a comment",
      "",
      'data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}',
      "",
      "data: [DONE]",
      "",
    ].join("\n");

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(sseBody));
        controller.close();
      },
    });

    globalThis.fetch = (async (): Promise<Response> => {
      return new Response(stream, { status: 200 });
    }) as typeof fetch;

    const collected: any[] = [];
    for await (const chunk of ZAIAdapter.chatCompletionStream(
      "m",
      { model: "m", messages: [{ role: "user", content: "x" }] },
      "key",
    )) {
      collected.push(chunk);
    }

    assert.equal(collected.length, 1);
    assert.equal(collected[0].choices[0].delta.content, "hi");
  });
});

// ─── Gemini Streaming — Tool Call Extraction ────────────────

describe("Gemini streaming — tool call handling", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("should extract functionCall parts from Gemini stream", async () => {
    globalThis.fetch = (async (): Promise<Response> => {
      return makeGeminiSSEResponse([
        [{ functionCall: { name: "get_weather", args: { city: "NYC" } } }],
      ]);
    }) as typeof fetch;

    const collected: any[] = [];
    for await (const chunk of GeminiAdapter.chatCompletionStream(
      "gemini-2.5-flash",
      { model: "gemini-2.5-flash", messages: [{ role: "user", content: "weather" }] },
      "test-key",
    )) {
      collected.push(chunk);
    }

    // First chunk should have tool_calls in delta
    const toolChunk = collected.find((c) => c.choices[0].delta.tool_calls);
    assert.ok(toolChunk, "Should have a chunk with tool_calls");
    const toolCall = toolChunk.choices[0].delta.tool_calls[0];
    assert.equal(toolCall.type, "function");
    assert.equal(toolCall.function.name, "get_weather");
    const args = JSON.parse(toolCall.function.arguments);
    assert.equal(args.city, "NYC");

    // The Gemini stream yields chunks with finish_reason: null for each part,
    // then the [DONE] marker ends the generator. The final chunk should be the
    // tool_call chunk (finish_reason: null — the adapter's terminal "stop" chunk
    // is skipped because [DONE] triggers an early generator return).
    const lastChunk = collected[collected.length - 1];
    assert.equal(lastChunk.choices[0].finish_reason, null, "Last data chunk should have null finish_reason before [DONE]");
  });

  it("should handle text parts in Gemini stream", async () => {
    globalThis.fetch = (async (): Promise<Response> => {
      return makeGeminiSSEResponse([
        [{ text: "Hello from Gemini stream" }],
      ]);
    }) as typeof fetch;

    const collected: any[] = [];
    for await (const chunk of GeminiAdapter.chatCompletionStream(
      "gemini-2.5-flash",
      { model: "gemini-2.5-flash", messages: [{ role: "user", content: "hi" }] },
      "test-key",
    )) {
      collected.push(chunk);
    }

    const textChunk = collected.find((c) => c.choices[0].delta.content);
    assert.ok(textChunk);
    assert.equal(textChunk.choices[0].delta.content, "Hello from Gemini stream");
  });

  it("should handle mixed text and functionCall in same part array", async () => {
    globalThis.fetch = (async (): Promise<Response> => {
      return makeGeminiSSEResponse([
        [
          { text: "Let me search that." },
          { functionCall: { name: "search", args: { q: "test" } } },
        ],
      ]);
    }) as typeof fetch;

    const collected: any[] = [];
    for await (const chunk of GeminiAdapter.chatCompletionStream(
      "gemini-2.5-flash",
      { model: "gemini-2.5-flash", messages: [{ role: "user", content: "search" }] },
      "test-key",
    )) {
      collected.push(chunk);
    }

    // Should have both content and tool_calls
    const payloadChunk = collected[0];
    assert.ok(payloadChunk.choices[0].delta.content, "Should have content");
    assert.ok(payloadChunk.choices[0].delta.tool_calls, "Should have tool_calls");
    assert.equal(payloadChunk.choices[0].delta.content, "Let me search that.");
    assert.equal(payloadChunk.choices[0].delta.tool_calls[0].function.name, "search");
  });
});

// ─── Ollama Adapter ─────────────────────────────────────────

describe("Ollama Adapter", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("should send requests to local Ollama endpoint", async () => {
    const { fetchMock, calls } = makeFetchCapture();
    globalThis.fetch = fetchMock;

    await OllamaAdapter.chatCompletion(
      "gemma4:latest",
      { model: "gemma4:latest", messages: [{ role: "user", content: "hello" }] },
      "",
    );

    assert.ok(calls[0].url.includes("localhost:11434"), "Should target localhost:11434");
    assert.ok(calls[0].url.includes("/chat/completions"), "Should target chat completions endpoint");
    assert.equal(calls[0].body.model, "gemma4:latest");
    assert.equal(calls[0].body.stream, false);
  });

  it("should set think=true when thinking is enabled", async () => {
    const { fetchMock, calls } = makeFetchCapture();
    globalThis.fetch = fetchMock;

    await OllamaAdapter.chatCompletion(
      "gemma4:latest",
      {
        model: "gemma4:latest",
        messages: [{ role: "user", content: "hello" }],
        thinking: "high",
      } as any,
      "",
    );

    assert.equal(calls[0].body.think, true, "think should be true when thinking is enabled");
  });

  it("should set think=false when thinking is none", async () => {
    const { fetchMock, calls } = makeFetchCapture();
    globalThis.fetch = fetchMock;

    await OllamaAdapter.chatCompletion(
      "gemma4:latest",
      { model: "gemma4:latest", messages: [{ role: "user", content: "hello" }] },
      "",
    );

    assert.equal(calls[0].body.think, false, "think should be false by default");
  });
});
