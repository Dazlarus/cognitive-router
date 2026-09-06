// tests/e2e.test.ts — End-to-end proxy lifecycle tests
// Run with: npx tsx --test tests/e2e.test.ts

import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import net from "node:net";
import { rmSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { ProxyServerStreaming } from "../src/proxy-stream.ts";
import { providerBackoff } from "../src/failure_classifier.ts";
import { loadConfig, type CognitiveRouterConfig } from "../src/config.ts";
import { DBService } from "../src/db_service.ts";

// ─── Test Helpers ───────────────────────────────────────────

function makeConfig(overrides: Partial<CognitiveRouterConfig> = {}): CognitiveRouterConfig {
  const base = loadConfig({
    enabled: true,
    logLevel: "warn",
    providerPriority: ["zai", "openrouter", "gemini", "ollama"],
    providers: {
      zai: { budgetType: "subscription", priority: "high" },
      openrouter: { budgetType: "free", priority: "high" },
      gemini: { budgetType: "credits", priority: "medium" },
      ollama: { budgetType: "free", priority: "low" },
    },
    weights: { capability: 0.5, reliability: 0.25, cost: 0.15, latency: 0.1 },
  });
  return { ...base, ...overrides };
}

async function getFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      server.close(() => {
        if (address && typeof address === "object") resolve(address.port);
        else reject(new Error("Unable to allocate test port"));
      });
    });
  });
}

function postChat(port: number, sessionKey: string, content: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify({
      model: "CognitiveRouter:latest",
      messages: [{ role: "user", content }],
      stream: false,
    });
    const req = http.request({
      hostname: "127.0.0.1", port,
      path: "/v1/chat/completions", method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(payload),
        "x-openclaw-session-key": sessionKey,
      },
    }, (res) => {
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => { body += chunk; });
      res.on("end", () => {
        if ((res.statusCode ?? 500) >= 400) {
          reject(new Error(`HTTP ${res.statusCode}: ${body}`));
          return;
        }
        resolve(JSON.parse(body));
      });
    });
    req.on("error", reject);
    req.write(payload);
    req.end();
  });
}

function postRawChat(
  port: number,
  payload: Record<string, any>,
  sessionKey = "test-session",
): Promise<{ statusCode: number; body: string }> {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(payload);
    const req = http.request({
      hostname: "127.0.0.1", port,
      path: "/v1/chat/completions", method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(body),
        "x-openclaw-session-key": sessionKey,
      },
    }, (res) => {
      let responseBody = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => { responseBody += chunk; });
      res.on("end", () => resolve({ statusCode: res.statusCode ?? 0, body: responseBody }));
    });
    req.on("error", reject);
    req.write(body);
    req.end();
  });
}

function streamChat(port: number, sessionKey: string, content: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify({
      model: "CognitiveRouter:latest",
      messages: [{ role: "user", content }],
      stream: true,
    });
    const req = http.request({
      hostname: "127.0.0.1", port,
      path: "/v1/chat/completions", method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(payload),
        "x-openclaw-session-key": sessionKey,
      },
    }, (res) => {
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => { body += chunk; });
      res.on("end", () => {
        if ((res.statusCode ?? 500) >= 400) {
          reject(new Error(`HTTP ${res.statusCode}: ${body}`));
          return;
        }
        resolve(body);
      });
    });
    req.on("error", reject);
    req.write(payload);
    req.end();
  });
}

function getModels(port: number): Promise<any> {
  return new Promise((resolve, reject) => {
    const req = http.request({
      hostname: "127.0.0.1", port, path: "/v1/models", method: "GET",
    }, (res) => {
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => { body += chunk; });
      res.on("end", () => {
        if ((res.statusCode ?? 500) >= 400) {
          reject(new Error(`HTTP ${res.statusCode}: ${body}`));
          return;
        }
        resolve(JSON.parse(body));
      });
    });
    req.on("error", reject);
    req.end();
  });
}

function getStats(port: number): Promise<any> {
  return new Promise((resolve, reject) => {
    const req = http.request({
      hostname: "127.0.0.1", port, path: "/stats", method: "GET",
    }, (res) => {
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => { body += chunk; });
      res.on("end", () => {
        if ((res.statusCode ?? 500) >= 400) {
          reject(new Error(`HTTP ${res.statusCode}: ${body}`));
          return;
        }
        resolve(JSON.parse(body));
      });
    });
    req.on("error", reject);
    req.end();
  });
}

function getHealth(port: number): Promise<any> {
  return new Promise((resolve, reject) => {
    const req = http.request({
      hostname: "127.0.0.1", port, path: "/health", method: "GET",
    }, (res) => {
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => { body += chunk; });
      res.on("end", () => resolve(JSON.parse(body)));
    });
    req.on("error", reject);
    req.end();
  });
}

// ─── Mock fetch helpers ─────────────────────────────────────

function makeOpenAIChatResponse(content: string, model: string): Response {
  return new Response(JSON.stringify({
    id: `chatcmpl-${Date.now()}`,
    object: "chat.completion",
    created: Date.now(),
    model,
    choices: [{
      index: 0,
      message: { role: "assistant", content },
      finish_reason: "stop",
    }],
  }), { status: 200 });
}

function makeOpenAIErrorResponse(status: number, message: string): Response {
  return new Response(JSON.stringify({ error: { message } }), { status });
}

// ─── Environment Management ─────────────────────────────────

const originalFetch = globalThis.fetch;
const originalMaxAttempts = process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER;
const originalZaiKey = process.env.ZAI_API_KEY;
const originalOpenRouterKey = process.env.OPENROUTER_API_KEY;
const originalGeminiKey = process.env.GEMINI_API_KEY;

let tempDbPaths: string[] = [];
let activeProxies: ProxyServerStreaming[] = [];

function makeTempDbPath(): string {
  const name = `tmp/e2e-${Date.now()}-${Math.random().toString(36).slice(2)}.db`;
  const fullPath = resolve(name);
  mkdirSync(resolve(fullPath, ".."), { recursive: true });
  tempDbPaths.push(fullPath);
  return fullPath;
}

beforeEach(() => {
  process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER = "1";
  delete process.env.ZAI_API_KEY;
});

afterEach(async () => {
  // Stop proxies
  for (const proxy of activeProxies) {
    try { await proxy.stop(); } catch { /* already stopped */ }
  }
  activeProxies = [];
  // Module-level backoff tracker survives across proxy instances in this
  // process — clear it so one test's 429s can't starve the next test's providers.
  providerBackoff.clearAll();

  // Restore env
  globalThis.fetch = originalFetch;
  if (originalMaxAttempts === undefined) delete process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER;
  else process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER = originalMaxAttempts;

  if (originalZaiKey === undefined) delete process.env.ZAI_API_KEY;
  else process.env.ZAI_API_KEY = originalZaiKey;

  if (originalOpenRouterKey === undefined) delete process.env.OPENROUTER_API_KEY;
  else process.env.OPENROUTER_API_KEY = originalOpenRouterKey;

  if (originalGeminiKey === undefined) delete process.env.GEMINI_API_KEY;
  else process.env.GEMINI_API_KEY = originalGeminiKey;

  // Clean up temp DBs
  for (const p of tempDbPaths) {
    for (const suffix of ["", "-wal", "-shm"]) {
      try { rmSync(p + suffix, { force: true }); } catch { /* ignore */ }
    }
  }
  tempDbPaths = [];
});

async function startProxy(config: CognitiveRouterConfig): Promise<{ proxy: ProxyServerStreaming; port: number }> {
  const proxy = new ProxyServerStreaming(config);
  await proxy.start();
  activeProxies.push(proxy);
  return { proxy, port: config.proxyPort! };
}

// ─── Tests ──────────────────────────────────────────────────

describe("E2E — Basic chat flow", () => {
  it("should route a chat request and return CognitiveRouter:latest as model name (no provider leakage)", async () => {
    let providerCalled = "";

    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);

      if (url.includes("/api/embeddings")) {
        return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      }
      if (url.includes("/api/tags")) {
        return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });
      }
      if (url.includes("/v1/chat/completions")) {
        const body = JSON.parse(String(init?.body ?? "{}"));
        providerCalled = body.model;
        return makeOpenAIChatResponse("Hello from provider!", body.model);
      }
      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = makeTempDbPath();
    const config = makeConfig({
      dbPath,
      providerPriority: ["ollama"],
      providers: { ollama: { budgetType: "free", priority: "high" } },
      proxyPort: port,
    });

    const { proxy } = await startProxy(config);

    try {
      const response = await postChat(port, "e2e-basic", "Hello there");

      assert.equal(response.model, "CognitiveRouter:latest", "Response model should be CognitiveRouter:latest");
      assert.ok(!response.model.includes("/"), "Model should not contain provider prefix");
      assert.ok(response.choices[0].message.content, "Should have non-empty content");
      assert.equal(response.choices[0].finish_reason, "stop");

      // The actual provider model should NOT leak into the response
      assert.notEqual(providerCalled, "CognitiveRouter:latest", "Provider should receive real model name");
    } finally {
      await proxy.stop();
    }
  });
});

/** Create a mock OpenAI SSE streaming response */
function makeOpenAIStreamResponse(content: string, model: string): Response {
  const words = content.split(" ");
  const chunks = words.map((word, i) => ({
    id: `chatcmpl-stream-${i}`,
    object: "chat.completion.chunk",
    created: Date.now(),
    model,
    choices: [{ index: 0, delta: { content: (i > 0 ? " " : "") + word }, finish_reason: null }],
  }));
  // Add role chunk first
  chunks.unshift({
    id: `chatcmpl-stream-0`,
    object: "chat.completion.chunk",
    created: Date.now(),
    model,
    choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
  });
  // Add final finish chunk
  chunks.push({
    id: `chatcmpl-stream-final`,
    object: "chat.completion.chunk",
    created: Date.now(),
    model,
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
  });

  const sseBody = chunks.map((c) => `data: ${JSON.stringify(c)}\n`).join("\n") + "data: [DONE]\n\n";
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(sseBody));
      controller.close();
    },
  });
  return new Response(stream, { status: 200 });
}

describe("E2E — Streaming chat", () => {
  it("should return SSE events with content and end with data: [DONE]", async () => {
    let streamRequested = false;
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);
      if (url.includes("/api/embeddings")) return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      if (url.includes("/api/tags")) return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });
      if (url.includes("/v1/chat/completions")) {
        const body = JSON.parse(String(init?.body ?? "{}"));
        if (body.stream === true) {
          streamRequested = true;
          return makeOpenAIStreamResponse("Streamed response content", body.model);
        }
        return makeOpenAIChatResponse("Streamed response content", body.model);
      }
      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = makeTempDbPath();
    const config = makeConfig({
      dbPath,
      providerPriority: ["ollama"],
      providers: { ollama: { budgetType: "free", priority: "high" } },
      proxyPort: port,
    });

    const { proxy } = await startProxy(config);

    try {
      const body = await streamChat(port, "e2e-stream", "Tell me a story");
      const events = body.split("\n\n").map((e) => e.trim()).filter(Boolean);

      // Verify the proxy actually requested streaming from the provider
      assert.ok(streamRequested, "Proxy should request stream:true from provider");

      assert.ok(events.length >= 2, "Should have at least 2 SSE events");
      assert.equal(events.at(-1), "data: [DONE]", "Last event should be [DONE]");

      // All data events should have CognitiveRouter:latest as model (no leakage)
      for (const evt of events) {
        if (evt === "data: [DONE]") continue;
        const payload = JSON.parse(evt.replace(/^data: /, ""));
        assert.equal(payload.model, "CognitiveRouter:latest", "SSE model should be CognitiveRouter:latest");
      }

      // Should have content spread across multiple chunks
      const contentChunks = events.filter((e, i) => {
        if (e === "data: [DONE]") return false;
        const p = JSON.parse(e.replace(/^data: /, ""));
        return p.choices?.[0]?.delta?.content;
      });
      assert.ok(contentChunks.length >= 1, "Should have content chunks");

      // Should have a finish_reason chunk
      const finishChunk = events.find((e) => {
        if (e === "data: [DONE]") return false;
        const p = JSON.parse(e.replace(/^data: /, ""));
        return p.choices?.[0]?.finish_reason === "stop";
      });
      assert.ok(finishChunk, "Should have a finish_reason: stop chunk");
    } finally {
      await proxy.stop();
    }
  });
});

describe("E2E — Multi-provider cascade", () => {
  it("should failover to the second provider when the first fails with 429", async () => {
    let callCount = 0;
    const providerUrlsHit: string[] = [];

    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);

      if (url.includes("/api/embeddings")) return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      if (url.includes("/api/tags")) return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });

      if (url.includes("openrouter.ai/api/v1/chat/completions")) {
        callCount++;
        providerUrlsHit.push(url);
        // First provider fails with 429
        return makeOpenAIErrorResponse(429, "Rate limit exceeded");
      }

      if (url.includes("/v1/chat/completions")) {
        callCount++;
        providerUrlsHit.push(url);
        const body = JSON.parse(String(init?.body ?? "{}"));
        // Second provider (ollama fallback) succeeds
        return makeOpenAIChatResponse("Success from fallback!", body.model);
      }

      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = makeTempDbPath();
    const config = makeConfig({
      dbPath,
      providerPriority: ["openrouter", "ollama"],
      providers: {
        openrouter: { budgetType: "free", priority: "high" },
        ollama: { budgetType: "free", priority: "low" },
      },
      proxyPort: port,
    });

    const { proxy } = await startProxy(config);

    try {
      const response = await postChat(port, "e2e-cascade", "Hello");

      assert.equal(response.model, "CognitiveRouter:latest");
      assert.ok(response.choices[0].message.content.includes("Success from fallback"));
      assert.ok(callCount >= 2, "Should have tried at least 2 providers");
    } finally {
      await proxy.stop();
    }
  });
});

describe("E2E — All providers exhausted", () => {
  it("should return 503 with all_providers_exhausted when every provider fails", async () => {
    globalThis.fetch = (async (input: RequestInfo | URL): Promise<Response> => {
      const url = String(input);
      if (url.includes("/api/embeddings")) return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      if (url.includes("/api/tags")) return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });
      if (url.includes("/v1/chat/completions")) return makeOpenAIErrorResponse(500, "Internal server error");
      if (url.includes("openrouter.ai")) return makeOpenAIErrorResponse(429, "Rate limited");
      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = makeTempDbPath();
    const config = makeConfig({
      dbPath,
      providerPriority: ["openrouter", "ollama"],
      providers: {
        openrouter: { budgetType: "free", priority: "high" },
        ollama: { budgetType: "free", priority: "low" },
      },
      proxyPort: port,
    });

    const { proxy } = await startProxy(config);

    try {
      const response = await postRawChat(port, {
        model: "CognitiveRouter:latest",
        messages: [{ role: "user", content: "This will fail" }],
        stream: false,
      });

      assert.equal(response.statusCode, 503, "Should return 503 when all providers exhausted");
      const body = JSON.parse(response.body);
      assert.equal(body.error.type, "all_providers_exhausted");
      assert.ok(body.error.message.includes("exhausted"));
    } finally {
      await proxy.stop();
    }
  });
});

describe("E2E — Tools routing", () => {
  it("should route tool-bearing requests to a tool-capable provider", async () => {
    let toolRequestReceived = false;

    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);
      if (url.includes("/api/embeddings")) return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      if (url.includes("/api/tags")) return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });
      if (url.includes("openrouter.ai/api/v1/chat/completions")) {
        const body = JSON.parse(String(init?.body ?? "{}"));
        if (Array.isArray(body.tools) && body.tools.length > 0) {
          toolRequestReceived = true;
        }
        return new Response(JSON.stringify({
          id: "chatcmpl-tool",
          object: "chat.completion",
          created: Date.now(),
          model: body.model,
          choices: [{
            index: 0,
            message: {
              role: "assistant",
              content: "",
              tool_calls: [{
                id: "call_tool_1",
                type: "function",
                function: { name: "example_tool", arguments: '{"result":42}' },
              }],
            },
            finish_reason: "tool_calls",
          }],
        }), { status: 200 });
      }
      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = makeTempDbPath();
    const config = makeConfig({
      dbPath,
      providerPriority: ["openrouter", "ollama"],
      providers: {
        openrouter: { budgetType: "free", priority: "high" },
        ollama: { budgetType: "free", priority: "low" },
      },
      proxyPort: port,
    });

    const { proxy } = await startProxy(config);

    try {
      const response = await postRawChat(port, {
        model: "CognitiveRouter:latest",
        messages: [{ role: "user", content: "Use the example tool" }],
        tools: [{
          type: "function",
          function: {
            name: "example_tool",
            description: "An example tool",
            parameters: { type: "object", properties: { result: { type: "integer" } } },
          },
        }],
        tool_choice: "auto",
        stream: false,
      }, "e2e-tools");

      assert.equal(response.statusCode, 200);
      assert.ok(toolRequestReceived, "Tool-capable provider should have received the tool request");
      const body = JSON.parse(response.body);
      assert.ok(body.choices[0].message.tool_calls, "Response should include tool_calls");
      assert.equal(body.choices[0].finish_reason, "tool_calls");
    } finally {
      await proxy.stop();
    }
  });
});

describe("E2E — GET /v1/models", () => {
  it("should return CognitiveRouter:latest, CogRouter:latest, and Embeddings:latest", async () => {
    globalThis.fetch = (async (input: RequestInfo | URL): Promise<Response> => {
      const url = String(input);
      if (url.includes("/api/embeddings")) return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      if (url.includes("/api/tags")) return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });
      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = makeTempDbPath();
    const config = makeConfig({
      dbPath,
      providerPriority: ["ollama"],
      providers: { ollama: { budgetType: "free", priority: "high" } },
      proxyPort: port,
    });

    const { proxy } = await startProxy(config);

    try {
      const models = await getModels(port);
      const ids = models.data.map((m: any) => m.id);

      assert.ok(ids.includes("CognitiveRouter:latest"), "Should list CognitiveRouter:latest");
      assert.ok(ids.includes("CogRouter:latest"), "Should list CogRouter:latest");
      assert.ok(ids.includes("Embeddings:latest"), "Should list Embeddings:latest");

      // No provider-specific model IDs should leak
      for (const id of ids) {
        assert.ok(!id.includes("/"), `Model ID "${id}" should not contain provider prefix`);
      }
    } finally {
      await proxy.stop();
    }
  });
});

describe("E2E — GET /health", () => {
  it("should return {status: 'ok'}", async () => {
    globalThis.fetch = (async (input: RequestInfo | URL): Promise<Response> => {
      const url = String(input);
      if (url.includes("/api/embeddings")) return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      if (url.includes("/api/tags")) return new Response(JSON.stringify({ models: [] }), { status: 200 });
      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = makeTempDbPath();
    const config = makeConfig({ dbPath, proxyPort: port });

    const { proxy } = await startProxy(config);

    try {
      const health = await getHealth(port);
      assert.equal(health.status, "ok");
    } finally {
      await proxy.stop();
    }
  });
});

describe("E2E — GET /stats", () => {
  it("should return provider health and model list", async () => {
    globalThis.fetch = (async (input: RequestInfo | URL): Promise<Response> => {
      const url = String(input);
      if (url.includes("/api/embeddings")) return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      if (url.includes("/api/tags")) return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });
      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = makeTempDbPath();
    const config = makeConfig({
      dbPath,
      providerPriority: ["ollama"],
      providers: { ollama: { budgetType: "free", priority: "high" } },
      proxyPort: port,
    });

    const { proxy } = await startProxy(config);

    try {
      const stats = await getStats(port);

      assert.ok(Array.isArray(stats.providers), "Should have providers array");
      assert.ok(Array.isArray(stats.models), "Should have models array");
      assert.equal(typeof stats.modelCount, "number", "Should have modelCount");
      assert.ok(stats.routing, "Should have routing info");
      assert.ok(stats.routing.providerPriority, "Should have providerPriority in routing");
    } finally {
      await proxy.stop();
    }
  });
});

describe("E2E — Session isolation", () => {
  it("should handle two concurrent requests from different sessions independently", async () => {
    const sessionCalls: Record<string, string[]> = {};

    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);
      if (url.includes("/api/embeddings")) return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      if (url.includes("/api/tags")) return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });

      if (url.includes("/v1/chat/completions")) {
        const body = JSON.parse(String(init?.body ?? "{}"));
        // Track which model was requested (for content tagging)
        const tag = body.model;
        return makeOpenAIChatResponse(`Response via ${tag}`, body.model);
      }
      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = makeTempDbPath();
    const config = makeConfig({
      dbPath,
      providerPriority: ["ollama"],
      providers: { ollama: { budgetType: "free", priority: "high" } },
      proxyPort: port,
    });

    const { proxy } = await startProxy(config);

    try {
      // Fire two concurrent requests from different sessions
      const [resA, resB] = await Promise.all([
        postChat(port, "session-alpha", "Hello from alpha"),
        postChat(port, "session-beta", "Hello from beta"),
      ]);

      // Both should succeed
      assert.equal(resA.model, "CognitiveRouter:latest");
      assert.equal(resB.model, "CognitiveRouter:latest");
      assert.ok(resA.choices[0].message.content.length > 0);
      assert.ok(resB.choices[0].message.content.length > 0);

      // Verify DB recorded both sessions
      const db = new DBService(dbPath);
      await db.initializeSchema();
      try {
        const decisions = db.getRecentDecisions(10);
        const recordedSessions = new Set(decisions.map((d: any) => d.session_key));
        assert.ok(recordedSessions.has("session-alpha"), "Should have recorded session-alpha");
        assert.ok(recordedSessions.has("session-beta"), "Should have recorded session-beta");
      } finally {
        db.close();
      }
    } finally {
      await proxy.stop();
    }
  });
});

describe("E2E — Streaming with tool calls", () => {
  it("should stream tool_calls from provider response as SSE", async () => {
    let streamRequested = false;
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);
      if (url.includes("/api/embeddings")) return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      if (url.includes("/api/tags")) return new Response(JSON.stringify({ models: [] }), { status: 200 });

      if (url.includes("openrouter.ai/api/v1/chat/completions")) {
        const body = JSON.parse(String(init?.body ?? "{}"));
        if (body.stream === true) {
          streamRequested = true;
          // Return SSE stream with tool_calls
          const chunks = [
            {
              id: "chatcmpl-stream-tool",
              object: "chat.completion.chunk",
              created: Date.now(),
              model: body.model,
              choices: [{ index: 0, delta: { role: "assistant", tool_calls: [{
                id: "call_stream_1",
                type: "function",
                function: { name: "stream_tool", arguments: "{\"x\":1}" },
              }] }, finish_reason: null }],
            },
            {
              id: "chatcmpl-stream-tool",
              object: "chat.completion.chunk",
              created: Date.now(),
              model: body.model,
              choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
            },
          ];
          const sseBody = chunks.map((c) => `data: ${JSON.stringify(c)}\n`).join("\n") + "data: [DONE]\n\n";
          const encoder = new TextEncoder();
          const stream = new ReadableStream({
            start(controller) {
              controller.enqueue(encoder.encode(sseBody));
              controller.close();
            },
          });
          return new Response(stream, { status: 200 });
        }
        return new Response(JSON.stringify({
          id: "chatcmpl-stream-tool",
          object: "chat.completion",
          created: Date.now(),
          model: body.model,
          choices: [{
            index: 0,
            message: {
              role: "assistant",
              content: "",
              tool_calls: [{
                id: "call_stream_1",
                type: "function",
                function: { name: "stream_tool", arguments: "{\"x\":1}" },
              }],
            },
            finish_reason: "tool_calls",
          }],
        }), { status: 200 });
      }
      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = makeTempDbPath();
    const config = makeConfig({
      dbPath,
      providerPriority: ["openrouter"],
      providers: { openrouter: { budgetType: "free", priority: "high" } },
      overrides: [{
        intent: "conversation",
        provider: "openrouter",
        model: "qwen/qwen3-coder:free",
        reason: "streaming tool test",
      }],
      proxyPort: port,
    });

    const { proxy } = await startProxy(config);

    try {
      const response = await postRawChat(port, {
        model: "CognitiveRouter:latest",
        messages: [{ role: "user", content: "Use a tool" }],
        tools: [{
          type: "function",
          function: {
            name: "stream_tool",
            description: "A streaming tool",
            parameters: { type: "object", properties: { x: { type: "integer" } } },
          },
        }],
        tool_choice: "auto",
        stream: true,
      }, "e2e-stream-tool");

      assert.equal(response.statusCode, 200);
      assert.ok(streamRequested, "Proxy should request stream:true from provider");

      const events = response.body.split("\n\n").map((e) => e.trim()).filter(Boolean);

      // Should have tool_calls in a chunk
      const toolChunk = events.find((e) => {
        if (e === "data: [DONE]") return false;
        const p = JSON.parse(e.replace(/^data: /, ""));
        return p.choices?.[0]?.delta?.tool_calls;
      });
      assert.ok(toolChunk, "Should have a chunk with tool_calls");
      const toolPayload = JSON.parse(toolChunk.replace(/^data: /, ""));
      assert.equal(toolPayload.model, "CognitiveRouter:latest", "Model should be scrubbed");
      assert.equal(toolPayload.choices[0].delta.tool_calls[0].function.name, "stream_tool");

      // Should have finish_reason: tool_calls
      const finishChunk = events.find((e) => {
        if (e === "data: [DONE]") return false;
        const p = JSON.parse(e.replace(/^data: /, ""));
        return p.choices?.[0]?.finish_reason === "tool_calls";
      });
      assert.ok(finishChunk, "Should have finish_reason: tool_calls chunk");

      // End with [DONE]
      assert.equal(events.at(-1), "data: [DONE]");
    } finally {
      await proxy.stop();
    }
  });
});

describe("E2E — Streaming mid-stream error handling", () => {
  it("should discard truncated content and emit a clean error + [DONE] when the provider stream errors mid-way", async () => {
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);
      if (url.includes("/api/embeddings")) return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      if (url.includes("/api/tags")) return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });

      if (url.includes("/v1/chat/completions")) {
        const body = JSON.parse(String(init?.body ?? "{}"));
        if (body.stream === true) {
          // Return a stream that sends one chunk, then aborts with an error
          const encoder = new TextEncoder();
          const chunk1 = `data: ${JSON.stringify({
            id: "chatcmpl-err",
            object: "chat.completion.chunk",
            created: Date.now(),
            model: body.model,
            choices: [{ index: 0, delta: { content: "partial" }, finish_reason: null }],
          })}\n\n`;
          // Simulate stream error by sending a chunk with error content
          // In reality this would be a network error, but for testing we
          // verify that a mid-stream HTTP error is handled gracefully
          const errorChunk = `data: ${JSON.stringify({
            error: { message: "server_error: internal error", type: "stream_error" },
          })}\n\n`;
          const done = "data: [DONE]\n\n";
          const stream = new ReadableStream({
            start(controller) {
              controller.enqueue(encoder.encode(chunk1 + errorChunk + done));
              controller.close();
            },
          });
          return new Response(stream, { status: 200 });
        }
        return makeOpenAIChatResponse("ok", body.model);
      }
      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = makeTempDbPath();
    const config = makeConfig({
      dbPath,
      providerPriority: ["ollama"],
      providers: { ollama: { budgetType: "free", priority: "high" } },
      proxyPort: port,
    });

    const { proxy } = await startProxy(config);

    try {
      const body = await streamChat(port, "e2e-midstream-err", "Hello");
      // Buffered-streaming contract (Daz, 2026-09-06: no passthrough):
      // a mid-stream provider death must NEVER surface truncated content.
      // The buffer is discarded, a retry is attempted invisibly, and when the
      // chain is exhausted the client gets one well-formed in-band error,
      // then a clean [DONE].
      assert.ok(body.includes("data: [DONE]"), "Should end with [DONE] even on error");
      assert.ok(!body.includes("partial"), "Truncated provider content must be discarded, never forwarded");
      // The failure is reported honestly as a structured SSE error event
      assert.ok(body.includes('"type":"all_providers_exhausted"') || body.includes('"type": "all_providers_exhausted"'),
        "Should surface a well-formed all_providers_exhausted error event");
    } finally {
      await proxy.stop();
    }
  });
});

describe("E2E — Provider error sanitization", () => {
  it("should sanitize sensitive data from provider error messages", async () => {
    globalThis.fetch = (async (input: RequestInfo | URL): Promise<Response> => {
      const url = String(input);
      if (url.includes("/api/embeddings")) return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      if (url.includes("/api/tags")) return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });
      if (url.includes("/v1/chat/completions")) {
        return new Response(JSON.stringify({
          error: {
            message: "Prompt tokens limit exceeded: 69529 > 66327. Visit https://openrouter.ai/workspaces/default/keys/7645a6e342a939c57a360638c770287a9880f3b80d86b7c8c89ea658cab59644 to adjust",
            code: 402,
          },
          user_id: "user_32kYpTck4JMfQFTf64cUi6QpWvw",
        }), { status: 402 });
      }
      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = makeTempDbPath();
    const config = makeConfig({
      dbPath,
      providerPriority: ["ollama"],
      providers: { ollama: { budgetType: "free", priority: "high" } },
      proxyPort: port,
    });

    const { proxy } = await startProxy(config);

    try {
      const response = await postRawChat(port, {
        model: "CognitiveRouter:latest",
        messages: [{ role: "user", content: "Trigger error" }],
        stream: true,
      }, "e2e-sanitize");

      // Even in streaming mode, errors should be sanitized
      assert.equal(response.body.includes("openrouter.ai/workspaces"), false, "Should not leak workspace URLs");
      assert.equal(response.body.includes("user_32kYp"), false, "Should not leak user IDs");
      assert.equal(response.body.includes("7645a6e342a939c5"), false, "Should not leak API keys");
    } finally {
      await proxy.stop();
    }
  });
});
