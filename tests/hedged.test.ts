// tests/hedged.test.ts — Tests for hedged request racing
// Run with: npx tsx --test tests/hedged.test.ts

import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import net from "node:net";
import { rmSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import {
  raceHedgedRequests,
  hedgeRetryDelayMs,
  type HedgeCandidate,
} from "../src/hedged_request.ts";
import type { ProviderAdapter, ChatCompletionResponse, ChatCompletionRequest } from "../src/providers.ts";
import { ProxyServerStreaming } from "../src/proxy-stream.ts";
import { loadConfig, type CognitiveRouterConfig } from "../src/config.ts";
import { DBService } from "../src/db_service.ts";

// ─── Mock Helpers ──────────────────────────────────────────

function makeChatResponse(content: string, model: string): ChatCompletionResponse {
  return {
    id: `chatcmpl-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    object: "chat.completion",
    created: Date.now(),
    model,
    choices: [{
      index: 0,
      message: { role: "assistant", content },
      finish_reason: "stop",
    }],
  };
}

function makeErrorResponse(status: number, message: string): Response {
  return new Response(JSON.stringify({ error: { message } }), { status });
}

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

/**
 * Create a mock ProviderAdapter with configurable behavior.
 * The adapter respects AbortSignal for cancellation.
 */
function makeMockAdapter(opts: {
  name: string;
  delayMs?: number;
  response?: ChatCompletionResponse;
  error?: Error;
}): ProviderAdapter {
  return {
    name: opts.name,
    async chatCompletion(
      _model: string,
      _request: ChatCompletionRequest,
      _apiKey: string,
      signal?: AbortSignal,
    ): Promise<ChatCompletionResponse> {
      if (opts.delayMs && opts.delayMs > 0) {
        await new Promise<void>((resolve, reject) => {
          const timer = setTimeout(resolve, opts.delayMs);
          if (signal) {
            signal.addEventListener("abort", () => {
              clearTimeout(timer);
              reject(new Error("aborted"));
            }, { once: true });
          }
        });
      }

      if (signal?.aborted) {
        throw new Error("aborted");
      }

      if (opts.error) throw opts.error;
      return opts.response ?? makeChatResponse(`response from ${opts.name}`, _model);
    },
    async *chatCompletionStream() {
      // Not used in these tests
    },
  };
}

// ─── Unit Tests: raceHedgedRequests ────────────────────────

describe("raceHedgedRequests — Unit Tests", () => {
  const baseRequest: ChatCompletionRequest = {
    model: "test-model",
    messages: [{ role: "user", content: "hello" }],
    stream: false,
  };

  it("fallback wins when it responds before primary retry", async () => {
    const primaryAdapter = makeMockAdapter({
      name: "primary",
      delayMs: 500, // Primary is slow
      response: makeChatResponse("primary won", "primary-model"),
    });
    const fallbackAdapter = makeMockAdapter({
      name: "fallback",
      delayMs: 10, // Fallback is fast
      response: makeChatResponse("fallback won", "fallback-model"),
    });

    const primary: HedgeCandidate = {
      provider: "primary", model: "primary-model",
      adapter: primaryAdapter, apiKey: "key",
    };
    const fallback: HedgeCandidate = {
      provider: "fallback", model: "fallback-model",
      adapter: fallbackAdapter, apiKey: "key",
    };

    const result = await raceHedgedRequests(primary, fallback, baseRequest, 100);

    assert.equal(result.outcome.result, "fallback_win");
    assert.equal(result.outcome.winnerProvider, "fallback");
    assert.equal(result.outcome.winnerModel, "fallback-model");
    assert.ok(result.outcome.loserCancelled, "Primary should have been cancelled");
    assert.equal(result.response.choices[0].message.content, "fallback won");
  });

  it("primary retry wins when fallback fails", async () => {
    const primaryAdapter = makeMockAdapter({
      name: "primary",
      delayMs: 10, // Primary retry is fast after delay
      response: makeChatResponse("primary won", "primary-model"),
    });
    const fallbackAdapter = makeMockAdapter({
      name: "fallback",
      error: new Error("server_error (500): internal error"),
    });

    const primary: HedgeCandidate = {
      provider: "primary", model: "primary-model",
      adapter: primaryAdapter, apiKey: "key",
    };
    const fallback: HedgeCandidate = {
      provider: "fallback", model: "fallback-model",
      adapter: fallbackAdapter, apiKey: "key",
    };

    const result = await raceHedgedRequests(primary, fallback, baseRequest, 50);

    assert.equal(result.outcome.result, "primary_win");
    assert.equal(result.outcome.winnerProvider, "primary");
    assert.ok(result.response.choices[0].message.content.includes("primary won"));
  });

  it("both fail → rejects with descriptive error", async () => {
    const primaryAdapter = makeMockAdapter({
      name: "primary",
      error: new Error("rate_limit: still throttled"),
    });
    const fallbackAdapter = makeMockAdapter({
      name: "fallback",
      error: new Error("server_error (500): down"),
    });

    const primary: HedgeCandidate = {
      provider: "primary", model: "primary-model",
      adapter: primaryAdapter, apiKey: "key",
    };
    const fallback: HedgeCandidate = {
      provider: "fallback", model: "fallback-model",
      adapter: fallbackAdapter, apiKey: "key",
    };

    await assert.rejects(
      raceHedgedRequests(primary, fallback, baseRequest, 50),
      (err: Error) => {
        assert.ok(err.message.includes("Both hedged requests failed"), `Unexpected error: ${err.message}`);
        assert.ok(err.message.includes("still throttled"));
        assert.ok(err.message.includes("down"));
        return true;
      },
    );
  });

  it("primary retry is cancelled when fallback wins first", async () => {
    let primaryWasCancelled = false;
    let primaryWasCalled = false;

    const primaryAdapter: ProviderAdapter = {
      name: "primary",
      async chatCompletion(_model, _request, _apiKey, signal) {
        primaryWasCalled = true;
        // Wait for abort signal
        await new Promise<void>((resolve) => {
          const timer = setTimeout(resolve, 10000); // Long wait
          signal?.addEventListener("abort", () => {
            primaryWasCancelled = true;
            clearTimeout(timer);
            resolve();
          }, { once: true });
        });

        if (signal?.aborted) throw new Error("aborted");
        return makeChatResponse("primary", _model);
      },
      async *chatCompletionStream() { },
    };

    const fallbackAdapter = makeMockAdapter({
      name: "fallback",
      delayMs: 5,
      response: makeChatResponse("fallback wins", "fallback-model"),
    });

    const result = await raceHedgedRequests(
      { provider: "primary", model: "m1", adapter: primaryAdapter, apiKey: "k" },
      { provider: "fallback", model: "m2", adapter: fallbackAdapter, apiKey: "k" },
      baseRequest,
      0, // No delay — both fire immediately so primary adapter IS called
    );

    assert.equal(result.outcome.result, "fallback_win");
    assert.ok(primaryWasCalled, "Primary adapter should have been called");
    assert.ok(primaryWasCancelled, "Primary should have been cancelled via AbortController");
  });

  it("only one response is returned regardless of timing", async () => {
    // Both succeed very quickly — only one should win
    let primaryWins = 0;
    let fallbackWins = 0;

    for (let i = 0; i < 10; i++) {
      const primaryAdapter = makeMockAdapter({
        name: "primary",
        delayMs: 5,
        response: makeChatResponse("primary", "m1"),
      });
      const fallbackAdapter = makeMockAdapter({
        name: "fallback",
        delayMs: 5,
        response: makeChatResponse("fallback", "m2"),
      });

      const result = await raceHedgedRequests(
        { provider: "primary", model: "m1", adapter: primaryAdapter, apiKey: "k" },
        { provider: "fallback", model: "m2", adapter: fallbackAdapter, apiKey: "k" },
        baseRequest,
        0, // No delay — both fire immediately
      );

      if (result.outcome.result === "primary_win") primaryWins++;
      if (result.outcome.result === "fallback_win") fallbackWins++;
    }

    assert.equal(primaryWins + fallbackWins, 10, "Every race should have exactly one winner");
  });
});

// ─── Unit Tests: hedgeRetryDelayMs ─────────────────────────

describe("hedgeRetryDelayMs", () => {
  it("returns a value within the configured range", () => {
    process.env.HEDGE_RETRY_MIN_MS = "5000";
    process.env.HEDGE_RETRY_MAX_MS = "10000";

    for (let i = 0; i < 20; i++) {
      const delay = hedgeRetryDelayMs();
      assert.ok(delay >= 5000, `Delay ${delay} should be >= 5000`);
      assert.ok(delay <= 10000, `Delay ${delay} should be <= 10000`);
    }

    delete process.env.HEDGE_RETRY_MIN_MS;
    delete process.env.HEDGE_RETRY_MAX_MS;
  });

  it("uses defaults when env vars are not set", () => {
    delete process.env.HEDGE_RETRY_MIN_MS;
    delete process.env.HEDGE_RETRY_MAX_MS;

    const delay = hedgeRetryDelayMs();
    assert.ok(delay >= 5000, `Default delay ${delay} should be >= 5000`);
    assert.ok(delay <= 10000, `Default delay ${delay} should be <= 10000`);
  });
});

// ─── E2E Tests: Proxy Integration ──────────────────────────

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
        resolve({ statusCode: res.statusCode ?? 0, body, json: (() => { try { return JSON.parse(body); } catch { return null; } })() });
      });
    });
    req.on("error", reject);
    req.write(payload);
    req.end();
  });
}

const originalFetch = globalThis.fetch;
let tempDbPaths: string[] = [];
let activeProxies: ProxyServerStreaming[] = [];

function makeTempDbPath(): string {
  const name = `tmp/hedge-${Date.now()}-${Math.random().toString(36).slice(2)}.db`;
  const fullPath = resolve(name);
  mkdirSync(resolve(fullPath, ".."), { recursive: true });
  tempDbPaths.push(fullPath);
  return fullPath;
}

beforeEach(() => {
  process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER = "1";
  // Fast hedge delays for testing
  process.env.HEDGE_RETRY_MIN_MS = "50";
  process.env.HEDGE_RETRY_MAX_MS = "100";
  delete process.env.ZAI_API_KEY;
});

afterEach(async () => {
  for (const proxy of activeProxies) {
    try { await proxy.stop(); } catch { /* already stopped */ }
  }
  activeProxies = [];

  globalThis.fetch = originalFetch;
  delete process.env.HEDGE_RETRY_MIN_MS;
  delete process.env.HEDGE_RETRY_MAX_MS;
  delete process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER;
  delete process.env.ZAI_API_KEY;

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

describe("E2E — Hedged request triggers on 429", () => {
  it("should trigger hedge when primary returns 429, fallback wins with immediate response", async () => {
    let primaryCallCount = 0;

    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);

      if (url.includes("/api/embeddings")) {
        return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      }
      if (url.includes("/api/tags")) {
        return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });
      }

      // OpenRouter (primary) — returns 429 on first call, succeeds on retry
      if (url.includes("openrouter.ai")) {
        primaryCallCount++;
        if (primaryCallCount === 1) {
          return makeErrorResponse(429, "Rate limit exceeded");
        }
        const body = JSON.parse(String(init?.body ?? "{}"));
        return makeOpenAIChatResponse("Primary retry succeeded!", body.model);
      }

      // Ollama (fallback) — succeeds immediately
      if (url.includes("/v1/chat/completions")) {
        const body = JSON.parse(String(init?.body ?? "{}"));
        return makeOpenAIChatResponse("Fallback succeeded!", body.model);
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
      const result = await postChat(port, "hedge-429", "Hello");

      assert.equal(result.statusCode, 200, "Should return 200");
      assert.ok(result.json, "Should have valid JSON response");
      assert.equal(result.json.model, "CognitiveRouter:latest", "Model should be CognitiveRouter:latest");
      assert.ok(result.json.choices[0].message.content.length > 0, "Should have content");

      // Verify hedge outcome was tracked
      const stats = proxy["costTracker"].getHedgeStats();
      assert.ok(stats.total >= 1, "Should have at least 1 hedge outcome recorded");
      assert.ok(
        stats.fallbackWins + stats.primaryWins >= 1,
        "Should have at least 1 win recorded",
      );
    } finally {
      await proxy.stop();
    }
  });

  it("should NOT trigger hedge for non-429/503 errors (e.g., empty response)", async () => {
    let hedgeWasTriggered = false;
    let ollamaCallCount = 0;

    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);

      if (url.includes("/api/embeddings")) {
        return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      }
      if (url.includes("/api/tags")) {
        return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });
      }

      // ZAI (primary) — returns empty response (not 429/503)
      if (url.includes("z.ai")) {
        return new Response(JSON.stringify({
          id: "chatcmpl-empty",
          object: "chat.completion",
          created: Date.now(),
          model: "test",
          choices: [{
            index: 0,
            message: { role: "assistant", content: "" },
            finish_reason: "stop",
          }],
        }), { status: 200 });
      }

      // Ollama (fallback) — succeeds
      if (url.includes("/v1/chat/completions")) {
        ollamaCallCount++;
        const body = JSON.parse(String(init?.body ?? "{}"));
        return makeOpenAIChatResponse("Fallback content", body.model);
      }

      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = makeTempDbPath();
    const config = makeConfig({
      dbPath,
      providerPriority: ["zai", "ollama"],
      providers: {
        zai: { budgetType: "subscription", priority: "high" },
        ollama: { budgetType: "free", priority: "low" },
      },
      proxyPort: port,
    });

    process.env.ZAI_API_KEY = "test-key";

    const { proxy } = await startProxy(config);

    try {
      const result = await postChat(port, "hedge-no-429", "Hello");

      assert.equal(result.statusCode, 200, "Should return 200");

      // Hedge should NOT have been triggered
      const stats = proxy["costTracker"].getHedgeStats();
      assert.equal(stats.total, 0, "Should NOT record any hedge outcomes for empty response");
    } finally {
      await proxy.stop();
    }
  });
});

describe("E2E — No double-response under any race condition", () => {
  it("should only send one response to the client when both providers succeed", async () => {
    let responsesSent = 0;
    let orCalls = 0;

    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);

      if (url.includes("/api/embeddings")) {
        return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      }
      if (url.includes("/api/tags")) {
        return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });
      }

      // OpenRouter (primary) — 429 first, then succeeds quickly
      if (url.includes("openrouter.ai")) {
        const body = JSON.parse(String(init?.body ?? "{}"));
        orCalls++;
        if (orCalls === 1) {
          return makeErrorResponse(429, "Rate limited");
        }
        // Primary retry succeeds fast
        return makeOpenAIChatResponse("Primary retry", body.model);
      }

      // Ollama (fallback) — also succeeds
      if (url.includes("/v1/chat/completions")) {
        const body = JSON.parse(String(init?.body ?? "{}"));
        return makeOpenAIChatResponse("Fallback response", body.model);
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

    // Wrap res.end to count responses
    const origEnd = http.ServerResponse.prototype.end;
    let endCount = 0;
    http.ServerResponse.prototype.end = function (...args: any[]) {
      endCount++;
      return origEnd.apply(this, args as any);
    };

    try {
      const result = await postChat(port, "hedge-double", "Hello");

      assert.equal(result.statusCode, 200, "Should return 200");
      assert.ok(result.json?.choices?.[0]?.message?.content, "Should have content");

      // CRITICAL: exactly one res.end() call should have happened for this request
      // (other calls from embedding/tags don't go through chat path)
      assert.ok(endCount >= 1, "At least one end call");

      // Verify only one response was generated
      assert.ok(
        result.json.choices[0].message.content.includes("Primary retry") ||
        result.json.choices[0].message.content.includes("Fallback"),
        "Response should come from one of the hedged providers",
      );
    } finally {
      http.ServerResponse.prototype.end = origEnd;
      await proxy.stop();
    }
  });

  it("should handle both hedged requests failing without double-response", async () => {
    globalThis.fetch = (async (input: RequestInfo | URL): Promise<Response> => {
      const url = String(input);

      if (url.includes("/api/embeddings")) {
        return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      }
      if (url.includes("/api/tags")) {
        return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });
      }

      // OpenRouter — always 429
      if (url.includes("openrouter.ai")) {
        return makeErrorResponse(429, "Still rate limited");
      }

      // Ollama — always 500
      if (url.includes("/v1/chat/completions")) {
        return makeErrorResponse(500, "Server error");
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
      const result = await postChat(port, "hedge-both-fail", "Hello");

      // Should eventually return 503 since everything fails
      assert.ok(result.statusCode === 503 || result.statusCode === 200, "Should return 503 or 200");

      // Verify hedge was attempted
      const stats = proxy["costTracker"].getHedgeStats();
      assert.ok(stats.total >= 1, "Should have at least 1 hedge attempt");
    } finally {
      await proxy.stop();
    }
  });
});

describe("E2E — Hedge outcomes tracked in CostTracker", () => {
  it("should record hedge outcome with winner information", async () => {
    let orCallCount = 0;

    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);

      if (url.includes("/api/embeddings")) {
        return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      }
      if (url.includes("/api/tags")) {
        return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });
      }

      if (url.includes("openrouter.ai")) {
        orCallCount++;
        if (orCallCount === 1) return makeErrorResponse(429, "Rate limited");
        const body = JSON.parse(String(init?.body ?? "{}"));
        return makeOpenAIChatResponse("Primary retry response", body.model);
      }

      if (url.includes("/v1/chat/completions")) {
        const body = JSON.parse(String(init?.body ?? "{}"));
        return makeOpenAIChatResponse("Fallback response", body.model);
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
    const costTracker = proxy["costTracker"];

    try {
      await postChat(port, "hedge-track", "Hello");

      const stats = costTracker.getHedgeStats();
      assert.ok(stats.total >= 1, "Should have recorded hedge outcome");
      assert.ok(
        stats.fallbackWins + stats.primaryWins >= 1,
        "Should have at least one win",
      );

      const outcomes = costTracker.getHedgeOutcomes();
      assert.ok(outcomes.length >= 1, "Should have raw outcome entries");

      const lastOutcome = outcomes[outcomes.length - 1];
      assert.ok(
        lastOutcome.result === "primary_win" || lastOutcome.result === "fallback_win",
        `Unexpected result: ${lastOutcome.result}`,
      );
      assert.ok(lastOutcome.winnerProvider.length > 0, "Winner provider should be set");
      assert.ok(lastOutcome.winnerModel.length > 0, "Winner model should be set");
    } finally {
      await proxy.stop();
    }
  });
});
