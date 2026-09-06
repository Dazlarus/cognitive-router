// tests/decision_source.test.ts — Decision-source taxonomy observability
// (Switchyard easy win #2, docs/switchyard-research-aug25.md §4)
// Run with: npx tsx --test tests/decision_source.test.ts

import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import net from "node:net";
import { rmSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { ProxyServerStreaming } from "../src/proxy-stream.ts";
import { loadConfig, type CognitiveRouterConfig } from "../src/config.ts";
import {
  DECISION_SOURCES,
  computeProxyDecisionSource,
  decisionSourceCounters,
} from "../src/decision_source.ts";
import { providerBackoff } from "../src/failure_classifier.ts";

// ─── Unit: taxonomy shape ──────────────────────────────────────────────────

describe("decision source taxonomy", () => {
  it("contains the fixed taxonomy with no duplicates", () => {
    assert.deepEqual([...DECISION_SOURCES], [
      "scored_pick",
      "provider_fallback",
      "exploration_probe",
      "circuit_breaker_skip",
      "context_window_skip",
      "tool_policy",
      "last_resort_local",
      "all_exhausted",
    ]);
    assert.equal(new Set(DECISION_SOURCES).size, DECISION_SOURCES.length);
  });
});

// ─── Unit: classifier precedence ───────────────────────────────────────────

const baseFacts = {
  served: true,
  routerSource: null,
  usesTools: false,
  contextGuardSkipped: false,
  circuitOrBackoffSkipped: false,
  attemptFailed: false,
  servedByAppendedLastResort: false,
};

describe("computeProxyDecisionSource precedence", () => {
  it("classifies all_exhausted when nothing served", () => {
    assert.equal(
      computeProxyDecisionSource({ ...baseFacts, served: false, attemptFailed: true }),
      "all_exhausted",
    );
  });

  it("classifies tool_policy when tools reshaped the candidate list", () => {
    assert.equal(
      computeProxyDecisionSource({ ...baseFacts, usesTools: true, attemptFailed: true }),
      "tool_policy",
    );
  });

  it("classifies context_window_skip from router pre-filter OR proxy guard", () => {
    assert.equal(
      computeProxyDecisionSource({ ...baseFacts, routerSource: "context_window_skip" }),
      "context_window_skip",
    );
    assert.equal(
      computeProxyDecisionSource({ ...baseFacts, contextGuardSkipped: true }),
      "context_window_skip",
    );
  });

  it("classifies last_resort_local from appended candidate OR router no-models fallback", () => {
    assert.equal(
      computeProxyDecisionSource({ ...baseFacts, servedByAppendedLastResort: true }),
      "last_resort_local",
    );
    assert.equal(
      computeProxyDecisionSource({ ...baseFacts, routerSource: "last_resort_local" }),
      "last_resort_local",
    );
  });

  it("classifies circuit_breaker_skip before provider_fallback", () => {
    assert.equal(
      computeProxyDecisionSource({ ...baseFacts, circuitOrBackoffSkipped: true, attemptFailed: true }),
      "circuit_breaker_skip",
    );
  });

  it("classifies provider_fallback when an earlier attempt failed", () => {
    assert.equal(computeProxyDecisionSource({ ...baseFacts, attemptFailed: true }), "provider_fallback");
  });

  it("defaults to scored_pick", () => {
    assert.equal(computeProxyDecisionSource({ ...baseFacts }), "scored_pick");
  });
});

// ─── Unit: counters ────────────────────────────────────────────────────────

describe("decisionSourceCounters", () => {
  beforeEach(() => decisionSourceCounters.reset());

  it("snapshot includes every taxonomy key plus a total", () => {
    decisionSourceCounters.increment("scored_pick");
    decisionSourceCounters.increment("scored_pick");
    decisionSourceCounters.increment("provider_fallback");
    const snap = decisionSourceCounters.snapshot();
    for (const s of DECISION_SOURCES) assert.ok(s in snap, `${s} missing`);
    assert.equal(snap.scored_pick, 2);
    assert.equal(snap.provider_fallback, 1);
    assert.equal(snap.all_exhausted, 0);
    assert.equal(snap.total, 3);
  });

  it("reset clears counts and bumps sinceIso", () => {
    decisionSourceCounters.increment("tool_policy");
    decisionSourceCounters.reset();
    assert.equal(decisionSourceCounters.snapshot().total, 0);
    assert.ok(decisionSourceCounters.sinceIso());
  });
});

// ─── E2E helpers (mirrors tests/e2e.test.ts) ───────────────────────────────

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

function makeOpenAIChatResponse(content: string, model: string): Response {
  return new Response(JSON.stringify({
    id: `chatcmpl-${Date.now()}`,
    object: "chat.completion",
    created: Date.now(),
    model,
    choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
  }), { status: 200 });
}

function makeOpenAIErrorResponse(status: number, message: string): Response {
  return new Response(JSON.stringify({ error: { message } }), { status });
}

function makeOpenAIStreamResponse(content: string, model: string): Response {
  const words = content.split(" ");
  const chunks = words.map((word, i) => ({
    id: `chatcmpl-stream-${i}`,
    object: "chat.completion.chunk",
    created: Date.now(),
    model,
    choices: [{ index: 0, delta: { content: (i > 0 ? " " : "") + word }, finish_reason: null }],
  }));
  chunks.unshift({
    id: "chatcmpl-stream-0",
    object: "chat.completion.chunk",
    created: Date.now(),
    model,
    choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
  });
  chunks.push({
    id: "chatcmpl-stream-final",
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

type RawResponse = { statusCode: number; headers: http.IncomingHttpHeaders; body: string };

function postRawChat(
  port: number,
  payload: Record<string, any>,
  sessionKey = "test-session",
): Promise<RawResponse> {
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
      res.on("end", () => resolve({ statusCode: res.statusCode ?? 0, headers: res.headers, body: responseBody }));
    });
    req.on("error", reject);
    req.write(body);
    req.end();
  });
}

function getStats(port: number): Promise<any> {
  return new Promise((resolve, reject) => {
    http.get({ hostname: "127.0.0.1", port, path: "/stats" }, (res) => {
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => { body += chunk; });
      res.on("end", () => resolve(JSON.parse(body)));
    }).once("error", reject);
  });
}

const originalFetch = globalThis.fetch;
const originalMaxAttempts = process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER;
const originalZaiKey = process.env.ZAI_API_KEY;

let tempDbPaths: string[] = [];
let activeProxies: ProxyServerStreaming[] = [];

function makeTempDbPath(): string {
  const name = `tmp/ds-${Date.now()}-${Math.random().toString(36).slice(2)}.db`;
  const fullPath = resolve(name);
  mkdirSync(resolve(fullPath, ".."), { recursive: true });
  tempDbPaths.push(fullPath);
  return fullPath;
}

beforeEach(() => {
  process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER = "1";
  delete process.env.ZAI_API_KEY;
  decisionSourceCounters.reset();
  providerBackoff.clearAll(); // module-level singleton — reset cross-test backoff state
});

afterEach(async () => {
  for (const proxy of activeProxies) {
    try { await proxy.stop(); } catch { /* already stopped */ }
  }
  activeProxies = [];
  globalThis.fetch = originalFetch;
  providerBackoff.clearAll();
  if (originalMaxAttempts === undefined) delete process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER;
  else process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER = originalMaxAttempts;
  if (originalZaiKey === undefined) delete process.env.ZAI_API_KEY;
  else process.env.ZAI_API_KEY = originalZaiKey;
  for (const p of tempDbPaths) {
    for (const suffix of ["", "-wal", "-shm"]) {
      try { rmSync(p + suffix, { force: true }); } catch { /* ignore */ }
    }
  }
  tempDbPaths = [];
});

async function startProxy(config: CognitiveRouterConfig): Promise<ProxyServerStreaming> {
  const proxy = new ProxyServerStreaming(config);
  await proxy.start();
  activeProxies.push(proxy);
  return proxy;
}

const HEADER = "x-model-router-selected-model";

function countFor(source: string): number {
  return decisionSourceCounters.snapshot()[source] ?? 0;
}

/** Ollama-only fetch mock: every chat completion succeeds and reports the model it received. */
function mockAllChatSucceed(capture: { model: string }): typeof fetch {
  return (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = String(input);
    if (url.includes("/api/embeddings")) return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
    if (url.includes("/api/tags")) return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });
    if (url.includes("/v1/chat/completions")) {
      const body = JSON.parse(String(init?.body ?? "{}"));
      capture.model = body.model;
      return body.stream === true
        ? makeOpenAIStreamResponse("streamed ok", body.model)
        : makeOpenAIChatResponse("ok", body.model);
    }
    return new Response(JSON.stringify({ data: [] }), { status: 200 });
  }) as typeof fetch;
}

// ─── E2E: header + counters ────────────────────────────────────────────────

describe("E2E — x-model-router-selected-model header", () => {
  it("non-streaming success carries the header naming the serving model (scored_pick)", async () => {
    const capture: { model: string } = { model: "" };
    globalThis.fetch = mockAllChatSucceed(capture);

    const port = await getFreePort();
    await startProxy(makeConfig({
      dbPath: makeTempDbPath(),
      providerPriority: ["ollama"],
      providers: { ollama: { budgetType: "free", priority: "high" } },
      proxyPort: port,
    }));

    const res = await postRawChat(port, {
      model: "CognitiveRouter:latest",
      messages: [{ role: "user", content: "Hello" }],
      stream: false,
    }, "ds-header-nonstream");

    assert.equal(res.statusCode, 200);
    assert.equal(res.headers[HEADER], capture.model, "header must name the model that actually served");
    assert.ok(capture.model && capture.model !== "CognitiveRouter:latest");
    assert.equal(countFor("scored_pick"), 1, "first-choice serve counts as scored_pick");
    assert.equal(countFor("all_exhausted"), 0);
  });

  it("streaming SSE carries the header, set before the first data frame", async () => {
    const capture: { model: string } = { model: "" };
    globalThis.fetch = mockAllChatSucceed(capture);

    const port = await getFreePort();
    await startProxy(makeConfig({
      dbPath: makeTempDbPath(),
      providerPriority: ["ollama"],
      providers: { ollama: { budgetType: "free", priority: "high" } },
      proxyPort: port,
    }));

    const res = await postRawChat(port, {
      model: "CognitiveRouter:latest",
      messages: [{ role: "user", content: "Stream me" }],
      stream: true,
    }, "ds-header-stream");

    assert.equal(res.statusCode, 200);
    assert.equal(res.headers["content-type"], "text/event-stream");
    // Header is part of the response head, so it necessarily preceded all SSE data.
    assert.equal(res.headers[HEADER], capture.model, "header must name the model that actually served");
    assert.ok(res.body.includes("data: "), "body should contain SSE frames");
    assert.ok(res.body.trimEnd().endsWith("data: [DONE]"));
    assert.equal(countFor("scored_pick"), 1);
  });

  it("provider_fallback after 429 carries the header of the fallback that served", async () => {
    const served: { model: string } = { model: "" };
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);
      if (url.includes("/api/embeddings")) return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      if (url.includes("/api/tags")) return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });
      if (url.includes("openrouter.ai/api/v1/chat/completions")) {
        return makeOpenAIErrorResponse(429, "Rate limit exceeded");
      }
      if (url.includes("/v1/chat/completions")) {
        const body = JSON.parse(String(init?.body ?? "{}"));
        served.model = body.model;
        return makeOpenAIChatResponse("recovered via fallback", body.model);
      }
      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    await startProxy(makeConfig({
      dbPath: makeTempDbPath(),
      providerPriority: ["openrouter", "ollama"],
      providers: {
        openrouter: { budgetType: "free", priority: "high" },
        ollama: { budgetType: "free", priority: "low" },
      },
      proxyPort: port,
    }));

    const res = await postRawChat(port, {
      model: "CognitiveRouter:latest",
      messages: [{ role: "user", content: "Trigger failover" }],
      stream: false,
    }, "ds-fallback");

    assert.equal(res.statusCode, 200);
    assert.equal(res.headers[HEADER], served.model, "header must name the fallback model that served");
    assert.equal(countFor("provider_fallback"), 1, "earlier attempt failed → provider_fallback");
  });

  it("all-exhausted sets no header and counts all_exhausted", async () => {
    globalThis.fetch = (async (input: RequestInfo | URL): Promise<Response> => {
      const url = String(input);
      if (url.includes("/api/embeddings")) return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      if (url.includes("/api/tags")) return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });
      if (url.includes("/v1/chat/completions")) return makeOpenAIErrorResponse(500, "Internal server error");
      if (url.includes("openrouter.ai")) return makeOpenAIErrorResponse(429, "Rate limited");
      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    await startProxy(makeConfig({
      dbPath: makeTempDbPath(),
      providerPriority: ["openrouter", "ollama"],
      providers: {
        openrouter: { budgetType: "free", priority: "high" },
        ollama: { budgetType: "free", priority: "low" },
      },
      proxyPort: port,
    }));

    const res = await postRawChat(port, {
      model: "CognitiveRouter:latest",
      messages: [{ role: "user", content: "This will fail" }],
      stream: false,
    }, "ds-exhausted");

    assert.equal(res.statusCode, 503);
    assert.equal(res.headers[HEADER], undefined, "no selected-model header when nothing served");
    assert.equal(countFor("all_exhausted"), 1);
  });

  it("tool-bearing request counts tool_policy", async () => {
    const served: { model: string } = { model: "" };
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);
      if (url.includes("/api/embeddings")) return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      if (url.includes("/api/tags")) return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });
      if (url.includes("openrouter.ai/api/v1/chat/completions")) {
        const body = JSON.parse(String(init?.body ?? "{}"));
        served.model = body.model;
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
    await startProxy(makeConfig({
      dbPath: makeTempDbPath(),
      providerPriority: ["openrouter", "ollama"],
      providers: {
        openrouter: { budgetType: "free", priority: "high" },
        ollama: { budgetType: "free", priority: "low" },
      },
      proxyPort: port,
    }));

    const res = await postRawChat(port, {
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
    }, "ds-tools");

    assert.equal(res.statusCode, 200);
    assert.equal(res.headers[HEADER], served.model);
    assert.equal(countFor("tool_policy"), 1, "tools reshaped the candidate list → tool_policy");
  });
});

describe("E2E — /stats decision-source counters", () => {
  it("exposes routing.decisionSources with the full taxonomy and counts", async () => {
    const capture: { model: string } = { model: "" };
    globalThis.fetch = mockAllChatSucceed(capture);

    const port = await getFreePort();
    await startProxy(makeConfig({
      dbPath: makeTempDbPath(),
      providerPriority: ["ollama"],
      providers: { ollama: { budgetType: "free", priority: "high" } },
      proxyPort: port,
    }));

    const stats0 = await getStats(port);
    assert.ok(stats0.routing.decisionSources, "stats should expose routing.decisionSources");
    assert.deepEqual(stats0.routing.decisionSources.taxonomy, [...DECISION_SOURCES]);
    assert.equal(stats0.routing.decisionSources.counts.total, 0, "fresh proxy has zero decisions");
    assert.ok(stats0.routing.decisionSources.sinceIso, "sinceIso present");

    const res = await postRawChat(port, {
      model: "CognitiveRouter:latest",
      messages: [{ role: "user", content: "One routed request" }],
      stream: false,
    }, "ds-stats");
    assert.equal(res.statusCode, 200);

    const stats1 = await getStats(port);
    const counts = stats1.routing.decisionSources.counts;
    for (const s of DECISION_SOURCES) assert.ok(s in counts, `${s} missing from stats counts`);
    assert.equal(counts.scored_pick, 1);
    assert.equal(counts.total, 1);
  });
});
