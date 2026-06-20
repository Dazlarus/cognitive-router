// tests/router.test.ts — Multi-session, multi-agent, and edge case tests
// Run with: node --import tsx tests/router.test.ts
// Or:       npx tsx --test tests/router.test.ts

import { describe, it, beforeEach, afterEach, mock } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import net from "node:net";
import { RoutingEngine } from "../src/router.ts";
import { CostTracker } from "../src/cost_tracker.ts";
import { ModelRegistry, type ModelCapability } from "../src/model_registry.ts";
import { DBService } from "../src/db_service.ts";
import { loadConfig, type CognitiveRouterConfig } from "../src/config.ts";
import { IntentClassifier, type Classification } from "../src/classifier.ts";
import { ProxyServerStreaming, fallbackModelForProvider } from "../src/proxy-stream.ts";
import { generationRoutingExclusionReason, isGenerationModel, modelTaskTypes } from "../src/model_policy.ts";

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

// In-memory DB mock
function makeMockDB(): DBService {
  const decisions: any[] = [];
  const outcomes: any[] = [];
  return {
    initializeSchema: async () => {},
    recordDecision: (d: any) => decisions.push(d),
    recordCallOutcome: (o: any) => outcomes.push(o),
    recordRetry: (r: any) => {},
    getDecisionByRequestId: (id: string) => decisions.find(d => d.requestId === id) ?? null,
    getRetryCount: (id: string) => 0,
    close: () => {},
    // expose for assertions
    _decisions: decisions,
    _outcomes: outcomes,
  } as any;
}

function makeClassification(intent: string = "conversation", confidence: number = 0.9): Classification {
  return { intent, confidence };
}

async function getFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      server.close(() => {
        if (address && typeof address === "object") {
          resolve(address.port);
        } else {
          reject(new Error("Unable to allocate test port"));
        }
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
      hostname: "127.0.0.1",
      port,
      path: "/v1/chat/completions",
      method: "POST",
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
      hostname: "127.0.0.1",
      port,
      path: "/v1/chat/completions",
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(body),
        "x-openclaw-session-key": sessionKey,
      },
    }, (res) => {
      let responseBody = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => { responseBody += chunk; });
      res.on("end", () => {
        resolve({ statusCode: res.statusCode ?? 0, body: responseBody });
      });
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
      hostname: "127.0.0.1",
      port,
      path: "/v1/chat/completions",
      method: "POST",
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
      hostname: "127.0.0.1",
      port,
      path: "/v1/models",
      method: "GET",
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
      hostname: "127.0.0.1",
      port,
      path: "/stats",
      method: "GET",
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

// ─── Tests ──────────────────────────────────────────────────

describe("RoutingEngine — Multi-Session Isolation", () => {
  let registry: ModelRegistry;
  let costTracker: CostTracker;
  let db: DBService;
  let config: CognitiveRouterConfig;
  let router: RoutingEngine;

  beforeEach(async () => {
    config = makeConfig();
    db = makeMockDB();
    costTracker = new CostTracker(db, config);
    await costTracker.refreshProviderStatus();
    registry = new ModelRegistry(db, config);
    await registry.loadCachedState();
    router = new RoutingEngine(registry, costTracker, db, config);
  });

  it("should handle two concurrent sessions with different intents independently", async () => {
    const sessionA = "agent:main:main";
    const sessionB = "discord:123:456";

    const codingTask: Classification = { intent: "coding", confidence: 0.95 };
    const chatTask: Classification = { intent: "conversation", confidence: 0.85 };

    const [decisionA, decisionB] = await Promise.all([
      router.decide(codingTask, sessionA, {}),
      router.decide(chatTask, sessionB, {}),
    ]);

    assert.ok(decisionA, "Session A should get a decision");
    assert.ok(decisionB, "Session B should get a decision");

    // Both should route to zai (healthy, highest priority subscription)
    assert.equal(decisionA!.provider, "zai");
    assert.equal(decisionB!.provider, "zai");

    // But potentially different models based on intent
    // Coding should prefer higher capability model
    assert.ok(decisionA!.scores.capability >= 0.8, "Coding should get high capability score");
  });

  it("should not let one session's failure affect another session's routing", async () => {
    const sessionA = "agent:main:main";
    const sessionB = "subagent:worker:1";

    // Session A triggers a rate limit on zai
    await costTracker.recordCall("zai", { durationMs: 1000, outcome: "rate_limit" });

    // Session A should now avoid zai
    const decisionA = await router.decide(makeClassification(), sessionA, {});
    assert.ok(decisionA, "Session A should still get a decision");
    assert.notEqual(decisionA!.provider, "zai", "Session A should avoid rate-limited zai");

    // Session B should ALSO avoid zai (circuit breaker is provider-level, not session-level)
    // This is intentional — the provider is objectively rate-limited
    const decisionB = await router.decide(makeClassification(), sessionB, {});
    assert.ok(decisionB, "Session B should get a decision");
    assert.notEqual(decisionB!.provider, "zai", "Session B should also avoid rate-limited zai");
  });

  it("should track request IDs per-session, not globally", async () => {
    // This tests the index.ts sessionRequestIds Map behavior
    // We simulate what the hook does: store requestId per session

    const sessionRequestIds = new Map<string, string>();

    const sessionA = "agent:main:main";
    const sessionB = "subagent:worker:2";

    // Session A routes
    const reqIdA = `req_A_${Date.now()}`;
    sessionRequestIds.set(sessionA, reqIdA);

    // Session B routes BEFORE A's model_call_ended fires
    const reqIdB = `req_B_${Date.now()}`;
    sessionRequestIds.set(sessionB, reqIdB);

    // Both should be tracked independently
    assert.equal(sessionRequestIds.get(sessionA), reqIdA);
    assert.equal(sessionRequestIds.get(sessionB), reqIdB);

    // Session A's call ends — look up A's request ID (not B's)
    const lookupA = sessionRequestIds.get(sessionA);
    assert.equal(lookupA, reqIdA, "Session A lookup should return A's request ID");

    // Session B's call ends — look up B's request ID (not A's)
    const lookupB = sessionRequestIds.get(sessionB);
    assert.equal(lookupB, reqIdB, "Session B lookup should return B's request ID");

    // Clear A on agent_end
    sessionRequestIds.delete(sessionA);
    assert.equal(sessionRequestIds.get(sessionA), undefined, "A cleared after agent_end");
    assert.equal(sessionRequestIds.get(sessionB), reqIdB, "B still tracked");
  });
});

describe("RoutingEngine — Provider Degradation & Failover", () => {
  let registry: ModelRegistry;
  let costTracker: CostTracker;
  let db: DBService;
  let config: CognitiveRouterConfig;
  let router: RoutingEngine;

  beforeEach(async () => {
    config = makeConfig();
    db = makeMockDB();
    costTracker = new CostTracker(db, config);
    await costTracker.refreshProviderStatus();
    registry = new ModelRegistry(db, config);
    await registry.loadCachedState();
    router = new RoutingEngine(registry, costTracker, db, config);
  });

  it("should return null when ALL providers are circuit_open (don't override)", async () => {
    // Kill all providers
    for (const provider of config.providerPriority) {
      await costTracker.recordCall(provider, { durationMs: 1000, outcome: "rate_limit" });
    }

    const decision = await router.decide(makeClassification(), "test-session", {});
    assert.equal(decision, null, "Should return null when all providers are unhealthy");
  });

  it("should failover to OpenRouter when ZAI is rate-limited", async () => {
    await costTracker.recordCall("zai", { durationMs: 500, outcome: "rate_limit" });

    const decision = await router.decide(makeClassification(), "test-session", {});
    assert.ok(decision, "Should get a decision");
    assert.notEqual(decision!.provider, "zai", "Should not use rate-limited zai");
  });

  it("should failover through the full priority chain", async () => {
    // Kill zai and openrouter
    await costTracker.recordCall("zai", { durationMs: 500, outcome: "rate_limit" });
    await costTracker.recordCall("openrouter", { durationMs: 500, outcome: "rate_limit" });

    const decision = await router.decide(makeClassification(), "test-session", {});
    assert.ok(decision, "Should get a decision");
    assert.notEqual(decision!.provider, "zai");
    assert.notEqual(decision!.provider, "openrouter");
  });

  it("should recover a provider after backoff period elapses", async () => {
    // Rate limit zai
    await costTracker.recordCall("zai", { durationMs: 500, outcome: "rate_limit" });
    assert.equal(costTracker.isAvailable("zai"), false);

    // Get state and simulate time passing (tier 1 = 30s)
    const state = costTracker.getProviderState("zai")!;
    state.lastFailureTime = Date.now() - 31_000; // 31 seconds ago

    // Should now be in throttled (half-open) state
    const updated = costTracker.getProviderState("zai")!;
    assert.equal(updated.status, "throttled", "Should be throttled after backoff elapses");
  });

  it("should reset to healthy on successful call after failure", async () => {
    // Cause some failures
    await costTracker.recordCall("zai", { durationMs: 500, outcome: "error" });
    await costTracker.recordCall("zai", { durationMs: 500, outcome: "error" });

    const state = costTracker.getProviderState("zai")!;
    assert.equal(state.consecutiveFailures, 2);

    // Successful call
    await costTracker.recordCall("zai", { durationMs: 500, outcome: "success" });

    const recovered = costTracker.getProviderState("zai")!;
    assert.equal(recovered.consecutiveFailures, 0, "Failures should reset on success");
    assert.equal(recovered.status, "healthy", "Should be healthy after success");
    assert.equal(recovered.backoffTier, 0, "Backoff tier should reset on success");
  });

  it("should escalate backoff tier progressively on repeated rate limits", async () => {
    const tiers: number[] = [];

    for (let i = 0; i < 4; i++) {
      await costTracker.recordCall("zai", { durationMs: 100, outcome: "rate_limit" });
      tiers.push(costTracker.getProviderState("zai")!.backoffTier);
    }

    assert.deepEqual(tiers, [1, 2, 3, 4], "Should escalate through all tiers");
    assert.equal(costTracker.getProviderState("zai")!.status, "circuit_open");
  });

  it("should cap backoff at tier 4 (30 minutes)", async () => {
    for (let i = 0; i < 10; i++) {
      await costTracker.recordCall("zai", { durationMs: 100, outcome: "rate_limit" });
    }

    const state = costTracker.getProviderState("zai")!;
    assert.equal(state.backoffTier, 4, "Should cap at tier 4");
  });

  it("should isolate Z.AI rate limits to the failed model when model is supplied", async () => {
    await costTracker.recordCall("zai", { durationMs: 500, outcome: "rate_limit" }, "glm-5.2");

    assert.equal(costTracker.isAvailable("zai"), true, "Z.AI provider should remain available");
    assert.equal(costTracker.isAvailable("zai", "glm-5.2"), false, "Failed model should be circuit-open");
    assert.equal(costTracker.isAvailable("zai", "glm-5.1"), true, "Sibling Z.AI model should remain available");
  });

  it("should route to another Z.AI model when the top Z.AI model is circuit-open", async () => {
    await costTracker.recordCall("zai", { durationMs: 500, outcome: "rate_limit" }, "glm-5.2");

    const decision = await router.decide(makeClassification("coding", 0.95), "test-session", {});

    assert.ok(decision, "Should still get a Z.AI decision");
    assert.equal(decision!.provider, "zai");
    assert.notEqual(decision!.model, "glm-5.2");
    assert.equal(decision!.model, "glm-5.1");
  });
});

describe("RoutingEngine — Circuit Breaker Tiers", () => {
  let config: CognitiveRouterConfig;
  let db: DBService;
  let costTracker: CostTracker;

  beforeEach(async () => {
    config = makeConfig();
    db = makeMockDB();
    costTracker = new CostTracker(db, config);
    await costTracker.refreshProviderStatus();
  });

  it("should apply correct backoff durations per tier", async () => {
    // This tests that all providers use the same unified backoff schedule
    const tracker = new CostTracker(db, config);
    await tracker.refreshProviderStatus();

    // Verify backoff is max 30 minutes (tier 4)
    const BACKOFF_TIER_4_MS = 1_800_000; // 30 minutes

    // Record 4 rate limits to reach tier 4
    for (let i = 0; i < 4; i++) {
      tracker.recordCall("zai", { durationMs: 100, outcome: "rate_limit" });
    }

    const state = tracker.getProviderState("zai")!;
    assert.equal(state.backoffTier, 4);

    // The lastFailureTime should be ~now
    const elapsed = Date.now() - state.lastFailureTime;
    assert.ok(elapsed < 5000, "lastFailureTime should be recent");

    // Provider should NOT be available (circuit_open, backoff not elapsed)
    assert.equal(tracker.isAvailable("zai"), false);
  });

  it("should use same backoff schedule for subscription and non-subscription providers", async () => {
    const tracker = new CostTracker(db, config);
    await tracker.refreshProviderStatus();

    // Rate limit both zai (subscription) and openrouter (free)
    await tracker.recordCall("zai", { durationMs: 100, outcome: "rate_limit" });
    await tracker.recordCall("openrouter", { durationMs: 100, outcome: "rate_limit" });

    const zaiState = tracker.getProviderState("zai")!;
    const orState = tracker.getProviderState("openrouter")!;

    assert.equal(zaiState.backoffTier, 1, "ZAI tier 1 after 1 rate limit");
    assert.equal(orState.backoffTier, 1, "OpenRouter tier 1 after 1 rate limit");

    // Both should have same status
    assert.equal(zaiState.status, "circuit_open");
    assert.equal(orState.status, "circuit_open");
  });
});

describe("RoutingEngine — Scoring Edge Cases", () => {
  let registry: ModelRegistry;
  let costTracker: CostTracker;
  let db: DBService;
  let config: CognitiveRouterConfig;
  let router: RoutingEngine;

  beforeEach(async () => {
    config = makeConfig();
    db = makeMockDB();
    costTracker = new CostTracker(db, config);
    await costTracker.refreshProviderStatus();
    registry = new ModelRegistry(db, config);
    await registry.loadCachedState();
    router = new RoutingEngine(registry, costTracker, db, config);
  });

  it("should handle empty provider priority list gracefully", async () => {
    const emptyConfig = makeConfig({ providerPriority: [] });
    const emptyRouter = new RoutingEngine(registry, costTracker, db, emptyConfig);

    const decision = await emptyRouter.decide(makeClassification(), "test", {});
    // Should either return null or a fallback — not crash
    assert.ok(decision === null || decision.provider === "ollama",
      "Empty priority list should return null or fallback");
  });

  it("should handle single provider in priority list", async () => {
    const singleConfig = makeConfig({ providerPriority: ["zai"] });
    const singleRouter = new RoutingEngine(registry, costTracker, db, singleConfig);

    const decision = await singleRouter.decide(makeClassification(), "test", {});
    assert.ok(decision, "Should get a decision with single provider");
    assert.equal(decision!.provider, "zai");
  });

  it("should handle single provider that is unhealthy", async () => {
    const singleConfig = makeConfig({ providerPriority: ["zai"] });
    const singleRouter = new RoutingEngine(registry, costTracker, db, singleConfig);

    await costTracker.recordCall("zai", { durationMs: 100, outcome: "rate_limit" });

    const decision = await singleRouter.decide(makeClassification(), "test", {});
    assert.equal(decision, null, "Single unhealthy provider should return null");
  });

  it("should handle very low confidence classification", async () => {
    const decision = await router.decide(
      makeClassification("unknown", 0.01),
      "test",
      {},
    );
    assert.ok(decision, "Should still route even with very low confidence");
  });

  it("should respect manual overrides", async () => {
    const overrideConfig = makeConfig({
      overrides: [
        { intent: "coding", provider: "ollama", model: "gemma4:latest", reason: "test override" },
      ],
    });
    const overrideRouter = new RoutingEngine(registry, costTracker, db, overrideConfig);

    const decision = await overrideRouter.decide(
      makeClassification("coding", 0.95),
      "test",
      {},
    );
    assert.ok(decision);
    assert.equal(decision!.provider, "ollama");
    assert.equal(decision!.model, "gemma4:latest");
    assert.ok(decision!.rationale.includes("override"));
  });

  it("should prefer cheaper model when scores are very close", async () => {
    const decision = await router.decide(makeClassification("conversation", 0.85), "test", {});
    assert.ok(decision);
    // With default config, zai should win for conversation
    // (subscription = highest cost score)
    assert.equal(decision!.provider, "zai");
  });
});

describe("CostTracker — Reliability Scoring", () => {
  let config: CognitiveRouterConfig;
  let db: DBService;
  let costTracker: CostTracker;

  beforeEach(async () => {
    config = makeConfig();
    db = makeMockDB();
    costTracker = new CostTracker(db, config);
    await costTracker.refreshProviderStatus();
  });

  it("should return 1.0 reliability for healthy provider with no failures", async () => {
    const tracker = new CostTracker(db, config);
    await tracker.refreshProviderStatus();
    const score = tracker.getReliabilityScore("zai");
    assert.equal(score, 1.0);
  });

  it("should return 0 reliability for circuit_open provider", async () => {
    const tracker = new CostTracker(db, config);
    await tracker.refreshProviderStatus();
    await tracker.recordCall("zai", { durationMs: 100, outcome: "rate_limit" });
    const score = tracker.getReliabilityScore("zai");
    assert.equal(score, 0, "Circuit-open provider should have 0 reliability");
  });

  it("should reduce reliability on consecutive errors (before circuit opens)", async () => {
    const tracker = new CostTracker(db, config);
    await tracker.refreshProviderStatus();

    // 1 error → throttled, not circuit_open
    await tracker.recordCall("zai", { durationMs: 100, outcome: "error" });
    let score = tracker.getReliabilityScore("zai");
    assert.ok(score > 0 && score < 1, `1 error should reduce score: got ${score}`);
    assert.ok(score >= 0.8, `1 error should be ~0.85: got ${score}`);

    // 2 errors
    await tracker.recordCall("zai", { durationMs: 100, outcome: "error" });
    score = tracker.getReliabilityScore("zai");
    assert.ok(score > 0 && score < 1, `2 errors should reduce score: got ${score}`);
    assert.ok(score >= 0.65, `2 errors should be ~0.70: got ${score}`);
  });

  it("should penalize subscription providers equally in reliability scoring", async () => {
    const tracker = new CostTracker(db, config);
    await tracker.refreshProviderStatus();

    // 1 rate limit on each
    await tracker.recordCall("zai", { durationMs: 100, outcome: "rate_limit" });
    await tracker.recordCall("openrouter", { durationMs: 100, outcome: "rate_limit" });

    const zaiScore = tracker.getReliabilityScore("zai");
    const orScore = tracker.getReliabilityScore("openrouter");

    // Both should be 0 (circuit_open from rate limit)
    assert.equal(zaiScore, 0);
    assert.equal(orScore, 0);
  });
});

describe("CostTracker — Throttled State Handling", () => {
  let config: CognitiveRouterConfig;
  let db: DBService;
  let costTracker: CostTracker;

  beforeEach(async () => {
    config = makeConfig();
    db = makeMockDB();
    costTracker = new CostTracker(db, config);
    await costTracker.refreshProviderStatus();
  });

  it("should mark provider as throttled after first generic error", async () => {
    const tracker = new CostTracker(db, config);
    await tracker.refreshProviderStatus();
    await tracker.recordCall("zai", { durationMs: 100, outcome: "error" });

    const state = tracker.getProviderState("zai")!;
    assert.equal(state.status, "throttled");
    assert.equal(state.consecutiveFailures, 1);
    assert.equal(state.backoffTier, 0, "Generic errors don't escalate backoff tier");
  });

  it("should open circuit after 3 consecutive generic errors", async () => {
    const tracker = new CostTracker(db, config);
    await tracker.refreshProviderStatus();

    for (let i = 0; i < 3; i++) {
      await tracker.recordCall("zai", { durationMs: 100, outcome: "error" });
    }

    const state = tracker.getProviderState("zai")!;
    assert.equal(state.status, "circuit_open");
    assert.ok(state.backoffTier > 0, "Should have escalated backoff tier");
  });

  it("should apply throttle penalty to cost score", async () => {
    const tracker = new CostTracker(db, config);
    await tracker.refreshProviderStatus();

    const healthyScore = tracker.getCostScore("zai");

    await tracker.recordCall("zai", { durationMs: 100, outcome: "error" });

    const throttledScore = tracker.getCostScore("zai");
    assert.ok(throttledScore < healthyScore,
      `Throttled cost (${throttledScore}) should be < healthy (${healthyScore})`);
  });

  it("should transition from circuit_open to throttled after backoff period", async () => {
    const tracker = new CostTracker(db, config);
    await tracker.refreshProviderStatus();

    // Rate limit → circuit_open, tier 1 (30s backoff)
    await tracker.recordCall("zai", { durationMs: 100, outcome: "rate_limit" });
    let state = tracker.getProviderState("zai")!;
    assert.equal(state.status, "circuit_open");

    // Simulate time passing
    state.lastFailureTime = Date.now() - 31_000;

    // Next access should transition to throttled
    state = tracker.getProviderState("zai")!;
    assert.equal(state.status, "throttled", "Should be half-open (throttled) after backoff");
  });
});

describe("Multi-Agent Isolation — Subagent Scenarios", () => {
  let registry: ModelRegistry;
  let costTracker: CostTracker;
  let db: DBService;
  let config: CognitiveRouterConfig;
  let router: RoutingEngine;

  beforeEach(async () => {
    config = makeConfig();
    db = makeMockDB();
    costTracker = new CostTracker(db, config);
    await costTracker.refreshProviderStatus();
    registry = new ModelRegistry(db, config);
    await registry.loadCachedState();
    router = new RoutingEngine(registry, costTracker, db, config);
  });

  it("should handle 5 concurrent subagent sessions without interference", async () => {
    const sessions = Array.from({ length: 5 }, (_, i) => `subagent:worker:${i}`);
    const intents: Classification[] = [
      { intent: "coding", confidence: 0.9 },
      { intent: "analysis", confidence: 0.85 },
      { intent: "conversation", confidence: 0.95 },
      { intent: "creative", confidence: 0.8 },
      { intent: "retrieval", confidence: 0.88 },
    ];

    const decisions = await Promise.all(
      sessions.map((s, i) => router.decide(intents[i], s, {})),
    );

    for (const d of decisions) {
      assert.ok(d, "Every session should get a decision");
      assert.ok(config.providerPriority.includes(d!.provider),
        `Provider ${d!.provider} should be in priority list`);
    }

    // All should route to zai (all healthy)
    for (let i = 0; i < decisions.length; i++) {
      assert.equal(decisions[i]!.provider, "zai",
        `Session ${sessions[i]} should route to zai when healthy`);
    }
  });

  it("should handle rapid sequential requests from different sessions", async () => {
    const sessions = Array.from({ length: 10 }, (_, i) => `session:${i}`);

    for (const s of sessions) {
      const decision = await router.decide(makeClassification(), s, {});
      assert.ok(decision);
      assert.equal(decision!.provider, "zai");
    }
  });

  it("should handle mixed subagent + main session routing under provider degradation", async () => {
    // Degrade zai slightly (1 error → throttled)
    await costTracker.recordCall("zai", { durationMs: 100, outcome: "error" });

    const mainSession = "agent:main:main";
    const subagent1 = "subagent:research:1";
    const subagent2 = "subagent:coding:2";

    const [main, sub1, sub2] = await Promise.all([
      router.decide(makeClassification("conversation"), mainSession, {}),
      router.decide(makeClassification("retrieval"), subagent1, {}),
      router.decide(makeClassification("coding"), subagent2, {}),
    ]);

    // All should still route to zai (throttled but still highest score)
    for (const [d, name] of [[main, "main"], [sub1, "sub1"], [sub2, "sub2"]] as const) {
      assert.ok(d, `${name} should get a decision`);
      assert.equal(d!.provider, "zai", `${name} should still use zai when throttled (not dead)`);
    }
  });
});

describe("Standalone Proxy — Concurrent Session Handling", () => {
  const originalFetch = globalThis.fetch;
  const originalMaxAttempts = process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER;
  const originalRequestTimeout = process.env.ROUTER_REQUEST_TIMEOUT_MS;
  const originalZaiKey = process.env.ZAI_API_KEY;
  const originalOpenRouterToolModel = process.env.ROUTER_OPENROUTER_TOOL_MODEL;
  const originalOpenRouterFallbackModel = process.env.ROUTER_OPENROUTER_FALLBACK_MODEL;

  afterEach(() => {
    globalThis.fetch = originalFetch;
    if (originalMaxAttempts === undefined) {
      delete process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER;
    } else {
      process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER = originalMaxAttempts;
    }
    if (originalRequestTimeout === undefined) {
      delete process.env.ROUTER_REQUEST_TIMEOUT_MS;
    } else {
      process.env.ROUTER_REQUEST_TIMEOUT_MS = originalRequestTimeout;
    }
    if (originalZaiKey === undefined) {
      delete process.env.ZAI_API_KEY;
    } else {
      process.env.ZAI_API_KEY = originalZaiKey;
    }
    if (originalOpenRouterToolModel === undefined) {
      delete process.env.ROUTER_OPENROUTER_TOOL_MODEL;
    } else {
      process.env.ROUTER_OPENROUTER_TOOL_MODEL = originalOpenRouterToolModel;
    }
    if (originalOpenRouterFallbackModel === undefined) {
      delete process.env.ROUTER_OPENROUTER_FALLBACK_MODEL;
    } else {
      process.env.ROUTER_OPENROUTER_FALLBACK_MODEL = originalOpenRouterFallbackModel;
    }
  });

  it("returns a router timeout before a slow provider can hit the outer client timeout", async () => {
    process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER = "1";
    process.env.ROUTER_REQUEST_TIMEOUT_MS = "25";
    delete process.env.ZAI_API_KEY;

    let chatCalls = 0;

    globalThis.fetch = (async (input: RequestInfo | URL): Promise<Response> => {
      const url = String(input);

      if (url.includes("/api/tags")) {
        return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });
      }

      if (url.includes("/api/embeddings")) {
        return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      }

      if (url.includes("/v1/chat/completions")) {
        chatCalls++;
        await new Promise((resolve) => setTimeout(resolve, 200));
        return new Response(JSON.stringify({
          id: "chatcmpl-too-late",
          object: "chat.completion",
          created: Date.now(),
          model: "gemma4:latest",
          choices: [{
            index: 0,
            message: { role: "assistant", content: "too late" },
            finish_reason: "stop",
          }],
        }), { status: 200 });
      }

      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = `tmp/proxy-request-timeout-${Date.now()}-${Math.random().toString(36).slice(2)}.db`;
    const config = makeConfig({
      dbPath,
      providerPriority: ["ollama"],
      providers: {
        ollama: { budgetType: "free", priority: "high" },
      },
      proxyPort: port,
    });

    const proxy = new ProxyServerStreaming(config);
    await proxy.start();

    try {
      const startedAt = Date.now();
      const response = await postRawChat(port, {
        model: "CognitiveRouter:latest",
        messages: [{ role: "user", content: "slow request" }],
        stream: false,
      });
      const elapsedMs = Date.now() - startedAt;

      assert.equal(response.statusCode, 503);
      assert.equal(chatCalls, 1);
      assert.ok(response.body.includes("router_request_timeout"));
      assert.ok(elapsedMs < 150, `Router should return before provider resolves; elapsed=${elapsedMs}ms`);
    } finally {
      await proxy.stop();
    }
  });

  it("records concurrent HTTP chat requests under their own session keys", async () => {
    process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER = "1";
    delete process.env.ZAI_API_KEY;

    let activeChatCalls = 0;
    let maxActiveChatCalls = 0;

    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);

      if (url.includes("/api/tags")) {
        return new Response(JSON.stringify({ models: [
          { name: "gemma4:latest" },
          { name: "nomic-embed-text:latest" },
        ] }), { status: 200 });
      }

      if (url.includes("/api/embeddings")) {
        return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      }

      if (url.includes("/v1/chat/completions")) {
        activeChatCalls++;
        maxActiveChatCalls = Math.max(maxActiveChatCalls, activeChatCalls);
        await new Promise((resolve) => setTimeout(resolve, 25));
        activeChatCalls--;

        const request = JSON.parse(String(init?.body ?? "{}"));
        const content = request.messages?.at(-1)?.content ?? "";
        if (Array.isArray(request.tools) && request.tools.length > 0) {
          return new Response(JSON.stringify({
            id: "chatcmpl-tool-test",
            object: "chat.completion",
            created: Date.now(),
            model: request.model,
            choices: [{
              index: 0,
              message: {
                role: "assistant",
                content: "",
                tool_calls: [{
                  id: "call_test",
                  type: "function",
                  function: { name: "example_tool", arguments: "{}" },
                }],
              },
              finish_reason: "tool_calls",
            }],
          }), { status: 200 });
        }
        return new Response(JSON.stringify({
          id: "chatcmpl-test",
          object: "chat.completion",
          created: Date.now(),
          model: request.model,
          choices: [{
            index: 0,
            message: { role: "assistant", content: `handled ${content}` },
            finish_reason: "stop",
          }],
        }), { status: 200 });
      }

      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = `tmp/proxy-concurrency-${Date.now()}-${Math.random().toString(36).slice(2)}.db`;
    const config = makeConfig({
      dbPath,
      providerPriority: ["ollama"],
      providers: {
        ollama: { budgetType: "free", priority: "high" },
      },
      proxyPort: port,
    });

    const proxy = new ProxyServerStreaming(config);
    await proxy.start();

    try {
      const sessions = [
        "agent:main",
        "agent:planner",
        "agent:worker",
        "subagent:research:1",
        "subagent:implement:2",
        "subagent:review:3",
      ];

      const responses = await Promise.all(
        sessions.map((session, index) => postChat(port, session, `request ${index}`)),
      );

      assert.equal(responses.length, sessions.length);
      for (const response of responses) {
        assert.equal(response.model, "CognitiveRouter:latest");
        assert.ok(!String(response.model).includes("/"), "Response model should not expose downstream provider/model");
      }
      assert.ok(maxActiveChatCalls > 1, "Provider calls should overlap under concurrent HTTP requests");

      const models = await getModels(port);
      const ids = models.data.map((model: any) => model.id);
      assert.deepEqual(ids, ["CognitiveRouter:latest", "CogRouter:latest", "Embeddings:latest"]);
      assert.equal(ids.some((id: string) => id.includes("ollama/") || id.includes("zai/")), false);

      const streamBody = await streamChat(port, "agent:stream", "stream request");
      const events = streamBody
        .split("\n\n")
        .map((event) => event.trim())
        .filter(Boolean);
      assert.ok(events.length >= 2, "Stream should include at least one content chunk and [DONE]");
      assert.equal(events.at(-1), "data: [DONE]");

      const firstPayload = JSON.parse(events[0].replace(/^data: /, ""));
      assert.equal(firstPayload.model, "CognitiveRouter:latest");
      assert.equal(firstPayload.choices[0].delta.role, "assistant");
      assert.ok(firstPayload.choices[0].delta.content.length > 0, "First SSE chunk must carry visible content");

    } finally {
      await proxy.stop();
    }

    const db = new DBService(dbPath);
    await db.initializeSchema();
    try {
      const decisions = db.getRecentDecisions(20);
      const recordedSessions = new Set(decisions.map((decision) => decision.session_key));

      for (const session of [
        "agent:main",
        "agent:planner",
        "agent:worker",
        "subagent:research:1",
        "subagent:implement:2",
        "subagent:review:3",
      ]) {
        assert.ok(recordedSessions.has(session), `Missing decision for ${session}`);
      }
    } finally {
      db.close();
    }
  });

  it("does not fall back to Ollama for tool-bearing requests", async () => {
    process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER = "1";
    process.env.ROUTER_OLLAMA_TOOL_MODELS = "not-a-local-model";
    delete process.env.ZAI_API_KEY;

    let ollamaChatCalls = 0;

    globalThis.fetch = (async (input: RequestInfo | URL): Promise<Response> => {
      const url = String(input);

      if (url.includes("/api/tags")) {
        return new Response(JSON.stringify({ models: [{ name: "deepseek-coder-v2:latest" }] }), { status: 200 });
      }

      if (url.includes("/api/embeddings")) {
        return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      }

      if (url.includes("/v1/chat/completions")) {
        ollamaChatCalls++;
        return new Response(JSON.stringify({
          error: { message: "registry.ollama.ai/library/deepseek-coder-v2:latest does not support tools" },
        }), { status: 400 });
      }

      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = `tmp/proxy-tools-${Date.now()}-${Math.random().toString(36).slice(2)}.db`;
    const config = makeConfig({
      dbPath,
      providerPriority: ["ollama"],
      providers: {
        ollama: { budgetType: "free", priority: "high" },
      },
      proxyPort: port,
    });

    const proxy = new ProxyServerStreaming(config);
    await proxy.start();

    try {
      const response = await postRawChat(port, {
        model: "CognitiveRouter:latest",
        messages: [{ role: "user", content: "Use a tool" }],
        tools: [{
          type: "function",
          function: {
            name: "example_tool",
            description: "Example tool",
            parameters: { type: "object", properties: {} },
          },
        }],
        tool_choice: "auto",
        stream: false,
      });

      assert.equal(response.statusCode, 503);
      assert.equal(ollamaChatCalls, 0, "Ollama should not be called for tool-bearing requests");
      assert.equal(response.body.includes("does not support tools"), false);
    } finally {
      await proxy.stop();
      delete process.env.ROUTER_OLLAMA_TOOL_MODELS;
    }
  });

  it("stats exposes model capabilities and decision scores", async () => {
    process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER = "1";
    process.env.ROUTER_OLLAMA_TOOL_MODELS = "gemma4:latest";
    delete process.env.ZAI_API_KEY;

    globalThis.fetch = (async (input: RequestInfo | URL): Promise<Response> => {
      const url = String(input);

      if (url.includes("/api/embeddings")) {
        return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      }

      if (url.includes("/api/tags")) {
        return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });
      }

      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = `tmp/proxy-stats-${Date.now()}-${Math.random().toString(36).slice(2)}.db`;
    const config = makeConfig({
      dbPath,
      providerPriority: ["gemini", "ollama"],
      providers: {
        gemini: { budgetType: "credits", priority: "medium" },
        ollama: { budgetType: "free", priority: "low" },
        requesty: { budgetType: "pay_per_token", priority: "medium", monthlyBudgetUsd: 20 },
      },
      proxyPort: port,
    });

    const proxy = new ProxyServerStreaming(config);
    await proxy.start();

    try {
      const stats = await getStats(port);

      assert.equal(Array.isArray(stats.providers), true);
      assert.equal(Array.isArray(stats.models), true);
      assert.equal(stats.modelCount, stats.models.length);
      assert.deepEqual(stats.routing.providerPriority, ["gemini", "ollama"]);
      assert.equal(stats.routing.fallbackModels.gemini.chat, "gemini-2.5-flash");
      assert.ok(stats.routing.pools.agentGeneration.includes("chat/generation"));

      const requesty = stats.providers.find((p: any) => p.name === "requesty");
      assert.ok(requesty, "Expected stats to include requesty provider health");
      assert.equal(requesty.priorityIndex, null);
      assert.equal(requesty.inProviderPriority, false);

      const gemma = stats.models.find((m: any) => m.id === "ollama/gemma4:latest");
      assert.ok(gemma, "Expected stats to include ollama/gemma4:latest");
      assert.deepEqual(gemma.taskTypes, ["chat", "generation", "local-emergency"]);
      assert.equal(gemma.toolCapable, true);
      assert.equal(gemma.capabilities.conversation, 0.70);
      assert.equal(gemma.decisionData.generationRoutingExclusion, null);
      assert.equal(gemma.decisionData.eligibleForRouting, true);
      assert.equal(typeof gemma.decisionData.intentScores.coding.overall, "number");
      assert.equal(typeof gemma.decisionData.intentScores.coding.capability, "number");
      assert.equal(typeof gemma.decisionData.effectiveScores.cost, "number");
      assert.equal(gemma.decisionData.weights.capability, 0.5);

    } finally {
      await proxy.stop();
      delete process.env.ROUTER_OLLAMA_TOOL_MODELS;
    }
  });

  it("prefers Gemini tool fallback before local Ollama tool models", async () => {
    process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER = "1";
    process.env.ROUTER_GEMINI_TOOL_MODEL = "gemini-2.5-flash";
    delete process.env.ZAI_API_KEY;

    let geminiChatCalls = 0;
    let ollamaChatCalls = 0;

    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);

      if (url.includes("/api/embeddings")) {
        return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      }

      if (url.includes("/api/tags")) {
        return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });
      }

      if (url.includes("generativelanguage.googleapis.com")) {
        geminiChatCalls++;
        const request = JSON.parse(String(init?.body ?? "{}"));
        assert.equal(request.tools[0].functionDeclarations[0].name, "example_tool");
        return new Response(JSON.stringify({
          candidates: [{
            content: {
              parts: [{
                functionCall: {
                  name: "example_tool",
                  args: { ok: true },
                },
              }],
            },
            finishReason: "STOP",
          }],
        }), { status: 200 });
      }

      if (url.includes("/v1/chat/completions")) {
        ollamaChatCalls++;
        return new Response(JSON.stringify({
          id: "ollama-should-not-run",
          object: "chat.completion",
          created: Date.now(),
          model: "gemma4:latest",
          choices: [{ index: 0, message: { role: "assistant", content: "ollama" }, finish_reason: "stop" }],
        }), { status: 200 });
      }

      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = `tmp/proxy-gemini-tools-${Date.now()}-${Math.random().toString(36).slice(2)}.db`;
    const config = makeConfig({
      dbPath,
      providerPriority: ["gemini", "ollama"],
      providers: {
        gemini: { budgetType: "credits", priority: "medium" },
        ollama: { budgetType: "free", priority: "low" },
      },
      proxyPort: port,
    });

    const proxy = new ProxyServerStreaming(config);
    await proxy.start();

    try {
      const response = await postRawChat(port, {
        model: "CognitiveRouter:latest",
        messages: [{ role: "user", content: "Use a tool" }],
        tools: [{
          type: "function",
          function: {
            name: "example_tool",
            description: "Example tool",
            parameters: { type: "object", properties: { ok: { type: "boolean" } } },
          },
        }],
        tool_choice: "auto",
        stream: false,
      }, "agent:gemini-tool");

      assert.equal(response.statusCode, 200);
      assert.equal(geminiChatCalls, 1);
      assert.equal(ollamaChatCalls, 0);
      const body = JSON.parse(response.body);
      assert.equal(body.choices[0].message.tool_calls[0].function.name, "example_tool");
      assert.equal(body.choices[0].finish_reason, "tool_calls");
    } finally {
      await proxy.stop();
      delete process.env.ROUTER_GEMINI_TOOL_MODEL;
    }
  });

  it("allows explicitly configured Ollama tool models", async () => {
    process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER = "1";
    process.env.ROUTER_OLLAMA_TOOL_MODELS = "gemma4:latest";
    delete process.env.ZAI_API_KEY;

    let ollamaChatCalls = 0;

    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);

      if (url.includes("/api/embeddings")) {
        return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      }

      if (url.includes("/api/tags")) {
        return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });
      }

      if (url.includes("/v1/chat/completions")) {
        ollamaChatCalls++;
        const request = JSON.parse(String(init?.body ?? "{}"));
        assert.equal(request.model, "gemma4:latest");
        assert.ok(Array.isArray(request.tools), "Tool request should reach allowlisted Ollama model");
        return new Response(JSON.stringify({
          id: "chatcmpl-ollama-tool",
          object: "chat.completion",
          created: Date.now(),
          model: request.model,
          choices: [{
            index: 0,
            message: {
              role: "assistant",
              content: "",
              tool_calls: [{
                id: "call_ollama",
                type: "function",
                function: { name: "example_tool", arguments: "{}" },
              }],
            },
            finish_reason: "tool_calls",
          }],
        }), { status: 200 });
      }

      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = `tmp/proxy-ollama-tool-${Date.now()}-${Math.random().toString(36).slice(2)}.db`;
    const config = makeConfig({
      dbPath,
      providerPriority: ["ollama"],
      providers: {
        ollama: { budgetType: "free", priority: "low" },
      },
      proxyPort: port,
    });

    const proxy = new ProxyServerStreaming(config);
    await proxy.start();

    try {
      const response = await postRawChat(port, {
        model: "CognitiveRouter:latest",
        messages: [{ role: "user", content: "Use a local tool model" }],
        tools: [{
          type: "function",
          function: {
            name: "example_tool",
            description: "Example tool",
            parameters: { type: "object", properties: {} },
          },
        }],
        tool_choice: "auto",
        stream: false,
      }, "agent:ollama-tool");

      assert.equal(response.statusCode, 200);
      assert.equal(ollamaChatCalls, 1);
      assert.ok(response.body.includes("call_ollama"));
    } finally {
      await proxy.stop();
      delete process.env.ROUTER_OLLAMA_TOOL_MODELS;
    }
  });

  it("forwards provider tool_calls in streaming responses", async () => {
    process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER = "1";
    delete process.env.ZAI_API_KEY;

    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);

      if (url.includes("/api/embeddings")) {
        return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      }

      if (url.includes("/api/tags")) {
        return new Response(JSON.stringify({ models: [] }), { status: 200 });
      }

      if (url.includes("openrouter.ai/api/v1/chat/completions")) {
        const request = JSON.parse(String(init?.body ?? "{}"));
        assert.ok(Array.isArray(request.tools), "Tool request should reach OpenRouter candidate");
        return new Response(JSON.stringify({
          id: "chatcmpl-tool-test",
          object: "chat.completion",
          created: Date.now(),
          model: request.model,
          choices: [{
            index: 0,
            message: {
              role: "assistant",
              content: "",
              tool_calls: [{
                id: "call_test",
                type: "function",
                function: { name: "example_tool", arguments: "{}" },
              }],
            },
            finish_reason: "tool_calls",
          }],
        }), { status: 200 });
      }

      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = `tmp/proxy-tool-stream-${Date.now()}-${Math.random().toString(36).slice(2)}.db`;
    const config = makeConfig({
      dbPath,
      providerPriority: ["openrouter", "ollama"],
      providers: {
        openrouter: { budgetType: "free", priority: "high" },
        ollama: { budgetType: "free", priority: "low" },
      },
      overrides: [
        {
          intent: "conversation",
          provider: "openrouter",
          model: "openrouter/free",
          reason: "tool-call streaming test",
        },
      ],
      proxyPort: port,
    });

    const proxy = new ProxyServerStreaming(config);
    await proxy.start();

    try {
      const response = await postRawChat(port, {
        model: "CognitiveRouter:latest",
        messages: [{ role: "user", content: "Use a tool" }],
        tools: [{
          type: "function",
          function: {
            name: "example_tool",
            description: "Example tool",
            parameters: { type: "object", properties: {} },
          },
        }],
        tool_choice: "auto",
        stream: true,
      }, "agent:tool-stream");

      assert.equal(response.statusCode, 200);
      const events = response.body
        .split("\n\n")
        .map((event) => event.trim())
        .filter(Boolean);
      const firstPayload = JSON.parse(events[0].replace(/^data: /, ""));
      assert.equal(firstPayload.choices[0].delta.role, "assistant");
      assert.equal(firstPayload.choices[0].delta.tool_calls[0].id, "call_test");
      const finalPayload = JSON.parse(events[1].replace(/^data: /, ""));
      assert.equal(finalPayload.choices[0].finish_reason, "tool_calls");
      assert.equal(events.at(-1), "data: [DONE]");
    } finally {
      await proxy.stop();
    }
  });

  it("ignores expensive OpenRouter fallback models from environment", async () => {
    process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER = "1";
    process.env.ROUTER_OPENROUTER_TOOL_MODEL = "openai/gpt-4o-mini";
    process.env.ROUTER_OPENROUTER_FALLBACK_MODEL = "anthropic/claude-3.5-sonnet";
    delete process.env.ZAI_API_KEY;

    assert.equal(fallbackModelForProvider("openrouter", true), "qwen/qwen3-coder:free");
    assert.equal(fallbackModelForProvider("openrouter", false), "openrouter/owl-alpha");
    process.env.ROUTER_OPENROUTER_FALLBACK_MODEL = "google/gemini-3-pro";
    assert.equal(fallbackModelForProvider("openrouter", false), "openrouter/owl-alpha");
    process.env.ROUTER_OPENROUTER_FALLBACK_MODEL = "poolside/laguna-m.1:free";
    process.env.ROUTER_OPENROUTER_TOOL_MODEL = "poolside/laguna-m.1:free";
    assert.equal(fallbackModelForProvider("openrouter", false), "poolside/laguna-m.1:free");

    let requestedModel = "";

    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);

      if (url.includes("/api/embeddings")) {
        return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      }

      if (url.includes("/api/tags")) {
        return new Response(JSON.stringify({ models: [] }), { status: 200 });
      }

      if (url.includes("openrouter.ai/api/v1/chat/completions")) {
        const request = JSON.parse(String(init?.body ?? "{}"));
        requestedModel = request.model;
        return new Response(JSON.stringify({
          id: "chatcmpl-cheap-tool",
          object: "chat.completion",
          created: Date.now(),
          model: request.model,
          choices: [{
            index: 0,
            message: {
              role: "assistant",
              content: "",
              tool_calls: [{
                id: "call_test",
                type: "function",
                function: { name: "example_tool", arguments: "{}" },
              }],
            },
            finish_reason: "tool_calls",
          }],
        }), { status: 200 });
      }

      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = `tmp/proxy-cheap-env-${Date.now()}-${Math.random().toString(36).slice(2)}.db`;
    const config = makeConfig({
      dbPath,
      providerPriority: ["openrouter"],
      providers: {
        openrouter: { budgetType: "free", priority: "high" },
      },
      overrides: [],
      proxyPort: port,
    });

    const proxy = new ProxyServerStreaming(config);
    await proxy.start();

    try {
      const response = await postRawChat(port, {
        model: "CognitiveRouter:latest",
        messages: [{ role: "user", content: "Use a tool" }],
        tools: [{
          type: "function",
          function: {
            name: "example_tool",
            description: "Example tool",
            parameters: { type: "object", properties: {} },
          },
        }],
        tool_choice: "auto",
        stream: true,
      }, "agent:cheap-env");

      assert.equal(response.statusCode, 200);
      assert.equal(requestedModel, "poolside/laguna-m.1:free");
      assert.equal(requestedModel.includes("openai/"), false);
      assert.equal(requestedModel.includes("anthropic/"), false);
    } finally {
      await proxy.stop();
      delete process.env.ROUTER_OPENROUTER_TOOL_MODEL;
      delete process.env.ROUTER_OPENROUTER_FALLBACK_MODEL;
    }
  });

  it("streams sanitized provider errors instead of assistant text when all providers are exhausted", async () => {
    process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER = "1";
    delete process.env.ZAI_API_KEY;

    globalThis.fetch = (async (input: RequestInfo | URL): Promise<Response> => {
      const url = String(input);

      if (url.includes("/api/embeddings")) {
        return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      }

      if (url.includes("/api/tags")) {
        return new Response(JSON.stringify({ models: [] }), { status: 200 });
      }

      if (url.includes("openrouter.ai/api/v1/chat/completions")) {
        return new Response(JSON.stringify({
          error: {
            message: "Prompt tokens limit exceeded: 69529 > 66327. To increase, visit https://openrouter.ai/workspaces/default/keys/7645a6e342a939c57a360638c770287a9880f3b80d86b7c8c89ea658cab59644 and adjust the key's monthly limit",
            code: 402,
          },
          user_id: "user_32kYpTck4JMfQFTf64cUi6QpWvw",
        }), { status: 402 });
      }

      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = `tmp/proxy-openrouter-402-${Date.now()}-${Math.random().toString(36).slice(2)}.db`;
    const config = makeConfig({
      dbPath,
      providerPriority: ["openrouter"],
      providers: {
        openrouter: { budgetType: "free", priority: "high" },
      },
      overrides: [
        {
          intent: "conversation",
          provider: "openrouter",
          model: "cohere/north-mini-code:free",
          reason: "quota error streaming test",
        },
      ],
      proxyPort: port,
    });

    const proxy = new ProxyServerStreaming(config);
    await proxy.start();

    try {
      const response = await postRawChat(port, {
        model: "CognitiveRouter:latest",
        messages: [{ role: "user", content: "Summarize a large context" }],
        stream: true,
      }, "agent:quota-stream");

      assert.equal(response.statusCode, 200);
      assert.ok(response.body.includes('"error"'));
      assert.ok(response.body.includes('"all_providers_exhausted"'));
      assert.equal(response.body.includes('"choices"'), false);
      assert.equal(response.body.includes("openrouter.ai/workspaces"), false);
      assert.equal(response.body.includes("user_32kYpTck4JMfQFTf64cUi6QpWvw"), false);
      assert.equal(response.body.trim().endsWith("data: [DONE]"), true);
    } finally {
      await proxy.stop();
    }
  });

  it("falls back when a provider returns an empty assistant payload", async () => {
    process.env.ROUTER_MAX_ATTEMPTS_PER_PROVIDER = "1";
    delete process.env.ZAI_API_KEY;

    let chatCalls = 0;

    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);

      if (url.includes("/api/embeddings")) {
        return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
      }

      if (url.includes("/api/tags")) {
        return new Response(JSON.stringify({ models: [{ name: "gemma4:latest" }] }), { status: 200 });
      }

      if (url.includes("chat/completions")) {
        chatCalls++;
        const request = JSON.parse(String(init?.body ?? "{}"));
        if (chatCalls === 1) {
          return new Response(JSON.stringify({
            id: "gen-empty",
            object: "chat.completion",
            created: Date.now(),
            model: request.model,
            choices: [{
              index: 0,
              message: { role: "assistant", content: "" },
              finish_reason: "length",
            }],
          }), { status: 200 });
        }
        return new Response(JSON.stringify({
          id: "chatcmpl-fallback",
          object: "chat.completion",
          created: Date.now(),
          model: request.model,
          choices: [{
            index: 0,
            message: { role: "assistant", content: "fallback ok" },
            finish_reason: "stop",
          }],
        }), { status: 200 });
      }

      return new Response(JSON.stringify({ data: [] }), { status: 200 });
    }) as typeof fetch;

    const port = await getFreePort();
    const dbPath = `tmp/proxy-empty-fallback-${Date.now()}-${Math.random().toString(36).slice(2)}.db`;
    const config = makeConfig({
      dbPath,
      providerPriority: ["openrouter", "ollama"],
      providers: {
        openrouter: { budgetType: "free", priority: "high" },
        ollama: { budgetType: "free", priority: "low" },
      },
      overrides: [
        {
          intent: "conversation",
          provider: "openrouter",
          model: "openrouter/free",
          reason: "empty response fallback test",
        },
      ],
      proxyPort: port,
    });

    const proxy = new ProxyServerStreaming(config);
    await proxy.start();

    try {
      const response = await postRawChat(port, {
        model: "CognitiveRouter:latest",
        messages: [{ role: "user", content: "Say ok" }],
        stream: true,
      }, "agent:empty-stream");

      assert.equal(response.statusCode, 200);
      assert.equal(chatCalls, 2);
      assert.ok(response.body.includes("fallback ok"));
      assert.equal(response.body.includes("gen-empty"), false);
      assert.equal(response.body.trim().endsWith("data: [DONE]"), true);
    } finally {
      await proxy.stop();
    }
  });
});

describe("Edge Cases — Boundary Conditions", () => {
  let registry: ModelRegistry;
  let costTracker: CostTracker;
  let db: DBService;
  let config: CognitiveRouterConfig;
  let router: RoutingEngine;

  beforeEach(async () => {
    config = makeConfig();
    db = makeMockDB();
    costTracker = new CostTracker(db, config);
    await costTracker.refreshProviderStatus();
    registry = new ModelRegistry(db, config);
    await registry.loadCachedState();
    router = new RoutingEngine(registry, costTracker, db, config);
  });

  it("should handle provider not in config", () => {
    const score = costTracker.getReliabilityScore("nonexistent-provider");
    assert.equal(score, 0, "Unknown provider should have 0 reliability");
  });

  it("should handle empty prompt classification", async () => {
    const decision = await router.decide(
      makeClassification("", 0.5),
      "test",
      {},
    );
    assert.ok(decision, "Should handle empty intent gracefully");
  });

  it("should handle VRAM limit filtering local models", async () => {
    // With default 11GB limit, models requiring more should be filtered
    // The registry should already filter these, but test the router layer
    const decision = await router.decide(makeClassification(), "test", {});
    assert.ok(decision);
    // If ollama is selected, it should be within VRAM limit
    if (decision!.provider === "ollama") {
      // Verify the model exists in registry
      const models = registry.getAvailableModels(["ollama"]);
      for (const m of models) {
        if (m.vramRequiredGb) {
          assert.ok(m.vramRequiredGb <= config.localVramLimitGb,
            `${m.model} VRAM (${m.vramRequiredGb}GB) should be <= limit (${config.localVramLimitGb}GB)`);
        }
      }
    }
  });

  it("should handle all providers having zero reliability (edge of exhaustion)", async () => {
    // Throttle all providers but don't circuit-open them
    for (const provider of config.providerPriority) {
      // 1 generic error each → throttled but not circuit_open
      await costTracker.recordCall(provider, { durationMs: 100, outcome: "error" });
    }

    const decision = await router.decide(makeClassification(), "test", {});
    // Should still return a decision (throttled != dead)
    assert.ok(decision, "Should route when all providers are throttled (not circuit_open)");
  });

  it("should handle unknown provider in priority list", async () => {
    const weirdConfig = makeConfig({ providerPriority: ["unknown_provider", "zai"] });
    const weirdRouter = new RoutingEngine(registry, costTracker, db, weirdConfig);

    const decision = await weirdRouter.decide(makeClassification(), "test", {});
    // Should skip unknown provider and route to zai
    assert.ok(decision);
    assert.equal(decision!.provider, "zai");
  });

  it("seeds deterministic OpenRouter free agent candidates", async () => {
    const openRouterIds = registry.getAvailableModels(["openrouter"]).map((m) => m.model);

    assert.ok(openRouterIds.includes("qwen/qwen3-coder:free"));
    assert.ok(openRouterIds.includes("poolside/laguna-m.1:free"));
    assert.ok(openRouterIds.includes("openrouter/owl-alpha"));
    assert.ok(openRouterIds.includes("nvidia/nemotron-3-ultra-550b-a55b:free"));
    assert.ok(openRouterIds.includes("nvidia/nemotron-3-super-120b-a12b:free"));
    assert.ok(openRouterIds.includes("qwen/qwen3.6-plus:free"));
    assert.ok(openRouterIds.includes("cohere/north-mini-code:free"));
    assert.equal(openRouterIds.includes("openrouter/free"), false);
  });
});

describe("Model Policy", () => {
  it("limits Z.AI generation routing to the three approved agent models", () => {
    assert.deepEqual(modelTaskTypes("zai", "glm-5.2"), ["chat", "generation"]);
    assert.deepEqual(modelTaskTypes("zai", "glm-5.1"), ["chat", "generation"]);
    assert.deepEqual(modelTaskTypes("zai", "glm-4.7-flash"), ["chat", "generation"]);
    assert.equal(generationRoutingExclusionReason("zai", "glm-4.7"), "not_in_zai_agent_generation_pool");
    assert.equal(isGenerationModel("zai", "glm-4.7"), false);
  });

  it("keeps embedding models out of generation while preserving their pool", () => {
    assert.deepEqual(modelTaskTypes("ollama", "nomic-embed-text:latest"), ["embedding"]);
    assert.equal(isGenerationModel("ollama", "nomic-embed-text:latest"), false);
    assert.equal(generationRoutingExclusionReason("ollama", "nomic-embed-text:latest"), "embedding_only");
  });

  it("limits Ollama generation routing to the emergency generation pool", () => {
    assert.deepEqual(modelTaskTypes("ollama", "gemma4:latest"), ["chat", "generation", "local-emergency"]);
    assert.equal(isGenerationModel("ollama", "gemma4:latest"), true);
    assert.equal(generationRoutingExclusionReason("ollama", "gemma4:latest"), null);

    assert.deepEqual(modelTaskTypes("ollama", "qwen2.5-coder:7b"), ["chat", "generation", "local-emergency"]);
    assert.equal(isGenerationModel("ollama", "qwen2.5-coder:7b"), true);
    assert.equal(generationRoutingExclusionReason("ollama", "qwen2.5-coder:3b"), "not_in_ollama_emergency_generation_pool");
    assert.equal(isGenerationModel("ollama", "qwen2.5-coder:3b"), false);
  });

  it("keeps local vision models in the vision pool instead of agent generation", () => {
    assert.deepEqual(modelTaskTypes("ollama", "llama3.2-vision:11b"), ["vision"]);
    assert.equal(isGenerationModel("ollama", "llama3.2-vision:11b"), false);
    assert.equal(generationRoutingExclusionReason("ollama", "llama3.2-vision:11b"), "vision_only_route");
  });
});

describe("DB Service — Retry Tracking Per Session", () => {
  it("should record and retrieve decisions by requestId", () => {
    const db = makeMockDB();

    db.recordDecision({
      timestamp: new Date().toISOString(),
      sessionKey: "session-A",
      messageHash: "hash1",
      intent: "coding",
      confidence: 0.9,
      provider: "zai",
      model: "glm-5.1",
      scores: {},
      overallScore: 0.85,
      outcome: "PENDING",
      requestId: "req_001",
    });

    db.recordDecision({
      timestamp: new Date().toISOString(),
      sessionKey: "session-B",
      messageHash: "hash2",
      intent: "conversation",
      confidence: 0.8,
      provider: "zai",
      model: "glm-5.1",
      scores: {},
      overallScore: 0.82,
      outcome: "PENDING",
      requestId: "req_002",
    });

    const lookup = db.getDecisionByRequestId("req_001");
    assert.ok(lookup);
    assert.equal(lookup.sessionKey, "session-A");
    assert.equal(lookup.intent, "coding");

    const notFound = db.getDecisionByRequestId("nonexistent");
    assert.equal(notFound, null);
  });

  it("should record call outcomes independently of decisions", () => {
    const db = makeMockDB();

    db.recordCallOutcome({
      provider: "zai",
      model: "glm-5.1",
      durationMs: 1500,
      outcome: "success",
      timestamp: new Date().toISOString(),
    });

    db.recordCallOutcome({
      provider: "openrouter",
      model: "free-model",
      durationMs: 800,
      outcome: "rate_limit",
      timestamp: new Date().toISOString(),
    });

    const outcomes = (db as any)._outcomes;
    assert.equal(outcomes.length, 2);
    assert.equal(outcomes[0].provider, "zai");
    assert.equal(outcomes[1].provider, "openrouter");
    assert.equal(outcomes[1].outcome, "rate_limit");
  });
});

describe("IntentClassifier — Startup Fallback", () => {
  it("supports keyword classification before embedding prototypes are initialized", () => {
    const classifier = new IntentClassifier({ tiebreakerThreshold: 0.7 });

    assert.equal(classifier.isInitialized(), false);

    const result = classifier.classifyByKeyword("fix this compile error in the API test");

    assert.equal(result.intent, "coding");
    assert.equal(Number.isFinite(result.confidence), true);
    assert.ok(result.confidence > 0);
  });
});
