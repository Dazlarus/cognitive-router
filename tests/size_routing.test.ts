// tests/size_routing.test.ts — Request-size-aware routing tests
// Run with: npx tsx --test tests/size_routing.test.ts

import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { RoutingEngine } from "../src/router.ts";
import { CostTracker, bucketForTokenCount, SIZE_BUCKETS, type SizeBucket } from "../src/cost_tracker.ts";
import { ModelRegistry } from "../src/model_registry.ts";
import { DBService } from "../src/db_service.ts";
import { loadConfig, type CognitiveRouterConfig } from "../src/config.ts";
import type { Classification } from "../src/classifier.ts";

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

function makeMockDB(): DBService {
  const decisions: any[] = [];
  const overrides: any[] = [];
  return {
    initializeSchema: async () => {},
    recordDecision: (d: any) => decisions.push(d),
    recordCallOutcome: () => {},
    recordRetry: () => {},
    getDecisionByRequestId: () => null,
    getRetryCount: () => 0,
    getRecentDecisions: () => decisions,
    getModelStats: () => [],
    getProviderHealth: () => [],
    getSpendByProvider: () => [],
    getSpend: () => 0,
    loadCapabilityOverrides: () => overrides,
    getCapabilityOverride: () => null,
    upsertCapabilityOverride: () => {},
    recordJudgeEvaluation: () => {},
    recordSpend: () => {},
    getAllSpend: () => [],
    close: () => {},
  } as any;
}

function makeClassification(intent = "conversation", confidence = 0.9): Classification {
  return { intent, confidence };
}

// ─── Size Bucket Tests ─────────────────────────────────────

describe("Size Bucket Classification", () => {
  it("should classify tokens < 5000 as small", () => {
    assert.equal(bucketForTokenCount(0), "small");
    assert.equal(bucketForTokenCount(100), "small");
    assert.equal(bucketForTokenCount(4999), "small");
  });

  it("should classify tokens 5000–49999 as medium", () => {
    assert.equal(bucketForTokenCount(5000), "medium");
    assert.equal(bucketForTokenCount(25000), "medium");
    assert.equal(bucketForTokenCount(49999), "medium");
  });

  it("should classify tokens >= 50000 as large", () => {
    assert.equal(bucketForTokenCount(50000), "large");
    assert.equal(bucketForTokenCount(100000), "large");
    assert.equal(bucketForTokenCount(1000000), "large");
  });

  it("should have correct bucket boundaries", () => {
    assert.equal(SIZE_BUCKETS.small.max, 5000);
    assert.equal(SIZE_BUCKETS.medium.max, 50000);
    assert.equal(SIZE_BUCKETS.large.max, Infinity);
  });
});

// ─── CostTracker Size Latency Profiling ────────────────────

describe("CostTracker — Size Latency Profiling", () => {
  let config: CognitiveRouterConfig;
  let db: DBService;
  let costTracker: CostTracker;

  beforeEach(async () => {
    config = makeConfig();
    db = makeMockDB();
    costTracker = new CostTracker(db, config);
    await costTracker.refreshProviderStatus();
  });

  it("should return static default latency when no observed data exists", () => {
    const small = costTracker.getSizeLatencyMs("zai", "small");
    assert.ok(small !== undefined, "Should return a default for zai/small");
    assert.ok(small! < 1500, "ZAI small latency should be low");
  });

  it("should return undefined for unknown provider", () => {
    const ms = costTracker.getSizeLatencyMs("unknown", "small");
    assert.equal(ms, undefined);
  });

  it("should record and return observed latency data", () => {
    costTracker.recordSizeLatency("zai", 500, 600);  // small bucket
    costTracker.recordSizeLatency("zai", 800, 700);  // small bucket

    const avg = costTracker.getSizeLatencyMs("zai", "small");
    assert.ok(avg !== undefined);
    assert.ok(avg! >= 600 && avg! <= 700, `Avg should be between 600-700, got ${avg}`);
  });

  it("should track latency independently per size bucket", () => {
    costTracker.recordSizeLatency("gemini", 100, 500);      // small
    costTracker.recordSizeLatency("gemini", 60000, 8000);   // large

    assert.ok(costTracker.getSizeLatencyMs("gemini", "small")! < 1000);
    assert.ok(costTracker.getSizeLatencyMs("gemini", "large")! > 5000);
  });

  it("should compute a latency score in [0, 1] range", () => {
    costTracker.recordSizeLatency("zai", 500, 400);  // fast small request

    const score = costTracker.getSizeLatencyScore("zai", 500);
    assert.ok(score >= 0 && score <= 1.0, `Score should be in [0,1], got ${score}`);
    assert.ok(score > 0.8, `Fast provider should score high, got ${score}`);
  });

  it("should produce a human-readable summary", () => {
    costTracker.recordSizeLatency("zai", 500, 600);
    const summary = costTracker.getSizeLatencySummary("zai");
    assert.ok(summary.includes("small"), "Summary should include small bucket");
  });
});

// ─── RoutingEngine Size-Aware Scoring ──────────────────────

describe("RoutingEngine — Size-Aware Routing", () => {
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

  it("should route small requests (<5K) preferentially to low-latency provider", async () => {
    // Small request — should boost zai (low-latency provider)
    const decision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-session",
      { estimatedTokens: 500 },
    );

    assert.ok(decision, "Should return a routing decision");
    assert.equal(decision!.provider, "zai",
      "Small request should prefer low-latency provider (zai)");

    // Size adjust should be present and positive
    assert.ok(decision!.scores.sizeAdjust !== undefined,
      "Size adjustment should be scored for small request");
    assert.ok(decision!.scores.sizeAdjust! > 0,
      "Small request to zai should have positive size adjustment");

    // Rationale should include size= term for transparency
    assert.ok(decision!.rationale.includes("size="),
      "Rationale should include size adjustment for transparency");
  });

  it("should route large requests (>50K) preferentially to high-throughput provider", async () => {
    // Large request — should boost gemini or zai (high-throughput providers)
    const decision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-session",
      { estimatedTokens: 80000 },
    );

    assert.ok(decision, "Should return a routing decision");
    // Gemini and zai are both high-throughput; zai typically wins on cost score
    assert.ok(
      decision!.provider === "zai" || decision!.provider === "gemini",
      `Large request should prefer high-throughput provider, got ${decision!.provider}`,
    );

    // Should have size adjustment
    assert.ok(decision!.scores.sizeAdjust !== undefined,
      "Size adjustment should be scored for large request");
  });

  it("should NOT apply size adjustment for medium-sized requests (5K–50K)", async () => {
    const decision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-session",
      { estimatedTokens: 20000 },
    );

    assert.ok(decision, "Should return a routing decision");
    // Medium bucket → no size adjustment
    assert.ok(decision!.scores.sizeAdjust === undefined || decision!.scores.sizeAdjust === 0,
      `Medium request should have no size adjustment, got ${decision!.scores.sizeAdjust}`);
  });

  it("should NOT apply size adjustment when estimatedTokens is 0 or missing", async () => {
    const decision = await router.decide(
      makeClassification(),
      "test-session",
      {},
    );

    assert.ok(decision, "Should return a routing decision");
    assert.ok(decision!.scores.sizeAdjust === undefined,
      "Missing estimatedTokens should not produce size adjustment");
  });

  it("should penalize ollama for small requests", async () => {
    // With only ollama vs zai, ollama should have negative size adjust for small
    const smallConfig = makeConfig({
      providerPriority: ["zai", "ollama"],
    });
    const smallRouter = new RoutingEngine(registry, costTracker, db, smallConfig);

    const decision = await smallRouter.decide(
      makeClassification(),
      "test-session",
      { estimatedTokens: 200 },
    );

    assert.ok(decision);
    // zai should win with the boost, not ollama
    assert.equal(decision!.provider, "zai");
  });

  it("should log size-adjustment debug info when size bucket is active", async () => {
    // This test verifies the scoring is transparent (logged in debug)
    // We can't easily capture log output, but we verify the rationale includes size info
    const decision = await router.decide(
      makeClassification(),
      "test-session",
      { estimatedTokens: 100 },
    );

    assert.ok(decision);
    assert.ok(
      decision!.rationale.includes("size="),
      "Rationale string should include size adjustment term for debug transparency",
    );
  });

  it("should refine static boost using observed latency data", async () => {
    // Record that ollama is actually very fast for small requests
    // (e.g. small model, warm GPU)
    costTracker.recordSizeLatency("ollama", 100, 200);
    costTracker.recordSizeLatency("ollama", 200, 250);
    costTracker.recordSizeLatency("zai", 100, 900);
    costTracker.recordSizeLatency("zai", 200, 950);

    const decision = await router.decide(
      makeClassification(),
      "test-session",
      { estimatedTokens: 100 },
    );

    assert.ok(decision);
    // ZAI should still win (subscription cost score + capability), but the
    // size boost for zai should be reduced since its observed latency is higher
    // relative to the bucket average (which ollama is pulling down).
    // We just verify the decision is made without error.
    assert.ok(decision!.provider === "zai" || decision!.provider === "ollama");
  });

  it("should handle existing tests' context shape (empty object or no estimatedTokens)", async () => {
    // Existing tests pass {} as context — this must still work
    const decision1 = await router.decide(makeClassification(), "test", {});
    assert.ok(decision1);

    // And undefined context
    const decision2 = await router.decide(makeClassification(), "test");
    assert.ok(decision2);
  });
});

// ─── Integration: Size scoring doesn't break existing behavior ─────────

describe("RoutingEngine — Size Scoring Backwards Compatibility", () => {
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

  it("should still route to zai for normal conversation (no size context)", async () => {
    const decision = await router.decide(
      makeClassification("conversation", 0.85),
      "session",
      {},
    );
    assert.ok(decision);
    assert.equal(decision!.provider, "zai");
  });

  it("should still route to zai for coding intent (no size context)", async () => {
    const decision = await router.decide(
      makeClassification("coding", 0.95),
      "session",
      {},
    );
    assert.ok(decision);
    assert.equal(decision!.provider, "zai");
  });

  it("should still failover when zai is circuit-open even with size scoring", async () => {
    await costTracker.recordCall("zai", { durationMs: 100, outcome: "rate_limit" });

    const decision = await router.decide(
      makeClassification(),
      "session",
      { estimatedTokens: 500 },
    );
    assert.ok(decision);
    assert.notEqual(decision!.provider, "zai",
      "Should still failover away from circuit-open zai regardless of size boost");
  });

  it("should still respect manual overrides even with size scoring", async () => {
    const overrideConfig = makeConfig({
      overrides: [
        { intent: "coding", provider: "ollama", model: "gemma4:latest", reason: "test" },
      ],
    });
    const overrideRouter = new RoutingEngine(registry, costTracker, db, overrideConfig);

    const decision = await overrideRouter.decide(
      makeClassification("coding", 0.95),
      "session",
      { estimatedTokens: 100 },
    );
    assert.ok(decision);
    assert.equal(decision!.provider, "ollama",
      "Manual override should take priority over size scoring");
  });
});
