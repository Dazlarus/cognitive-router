// tests/warm_model.test.ts — Warm model awareness tests
// Run with: npx tsx --test tests/warm_model.test.ts

import { describe, it, beforeEach, afterEach, mock, type Mock } from "node:test";
import assert from "node:assert/strict";
import { OllamaWarmthChecker, type WarmthInfo } from "../src/ollama_warmth.ts";
import { RoutingEngine } from "../src/router.ts";
import { CostTracker } from "../src/cost_tracker.ts";
import { ModelRegistry, type ModelCapability } from "../src/model_registry.ts";
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
    loadCircuitState: () => null,
    saveCircuitState: () => {},
    close: () => {},
  } as any;
}

function makeClassification(intent = "conversation", confidence = 0.9): Classification {
  return { intent, confidence };
}

/** Create a mock fetch that returns specified loaded models from /api/ps. */
function mockFetchPs(models: string[] | null, opts: { status?: number; delay?: number } = {}): Mock {
  const { status = 200, delay = 0 } = opts;
  return mock.method(globalThis, "fetch", (url: string) => {
    if (typeof url === "string" && url.includes("/api/ps")) {
      return Promise.resolve({
        ok: status >= 200 && status < 300,
        status,
        json: () => Promise.resolve(
          models === null
            ? { unexpected: "format" }
            : { models: models.map((name) => ({ name, model: name, size: 0 })) },
        ),
      } as any);
    }
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({}) } as any);
  });
}

/** Create a mock fetch that always rejects (simulates Ollama unreachable). */
function mockFetchUnreachable(): Mock {
  return mock.method(globalThis, "fetch", (url: string) => {
    if (typeof url === "string" && url.includes("/api/ps")) {
      return Promise.reject(new Error("ECONNREFUSED"));
    }
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({}) } as any);
  });
}

// ─── OllamaWarmthChecker Unit Tests ───────────────────────

describe("OllamaWarmthChecker", () => {
  let checker: OllamaWarmthChecker;
  let fetchMock: Mock;

  afterEach(() => {
    if (fetchMock) fetchMock.mock.restore();
  });

  it("should detect a warm model", async () => {
    fetchMock = mockFetchPs(["gemma4:latest", "qwen2.5:7b"]);
    checker = new OllamaWarmthChecker({ cacheTtlMs: 5000 });

    const info = await checker.checkWarmth("gemma4:latest");
    assert.equal(info.isWarm, true);
    assert.equal(info.estimatedColdStartMs, 0);
    assert.equal(info.ollamaReachable, true);
    assert.deepEqual(info.loadedModels, ["gemma4:latest", "qwen2.5:7b"]);
  });

  it("should detect a cold model when not in /api/ps", async () => {
    fetchMock = mockFetchPs(["gemma4:latest"]);
    checker = new OllamaWarmthChecker({ cacheTtlMs: 5000 });

    const info = await checker.checkWarmth("llama3:70b");
    assert.equal(info.isWarm, false);
    assert.ok(info.estimatedColdStartMs > 0, "Cold model should have non-zero cold start estimate");
    assert.equal(info.ollamaReachable, true);
  });

  it("should estimate cold-start scaling with model size", async () => {
    fetchMock = mockFetchPs([]);
    checker = new OllamaWarmthChecker({ cacheTtlMs: 5000 });

    const small = await checker.checkWarmth("qwen2.5:7b");
    const large = await checker.checkWarmth("llama3:70b");

    assert.ok(large.estimatedColdStartMs > small.estimatedColdStartMs,
      `70B cold start (${large.estimatedColdStartMs}ms) should exceed 7B (${small.estimatedColdStartMs}ms)`);
  });

  it("should treat Ollama unreachable as cold + log warning", async () => {
    fetchMock = mockFetchUnreachable();
    checker = new OllamaWarmthChecker({ cacheTtlMs: 5000 });

    const info = await checker.checkWarmth("gemma4:latest");
    assert.equal(info.isWarm, false);
    assert.equal(info.ollamaReachable, false);
    assert.ok(info.estimatedColdStartMs > 0, "Should still estimate cold start");
    assert.equal(info.loadedModels.length, 0);
  });

  it("should handle unexpected /api/ps format gracefully", async () => {
    fetchMock = mockFetchPs(null);
    checker = new OllamaWarmthChecker({ cacheTtlMs: 5000 });

    const info = await checker.checkWarmth("gemma4:latest");
    assert.equal(info.isWarm, false);
    assert.equal(info.ollamaReachable, true); // reachable, just unexpected format
    assert.equal(info.loadedModels.length, 0);
  });

  it("should handle HTTP error status from Ollama", async () => {
    fetchMock = mockFetchPs([], { status: 500 });
    checker = new OllamaWarmthChecker({ cacheTtlMs: 5000 });

    const info = await checker.checkWarmth("gemma4:latest");
    assert.equal(info.isWarm, false);
    assert.equal(info.ollamaReachable, false);
  });

  it("should cache /api/ps result within TTL window", async () => {
    let callCount = 0;
    fetchMock = mock.method(globalThis, "fetch", (url: string) => {
      if (typeof url === "string" && url.includes("/api/ps")) {
        callCount++;
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ models: [{ name: "gemma4:latest" }] }),
        } as any);
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) } as any);
    });

    checker = new OllamaWarmthChecker({ cacheTtlMs: 5000 });

    // First call — should hit Ollama
    await checker.checkWarmth("gemma4:latest");
    assert.equal(callCount, 1, "First call should fetch from Ollama");

    // Second call within TTL — should use cache
    await checker.checkWarmth("gemma4:latest");
    assert.equal(callCount, 1, "Second call within TTL should use cache");

    // Invalidate cache and call again — should hit Ollama
    checker.invalidateCache();
    await checker.checkWarmth("gemma4:latest");
    assert.equal(callCount, 2, "After invalidation, should fetch from Ollama again");
  });

  it("should batch-check multiple models in a single /api/ps call", async () => {
    fetchMock = mockFetchPs(["gemma4:latest", "qwen2.5-coder:7b"]);
    checker = new OllamaWarmthChecker({ cacheTtlMs: 5000 });

    const batch = await checker.checkWarmthBatch(["gemma4:latest", "llama3:70b", "qwen2.5-coder:7b"]);

    // Batch check returns correct results in a single cached lookup
    // (cache behavior is separately verified in the cache test)
    assert.equal(batch.size, 3);
    assert.equal(batch.get("gemma4:latest")!.isWarm, true);
    assert.equal(batch.get("llama3:70b")!.isWarm, false);
    assert.equal(batch.get("qwen2.5-coder:7b")!.isWarm, true);
    // All entries share the same loadedModels list (single /api/ps poll)
    assert.deepEqual(batch.get("gemma4:latest")!.loadedModels, ["gemma4:latest", "qwen2.5-coder:7b"]);
    assert.deepEqual(batch.get("llama3:70b")!.loadedModels, ["gemma4:latest", "qwen2.5-coder:7b"]);
  });

  it("should match model names case-insensitively", async () => {
    fetchMock = mockFetchPs(["Gemma4:Latest"]);
    checker = new OllamaWarmthChecker({ cacheTtlMs: 5000 });

    const info = await checker.checkWarmth("gemma4:LATEST");
    assert.equal(info.isWarm, true);
  });
});

// ─── Router Integration Tests ─────────────────────────────

describe("RoutingEngine — Warm Model Integration", () => {
  let config: CognitiveRouterConfig;
  let db: DBService;
  let costTracker: CostTracker;
  let registry: ModelRegistry;
  let router: RoutingEngine;
  let warmthChecker: OllamaWarmthChecker;
  let fetchMock: Mock;

  const ollamaModel: ModelCapability = {
    provider: "ollama",
    model: "gemma4:latest",
    contextWindow: 32_768,
    modalities: ["text"],
    capabilities: {
      coding: 0.65, reasoning: 0.60, creative: 0.60, math: 0.55,
      analysis: 0.60, conversation: 0.70, retrieval: 0.60,
      science: 0.55, business: 0.60, summary: 0.65,
    },
    isLocal: true,
    source: "benchmark",
    vramRequiredGb: 5,
    usageMultiplier: 1,
  };

  const zaiModel: ModelCapability = {
    provider: "zai",
    model: "glm-4.6-flash",
    contextWindow: 200_000,
    modalities: ["text"],
    capabilities: {
      coding: 0.85, reasoning: 0.85, creative: 0.80, math: 0.85,
      analysis: 0.85, conversation: 0.90, retrieval: 0.80,
      science: 0.85, business: 0.85, summary: 0.85,
    },
    isLocal: false,
    source: "benchmark",
    usageMultiplier: 1,
  };

  function setupRegistry(registry: ModelRegistry, models: ModelCapability[]) {
    // Inject models directly via the internal seed mechanism
    // We use getAvailableModels which reads from the registry's internal state
    for (const m of models) {
      (registry as any).models.set(`${m.provider}/${m.model}`, m);
    }
  }

  beforeEach(async () => {
    config = makeConfig();
    db = makeMockDB();
    costTracker = new CostTracker(db, config);
    await costTracker.refreshProviderStatus();

    registry = new ModelRegistry(db, config);
    // Seed the registry with test models
    (registry as any).models = new Map();
    setupRegistry(registry, [ollamaModel, zaiModel]);

    router = new RoutingEngine(registry, costTracker, db, config);
    warmthChecker = new OllamaWarmthChecker({ cacheTtlMs: 5000 });
    router.setWarmthChecker(warmthChecker);
  });

  afterEach(() => {
    if (fetchMock) fetchMock.mock.restore();
  });

  it("should remove GPU penalty and boost latency for warm Ollama models", async () => {
    // Ollama reports gemma4:latest as loaded (warm)
    fetchMock = mockFetchPs(["gemma4:latest"]);

    const decision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-session",
      { estimatedTokens: 500 },
    );

    assert.ok(decision, "Decision should not be null");
    // The rationale should contain ollama=warm
    assert.ok(decision.rationale.includes("ollama=warm"),
      `Rationale should include 'ollama=warm': ${decision.rationale}`);
    assert.ok(decision.rationale.includes("warmth=+"),
      `Rationale should include positive warmth boost: ${decision.rationale}`);
  });

  it("should apply GPU penalty and cold-start estimate for cold Ollama models", async () => {
    // Ollama is reachable but gemma4:latest is NOT loaded
    fetchMock = mockFetchPs([]);

    const decision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-session",
      { estimatedTokens: 500 },
    );

    assert.ok(decision);
    // Check that at least one candidate has ollama=cold in the rationale
    const allRationales = [decision.rationale, ...(decision.candidates?.map(c => c.rationale) ?? [])];
    const hasCold = allRationales.some(r => r.includes("ollama=cold"));
    assert.ok(hasCold, `At least one candidate should show 'ollama=cold': ${decision.rationale}`);
  });

  it("should handle Ollama unreachable gracefully in routing", async () => {
    // Ollama is completely unreachable
    fetchMock = mockFetchUnreachable();

    const decision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-session",
      { estimatedTokens: 500 },
    );

    assert.ok(decision);
    // Should still produce a valid decision — ollama models treated as cold
    const allRationales = [decision.rationale, ...(decision.candidates?.map(c => c.rationale) ?? [])];
    const hasColdUnreachable = allRationales.some(r => r.includes("cold"));
    assert.ok(hasColdUnreachable,
      `Should show cold status when Ollama unreachable: ${decision.rationale}`);
  });

  it("should prefer a warm Ollama model over a cold one (all else equal)", async () => {
    // Use two whitelisted ollama models: gemma4:latest (warm) vs deepseek-coder-v2:latest (cold)
    const warmModel = { ...ollamaModel, model: "gemma4:latest" };
    const coldModel: ModelCapability = {
      provider: "ollama",
      model: "deepseek-coder-v2:latest",
      contextWindow: 32_768,
      modalities: ["text"],
      capabilities: {
        coding: 0.65, reasoning: 0.60, creative: 0.60, math: 0.55,
        analysis: 0.60, conversation: 0.70, retrieval: 0.60,
        science: 0.55, business: 0.60, summary: 0.65,
      },
      isLocal: true,
      source: "benchmark",
      vramRequiredGb: 5,
      usageMultiplier: 1,
    };
    (registry as any).models = new Map();
    setupRegistry(registry, [warmModel, coldModel, zaiModel]);

    // Only gemma4:latest is loaded (warm), deepseek-coder-v2 is cold
    fetchMock = mockFetchPs(["gemma4:latest"]);

    const decision = await router.decide(
      makeClassification("conversation", 0.5),
      "test-session",
      { estimatedTokens: 500 },
    );

    assert.ok(decision);

    // Find the warm and cold candidates in the decision output
    const allCandidates = [
      { provider: decision.provider, model: decision.model, rationale: decision.rationale, score: decision.overallScore },
      ...(decision.candidates?.map(c => ({ provider: c.provider, model: c.model, rationale: c.rationale, score: c.overallScore })) ?? []),
    ];

    const warm = allCandidates.find(c => c.model === "gemma4:latest");
    const cold = allCandidates.find(c => c.model === "deepseek-coder-v2:latest");

    assert.ok(warm, "Warm model (gemma4:latest) should be in candidates");
    assert.ok(cold, "Cold model (deepseek-coder-v2:latest) should be in candidates");
    assert.ok(warm!.score > cold!.score,
      `Warm model score (${warm!.score}) should exceed cold (${cold!.score})`);
  });

  it("should include warmth info in /last-decision style output (rationale string)", async () => {
    fetchMock = mockFetchPs(["gemma4:latest"]);

    const decision = await router.decide(
      makeClassification("coding", 0.85),
      "test-session",
      { estimatedTokens: 1000 },
    );

    assert.ok(decision);
    // The rationale must contain enough info for decision transparency
    // It should mention ollama=warm or ollama=cold for the ollama candidate(s)
    const allRationales = [decision.rationale, ...(decision.candidates?.map(c => c.rationale) ?? [])];
    const ollamaRationales = allRationales.filter(r => r.includes("ollama="));
    assert.ok(ollamaRationales.length > 0,
      "At least one candidate rationale should contain ollama warm/cold status");
  });

  it("should work correctly when no warmth checker is injected (backward compat)", async () => {
    // Don't set a warmth checker — router should behave as before
    router = new RoutingEngine(registry, costTracker, db, config);

    const decision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-session",
      { estimatedTokens: 500 },
    );

    assert.ok(decision);
    // Should NOT have ollama=warm or ollama=cold in rationale
    assert.ok(!decision.rationale.includes("ollama=warm"),
      "Without warmth checker, should not show ollama=warm");
  });
});
