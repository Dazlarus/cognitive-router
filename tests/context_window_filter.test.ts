// tests/context_window_filter.test.ts — Unit tests for the context-window
// pre-filter in candidate selection (Switchyard easy win #1, Aug 2026).
// Run with: npx tsx --test tests/context_window_filter.test.ts
//
// Covers:
//  1. Filter logic — small-window candidates skipped with a structured
//     skipReason (context_window_exceeded) visible on the decision.
//  2. Candidate ordering with mixed context windows — window filter
//     overrides raw score when the request does not fit.
//  3. All-filtered degradation path — best-scoring model still wins,
//     allFilteredDegraded flag set, warn logged, never a hard-fail.
//  4. Token estimator invariants (heuristic, no new tokenizer dependency).

import { describe, it, beforeEach, afterEach, mock } from "node:test";
import assert from "node:assert/strict";
import { RoutingEngine } from "../src/router.ts";
import { CostTracker } from "../src/cost_tracker.ts";
import { ModelRegistry, type ModelCapability } from "../src/model_registry.ts";
import { DBService } from "../src/db_service.ts";
import { loadConfig, type CognitiveRouterConfig } from "../src/config.ts";
import { logger } from "../src/logger.ts";
import { estimateTokenCount } from "../src/proxy-stream.ts";
import type { Classification } from "../src/classifier.ts";

// ─── Helpers (mirroring router.test.ts patterns) ───

function makeConfig(overrides: Partial<CognitiveRouterConfig> = {}): CognitiveRouterConfig {
  const base = loadConfig({
    enabled: true,
    logLevel: "warn",
    providerPriority: ["openrouter", "gemini"],
    providers: {
      openrouter: { budgetType: "free", priority: "high" },
      gemini: { budgetType: "credits", priority: "medium" },
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
    getRecentDecisions: (limit?: number) => decisions.slice(-(limit ?? 10)),
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
    getAllLatestChatBenchmarks: () => new Map(),
    close: () => {},
    _decisions: decisions,
  } as any;
}

const NO_CAPS = {
  coding: 0, reasoning: 0, creative: 0, math: 0, analysis: 0,
  conversation: 0, retrieval: 0, science: 0, business: 0, summary: 0,
};

/** Build a minimal ModelCapability for the stub registry. */
function makeCap(provider: string, model: string, contextWindow: number): ModelCapability {
  return {
    provider,
    model,
    contextWindow,
    modalities: ["text"],
    capabilities: { ...NO_CAPS },
    isLocal: false,
    // Remotes must carry explicit pricing (free-vs-unknown rule) or the
    // quarantine filter correctly drops them (test would hit last-resort).
    costPer1kInput: 0.001,
    costPer1kOutput: 0.002,
    source: "benchmark",
    planEligible: true,
  };
}

/** Stub registry: full control over candidates + per-model capability score.
 *  Only the surface RoutingEngine touches is implemented. */
class StubRegistry {
  constructor(
    private models: ModelCapability[],
    private scoreFor: (m: ModelCapability) => number,
  ) {}
  getAvailableModels(availableProviders: string[]): ModelCapability[] {
    return this.models.filter((m) => availableProviders.includes(m.provider));
  }
  getCapabilityScore(provider: string, model: string): number {
    const m = this.models.find((x) => x.provider === provider && x.model === model);
    return m ? this.scoreFor(m) : 0.5;
  }
  getExplorationScore(provider: string, model: string): number {
    return this.getCapabilityScore(provider, model);
  }
}

function makeClassification(intent = "coding", confidence = 0.9): Classification {
  return { intent, confidence };
}

// Shared fixture: mixed context windows on the same provider so cost/latency
// adjustments stay symmetric and capability drives ordering.
const SMALL = makeCap("openrouter", "small-window-model", 8_000);   // effective limit 6,400
const BIG = makeCap("openrouter", "big-window-model", 200_000);     // effective limit 160,000
const MIXED_MODELS = [SMALL, BIG];

describe("Context window pre-filter — filter logic", () => {
  let config: CognitiveRouterConfig;
  let db: DBService;
  let costTracker: CostTracker;

  beforeEach(async () => {
    config = makeConfig();
    db = makeMockDB();
    costTracker = new CostTracker(db, config);
    await costTracker.refreshProviderStatus();
  });

  it("skips candidates whose window is smaller than the estimate, with structured skipReason", async () => {
    const registry = new StubRegistry(MIXED_MODELS, () => 0.8) as unknown as ModelRegistry;
    const router = new RoutingEngine(registry, costTracker, db, config);

    const decision = await router.decide(makeClassification(), "test-session", { estimatedTokens: 50_000 });

    assert.ok(decision, "should return a decision");
    assert.equal(decision!.provider, "openrouter");
    assert.equal(decision!.model, "big-window-model", "only the big-window model fits");

    // Structured filter info attached to the decision (→ contextFilterJson in decision logs)
    assert.ok(decision!.contextFilter, "contextFilter should be attached");
    const cf = decision!.contextFilter!;
    assert.equal(cf.estimatedTokens, 50_000);
    assert.equal(cf.safetyFactor, 0.8);
    assert.equal(cf.maxContextWindow, 200_000);
    assert.equal(cf.allFilteredDegraded, undefined, "not degraded — a candidate fit");

    assert.equal(cf.filteredOut.length, 1, "exactly the small-window model is filtered");
    const entry = cf.filteredOut[0];
    assert.equal(entry.provider, "openrouter");
    assert.equal(entry.model, "small-window-model");
    assert.equal(entry.contextWindow, 8_000);
    assert.equal(entry.effectiveLimit, 6_400, "effective limit = floor(0.8 × 8000)");
    assert.equal(entry.skipReason, "context_window_exceeded", "machine-readable skip reason");

    // Filtered model must not appear in the runner-up candidates
    const candidateNames = (decision!.candidates ?? []).map((c) => c.model);
    assert.equal(candidateNames.includes("small-window-model"), false,
      "filtered model must not appear in candidates list");
  });

  it("boundary: estimate at exactly the effective limit passes, one over is filtered", async () => {
    const registry = new StubRegistry(MIXED_MODELS, () => 0.8) as unknown as ModelRegistry;
    const router = new RoutingEngine(registry, costTracker, db, config);

    // 6,400 == effectiveLimit of the small model → not "greater than" → passes
    const atLimit = await router.decide(makeClassification(), "s1", { estimatedTokens: 6_400 });
    assert.equal(atLimit!.model, "small-window-model", "estimate == limit should pass");
    assert.equal(atLimit!.contextFilter!.filteredOut.length, 0);

    // 6,401 > 6,400 → filtered
    const overLimit = await router.decide(makeClassification(), "s2", { estimatedTokens: 6_401 });
    assert.equal(overLimit!.model, "big-window-model", "estimate one over limit should filter small model");
    assert.equal(overLimit!.contextFilter!.filteredOut.length, 1);
  });

  it("no estimate (estimatedTokens=0) → no filtering, contextFilter absent", async () => {
    const registry = new StubRegistry(MIXED_MODELS, () => 0.8) as unknown as ModelRegistry;
    const router = new RoutingEngine(registry, costTracker, db, config);

    const decision = await router.decide(makeClassification(), "s3", {});
    assert.ok(decision);
    assert.equal(decision!.contextFilter, undefined, "no context filter when size unknown");
  });
});

describe("Context window pre-filter — candidate ordering with mixed windows", () => {
  let config: CognitiveRouterConfig;
  let db: DBService;
  let costTracker: CostTracker;

  beforeEach(async () => {
    config = makeConfig();
    db = makeMockDB();
    costTracker = new CostTracker(db, config);
    await costTracker.refreshProviderStatus();
  });

  it("a higher-scoring small-window model wins small requests but loses large ones", async () => {
    // Small-window model is the BETTER scorer — the window filter is the only
    // thing that can demote it.
    const registry = new StubRegistry(MIXED_MODELS, (m) =>
      m.model === "small-window-model" ? 0.95 : 0.90,
    ) as unknown as ModelRegistry;
    const router = new RoutingEngine(registry, costTracker, db, config);

    // Small request: capability decides → small-window model wins
    const smallReq = await router.decide(makeClassification(), "s4", { estimatedTokens: 500 });
    assert.equal(smallReq!.model, "small-window-model",
      "small request: high-scoring small-window model wins");
    assert.equal(smallReq!.contextFilter!.filteredOut.length, 0);

    // Large request: small-window model can't fit → runner-up (big-window) wins
    const largeReq = await router.decide(makeClassification(), "s5", { estimatedTokens: 50_000 });
    assert.equal(largeReq!.model, "big-window-model",
      "large request: window filter must override raw capability ordering");
    assert.ok(largeReq!.overallScore > 0, "winning decision carries a real score");
  });
});

describe("Context window pre-filter — all-filtered degradation path", () => {
  let config: CognitiveRouterConfig;
  let db: DBService;
  let costTracker: CostTracker;
  let warnSpy: ReturnType<typeof mock.method>;

  beforeEach(async () => {
    config = makeConfig();
    db = makeMockDB();
    costTracker = new CostTracker(db, config);
    await costTracker.refreshProviderStatus();
    warnSpy = mock.method(logger, "warn");
  });

  afterEach(() => {
    warnSpy.mock.restore();
  });

  it("degrades gracefully: best-scoring model + warn log, never a hard-fail", async () => {
    // Both models exceed: 5M tokens >> 160K effective limit of the biggest.
    // The STRONGER scorer must win — that is "current behavior" preserved.
    const registry = new StubRegistry(MIXED_MODELS, (m) =>
      m.model === "small-window-model" ? 0.5 : 0.9,
    ) as unknown as ModelRegistry;
    const router = new RoutingEngine(registry, costTracker, db, config);

    const decision = await router.decide(makeClassification(), "s6", { estimatedTokens: 5_000_000 });

    // Never hard-fail: a real model is returned, no error block
    assert.ok(decision, "decision must exist");
    assert.equal(decision!.error, undefined, "degradation must not produce an error decision");
    assert.equal(decision!.provider, "openrouter");
    assert.equal(decision!.model, "big-window-model",
      "best-scoring model from the unfiltered set wins");

    // Degradation is visible + structured on the decision
    assert.ok(decision!.contextFilter, "contextFilter attached in degraded mode");
    const cf = decision!.contextFilter!;
    assert.equal(cf.allFilteredDegraded, true, "allFilteredDegraded flag must be set");
    assert.equal(cf.filteredOut.length, 2, "every candidate recorded as filtered out");
    for (const entry of cf.filteredOut) {
      assert.equal(entry.skipReason, "context_window_exceeded");
    }

    // Warn logged so operators see the fallback in logs
    const warnMessages = warnSpy.mock.calls.map((c: any) => String(c.arguments[0]));
    const degradationWarning = warnMessages.find((m) => m.includes("ALL") && m.includes("Degrading gracefully"));
    assert.ok(degradationWarning, `expected degradation warn log, got: ${JSON.stringify(warnMessages)}`);

    // Decision is serializable for the DB decision log (contextFilterJson path)
    const serialized = JSON.stringify(cf);
    assert.ok(serialized.includes("context_window_exceeded"));
    assert.ok(serialized.includes("\"allFilteredDegraded\":true"));
  });
});

describe("Context window pre-filter — token estimator", () => {
  it("covers messages and tools schema; monotonic; no zero-request inflation", () => {
    const empty = estimateTokenCount({ messages: [] } as any);
    assert.ok(empty >= 0 && empty < 10, `empty request should estimate ~0, got ${empty}`);

    // Realistic filler: repeated single chars get BPE-merged aggressively
    // (~8 chars/token), which is correct tokenizer behavior - the ~4
    // chars/token heuristic only holds for word-like text.
    const filler = (n: number) => "the quick brown fox jumps over the lazy dog ".repeat(Math.ceil(n / 45)).slice(0, n);
    const short = estimateTokenCount({ messages: [{ role: "user", content: filler(400) }] } as any);
    const long = estimateTokenCount({ messages: [{ role: "user", content: filler(4_000) }] } as any);
    assert.ok(long > short, "estimate must grow with content size");

    // Heuristic scale: 4,000 chars ≈ 800–1,400 tokens under both the
    // tiktoken path (~4 chars/token) and the chars/3.5 fallback.
    assert.ok(long >= 700 && long <= 1_500, `4,000 chars should estimate ~1K tokens, got ${long}`);

    const withTools = estimateTokenCount({
      messages: [{ role: "user", content: "x".repeat(400) }],
      tools: [{
        type: "function",
        function: {
          name: "example_tool",
          description: "x".repeat(1_000),
          parameters: { type: "object", properties: { a: { type: "string" } } },
        },
      }],
    } as any);
    assert.ok(withTools > short + 100,
      `tool schema must add to the estimate: ${withTools} vs ${short}`);
  });
});
