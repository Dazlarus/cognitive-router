// tests/latency_skip.test.ts — Proactive latency-degradation skip (throttle avoidance)
// Daz-approved 2026-09-10: skip providers whose health shows SUSTAINED latency
// degradation BEFORE a request fails, instead of only failing over on hard errors.
// Run with: npx tsx --test tests/latency_skip.test.ts

import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { RoutingEngine } from "../src/router.ts";
import { CostTracker } from "../src/cost_tracker.ts";
import { ModelRegistry } from "../src/model_registry.ts";
import { DBService } from "../src/db_service.ts";
import { loadConfig, type CognitiveRouterConfig } from "../src/config.ts";
import type { Classification } from "../src/classifier.ts";

// ─── Test Helpers (mirrors router.test.ts) ──────────────────

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
  const outcomes: any[] = [];
  const overrides: any[] = [];
  return {
    initializeSchema: async () => {},
    recordDecision: (d: any) => decisions.push(d),
    recordCallOutcome: (o: any) => outcomes.push(o),
    recordRetry: (r: any) => {},
    getDecisionByRequestId: (id: string) => decisions.find((d) => d.requestId === id) ?? null,
    getRetryCount: (id: string) => 0,
    getRecentDecisions: (limit?: number) => decisions.slice(-(limit ?? 10)),
    getModelStats: () => [],
    getProviderHealth: () => [],
    getSpendByProvider: () => [],
    getSpend: () => 0,
    loadCapabilityOverrides: () => overrides,
    getCapabilityOverride: (provider: string, model: string, intent: string) =>
      overrides.find((o) => o.provider === provider && o.model === model && o.intent === intent) ?? null,
    upsertCapabilityOverride: () => {},
    recordJudgeEvaluation: () => {},
    recordSpend: () => {},
    getAllSpend: () => [],
    loadCircuitState: () => null,
    saveCircuitState: () => {},
    getAllLatestChatBenchmarks: () => new Map(),
    close: () => {},
    _decisions: decisions,
    _outcomes: outcomes,
  } as any;
}

function makeClassification(intent: string = "conversation", confidence: number = 0.9): Classification {
  return { intent, confidence };
}

async function buildStack(config: CognitiveRouterConfig, db: DBService) {
  const costTracker = new CostTracker(db, config);
  await costTracker.refreshProviderStatus();
  const registry = new ModelRegistry(db, config);
  await registry.loadCachedState();
  const router = new RoutingEngine(registry, costTracker, db, config);
  return { costTracker, registry, router };
}

// ─── Tests ──────────────────────────────────────────────────

const ORIGINAL_SKIP_MS = process.env.ROUTER_SKIP_LATENCY_MS;
const ORIGINAL_MIN_SAMPLES = process.env.ROUTER_SKIP_LATENCY_MIN_SAMPLES;
const ORIGINAL_TTL = process.env.ROUTER_SKIP_SAMPLE_TTL_MS;

describe("Proactive latency-degradation skip (throttle avoidance)", () => {
  let db: DBService;
  let config: CognitiveRouterConfig;
  let costTracker: CostTracker;
  let router: RoutingEngine;

  beforeEach(async () => {
    // Low threshold so synthetic 6s samples read as "sustained degradation".
    // Must be set BEFORE the CostTracker constructor (reads env once).
    process.env.ROUTER_SKIP_LATENCY_MS = "5000";
    delete process.env.ROUTER_SKIP_LATENCY_MIN_SAMPLES;
    delete process.env.ROUTER_SKIP_SAMPLE_TTL_MS;
    config = makeConfig();
    db = makeMockDB();
    ({ costTracker, router } = await buildStack(config, db));
  });

  afterEach(() => {
    if (ORIGINAL_SKIP_MS === undefined) delete process.env.ROUTER_SKIP_LATENCY_MS;
    else process.env.ROUTER_SKIP_LATENCY_MS = ORIGINAL_SKIP_MS;
    if (ORIGINAL_MIN_SAMPLES === undefined) delete process.env.ROUTER_SKIP_LATENCY_MIN_SAMPLES;
    else process.env.ROUTER_SKIP_LATENCY_MIN_SAMPLES = ORIGINAL_MIN_SAMPLES;
    if (ORIGINAL_TTL === undefined) delete process.env.ROUTER_SKIP_SAMPLE_TTL_MS;
    else process.env.ROUTER_SKIP_SAMPLE_TTL_MS = ORIGINAL_TTL;
  });

  it("skips a provider with sustained latency degradation while a healthy alternative exists", async () => {
    // Synthetic health data: 4 fresh slow successes — degraded but never FAILED,
    // exactly the pre-failure window this feature exists for.
    for (let i = 0; i < 4; i++) {
      await costTracker.recordCall("zai", { durationMs: 17_000, outcome: "success" });
    }
    assert.equal(costTracker.isLatencyDegraded("zai"), true, "4x17s samples must read as degraded");

    const decision = await router.decide(makeClassification(), "session:lat-skip", {});
    assert.ok(decision, "decision must still be produced");
    assert.notEqual(decision!.provider, "zai", "degraded provider must not win ranking");
    const runnerUps = (decision!.candidates ?? []).map((c) => c.provider);
    assert.ok(!runnerUps.includes("zai"), "degraded provider must be excluded from candidate ranking");
  });

  it("recovers automatically when provider health improves", async () => {
    for (let i = 0; i < 3; i++) {
      await costTracker.recordCall("zai", { durationMs: 6_000, outcome: "success" });
    }
    assert.equal(costTracker.isLatencyDegraded("zai"), true);

    // Health improves: fresh fast samples pull the window average back under
    // the threshold — no flags to clear, recovery is inherent.
    for (let i = 0; i < 3; i++) {
      await costTracker.recordCall("zai", { durationMs: 500, outcome: "success" });
    }
    assert.equal(
      costTracker.isLatencyDegraded("zai"),
      false,
      "(3x6000 + 3x500)/6 = 3250ms < 5000ms — must recover",
    );

    const decision = await router.decide(makeClassification(), "session:lat-recover", {});
    assert.ok(decision);
    assert.equal(decision!.provider, "zai", "recovered provider must rank again");
  });

  it("does not skip on a single slow outlier (min-sample guard)", async () => {
    await costTracker.recordCall("zai", { durationMs: 17_000, outcome: "success" });
    assert.equal(costTracker.isLatencyDegraded("zai"), false, "one slow sample is not a pattern");

    const decision = await router.decide(makeClassification(), "session:outlier", {});
    assert.ok(decision);
    const ranked = [decision!.provider, ...(decision!.candidates ?? []).map((c) => c.provider)];
    assert.ok(ranked.includes("zai"), "outlier must not exclude zai from ranking");
  });

  it("keeps degraded providers ranked when EVERY candidate is degraded (never empty the pool)", async () => {
    const db2 = makeMockDB();
    const cfg2 = makeConfig({ providerPriority: ["zai", "openrouter"] });
    const { costTracker: tracker2, router: router2 } = await buildStack(cfg2, db2);

    for (const p of ["zai", "openrouter"]) {
      for (let i = 0; i < 3; i++) {
        await tracker2.recordCall(p, { durationMs: 17_000, outcome: "success" });
      }
    }
    assert.equal(tracker2.isLatencyDegraded("zai"), true);
    assert.equal(tracker2.isLatencyDegraded("openrouter"), true);

    const decision = await router2.decide(makeClassification(), "session:all-degraded", {});
    assert.ok(decision, "must still route when the whole candidate pool is degraded");
    assert.ok(
      decision!.provider === "zai" || decision!.provider === "openrouter",
      "degraded-but-only providers must still serve",
    );
  });

  it("ignores stale samples so a skipped provider is re-probed (automatic recovery)", async () => {
    for (let i = 0; i < 3; i++) {
      await costTracker.recordCall("zai", { durationMs: 17_000, outcome: "success" });
    }
    assert.equal(costTracker.isLatencyDegraded("zai"), true);

    // A skipped provider receives no traffic, so its window would freeze.
    // Simulate the TTL aging those samples out (default TTL 120s).
    const state = costTracker.getProviderState("zai")!;
    const stale = Date.now() - 10 * 60_000;
    state.recentLatencyTimes = state.recentLatencyTimes.map(() => stale);
    assert.equal(
      costTracker.isLatencyDegraded("zai"),
      false,
      "stale window = no current evidence — provider must be eligible for re-probe",
    );
  });

  it("skips a rate-limit-throttled provider while healthy alternatives exist", async () => {
    // Synthetic "throttled status" via the rate-limit pattern flag.
    for (let i = 0; i < 3; i++) {
      await costTracker.recordCall("zai", { durationMs: 100, outcome: "rate_limit" });
    }
    assert.equal(costTracker.isThrottled("zai"), true);

    const decision = await router.decide(makeClassification(), "session:throttled", {});
    assert.ok(decision);
    assert.notEqual(decision!.provider, "zai", "throttled provider must be excluded");
    const runnerUps = (decision!.candidates ?? []).map((c) => c.provider);
    assert.ok(!runnerUps.includes("zai"), "throttled provider must not appear in runner-ups");
  });

  it("can be disabled with ROUTER_SKIP_LATENCY_MS<=0", async () => {
    process.env.ROUTER_SKIP_LATENCY_MS = "0";
    const db2 = makeMockDB();
    const { costTracker: tracker2 } = await buildStack(makeConfig(), db2);

    for (let i = 0; i < 4; i++) {
      await tracker2.recordCall("zai", { durationMs: 60_000, outcome: "success" });
    }
    assert.equal(tracker2.getLatencySkipMs(), 0);
    assert.equal(tracker2.isLatencyDegraded("zai"), false, "disabled skip never degrades a provider");
  });

  it("default threshold is generous — never trips on realistic degraded-but-alive latencies", async () => {
    delete process.env.ROUTER_SKIP_LATENCY_MS;
    const db2 = makeMockDB();
    const { costTracker: tracker2 } = await buildStack(makeConfig(), db2);

    assert.equal(tracker2.getLatencySkipMs(), 30_000, "default must be 30s (generous)");
    // The observed zai degradation (17.4s avg) is below the generous default:
    // healthy fleets can never trip it, and tuning down is explicit.
    for (let i = 0; i < 5; i++) {
      await tracker2.recordCall("zai", { durationMs: 17_400, outcome: "success" });
    }
    assert.equal(tracker2.isLatencyDegraded("zai"), false);
  });
});
