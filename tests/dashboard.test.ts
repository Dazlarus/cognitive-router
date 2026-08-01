// tests/dashboard.test.ts — Tests for the /v1/dashboard aggregation queries
// Run with: npx tsx --test tests/dashboard.test.ts
//
// Uses a REAL temporary SQLite database — no mocks.

import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, rmSync } from "node:fs";
import { resolve } from "node:path";
import { DBService } from "../src/db_service.ts";

// ─── Helpers ────────────────────────────────────────────────

let dbPath: string;
let db: DBService;
let counter = 0;

function makeTempDbPath(): string {
  counter++;
  const name = `data/test-dash-${Date.now()}-${counter}-${Math.random().toString(36).slice(2)}.db`;
  return resolve(name);
}

function isoTime(minutesAgo: number): string {
  return new Date(Date.now() - minutesAgo * 60_000).toISOString();
}

// ─── Setup / Teardown ───────────────────────────────────────

beforeEach(() => {
  dbPath = makeTempDbPath();
  mkdirSync(resolve(dbPath, ".."), { recursive: true });
  db = new DBService(dbPath);
});

afterEach(() => {
  try { db.close(); } catch { /* already closed */ }
  try { rmSync(dbPath, { force: true }); } catch { /* ignore */ }
  for (const suffix of ["-wal", "-shm"]) {
    try { rmSync(dbPath + suffix, { force: true }); } catch { /* ignore */ }
  }
});

// ─── Tests ──────────────────────────────────────────────────

describe("DBService.parseTimeRange", () => {
  it("should parse '1h' as 1 hour back", () => {
    const since = DBService.parseTimeRange("1h");
    const expectedMs = Date.now() - 3_600_000;
    const actualMs = new Date(since).getTime();
    assert.ok(Math.abs(actualMs - expectedMs) < 5000, "Should be within 5s of 1h ago");
  });

  it("should parse '24h' as 24 hours back", () => {
    const since = DBService.parseTimeRange("24h");
    const expectedMs = Date.now() - 24 * 3_600_000;
    const actualMs = new Date(since).getTime();
    assert.ok(Math.abs(actualMs - expectedMs) < 5000);
  });

  it("should parse '7d' as 7 days back", () => {
    const since = DBService.parseTimeRange("7d");
    const expectedMs = Date.now() - 7 * 86_400_000;
    const actualMs = new Date(since).getTime();
    assert.ok(Math.abs(actualMs - expectedMs) < 5000);
  });

  it("should parse '1w' as 7 days back", () => {
    const since = DBService.parseTimeRange("1w");
    const expectedMs = Date.now() - 7 * 86_400_000;
    const actualMs = new Date(since).getTime();
    assert.ok(Math.abs(actualMs - expectedMs) < 5000);
  });

  it("should default to 24h for invalid input", () => {
    const since = DBService.parseTimeRange("invalid");
    const expectedMs = Date.now() - 24 * 3_600_000;
    const actualMs = new Date(since).getTime();
    assert.ok(Math.abs(actualMs - expectedMs) < 5000);
  });
});

describe("Dashboard — providerDistribution", () => {
  beforeEach(async () => {
    await db.initializeSchema();
  });

  it("should return provider distribution with counts and percentages", () => {
    const since = isoTime(60);

    // Insert test decisions
    for (let i = 0; i < 10; i++) {
      db.recordDecision({
        timestamp: isoTime(10),
        sessionKey: "sess",
        messageHash: `h${i}`,
        intent: "coding",
        confidence: 0.9,
        provider: "zai",
        model: "glm-5.1",
        scores: {},
        overallScore: 0.85,
        outcome: "success",
        requestId: `req_zai_${i}`,
      });
    }
    for (let i = 0; i < 5; i++) {
      db.recordDecision({
        timestamp: isoTime(10),
        sessionKey: "sess",
        messageHash: `h2${i}`,
        intent: "coding",
        confidence: 0.9,
        provider: "openrouter",
        model: "free-model",
        scores: {},
        overallScore: 0.75,
        outcome: "success",
        requestId: `req_or_${i}`,
      });
    }

    const dist = db.getDashboardProviderDistribution(since);
    assert.equal(dist.length, 2);

    assert.equal(dist[0].provider, "zai");
    assert.equal(dist[0].requests, 10);
    assert.equal(dist[0].percentage, 66.67);

    assert.equal(dist[1].provider, "openrouter");
    assert.equal(dist[1].requests, 5);
    assert.equal(dist[1].percentage, 33.33);
  });

  it("should calculate success and failure rates per provider", () => {
    const since = isoTime(120);

    db.recordDecision({
      timestamp: isoTime(10), sessionKey: "s", messageHash: "h1",
      intent: "coding", confidence: 0.9, provider: "zai", model: "m1",
      scores: {}, overallScore: 0.8, outcome: "success", requestId: "r1",
    });
    db.recordDecision({
      timestamp: isoTime(10), sessionKey: "s", messageHash: "h2",
      intent: "coding", confidence: 0.9, provider: "zai", model: "m1",
      scores: {}, overallScore: 0.8, outcome: "error", requestId: "r2",
    });
    db.recordDecision({
      timestamp: isoTime(10), sessionKey: "s", messageHash: "h3",
      intent: "coding", confidence: 0.9, provider: "zai", model: "m1",
      scores: {}, overallScore: 0.8, outcome: "timeout", requestId: "r3",
    });

    const dist = db.getDashboardProviderDistribution(since);
    assert.equal(dist.length, 1);
    assert.equal(dist[0].requests, 3);
    assert.equal(dist[0].successes, 1);
    assert.equal(dist[0].failures, 2);
    assert.equal(dist[0].successRate, 33.33);
  });

  it("should return empty array when no data", () => {
    const dist = db.getDashboardProviderDistribution(isoTime(60));
    assert.equal(dist.length, 0);
  });
});

describe("Dashboard — costByIntent", () => {
  beforeEach(async () => {
    await db.initializeSchema();
  });

  it("should aggregate decisions by intent", () => {
    const since = isoTime(120);

    for (let i = 0; i < 8; i++) {
      db.recordDecision({
        timestamp: isoTime(10), sessionKey: "s", messageHash: `c${i}`,
        intent: "coding", confidence: 0.9, provider: "zai", model: "m1",
        scores: {}, overallScore: 0.8, outcome: "success", requestId: `ci${i}`,
      });
    }
    for (let i = 0; i < 4; i++) {
      db.recordDecision({
        timestamp: isoTime(10), sessionKey: "s", messageHash: `r${i}`,
        intent: "research", confidence: 0.9, provider: "gemini", model: "m2",
        scores: {}, overallScore: 0.8, outcome: "success", requestId: `ri${i}`,
      });
    }

    const costs = db.getDashboardCostByIntent(since);
    assert.ok(costs.length >= 2);

    const coding = costs.find(c => c.intent === "coding");
    assert.ok(coding);
    assert.equal(coding!.decisions, 8);

    const research = costs.find(c => c.intent === "research");
    assert.ok(research);
    assert.equal(research!.decisions, 4);
  });

  it("should estimate spend proportionally from provider_spend", () => {
    const since = isoTime(120);

    // Record spend for zai
    db.recordSpend("zai", 1.50, "daily");

    // All coding decisions go to zai
    for (let i = 0; i < 5; i++) {
      db.recordDecision({
        timestamp: isoTime(5), sessionKey: "s", messageHash: `s${i}`,
        intent: "coding", confidence: 0.9, provider: "zai", model: "m1",
        scores: {}, overallScore: 0.8, outcome: "success", requestId: `sp${i}`,
      });
    }

    const costs = db.getDashboardCostByIntent(since);
    const coding = costs.find(c => c.intent === "coding");
    assert.ok(coding);
    assert.ok(coding!.estimatedSpendUsd > 0, "Should have non-zero spend");
  });
});

describe("Dashboard — latencyTrends", () => {
  beforeEach(async () => {
    await db.initializeSchema();
  });

  it("should return per-provider latency for multiple bucket sizes", () => {
    const since = isoTime(60);

    // Record some call outcomes
    db.recordCallOutcome({
      provider: "zai", model: "glm-5.1", durationMs: 800,
      outcome: "success", timestamp: isoTime(5),
    });
    db.recordCallOutcome({
      provider: "zai", model: "glm-5.1", durationMs: 1200,
      outcome: "success", timestamp: isoTime(3),
    });
    db.recordCallOutcome({
      provider: "openrouter", model: "free-model", durationMs: 2000,
      outcome: "success", timestamp: isoTime(2),
    });

    const trends = db.getDashboardLatencyTrends(since);
    assert.ok(trends.length >= 2);

    const zai = trends.find(t => t.provider === "zai");
    assert.ok(zai);

    // Check that all 4 bucket sizes exist
    const bucketNames = zai!.buckets.map(b => b.bucket);
    assert.ok(bucketNames.includes("1h"));
    assert.ok(bucketNames.includes("6h"));
    assert.ok(bucketNames.includes("24h"));
    assert.ok(bucketNames.includes("7d"));

    // 1h bucket should have data (our calls were within last hour)
    const bucket1h = zai!.buckets.find(b => b.bucket === "1h");
    assert.ok(bucket1h);
    assert.ok(bucket1h!.samples >= 2, "1h bucket should have at least 2 samples");
    assert.ok(bucket1h!.avgMs != null);
    assert.ok(bucket1h!.avgMs! >= 800 && bucket1h!.avgMs! <= 1200);
  });

  it("should return empty buckets for providers with no data", () => {
    const since = isoTime(60);
    const trends = db.getDashboardLatencyTrends(since);
    assert.equal(trends.length, 0);
  });
});

describe("Dashboard — modelMarketShare", () => {
  beforeEach(async () => {
    await db.initializeSchema();
  });

  it("should show model market share with counts and percentages", () => {
    const since = isoTime(60);

    for (let i = 0; i < 15; i++) {
      db.recordDecision({
        timestamp: isoTime(10), sessionKey: "s", messageHash: `a${i}`,
        intent: "coding", confidence: 0.9, provider: "zai", model: "glm-5.1",
        scores: {}, overallScore: 0.85, outcome: "success", requestId: `ms${i}`,
      });
    }
    for (let i = 0; i < 5; i++) {
      db.recordDecision({
        timestamp: isoTime(10), sessionKey: "s", messageHash: `b${i}`,
        intent: "coding", confidence: 0.9, provider: "openrouter", model: "free-model",
        scores: {}, overallScore: 0.75, outcome: "success", requestId: `ms2${i}`,
      });
    }

    const share = db.getDashboardModelMarketShare(since, "1h");
    assert.ok(share.length >= 2);

    const top = share[0];
    assert.equal(top.provider, "zai");
    assert.equal(top.model, "glm-5.1");
    assert.equal(top.requests, 15);
    assert.equal(top.percentage, 75);

    // Timeseries should have at least one bucket
    assert.ok(top.timeseries.length >= 1, "Should have timeseries data");
    assert.ok(top.timeseries[0].count > 0);
  });

  it("should return empty array when no data", () => {
    const share = db.getDashboardModelMarketShare(isoTime(60), "1h");
    assert.equal(share.length, 0);
  });
});

describe("Dashboard — modelSuccessRates", () => {
  beforeEach(async () => {
    await db.initializeSchema();
  });

  it("should compute success rates per provider/model from call_outcomes", () => {
    const since = isoTime(60);

    // zai/glm-5.1: 3 success, 1 error
    db.recordCallOutcome({ provider: "zai", model: "glm-5.1", durationMs: 800, outcome: "success", timestamp: isoTime(5) });
    db.recordCallOutcome({ provider: "zai", model: "glm-5.1", durationMs: 900, outcome: "success", timestamp: isoTime(4) });
    db.recordCallOutcome({ provider: "zai", model: "glm-5.1", durationMs: 1000, outcome: "success", timestamp: isoTime(3) });
    db.recordCallOutcome({ provider: "zai", model: "glm-5.1", durationMs: 500, outcome: "error", timestamp: isoTime(2) });

    // openrouter/free-model: 1 success, 1 timeout
    db.recordCallOutcome({ provider: "openrouter", model: "free-model", durationMs: 1500, outcome: "success", timestamp: isoTime(4) });
    db.recordCallOutcome({ provider: "openrouter", model: "free-model", durationMs: 30000, outcome: "timeout", timestamp: isoTime(2) });

    const rates = db.getDashboardModelSuccessRates(since);
    assert.ok(rates.length >= 2);

    const zai = rates.find(r => r.provider === "zai" && r.model === "glm-5.1");
    assert.ok(zai);
    assert.equal(zai!.total, 4);
    assert.equal(zai!.successes, 3);
    assert.equal(zai!.failures, 1);
    assert.equal(zai!.successRate, 75);
    assert.ok(zai!.avgLatencyMs != null);
    // avg latency = (800 + 900 + 1000) / 3 = 900
    assert.equal(zai!.avgLatencyMs, 900);

    const or = rates.find(r => r.provider === "openrouter" && r.model === "free-model");
    assert.ok(or);
    assert.equal(or!.total, 2);
    assert.equal(or!.successRate, 50);
  });

  it("should return empty array when no call_outcomes data", () => {
    const rates = db.getDashboardModelSuccessRates(isoTime(60));
    assert.equal(rates.length, 0);
  });
});

describe("Dashboard — outcomeDistribution", () => {
  beforeEach(async () => {
    await db.initializeSchema();
  });

  it("should return outcome distribution with percentages", () => {
    const since = isoTime(60);

    const outcomes = ["success", "success", "success", "error", "timeout", "rate_limit"];
    for (let i = 0; i < outcomes.length; i++) {
      db.recordDecision({
        timestamp: isoTime(10), sessionKey: "s", messageHash: `o${i}`,
        intent: "coding", confidence: 0.9, provider: "zai", model: "m1",
        scores: {}, overallScore: 0.8, outcome: outcomes[i], requestId: `od${i}`,
      });
    }

    const dist = db.getDashboardOutcomeDistribution(since);
    assert.ok(dist.length >= 3);

    const success = dist.find(d => d.outcome === "success");
    assert.ok(success);
    assert.equal(success!.count, 3);
    assert.equal(success!.percentage, 50);

    const error = dist.find(d => d.outcome === "error");
    assert.ok(error);
    assert.equal(error!.count, 1);
    assert.equal(error!.percentage, 16.67);

    const timeout = dist.find(d => d.outcome === "timeout");
    assert.ok(timeout);
    assert.equal(timeout!.count, 1);
    assert.equal(timeout!.percentage, 16.67);
  });

  it("should return empty array when no data", () => {
    const dist = db.getDashboardOutcomeDistribution(isoTime(60));
    assert.equal(dist.length, 0);
  });
});

describe("Dashboard — getDashboardData (full payload)", () => {
  beforeEach(async () => {
    await db.initializeSchema();
  });

  it("should return a complete dashboard payload with all sections", () => {
    // Seed some data
    for (let i = 0; i < 5; i++) {
      db.recordDecision({
        timestamp: isoTime(10), sessionKey: "s", messageHash: `f${i}`,
        intent: "coding", confidence: 0.9, provider: "zai", model: "glm-5.1",
        scores: {}, overallScore: 0.85, outcome: i < 4 ? "success" : "error",
        requestId: `full${i}`,
      });
    }
    db.recordCallOutcome({
      provider: "zai", model: "glm-5.1", durationMs: 1000,
      outcome: "success", timestamp: isoTime(5),
    });
    db.recordSpend("zai", 0.50, "daily");

    const data = db.getDashboardData("1h");

    // Verify top-level structure
    assert.ok(data.timeRange);
    assert.ok((data.timeRange as any).label === "1h");
    assert.ok((data.timeRange as any).since);
    assert.ok((data.timeRange as any).generatedAt);

    assert.ok(data.summary);
    assert.ok(data.providerDistribution);
    assert.ok(data.costByIntent);
    assert.ok(data.latencyTrends);
    assert.ok(data.modelMarketShare);
    assert.ok(data.modelSuccessRates);
    assert.ok(data.outcomeDistribution);

    // Summary should have totalRequests
    const summary = data.summary as Record<string, unknown>;
    // summary.totalRequests is { cnt: number } from the query
    const totalReqs = (summary.totalRequests as any)?.cnt ?? summary.totalRequests;
    assert.ok(typeof totalReqs === "number");
    assert.ok(totalReqs >= 5, `Expected at least 5 total requests, got ${totalReqs}`);
  });

  it("should handle empty database gracefully", () => {
    const data = db.getDashboardData("24h");

    assert.ok(data.timeRange);
    assert.ok(Array.isArray(data.providerDistribution));
    assert.ok(Array.isArray(data.costByIntent));
    assert.ok(Array.isArray(data.latencyTrends));
    assert.ok(Array.isArray(data.modelMarketShare));
    assert.ok(Array.isArray(data.modelSuccessRates));
    assert.ok(Array.isArray(data.outcomeDistribution));

    // All arrays should be empty
    assert.equal((data.providerDistribution as any[]).length, 0);
    assert.equal((data.modelMarketShare as any[]).length, 0);
  });

  it("should respect different time ranges", () => {
    // Insert data from 2 hours ago and 30 minutes ago
    db.recordDecision({
      timestamp: isoTime(120), sessionKey: "old", messageHash: "old",
      intent: "coding", confidence: 0.9, provider: "zai", model: "old-model",
      scores: {}, overallScore: 0.8, outcome: "success", requestId: "old_req",
    });
    db.recordDecision({
      timestamp: isoTime(10), sessionKey: "new", messageHash: "new",
      intent: "coding", confidence: 0.9, provider: "zai", model: "new-model",
      scores: {}, overallScore: 0.8, outcome: "success", requestId: "new_req",
    });

    // 1h range should only include the recent one
    const data1h = db.getDashboardData("1h");
    const dist1h = data1h.providerDistribution as any[];
    assert.equal(dist1h.length, 1);
    assert.equal(dist1h[0].requests, 1);

    // 7d range should include both
    const data7d = db.getDashboardData("7d");
    const dist7d = data7d.providerDistribution as any[];
    assert.ok(dist7d.length >= 1);
    assert.ok(dist7d[0].requests >= 2);
  });
});
