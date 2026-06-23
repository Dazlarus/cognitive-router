// tests/db-integration.test.ts — Integration tests for db_service.ts
// Run with: npx tsx --test tests/db-integration.test.ts
//
// Uses a REAL temporary SQLite database — no mocks.

import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, rmSync } from "node:fs";
import { resolve } from "node:path";
import Database from "better-sqlite3";
import { DBService } from "../src/db_service.ts";

// ─── Helpers ────────────────────────────────────────────────

let dbPath: string;
let db: DBService;
let counter = 0;

function makeTempDbPath(): string {
  counter++;
  const name = `data/test-${Date.now()}-${counter}-${Math.random().toString(36).slice(2)}.db`;
  return resolve(name);
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
  // Clean up WAL/SHM files too
  for (const suffix of ["-wal", "-shm"]) {
    try { rmSync(dbPath + suffix, { force: true }); } catch { /* ignore */ }
  }
});

// ─── Tests ──────────────────────────────────────────────────

describe("DBService — schema initialization", () => {
  it("should create all required tables", async () => {
    await db.initializeSchema();

    // Open a separate read-only connection to inspect schema
    const raw = new Database(dbPath, { readonly: true, fileMustExist: true });
    const tables = raw.prepare(`SELECT name FROM sqlite_master WHERE type='table' ORDER BY name`).all() as Array<{ name: string }>;
    raw.close();
    const tableNames = tables.map((t) => t.name);

    assert.ok(tableNames.includes("routing_decisions"), "routing_decisions table should exist");
    assert.ok(tableNames.includes("retry_attempts"), "retry_attempts table should exist");
    assert.ok(tableNames.includes("call_outcomes"), "call_outcomes table should exist");
    assert.ok(tableNames.includes("model_stats"), "model_stats table should exist");
    assert.ok(tableNames.includes("provider_health"), "provider_health table should exist");
    assert.ok(tableNames.includes("benchmark_cache"), "benchmark_cache table should exist");
    assert.ok(tableNames.includes("capability_overrides"), "capability_overrides table should exist");
    assert.ok(tableNames.includes("judge_history"), "judge_history table should exist");
  });

  it("should be idempotent — calling initializeSchema twice does not error", async () => {
    await db.initializeSchema();
    // Second call should not throw
    await db.initializeSchema();

    // Verify tables still exist and are usable via a separate connection
    const raw = new Database(dbPath, { readonly: true, fileMustExist: true });
    const tables = raw.prepare(`SELECT name FROM sqlite_master WHERE type='table'`).all() as Array<{ name: string }>;
    raw.close();
    assert.ok(tables.length >= 8, "Should still have all tables");
  });
});

describe("DBService — routing decisions", () => {
  beforeEach(async () => {
    await db.initializeSchema();
  });

  it("should record a decision and retrieve it by requestId", () => {
    const timestamp = new Date().toISOString();
    db.recordDecision({
      timestamp,
      sessionKey: "agent:main:test",
      messageHash: "abc123",
      intent: "coding",
      confidence: 0.95,
      provider: "zai",
      model: "glm-5.1",
      scores: { capability: 0.9, reliability: 1.0, cost: 0.8, latency: 0.7 },
      overallScore: 0.88,
      outcome: "PENDING",
      requestId: "req_test_001",
    });

    const result = db.getDecisionByRequestId("req_test_001");

    assert.ok(result, "Should find the decision");
    assert.equal(result!.sessionKey, "agent:main:test");
    assert.equal(result!.intent, "coding");
    assert.equal(result!.confidence, 0.95);
    assert.equal(result!.provider, "zai");
    assert.equal(result!.model, "glm-5.1");
    assert.equal(result!.outcome, "PENDING");
    assert.equal(result!.requestId, "req_test_001");
    assert.deepEqual(result!.scores, { capability: 0.9, reliability: 1.0, cost: 0.8, latency: 0.7 });
  });

  it("should return null for non-existent requestId", () => {
    const result = db.getDecisionByRequestId("does_not_exist");
    assert.equal(result, null);
  });

  it("should retrieve the most recent decision when multiple share the same requestId", () => {
    const ts1 = new Date("2024-01-01T10:00:00Z").toISOString();
    const ts2 = new Date("2024-01-01T11:00:00Z").toISOString();

    db.recordDecision({
      timestamp: ts1,
      sessionKey: "session-A",
      messageHash: "h1",
      intent: "coding",
      confidence: 0.9,
      provider: "zai",
      model: "glm-5.1",
      scores: {},
      overallScore: 0.85,
      outcome: "PENDING",
      requestId: "req_dup",
    });

    db.recordDecision({
      timestamp: ts2,
      sessionKey: "session-B",
      messageHash: "h2",
      intent: "conversation",
      confidence: 0.8,
      provider: "openrouter",
      model: "free-model",
      scores: {},
      overallScore: 0.75,
      outcome: "PENDING",
      requestId: "req_dup",
    });

    const result = db.getDecisionByRequestId("req_dup");
    assert.ok(result);
    // Should return the most recent (ORDER BY timestamp DESC)
    assert.equal(result!.sessionKey, "session-B");
  });
});

describe("DBService — getRecentDecisions", () => {
  beforeEach(async () => {
    await db.initializeSchema();
  });

  it("should return decisions ordered by timestamp DESC (most recent first)", () => {
    for (let i = 0; i < 5; i++) {
      db.recordDecision({
        timestamp: new Date(2024, 0, 1, 12, i, 0).toISOString(),
        sessionKey: `session-${i}`,
        messageHash: `hash-${i}`,
        intent: "conversation",
        confidence: 0.8,
        provider: "zai",
        model: "glm-5.1",
        scores: {},
        overallScore: 0.8,
        outcome: "PENDING",
        requestId: `req_${i}`,
      });
    }

    const recent = db.getRecentDecisions(3);
    assert.equal(recent.length, 3, "Should return requested number of decisions");

    // Should be ordered by timestamp DESC
    const timestamps = recent.map((r: any) => r.timestamp);
    assert.ok(timestamps[0] >= timestamps[1], "First should be more recent than second");
    assert.ok(timestamps[1] >= timestamps[2], "Second should be more recent than third");

    // The most recent should be session-4
    assert.equal((recent[0] as any).session_key, "session-4");
  });

  it("should return empty array when no decisions exist", () => {
    const recent = db.getRecentDecisions(10);
    assert.equal(recent.length, 0);
  });
});

describe("DBService — call outcomes and model stats", () => {
  beforeEach(async () => {
    await db.initializeSchema();
  });

  it("should record call outcomes and update model_stats", () => {
    db.recordCallOutcome({
      provider: "zai",
      model: "glm-5.1",
      durationMs: 1500,
      outcome: "success",
      timestamp: new Date().toISOString(),
    });

    db.recordCallOutcome({
      provider: "zai",
      model: "glm-5.1",
      durationMs: 2000,
      outcome: "success",
      timestamp: new Date().toISOString(),
    });

    db.recordCallOutcome({
      provider: "zai",
      model: "glm-5.1",
      durationMs: 500,
      outcome: "error",
      timestamp: new Date().toISOString(),
    });

    const stats = db.getModelStats();
    assert.ok(stats.length >= 1, "Should have at least one model stat");

    const zaiStat = stats.find((s: any) => s.provider_model === "zai/glm-5.1");
    assert.ok(zaiStat, "Should have stats for zai/glm-5.1");
    assert.equal((zaiStat as any).total_calls, 3, "Should have 3 total calls");
    assert.equal((zaiStat as any).success_count, 2, "Should have 2 successes");
    assert.equal((zaiStat as any).failure_count, 1, "Should have 1 failure");

    // avg_latency_ms = total_latency_ms / total_calls = (1500 + 2000 + 500) / 3 = 1333.33
    const avgLatency = (zaiStat as any).avg_latency_ms;
    assert.ok(Math.abs(avgLatency - 1333.33) < 1, `Expected avg_latency ~1333.33, got ${avgLatency}`);

    // failure_rate = failure_count / total_calls = 1/3 ≈ 0.333
    const failureRate = (zaiStat as any).failure_rate;
    assert.ok(Math.abs(failureRate - 0.333) < 0.01, `Expected failure_rate ~0.333, got ${failureRate}`);
  });

  it("should track model stats independently for different models", () => {
    db.recordCallOutcome({
      provider: "zai", model: "glm-5.1", durationMs: 1000,
      outcome: "success", timestamp: new Date().toISOString(),
    });
    db.recordCallOutcome({
      provider: "openrouter", model: "free-model", durationMs: 2000,
      outcome: "error", timestamp: new Date().toISOString(),
    });

    const stats = db.getModelStats();
    const zai = stats.find((s: any) => s.provider_model === "zai/glm-5.1");
    const or = stats.find((s: any) => s.provider_model === "openrouter/free-model");

    assert.ok(zai);
    assert.ok(or);
    assert.equal((zai as any).total_calls, 1);
    assert.equal((zai as any).success_count, 1);
    assert.equal((or as any).total_calls, 1);
    assert.equal((or as any).failure_count, 1);
  });
});

describe("DBService — retry tracking", () => {
  beforeEach(async () => {
    await db.initializeSchema();
  });

  it("should record retries and count them by requestId", () => {
    const requestId = "req_retry_001";

    db.recordRetry({
      requestId,
      failedProvider: "zai",
      failedOutcome: "rate_limit",
      retryProvider: "openrouter",
      timestamp: new Date().toISOString(),
    });

    db.recordRetry({
      requestId,
      failedProvider: "openrouter",
      failedOutcome: "error",
      retryProvider: "gemini",
      timestamp: new Date().toISOString(),
    });

    const count = db.getRetryCount(requestId);
    assert.equal(count, 2, "Should have 2 retries recorded");
  });

  it("should return 0 for requestId with no retries", () => {
    const count = db.getRetryCount("never_retried");
    assert.equal(count, 0);
  });
});

describe("DBService — capability overrides", () => {
  beforeEach(async () => {
    await db.initializeSchema();
  });

  it("should upsert and get a capability override", () => {
    db.upsertCapabilityOverride("zai", "glm-5.1", "coding", 0.92, 10);

    const result = db.getCapabilityOverride("zai", "glm-5.1", "coding");
    assert.ok(result);
    assert.equal(result!.score, 0.92);
    assert.equal(result!.sampleCount, 10);
  });

  it("should update existing override on upsert (not duplicate)", () => {
    db.upsertCapabilityOverride("zai", "glm-5.1", "coding", 0.85, 5);
    db.upsertCapabilityOverride("zai", "glm-5.1", "coding", 0.92, 15);

    const result = db.getCapabilityOverride("zai", "glm-5.1", "coding");
    assert.ok(result);
    assert.equal(result!.score, 0.92, "Score should be updated to latest value");
    assert.equal(result!.sampleCount, 15, "Sample count should be updated");
  });

  it("should return null/undefined for non-existent override", () => {
    const result = db.getCapabilityOverride("zai", "nonexistent", "coding");
    assert.ok(!result, "Non-existent override should be falsy (null or undefined)");
  });

  it("should load all capability overrides with loadCapabilityOverrides", () => {
    db.upsertCapabilityOverride("zai", "glm-5.1", "coding", 0.92, 10);
    db.upsertCapabilityOverride("zai", "glm-5.1", "conversation", 0.88, 8);
    db.upsertCapabilityOverride("openrouter", "free-model", "coding", 0.70, 5);

    const overrides = db.loadCapabilityOverrides();
    assert.equal(overrides.length, 3, "Should load all 3 overrides");

    // Verify structure
    for (const o of overrides) {
      assert.ok(typeof o.provider === "string");
      assert.ok(typeof o.model === "string");
      assert.ok(typeof o.intent === "string");
      assert.ok(typeof o.score === "number");
      assert.ok(typeof o.sampleCount === "number");
    }
  });

  it("should handle different intents for same provider/model independently", () => {
    db.upsertCapabilityOverride("zai", "glm-5.1", "coding", 0.92, 10);
    db.upsertCapabilityOverride("zai", "glm-5.1", "creative", 0.75, 5);

    const coding = db.getCapabilityOverride("zai", "glm-5.1", "coding");
    const creative = db.getCapabilityOverride("zai", "glm-5.1", "creative");

    assert.equal(coding!.score, 0.92);
    assert.equal(creative!.score, 0.75);
  });
});

describe("DBService — judge evaluations", () => {
  beforeEach(async () => {
    await db.initializeSchema();
  });

  it("should record a judge evaluation without error", () => {
    assert.doesNotThrow(() => {
      db.recordJudgeEvaluation(
        "zai",
        "glm-5.1",
        "coding",
        0.88,
        "Good code quality with minor issues",
        "gemini-2.5-flash",
      );
    });
  });

  it("should record multiple judge evaluations", () => {
    for (let i = 0; i < 3; i++) {
      db.recordJudgeEvaluation(
        "zai",
        "glm-5.1",
        "coding",
        0.8 + i * 0.05,
        `Evaluation ${i}`,
        "gemini-2.5-flash",
      );
    }

    // Verify by querying directly via a separate connection
    const raw = new Database(dbPath, { readonly: true, fileMustExist: true });
    const rows = raw.prepare(`SELECT * FROM judge_history WHERE provider = ? AND model = ?`)
      .all("zai", "glm-5.1") as any[];
    raw.close();
    assert.equal(rows.length, 3, "Should have 3 judge evaluations");
  });
});

describe("DBService — spend by provider", () => {
  beforeEach(async () => {
    await db.initializeSchema();
  });

  it("should aggregate decisions by provider", () => {
    const providers = ["zai", "zai", "openrouter", "gemini"];

    for (let i = 0; i < providers.length; i++) {
      db.recordDecision({
        timestamp: new Date(2024, 0, 1, 12, i, 0).toISOString(),
        sessionKey: `session-${i}`,
        messageHash: `hash-${i}`,
        intent: "conversation",
        confidence: 0.8,
        provider: providers[i],
        model: "some-model",
        scores: {},
        overallScore: 0.8,
        outcome: i % 2 === 0 ? "SUCCESS" : "PENDING",
        requestId: `req_${i}`,
      });
    }

    const spend = db.getSpendByProvider();
    assert.ok(spend.length >= 3, "Should have at least 3 providers");

    const zaiSpend = spend.find((s: any) => s.provider === "zai");
    assert.ok(zaiSpend);
    assert.equal((zaiSpend as any).decisions, 2, "ZAI should have 2 decisions");

    const orSpend = spend.find((s: any) => s.provider === "openrouter");
    assert.ok(orSpend);
    assert.equal((orSpend as any).decisions, 1);
  });
});

describe("DBService — request_id column migration", () => {
  it("should have request_id column after addColumnIfMissing runs", async () => {
    await db.initializeSchema();

    const raw = new Database(dbPath, { readonly: true, fileMustExist: true });
    const columns = raw.prepare(`PRAGMA table_info(routing_decisions)`).all() as Array<{ name: string }>;
    raw.close();
    const columnNames = columns.map((c) => c.name);

    assert.ok(columnNames.includes("request_id"), "request_id column should exist after schema init");
  });
});
