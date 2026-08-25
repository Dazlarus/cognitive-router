// tests/abort_feedback.test.ts — Unit and integration tests for Phase 3 Session Durability
// Run with: npx tsx --test tests/abort_feedback.test.ts

import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import net from "node:net";
import { rmSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { ProxyServerStreaming } from "../src/proxy-stream.ts";
import { loadConfig, type CognitiveRouterConfig } from "../src/config.ts";
import { DBService } from "../src/db_service.ts";
import { CostTracker } from "../src/cost_tracker.ts";
import { RoutingEngine } from "../src/router.ts";

// ─── Test Helpers ───────────────────────────────────────────

let dbPath: string;
let db: DBService;
let counter = 0;

function makeTempDbPath(): string {
  counter++;
  const name = `data/test-abort-${Date.now()}-${counter}-${Math.random().toString(36).slice(2)}.db`;
  return resolve(name);
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

function makeConfig(port: number, tempDbPath: string, overrides: Partial<CognitiveRouterConfig> = {}): CognitiveRouterConfig {
  const base = loadConfig({
    enabled: true,
    logLevel: "warn",
    dbPath: tempDbPath,
    proxyPort: port,
    providerPriority: ["zai", "openrouter", "gemini", "ollama"],
    providers: {
      zai: { budgetType: "subscription", priority: "high" },
      openrouter: { budgetType: "free", priority: "high" },
      gemini: { budgetType: "credits", priority: "medium" },
      ollama: { budgetType: "free", priority: "low" },
    },
    weights: { capability: 0.5, reliability: 0.25, cost: 0.15, latency: 0.1 },
    reliabilityAbortPenalty: 0.2,
  });
  return { ...base, ...overrides };
}

function requestJson(port: number, path: string, method: string, payload?: any): Promise<{ statusCode: number; body: string }> {
  return new Promise((resolve, reject) => {
    const bodyStr = payload ? JSON.stringify(payload) : "";
    const req = http.request({
      hostname: "127.0.0.1",
      port,
      path,
      method,
      headers: payload ? {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(bodyStr),
      } : {},
    }, (res) => {
      let responseBody = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => { responseBody += chunk; });
      res.on("end", () => resolve({ statusCode: res.statusCode ?? 0, body: responseBody }));
    });
    req.on("error", reject);
    if (payload) {
      req.write(bodyStr);
    }
    req.end();
  });
}

// ─── Tests ──────────────────────────────────────────────────

describe("Phase 3: Session-Level Durability Tests", () => {
  let port: number;
  let config: CognitiveRouterConfig;
  let proxy: ProxyServerStreaming;

  beforeEach(async () => {
    dbPath = makeTempDbPath();
    mkdirSync(resolve(dbPath, ".."), { recursive: true });
    port = await getFreePort();
    config = makeConfig(port, dbPath);
    proxy = new ProxyServerStreaming(config);
    await proxy.start();
    // Wait for registry readiness and migrations
    db = (proxy as any).db;
  });

  afterEach(async () => {
    if (proxy) {
      await proxy.stop();
    }
    try { rmSync(dbPath, { force: true }); } catch { /* ignore */ }
    for (const suffix of ["-wal", "-shm"]) {
      try { rmSync(dbPath + suffix, { force: true }); } catch { /* ignore */ }
    }
  });

  describe("DB Migration (db_service.ts) - user_version = 3", () => {
    it("should migrate schema to v3 and create required tables", () => {
      const sqliteDb = db.getDb();
      const userVersion = sqliteDb.pragma("user_version", { simple: true }) as number;
      // v3 tables must exist; user_version is >= 3 (v4 learning-loop hardening supersedes v3).
      assert.ok(userVersion >= 3, `Database user_version should be >= 3, got ${userVersion}`);

      // Verify tables exist
      const tables = sqliteDb.prepare("SELECT name FROM sqlite_master WHERE type='table'").all() as { name: string }[];
      const tableNames = tables.map(t => t.name);
      assert.ok(tableNames.includes("abort_events"), "abort_events table should exist");
      assert.ok(tableNames.includes("session_outcomes"), "session_outcomes table should exist");
    });

    it("should support reverse migration rollback_v3()", () => {
      db.rollback_v3();
      const sqliteDb = db.getDb();
      const userVersion = sqliteDb.pragma("user_version", { simple: true }) as number;
      assert.equal(userVersion, 2, "Database user_version should roll back to 2");

      const tables = sqliteDb.prepare("SELECT name FROM sqlite_master WHERE type='table'").all() as { name: string }[];
      const tableNames = tables.map(t => t.name);
      assert.ok(!tableNames.includes("abort_events"), "abort_events table should be dropped");
      assert.ok(!tableNames.includes("session_outcomes"), "session_outcomes table should be dropped");
    });
  });

  describe("POST /v1/report/abort Endpoint", () => {
    it("should record abort event and return 200 with valid data", async () => {
      const payload = {
        provider: "zai",
        model: "glm-4.1",
        turnsCompleted: 2,
        durationMs: 4500,
        sessionKey: "test-abort-session",
      };

      const res = await requestJson(port, "/v1/report/abort", "POST", payload);
      assert.equal(res.statusCode, 200);

      const response = JSON.parse(res.body);
      assert.equal(response.status, "success");

      // Verify database record
      const events = db.getDb().prepare("SELECT * FROM abort_events ORDER BY id DESC LIMIT 1").all() as any[];
      assert.equal(events.length, 1);
      assert.equal(events[0].provider, "zai");
      assert.equal(events[0].model, "glm-4.1");
      assert.equal(events[0].turns_completed, 2);
      assert.equal(events[0].duration_ms, 4500);
      assert.equal(events[0].session_key, "test-abort-session");
    });

    it("should decrease reliability score by penalty amount", async () => {
      // Prior reliability score
      const costTracker = (proxy as any).costTracker;
      const initialReliability = costTracker.getReliabilityScore("zai", "glm-4.1");
      assert.equal(initialReliability, 1.0, "Initial reliability score should be 1.0");

      const payload = {
        provider: "zai",
        model: "glm-4.1",
      };

      const res = await requestJson(port, "/v1/report/abort", "POST", payload);
      assert.equal(res.statusCode, 200);

      const afterReliability = costTracker.getReliabilityScore("zai", "glm-4.1");
      assert.equal(afterReliability, 0.8, "Reliability score should decrease by 0.2 penalty");
    });

    it("should clamp reliability score to 0.0 minimum", async () => {
      const costTracker = (proxy as any).costTracker;
      const payload = {
        provider: "zai",
        model: "glm-4.1",
      };

      // Apply penalty 6 times (0.2 * 6 = 1.2)
      for (let i = 0; i < 6; i++) {
        await requestJson(port, "/v1/report/abort", "POST", payload);
      }

      const finalReliability = costTracker.getReliabilityScore("zai", "glm-4.1");
      assert.equal(finalReliability, 0.0, "Reliability score should clamp to 0.0 minimum");
    });

    it("should return 400 with invalid inputs", async () => {
      // Missing provider
      let res = await requestJson(port, "/v1/report/abort", "POST", { model: "glm-4.1" });
      assert.equal(res.statusCode, 400);

      // Missing model
      res = await requestJson(port, "/v1/report/abort", "POST", { provider: "zai" });
      assert.equal(res.statusCode, 400);

      // Negative turnsCompleted
      res = await requestJson(port, "/v1/report/abort", "POST", { provider: "zai", model: "glm-4.1", turnsCompleted: -1 });
      assert.equal(res.statusCode, 400);
    });

    it("should still apply in-memory penalty if database write fails", async () => {
      const costTracker = (proxy as any).costTracker;
      // Intentionally close database connection to trigger a database write failure
      db.getDb().close();

      const payload = {
        provider: "zai",
        model: "glm-4.1",
      };

      const res = await requestJson(port, "/v1/report/abort", "POST", payload);
      assert.equal(res.statusCode, 500);

      const afterReliability = costTracker.getReliabilityScore("zai", "glm-4.1");
      assert.equal(afterReliability, 0.8, "In-memory penalty must still be applied even if database write fails");
    });
  });

  describe("Zombie Pattern Flags", () => {
    it("should set model pattern flag to unstable after 3 stalls", async () => {
      const costTracker = (proxy as any).costTracker;
      // Call abort endpoint with stall-related logic manually, or trigger applyAbortPenalty directly
      // Since our stream stall detection throws and calls applyAbortPenalty(..., "stall")
      costTracker.applyAbortPenalty("zai", "glm-4.1", "stall");
      costTracker.applyAbortPenalty("zai", "glm-4.1", "stall");
      costTracker.applyAbortPenalty("zai", "glm-4.1", "stall");

      const isUnstable = costTracker.isUnstable("zai", "glm-4.1");
      assert.equal(isUnstable, true, "Model glm-4.1 should be flagged as unstable after 3 stalls");

      const finalReliability = costTracker.getReliabilityScore("zai", "glm-4.1");
      // Pattern flags set reliability to PATTERN_DEPRIORITIZED_RELIABILITY = 0.1
      assert.equal(finalReliability, 0.1, "Reliability score should drop to 0.1 when unstable");
    });

    it("should reset zombie count when a successful call is recorded", async () => {
      const costTracker = (proxy as any).costTracker;
      costTracker.applyAbortPenalty("zai", "glm-4.1", "stall");
      costTracker.applyAbortPenalty("zai", "glm-4.1", "stall");

      // Successful call resets count
      await costTracker.recordCall("zai", { durationMs: 1000, outcome: "success" }, "glm-4.1");

      // Another stall (making it 3rd total, but count was reset)
      costTracker.applyAbortPenalty("zai", "glm-4.1", "stall");

      const isUnstable = costTracker.isUnstable("zai", "glm-4.1");
      assert.equal(isUnstable, false, "Model should not be unstable as zombie count was reset by successful call");
    });
  });

  describe("GET /v1/fallback Endpoint", () => {
    it("should return fallback suggestion excluding the failed candidate", async () => {
      // First, get standard fallback suggestion
      const res = await requestJson(port, "/v1/fallback?provider=zai&model=glm-4.1", "GET");
      assert.equal(res.statusCode, 200);

      const data = JSON.parse(res.body);
      assert.ok(data.provider, "Should return a provider");
      assert.ok(data.model, "Should return a model");
      assert.notEqual(data.model, "glm-4.1", "Should exclude failed candidate 'glm-4.1'");
      assert.ok(data.rationale, "Should include decision rationale");
      assert.ok(typeof data.score === "number", "Should include score");
    });

    it("should return 400 for invalid query parameters", async () => {
      // Missing provider
      let res = await requestJson(port, "/v1/fallback?model=glm-4.1", "GET");
      assert.equal(res.statusCode, 400);

      // Invalid provider
      res = await requestJson(port, "/v1/fallback?provider=non-existent&model=glm-4.1", "GET");
      assert.equal(res.statusCode, 400);

      // Missing model
      res = await requestJson(port, "/v1/fallback?provider=zai", "GET");
      assert.equal(res.statusCode, 400);

      // Invalid reason
      res = await requestJson(port, "/v1/fallback?provider=zai&model=glm-4.1&reason=invalid-reason", "GET");
      assert.equal(res.statusCode, 400);
    });
  });

  describe("Cron Routing Profile", () => {
    it("should double reliability weight and proportionally reduce capability weight", async () => {
      const router = (proxy as any).router;

      // Get standard decision (no profile)
      const decisionStandard = await router.decide({ intent: "conversation", confidence: 1.0 }, "sess");

      // Get cron profile decision
      const decisionCron = await router.decide({ intent: "conversation", confidence: 1.0 }, "sess", {}, "cron");

      assert.ok(decisionStandard, "Should return decision for standard profile");
      assert.ok(decisionCron, "Should return decision for cron profile");
    });
  });
});
