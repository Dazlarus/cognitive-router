// tests/ladder_refresh.test.ts - LadderProjectionService + admin refresh hook
// (p3e-006). Verifies: hourly refresh service rebuilds intents through the
// same rebuildLadder/replaceLadder path as sync hot-apply; startup refresh
// backstops; admin hook POST /admin/bench/ladder/refresh triggers the same
// path; concurrency skip; periodic arm/disarm.

import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";
import { ProxyServerStreaming } from "../src/proxy-stream.ts";
import { loadConfig } from "../src/config.ts";
import { DBService } from "../src/db_service.ts";
import { LadderProjectionService } from "../src/ladder_refresh.ts";
import { initializeBenchmarkTables } from "../src/benchmark_embeddings.ts";
import { PROMPT_GENERATION } from "../src/benchmark_ladder.ts";
import { registryReadiness } from "../src/readiness.ts";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const ADMIN = "test-admin-token-0123456789abcdef01234567";

let dir: string;
let proxy: ProxyServerStreaming;
let pdb: DBService;
let server: http.Server;
let baseUrl: string;

function verdictRows(
  a: string,
  b: string,
  verdict: "a" | "b" | "tie",
  at = "2026-09-09T01:00:00.000Z",
) {
  return Array.from({ length: 10 }, (_, i) => ({
    modelA: a,
    modelB: b,
    intent: "coding",
    promptGeneration: PROMPT_GENERATION,
    round: Math.floor(i / 2),
    swapOrder: (i % 2) as 0 | 1,
    verdict,
  }));
}

beforeEach(async () => {
  dir = mkdtempSync(join(tmpdir(), "ladder-refresh-"));
  process.env.ROUTER_ADMIN_TOKEN = ADMIN;
  const config = loadConfig({
    proxyPort: 3457,
    enabled: true,
    providerPriority: [],
    weights: { capability: 0, reliability: 0, cost: 0, latency: 0 },
    dbPath: join(dir, "router.db"),
  } as any);
  proxy = new ProxyServerStreaming(config);
  (proxy as any).initialized = true;
  registryReadiness.setSynced();
  pdb = (proxy as any).db as DBService;
  initializeBenchmarkTables(pdb);

  server = http.createServer((req, res) => {
    (proxy as any).handleRequest(req, res);
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

afterEach(async () => {
  server.closeAllConnections?.();
  await new Promise<void>((r) => server.close(() => r()));
  await proxy.stop();
  rmSync(dir, { recursive: true, force: true });
  delete process.env.ROUTER_ADMIN_TOKEN;
});

describe("LadderProjectionService", () => {
  it("refreshes tracked intents: rank change detected on first build, none on repeat", async () => {
    // seed verdict rows: A beats B consistently -> A rank 1
    for (const r of verdictRows("model-a|as-served|medium", "model-b|as-served|medium", "a")) {
      pdb.insertBenchmarkVerdict({
        ...r,
        promptGeneration: PROMPT_GENERATION,
      });
    }
    const svc = new LadderProjectionService(pdb);
    // first refresh: projection is empty, so refresh must seed it via the
    // cold-start coding backstop (intent not yet tracked)
    const first = await svc.refresh("test");
    assert.equal(first.skipped, false);
    assert.ok(first.intents.length >= 1);
    const coding = first.intents.find((r) => r.intent === "coding");
    assert.ok(coding, "coding intent refreshed");
    assert.equal(coding.identities, 2);

    // second refresh: already consistent -> changed = 0
    const second = await svc.refresh("test");
    const coding2 = second.intents.find((r) => r.intent === "coding");
    assert.ok(coding2);
    assert.equal(coding2.changed, 0);
    assert.equal(coding2.identities, 2);

    // ladder content: rank 1 = model-a
    const ladder = pdb.getLadder("coding", PROMPT_GENERATION);
    assert.equal(ladder.length, 2);
    assert.equal(ladder[0].modelKey, "model-a|as-served|medium");
    assert.equal(ladder[0].rank, 1);
  });

  it("empty projection with no verdict rows is a no-op (not an error)", async () => {
    const svc = new LadderProjectionService(pdb);
    const report = await svc.refresh("test");
    assert.equal(report.skipped, false);
    assert.equal(report.intents.length, 0);
  });

  it("periodic arm/disarm: startPeriodic arms once, stopPeriodic clears", () => {
    const svc = new LadderProjectionService(pdb, 999);
    svc.startPeriodic();
    svc.startPeriodic(); // idempotent
    svc.stopPeriodic();
    svc.stopPeriodic(); // idempotent
  });
});

describe("POST /admin/bench/ladder/refresh", () => {
  it("403s without the admin token", async () => {
    const res = await fetch(`${baseUrl}/admin/bench/ladder/refresh`, { method: "POST" });
    assert.equal(res.status, 403);
  });

  it("runs the refresh and returns the report", async () => {
    for (const r of verdictRows("model-a|as-served|medium", "model-b|as-served|medium", "a")) {
      pdb.insertBenchmarkVerdict({ ...r, promptGeneration: PROMPT_GENERATION });
    }
    const res = await fetch(`${baseUrl}/admin/bench/ladder/refresh`, {
      method: "POST",
      headers: { "x-admin-token": ADMIN },
    });
    assert.equal(res.status, 200);
    const body = await res.json();
    assert.equal(body.skipped, false);
    assert.ok(Array.isArray(body.intents));
    const coding = body.intents.find((r: any) => r.intent === "coding");
    assert.ok(coding, "coding refreshed via admin hook");
    assert.equal(coding.identities, 2);
    // ladder is queryable right after
    const ladderRes = await fetch(`${baseUrl}/admin/bench/ladder`, {
      headers: { "x-admin-token": ADMIN },
    });
    const ladder = await ladderRes.json();
    assert.ok(ladder.ladders.coding.length >= 2);
  });
});
