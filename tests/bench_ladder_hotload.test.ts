// tests/bench_ladder_hotload.test.ts - P1 hot-load: POST /admin/bench/sync
// applies verdict rows to the live ladder projection IN-PROCESS. Sync-then-
// query (GET /admin/bench/ladder) reflects new verdicts with NO restart.
// Restart/curator rebuild remains a reconciliation backstop only.

import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";
import { ProxyServerStreaming } from "../src/proxy-stream.ts";
import { loadConfig } from "../src/config.ts";
import { registryReadiness } from "../src/readiness.ts";
import { PROMPT_GENERATION } from "../src/benchmark_ladder.ts";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const TOKEN = "test-sync-token-0123456789abcdef0123456789abcdef";
const ADMIN = "test-admin-token-0123456789abcdef01234567";

let dir: string;
let proxy: ProxyServerStreaming;
let server: http.Server;
let baseUrl: string;

beforeEach(async () => {
  dir = mkdtempSync(join(tmpdir(), "bench-hotload-"));
  process.env.BENCH_SYNC_TOKEN = TOKEN;
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
  delete process.env.BENCH_SYNC_TOKEN;
  delete process.env.ROUTER_ADMIN_TOKEN;
});

async function postSync(rows: unknown[]) {
  const payload = JSON.stringify({ schema_version: 1, verdicts: rows });
  const res = await fetch(`${baseUrl}/admin/bench/sync`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "content-length": String(Buffer.byteLength(payload)),
      authorization: `Bearer ${TOKEN}`,
    },
    body: payload,
  });
  return { status: res.status, body: await res.json().catch(() => ({})) };
}

async function getLadder() {
  const res = await fetch(`${baseUrl}/admin/bench/ladder`, {
    headers: { "x-admin-token": ADMIN },
  });
  return { status: res.status, body: await res.json().catch(() => ({})) };
}

function rows(
  a: string,
  b: string,
  verdict: "a" | "b" | "tie",
  gen = PROMPT_GENERATION,
  at = "2026-09-09T01:00:00.000Z",
) {
  return Array.from({ length: 10 }, (_, i) => ({
    model_a: a,
    model_b: b,
    intent: "coding",
    prompt_generation: gen,
    round: Math.floor(i / 2),
    swap_order: (i % 2) as 0 | 1,
    verdict,
    judge_provider: "zai",
    judge_model: "glm-5.2",
    timestamp: at,
    generated_at: at,
  }));
}

describe("hot-load: sync applies to the live ladder without restart", () => {
  it("sync-then-query reflects the new verdict, same process", async () => {
    // BEFORE: empty projection
    const before = await getLadder();
    assert.equal(before.status, 200);
    assert.deepEqual(before.body.ladders, {});

    // SYNC: alpha beats beta, 5 rounds x 2 order-swapped asks
    const r1 = await postSync(rows("alpha|as-served|medium", "beta|as-served|medium", "a"));
    assert.equal(r1.status, 200);
    assert.equal(r1.body.applied, 10);
    assert.deepEqual(r1.body.ladder, { applied: ["coding"], skipped: [] });

    // AFTER (same process, no restart): ladder reflects the verdict
    const after = await getLadder();
    assert.equal(after.status, 200);
    assert.equal(after.body.generation, PROMPT_GENERATION);
    const coding = after.body.ladders.coding;
    assert.equal(coding.length, 2);
    assert.equal(coding[0].modelKey, "alpha|as-served|medium");
    assert.equal(coding[0].rank, 1);
    assert.ok(coding[0].strength > coding[1].strength);
    assert.equal(coding[1].modelKey, "beta|as-served|medium");

    // A LATER sync re-shapes the same live projection: gamma beats alpha
    const r2 = await postSync(
      rows("gamma|as-served|medium", "alpha|as-served|medium", "a", PROMPT_GENERATION, "2026-09-09T02:00:00.000Z"),
    );
    assert.equal(r2.status, 200);
    const after2 = await getLadder();
    const coding2 = after2.body.ladders.coding;
    assert.equal(coding2.length, 3);
    assert.equal(coding2[0].modelKey, "gamma|as-served|medium"); // new verdict reordered ranks in-process
    assert.equal(coding2[1].modelKey, "alpha|as-served|medium");
    assert.equal(coding2[2].modelKey, "beta|as-served|medium");
  });

  it("other-generation rows defer to the restart backstop (skipped, not applied)", async () => {
    const r = await postSync(rows("old-a|as-served|medium", "old-b|as-served|medium", "a", "gen-2020-01-01"));
    assert.equal(r.status, 200);
    assert.deepEqual(r.body.ladder, { applied: [], skipped: ["coding"] });
    const ladder = await getLadder();
    assert.deepEqual(ladder.body.ladders, {}); // current-gen projection untouched
  });

  it("admin ladder read is token-gated", async () => {
    const res = await fetch(`${baseUrl}/admin/bench/ladder`);
    assert.equal(res.status, 403);
  });
});
