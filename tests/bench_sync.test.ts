// tests/bench_sync.test.ts - POST /admin/bench/sync endpoint behavior
// (CONTRACT.md §4): auth, schema validation, row caps, single-transaction
// upsert, watermark staleness. Uses a REAL http server on an ephemeral port
// wrapping the proxy's handler — mock ServerResponse never emits 'close'
// without a socket (found the hard way), so real sockets it is.

import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";
import { ProxyServerStreaming } from "../src/proxy-stream.ts";
import { loadConfig } from "../src/config.ts";
import { DBService } from "../src/db_service.ts";
import { registryReadiness } from "../src/readiness.ts";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const TOKEN = "test-sync-token-0123456789abcdef0123456789abcdef";

let dir: string;
let proxy: ProxyServerStreaming;
let pdb: DBService; // the PROXY's DBService — the one the handler writes to
let server: http.Server;
let baseUrl: string;

beforeEach(async () => {
  dir = mkdtempSync(join(tmpdir(), "bench-sync-"));
  process.env.BENCH_SYNC_TOKEN = TOKEN;
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

  server = http.createServer((req, res) => {
    (proxy as any).handleRequest(req, res);
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

afterEach(async () => {
  server.closeAllConnections?.();
  await new Promise<void>((r) => server.close(() => r()));
  await proxy.stop(); // curator/discovery timers keep the process alive otherwise
  rmSync(dir, { recursive: true, force: true });
  delete process.env.BENCH_SYNC_TOKEN;
});

async function post(body: unknown, headers: Record<string, string> = {}) {
  const payload = JSON.stringify(body);
  const res = await fetch(`${baseUrl}/admin/bench/sync`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "content-length": String(Buffer.byteLength(payload)),
      ...headers,
    },
    body: payload,
  });
  return { status: res.status, body: await res.json().catch(() => ({})) };
}

function validRow(overrides: Record<string, unknown> = {}) {
  return {
    model_a: "glm-5.3|as-served|medium",
    model_b: "glm-5-turbo|as-served|medium",
    intent: "coding",
    prompt_generation: "gen-2026-09",
    round: 0,
    swap_order: 0,
    verdict: "a",
    judge_provider: "zai",
    judge_model: "glm-5.2",
    timestamp: "2026-09-07T20:00:00.000Z",
    generated_at: "2026-09-07T20:00:00.000Z",
    ...overrides,
  };
}

describe("POST /admin/bench/sync", () => {
  it("rejects missing/invalid token with 401", async () => {
    const noToken = await post({ schema_version: 1, verdicts: [validRow()] });
    assert.equal(noToken.status, 401);
    const badToken = await post(
      { schema_version: 1, verdicts: [validRow()] },
      { authorization: "Bearer wrong-token" },
    );
    assert.equal(badToken.status, 401);
  });

  it("applies a valid payload and advances the watermark", async () => {
    const r = await post(
      { schema_version: 1, generated_at: "2026-09-07T20:00:00.000Z", verdicts: [validRow()] },
      { authorization: `Bearer ${TOKEN}` },
    );
    assert.equal(r.status, 200);
    assert.equal(r.body.applied, 1);
    assert.equal(pdb.getBenchSyncWatermark(), "2026-09-07T20:00:00.000Z");
    assert.equal(pdb.getAllBenchmarkVerdicts("coding", "gen-2026-09").length, 1);
  });

  it("rejects invalid rows wholesale (400 invalid_row), nothing applied", async () => {
    const r = await post(
      { schema_version: 1, verdicts: [validRow(), validRow({ verdict: "bogus" })] },
      { authorization: `Bearer ${TOKEN}` },
    );
    assert.equal(r.status, 400);
    assert.equal(r.body.error.code, "invalid_row");
    assert.equal(pdb.getAllBenchmarkVerdicts("coding", "gen-2026-09").length, 0);
  });

  it("rejects >500 rows (400 bad_row_count)", async () => {
    const rows = Array.from({ length: 501 }, () => validRow());
    const r = await post(
      { schema_version: 1, verdicts: rows },
      { authorization: `Bearer ${TOKEN}` },
    );
    assert.equal(r.status, 400);
    assert.equal(r.body.error.code, "bad_row_count");
  });

  it("rejects unknown schema_version (400 unsupported_schema)", async () => {
    const r = await post(
      { schema_version: 2, verdicts: [validRow()] },
      { authorization: `Bearer ${TOKEN}` },
    );
    assert.equal(r.status, 400);
    assert.equal(r.body.error.code, "unsupported_schema");
  });

  it("rejects stale payloads (409); equal timestamp = idempotent no-op", async () => {
    const at = (t: string) => validRow({ generated_at: t, timestamp: t });
    const auth = { authorization: `Bearer ${TOKEN}` };
    const r1 = await post({ schema_version: 1, verdicts: [at("2026-09-07T20:00:00.000Z")] }, auth);
    assert.equal(r1.status, 200);
    const r2 = await post({ schema_version: 1, verdicts: [at("2026-09-07T19:00:00.000Z")] }, auth);
    assert.equal(r2.status, 409);
    assert.equal(r2.body.error.code, "stale_payload");
    const r3 = await post({ schema_version: 1, verdicts: [at("2026-09-07T20:00:00.000Z")] }, auth);
    assert.equal(r3.status, 200);
  });

  it("returns 503 sync_disabled when token not configured", async () => {
    delete process.env.BENCH_SYNC_TOKEN;
    const r = await post({ schema_version: 1, verdicts: [validRow()] });
    assert.equal(r.status, 503);
    assert.equal(r.body.error.code, "sync_disabled");
  });
});
