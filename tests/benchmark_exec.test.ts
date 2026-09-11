// tests/benchmark_exec.test.ts - Tests for src/benchmark_exec.ts
//
// Exec sandbox backend is WIP/blocked (cogrouter-p3e-001); benchmark_exec
// is tested with DI mocks per task guidance: "structure benchmark_exec against
// the documented sandbox interface and cover persistence/interval logic with
// unit tests instead."

import { describe, it, beforeEach, afterEach, after } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, rmSync } from "node:fs";
import { resolve } from "node:path";
import {
  ExecBenchmark,
  wilsonInterval,
  REMOTE_CACHE_TTL_MS,
  type ExecBenchDeps,
  type ExecProbeOutcome,
} from "../src/benchmark_exec.js";
import { DBService } from "../src/db_service.js";
import type { ExecRunResult, FailReason } from "../src/exec_sandbox.js";
import type { CodingProbe } from "../src/probes.js";

// ─── Helpers ─────────────────────────────────────────────────

let _counter = 0;
const _tempPaths: string[] = [];

async function makeTempDb(): Promise<DBService> {
  _counter++;
  const dir = resolve(`data/test-${Date.now()}-${_counter}`);
  mkdirSync(dir, { recursive: true });
  const path = `${dir}/bench.db`;
  _tempPaths.push(dir);
  const db = new DBService(path);
  await db.initializeSchema();
  return db;
}

function closeRemoveDb(db: DBService): void {
  try { db.close(); } catch { /* ok */ }
}

after(() => {
  for (const p of _tempPaths) {
    try { rmSync(p, { force: true, recursive: true }); } catch { /* ok */ }
  }
});

/** A minimal CodingProbe for test injection */
function makeProbe(overrides: Partial<CodingProbe> = {}): CodingProbe {
  return {
    id: `probe-${overrides.id ?? "test"}`,
    generation: "gen-2026-09-08b",
    subIntent: "quick",
    prompt: "Write a Python function is_balanced.",
    language: "python",
    entry: "is_balanced",
    harness: {
      setup: "from solution import is_balanced",
      cases: [
        { call: "is_balanced('()[]{}')", expect: "True" },
        { call: "is_balanced('([)]')", expect: "False" },
      ],
      casesHidden: 2,
    },
    ...overrides,
  };
}

/** Mock sandbox: returns whatever you pre-configure in persistence[] (round-robin). */
function makeMockSandbox(
  persistence: Array<{
    casesPassed?: number;
    casesTotal?: number;
    failReason?: FailReason | null;
    durationMs?: number;
    backend?: "docker" | "win32-job" | null;
  }> = [],
) {
  let idx = 0;
  return {
    run: async (): Promise<ExecRunResult> => {
      const p = persistence[idx] ?? persistence[persistence.length - 1] ?? {};
      idx++;
      const total = p.casesTotal ?? 4;
      const passed = p.casesPassed ?? total;
      return {
        casesPassed: passed,
        casesTotal: total,
        passRate: total > 0 ? passed / total : 0,
        failReason: p.failReason ?? null,
        durationMs: p.durationMs ?? 10,
        backend: p.backend ?? "docker",
      };
    },
    available: async (): Promise<"docker" | "win32-job" | null> => "docker",
  };
}

/** Mock model caller: returns a fenced code block, or throws on demand. */
function makeModelCaller(
  responseText: string = "Here's the code:\n```python\ndef foo():\n    pass\n```",
  _latencyMs: number = 50,
  fail: boolean = false,
) {
  let calls = 0;
  return async (_key: any, _prompt: string): Promise<string> => {
    calls++;
    if (fail) throw new Error(`model call failed (call ${calls})`);
    await new Promise((r) => setTimeout(r, 5));
    return responseText;
  };
}

/** Build a fresh ExecBenchmark with all-DI mocks. */
function buildBench(
  db: DBService,
  overrides: Partial<ExecBenchDeps> = {},
): ExecBenchmark {
  return new ExecBenchmark({
    db,
    callModel: makeModelCaller(),
    sandbox: makeMockSandbox(),
    versionHash: async () => "vhash001",
    gpuIdle: async () => true, // idle — allows local runs
    now: () => 1000000,
    ...overrides,
  });
}

// ─── wilsonInterval ──────────────────────────────────────────

describe("wilsonInterval", () => {
  it("returns null for n <= 0", () => {
    assert.equal(wilsonInterval(0, 0), null);
    assert.equal(wilsonInterval(1, -1), null);
  });

  it("returns 0 interval for 0 successes", () => {
    const iv = wilsonInterval(0, 3);
    assert.ok(iv !== null);
    assert.equal(iv.low, 0);
    assert.ok(iv.high > 0);
  });

  it("returns 1 interval for all successes", () => {
    const iv = wilsonInterval(3, 3);
    assert.ok(iv !== null);
    assert.ok(iv.low > 0);
    assert.equal(iv.high, 1);
  });

  it("wraps bounds to [0,1]", () => {
    const iv = wilsonInterval(1, 10);
    assert.ok(iv !== null);
    assert.ok(iv.low >= 0);
    assert.ok(iv.high <= 1);
    assert.ok(iv.low < iv.high);
  });

  it("interval widens with fewer samples", () => {
    const iv3 = wilsonInterval(1, 3);
    const iv30 = wilsonInterval(10, 30);
    assert.ok(iv3 !== null && iv30 !== null);
    assert.ok((iv3.high - iv3.low) > (iv30.high - iv30.low));
  });
});

// ─── ExecBenchmark ───────────────────────────────────────────

describe("ExecBenchmark", () => {
  let db: DBService;
  let bench: ExecBenchmark;

  beforeEach(async () => {
    db = await makeTempDb();
    bench = buildBench(db);
  });

  afterEach(() => {
    closeRemoveDb(db);
  });

  // ─── summarize ────────────────────────────────────

  describe("summarize", () => {
    it("returns empty summary for no results", () => {
      const s = bench.summarize("ollama", "m|fp|none", "g", "", []);
      assert.equal(s.nProbes, 0);
      assert.equal(s.meanPassAt1, null);
      assert.equal(s.eligible, false);
    });

    it("excludes sandbox_unavailable and modelError from summary", () => {
      const results: ExecProbeOutcome[] = [
        { probeId: "p1", cached: false, passRate: 1, casesPassed: 3, casesTotal: 3, failReason: null, durationMs: 10, latencyMs: 50 },
        { probeId: "p2", cached: false, passRate: 0, casesPassed: 0, casesTotal: 3, failReason: "sandbox_unavailable", durationMs: 0, latencyMs: 0 },
        { probeId: "p3", cached: false, passRate: 0, casesPassed: 0, casesTotal: 3, failReason: null, durationMs: 0, latencyMs: 10, modelError: true },
      ];
      const s = bench.summarize("ollama", "m|fp|none", "g", "", results);
      assert.equal(s.nProbes, 1);
      assert.equal(s.meanPassAt1, 1);
      assert.equal(s.eligible, false); // < 3
    });

    it("computes meanPassAt1 with partial success", () => {
      const results: ExecProbeOutcome[] = [
        { probeId: "p1", cached: false, passRate: 1, casesPassed: 3, casesTotal: 3, failReason: null, durationMs: 10, latencyMs: 50 },
        { probeId: "p2", cached: false, passRate: 1, casesPassed: 3, casesTotal: 3, failReason: null, durationMs: 10, latencyMs: 50 },
        { probeId: "p3", cached: false, passRate: 0.5, casesPassed: 1, casesTotal: 2, failReason: null, durationMs: 10, latencyMs: 50 },
      ];
      const s = bench.summarize("ollama", "m|fp|none", "g", "", results);
      assert.equal(s.nProbes, 3);
      assert.equal(s.meanPassAt1, 2 / 3);
      assert.equal(s.meanCasePassRate, (1 + 1 + 0.5) / 3);
      assert.equal(s.eligible, true);
      assert.equal(s.eligible, s.nProbes >= 3);
    });

    it("deduplicates by probe id (latest wins)", () => {
      const results: ExecProbeOutcome[] = [
        { probeId: "p1", cached: false, passRate: 0, casesPassed: 0, casesTotal: 3, failReason: "runtime_error", durationMs: 10, latencyMs: 50 },
        { probeId: "p1", cached: false, passRate: 1, casesPassed: 3, casesTotal: 3, failReason: null, durationMs: 10, latencyMs: 50 },
      ];
      const s = bench.summarize("ollama", "m|fp|none", "g", "", results);
      assert.equal(s.nProbes, 1);
      assert.equal(s.meanPassAt1, 1);
    });
  });

  // ─── summarizePersisted ───────────────────────────

  describe("summarizePersisted", () => {
    it("returns empty when nothing persisted", () => {
      const s = bench.summarizePersisted("ollama", "m|fp|none");
      assert.equal(s.nProbes, 0);
      assert.equal(s.eligible, false);
    });

    it("aggregates from stored rows, newest per probe", () => {
      // Persist two rows for same probe: first old (fail), then new (pass)
      db.saveExecBenchmarkResult({
        modelKey: "m|fp|none", provider: "ollama", probeId: "p1",
        generation: "gen-2026-09-08b",
        passRate: 0, casesPassed: 0, casesTotal: 3,
        failReason: "runtime_error", durationMs: 10,
        modelVersionHash: "v1", backend: "docker",
        timestamp: "2026-01-01T00:00:00Z",
      });
      db.saveExecBenchmarkResult({
        modelKey: "m|fp|none", provider: "ollama", probeId: "p1",
        generation: "gen-2026-09-08b",
        passRate: 1, casesPassed: 3, casesTotal: 3,
        failReason: null, durationMs: 10,
        modelVersionHash: "v1", backend: "docker",
        timestamp: "2026-06-01T00:00:00Z",
      });
      // Unrelated probe
      db.saveExecBenchmarkResult({
        modelKey: "m|fp|none", provider: "ollama", probeId: "p2",
        generation: "gen-2026-09-08b",
        passRate: 1, casesPassed: 3, casesTotal: 3,
        failReason: null, durationMs: 10,
        modelVersionHash: "v1", backend: "docker",
        timestamp: "2026-06-01T00:00:00Z",
      });
      const s = bench.summarizePersisted("ollama", "m|fp|none");
      assert.equal(s.nProbes, 2);
      // both newest rows pass, so no fail reasons
      assert.deepEqual(s.failReasonCounts, {});
      assert.equal(s.eligible, false); // 2 < 3
    });
  });

  // ─── benchmarkIdentity ────────────────────────────

  describe("benchmarkIdentity", () => {
    it("runs probes and reports outcomes", async () => {
      const sandbox = makeMockSandbox([
        { casesPassed: 4, casesTotal: 4 },
        { casesPassed: 2, casesTotal: 4, failReason: "wrong_output" },
        { casesPassed: 4, casesTotal: 4 },
      ]);
      const mc = makeModelCaller();
      const b = buildBench(db, { sandbox, callModel: mc, versionHash: async () => "v1" });
      const probes = [
        makeProbe({ id: "p1" }),
        makeProbe({ id: "p2" }),
        makeProbe({ id: "p3" }),
      ];
      const report = await b.benchmarkIdentity("ollama", { model: "m", quant: "q4", effort: "medium" }, { probes });
      assert.equal(report.skipped, null);
      assert.equal(report.results.length, 3);
      assert.equal(report.summary.nProbes, 3);
      assert.equal(report.summary.meanPassAt1, 2 / 3);
      assert.equal(report.summary.eligible, true);
    });

    it("caches results and returns cached=true", async () => {
      // First run: persist a fresh result
      const sandbox = makeMockSandbox([{ casesPassed: 4, casesTotal: 4 }]);
      const b = buildBench(db, { sandbox, versionHash: async () => "v1", now: () => 2000000 });
      const probes = [makeProbe({ id: "p-cached" })];
      const report1 = await b.benchmarkIdentity("openai", { model: "m", quant: "as-served", effort: "medium" }, { probes });
      assert.equal(report1.results[0].cached, false);

      // Second run with same model key/probe/generation/version hash -> cache hit
      const b2 = buildBench(db, { sandbox: makeMockSandbox([]), versionHash: async () => "v1", now: () => 3000000 });
      const report2 = await b2.benchmarkIdentity("openai", { model: "m", quant: "as-served", effort: "medium" }, { probes });
      assert.equal(report2.results[0].cached, true);
      assert.equal(report2.results[0].passRate, 1);
    });

    it("skips cache on sandbox_unavailable entries", async () => {
      // Persist a sandbox_unavailable entry
      db.saveExecBenchmarkResult({
        modelKey: "m|as-served|medium", provider: "openai", probeId: "p-sandown",
        generation: "gen-2026-09-08b",
        passRate: 0, casesPassed: 0, casesTotal: 3,
        failReason: "sandbox_unavailable", durationMs: 0,
        modelVersionHash: "v1", backend: null,
        timestamp: new Date().toISOString(),
      });
      const sandbox = makeMockSandbox([{ casesPassed: 3, casesTotal: 3 }]);
      const b = buildBench(db, { sandbox, versionHash: async () => "v1", now: () => Date.now() });
      const probes = [makeProbe({ id: "p-sandown" })];
      const report = await b.benchmarkIdentity("openai", { model: "m", quant: "as-served", effort: "medium" }, { probes });
      assert.equal(report.results[0].cached, false);
      assert.equal(report.results[0].passRate, 1);
    });

    it("recognizes modelError and excludes from summary", async () => {
      const mc = makeModelCaller("", 50, true); // throws every call
      const b = buildBench(db, { callModel: mc, versionHash: async () => "v1", now: () => Date.now() });
      const probes = [makeProbe({ id: "p1" })];
      const report = await b.benchmarkIdentity("openai", { model: "m", quant: "as-served", effort: "medium" }, { probes });
      assert.equal(report.results[0].modelError, true, "model call failure should set modelError");
      assert.equal(report.summary.nProbes, 0, "model error should not count toward summary");
    });

    it("skipped when GPU busy (local)", async () => {
      const b = buildBench(db, { gpuIdle: async () => false, versionHash: async () => "v1", now: () => Date.now() });
      const probes = [makeProbe({ id: "p1" })];
      const report = await b.benchmarkIdentity("ollama", { model: "m", quant: "q4", effort: "medium" }, { probes });
      assert.equal(report.skipped, "gpu_busy");
      assert.equal(report.results.length, 0);
    });

    it("skipped when GPU unknown (local)", async () => {
      const b = buildBench(db, { gpuIdle: async () => null, versionHash: async () => "v1", now: () => Date.now() });
      const probes = [makeProbe({ id: "p1" })];
      const report = await b.benchmarkIdentity("ollama", { model: "m", quant: "q4", effort: "medium" }, { probes });
      assert.equal(report.skipped, "gpu_unknown");
      assert.equal(report.results.length, 0);
    });

    it("honors cache hits during GPU-busy gating", async () => {
      // Pre-populate
      db.saveExecBenchmarkResult({
        modelKey: "m|q4|medium", provider: "ollama", probeId: "p-cached",
        generation: "gen-2026-09-08b",
        passRate: 1, casesPassed: 3, casesTotal: 3,
        failReason: null, durationMs: 10,
        modelVersionHash: "v1", backend: "docker",
        timestamp: new Date(Date.now() - 1000).toISOString(),
      });
      const b = buildBench(db, { gpuIdle: async () => false, versionHash: async () => "v1", now: () => Date.now() });
      const probes = [makeProbe({ id: "p-cached" })];
      const report = await b.benchmarkIdentity("ollama", { model: "m", quant: "q4", effort: "medium" }, { probes });
      assert.equal(report.skipped, "gpu_busy");
      assert.equal(report.results.length, 1, "cache hits should emit during GPU-gating");
      assert.equal(report.results[0].cached, true);
      assert.equal(report.results[0].passRate, 1);
    });

    it("passes cached=false when forceRefresh is set", async () => {
      db.saveExecBenchmarkResult({
        modelKey: "m|as-served|medium", provider: "openai", probeId: "p-force",
        generation: "gen-2026-09-08b",
        passRate: 0.5, casesPassed: 2, casesTotal: 4,
        failReason: "wrong_output", durationMs: 10,
        modelVersionHash: "v1", backend: "docker",
        timestamp: new Date().toISOString(),
      });
      const sandbox = makeMockSandbox([{ casesPassed: 4, casesTotal: 4 }]);
      const b = buildBench(db, { sandbox, versionHash: async () => "v1", now: () => Date.now() });
      const probes = [makeProbe({ id: "p-force" })];
      const report = await b.benchmarkIdentity("openai", { model: "m", quant: "as-served", effort: "medium" }, { probes, forceRefresh: true });
      assert.equal(report.results[0].cached, false);
      assert.equal(report.results[0].passRate, 1);
    });

    it("returns skipped=no_probes when no probes match", async () => {
      const report = await bench.benchmarkIdentity("openai", { model: "m", quant: "as-served", effort: "medium" }, { probes: [] });
      assert.equal(report.skipped, "no_probes");
      assert.equal(report.results.length, 0);
    });

    it("handles no_code_block extraction failure", async () => {
      const mc = makeModelCaller("Just text, no code fence");
      const sandbox = makeMockSandbox([]);
      const b = buildBench(db, { callModel: mc, sandbox, versionHash: async () => "v1", now: () => Date.now() });
      const probes = [makeProbe({ id: "p-nocode" })];
      const report = await b.benchmarkIdentity("openai", { model: "m", quant: "as-served", effort: "medium" }, { probes });
      assert.equal(report.results[0].failReason, "no_code_block");
      assert.equal(report.results[0].passRate, 0);
      // Verify persisted
      const cached = db.getLatestExecBenchmarkResult("m|as-served|medium", "p-nocode", "gen-2026-09-08b");
      assert.ok(cached !== null);
      assert.equal(cached.failReason, "no_code_block");
    });
  });

  // ─── Cache TTL ────────────────────────────────────

  describe("cache TTL", () => {
    it("respects 48h TTL for remote identities", async () => {
      const staleTime = Date.now() - 47 * 3_600_000; // 47h ago
      const now = staleTime + REMOTE_CACHE_TTL_MS - 1; // just inside TTL
      db.saveExecBenchmarkResult({
        modelKey: "m|as-served|medium", provider: "openai", probeId: "p-ttl",
        generation: "gen-2026-09-08b",
        passRate: 1, casesPassed: 3, casesTotal: 3,
        failReason: null, durationMs: 10,
        modelVersionHash: "v1", backend: "docker",
        timestamp: new Date(staleTime).toISOString(),
      });
      const b = buildBench(db, { versionHash: async () => "v1", now: () => now });
      const probes = [makeProbe({ id: "p-ttl" })];
      const report = await b.benchmarkIdentity("openai", { model: "m", quant: "as-served", effort: "medium" }, { probes });
      assert.equal(report.results[0].cached, true);
    });

    it("misses cache when stale (beyond 48h)", async () => {
      const staleTime = Date.now() - (REMOTE_CACHE_TTL_MS + 1);
      const now = staleTime + REMOTE_CACHE_TTL_MS + 1000;
      const sandbox = makeMockSandbox([{ casesPassed: 4, casesTotal: 4 }]);
      db.saveExecBenchmarkResult({
        modelKey: "m|as-served|medium", provider: "openai", probeId: "p-ttl-stale",
        generation: "gen-2026-09-08b",
        passRate: 0.5, casesPassed: 1, casesTotal: 2,
        failReason: "wrong_output", durationMs: 10,
        modelVersionHash: "v1", backend: "docker",
        timestamp: new Date(staleTime).toISOString(),
      });
      const b = buildBench(db, { sandbox, versionHash: async () => "v1", now: () => now });
      const probes = [makeProbe({ id: "p-ttl-stale" })];
      const report = await b.benchmarkIdentity("openai", { model: "m", quant: "as-served", effort: "medium" }, { probes });
      assert.equal(report.results[0].cached, false);
    });

    it("misses cache on version hash change", async () => {
      db.saveExecBenchmarkResult({
        modelKey: "m|as-served|medium", provider: "openai", probeId: "p-version",
        generation: "gen-2026-09-08b",
        passRate: 0.5, casesPassed: 1, casesTotal: 2,
        failReason: "wrong_output", durationMs: 10,
        modelVersionHash: "v1", backend: "docker",
        timestamp: new Date().toISOString(),
      });
      const sandbox = makeMockSandbox([{ casesPassed: 4, casesTotal: 4 }]);
      const b = buildBench(db, { sandbox, versionHash: async () => "new-v2", now: () => Date.now() });
      const probes = [makeProbe({ id: "p-version" })];
      const report = await b.benchmarkIdentity("openai", { model: "m", quant: "as-served", effort: "medium" }, { probes });
      assert.equal(report.results[0].cached, false);
    });
  });

  // ─── failReason counts ────────────────────────────

  describe("failReason counts", () => {
    it("accumulates fail reasons across probes", () => {
      const results: ExecProbeOutcome[] = [
        { probeId: "p1", cached: false, passRate: 1, casesPassed: 3, casesTotal: 3, failReason: null, durationMs: 10, latencyMs: 50 },
        { probeId: "p2", cached: false, passRate: 0.5, casesPassed: 1, casesTotal: 2, failReason: "wrong_output", durationMs: 10, latencyMs: 50 },
        { probeId: "p3", cached: false, passRate: 0, casesPassed: 0, casesTotal: 2, failReason: "runtime_error", durationMs: 10, latencyMs: 50 },
      ];
      const s = bench.summarize("ollama", "m|fp|none", "g", "", results);
      assert.deepEqual(s.failReasonCounts, {
        wrong_output: 1,
        runtime_error: 1,
      });
    });
  });
});