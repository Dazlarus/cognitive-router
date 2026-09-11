// tests/judge_calibration.test.ts - Tests for p3e-005
//
// Covers: (1) pure stats (pearson, computeCalibrationStats — agreement,
// buckets, degenerate inputs); (2) DB round-trip on judge_calibration;
// (3) ExecBenchmark judge hook — pair logging on fresh runs, suppression
// on sandbox_unavailable and judge failure; (4) service report shape +
// divergence flag thresholds.

import { describe, it, beforeEach, afterEach, after } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, rmSync } from "node:fs";
import { resolve } from "node:path";
import {
  computeCalibrationStats,
  pearson,
  JudgeCalibrationService,
  MIN_PAIRS_FOR_REPORT,
  DIVERGENCE_AGREEMENT_PCT,
  type CalibrationRowInput,
} from "../src/judge_calibration.js";
import {
  ExecBenchmark,
  type ExecBenchDeps,
} from "../src/benchmark_exec.js";
import { DBService } from "../src/db_service.js";
import type { ExecRunResult, FailReason } from "../src/exec_sandbox.js";
import type { CodingProbe } from "../src/probes.js";

// ─── Helpers ─────────────────────────────────────────────────

let _counter = 0;
const _tempPaths: string[] = [];

async function makeTempDb(): Promise<DBService> {
  _counter++;
  const dir = resolve(`data/test-calib-${Date.now()}-${_counter}`);
  mkdirSync(dir, { recursive: true });
  const path = `${dir}/bench.db`;
  _tempPaths.push(dir);
  const db = new DBService(path);
  await db.initializeSchema();
  return db;
}

after(() => {
  for (const p of _tempPaths) {
    try { rmSync(p, { force: true, recursive: true }); } catch { /* ok */ }
  }
});

function row(judgeScore: number, passRate: number, extra: Partial<CalibrationRowInput> = {}): CalibrationRowInput {
  return { id: 0, judgeScore, passRate, judgeProvider: null, judgeModel: null, ...extra };
}

function makeProbe(overrides: Partial<CodingProbe> = {}): CodingProbe {
  return {
    id: `calib-probe`,
    generation: "gen-2026-09-08b",
    subIntent: "quick",
    prompt: "Write a Python function.",
    language: "python",
    entry: "f",
    harness: {
      setup: "from solution import f",
      cases: [
        { call: "f(1)", expect: "2" },
        { call: "f(2)", expect: "4" },
      ],
      casesHidden: 0,
    },
    ...overrides,
  };
}

function makeMockSandbox(result: { casesPassed: number; casesTotal: number; failReason?: FailReason | null }) {
  return {
    run: async (): Promise<ExecRunResult> => ({
      casesPassed: result.casesPassed,
      casesTotal: result.casesTotal,
      passRate: result.casesTotal > 0 ? result.casesPassed / result.casesTotal : 0,
      failReason: result.failReason ?? null,
      durationMs: 5,
      backend: "docker",
    }),
    available: async (): Promise<"docker" | "win32-job" | null> => "docker",
  };
}

async function makeBench(
  db: DBService,
  opts: {
    judge?: (p: string, r: string) => Promise<{ rawScore: number } | null>;
    sandboxResult: { casesPassed: number; casesTotal: number; failReason?: FailReason | null };
  },
): Promise<ExecBenchmark> {
  const deps: ExecBenchDeps = {
    db,
    callModel: async () => "Sure:\n```python\ndef f(x):\n    return x * 2\n```",
    judgeScore: opts.judge,
    sandbox: makeMockSandbox(opts.sandboxResult),
    versionHash: async () => "v1",
    gpuIdle: async () => true,
    now: () => 1_700_000_000_000,
  };
  return new ExecBenchmark(deps);
}

// ─── pearson ─────────────────────────────────────────────────

describe("pearson", () => {
  it("returns +1 for perfectly increasing pairs", () => {
    assert.equal(pearson([1, 2, 3, 4], [0.1, 0.4, 0.7, 1.0]), 1);
  });

  it("returns -1 for perfectly inverted pairs", () => {
    assert.equal(pearson([1, 2, 3, 4], [1.0, 0.7, 0.4, 0.1]), -1);
  });

  it("returns null on n<2", () => {
    assert.equal(pearson([1], [0.5]), null);
  });

  it("returns null on zero variance", () => {
    assert.equal(pearson([7, 7, 7], [0.1, 0.5, 0.9]), null);
    assert.equal(pearson([3, 5, 8], [0.5, 0.5, 0.5]), null);
  });
});

// ─── computeCalibrationStats ─────────────────────────────────

describe("computeCalibrationStats", () => {
  it("computes perfect threshold agreement", () => {
    const s = computeCalibrationStats([
      row(8, 1.0), row(9, 0.9), row(7, 0.75),
      row(3, 0.0), row(2, 0.1), row(4, 0.25),
    ]);
    assert.equal(s.n, 6);
    assert.equal(s.agreementPct, 100);
    assert.equal(s.judgePassExecPass, 3);
    assert.equal(s.judgePassExecFail, 0);
    assert.equal(s.judgeFailExecPass, 0);
    assert.equal(s.judgeFailExecFail, 3);
  });

  it("computes disagreement split", () => {
    const s = computeCalibrationStats([
      row(8, 1.0), row(8, 0.0),   // one agree, one judge-pass/exec-fail
      row(3, 1.0), row(3, 0.0),   // one judge-fail/exec-pass, one agree
    ]);
    assert.equal(s.agreementPct, 50);
    assert.equal(s.judgePassExecFail, 1);
    assert.equal(s.judgeFailExecPass, 1);
  });

  it("empty input yields nulls, not NaN", () => {
    const s = computeCalibrationStats([]);
    assert.equal(s.n, 0);
    assert.equal(s.agreementPct, null);
    assert.equal(s.pearsonR, null);
    assert.equal(s.meanJudgeScore, null);
    assert.equal(s.meanPassRate, null);
    assert.deepEqual(s.buckets, []);
  });

  it("buckets mean pass@1 by raw judge score", () => {
    const s = computeCalibrationStats([row(7, 1.0), row(7, 0.5), row(9, 1.0)]);
    assert.deepEqual(s.buckets, [
      { judgeScore: 7, n: 2, meanPassRate: 0.75 },
      { judgeScore: 9, n: 1, meanPassRate: 1 },
    ]);
  });

  it("honors custom thresholds", () => {
    const s = computeCalibrationStats([row(9, 0.4)], { passThreshold: 0.3 });
    assert.equal(s.judgePassExecPass, 1);
  });
});

// ─── DB round-trip ───────────────────────────────────────────

describe("judge_calibration table", () => {
  let db: DBService;

  beforeEach(async () => { db = await makeTempDb(); });
  afterEach(() => { try { db.close(); } catch { /* ok */ } });

  it("inserts and reads back rows with provenance", () => {
    db.insertJudgeCalibration({
      probeId: "coding-exec-001",
      modelKey: "zai/glm-5.3|effort=low",
      generation: "gen-2026-09-08b",
      judgeScore: 8,
      passRate: 1.0,
      casesPassed: 5,
      casesTotal: 5,
      judgeProvider: "openrouter",
      judgeModel: "qwen/qwen3-30b-a3b-instruct-2507",
      execBackend: "docker",
    });
    db.insertJudgeCalibration({
      probeId: "coding-exec-002",
      modelKey: "zai/glm-5.3|effort=low",
      generation: "gen-2026-09-08b",
      judgeScore: 3,
      passRate: 0,
      casesPassed: 0,
      casesTotal: 5,
    });
    const rows = db.getJudgeCalibrationRows("gen-2026-09-08b");
    assert.equal(rows.length, 2);
    assert.equal(rows[0].judgeScore, 8);
    assert.equal(rows[0].judgeProvider, "openrouter");
    assert.equal(rows[0].execBackend, "docker");
    assert.equal(rows[1].judgeProvider, null);
    // generation filter
    assert.equal(db.getJudgeCalibrationRows("gen-older").length, 0);
    assert.equal(db.getJudgeCalibrationRows().length, 2);
  });

  it("latest timestamp lookup is per (model, probe, generation)", () => {
    db.insertJudgeCalibration({
      probeId: "p1", modelKey: "m1", generation: "g1",
      judgeScore: 5, passRate: 0.5, casesPassed: 2, casesTotal: 4,
      timestamp: "2026-09-11T00:00:00.000Z",
    });
    db.insertJudgeCalibration({
      probeId: "p1", modelKey: "m1", generation: "g1",
      judgeScore: 6, passRate: 0.75, casesPassed: 3, casesTotal: 4,
      timestamp: "2026-09-11T01:00:00.000Z",
    });
    db.insertJudgeCalibration({
      probeId: "p2", modelKey: "m1", generation: "g1",
      judgeScore: 7, passRate: 1.0, casesPassed: 4, casesTotal: 4,
      timestamp: "2026-09-11T02:00:00.000Z",
    });
    assert.equal(db.getLatestJudgeCalibrationTimestamp("m1", "p1", "g1"), "2026-09-11T01:00:00.000Z");
    assert.equal(db.getLatestJudgeCalibrationTimestamp("m1", "p2", "g1"), "2026-09-11T02:00:00.000Z");
    assert.equal(db.getLatestJudgeCalibrationTimestamp("m1", "p3", "g1"), null);
  });
});

// ─── ExecBenchmark judge hook ────────────────────────────────

describe("ExecBenchmark judge hook", () => {
  let db: DBService;

  beforeEach(async () => { db = await makeTempDb(); });
  afterEach(() => { try { db.close(); } catch { /* ok */ } });

  it("logs a (judge_score, pass@1) pair on a fresh passing run", async () => {
    const bench = await makeBench(db, {
      judge: async () => ({ rawScore: 8, judgeProvider: "openrouter", judgeModel: "qwen/x" }),
      sandboxResult: { casesPassed: 5, casesTotal: 5 },
    });
    const report = await bench.benchmarkIdentity("zai", {
      provider: "zai", model: "glm-5.3", effort: "low",
    } as any, { probes: [makeProbe()] });

    assert.equal(report.results.length, 1);
    assert.equal(report.results[0].judgeScore, 8);
    const rows = db.getJudgeCalibrationRows();
    assert.equal(rows.length, 1);
    assert.equal(rows[0].judgeScore, 8);
    assert.equal(rows[0].passRate, 1.0);
    assert.equal(rows[0].judgeProvider, "openrouter");
    assert.equal(rows[0].execBackend, "docker");
    // exec result persisted too — the pair is additive, never a replacement
    const exec = db.getLatestExecBenchmarkResult(report.modelKey, "calib-probe", "gen-2026-09-08b");
    assert.ok(exec);
    assert.equal(exec!.passRate, 1.0);
  });

  it("logs the pair on no_code_block (judge saw prose, exec floor is a real 0)", async () => {
    const deps: ExecBenchDeps = {
      db,
      callModel: async () => "I would write a function that doubles x. No code though.",
      judgeScore: async () => ({ rawScore: 7 }),
      sandbox: makeMockSandbox({ casesPassed: 0, casesTotal: 2 }),
      versionHash: async () => "v1",
      gpuIdle: async () => true,
      now: () => 1_700_000_000_000,
    };
    const bench = new ExecBenchmark(deps);
    const report = await bench.benchmarkIdentity("zai", {
      provider: "zai", model: "glm-5.3", effort: "low",
    } as any, { probes: [makeProbe()] });

    assert.equal(report.results[0].failReason, "no_code_block");
    assert.equal(db.getJudgeCalibrationRows().length, 1);
  });

  it("suppresses the pair when the sandbox is unavailable (no exec signal)", async () => {
    const bench = await makeBench(db, {
      judge: async () => ({ rawScore: 9 }),
      sandboxResult: { casesPassed: 0, casesTotal: 0, failReason: "sandbox_unavailable" },
    });
    await bench.benchmarkIdentity("zai", {
      provider: "zai", model: "glm-5.3", effort: "low",
    } as any, { probes: [makeProbe()] });

    assert.equal(db.getJudgeCalibrationRows().length, 0);
  });

  it("judge failure never breaks the exec run (pair skipped)", async () => {
    const bench = await makeBench(db, {
      judge: async () => { throw new Error("judge down"); },
      sandboxResult: { casesPassed: 4, casesTotal: 5 },
    });
    const report = await bench.benchmarkIdentity("zai", {
      provider: "zai", model: "glm-5.3", effort: "low",
    } as any, { probes: [makeProbe()] });

    assert.equal(report.results[0].passRate, 0.8);
    assert.equal(report.results[0].judgeScore, undefined);
    assert.equal(db.getJudgeCalibrationRows().length, 0);
  });

  it("judge returning null skips the pair without failing", async () => {
    const bench = await makeBench(db, {
      judge: async () => null,
      sandboxResult: { casesPassed: 5, casesTotal: 5 },
    });
    const report = await bench.benchmarkIdentity("zai", {
      provider: "zai", model: "glm-5.3", effort: "low",
    } as any, { probes: [makeProbe()] });

    assert.equal(report.results[0].passRate, 1.0);
    assert.equal(db.getJudgeCalibrationRows().length, 0);
  });

  it("cached hits produce no new pairs (judge scores fresh responses only)", async () => {
    const bench = await makeBench(db, {
      judge: async () => ({ rawScore: 8 }),
      sandboxResult: { casesPassed: 5, casesTotal: 5 },
    });
    const probe = makeProbe();
    await bench.benchmarkIdentity("zai", {
      provider: "zai", model: "glm-5.3", effort: "low",
    } as any, { probes: [probe] });
    assert.equal(db.getJudgeCalibrationRows().length, 1);

    // Second run within TTL: cache hit, no judge call, no new pair.
    const bench2 = await makeBench(db, {
      judge: async () => { throw new Error("must not be called on cache hit"); },
      sandboxResult: { casesPassed: 5, casesTotal: 5 },
    });
    const report2 = await bench2.benchmarkIdentity("zai", {
      provider: "zai", model: "glm-5.3", effort: "low",
    } as any, { probes: [probe] });
    assert.equal(report2.results[0].cached, true);
    assert.equal(db.getJudgeCalibrationRows().length, 1);
  });
});

// ─── JudgeCalibrationService ─────────────────────────────────

describe("JudgeCalibrationService", () => {
  let db: DBService;

  beforeEach(async () => { db = await makeTempDb(); });
  afterEach(() => { try { db.close(); } catch { /* ok */ } });

  it("reports zero pairs without error", async () => {
    const svc = new JudgeCalibrationService(db);
    const rep = await svc.report("test");
    assert.equal(rep.skipped, false);
    assert.equal(rep.overall.n, 0);
    assert.equal(rep.divergenceSuspected, false);
  });

  it("aggregates pairs and flags divergence past thresholds", async () => {
    // 20 pairs, only 8 agreeing = 40% < 60% threshold → divergence.
    for (let i = 0; i < 10; i++) {
      db.insertJudgeCalibration({
        probeId: `agree-${i}`, modelKey: "m1", generation: "gen-2026-09-08b",
        judgeScore: 8, passRate: 1.0, casesPassed: 5, casesTotal: 5,
      });
      db.insertJudgeCalibration({
        probeId: `disagree-${i}`, modelKey: "m1", generation: "gen-2026-09-08b",
        judgeScore: 8, passRate: 0.0, casesPassed: 0, casesTotal: 5,
      });
    }
    const svc = new JudgeCalibrationService(db);
    const rep = await svc.report("test");
    assert.equal(rep.overall.n, MIN_PAIRS_FOR_REPORT);
    assert.ok(rep.overall.agreementPct != null && rep.overall.agreementPct < DIVERGENCE_AGREEMENT_PCT);
    assert.equal(rep.divergenceSuspected, true);
  });

  it("does not flag divergence below the minimum sample size", async () => {
    // 2 pairs, 0 agreement — real divergence, but n < MIN_PAIRS_FOR_REPORT.
    db.insertJudgeCalibration({
      probeId: "d1", modelKey: "m1", generation: "gen-2026-09-08b",
      judgeScore: 9, passRate: 0.0, casesPassed: 0, casesTotal: 5,
    });
    db.insertJudgeCalibration({
      probeId: "d2", modelKey: "m1", generation: "gen-2026-09-08b",
      judgeScore: 9, passRate: 0.0, casesPassed: 0, casesTotal: 5,
    });
    const svc = new JudgeCalibrationService(db);
    const rep = await svc.report("test");
    assert.equal(rep.overall.agreementPct, 0);
    assert.equal(rep.divergenceSuspected, false);
  });

  it("splits stats per judge identity from provenance", async () => {
    db.insertJudgeCalibration({
      probeId: "p1", modelKey: "m1", generation: "gen-2026-09-08b",
      judgeScore: 8, passRate: 1.0, casesPassed: 5, casesTotal: 5,
      judgeProvider: "openrouter", judgeModel: "qwen/x",
    });
    db.insertJudgeCalibration({
      probeId: "p2", modelKey: "m1", generation: "gen-2026-09-08b",
      judgeScore: 2, passRate: 0.0, casesPassed: 0, casesTotal: 5,
      judgeProvider: "ollama", judgeModel: "gemma4:latest",
    });
    const svc = new JudgeCalibrationService(db);
    const rep = await svc.report("test");
    assert.equal(rep.perJudge.length, 2);
    assert.ok(rep.perJudge.some((j) => j.judge === "openrouter/qwen/x" && j.n === 1));
    assert.ok(rep.perJudge.some((j) => j.judge === "ollama/gemma4:latest" && j.n === 1));
  });

  it("startPeriodic arms and stopPeriodic disarms without leaks", async () => {
    const svc = new JudgeCalibrationService(db, 1);
    svc.startPeriodic();
    svc.startPeriodic(); // idempotent
    svc.stopPeriodic();
    svc.stopPeriodic(); // idempotent
    assert.ok(true);
  });
});
