// tests/coding_blend.test.ts - Unit tests for CodingBlendService (p3e-007)
// Run with: npx tsx --test tests/coding_blend.test.ts

import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { CodingBlendService } from "../src/coding_blend.ts";
import { PROMPT_GENERATION } from "../src/benchmark_ladder.ts";
import { MIN_PROBE_RESULTS } from "../src/benchmark_exec.ts";

// ─── Fixtures ───

/** Minimal mock DB covering only the 4 methods CodingBlendService touches. */
function makeMockDB(overrides: Partial<Record<string, Function>> = {}): any {
  return {
    getLadder: (_intent: string, _gen: string) => [
      { modelKey: "glm-5.2|as-served|medium", rank: 1, strength: 3.0 },
      { modelKey: "qwen3-235b|as-served|medium", rank: 2, strength: 1.2 },
      { modelKey: "glm-4.7|as-served|medium", rank: 3, strength: 0.8 },
    ],
    getAllBenchmarkVerdicts: (_intent: string, _gen: string) => [],
    getExecResultsForIdentity: (_modelKey: string, _gen: string) => [],
    getCapabilityOverride: (_p: string, _m: string, _i: string) => null,
    ...overrides,
  };
}

function makeMockRegistry(score = 0): any {
  return { getCapabilityScore: () => score };
}

const LADDER_KEY = "glm-5.2|as-served|medium";

// ─── Tests ───

describe("CodingBlendService", () => {
  let svc: CodingBlendService;
  let db: any;

  beforeEach(() => {
    db = makeMockDB();
    svc = new CodingBlendService(db, makeMockRegistry());
  });

  describe("blendModelKey", () => {
    it("returns null for identity without a ladder entry (fall-through path)", () => {
      assert.equal(svc.blendModelKey("unknown-model|as-served|medium"), null);
    });

    it("with zero comparisons, shrunk BT strength equals neutral 1.0", () => {
      const r = svc.blendModelKey(LADDER_KEY)!;
      assert.ok(r);
      assert.equal(r.btShrunk, 1.0);
      assert.equal(r.btStrength, 3.0);
      assert.equal(r.ewmaSamples, 0);
    });

    it("shrinkage increases with comparison count and approaches raw strength", () => {
      const verdicts = [
        { modelA: LADDER_KEY, modelB: "opponent-a", round: 1 },
        { modelA: "opponent-a", modelB: LADDER_KEY, round: 2 },
        { modelA: LADDER_KEY, modelB: "opponent-b", round: 1 },
      ];
      const dbN = makeMockDB({
        getAllBenchmarkVerdicts: () => verdicts,
      });
      const svcN = new CodingBlendService(dbN, makeMockRegistry());
      const r = svcN.blendModelKey(LADDER_KEY)!;
      // n=3 unique (opponent,round) pairs → factor 3/6 = 0.5 → shrunk = 2.0
      assert.equal(r.btShrunk, 2.0);
    });

    it("zero-comparison ladder normalizes to neutral btNorm=0.5 for all (all shrink to 1.0)", () => {
      const top = svc.blendModelKey(LADDER_KEY)!;
      const bottom = svc.blendModelKey("glm-4.7|as-served|medium")!;
      assert.equal(top.btNorm, 0.5);
      assert.equal(bottom.btNorm, 0.5);
      assert.equal(top.blended, bottom.blended);
    });

    it("comparisons differentiate identities: more evidence pulls btNorm toward extremes", () => {
      // Top identity has 3 comparisons (shrunk toward 2.0), others have 0 (shrunk to 1.0)
      const verdicts = [
        { modelA: LADDER_KEY, modelB: "qwen3-235b|as-served|medium", round: 1 },
        { modelA: LADDER_KEY, modelB: "qwen3-235b|as-served|medium", round: 2 },
        { modelA: LADDER_KEY, modelB: "glm-4.7|as-served|medium", round: 1 },
      ];
      const svcN = new CodingBlendService(makeMockDB({ getAllBenchmarkVerdicts: () => verdicts }), makeMockRegistry());
      const top = svcN.blendModelKey(LADDER_KEY)!;
      const bottom = svcN.blendModelKey("glm-4.7|as-served|medium")!;
      assert.equal(top.btShrunk, 2.0);
      assert.ok(top.btNorm > bottom.btNorm);
      assert.ok(top.blended > bottom.blended);
    });

    it("exec floor is 0 when fewer than MIN_PROBE_RESULTS probes", () => {
      db.getExecResultsForIdentity = () =>
        Array.from({ length: MIN_PROBE_RESULTS - 1 }, (_, i) => ({
          probeId: `p${i}`, passRate: 1.0, failReason: null,
        }));
      const r = svc.blendModelKey(LADDER_KEY)!;
      assert.equal(r.execNProbes, MIN_PROBE_RESULTS - 1);
      assert.equal(r.execFloor, 0);
    });

    it("exec floor = mean pass@1 across probes once eligible", () => {
      db.getExecResultsForIdentity = () => [
        { probeId: "p1", passRate: 1.0, failReason: null },
        { probeId: "p2", passRate: 0.0, failReason: null },
        { probeId: "p3", passRate: 1.0, failReason: null },
      ];
      const r = svc.blendModelKey(LADDER_KEY)!;
      assert.equal(r.execNProbes, 3);
      assert.ok(Math.abs(r.execFloor - 2 / 3) < 1e-9);
    });

    it("excludes sandbox_unavailable probes and dedupes by probeId keeping first row", () => {
      db.getExecResultsForIdentity = () => [
        { probeId: "p1", passRate: 1.0, failReason: null },
        { probeId: "p2", passRate: 0.5, failReason: "sandbox_unavailable" }, // excluded
        { probeId: "p3", passRate: 0.0, failReason: null },
        { probeId: "p4", passRate: 1.0, failReason: null },
      ];
      const r = svc.blendModelKey(LADDER_KEY)!;
      assert.equal(r.execNProbes, 3);
      assert.ok(Math.abs(r.execFloor - 2 / 3) < 1e-9);
    });

    it("live EWMA falls back to neutral 0.5 when no override exists", () => {
      const r = svc.blendModelKey(LADDER_KEY)!;
      assert.equal(r.ewmaScore, 0.5);
      assert.equal(r.ewmaSamples, 0);
    });

    it("live EWMA uses capability override when present", () => {
      db.getCapabilityOverride = () => ({ score: 0.9, sampleCount: 12, pinned: false });
      const svcO = new CodingBlendService(db, makeMockRegistry(0.9));
      const r = svcO.blendModelKey(LADDER_KEY)!;
      assert.equal(r.ewmaSamples, 12);
      assert.ok(r.ewmaScore > 0.5);
    });

    it("blended score is clamped to [0,1]", () => {
      db.getCapabilityOverride = () => ({ score: 1.0, sampleCount: 50, pinned: false });
      db.getExecResultsForIdentity = () =>
        Array.from({ length: MIN_PROBE_RESULTS }, (_, i) => ({
          probeId: `p${i}`, passRate: 1.0, failReason: null,
        }));
      const r = svc.blendModelKey(LADDER_KEY)!;
      assert.ok(r.blended <= 1.0 && r.blended >= 0.0);
    });

    it("blended equals weighted sum with default weights (sanity)", () => {
      const r = svc.blendModelKey(LADDER_KEY)!;
      // zero comparisons: btNorm=0.5, execFloor=0, ewma=0.5 → 0.5*0.5 + 0.3*0 + 0.2*0.5 = 0.35
      assert.ok(Math.abs(r.blended - 0.35) < 1e-9);
    });
  });

  describe("blendProviderModel", () => {
    it("maps provider+model to medium-effort bench identity", () => {
      const r = svc.blendProviderModel("zai", "glm-5.2");
      assert.ok(r);
      assert.equal(r.modelKey, LADDER_KEY);
    });

    it("falls back to null when no effort level matches", () => {
      assert.equal(svc.blendProviderModel("zai", "nonexistent-model"), null);
    });
  });

  describe("report", () => {
    it("counts full vs partial blends and computes summary stats", () => {
      // Give every identity eligible exec + live overrides → all full
      db.getExecResultsForIdentity = () =>
        Array.from({ length: MIN_PROBE_RESULTS }, (_, i) => ({
          probeId: `p${i}`, passRate: 1.0, failReason: null,
        }));
      db.getCapabilityOverride = () => ({ score: 0.8, sampleCount: 5, pinned: false });
      const rep = svc.report();
      assert.equal(rep.identities.length, 3);
      assert.equal(rep.fullBlendCount, 3);
      assert.equal(rep.partialBlendCount, 0);
      assert.ok(rep.maxBlended >= rep.meanBlended && rep.meanBlended >= rep.minBlended);
      assert.equal(rep.generation, PROMPT_GENERATION);
    });

    it("reports partial when exec or live data missing", () => {
      const rep = svc.report(); // no exec rows, no overrides
      assert.equal(rep.fullBlendCount, 0);
      assert.equal(rep.partialBlendCount, 3);
    });
  });

  describe("setWeights", () => {
    it("normalizes weights that do not sum to 1", () => {
      svc.setWeights({ bt: 0.6, exec: 0.6, live: 0.6 }); // sum 1.8
      const rep = svc.report();
      const w = rep.weights;
      assert.ok(Math.abs(w.bt + w.exec + w.live - 1) < 0.01);
      // proportions preserved: each = 1/3
      assert.ok(Math.abs(w.bt - 1 / 3) < 0.01);
    });

    it("partial update triggers renormalization of all weights (sum stays 1)", () => {
      svc.setWeights({ exec: 0.4 }); // 0.5+0.4+0.2=1.1 → scaled by 1/1.1
      const w = svc.report().weights;
      assert.ok(Math.abs(w.bt + w.exec + w.live - 1) < 0.01);
      assert.ok(Math.abs(w.exec - 0.4 / 1.1) < 1e-9);
      assert.ok(Math.abs(w.bt - 0.5 / 1.1) < 1e-9);
    });
  });

  describe("hasFullBlend", () => {
    it("false when all blends are partial", () => {
      assert.equal(svc.hasFullBlend(), false);
    });

    it("true when at least one identity is fully blended", () => {
      db.getExecResultsForIdentity = (key: string) =>
        key === LADDER_KEY
          ? Array.from({ length: MIN_PROBE_RESULTS }, (_, i) => ({
              probeId: `p${i}`, passRate: 1.0, failReason: null,
            }))
          : [];
      db.getCapabilityOverride = (_p: string, m: string) =>
        m === "glm-5.2" ? { score: 0.8, sampleCount: 5, pinned: false } : null;
      assert.equal(svc.hasFullBlend(), true);
    });
  });
});
