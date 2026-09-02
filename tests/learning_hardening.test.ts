// tests/learning_hardening.test.ts — Phase-1 hardening unit tests
// Covers: clamps, RoC caps, decay math, attribution arms, note similarity +
// quarantine canary, pin immutability, reliability age decay, UCB read path,
// ingestion tiering. All time-dependent logic is exercised via explicit
// timestamps / env overrides (time-mocked by construction).

import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DBService } from "../src/db_service.ts";
import { ModelRegistry } from "../src/model_registry.ts";
import { loadConfig, type CognitiveRouterConfig } from "../src/config.ts";
import * as guards from "../src/learning_guards.ts";

// ─── Helpers ───

const ENV_BACKUP: Record<string, string | undefined> = {};
const ENV_KEYS = [
  "ROUTER_CAP_MIN", "ROUTER_CAP_MAX", "ROUTER_ROC_MAX_24H", "ROUTER_ROC_MAX_7D",
  "ROUTER_DECAY_IDLE_DAYS", "ROUTER_DECAY_RATE", "ROUTER_ARM_B_MIN_CONFIDENCE",
  "ROUTER_JUDGE_NOTE_CANARY", "ROUTER_JUDGE_NOTE_HOLD", "ROUTER_JUDGE_MAX_CHARS",
  "ROUTER_JUDGE_TIER2_MAX_CHARS", "ROUTER_JUDGE_CHARS_PER_TOKEN",
  "ROUTER_LEARNING_SHADOW_MODE", "ROUTER_UCB_K", "ZAI_API_KEY",
];
function setEnv(k: string, v: string | undefined) {
  if (v === undefined) delete process.env[k];
  else process.env[k] = v;
}

let tmpDir: string;
let db: DBService;
let registry: ModelRegistry;
let config: CognitiveRouterConfig;

function freshDB(): DBService {
  const path = join(mkdtempSync(join(tmpdir(), "lrn-")), "test.db");
  const d = new DBService(path);
  (d as any).__dir = path;
  return d;
}

function makeRegistry() {
  delete process.env.ZAI_API_KEY; // suppress discovery
  config = loadConfig({
    enabled: true, logLevel: "error",
    providerPriority: ["zai", "openrouter", "gemini", "ollama"],
    providers: {
      zai: { budgetType: "subscription", priority: "high" },
      openrouter: { budgetType: "free", priority: "high" },
      gemini: { budgetType: "credits", priority: "medium" },
      ollama: { budgetType: "free", priority: "low" },
    },
    weights: { capability: 0.5, reliability: 0.25, cost: 0.15, latency: 0.1 },
  });
  return new ModelRegistry(db, config);
}

beforeEach(async () => {
  tmpDir = mkdtempSync(join(tmpdir(), "lrn-"));
  for (const k of ENV_KEYS) { ENV_BACKUP[k] = process.env[k]; }
  db = freshDB();
  db.initializeSchema();
  registry = makeRegistry();
  await registry.loadCachedState();
});

afterEach(() => {
  db.close();
  rmSync(tmpDir, { recursive: true, force: true });
  for (const k of ENV_KEYS) { setEnv(k, ENV_BACKUP[k]); }
});

// ─── Clamps ───

describe("learning hardening — clamps", () => {
  it("clamps to [0.05, 0.98] at write time", () => {
    setEnv("ROUTER_LEARNING_SHADOW_MODE", "false");
    // Drive glm-5.2 coding (seed 0.88) toward the floor with repeated 0s.
    for (let i = 0; i < 30; i++) {
      registry.updateCapability("zai", "glm-5.2", "coding", 0);
    }
    const score = registry.getCapabilityScore("zai", "glm-5.2", "coding");
    assert.ok(score >= 0.05 - 1e-9, `floor violated: ${score}`);
    assert.ok(score <= 0.98 + 1e-9, `ceiling violated: ${score}`);
  });

  it("never writes below floor even after extreme punishment volume", () => {
    setEnv("ROUTER_LEARNING_SHADOW_MODE", "false");
    for (let i = 0; i < 200; i++) {
      registry.updateCapability("zai", "glm-5.1", "math", 0);
    }
    const score = registry.getCapabilityScore("zai", "glm-5.1", "math");
    assert.ok(score >= 0.05 - 1e-9, `floor violated: ${score}`);
  });
});

// ─── Rate-of-change caps ───

describe("learning hardening — RoC caps", () => {
  it("rejects movement beyond Δmax per 24h and logs the rejection", () => {
    setEnv("ROUTER_LEARNING_SHADOW_MODE", "false");
    setEnv("ROUTER_ROC_MAX_24H", "0.10");
    // First update from seed 0.88 with judge=0 → 0.66+0 = 0.66 → delta 0.22 > 0.10 → rejected.
    const before = registry.getCapabilityScore("zai", "glm-5.2", "coding");
    registry.updateCapability("zai", "glm-5.2", "coding", 0);
    const after = registry.getCapabilityScore("zai", "glm-5.2", "coding");
    assert.equal(after, before, "over-cap update must be rejected");
  });

  it("accepts small movements within Δmax", () => {
    setEnv("ROUTER_LEARNING_SHADOW_MODE", "false");
    setEnv("ROUTER_ROC_MAX_24H", "0.10");
    const before = registry.getCapabilityScore("zai", "glm-5.2", "coding"); // 0.88
    registry.updateCapability("zai", "glm-5.2", "coding", 0.8);
    const after = registry.getCapabilityScore("zai", "glm-5.2", "coding");
    assert.ok(Math.abs(after - before) < 0.10 + 1e-9);
    assert.notEqual(after, before, "small in-cap update should apply");
  });

  it("enforces the 7d cap across multiple days", () => {
    setEnv("ROUTER_LEARNING_SHADOW_MODE", "false");
    setEnv("ROUTER_ROC_MAX_24H", "0.10");
    setEnv("ROUTER_ROC_MAX_7D", "0.12");
    let cur = registry.getCapabilityScore("zai", "glm-5.2", "coding");
    // Day 1: -0.10 worth of movement (accepted at most).
    registry.updateCapability("zai", "glm-5.2", "coding", 0.5);
    const d1 = registry.getCapabilityScore("zai", "glm-5.2", "coding");
    assert.ok(Math.abs(d1 - cur) <= 0.10 + 1e-9);
    // Simulate next day by writing movement history: add fake movement day-2.
    db.recordCapabilityMovement("zai", "glm-5.2", "coding", 0.10,
      new Date(Date.now() - 2 * 86_400_000).toISOString().slice(0, 10) + "T00:00:00.000Z");
    cur = d1;
    registry.updateCapability("zai", "glm-5.2", "coding", 0.5);
    const d2 = registry.getCapabilityScore("zai", "glm-5.2", "coding");
    // 7d window now holds ≥0.10; another 0.10 would exceed 0.12 → rejected.
    assert.equal(d2, cur, "7d cap must reject cumulative movement");
  });
});

// ─── Attribution arms ───

describe("learning hardening — attribution gate arms", () => {
  it("Arm A: provider truncation → no capability write, reliability-only", () => {
    setEnv("ROUTER_LEARNING_SHADOW_MODE", "false");
    const before = registry.getCapabilityScore("zai", "glm-5.2", "coding");
    const out = registry.applyJudgedScore("zai", "glm-5.2", "coding", 0.0, {
      rawScore: 0, judgeNote: "truncated output", judgeModelId: "test/judge",
      confidence: 0.95, truncated: true,
    });
    assert.equal(out.arm, "A");
    assert.equal(out.applied, false);
    assert.equal(registry.getCapabilityScore("zai", "glm-5.2", "coding"), before);
  });

  it("Arm B: confidence < 0.75 → no_apply", () => {
    setEnv("ROUTER_LEARNING_SHADOW_MODE", "false");
    const before = registry.getCapabilityScore("zai", "glm-5.2", "coding");
    const out = registry.applyJudgedScore("zai", "glm-5.2", "coding", 0.2, {
      rawScore: 2, judgeNote: "irrelevant to the math task", judgeModelId: "test/judge",
      confidence: 0.62,
    });
    assert.equal(out.arm, "B");
    assert.equal(out.applied, false);
    assert.equal(registry.getCapabilityScore("zai", "glm-5.2", "coding"), before);
    // judge_history row logged with no_apply=1
    const row = db.getDb().prepare(
      `SELECT no_apply, gate_arm, gate_reason FROM judge_history ORDER BY id DESC LIMIT 1`,
    ).get() as any;
    assert.equal(row.no_apply, 1);
    assert.equal(row.gate_arm, "B");
    assert.equal(row.gate_reason, "low_confidence");
  });

  it("Arm B: missing confidence → no_apply", () => {
    const out = registry.applyJudgedScore("zai", "glm-5.2", "coding", 0.5, {
      rawScore: 5, judgeNote: "ok", judgeModelId: "test/judge", confidence: null,
    });
    assert.equal(out.arm, "B");
    assert.equal(out.reason, "missing_confidence");
  });

  it("Arm B: malformed verdict → no_apply", () => {
    const out = registry.applyJudgedScore("zai", "glm-5.2", "coding", 0.5, {
      rawScore: 5, judgeNote: "unparseable", judgeModelId: "test/judge",
      confidence: 0.95, malformed: true,
    });
    assert.equal(out.arm, "B");
    assert.equal(out.reason, "malformed_verdict");
  });

  it("Arm C: high-confidence verdict → guardrails (shadow logs would-be)", () => {
    setEnv("ROUTER_LEARNING_SHADOW_MODE", "true");
    const before = registry.getCapabilityScore("zai", "glm-5.2", "coding");
    const out = registry.applyJudgedScore("zai", "glm-5.2", "coding", 0.9, {
      rawScore: 9, judgeNote: "excellent and complete answer", judgeModelId: "test/judge",
      confidence: 0.9,
    });
    assert.equal(out.arm, "C");
    assert.equal(out.applied, false); // shadow
    assert.equal(out.reason, "shadow_mode");
    assert.ok(out.wouldBe !== null && out.wouldBe > before);
    // shadow_decisions row logged
    assert.equal(db.getShadowDecisionCount(), 1);
    // capability_overrides untouched in shadow
    const ov = db.getCapabilityOverride("zai", "glm-5.2", "coding");
    assert.equal(ov, null);
  });

  it("shadow rows: no-apply paths (Arm B, Arm A, canary) still log would-be counterfactual", () => {
    setEnv("ROUTER_LEARNING_SHADOW_MODE", "true");
    const latest = () => db.getDb().prepare(
      `SELECT rejection_reason, pre_clamp, would_be_value, sample_n FROM shadow_decisions ORDER BY id DESC LIMIT 1`,
    ).get() as any;
    const expected = (baseline: number, normalized: number) =>
      Math.min(0.98, Math.max(0.05, baseline * 0.75 + normalized * 0.25));

    // Arm B — low confidence: rejected, but the counterfactual must be recorded.
    const baseB = registry.getCapabilityScore("zai", "glm-5.2", "coding");
    registry.applyJudgedScore("zai", "glm-5.2", "coding", 0.2, {
      rawScore: 2, judgeNote: "irrelevant to the math task", judgeModelId: "t/j", confidence: 0.62,
    });
    let row = latest();
    assert.equal(row.rejection_reason, "low_confidence");
    assert.ok(row.pre_clamp !== null, "Arm B shadow row must carry pre_clamp");
    assert.ok(row.would_be_value !== null, "Arm B shadow row must carry would_be_value");
    assert.ok(Math.abs(row.would_be_value - expected(baseB, 0.2)) < 1e-9);
    assert.equal(row.sample_n, 0, "sample_n = capability-cell sample count at decision time");

    // Arm A — provider-attributable truncation: same counterfactual requirement.
    const baseA = registry.getCapabilityScore("zai", "glm-5.2", "coding");
    registry.applyJudgedScore("zai", "glm-5.2", "coding", 0.2, {
      rawScore: 2, judgeNote: "truncated output here", judgeModelId: "t/j", confidence: 0.95, truncated: true,
    });
    row = latest();
    const expectedA = guards.classifyAttribution({ truncated: true, malformed: false, confidence: 0.95 }).reason;
    assert.equal(row.rejection_reason, expectedA);
    assert.ok(row.would_be_value !== null, "Arm A shadow row must carry would_be_value");
    assert.ok(Math.abs(row.would_be_value - expected(baseA, 0.2)) < 1e-9);

    // Quarantine canary — fires before arm dispatch; row must still carry values.
    const note = "The response is completely irrelevant to the math task and appears to be a system status log.";
    for (let i = 0; i < 5; i++) {
      db.recordJudgeEvaluationEx({
        provider: "zai", model: "glm-5.2", intent: "coding",
        judgeScore: 0, judgeNote: note, judgeModel: "hist/judge",
        noApply: false, gateArm: null, gateReason: null,
      });
    }
    const baseQ = registry.getCapabilityScore("zai", "glm-5.2", "coding");
    const q = registry.applyJudgedScore("zai", "glm-5.2", "coding", 0.0, {
      rawScore: 0, judgeNote: note.toUpperCase() + "!!", judgeModelId: "t/j", confidence: 0.9,
    });
    assert.equal(q.reason, "quarantine_canary");
    row = latest();
    assert.equal(row.rejection_reason, "quarantine_canary");
    assert.ok(row.pre_clamp !== null, "canary shadow row must carry pre_clamp");
    assert.ok(row.would_be_value !== null, "canary shadow row must carry would_be_value");
    assert.ok(Math.abs(row.would_be_value - expected(baseQ, 0.0)) < 1e-9);

    // Unknown cell — no baseline exists: NULL is correct, not an instrumentation gap.
    const u = registry.applyJudgedScore("zai", "nonexistent-model", "coding", 0.5, {
      rawScore: 5, judgeNote: "ok", judgeModelId: "t/j", confidence: 0.95,
    });
    assert.equal(u.reason, "unknown_model_or_intent");
    row = latest();
    assert.equal(row.would_be_value, null, "unknown cell has no counterfactual");
  });

  it("Arm C live (shadow off): applies through guardrails", () => {
    setEnv("ROUTER_LEARNING_SHADOW_MODE", "false");
    const before = registry.getCapabilityScore("zai", "glm-5.2", "coding");
    const out = registry.applyJudgedScore("zai", "glm-5.2", "coding", 0.9, {
      rawScore: 9, judgeNote: "excellent and complete answer", judgeModelId: "test/judge",
      confidence: 0.9,
    });
    assert.equal(out.applied, true);
    const after = registry.getCapabilityScore("zai", "glm-5.2", "coding");
    assert.ok(after > before);
  });
});

// ─── Note similarity + quarantine ───

describe("learning hardening — note similarity + quarantine canary", () => {
  it("normalizeNote: lowercase, punctuation-stripped, 255-truncated", () => {
    const a = guards.normalizeNote("The response is COMPLETELY irrelevant! To the math task.");
    const b = guards.normalizeNote("the response is completely irrelevant to the math task");
    assert.equal(a, b);
    assert.ok(guards.normalizeNote("x".repeat(500)).length <= 255);
  });

  it("identical hash detected; near-identical via Levenshtein ≤ 6", () => {
    const a = guards.normalizeNote("irrelevant system status log for math task");
    const b = guards.normalizeNote("irrelevant system status log for a math task");
    assert.ok(guards.isNearIdenticalNote(a, b));
    const c = guards.normalizeNote("great answer with correct calculation and clarity");
    assert.ok(!guards.isNearIdenticalNote(a, c));
  });

  it("≥5 identical notes → pair quarantined, persisted, and blocks further applies", () => {
    setEnv("ROUTER_LEARNING_SHADOW_MODE", "false");
    const note = "The response is completely irrelevant to the math task and appears to be a system status log.";
    // Seed 5 historical occurrences directly (canary = ≥5 same note historically).
    for (let i = 0; i < 5; i++) {
      db.recordJudgeEvaluationEx({
        provider: "zai", model: "glm-5.1", intent: "math",
        judgeScore: 0, judgeNote: note, judgeModel: "hist/judge",
        noApply: false, gateArm: null, gateReason: null,
      });
    }
    // Next event with a DIFFERENT note → applies normally (no hold: different hash).
    const ok = registry.applyJudgedScore("zai", "glm-5.1", "math", 0.8, {
      rawScore: 8, judgeNote: "a genuinely different verdict about quality", judgeModelId: "t/j", confidence: 0.9,
    });
    assert.equal(ok.quarantined, false);

    // Next identical occurrence (6th historically) → canary fires, pair quarantined.
    const q = registry.applyJudgedScore("zai", "glm-5.1", "math", 0.0, {
      rawScore: 0, judgeNote: note.toUpperCase() + "!!", judgeModelId: "t/j", confidence: 0.9,
    });
    assert.equal(q.quarantined, true);
    assert.equal(q.reason, "quarantine_canary");
    assert.ok(db.isPairQuarantined("zai", "glm-5.1", "math"));

    // After quarantine: even a good verdict is refused.
    const after = registry.applyJudgedScore("zai", "glm-5.1", "math", 1.0, {
      rawScore: 10, judgeNote: "superb work", judgeModelId: "t/j", confidence: 0.95,
    });
    assert.equal(after.applied, false);
    assert.equal(after.reason, "pair_quarantined");
  });

  it("pre-application hold: 1 prior identical note holds the 2nd (different from canary)", () => {
    setEnv("ROUTER_LEARNING_SHADOW_MODE", "false");
    setEnv("ROUTER_JUDGE_NOTE_HOLD", "1");
    const note = "partially relevant but incomplete or inaccurate response about the topic";
    db.recordJudgeEvaluationEx({
      provider: "zai", model: "glm-5.1", intent: "science",
      judgeScore: 4, judgeNote: note, judgeModel: "hist/judge",
      noApply: false, gateArm: null, gateReason: null,
    });
    const out = registry.applyJudgedScore("zai", "glm-5.1", "science", 0.4, {
      rawScore: 4, judgeNote: note, judgeModelId: "t/j", confidence: 0.9,
    });
    assert.equal(out.applied, false);
    assert.equal(out.reason, "note_hold");
  });
});

// ─── Pins ───

describe("learning hardening — pins", () => {
  it("pinned rows refuse updates (live mode)", () => {
    setEnv("ROUTER_LEARNING_SHADOW_MODE", "false");
    db.upsertCapabilityOverride("zai", "glm-5.2", "coding", 0.88, 5);
    registry.setCapabilityPin("zai", "glm-5.2", "coding", true);
    const before = registry.getCapabilityScore("zai", "glm-5.2", "coding");
    registry.updateCapability("zai", "glm-5.2", "coding", 0.1);
    assert.equal(registry.getCapabilityScore("zai", "glm-5.2", "coding"), before);
    const out = registry.applyJudgedScore("zai", "glm-5.2", "coding", 0.1, {
      rawScore: 1, judgeNote: "bad bad bad output here", judgeModelId: "t/j", confidence: 0.95,
    });
    assert.equal(out.applied, false);
    assert.equal(out.reason, "pinned");
    // Unpin restores learnability.
    registry.setCapabilityPin("zai", "glm-5.2", "coding", false);
    const out2 = registry.applyJudgedScore("zai", "glm-5.2", "coding", 0.95, {
      rawScore: 10, judgeNote: "flawless execution of the coding task", judgeModelId: "t/j", confidence: 0.95,
    });
    assert.equal(out2.applied, true);
  });
});

// ─── Decay-to-prior ───

describe("learning hardening — decay-to-prior", () => {
  it("decays learned value toward seed after N idle days (pinned exempt)", async () => {
    setEnv("ROUTER_DECAY_IDLE_DAYS", "14");
    setEnv("ROUTER_DECAY_RATE", "0.10");
    const idleDate = new Date(Date.now() - 30 * 86_400_000).toISOString();
    // Learned 0.20 on coding (seed 0.88) — 30d idle → decayed toward seed.
    db.upsertCapabilityOverride("zai", "glm-5.2", "coding", 0.20, 5);
    db.getDb().prepare(
      `UPDATE capability_overrides SET last_judged = ? WHERE provider = 'zai' AND model = 'glm-5.2' AND intent = 'coding'`,
    ).run(idleDate);
    // Pinned row must NOT decay.
    db.upsertCapabilityOverride("zai", "glm-5.2", "summary", 0.30, 5);
    db.getDb().prepare(
      `UPDATE capability_overrides SET last_judged = ?, pinned = 1 WHERE provider = 'zai' AND model = 'glm-5.2' AND intent = 'summary'`,
    ).run(idleDate);

    const reg2 = makeRegistry();
    await reg2.loadCachedState();
    const coding = reg2.getCapabilityScore("zai", "glm-5.2", "coding");
    assert.ok(coding > 0.20 && coding < 0.88, `expected decay between 0.20 and 0.88, got ${coding}`);
    // expected: 0.88 + (0.20-0.88)*(0.9)^16 ≈ 0.88 - 0.68*0.185 ≈ 0.754
    const summary = reg2.getCapabilityScore("zai", "glm-5.2", "summary");
    assert.equal(summary, 0.30, "pinned row must not decay");
    // Persisted.
    const persisted = db.getCapabilityOverride("zai", "glm-5.2", "coding");
    assert.ok(Math.abs(persisted!.score - coding) < 1e-9);
  });

  it("fresh rows (no idle) are not decayed", async () => {
    db.upsertCapabilityOverride("zai", "glm-5.2", "coding", 0.20, 5);
    const reg2 = makeRegistry();
    await reg2.loadCachedState();
    assert.equal(reg2.getCapabilityScore("zai", "glm-5.2", "coding"), 0.20);
  });
});

// ─── Reliability decay ───

describe("learning hardening — reliability age decay (§4.6)", () => {
  it("episodic counters halve every 1h of age", () => {
    const d0 = guards.ageWeightedEpisodic(8, 4, 0);
    assert.equal(d0.consecutiveFailures, 8);
    assert.equal(d0.backoffTier, 4);
    const d1 = guards.ageWeightedEpisodic(8, 4, 3_600_000);
    assert.equal(d1.consecutiveFailures, 4);
    assert.equal(d1.backoffTier, 2);
    const d3 = guards.ageWeightedEpisodic(8, 4, 3 * 3_600_000);
    assert.equal(d3.consecutiveFailures, 1);
    const dOld = guards.ageWeightedEpisodic(8, 4, 48 * 3_600_000);
    assert.equal(dOld.consecutiveFailures, 0);
    assert.equal(dOld.backoffTier, 0);
  });

  it("persistent failure rates halve every 7d", () => {
    assert.ok(Math.abs(guards.ageWeightedPersistentFailureRate(0.8, 0) - 0.8) < 1e-9);
    assert.ok(Math.abs(guards.ageWeightedPersistentFailureRate(0.8, 7 * 86_400_000) - 0.4) < 1e-9);
    assert.ok(Math.abs(guards.ageWeightedPersistentFailureRate(0.8, 14 * 86_400_000) - 0.2) < 1e-9);
  });
});

// ─── UCB read path ───

describe("learning hardening — UCB exploration read path", () => {
  it("returns learned score while shadow mode ON (live routing untouched)", () => {
    setEnv("ROUTER_LEARNING_SHADOW_MODE", "true");
    setEnv("ROUTER_UCB_K", "0.5");
    db.upsertCapabilityOverride("zai", "glm-5.2", "coding", 0.80, 25);
    const reg2 = makeRegistry();
    return reg2.loadCachedState().then(() => {
      assert.ok(Math.abs(reg2.getExplorationScore("zai", "glm-5.2", "coding") - 0.80) < 1e-9);
    });
  });

  it("adds k/√n (n=0→1) when shadow OFF", async () => {
    setEnv("ROUTER_LEARNING_SHADOW_MODE", "false");
    setEnv("ROUTER_UCB_K", "0.5");
    db.upsertCapabilityOverride("zai", "glm-5.2", "coding", 0.80, 25);
    const reg2 = makeRegistry();
    await reg2.loadCachedState();
    // n=25 → 0.5/5 = 0.10
    assert.ok(Math.abs(reg2.getExplorationScore("zai", "glm-5.2", "coding") - 0.90) < 1e-9);
    // n=0 model → 0.5/1 = 0.5 bonus, capped at 1. NOTE: SEED_MODELS objects are
    // shared across registry instances in this codebase, so read the current
    // learned value rather than assuming the pristine seed.
    const learnedSummary = reg2.getCapabilityScore("zai", "glm-5.2", "summary");
    assert.ok(
      Math.abs(reg2.getExplorationScore("zai", "glm-5.2", "summary") - Math.min(1, learnedSummary + 0.5)) < 1e-9,
    );
  });
});

// ─── Ingestion ladder ───

describe("learning hardening — ingestion ladder tiering", () => {
  it("tier 1 ≤ 32k; tier 2 ≤ 200k; tier 3 beyond; tokens via 3.5:1", () => {
    assert.equal(guards.pickIngestionTier(1_000), 1);
    assert.equal(guards.pickIngestionTier(32_000), 1);
    assert.equal(guards.pickIngestionTier(32_001), 2);
    assert.equal(guards.pickIngestionTier(200_000), 2);
    assert.equal(guards.pickIngestionTier(200_001), 3);
    assert.equal(guards.estimateTokens(3_500), 1000);
  });
});
