// tests/learning_replay.test.ts — Phase-1 integration tests
// 1) The 37-turn status-log poisoning cluster replayed through the hardened
//    gate: confidence band 0.50–0.70 (as observed live), identical judge notes,
//    score 0 — REQUIRED outcome: 100% no-apply (zero capability applications)
//    AND the quarantine canary fires. Proven in LIVE mode (shadow off).
// 2) Same replay in SHADOW mode (the deployment default): identical gating,
//    zero capability_overrides writes, shadow_decisions rows logged.
// 3) capability_repair.ts exercised against a temp DB (repair idempotent +
//    --restore round-trip).

import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DBService } from "../src/db_service.ts";
import { ModelRegistry } from "../src/model_registry.ts";
import { loadConfig, type CognitiveRouterConfig } from "../src/config.ts";

const ENV_BACKUP: Record<string, string | undefined> = {};
const ENV_KEYS = [
  "ROUTER_LEARNING_SHADOW_MODE", "ROUTER_ARM_B_MIN_CONFIDENCE",
  "ROUTER_JUDGE_NOTE_CANARY", "ROUTER_JUDGE_NOTE_HOLD",
  "ROUTER_ROC_MAX_24H", "ROUTER_ROC_MAX_7D", "ZAI_API_KEY",
];

let tmpDir: string;
let db: DBService;
let registry: ModelRegistry;
let config: CognitiveRouterConfig;

function setEnv(k: string, v: string | undefined) {
  if (v === undefined) delete process.env[k];
  else process.env[k] = v;
}

beforeEach(async () => {
  tmpDir = mkdtempSync(join(tmpdir(), "replay-"));
  for (const k of ENV_KEYS) ENV_BACKUP[k] = process.env[k];
  delete process.env.ZAI_API_KEY; // suppress discovery
  db = new DBService(join(tmpDir, "replay.db"));
  db.initializeSchema();
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
  registry = new ModelRegistry(db, config);
  await registry.loadCachedState();
});

afterEach(() => {
  db.close();
  rmSync(tmpDir, { recursive: true, force: true });
  for (const k of ENV_KEYS) setEnv(k, ENV_BACKUP[k]);
});

const POISON_NOTE = "The response is completely irrelevant to the math task and appears to be a system status log.";

describe("integration — 37-turn status-log poisoning replay (§4.5 gate 1)", () => {
  it("LIVE mode: 100% no-apply, zero applications, canary fires", () => {
    setEnv("ROUTER_LEARNING_SHADOW_MODE", "false");
    // Seed the pre-existing poisoned state exactly as the live DB had it.
    db.upsertCapabilityOverride("zai", "glm-5.1", "math", 0.00002, 40);

    const outcomes = [];
    for (let turn = 0; turn < 37; turn++) {
      // Confidence band as measured live: 0.50–0.70, all below the 0.75 arm-B bar.
      const confidence = 0.50 + (turn % 21) * 0.01;
      outcomes.push(registry.applyJudgedScore("zai", "glm-5.1", "math", 0.0, {
        rawScore: 0,
        judgeNote: POISON_NOTE,
        judgeModelId: "openrouter/qwen/qwen3-30b-a3b-instruct-2507",
        confidence,
      }));
    }

    // Gate 1 requirement: 100% no-apply.
    const applied = outcomes.filter((o) => o.applied);
    assert.equal(applied.length, 0, `expected zero applications, got ${applied.length}`);

    // Every gated event was arm-B (low confidence) or canary-quarantined.
    const arms = new Set(outcomes.map((o) => o.arm));
    assert.ok(arms.has("B"), "arm B must appear (low_confidence)");

    // Zero capability applications: the poisoned value never moved.
    const cell = db.getCapabilityOverride("zai", "glm-5.1", "math");
    assert.equal(cell!.score, 0.00002);
    assert.equal(cell!.sampleCount, 40, "sample count must not advance on no-apply events");

    // All 37 events logged to judge_history with no_apply=1.
    const rows = db.getDb().prepare(
      `SELECT no_apply, COUNT(*) AS c FROM judge_history WHERE provider='zai' AND model='glm-5.1' AND intent='math'`,
    ).get() as { no_apply: number; c: number };
    assert.equal(rows.no_apply, 1);
    assert.equal(rows.c, 37);

    // Canary fires: ≥5 identical notes → pair quarantined (persisted).
    assert.ok(db.isPairQuarantined("zai", "glm-5.1", "math"), "canary must quarantine the pair");
    const q = db.getQuarantinedPairs().find((p) => p.model === "glm-5.1" && p.intent === "math");
    assert.ok(q, "quarantine row persisted");
    assert.ok(q!.occurrences >= 5, `occurrences >= 5, got ${q!.occurrences}`);
  });

  it("SHADOW mode (deployment default): identical gating, zero writes, shadow rows logged", () => {
    setEnv("ROUTER_LEARNING_SHADOW_MODE", "true");
    db.upsertCapabilityOverride("zai", "glm-5.2", "math", 0.0901, 8);

    let quarantinedCount = 0;
    for (let turn = 0; turn < 37; turn++) {
      const out = registry.applyJudgedScore("zai", "glm-5.2", "math", 0.0, {
        rawScore: 0, judgeNote: POISON_NOTE, judgeModelId: "t/j",
        confidence: 0.50 + (turn % 21) * 0.01,
      });
      assert.equal(out.applied, false);
      if (out.quarantined) quarantinedCount++;
    }
    assert.ok(quarantinedCount > 0, "canary must fire during shadow replay");
    const cell = db.getCapabilityOverride("zai", "glm-5.2", "math");
    assert.equal(cell!.score, 0.0901, "shadow mode must never write capability_overrides");
    assert.equal(cell!.sampleCount, 8);

    // Shadow rows logged for gated events (metadata only).
    const shadowRows = db.getDb().prepare(
      `SELECT COUNT(*) AS c FROM shadow_decisions WHERE provider='zai' AND model='glm-5.2' AND intent='math'`,
    ).get() as { c: number };
    assert.ok(shadowRows.c >= 30, `expected ~37 shadow rows, got ${shadowRows.c}`);
    // ...and the table has NO prompt/response content columns at all.
    const cols = db.getDb().prepare(`PRAGMA table_info(shadow_decisions)`).all().map((c: any) => c.name);
    for (const forbidden of ["prompt", "response", "content", "note"]) {
      assert.ok(!cols.includes(forbidden), `shadow_decisions must not store ${forbidden}`);
    }
  });
});

describe("integration — capability repair script (§4.3) on temp DB", () => {
  it("repair: archives, resets poisoned cells, reseeds placeholders, idempotent", async () => {
    // Build a replica of the poisoned live state.
    db.upsertCapabilityOverride("zai", "glm-5.1", "math", 0.00002, 40);
    db.upsertCapabilityOverride("zai", "glm-5.2", "math", 0.0901, 8);
    db.upsertCapabilityOverride("zai", "glm-5.1", "coding", 0.58026, 207); // healthy — keep
    db.upsertCapabilityOverride("zai", "glm-5.1", "summary", 0.79454, 179); // healthy — keep
    db.quarantinePair("zai", "glm-5.1", "math", "deadbeef", "old poisoned note", 6);

    // Run the repair script via shell (execSync uses cmd on Windows) on the temp DB.
    const { execSync } = await import("node:child_process");
    const dbPath = join(tmpDir, "replay.db");
    // db handle must be released so the child process can write.
    db.close();
    const run = () => execSync(`npx tsx scripts/capability_repair.ts --db "${dbPath}"`, {
      encoding: "utf8", timeout: 120_000,
    });

    const out1 = run();
    assert.match(out1, /Archived 4 capability_overrides rows/);
    assert.match(out1, /Reset zai\/glm-5\.1 \[math\]/);
    assert.match(out1, /Reset zai\/glm-5\.2 \[math\]/);
    assert.match(out1, /no quarantine rows remain on repaired pairs/);

    // Reopen to verify.
    const verify = new DBService(dbPath);
    try {
      // Poisoned cells gone → registry falls back to seed.
      assert.equal(verify.getCapabilityOverride("zai", "glm-5.1", "math"), null);
      assert.equal(verify.getCapabilityOverride("zai", "glm-5.2", "math"), null);
      // Healthy rows kept.
      assert.equal(verify.getCapabilityOverride("zai", "glm-5.1", "coding")!.score, 0.58026);
      assert.equal(verify.getCapabilityOverride("zai", "glm-5.1", "summary")!.score, 0.79454);
      // Placeholders seeded at 0.5 n=0 (3 models × 10 intents).
      const ph = verify.getCapabilityOverride("openrouter", "qwen/qwen3-coder:free", "coding");
      assert.ok(ph, "placeholder row must exist");
      assert.equal(ph!.score, 0.5);
      assert.equal(ph!.sampleCount, 0);
      const phCount = (verify.getDb().prepare(
        `SELECT COUNT(*) AS c FROM capability_overrides WHERE provider='openrouter' AND score=0.5 AND sample_count=0`,
      ).get() as { c: number }).c;
      assert.equal(phCount, 30, "3 models × 10 intents");
      // Quarantine cleared on repaired pair.
      assert.ok(!verify.isPairQuarantined("zai", "glm-5.1", "math"));
    } finally {
      verify.close();
    }

    // Idempotent: second run → identical end state (extra archive batch appended).
    const out2 = run();
    assert.match(out2, /Archived 32 capability_overrides rows/); // 2 healthy + 30 placeholders
    const verify2 = new DBService(dbPath);
    try {
      assert.equal(verify2.getCapabilityOverride("zai", "glm-5.1", "coding")!.score, 0.58026);
      assert.equal(verify2.getCapabilityOverride("zai", "glm-5.1", "math"), null);
      const c = (verify2.getDb().prepare(
        `SELECT COUNT(*) AS c FROM capability_overrides WHERE score=0.5 AND sample_count=0 AND provider='openrouter'`,
      ).get() as any).c;
      assert.equal(c, 30);
    } finally {
      verify2.close();
    }
  });

  it("--restore round-trips the latest archive batch into capability_overrides", async () => {
    // Fresh replica of the poisoned live state.
    db.upsertCapabilityOverride("zai", "glm-5.1", "math", 0.00002, 40);
    db.upsertCapabilityOverride("zai", "glm-5.2", "math", 0.0901, 8);
    db.upsertCapabilityOverride("zai", "glm-5.1", "coding", 0.58026, 207);

    const { execSync } = await import("node:child_process");
    const dbPath = join(tmpDir, "replay.db");
    db.close();
    execSync(`npx tsx scripts/capability_repair.ts --db "${dbPath}"`, { encoding: "utf8", timeout: 120_000 });

    // Poisoned rows gone after repair...
    const v1 = new DBService(dbPath);
    assert.equal(v1.getCapabilityOverride("zai", "glm-5.1", "math"), null);
    v1.close();

    // ...and --restore reinstates the pre-repair archive batch.
    const outRestore = execSync(`npx tsx scripts/capability_repair.ts --db "${dbPath}" --restore`, {
      encoding: "utf8", timeout: 120_000,
    });
    assert.match(outRestore, /Restored 3 rows from archive batch/);
    const v2 = new DBService(dbPath);
    const mathCell = v2.getCapabilityOverride("zai", "glm-5.1", "math");
    assert.ok(mathCell, "restored poisoned row returns");
    assert.equal(mathCell!.score, 0.00002);
    assert.equal(mathCell!.sampleCount, 40);
    assert.equal(v2.getCapabilityOverride("zai", "glm-5.1", "coding")!.score, 0.58026);
    v2.close();
  });
});
