// scripts/verify_shadow_gates.ts — Phase-1 shadow-window acceptance gates
// (LEARNING_LOOP_DESIGN.md §4.5). Runnable on demand; exits non-zero on any
// failed gate. Gates, not calendar, decide the apply-flag flip.
//
//   Gate 1 — Misclassification replay: the historical status-log cluster is
//            replayed through the hardened gate (in-memory DB); required
//            outcome 100% no-apply.
//   Gate 2 — Healthy-row stability: zero unexpected quarantine alerts on
//            non-reseeded cells (checked against the LIVE db's
//            quarantined_pairs + shadow_decisions tables).
//   Gate 3 — Bounded drift: shadow would-be values on non-reseeded cells sit
//            within the RoC envelope of the live learned values.
//   Gate 4 — Coding detector: coding-intent divergence report (would-be vs
//            learned; flags systematic truncation-bias divergence).
//
// Usage: npx tsx scripts/verify_shadow_gates.ts [--db data/cognitive-router.db]

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DBService } from "../src/db_service.js";
import { ModelRegistry } from "../src/model_registry.js";
import { loadConfig } from "../src/config.js";
import { rocMax24h, rocMax7d, normalizeNote, noteHash } from "../src/learning_guards.js";

const args = process.argv.slice(2);
const dbPath = args.includes("--db") ? args[args.indexOf("--db") + 1] : "data/cognitive-router.db";

/** The historical status-log cluster (from judge_history: 40 zero-scores on
 *  glm-5.1 math with near-identical "system status log" notes; the fixed
 *  37-turn cluster + variants). Confidence band as measured live: 0.50–0.70. */
const CLUSTER_NOTE = "The response is completely irrelevant to the math task and appears to be a system status log.";
const CLUSTER_TURNS = 37;

let failures = 0;
function report(gate: string, pass: boolean, detail: string): void {
  console.log(`${pass ? "✅" : "❌"} ${gate}: ${detail}`);
  if (!pass) failures++;
}

process.env.ZAI_API_KEY = process.env.ZAI_API_KEY ?? ""; // keep discovery off
delete process.env.ZAI_API_KEY;

// ─── Gate 1: misclassification replay (in-memory temp DB) ───
{
  const tmp = mkdtempSync(join(tmpdir(), "gates-"));
  const tdb = new DBService(join(tmp, "gates.db"));
  tdb.initializeSchema();
  const config = loadConfig({
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
  const registry = new ModelRegistry(tdb, config);
  await registry.loadCachedState();

  // Replicate pre-hardening poisoned state.
  tdb.upsertCapabilityOverride("zai", "glm-5.1", "math", 0.00002, 40);
  // NOTE: gate 1 is a LIVE-mode proof (shadow off) — the replay must show the
  // gate itself blocks the cluster even when apply is enabled.
  process.env.ROUTER_LEARNING_SHADOW_MODE = "false";
  // Re-read env-dependent behavior through fresh calls:
  let applied = 0;
  let quarantined = false;
  for (let i = 0; i < CLUSTER_TURNS; i++) {
    const out = registry.applyJudgedScore("zai", "glm-5.1", "math", 0.0, {
      rawScore: 0,
      judgeNote: CLUSTER_NOTE,
      judgeModelId: "openrouter/qwen/qwen3-30b-a3b-instruct-2507",
      confidence: 0.50 + (i % 21) * 0.01,
    });
    if (out.applied) applied++;
    if (out.quarantined) quarantined = true;
  }
  delete process.env.ROUTER_LEARNING_SHADOW_MODE;
  report("Gate 1 (misclassification replay)",
    applied === 0 && quarantined,
    `${applied}/${CLUSTER_TURNS} applications (need 0), canary ${quarantined ? "fired" : "did NOT fire"} (need fired)`);
  tdb.close();
  rmSync(tmp, { recursive: true, force: true });
}

// ─── Gates 2-4: live shadow_decisions analysis ───
{
  const db = new DBService(dbPath);
  db.initializeSchema(); // idempotent; ensures v4 tables exist

  // Gate 2: zero unexpected quarantines on non-reseeded cells.
  const RESEEDED = new Set([
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/qwen/qwen3-coder:free",
    "openrouter/google/gemma-4-26b-a4b-it:free",
    "zai/glm-5.1", "zai/glm-5.2", // math reset (pair-level check below)
  ]);
  const quarantines = db.getQuarantinedPairs();
  const unexpected = quarantines.filter((q) => {
    const pair = `${q.provider}/${q.model}`;
    if (pair === "zai/glm-5.1" || pair === "zai/glm-5.2") {
      return q.intent !== "math"; // reseeded cells = math only
    }
    return !RESEEDED.has(pair);
  });
  report("Gate 2 (healthy-row stability)",
    unexpected.length === 0,
    unexpected.length === 0
      ? `${quarantines.length} quarantine row(s), all on reseeded/poisoned cells`
      : `UNEXPECTED quarantines: ${unexpected.map((q) => `${q.provider}/${q.model}[${q.intent}]`).join(", ")}`);

  // Gate 3: would-be values within the RoC envelope of live learned values.
  const shadows = db.getDb().prepare(
    `SELECT provider, model, intent, would_be_value, sample_n, rejection_reason
     FROM shadow_decisions WHERE would_be_value IS NOT NULL AND rejection_reason = 'shadow_mode'`,
  ).all() as Array<{ provider: string; model: string; intent: string; would_be_value: number; sample_n: number | null; rejection_reason: string }>;

  let violations = 0;
  let checked = 0;
  for (const s of shadows) {
    const live = db.getCapabilityOverride(s.provider, s.model, s.intent);
    if (!live) continue;
    const drift = Math.abs(s.would_be_value - live.score);
    checked++;
    if (drift > rocMax24h() + 1e-9) violations++;
  }
  report("Gate 3 (bounded drift)",
    violations === 0,
    `${violations}/${checked} would-be values outside the 24h RoC envelope (Δmax=${rocMax24h()})`);

  // Gate 4: coding-intent divergence report.
  const coding = db.getDb().prepare(
    `SELECT provider, model,
            COUNT(*) AS n,
            AVG(would_be_value) AS avg_would_be,
            MIN(would_be_value) AS min_would,
            MAX(would_be_value) AS max_would
     FROM shadow_decisions
     WHERE intent = 'coding' AND would_be_value IS NOT NULL
     GROUP BY provider, model`,
  ).all() as Array<{ provider: string; model: string; n: number; avg_would_be: number; min_would: number; max_would: number }>;

  console.log("\n📊 Gate 4 (coding-intent divergence — de-truncation bias detector):");
  if (coding.length === 0) {
    console.log("   (no coding-intent shadow rows yet — needs live traffic; run again after the shadow window accrues)");
  } else {
    for (const c of coding) {
      const live = db.getCapabilityOverride(c.provider, c.model, "coding");
      const liveScore = live?.score ?? 0.5;
      const delta = c.avg_would_be - liveScore;
      const flag = delta > rocMax24h() ? " ⚠ DIVERGENT — investigate truncation bias (reset-and-relearn candidate)" : "";
      console.log(
        `   ${c.provider}/${c.model} [coding]: n=${c.n} avg-would-be=${c.avg_would_be.toFixed(3)} ` +
        `live=${liveScore.toFixed(3)} Δ=${delta >= 0 ? "+" : ""}${delta.toFixed(3)}${flag}`,
      );
    }
  }
  report("Gate 4 (coding detector)", true, `${coding.length} coding cell(s) reported (informational gate — divergence list above)`);

  // Bonus diagnostic: shadow volume.
  const total = db.getShadowDecisionCount();
  console.log(`\nShadow volume: ${total} decision row(s) logged.`);

  db.close();
}

console.log("");
if (failures > 0) {
  console.log(`❌ ${failures} gate(s) FAILED — do NOT flip apply; extend the shadow window and investigate.`);
  process.exit(1);
}
console.log("✅ All shadow gates passed — safe to flip ROUTER_LEARNING_SHADOW_MODE=false (human decision).");
