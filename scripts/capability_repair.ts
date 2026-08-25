// scripts/capability_repair.ts — Phase-1 data repair (LEARNING_LOOP_DESIGN.md §4.3)
//
// Actions (default mode):
//   1. Archive ALL capability_overrides rows to capability_overrides_archive
//      with a batch timestamp (reversibility).
//   2. Reset poisoned cells to seed: zai/glm-5.1 math + zai/glm-5.2 math
//      (misclassification casualties) — rows DELETED so the registry falls
//      back to benchmark seeds (glm-5.1 math=0.84, glm-5.2 math=0.90).
//   3. Keep healthy learned rows (no action).
//   4. Reseed the generic-0.5 OpenRouter "inferred" placeholders: neutral
//      score 0.5, sample_count 0 (n=0 → full UCB exploration), for the five
//      routed OpenRouter models that are NOT benchmark-seeded. Seeded models
//      (cohere/north-mini-code:free, deepseek/deepseek-v4-flash) keep their
//      benchmark seed values — no placeholder rows.
//   5. Clear any quarantined_pairs rows for repaired pairs; assert none remain.
//
// Idempotent: a second run produces the identical end state (extra archive
// batches are appended — the archive is append-only audit history).
//
// --restore  : restore the LATEST archive batch into capability_overrides
//              (deletes current rows, reinserts the archived snapshot).
// --db PATH  : database path (default data/cognitive-router.db).
//
// Tested once against a temp DB (tests/learning_repair.test.ts).

import { DBService } from "../src/db_service.js";
import { logger } from "../src/logger.js";

const args = process.argv.slice(2);
const restore = args.includes("--restore");
const dbPath = args.includes("--db") ? args[args.indexOf("--db") + 1] : "data/cognitive-router.db";

/** Poisoned cells → reset to benchmark seed (row deleted = seed applies). */
const RESET_TO_SEED: Array<{ provider: string; model: string; intent: string; seed: number }> = [
  { provider: "zai", model: "glm-5.1", intent: "math", seed: 0.84 },
  { provider: "zai", model: "glm-5.2", intent: "math", seed: 0.90 },
];

/** The five generic-0.5 OpenRouter "inferred" placeholders. The two models
 *  with benchmark seeds are excluded (seed = fresh benchmark already). */
const OPENROUTER_PLACEHOLDERS = [
  "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
  "openrouter/qwen/qwen3-coder:free",
  "openrouter/google/gemma-4-26b-a4b-it:free",
];

const PLACEHOLDER_INTENTS = [
  "coding", "reasoning", "creative", "math", "analysis",
  "conversation", "retrieval", "science", "business", "summary",
];

const db = new DBService(dbPath);
db.initializeSchema();

try {
  if (restore) {
    // ─── Restore mode ───
    const latest = (db.getDb().prepare(
      `SELECT archived_at, COUNT(*) AS c FROM capability_overrides_archive
       GROUP BY archived_at ORDER BY archived_at DESC LIMIT 1`,
    ).get() as { archived_at: string; c: number } | undefined);
    if (!latest) {
      console.log("No archive batches found — nothing to restore.");
      process.exit(0);
    }
    const rows = db.getDb().prepare(
      `SELECT provider, model, intent, score, sample_count, last_judged, pinned
       FROM capability_overrides_archive WHERE archived_at = ?`,
    ).all(latest.archived_at) as Array<{ provider: string; model: string; intent: string; score: number; sample_count: number; last_judged: string; pinned: number }>;

    const tx = db.getDb().transaction(() => {
      db.getDb().prepare(`DELETE FROM capability_overrides`).run();
      const ins = db.getDb().prepare(
        `INSERT INTO capability_overrides (provider, model, intent, score, sample_count, last_judged, pinned)
         VALUES (?, ?, ?, ?, ?, ?, ?)`,
      );
      for (const r of rows) {
        ins.run(r.provider, r.model, r.intent, r.score, r.sample_count, r.last_judged, r.pinned);
      }
    });
    tx();
    console.log(`Restored ${rows.length} rows from archive batch ${latest.archived_at}.`);
    process.exit(0);
  }

  // ─── Repair mode ───
  const batchTs = new Date().toISOString();
  const archived = db.archiveAllCapabilityOverrides(batchTs);
  console.log(`[1/4] Archived ${archived} capability_overrides rows (batch ${batchTs}).`);

  // Reset poisoned cells to seed.
  const del = db.getDb().prepare(
    `DELETE FROM capability_overrides WHERE provider = ? AND model = ? AND intent = ?`,
  );
  let resetCount = 0;
  for (const cell of RESET_TO_SEED) {
    const info = del.run(cell.provider, cell.model, cell.intent);
    resetCount += info.changes;
    console.log(`[2/4] Reset ${cell.provider}/${cell.model} [${cell.intent}] → seed ${cell.seed} (${info.changes} row removed).`);
  }

  // Reseed generic-0.5 OpenRouter inferred placeholders (idempotent upsert).
  const upsert = db.getDb().prepare(
    `INSERT INTO capability_overrides (provider, model, intent, score, sample_count, last_judged, pinned)
     VALUES (?, ?, ?, 0.5, 0, ?, 0)
     ON CONFLICT(provider, model, intent) DO UPDATE SET
       score = 0.5, sample_count = 0, last_judged = excluded.last_judged`,
  );
  let seeded = 0;
  for (const full of OPENROUTER_PLACEHOLDERS) {
    const slash = full.indexOf("/");
    const provider = full.slice(0, slash);
    const model = full.slice(slash + 1);
    for (const intent of PLACEHOLDER_INTENTS) {
      upsert.run(provider, model, intent, batchTs);
      seeded++;
    }
  }
  console.log(`[3/4] Reseeded ${OPENROUTER_PLACEHOLDERS.length} OpenRouter placeholder models × ${PLACEHOLDER_INTENTS.length} intents at 0.5 (n=0) = ${seeded} rows.`);

  // Clear quarantines on repaired pairs; assert none remain.
  const clearQ = db.getDb().prepare(
    `DELETE FROM quarantined_pairs WHERE provider = ? AND model = ? AND intent = ?`,
  );
  let cleared = 0;
  for (const cell of RESET_TO_SEED) {
    cleared += clearQ.run(cell.provider, cell.model, cell.intent).changes;
  }
  for (const full of OPENROUTER_PLACEHOLDERS) {
    const slash = full.indexOf("/");
    const provider = full.slice(0, slash);
    const model = full.slice(slash + 1);
    for (const intent of PLACEHOLDER_INTENTS) {
      cleared += clearQ.run(provider, model, intent).changes;
    }
  }
  console.log(`[4/4] Cleared ${cleared} quarantine rows on repaired pairs.`);

  // Assertions.
  const assertNotQuarantined = (provider: string, model: string, intent: string) => {
    if (db.isPairQuarantined(provider, model, intent)) {
      throw new Error(`ASSERT FAILED: ${provider}/${model} [${intent}] still quarantined after repair`);
    }
  };
  for (const cell of RESET_TO_SEED) assertNotQuarantined(cell.provider, cell.model, cell.intent);
  console.log("Assert: no quarantine rows remain on repaired pairs. OK");

  const finalCount = (db.getDb().prepare(`SELECT COUNT(*) AS c FROM capability_overrides`).get() as { c: number }).c;
  console.log(`Repair complete: ${finalCount} capability_overrides rows remain (healthy rows kept).`);
  console.log(`Rollback: npx tsx scripts/capability_repair.ts --restore  (restores batch ${batchTs})`);
} finally {
  db.close();
}
