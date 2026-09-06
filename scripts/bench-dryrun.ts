// scripts/bench-dryrun.ts - Sanity layer 2: FREE end-to-end ladder run on
// local ollama. Ground truth: a 7b coding model must outrank its 1.5b
// sibling when judged by a third family (gemma4 — the same local fallback
// judge production uses).
//
// Exercises the REAL production paths: makeModelCaller (PINNED_DECODE,
// endpoint resolution, timeouts), makeJudgeCaller (A/B prompt, verdict
// parsing, family recusal), comparePair (k rounds x swapped asks, raw
// persistence), insertIntoLadder (binary insert), rebuildLadder (BT ranks).
// Writes only to a throwaway sqlite file — never the production DB.
//
// Usage:
//   npm run bench:dryrun
//   npm run bench:dryrun -- --a qwen2.5-coder:7b --b qwen2.5-coder:1.5b \
//       --judge gemma4:latest --intent coding --rounds 5
//
// Expect ~15-25 min on 12GB VRAM (models swap in/out per call; ollama
// keeps them cached ~5m — a cold timeout on first run usually passes on
// an immediate rerun). Exit 0 = ground truth reproduced, 1 = it did not.

import { unlinkSync, existsSync } from "node:fs";
import { join } from "node:path";
import { DBService } from "../src/db_service.js";
import {
  comparePair,
  encodeKey,
  insertIntoLadder,
  parseOllamaQuant,
  rebuildLadder,
  type BenchIntent,
  type BenchModelKey,
} from "../src/benchmark_ladder.js";
import { makeJudgeCaller, makeModelCaller, mapResolver } from "../src/benchmark_wiring.js";

// ---------- args ----------

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const A_TAG = arg("a", "qwen2.5-coder:7b");
const B_TAG = arg("b", "qwen2.5-coder:1.5b");
const JUDGE_TAG = arg("judge", "gemma4:latest");
const INTENT = arg("intent", "coding") as BenchIntent;
const ROUNDS = Number(arg("rounds", "5"));
const ALLOW_REMOTE_JUDGE = process.argv.includes("--allow-remote-judge");

// Spend guard: the dry-run must be free. Refuse a remote judge unless
// explicitly allowed (that would spend tokens on a test).
if (
  !ALLOW_REMOTE_JUDGE &&
  process.env.ROUTER_BENCH_JUDGE_PROVIDER &&
  process.env.ROUTER_BENCH_JUDGE_PROVIDER !== "ollama"
) {
  console.error(
    `refusing to run: ROUTER_BENCH_JUDGE_PROVIDER=${process.env.ROUTER_BENCH_JUDGE_PROVIDER} ` +
      `would spend money. Use --allow-remote-judge to override.`,
  );
  process.exit(2);
}

// Force the judge onto local ollama for this process (getJudgeCandidates
// reads env lazily per call, so setting it here wins over .env values).
process.env.ROUTER_BENCH_JUDGE_PROVIDER = "ollama";
process.env.ROUTER_BENCH_JUDGE_MODEL = JUDGE_TAG;

// ---------- preflight ----------

async function preflight(): Promise<void> {
  const resp = await fetch("http://127.0.0.1:11434/api/tags", {
    signal: AbortSignal.timeout(10_000),
  });
  if (!resp.ok) throw new Error(`ollama /api/tags HTTP ${resp.status}`);
  const tags = ((await resp.json()) as any).models?.map((m: any) => m.name) ?? [];
  for (const need of [A_TAG, B_TAG, JUDGE_TAG]) {
    if (!tags.includes(need)) {
      throw new Error(`ollama does not have "${need}". Pull it first: ollama pull ${need}`);
    }
  }
}

// ---------- run ----------

function keyFor(tag: string): BenchModelKey {
  const { model, quant } = parseOllamaQuant(tag);
  return { model, quant, effort: "none" };
}

async function main(): Promise<number> {
  const t0 = Date.now();
  console.log(`bench dry-run: ${A_TAG} vs ${B_TAG}, judge ${JUDGE_TAG}, intent ${INTENT}, ${ROUNDS} rounds`);
  console.log(`(models swap in/out of VRAM between calls — expect ~15-25 min on 12GB)\n`);
  await preflight();

  const dbPath = join(process.cwd(), "data", "bench-dryrun.db");
  if (existsSync(dbPath)) unlinkSync(dbPath);
  const db = new DBService(dbPath);

  const keyA = keyFor(A_TAG);
  const keyB = keyFor(B_TAG);
  const endpointMap = {
    [encodeKey(keyA)]: { provider: "ollama", model: A_TAG },
    [encodeKey(keyB)]: { provider: "ollama", model: B_TAG },
  };
  const deps = {
    db,
    callModel: makeModelCaller(mapResolver(endpointMap)),
    judge: makeJudgeCaller(),
    rounds: ROUNDS,
  };

  // 1. Insert the presumed-weaker model into an empty ladder.
  console.log(`[1/3] inserting ${B_TAG} into empty ladder...`);
  const first = await insertIntoLadder(deps, [], keyB, INTENT);
  console.log(`      index ${first.index}, ${first.comparisons.length} comparisons\n`);

  // 2. Insert the presumed-stronger model — must land above (index 0).
  console.log(`[2/3] inserting ${A_TAG} above ${B_TAG}...`);
  const second = await insertIntoLadder(deps, [encodeKey(keyB)], keyA, INTENT);
  console.log(`      index ${second.index} (0 = above), cached=${second.comparisons.map((c) => c.cached).join(",")}\n`);

  // 3. Rebuild from raw rows; verify the ground-truth ordering.
  console.log(`[3/3] rebuilding ladder from raw verdicts...`);
  const ladder = rebuildLadder(db, INTENT);
  const raw = db.getAllBenchmarkVerdicts(INTENT);
  console.log(`\nraw asks (${raw.length}):`);
  for (const r of raw) {
    console.log(
      `  r${r.round} swap${r.swapOrder}: ${r.modelA.slice(0, 40)} vs ${r.modelB.slice(0, 40)} -> ${r.verdict}`,
    );
  }
  console.log(`\nladder:`);
  for (const e of ladder) {
    console.log(`  #${e.rank} (BT ${e.strength.toFixed(3)}) ${e.modelKey}`);
  }

  const strongKey = encodeKey(keyA);
  const weakKey = encodeKey(keyB);
  const pass =
    ladder[0]?.modelKey === strongKey &&
    ladder.find((e) => e.modelKey === weakKey) !== undefined &&
    (ladder.find((e) => e.modelKey === weakKey)?.rank ?? 99) >
      (ladder.find((e) => e.modelKey === strongKey)?.rank ?? 0);

  const mins = ((Date.now() - t0) / 60_000).toFixed(1);
  console.log(`\n${pass ? "PASS" : "FAIL"}: ${A_TAG} ranks above ${B_TAG} on ${INTENT} (${mins} min)`);
  return pass ? 0 : 1;
}

main()
  .then((code) => process.exit(code))
  .catch((err) => {
    console.error(`dry-run failed: ${err instanceof Error ? err.stack : err}`);
    process.exit(1);
  });
