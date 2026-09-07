// tests/benchmark.test.ts - Sanity layer 1: pure-logic tests for the
// benchmark ladder, wiring helpers, and pricing lookups. Zero network,
// zero spend. Ground-truth E2E (layer 2) lives in scripts/bench-dryrun.ts.

import test from "node:test";
import assert from "node:assert/strict";
import {
  AS_SERVED,
  collapseRounds,
  comparePair,
  computeBradleyTerry,
  decodeKey,
  encodeKey,
  insertIntoLadder,
  parseOllamaQuant,
  promptsFor,
  rebuildLadder,
  type BenchModelKey,
  type RawAskRow,
  type Verdict,
} from "../src/benchmark_ladder.js";
import {
  catalogKeyCandidates,
} from "../src/discovery.js";
import { zaiBillingMode } from "../src/providers.js";

// ---------- identity ----------

test("encodeKey/decodeKey round-trip", () => {
  const k: BenchModelKey = { model: "claude-opus-4.5", quant: "q4", effort: "high" };
  assert.deepEqual(decodeKey(encodeKey(k)), k);
  const k2: BenchModelKey = { model: "glm-5.2", quant: AS_SERVED, effort: "none" };
  assert.deepEqual(decodeKey(encodeKey(k2)), k2);
});

test("decodeKey tolerates missing segments", () => {
  const d = decodeKey("somemodel");
  assert.equal(d.model, "somemodel");
  assert.equal(d.quant, AS_SERVED);
  assert.equal(d.effort, "none");
});

test("parseOllamaQuant extracts quant families", () => {
  assert.deepEqual(parseOllamaQuant("qwen2.5:3b-instruct-q4_K_M"), {
    model: "qwen2.5:3b-instruct",
    quant: "q4",
  });
  assert.deepEqual(parseOllamaQuant("mixtral-q8_0"), { model: "mixtral", quant: "q8" });
  assert.deepEqual(parseOllamaQuant("gemma4:latest"), { model: "gemma4:latest", quant: AS_SERVED });
  assert.deepEqual(parseOllamaQuant("qwen2.5-coder:7b"), {
    model: "qwen2.5-coder:7b",
    quant: AS_SERVED,
  });
});

// ---------- collapseRounds ----------

const row = (round: number, swap: 0 | 1, verdict: Verdict): RawAskRow => ({
  modelA: "A", modelB: "B", round, swapOrder: swap, verdict,
});

test("collapseRounds: swap agreement passes through", () => {
  assert.equal(collapseRounds([row(0, 0, "a"), row(0, 1, "a")]), "a");
});

test("collapseRounds: swap disagreement = tie for the round", () => {
  assert.equal(collapseRounds([row(0, 0, "a"), row(0, 1, "b")]), "tie");
});

test("collapseRounds: majority across rounds, ties count half", () => {
  // rounds: a, a, tie -> 2.5/3 -> a
  const rows = [
    row(0, 0, "a"), row(0, 1, "a"),
    row(1, 0, "a"), row(1, 1, "a"),
    row(2, 0, "a"), row(2, 1, "b"), // disagree -> tie
  ];
  assert.equal(collapseRounds(rows), "a");
  // rounds: b, tie, tie -> 0 + 0.5 + 0.5 = 1.0 < half(1.5) -> b
  const rows2 = [
    row(0, 0, "b"), row(0, 1, "b"),
    row(1, 0, "a"), row(1, 1, "b"), // tie
    row(2, 0, "b"), row(2, 1, "a"), // tie
  ];
  assert.equal(collapseRounds(rows2), "b");
});

test("collapseRounds: single-ask rounds pass through; empty = null", () => {
  assert.equal(collapseRounds([row(0, 0, "b")]), "b");
  assert.equal(collapseRounds([]), null);
});

// ---------- Bradley-Terry ----------

test("computeBradleyTerry recovers a deterministic dominance chain", () => {
  // A beats B, C, D; B beats C, D; C beats D. Each once.
  const pairs = [
    { a: "A", b: "B", verdict: "a" as Verdict },
    { a: "A", b: "C", verdict: "a" as Verdict },
    { a: "A", b: "D", verdict: "a" as Verdict },
    { a: "B", b: "C", verdict: "a" as Verdict },
    { a: "B", b: "D", verdict: "a" as Verdict },
    { a: "C", b: "D", verdict: "a" as Verdict },
  ];
  const s = computeBradleyTerry(pairs);
  assert.ok(s.get("A")! > s.get("B")!, "A > B");
  assert.ok(s.get("B")! > s.get("C")!, "B > C");
  assert.ok(s.get("C")! > s.get("D")!, "C > D");
});

test("computeBradleyTerry anchors geometric mean to 1", () => {
  const pairs = [
    { a: "A", b: "B", verdict: "a" as Verdict },
    { a: "A", b: "C", verdict: "tie" as Verdict },
    { a: "B", b: "C", verdict: "b" as Verdict },
  ];
  const s = computeBradleyTerry(pairs);
  const vals = [...s.values()];
  const logSum = vals.reduce((acc, v) => acc + Math.log(v), 0);
  assert.ok(Math.abs(logSum) < 1e-6, `geo mean anchor, logSum=${logSum}`);
});

test("computeBradleyTerry: winless model stays positive (prior), ordered last", () => {
  const pairs = [
    { a: "A", b: "B", verdict: "a" as Verdict },
    { a: "A", b: "Z", verdict: "a" as Verdict },
    { a: "B", b: "Z", verdict: "a" as Verdict },
  ];
  const s = computeBradleyTerry(pairs);
  assert.ok(s.get("Z")! > 0, `winless Z positive, got ${s.get("Z")}`);
  assert.ok(s.get("A")! > s.get("B")! && s.get("B")! > s.get("Z")!, "order preserved");
});

test("computeBradleyTerry tie splits evenly between equals", () => {
  const pairs = [
    { a: "X", b: "Y", verdict: "tie" as Verdict },
    { a: "X", b: "Y", verdict: "tie" as Verdict },
  ];
  const s = computeBradleyTerry(pairs);
  assert.ok(Math.abs(s.get("X")! - s.get("Y")!) < 1e-9);
});

// ---------- mock-driven compare/insert ----------

interface MockDb {
  inserted: RawAskRow[];
  verdicts: Map<string, RawAskRow[]>;
}

function makeMockDb(): MockDb {
  return { inserted: [], verdicts: new Map() };
}

function mockDbAs(db: MockDb): any {
  return {
    insertBenchmarkVerdict: (r: RawAskRow) => db.inserted.push(r),
    getBenchmarkVerdicts: (a: string, b: string) => db.verdicts.get(`${a}\u0000${b}`) ?? [],
    getAllBenchmarkVerdicts: () => db.inserted,
    getLadderKeys: () => [...new Set(db.inserted.flatMap((r) => [r.modelA, r.modelB]))],
  };
}

const STRONG: BenchModelKey = { model: "strong-model", quant: AS_SERVED, effort: "none" };
const WEAK: BenchModelKey = { model: "weak-model", quant: AS_SERVED, effort: "none" };

/** Position-agnostic judge: prefers the longer response either slot. */
function longerWinsJudge() {
  return async (_p: string, ra: string, rb: string): Promise<Verdict> =>
    ra.length > rb.length ? "a" : rb.length > ra.length ? "b" : "tie";
}

/** Position-biased judge: always picks slot A. Swap rule must neutralize. */
function biasedJudge() {
  return async (): Promise<Verdict> => "a";
}

test("comparePair: unbiased judge + length gap -> consistent winner; rows persisted", async () => {
  const db = makeMockDb();
  const deps = {
    db: mockDbAs(db),
    callModel: async (k: BenchModelKey) => (k.model === "strong-model" ? "a thorough correct answer with detail" : "x"),
    judge: longerWinsJudge(),
    rounds: 3,
  };
  const v = await comparePair(deps, STRONG, WEAK, "coding");
  assert.equal(v, "a");
  assert.equal(db.inserted.length, 6, "2 asks per round");
  assert.equal(new Set(db.inserted.map((r) => r.round)).size, 3);
});

test("comparePair: position-biased judge is neutralized to ties", async () => {
  const deps = {
    db: mockDbAs(makeMockDb()),
    callModel: async () => "same response",
    judge: biasedJudge(),
    rounds: 3,
  };
  const v = await comparePair(deps, STRONG, WEAK, "coding");
  assert.equal(v, "tie");
});

test("insertIntoLadder: empty ladder -> index 0, no comparisons", async () => {
  const deps = {
    db: mockDbAs(makeMockDb()),
    callModel: async () => "x",
    judge: longerWinsJudge(),
  };
  const { index, comparisons } = await insertIntoLadder(deps, [], WEAK, "coding");
  assert.equal(index, 0);
  assert.equal(comparisons.length, 0);
});

test("insertIntoLadder: stronger newcomer lands above; weaker below; tie below", async () => {
  const deps = {
    db: mockDbAs(makeMockDb()),
    callModel: async (k: BenchModelKey) => (k.model === "strong-model" ? "long detailed answer" : "short"),
    judge: longerWinsJudge(),
  };
  const mid: BenchModelKey = { model: "mid-model", quant: AS_SERVED, effort: "none" };
  // ladder: [weak]
  const weakFirst = await insertIntoLadder(deps, [encodeKey(WEAK)], STRONG, "coding");
  assert.equal(weakFirst.index, 0, "strong above weak");
  // ladder: [mid, weak] -> strong above mid
  const vsMid = await insertIntoLadder(deps, [encodeKey(mid), encodeKey(WEAK)], STRONG, "coding");
  assert.equal(vsMid.index, 0);
  // weak into [strong, mid] -> below both
  const weakLast = await insertIntoLadder(deps, [encodeKey(STRONG), encodeKey(mid)], WEAK, "coding");
  assert.equal(weakLast.index, 2);
  // equal responses -> tie -> below the tied entry
  const tieDeps = {
    db: mockDbAs(makeMockDb()),
    callModel: async () => "identical",
    judge: longerWinsJudge(),
  };
  const tied = await insertIntoLadder(tieDeps, [encodeKey(STRONG)], mid, "coding");
  assert.equal(tied.index, 1, "tie places newcomer below");
});

test("insertIntoLadder: cached verdicts skip model + judge calls entirely", async () => {
  const db = makeMockDb();
  // Pre-seed a collapsed 'b' (newcomer loses) for strong vs weak.
  db.verdicts.set(`${encodeKey(WEAK)}\u0000${encodeKey(STRONG)}`, [
    { modelA: encodeKey(WEAK), modelB: encodeKey(STRONG), round: 0, swapOrder: 0, verdict: "b" },
    { modelA: encodeKey(WEAK), modelB: encodeKey(STRONG), round: 0, swapOrder: 1, verdict: "b" },
  ]);
  let calls = 0;
  const deps = {
    db: mockDbAs(db),
    callModel: async () => {
      calls++;
      return "x";
    },
    judge: async () => {
      calls++;
      return "a" as Verdict;
    },
  };
  const { index, comparisons } = await insertIntoLadder(deps, [encodeKey(STRONG)], WEAK, "coding");
  assert.equal(index, 1, "cached loss -> below");
  assert.equal(comparisons[0]?.cached, true);
  assert.equal(calls, 0, "no live calls when cached");
});

test("rebuildLadder: rank 1 to the consistent winner; unpaired keys rank last", () => {
  const db = makeMockDb();
  const sk = encodeKey(STRONG);
  const wk = encodeKey(WEAK);
  const lonely = "lonely|as-served|none";
  for (let r = 0; r < 3; r++) {
    db.inserted.push(
      { modelA: wk, modelB: sk, round: r, swapOrder: 0, verdict: "b" },
      { modelA: wk, modelB: sk, round: r, swapOrder: 1, verdict: "b" },
    );
  }
  const ladder = rebuildLadder(
    Object.assign(mockDbAs(db), { getLadderKeys: () => [sk, wk, lonely] }),
    "coding",
  );
  assert.equal(ladder[0]?.modelKey, sk, "strong ranks first");
  assert.equal(ladder[1]?.modelKey, wk);
  // lonely has no comparisons -> strength 0, ranked last, present
  const lonelyEntry = ladder.find((e) => e.modelKey === lonely);
  assert.equal(lonelyEntry?.strength, 0);
  assert.equal(lonelyEntry?.rank, ladder.length);
});

test("promptsFor: starter pool has 3 prompts per intent", () => {
  for (const intent of ["coding", "reasoning", "conversation"] as const) {
    assert.equal(promptsFor(intent).length, 3);
  }
});

// ---------- wiring: families, recusal, verdict parsing ----------
// (familyOf/judgeRecused/parseAbVerdict moved to cogrouter-bench with the
//  extraction — coverage lives there now; router keeps no judge wiring)

// ---------- pricing ----------

test("catalogKeyCandidates: slash, dot, alias, bare forms in priority order", () => {
  assert.deepEqual(catalogKeyCandidates("gemini", "gemini-2.5-flash"), [
    "gemini/gemini-2.5-flash",
    "gemini.gemini-2.5-flash",
    "google/gemini-2.5-flash",
    "google.gemini-2.5-flash",
    "gemini-2.5-flash",
  ]);
  assert.deepEqual(catalogKeyCandidates("openai", "gpt-5.2"), [
    "openai/gpt-5.2",
    "openai.gpt-5.2",
    "gpt-5.2",
  ]);
});

test("zaiBillingMode: billing follows the endpoint URL", () => {
  const saved = process.env.ZAI_BASE_URL;
  try {
    process.env.ZAI_BASE_URL = "https://api.z.ai/api/coding/paas/v4";
    assert.equal(zaiBillingMode(), "coding_plan");
    process.env.ZAI_BASE_URL = "https://api.z.ai/api/paas/v4";
    assert.equal(zaiBillingMode(), "platform");
    delete process.env.ZAI_BASE_URL;
    assert.equal(zaiBillingMode(), "coding_plan", "default adapter = coding plan");
  } finally {
    if (saved === undefined) delete process.env.ZAI_BASE_URL;
    else process.env.ZAI_BASE_URL = saved;
  }
});
