// src/benchmark_ladder.ts - Comparative ordinal benchmarking (v2 foundation)
//
// Design (Daz-spec, 2026-09-06):
//   - Ordinal, not absolute: models are ranked pairwise ("is A or B better")
//     per intent. Bradley-Terry strength preserves gaps; rank derives from it.
//   - Identity = (model, quantization, effort): Opus@q4/low != Opus@fp/max.
//     Quant is explicit for locals (Ollama tags), "as-served" for remotes.
//   - Capability is model-intrinsic; speed/cost/reliability stay endpoint-side
//     (live telemetry). The ModelCaller wiring must resolve the CHEAPEST
//     endpoint serving each identity.
//   - Pinned decode params: same weights + effort -> same response regardless
//     of serving speed (1 tok/s or 1000 tok/s must not change the verdict).
//   - Position bias: every round asks the judge twice with swapped order;
//     disagreement = tie for that round.
//   - Rounds: k=5 default (ceiling 20). Majority across rounds, ties half.
//   - Raw verdicts stored uncollapsed (one row per ask) so aggregation can be
//     re-derived later without re-spending tokens.
//   - Judge recusal is family-wide (matches judge.ts); the JudgeCaller
//     wiring must enforce it — this module does not call providers directly.
//
// Foundation only: nothing imports this yet. Wiring (discovery modes
// auto/safe/manual, hourly refresh, admin hook) comes next.

import { logger } from "./logger.js";
import type { DBService } from "./db_service.js";

// ---------- identity ----------

export type EffortLevel = "none" | "low" | "medium" | "high";

export interface BenchModelKey {
  /** Provider-agnostic model id, e.g. "claude-opus-4.5" */
  model: string;
  /** "fp" | "q4" | "q8" | ... for locals; AS_SERVED for remotes */
  quant: string;
  effort: EffortLevel;
}

export const AS_SERVED = "as-served";

export function encodeKey(k: BenchModelKey): string {
  return `${k.model}|${k.quant}|${k.effort}`;
}

export function decodeKey(s: string): BenchModelKey {
  const parts = s.split("|");
  return {
    model: parts[0] ?? s,
    quant: parts[1] ?? AS_SERVED,
    effort: (parts[2] as EffortLevel) ?? "none",
  };
}

/** Extract quant family from an Ollama tag.
 *  "qwen3:14b-instruct-q4_K_M" -> { model: "qwen3:14b-instruct", quant: "q4" } */
export function parseOllamaQuant(tag: string): { model: string; quant: string } {
  const m = tag.match(/[-:_]q(\d)(?:_[A-Za-z0-9]+)*$/i);
  if (!m) return { model: tag, quant: AS_SERVED };
  return { model: tag.slice(0, m.index), quant: `q${m[1].toLowerCase()}` };
}

// ---------- pinned decode ----------

/** Same weights + effort must yield the same response regardless of endpoint
 *  speed. Wiring passes these on every completion call. */
export const PINNED_DECODE = {
  temperature: 0.1,
  // 6144 not 2048: reasoning-always-on models (GLM-5.3 at mapped effort
  // high) burn the whole ceiling on reasoning before any content - live
  // 2026-09-07: every zai bench gen returned empty content at 2048. Ceiling
  // only; temp-0 concrete prompts make non-reasoning models stop far below.
  max_tokens: 6144,
} as const;

// ---------- prompt generations ----------

/** Generation tag invalidates caches when prompts OR the decode protocol
 *  change. gen-2026-09-02 (2026-09-08): as-served decode — uncapped, full
 *  response (reasoning + answer) captured and judged by the bench sidecar
 *  (cogrouter-bench ladder.ts is the twin; keep both tags in lockstep —
 *  CONTRACT.md §3 records the re-integration diff). */
export const PROMPT_GENERATION = "gen-2026-09-08";

export type BenchIntent = "coding" | "reasoning" | "conversation";

const STARTER_PROMPTS: Record<BenchIntent, string[]> = {
  coding: [
    "Write a TypeScript function `debounceAsync(fn, waitMs)` that debounces an async function: only the last call within the window runs, and its result (not a stale one) resolves. Include cancellation of in-flight work. Show the code and one usage example.",
    "Given an array of log entries {ts: number, level: 'info'|'warn'|'error', msg: string}, write a Python function that returns the busiest 5-minute window by error count, with ties broken by earliest window. Explain your approach in two sentences before the code.",
    "A SQL table `orders(id, customer_id, created_at, total_cents)` has 200M rows. Write a query for each customer's latest order, and explain in one paragraph why your formulation avoids a full sort.",
  ],
  reasoning: [
    "A factory's output doubles every 2 days but the machine degrades and loses 10% of its current output rate each day it runs continuously. Starting at 100 units/day, is there a steady state, and if so what is it? Show your reasoning step by step.",
    "Three boxes: one contains only apples, one only oranges, one both. All three labels are wrong. You may pull one fruit from one box. Explain the exact procedure to relabel all boxes correctly and why it works.",
    "A train leaves A at 60 mph; another leaves B at 90 mph, 180 miles away, 30 minutes later, toward A. A bird flies at 120 mph between the trains, turning around instantly at each. How far does the bird fly before the trains meet? Reason it out.",
  ],
  conversation: [
    "A friend says: 'I've been learning to paint for a year and my work still looks childish. I think I should quit.' Write a warm, honest reply that neither flatters nor dismisses them. 3-5 sentences.",
    "Explain to a curious 10-year-old why the sky is blue but sunsets are red. Keep it conversational, accurate, and under 120 words.",
    "You're recommending a podcast guest who studies sleep. Draft three interview questions that avoid the usual cliches, each with a one-line reason why it's interesting.",
  ],
};

export function promptsFor(intent: BenchIntent): string[] {
  return STARTER_PROMPTS[intent] ?? STARTER_PROMPTS.conversation;
}

// ---------- verdicts & comparison ----------

export type Verdict = "a" | "b" | "tie";

export const LADDER_ROUNDS = 5;
export const LADDER_MAX_ROUNDS = 20;

/** Runs the same prompt against a model identity; returns response text.
 *  Wiring must resolve the cheapest endpoint serving the identity, apply
 *  PINNED_DECODE, and map effort onto provider-specific fields. */
export type ModelCaller = (key: BenchModelKey, prompt: string) => Promise<string>;

/** Judges two responses to one prompt; returns a/b/tie from A's perspective.
 *  Receives both identities so wiring can enforce family-wide judge recusal
 *  (judge family == candidate family -> alternate judge or throw). */
export type JudgeCaller = (
  prompt: string,
  responseA: string,
  responseB: string,
  keyA?: BenchModelKey,
  keyB?: BenchModelKey,
) => Promise<Verdict>;

export interface CompareDeps {
  db: DBService;
  callModel: ModelCaller;
  judge: JudgeCaller;
  rounds?: number;
}

export interface RawAskRow {
  modelA: string;
  modelB: string;
  round: number;
  swapOrder: 0 | 1;
  verdict: Verdict; // from modelA's perspective (getter normalizes)
}

function majority(vs: Verdict[]): Verdict {
  let s = 0;
  for (const v of vs) s += v === "a" ? 1 : v === "b" ? 0 : 0.5;
  return s > vs.length / 2 ? "a" : s < vs.length / 2 ? "b" : "tie";
}

/** Collapse raw asks into round verdicts (swap-agreement rule), then majority.
 *  Rows must already be normalized to a single (a,b) perspective. */
export function collapseRounds(rows: RawAskRow[]): Verdict | null {
  if (rows.length === 0) return null;
  const byRound = new Map<number, Verdict[]>();
  for (const r of rows) {
    const list = byRound.get(r.round) ?? [];
    list.push(r.verdict);
    byRound.set(r.round, list);
  }
  const roundVerdicts: Verdict[] = [];
  for (const [round, asks] of byRound) {
    if (asks.length >= 2) {
      // first ask + swapped ask: agreement wins, disagreement = tie
      roundVerdicts.push(asks[0] === asks[1] ? asks[0] : "tie");
    } else {
      roundVerdicts.push(asks[0]);
    }
    void round;
  }
  return majority(roundVerdicts);
}

/** Aggregated verdict for a pair from stored rows (either direction),
 *  current generation. Null when never compared. */
export function cachedPairVerdict(
  db: DBService,
  a: string,
  b: string,
  intent: BenchIntent,
  generation = PROMPT_GENERATION,
): Verdict | null {
  const rows = db.getBenchmarkVerdicts(a, b, intent, generation);
  return collapseRounds(rows);
}

/** One round: same prompt judged twice, order swapped. Disagreement = tie. */
async function askRound(
  judge: JudgeCaller,
  prompt: string,
  ra: string,
  rb: string,
  keyA: BenchModelKey,
  keyB: BenchModelKey,
): Promise<{ first: Verdict; swapped: Verdict; round: Verdict }> {
  const first = await judge(prompt, ra, rb, keyA, keyB);
  const swappedRaw = await judge(prompt, rb, ra); // B in slot A
  const swapped: Verdict =
    swappedRaw === "a" ? "b" : swappedRaw === "b" ? "a" : "tie";
  const round: Verdict = first === swapped ? first : "tie";
  return { first, swapped, round };
}

/** Full pair comparison: k rounds x 2 order-swapped judge asks.
 *  Persists every raw ask uncollapsed, then returns the majority. */
export async function comparePair(
  deps: CompareDeps,
  a: BenchModelKey,
  b: BenchModelKey,
  intent: BenchIntent,
): Promise<Verdict> {
  const rounds = Math.min(deps.rounds ?? LADDER_ROUNDS, LADDER_MAX_ROUNDS);
  const prompts = promptsFor(intent);
  const aKey = encodeKey(a);
  const bKey = encodeKey(b);
  const roundVerdicts: Verdict[] = [];

  // Failure path: a pair that dies mid-comparison (dead generator, judge
  // outage) must leave NO partial rounds behind — a later retry would
  // collapse them into a majority computed over fewer, order-biased rounds.
  // Earlier completed rounds are deleted too: the retry re-spends them,
  // the honest price of a clean verdict.
  try {
    for (let r = 0; r < rounds; r++) {
      const prompt = prompts[r % prompts.length];
      const [ra, rb] = await Promise.all([
        deps.callModel(a, prompt),
        deps.callModel(b, prompt),
      ]);
      const { first, swapped, round } = await askRound(deps.judge, prompt, ra, rb, a, b);
      deps.db.insertBenchmarkVerdict({
        modelA: aKey,
        modelB: bKey,
        intent,
        promptGeneration: PROMPT_GENERATION,
        round: r,
        swapOrder: 0,
        verdict: first,
      });
      deps.db.insertBenchmarkVerdict({
        modelA: aKey,
        modelB: bKey,
        intent,
        promptGeneration: PROMPT_GENERATION,
        round: r,
        swapOrder: 1,
        verdict: swapped, // stored from A's perspective, flag marks the swap
      });
      roundVerdicts.push(round);
      logger.debug(
        `bench ladder [${intent}] ${aKey} vs ${bKey} round ${r}: ${round} ` +
          `(first=${first} swapped=${swapped})`,
      );
    }
  } catch (err) {
    deps.db.deleteBenchmarkVerdicts?.(aKey, bKey, intent, PROMPT_GENERATION);
    throw err;
  }

  const result = majority(roundVerdicts);
  logger.info(
    `bench ladder [${intent}] ${aKey} vs ${bKey} => ${result} ` +
      `(${rounds} rounds)`,
  );
  return result;
}

// ---------- ladder ----------

export interface LadderEntry {
  modelKey: string; // encoded
  rank: number; // 1 = strongest
  strength: number; // Bradley-Terry (geometric mean anchored to 1)
}

export interface LadderComparison {
  vs: string;
  verdict: Verdict; // newcomer's perspective
  cached: boolean;
}

/** Binary-insert a newcomer into a sorted ladder (rank 1 = best).
 *  Reuses cached pair verdicts for the current generation; benches only
 *  missing pairs (~log2(n) comparisons). Tie places newcomer just below
 *  the tied entry. Returns 0-based insertion index. */
export async function insertIntoLadder(
  deps: CompareDeps,
  ladder: string[],
  newcomer: BenchModelKey,
  intent: BenchIntent,
): Promise<{ index: number; comparisons: LadderComparison[] }> {
  const newcomerKey = encodeKey(newcomer);
  const comparisons: LadderComparison[] = [];
  let lo = 0;
  let hi = ladder.length;

  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    const vs = ladder[mid];
    let verdict = cachedPairVerdict(deps.db, newcomerKey, vs, intent);
    let cached = true;
    if (!verdict) {
      verdict = await comparePair(deps, newcomer, decodeKey(vs), intent);
      cached = false;
    }
    comparisons.push({ vs, verdict, cached });
    if (verdict === "a") hi = mid; // newcomer beats mid -> insert above
    else lo = mid + 1; // loses or ties -> below mid
  }

  return { index: lo, comparisons };
}

// ---------- Bradley-Terry strength ----------

/** Pair verdict from a's perspective, re-used for aggregation. */
export interface PairRecord {
  a: string;
  b: string;
  verdict: Verdict;
}

/** Bradley-Terry strength via MM iteration over collapsed pair verdicts.
 *  Ties split 0.5/0.5 (simple Davidson approximation — documented
 *  convention). Every model carries a weak prior: half a win against a
 *  virtual strength-1 opponent. Without it a winless model collapses to
 *  zero, MM stops converging, and the geometric-mean anchor breaks
 *  (found by tests/benchmark.test.ts). Strengths are anchored to
 *  geometric mean 1. */
export function computeBradleyTerry(
  pairs: PairRecord[],
  iterations = 64,
): Map<string, number> {
  const keys = new Set<string>();
  const wins = new Map<string, number>();
  const n = new Map<string, number>(); // sorted pair -> comparison count

  for (const p of pairs) {
    keys.add(p.a);
    keys.add(p.b);
    const wa = p.verdict === "a" ? 1 : p.verdict === "b" ? 0 : 0.5;
    wins.set(p.a, (wins.get(p.a) ?? 0) + wa);
    wins.set(p.b, (wins.get(p.b) ?? 0) + (1 - wa));
    const pk = [p.a, p.b].sort().join("\u0000");
    n.set(pk, (n.get(pk) ?? 0) + 1);
  }

  const s = new Map<string, number>();
  for (const k of keys) s.set(k, 1);

  for (let it = 0; it < iterations; it++) {
    const next = new Map<string, number>();
    for (const i of keys) {
      let denom = 1 / ((s.get(i) ?? 1) + 1); // virtual tie vs strength-1
      for (const j of keys) {
        if (i === j) continue;
        const pk = [i, j].sort().join("\u0000");
        const nij = n.get(pk) ?? 0;
        if (nij > 0) denom += nij / ((s.get(i) ?? 1) + (s.get(j) ?? 1));
      }
      const w = (wins.get(i) ?? 0) + 0.5; // prior half-win
      next.set(i, denom > 0 ? w / denom : 1);
    }
    // anchor: geometric mean = 1
    let logSum = 0;
    let cnt = 0;
    for (const v of next.values()) {
      logSum += Math.log(Math.max(v, 1e-9));
      cnt++;
    }
    const g = Math.exp(logSum / Math.max(cnt, 1));
    for (const k of keys) next.set(k, (next.get(k) ?? 1) / g);
    s.clear();
    for (const [k, v] of next) s.set(k, v);
  }

  return s;
}

/** Rebuild the full ladder for an intent from stored raw asks.
 *  Collapses rounds per pair, computes BT strengths, assigns ranks.
 *  Pairs never compared get no strength (ranked last, alphabetical).
 *  extraKeys admits identities with no verdicts yet (cold-start inserts:
 *  the first identity into an empty ladder compares against nothing). */
export function rebuildLadder(
  db: DBService,
  intent: BenchIntent,
  generation = PROMPT_GENERATION,
  extraKeys: string[] = [],
): LadderEntry[] {
  const rows = db.getAllBenchmarkVerdicts(intent, generation);

  // group raw asks by unordered pair
  const byPair = new Map<string, RawAskRow[]>();
  for (const r of rows) {
    const pk = [r.modelA, r.modelB].sort().join("\u0000");
    const list = byPair.get(pk) ?? [];
    list.push(r);
    byPair.set(pk, list);
  }

  const pairs: PairRecord[] = [];
  for (const [pk, asks] of byPair) {
    // normalize to (first-sorted-key wins) perspective per row
    const [k1, k2] = pk.split("\u0000");
    const normalized: RawAskRow[] = asks.map((r) =>
      r.modelA === k1
        ? r
        : {
            modelA: r.modelB,
            modelB: r.modelA,
            round: r.round,
            swapOrder: r.swapOrder,
            verdict: r.verdict === "a" ? "b" : r.verdict === "b" ? "a" : "tie",
          },
    );
    const v = collapseRounds(normalized);
    if (v) pairs.push({ a: k1, b: k2, verdict: v });
  }

  const strengths = computeBradleyTerry(pairs);
  const seen = new Set<string>();
  for (const p of pairs) {
    seen.add(p.a);
    seen.add(p.b);
  }
  const allKeys = new Set<string>(db.getLadderKeys(intent, generation));
  for (const k of seen) allKeys.add(k);
  for (const k of extraKeys) allKeys.add(k);

  const entries: Array<{ key: string; strength: number }> = [];
  for (const k of allKeys) {
    entries.push({ key: k, strength: strengths.get(k) ?? 0 });
  }
  entries.sort((x, y) =>
    y.strength - x.strength || x.key.localeCompare(y.key),
  );

  return entries.map((e, i) => ({
    modelKey: e.key,
    rank: i + 1,
    strength: e.strength,
  }));
}
