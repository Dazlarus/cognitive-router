// scripts/tune_ucb_k.ts — OFFLINE grid-tune of the UCB exploration constant
// (LEARNING_LOOP_DESIGN.md §4.2). Replays the historical judge_history evals
// and, for each candidate k, counts route flips (argmax over learned values
// with vs without the k/√n bonus) and whether a flip was harmful (the
// flipped-to model's observed average score at that time was lower).
// NO live tuning: this writes a report only; the chosen k is applied via
// ROUTER_UCB_K env, never adjusted at runtime.
//
// Usage: npx tsx scripts/tune_ucb_k.ts [--db data/cognitive-router.db]

import Database from "better-sqlite3";

const dbPath = process.argv.includes("--db")
  ? process.argv[process.argv.indexOf("--db") + 1]
  : "data/cognitive-router.db";

const K_GRID = [0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30];

interface Eval {
  timestamp: string;
  provider: string;
  model: string;
  intent: string;
  judge_score: number;
}

const db = new Database(dbPath, { readonly: true });
const evals = db.prepare(
  `SELECT timestamp, provider, model, intent, judge_score FROM judge_history ORDER BY timestamp ASC`,
).all() as Eval[];

console.log(`Replaying ${evals.length} judge_history evals (k grid: ${K_GRID.join(", ")})`);

// State per (intent, pair): EMA-learned value + sample count, replayed in time order.
const ALPHA = 0.25;
function pairKey(e: { provider: string; model: string }) { return `${e.provider}/${e.model}`; }

interface PairState { value: number; n: number; obsSum: number; obsN: number; }

const results = K_GRID.map((k) => ({ k, flips: 0, harmfulFlips: 0, beneficialFlips: 0, decisions: 0 }));

const learned = new Map<string, PairState>(); // intent -> (pairKey -> state)

for (const e of evals) {
  const intentMap = learned.get(e.intent) ?? new Map<string, PairState>();
  learned.set(e.intent, intentMap);

  // Candidate set = pairs that have ≥1 eval for this intent BEFORE this event.
  const candidates = Array.from(intentMap.entries()).filter(([, s]) => s.n > 0);
  if (candidates.length >= 2) {
    // Observed average score at this time (proxy for "how good is this model really").
    const obs = (s: PairState) => (s.obsN > 0 ? s.obsSum / s.obsN : 0);

    let basePick: [string, PairState] | null = null;
    let baseBest = -Infinity;
    for (const c of candidates) {
      if (c[1].value > baseBest) { baseBest = c[1].value; basePick = c; }
    }
    for (const r of results) {
      const k = r.k;
      let ucbPick: [string, PairState] | null = null;
      let ucbBest = -Infinity;
      for (const c of candidates) {
        const s = c[1].value + (k > 0 ? k / Math.sqrt(c[1].n) : 0);
        if (s > ucbBest) { ucbBest = s; ucbPick = c; }
      }
      if (basePick && ucbPick && basePick[0] !== ucbPick[0]) {
        r.flips++;
        if (obs(ucbPick[1]) < obs(basePick[1]) - 1e-9) r.harmfulFlips++;
        else if (obs(ucbPick[1]) > obs(basePick[1]) + 1e-9) r.beneficialFlips++;
      }
      r.decisions++;
    }
  }

  // Apply this eval to the pair's state (EMA + running observed average).
  const key = pairKey(e);
  const st = intentMap.get(key) ?? { value: 0.5, n: 0, obsSum: 0, obsN: 0 };
  st.value = st.value * (1 - ALPHA) + (e.judge_score / 10) * ALPHA;
  st.n += 1;
  st.obsSum += e.judge_score;
  st.obsN += 1;
  intentMap.set(key, st);
}

console.table(results.map((r) => ({
  k: r.k,
  decisions: r.decisions,
  flips: r.flips,
  flipRate: r.decisions > 0 ? (r.flips / r.decisions).toFixed(4) : "0",
  harmful: r.harmfulFlips,
  beneficial: r.beneficialFlips,
  harmRatio: r.flips > 0 ? (r.harmfulFlips / r.flips).toFixed(3) : "0",
})));

// Selection: k=0 trivially has zero harmful flips but zero exploration — it
// defeats the approved guardrail (low-n models never re-enter rotation), so
// the baseline is excluded. Among exploratory k>0: minimize harmful flips,
// tie-break on beneficial flips.
const exploratory = results.filter((r) => r.k > 0);
const best = exploratory.reduce((a, b) => {
  if (b.harmfulFlips !== a.harmfulFlips) return b.harmfulFlips < a.harmfulFlips ? b : a;
  return b.beneficialFlips > a.beneficialFlips ? b : a;
});

console.log(`\nSelected k = ${best.k} (min harmful flips among exploratory k: ${best.harmfulFlips}; ` +
  `k=0 baseline excluded — no exploration = no guardrail)`);

// Write the tuning log.
const fs = await import("node:fs");
const rows = results.map((r) => `| ${r.k} | ${r.decisions} | ${r.flips} | ${r.harmfulFlips} | ${r.beneficialFlips} | ${r.flips > 0 ? (r.harmfulFlips / r.flips).toFixed(3) : "0"} |`).join("\n");
const report = `# UCB k Tuning Log — ${new Date().toISOString()}

Offline grid-tune of the exploration constant k (learned + k/√n) against the
full judge_history replay (\`scripts/tune_ucb_k.ts\`). No live tuning: the
selected k is a static env value (\`ROUTER_UCB_K\`).

- Eval count: ${evals.length}
- Method: replay EMA (alpha=0.25) per (intent, pair) in time order; at each
  event with ≥2 candidate pairs, compare argmax with vs without the bonus.
  Harmful flip = flipped-to model's observed mean score was lower than the
  baseline pick's at that moment.
- Selection rule: k=0 (no exploration) is the degenerate baseline — trivially
  zero harmful flips but it defeats the exploration guardrail, so it is
  excluded. Among k>0: minimize harmful flips; tie-break on beneficial flips.

| k | decisions | flips | harmful | beneficial | harm ratio |
|---|---|---|---|---|---|
${rows}

**Selected k = ${best.k}** (harmful flips: ${best.harmfulFlips}, beneficial: ${best.beneficialFlips}).
Apply by setting \`ROUTER_UCB_K=${best.k}\` in the environment.
`;

fs.writeFileSync("docs/TUNING_LOG.md", report, "utf8");
console.log("Wrote docs/TUNING_LOG.md");
db.close();
