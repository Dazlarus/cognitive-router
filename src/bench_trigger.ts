// src/bench_trigger.ts - Bench trigger + canary runner (Daz-spec, 2026-09-06)
//
// Cold-start from zero: the ladder starts EMPTY; benchmark verdicts are the
// only ranking source. This module owns the deliberate step between
// discovery (registers models, never spends) and the ladder (ranks them):
//
//   - collectBenchable: registry -> benchable identities (chat-capable,
//     non-embedding, priced-or-local per the free-vs-unknown rule)
//   - pending = identities with NO ladder presence and NO stored verdicts
//     for the bench intent (genuinely new, not just re-registered)
//   - benchPass: ONE identity per discovery pass (ROUTER_BENCH_MAX_PER_PASS,
//     default 1), inserted via binary ladder insert, then the ladder is
//     rebuilt + persisted (Bradley-Terry)
//   - cheapest-endpoint resolution per identity (free remotes + locals first)
//   - intent: coding only (v1); effort pinned to "medium" as the canonical
//     probing effort — identity = (model, quant, medium)
//
// Canary runner (Layer 3) proves the judge wiring live, no persistence:
//   - mixed-family pair judged by the primary bench judge (ZAI glm-5.2)
//   - glm-vs-glm pair proving family-wide recusal falls to ollama/gemma4
//
// Spending gates: ROUTER_BENCH_TRIGGER=1 arms discovery-triggered passes
// (default off — tests and fresh installs stay silent); /admin/bench and
// /admin/bench/canary are explicit admin actions and always allowed.

import { logger } from "./logger.js";
import type { ModelRegistry } from "./model_registry.js";
import type { DBService } from "./db_service.js";
import { isEmbeddingOnlyModel } from "./model_policy.js";
import {
  AS_SERVED,
  PINNED_DECODE,
  PROMPT_GENERATION,
  type BenchIntent,
  type BenchModelKey,
  type LadderComparison,
  type Verdict,
  decodeKey,
  encodeKey,
  insertIntoLadder,
  parseOllamaQuant,
  promptsFor,
  rebuildLadder,
} from "./benchmark_ladder.js";
import {
  A_B_JUDGE_PROMPT,
  familyOf,
  judgeRecused,
  parseAbVerdict,
  type BenchEndpoint,
} from "./benchmark_wiring.js";
import { getProvider } from "./providers.js";

export const BENCH_INTENT: BenchIntent = "coding";
export const BENCH_EFFORT = "medium" as const;

// ---------- identity collection ----------

export interface BenchableIdentity {
  key: BenchModelKey;
  /** Every endpoint serving this identity, cheapest first. */
  endpoints: BenchEndpoint[];
  /** Cheapest total per-1k cost across endpoints (0 = free/local). */
  cheapestCostPer1k: number;
}

/** Registry -> benchable identities grouped by (model, quant).
 *  Remotes with undefined pricing are skipped (free-vs-unknown rule: they
 *  are quarantined from routing, so benching them would be spend without
 *  a routing payoff). Embedding-only models are skipped. */
export function collectBenchable(registry: ModelRegistry): BenchableIdentity[] {
  interface Serving {
    endpoint: BenchEndpoint;
    cost: number;
    local: boolean;
    remote: boolean;
  }
  const grouped = new Map<string, { key: BenchModelKey; servings: Serving[] }>();

  for (const m of registry.getAllModels()) {
    if (isEmbeddingOnlyModel(m.provider, m.model)) continue;
    // The ENTIRE openrouter/* namespace is meta, not models: their public
    // listing exposes only routers/utilities there (auto, auto-beta, fusion,
    // pareto-code, free, bodybuilder — verified against the live listing
    // 2026-09-06). Some appear double-prefixed (model id itself starts with
    // openrouter/), some bare (auto-beta). None is a benchable identity:
    // each delegates to a moving target (pareto-code literally re-ranks by
    // Artificial Analysis). Exclude the namespace + defensive bare forms.
    if (
      m.provider === "openrouter" &&
      (m.model.toLowerCase().startsWith("openrouter/") ||
        /(?:^|\/)(auto(?=-|$)|fusion$|pareto-code$)/i.test(m.model))
    ) continue;
    const priced = m.costPer1kInput !== undefined && m.costPer1kOutput !== undefined;
    if (!m.isLocal && !priced) continue; // quarantined remote

    let key: BenchModelKey;
    if (m.provider === "ollama") {
      const { model, quant } = parseOllamaQuant(m.model);
      key = { model, quant, effort: BENCH_EFFORT };
    } else {
      key = { model: m.model, quant: AS_SERVED, effort: BENCH_EFFORT };
    }
    const encoded = encodeKey(key);
    const entry = grouped.get(encoded) ?? { key, servings: [] };
    entry.servings.push({
      endpoint: { provider: m.provider, model: m.model },
      cost: (m.costPer1kInput ?? 0) + (m.costPer1kOutput ?? 0),
      local: m.isLocal,
      remote: !m.isLocal,
    });
    grouped.set(encoded, entry);
  }

  const out: BenchableIdentity[] = [];
  for (const { key, servings } of grouped.values()) {
    // Cheapest first; free remotes before locals (faster, same $0);
    // equal cost -> remote before local.
    servings.sort((a, b) => a.cost - b.cost || Number(b.remote) - Number(a.remote));
    out.push({
      key,
      endpoints: servings.map((s) => s.endpoint),
      cheapestCostPer1k: servings[0]?.cost ?? 0,
    });
  }
  // Pending queue order: cheapest identities enter the ladder first;
  // at equal cost free remotes before locals (faster, same $0).
  out.sort(
    (a, b) =>
      a.cheapestCostPer1k - b.cheapestCostPer1k ||
      Number(b.endpoints[0]?.provider !== "ollama") - Number(a.endpoints[0]?.provider !== "ollama"),
  );
  return out;
}

/** Identities with no ladder presence and no stored verdicts — genuinely
 *  new to the bench for this intent + generation. */
export function pendingIdentities(
  db: DBService,
  registry: ModelRegistry,
  intent: BenchIntent = BENCH_INTENT,
): BenchableIdentity[] {
  const seen = new Set<string>(db.getLadderKeys(intent, PROMPT_GENERATION));
  for (const row of db.getAllBenchmarkVerdicts(intent, PROMPT_GENERATION)) {
    seen.add(row.modelA);
    seen.add(row.modelB);
  }
  return collectBenchable(registry).filter((i) => !seen.has(encodeKey(i.key)));
}

// ---------- bench pass ----------

export interface BenchPassResult {
  trigger: "discovery" | "admin";
  enabled: boolean;
  benched: string | null;
  pendingBefore: number;
  pendingAfter: number;
  ladderSize: number;
  comparisons?: LadderComparison[];
  error?: string;
}

function identityResolver(identity: BenchableIdentity) {
  const byEncoded = new Map<string, BenchEndpoint>(
    identity.endpoints.map((e) => [encodeKey(identity.key), e]),
  );
  return (key: BenchModelKey): BenchEndpoint | null =>
    byEncoded.get(encodeKey(key)) ?? null;
}

/** One deliberate bench pass: at most ROUTER_BENCH_MAX_PER_PASS new
 *  identities (default 1), binary-inserted into the coding ladder.
 *  Discovery-triggered passes require ROUTER_BENCH_TRIGGER=1; admin passes
 *  are explicit and always run. */
export async function benchPass(
  db: DBService,
  registry: ModelRegistry,
  trigger: "discovery" | "admin",
  opts: { rounds?: number } = {},
): Promise<BenchPassResult> {
  const enabled =
    trigger === "admin" || process.env.ROUTER_BENCH_TRIGGER === "1";
  if (!enabled) {
    // Zero work when disabled — pendingBefore reports -1 (unknown), not a
    // scan result, so callers cannot mistake it for "nothing to do".
    return {
      trigger,
      enabled: false,
      benched: null,
      pendingBefore: -1,
      pendingAfter: -1,
      ladderSize: -1,
    };
  }
  const pending = pendingIdentities(db, registry);
  const base: BenchPassResult = {
    trigger,
    enabled,
    benched: null,
    pendingBefore: pending.length,
    pendingAfter: pending.length,
    ladderSize: db.getLadderKeys(BENCH_INTENT, PROMPT_GENERATION).length,
  };
  if (pending.length === 0) return base;

  const maxPerPass = Math.max(
    1,
    parseInt(process.env.ROUTER_BENCH_MAX_PER_PASS ?? "1", 10) || 1,
  );
  void maxPerPass; // admission cap is 1 per pass; attempts below may exceed it

  // A dead generator (free-tier corpse, observed live: cohere free tier)
  // must not stall the cheapest-first queue forever: try up to 3 pending
  // identities per pass, but admit at most one. Failed attempts leave no
  // verdict rows (comparePair deletes partials), so they stay pending.
  const MAX_ATTEMPTS = 3;
  let lastError = "";
  for (let attempt = 0; attempt < Math.min(MAX_ATTEMPTS, pending.length); attempt++) {
    const chosen = pending[attempt];
    const { makeModelCaller, makeJudgeCaller } = await import("./benchmark_wiring.js");
    const deps = {
      db,
      callModel: makeModelCaller(identityResolver(chosen)),
      judge: makeJudgeCaller(),
      rounds: opts.rounds,
    };

    try {
      const ladder = db.getLadderKeys(BENCH_INTENT, PROMPT_GENERATION);
      const { comparisons } = await insertIntoLadder(
        deps,
        ladder,
        chosen.key,
        BENCH_INTENT,
      );
      // extraKeys admits the newcomer even with zero comparisons (cold-start
      // insert into an empty ladder compares against nothing).
      const rebuilt = rebuildLadder(db, BENCH_INTENT, PROMPT_GENERATION, [
        encodeKey(chosen.key),
      ]);
      db.replaceLadder(BENCH_INTENT, PROMPT_GENERATION, rebuilt);
      logger.info(
        `bench pass (${trigger}): inserted ${encodeKey(chosen.key)} at rank ` +
          `${(rebuilt.find((e) => e.modelKey === encodeKey(chosen.key))?.rank) ?? "?"}, ` +
          `ladder now ${rebuilt.length} identities`,
      );
      return {
        ...base,
        benched: encodeKey(chosen.key),
        pendingAfter: pendingIdentities(db, registry).length,
        ladderSize: rebuilt.length,
        comparisons,
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      lastError = message;
      logger.warn(`bench pass (${trigger}) failed for ${encodeKey(chosen.key)}: ${message}`);
    }
  }
  return { ...base, error: lastError || "all attempts failed" };
}

// ---------- canary (Layer 3) ----------

export interface CanaryPairResult {
  label: string;
  a: string;
  b: string;
  judgeConsidered: string[];
  judgeRecusedFrom: string[];
  judgeUsed: string;
  verdict: Verdict;
  roundTripMs: number;
  error?: string;
}

export interface CanaryResult {
  mixed?: CanaryPairResult;
  recusal?: CanaryPairResult;
  error?: string;
}

async function callEndpointText(
  endpoint: BenchEndpoint,
  prompt: string,
): Promise<string> {
  const adapter = getProvider(endpoint.provider);
  if (!adapter) throw new Error(`no adapter for ${endpoint.provider}`);
  const apiKey = process.env[endpoint.provider.toUpperCase() + "_API_KEY"] ?? "";
  if (!apiKey && endpoint.provider !== "ollama") {
    throw new Error(`no API key for ${endpoint.provider}`);
  }
  const result = await adapter.chatCompletion(
    endpoint.model,
    {
      model: endpoint.model,
      messages: [{ role: "user", content: prompt }],
      stream: false,
      temperature: PINNED_DECODE.temperature,
      max_tokens: PINNED_DECODE.max_tokens,
    } as any,
    apiKey,
    AbortSignal.timeout(120_000),
  );
  const content = result.choices?.[0]?.message?.content ?? "";
  if (!content.trim()) {
    throw new Error(`empty response from ${endpoint.provider}/${endpoint.model}`);
  }
  return content;
}

/** One canary comparison between two identities: single generation each,
 *  single judge ask, full observability (which judges were considered,
 *  recused, and which served). Persists NOTHING — canaries prove wiring,
 *  they do not feed the ladder. */
async function canaryPair(
  label: string,
  a: BenchableIdentity,
  b: BenchableIdentity,
  judgeCandidates: BenchEndpoint[],
): Promise<CanaryPairResult> {
  const started = Date.now();
  const result: CanaryPairResult = {
    label,
    a: encodeKey(a.key),
    b: encodeKey(b.key),
    judgeConsidered: judgeCandidates.map((j) => `${j.provider}/${j.model}`),
    judgeRecusedFrom: [],
    judgeUsed: "",
    verdict: "tie",
    roundTripMs: 0,
  };
  try {
    const prompt = promptsFor(BENCH_INTENT)[0];
    const [ra, rb] = await Promise.all([
      callEndpointText(a.endpoints[0], prompt),
      callEndpointText(b.endpoints[0], prompt),
    ]);

    for (const judge of judgeCandidates) {
      if (judgeRecused(judge.provider, judge.model, { provider: "", model: a.key.model }) ||
          judgeRecused(judge.provider, judge.model, { provider: "", model: b.key.model })) {
        result.judgeRecusedFrom.push(`${judge.provider}/${judge.model}`);
        continue;
      }
      const adapter = getProvider(judge.provider);
      if (!adapter) continue;
      const apiKey = process.env[judge.provider.toUpperCase() + "_API_KEY"] ?? "";
      if (!apiKey && judge.provider !== "ollama") continue;

      const filled = A_B_JUDGE_PROMPT
        .replace("{prompt}", prompt)
        .replace("{responseA}", ra)
        .replace("{responseB}", rb);
      const jr = await adapter.chatCompletion(
        judge.model,
        {
          model: judge.model,
          messages: [{ role: "user", content: filled }],
          stream: false,
          temperature: 0.1,
          // 1024 not 100: thinking models burn the budget on reasoning
          // before content (see makeJudgeCaller note; found by live canary).
          max_tokens: 1024,
        } as any,
        apiKey,
      );
      const verdict = parseAbVerdict(jr.choices?.[0]?.message?.content ?? "");
      if (!verdict) throw new Error("judge returned unparseable verdict");
      result.judgeUsed = `${judge.provider}/${judge.model}`;
      result.verdict = verdict;
      break;
    }
    if (!result.judgeUsed) throw new Error("all judge candidates recused or unavailable");
    result.roundTripMs = Date.now() - started;
    return result;
  } catch (err) {
    result.error = err instanceof Error ? err.message : String(err);
    result.roundTripMs = Date.now() - started;
    return result;
  }
}

/** Family of an identity, for pairing + recusal logic. */
function identityFamily(i: BenchableIdentity): string {
  return familyOf(i.endpoints[0]?.provider ?? "", i.key.model);
}

/** Layer 3 canary: prove the live judge wiring end-to-end.
 *  - mixed: two identities from different families, neither sharing the
 *    primary judge's family -> primary judge (ZAI) must serve.
 *  - recusal: two identities sharing the primary judge's family (glm) ->
 *    primary recuses, local fallback (gemma4) must serve. */
export async function runCanary(
  db: DBService,
  registry: ModelRegistry,
): Promise<CanaryResult> {
  const { getJudgeCandidates } = await import("./benchmark_wiring.js");
  const judges = getJudgeCandidates();
  const primary = judges[0];
  const primaryFamily = familyOf(primary.provider, primary.model);

  const benchable = collectBenchable(registry);
  const glmIdentities = benchable.filter((i) => identityFamily(i) === primaryFamily);
  const nonJudgeFamilies = benchable.filter(
    (i) => identityFamily(i) !== primaryFamily,
  );

  const result: CanaryResult = {};

  // Recusal pair first (cheaper: local fallback proves even if remotes fail)
  if (glmIdentities.length >= 2) {
    result.recusal = await canaryPair(
      "recusal",
      glmIdentities[0],
      glmIdentities[1],
      judges,
    );
  } else {
    result.recusal = {
      label: "recusal",
      a: "",
      b: "",
      judgeConsidered: judges.map((j) => `${j.provider}/${j.model}`),
      judgeRecusedFrom: [],
      judgeUsed: "",
      verdict: "tie",
      roundTripMs: 0,
      error: `need >=2 ${primaryFamily}-family identities, have ${glmIdentities.length}`,
    };
  }

  // Mixed pair: cheapest representatives of distinct non-judge families;
  // try up to 3 pairings so one dead free-tier generator (observed live:
  // cohere/north-mini-code:free empty response) can't sink the canary.
  const byFamily = new Map<string, BenchableIdentity>();
  for (const i of nonJudgeFamilies) {
    if (!byFamily.has(identityFamily(i))) byFamily.set(identityFamily(i), i);
  }
  const reps = [...byFamily.values()];
  const pairings: Array<[BenchableIdentity, BenchableIdentity]> = [
    [reps[0], reps[1]], [reps[0], reps[2]], [reps[1], reps[2]],
  ].filter((p): p is [BenchableIdentity, BenchableIdentity] =>
    p[0] !== undefined && p[1] !== undefined);

  result.mixed = undefined;
  for (const [x, y] of pairings) {
    const attempt = await canaryPair("mixed", x, y, judges);
    if (!attempt.error) {
      result.mixed = attempt;
      break;
    }
    result.mixed = attempt; // keep last error for reporting
  }
  if (!result.mixed) {
    result.mixed = {
      label: "mixed",
      a: "",
      b: "",
      judgeConsidered: judges.map((j) => `${j.provider}/${j.model}`),
      judgeRecusedFrom: [],
      judgeUsed: "",
      verdict: "tie",
      roundTripMs: 0,
      error: `need 2 distinct non-${primaryFamily} families, have ${reps.length}`,
    };
  }
  void db;
  return result;
}

export { decodeKey };
