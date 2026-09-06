// tests/bench_trigger.test.ts - Bench trigger + canary support (offline)
// Run with: npx tsx --test tests/bench_trigger.test.ts

import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import {
  BENCH_INTENT,
  benchPass,
  collectBenchable,
  pendingIdentities,
} from "../src/bench_trigger.ts";
import { PROMPT_GENERATION, encodeKey, AS_SERVED } from "../src/benchmark_ladder.ts";
import type { ModelRegistry } from "../src/model_registry.ts";
import type { DBService } from "../src/db_service.ts";

// ---------- fakes ----------

function makeRegistry(models: Array<Record<string, unknown>>): ModelRegistry {
  return {
    getAllModels: () => models.map((m) => ({
      provider: "openrouter",
      model: "x",
      isLocal: false,
      ...m,
    })),
  } as unknown as ModelRegistry;
}

function makeDB(opts: {
  ladderKeys?: string[];
  verdictModels?: string[];
  capture?: Array<{ intent: string; entries: Array<{ modelKey: string; rank: number; strength: number }> }>;
} = {}): DBService {
  const verdictModels = opts.verdictModels ?? [];
  const rows = verdictModels.length >= 2
    ? [{ modelA: verdictModels[0], modelB: verdictModels[1], round: 0, swapOrder: 0 as const, verdict: "a" as const }]
    : [];
  // replaceLadder -> getLadderKeys round-trip so rebuildLadder sees persisted state
  let storedKeys = [...(opts.ladderKeys ?? [])];
  return {
    getLadderKeys: () => [...storedKeys],
    getAllBenchmarkVerdicts: () => rows,
    getBenchmarkVerdicts: () => [],
    insertBenchmarkVerdict: () => {},
    replaceLadder: (intent: string, _gen: string, entries: Array<{ modelKey: string; rank: number; strength: number }>) => {
      storedKeys = entries.map((e) => e.modelKey);
      opts.capture?.push({ intent, entries });
    },
  } as unknown as DBService;
}

// ---------- collectBenchable ----------

describe("bench trigger - collectBenchable", () => {
  it("includes priced remotes and locals, excludes unpriced remotes", () => {
    const reg = makeRegistry([
      { provider: "openrouter", model: "paid/model", costPer1kInput: 0.001, costPer1kOutput: 0.002 },
      { provider: "openrouter", model: "unknown/model" }, // no pricing -> quarantined
      { provider: "ollama", model: "gemma4:latest", isLocal: true },
    ]);
    const ids = collectBenchable(reg);
    const keys = ids.map((i) => encodeKey(i.key));
    assert.ok(keys.includes("paid/model|as-served|medium"));
    assert.ok(keys.includes("gemma4:latest|as-served|medium"));
    assert.ok(!keys.includes("unknown/model|as-served|medium"));
  });

  it("derives quant from ollama tags", () => {
    const reg = makeRegistry([
      { provider: "ollama", model: "qwen3:14b-instruct-q4_K_M", isLocal: true },
    ]);
    const ids = collectBenchable(reg);
    assert.equal(ids.length, 1);
    assert.equal(ids[0].key.model, "qwen3:14b-instruct");
    assert.equal(ids[0].key.quant, "q4");
    assert.equal(ids[0].key.effort, "medium");
  });

  it("sorts pending queue cheapest-first with free remotes before locals at equal cost", () => {
    const reg = makeRegistry([
      { provider: "ollama", model: "biglocal:70b", isLocal: true },
      { provider: "openrouter", model: "free/model:free", costPer1kInput: 0, costPer1kOutput: 0 },
      { provider: "openrouter", model: "paid/model", costPer1kInput: 0.001, costPer1kOutput: 0.002 },
    ]);
    const ids = collectBenchable(reg);
    assert.equal(ids[0].key.model, "free/model:free");
    assert.equal(ids[0].cheapestCostPer1k, 0);
    assert.equal(ids[1].key.model, "biglocal:70b");
    assert.equal(ids[2].key.model, "paid/model");
  });

  it("groups multiple servings of one identity, cheapest endpoint first", () => {
    const reg = makeRegistry([
      { provider: "ollama", model: "gemma4:latest", isLocal: true },
      { provider: "openrouter", model: "gemma4:latest", costPer1kInput: 0, costPer1kOutput: 0 },
    ]);
    const ids = collectBenchable(reg);
    const gemma = ids.find((i) => i.key.model === "gemma4:latest");
    assert.ok(gemma);
    assert.equal(gemma.endpoints.length, 2);
    assert.equal(gemma.endpoints[0].provider, "openrouter"); // free remote first
    assert.equal(gemma.endpoints[1].provider, "ollama");
  });
});

// ---------- pendingIdentities ----------

describe("bench trigger - pendingIdentities", () => {
  it("excludes identities already in the ladder", () => {
    const reg = makeRegistry([
      { provider: "openrouter", model: "a", costPer1kInput: 0, costPer1kOutput: 0 },
      { provider: "openrouter", model: "b", costPer1kInput: 0, costPer1kOutput: 0 },
    ]);
    const db = makeDB({ ladderKeys: ["a|as-served|medium"] });
    const pending = pendingIdentities(db, reg);
    assert.equal(pending.length, 1);
    assert.equal(pending[0].key.model, "b");
  });

  it("excludes identities seen in stored verdicts even without ladder rows", () => {
    const reg = makeRegistry([
      { provider: "openrouter", model: "a", costPer1kInput: 0, costPer1kOutput: 0 },
      { provider: "openrouter", model: "c", costPer1kInput: 0, costPer1kOutput: 0 },
    ]);
    const db = makeDB({ verdictModels: ["a|as-served|medium", "gone|as-served|medium"] });
    const pending = pendingIdentities(db, reg);
    assert.equal(pending.length, 1);
    assert.equal(pending[0].key.model, "c");
  });
});

// ---------- benchPass ----------

describe("bench trigger - benchPass", () => {
  beforeEach(() => { delete process.env.ROUTER_BENCH_TRIGGER; });
  afterEach(() => { delete process.env.ROUTER_BENCH_TRIGGER; });

  it("discovery trigger is a no-op without ROUTER_BENCH_TRIGGER=1", async () => {
    const reg = makeRegistry([
      { provider: "openrouter", model: "a", costPer1kInput: 0, costPer1kOutput: 0 },
    ]);
    const captured: any[] = [];
    const db = makeDB({ capture: captured });
    const result = await benchPass(db, reg, "discovery");
    assert.equal(result.enabled, false);
    assert.equal(result.benched, null);
    assert.equal(result.pendingBefore, -1); // unknown: no scan when disabled
    assert.equal(captured.length, 0); // nothing persisted
  });

  it("admin trigger cold-starts an empty ladder with zero model calls", async () => {
    // First identity into an EMPTY ladder: binary insert needs no comparisons,
    // so no provider calls happen — the pure cold-start path is offline-testable.
    const reg = makeRegistry([
      { provider: "openrouter", model: "a", costPer1kInput: 0, costPer1kOutput: 0 },
    ]);
    const captured: any[] = [];
    const db = makeDB({ capture: captured });
    const result = await benchPass(db, reg, "admin");
    assert.equal(result.enabled, true);
    assert.equal(result.benched, "a|as-served|medium");
    assert.equal(result.comparisons?.length ?? 0, 0); // no comparisons on empty ladder
    assert.equal(result.ladderSize, 1);
    assert.equal(result.pendingAfter, 0);
    // ladder persisted via replaceLadder
    assert.equal(captured.length, 1);
    assert.equal(captured[0].intent, BENCH_INTENT);
    assert.equal(captured[0].entries[0].modelKey, "a|as-served|medium");
    assert.equal(captured[0].entries[0].rank, 1);
  });

  it("no-op when nothing is pending", async () => {
    const reg = makeRegistry([
      { provider: "openrouter", model: "a", costPer1kInput: 0, costPer1kOutput: 0 },
    ]);
    const db = makeDB({ ladderKeys: ["a|as-served|medium"] });
    const result = await benchPass(db, reg, "admin");
    assert.equal(result.benched, null);
    assert.equal(result.pendingBefore, 0);
  });
});

// sanity: exported constants stay stable (they key ladder rows)
describe("bench trigger - constants", () => {
  it("pins intent=coding and the prompt generation", () => {
    assert.equal(BENCH_INTENT, "coding");
    assert.ok(PROMPT_GENERATION.length > 0);
    assert.equal(AS_SERVED, "as-served");
  });
});
