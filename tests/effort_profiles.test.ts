// tests/effort_profiles.test.ts - Phase 2 config cards: normalization + merge precedence
import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import {
  normalizeEffortLevel,
  SEED_CONFIG_CARDS,
  heuristicCardFor,
  mergeConfigCard,
  type ConfigCard,
} from "../src/effort_profiles.ts";
import { ModelRegistry } from "../src/model_registry.ts";
import type { DBService } from "../src/db_service.ts";
import { loadConfig, type CognitiveRouterConfig } from "../src/config.ts";

function makeConfig(): CognitiveRouterConfig {
  return loadConfig({
    enabled: true, logLevel: "warn",
    providerPriority: ["zai", "openrouter"],
    providers: {
      zai: { budgetType: "subscription", priority: "high" },
      openrouter: { budgetType: "free", priority: "high" },
    },
    weights: { capability: 0.5, reliability: 0.25, cost: 0.15, latency: 0.1 },
  });
}

function makeMockDB(): DBService {
  return {
    initializeSchema: async () => {},
    loadCapabilityOverrides: () => [],
    getCapabilityOverride: () => null,
    upsertCapabilityOverride: () => {},
    recordJudgeEvaluation: () => {},
    close: () => {},
  } as any;
}

describe("normalizeEffortLevel - per-provider tables", () => {
  it("zai: maps raw names incl. thinking toggle + clamps xhigh/ultra to max", () => {
    assert.equal(normalizeEffortLevel("zai", "low"), "low");
    assert.equal(normalizeEffortLevel("zai", "max"), "max");
    assert.equal(normalizeEffortLevel("zai", "xhigh"), "max");
    assert.equal(normalizeEffortLevel("zai", "ultra"), "max");
    assert.equal(normalizeEffortLevel("zai", "disabled"), "thinking-off");
    assert.equal(normalizeEffortLevel("zai", "enabled"), "thinking-on");
  });

  it("openrouter/openai: minimal clamps, xhigh keeps ultra on openrouter", () => {
    assert.equal(normalizeEffortLevel("openrouter", "minimal"), "none");
    assert.equal(normalizeEffortLevel("openrouter", "xhigh"), "ultra");
    assert.equal(normalizeEffortLevel("openai", "minimal"), "low");
    assert.equal(normalizeEffortLevel("openai", "xhigh"), "ultra");
  });

  it("gemini/anthropic: budget-style names normalize; unknown provider -> undefined", () => {
    assert.equal(normalizeEffortLevel("gemini", "off"), "thinking-off");
    assert.equal(normalizeEffortLevel("anthropic", "high"), "high");
    assert.equal(normalizeEffortLevel("ollama", "high"), undefined);
  });

  it("case-insensitive + unknown raw -> undefined", () => {
    assert.equal(normalizeEffortLevel("zai", "MAX"), "max");
    assert.equal(normalizeEffortLevel("zai", "banana"), undefined);
  });
});

describe("SEED_CONFIG_CARDS - roster coverage", () => {
  it("covers the zai effort-matrix roster with coherent profiles", () => {
    for (const key of ["zai/glm-4.7", "zai/glm-5.1", "zai/glm-5.3", "zai/glm-5.3-flash"]) {
      assert.ok(SEED_CONFIG_CARDS[key], `missing seed card: ${key}`);
      const p = SEED_CONFIG_CARDS[key].effortProfile;
      assert.ok(p, `missing effort profile: ${key}`);
      assert.ok(p.levels.length >= 2);
      assert.ok(p.levels.includes(p.defaultLevel), "defaultLevel must be in levels");
    }
  });

  it("glm-5.3 is reasoning-always-on ladder low|high|max (docs.z.ai glm-5.3)", () => {
    const p = SEED_CONFIG_CARDS["zai/glm-5.3"].effortProfile!;
    assert.equal(p.axis, "effort_ladder");
    assert.deepEqual(p.levels, ["low", "high", "max"]);
    assert.ok(!p.levels.includes("thinking-off"));
  });

  it("glm-5.2 accepts only high|max plus disable (github.com/zai-org/GLM-5)", () => {
    const p = SEED_CONFIG_CARDS["zai/glm-5.2"].effortProfile!;
    assert.deepEqual(p.levels, ["thinking-off", "high", "max"]);
    assert.equal(p.defaultLevel, "max");
  });

  it("glm-4.7 uses the thinking_on_off axis", () => {
    assert.equal(SEED_CONFIG_CARDS["zai/glm-4.7"].effortProfile!.axis, "thinking_on_off");
  });
});

describe("mergeConfigCard - precedence (seed > discovery)", () => {
  const seedCard: ConfigCard = {
    cacheUsage: false,
    fastMode: { supported: false },
    effortProfile: { axis: "effort_ladder", levels: ["low", "high", "max"], defaultLevel: "high" },
    confidence: "high", source: "seed",
  };

  it("seed card wins on every defined field; discovery cannot override", () => {
    const discovered: ConfigCard = {
      cacheUsage: true,
      fastMode: { supported: true, pricing: "$0.1/1M" },
      effortProfile: { axis: "thinking_on_off", levels: ["thinking-off", "thinking-on"], defaultLevel: "thinking-on" },
      confidence: "low", source: "discovery",
    };
    const merged = mergeConfigCard(seedCard, discovered)!;
    assert.equal(merged.cacheUsage, false);
    assert.equal(merged.fastMode.supported, false);
    assert.equal(merged.effortProfile!.axis, "effort_ladder");
    assert.deepEqual(merged.effortProfile!.levels, ["low", "high", "max"]);
    assert.equal(merged.source, "seed");
  });

  it("discovery fills gaps when seed fields are absent (no seed card)", () => {
    const discovered: ConfigCard = {
      cacheUsage: true, fastMode: { supported: true }, effortProfile: null,
      confidence: "low", source: "discovery",
    };
    const merged = mergeConfigCard(undefined, discovered)!;
    assert.equal(merged.cacheUsage, true);
    assert.equal(merged.source, "discovery");
  });

  it("no discovery card -> seed untouched", () => {
    const merged = mergeConfigCard(seedCard, undefined)!;
    assert.equal(merged, seedCard);
  });
});

describe("ModelRegistry - card attach + discovery gap-fill", () => {
  let registry: ModelRegistry;
  const originalZaiKey = process.env.ZAI_API_KEY;

  beforeEach(() => {
    delete process.env.ZAI_API_KEY;
    registry = new ModelRegistry(makeMockDB(), makeConfig());
  });

  it("loadCachedState attaches seed cards to seeded models", async () => {
    await registry.loadCachedState();
    const glm47 = registry.getCapability("zai", "glm-4.7");
    assert.ok(glm47?.configCard, "glm-4.7 card attached");
    assert.equal(glm47!.configCard!.effortProfile!.axis, "thinking_on_off");
    const glm53 = registry.getCapability("zai", "glm-5.3");
    assert.ok(glm53?.configCard);
    assert.deepEqual(glm53!.configCard!.effortProfile!.levels, ["low", "high", "max"]);
  });

  it("applyDiscoveredConfigCard cannot override a seed card (seed wins end-to-end)", async () => {
    await registry.loadCachedState();
    const changed = registry.applyDiscoveredConfigCard("zai", "glm-5.3", {
      cacheUsage: true,
      fastMode: { supported: true },
      effortProfile: { axis: "thinking_on_off", levels: ["thinking-off", "thinking-on"], defaultLevel: "thinking-on" },
      confidence: "low", source: "discovery",
    });
    const card = registry.getCapability("zai", "glm-5.3")!.configCard!;
    assert.equal(card.effortProfile!.axis, "effort_ladder");
    assert.equal(card.source, "seed");
    // seed card already had every field defined -> merge is a no-op rebuild
    assert.equal(changed, true); // card object replaced but content seed-authored
  });

  it("applyDiscoveredConfigCard fills cardless models", async () => {
    await registry.loadCachedState();
    // ollama model has no seed card
    const changed = registry.applyDiscoveredConfigCard("ollama", "gemma4:latest", {
      cacheUsage: false, fastMode: { supported: false },
      effortProfile: null, confidence: "low", source: "discovery",
    });
    assert.equal(changed, true);
    assert.equal(registry.getCapability("ollama", "gemma4:latest")!.configCard!.source, "discovery");
  });

  it("registerExternalModel merges discovery card without clobbering seeds", async () => {
    await registry.loadCachedState();
    const wasNew = registry.registerExternalModel("openrouter", "z-ai/glm-5.3", {
      costPer1kInput: 0.0014, costPer1kOutput: 0.0044,
      configCard: { cacheUsage: true, fastMode: { supported: false }, effortProfile: null, confidence: "low", source: "discovery" },
    });
    assert.equal(wasNew, false); // seeded mirror already known
    const card = registry.getCapability("openrouter", "z-ai/glm-5.3")!.configCard!;
    assert.equal(card.effortProfile!.levels.join(","), "low,high,max"); // seed ladder survives
    assert.equal(card.cacheUsage, false); // seed value survives
  });

  it("heuristicCardFor: unseeded glm variants get low-confidence discovery cards; seeded keys return undefined", () => {
    assert.equal(heuristicCardFor("zai", "glm-5.3"), undefined);
    const h = heuristicCardFor("zai", "glm-5.4");
    assert.ok(h);
    assert.equal(h!.source, "discovery");
    assert.equal(h!.confidence, "low");
    const orMirror = heuristicCardFor("openrouter", "z-ai/glm-5.4");
    assert.ok(orMirror);
    // flash-tier heuristic marks fast mode
    const flash = heuristicCardFor("openrouter", "z-ai/glm-5.4-flash", { supportsReasoningEffort: true });
    assert.ok(flash);
    assert.equal(flash!.fastMode.supported, true);
    assert.ok(flash!.effortProfile);
  });

  if (originalZaiKey !== undefined) process.env.ZAI_API_KEY = originalZaiKey;
});
