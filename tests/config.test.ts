// tests/config.test.ts — Unit tests for config.ts
// Run with: npx tsx --test tests/config.test.ts

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { loadConfig, type CognitiveRouterConfig, type Weights } from "../src/config.ts";

// ─── Tests ──────────────────────────────────────────────────

describe("config — default values", () => {
  it("should return enabled=false when not specified", () => {
    const config = loadConfig({});
    assert.equal(config.enabled, false);
  });

  it("should return correct DEFAULT_WEIGHTS", () => {
    const config = loadConfig({});
    const expectedWeights: Weights = {
      capability: 0.50,
      reliability: 0.25,
      cost: 0.15,
      latency: 0.10,
    };
    assert.deepEqual(config.weights, expectedWeights);
  });

  it("should default providerPriority to [zai, openrouter, gemini, ollama]", () => {
    const config = loadConfig({});
    assert.deepEqual(config.providerPriority, ["zai", "openrouter", "gemini", "ollama"]);
  });

  it("should default probeRate to 0.05", () => {
    const config = loadConfig({});
    assert.equal(config.probeRate, 0.05);
  });

  it("should default tiebreakerThreshold to 0.70", () => {
    const config = loadConfig({});
    assert.equal(config.tiebreakerThreshold, 0.70);
  });

  it("should default benchmarkSyncIntervalHours to 168", () => {
    const config = loadConfig({});
    assert.equal(config.benchmarkSyncIntervalHours, 168);
  });

  it('should default logLevel to "info"', () => {
    const config = loadConfig({});
    assert.equal(config.logLevel, "info");
  });

  it('should default dbPath to "data/cognitive-router.db"', () => {
    const config = loadConfig({});
    assert.equal(config.dbPath, "data/cognitive-router.db");
  });

  it("should default localVramLimitGb to 11", () => {
    const config = loadConfig({});
    assert.equal(config.localVramLimitGb, 11);
  });

  it("should default proxyPort to 3456", () => {
    const config = loadConfig({});
    assert.equal(config.proxyPort, 3456);
  });

  it("should default overrides to empty array", () => {
    const config = loadConfig({});
    assert.deepEqual(config.overrides, []);
  });

  it("should include all default providers", () => {
    const config = loadConfig({});
    assert.ok(config.providers.openrouter, "openrouter should exist");
    assert.ok(config.providers.zai, "zai should exist");
    assert.ok(config.providers.gemini, "gemini should exist");
    assert.ok(config.providers.ollama, "ollama should exist");
    assert.ok(config.providers.requesty, "requesty should exist");
  });

  it("should set correct budget types for default providers", () => {
    const config = loadConfig({});
    assert.equal(config.providers.openrouter.budgetType, "free");
    assert.equal(config.providers.zai.budgetType, "subscription");
    assert.equal(config.providers.gemini.budgetType, "credits");
    assert.equal(config.providers.ollama.budgetType, "free");
    assert.equal(config.providers.requesty.budgetType, "pay_per_token");
  });
});

describe("config — partial overrides", () => {
  it("should keep all defaults when only enabled is passed", () => {
    const config = loadConfig({ enabled: true });
    assert.equal(config.enabled, true);
    // All other defaults should be intact
    assert.deepEqual(config.weights, { capability: 0.50, reliability: 0.25, cost: 0.15, latency: 0.10 });
    assert.deepEqual(config.providerPriority, ["zai", "openrouter", "gemini", "ollama"]);
    assert.equal(config.localVramLimitGb, 11);
    assert.equal(config.proxyPort, 3456);
    assert.equal(config.probeRate, 0.05);
    assert.equal(config.tiebreakerThreshold, 0.70);
    assert.equal(config.benchmarkSyncIntervalHours, 168);
  });

  it("should keep all defaults when only proxyPort is passed", () => {
    const config = loadConfig({ proxyPort: 8080 });
    assert.equal(config.proxyPort, 8080);
    assert.equal(config.enabled, false);
    assert.deepEqual(config.weights, { capability: 0.50, reliability: 0.25, cost: 0.15, latency: 0.10 });
  });
});

describe("config — custom weights merge with defaults", () => {
  it("should override only the specified weight fields", () => {
    const config = loadConfig({ weights: { capability: 0.70 } });
    assert.equal(config.weights.capability, 0.70);
    // Other weights should retain defaults
    assert.equal(config.weights.reliability, 0.25);
    assert.equal(config.weights.cost, 0.15);
    assert.equal(config.weights.latency, 0.10);
  });

  it("should override all weights when all are specified", () => {
    const config = loadConfig({
      weights: { capability: 0.4, reliability: 0.3, cost: 0.2, latency: 0.1 },
    });
    assert.deepEqual(config.weights, { capability: 0.4, reliability: 0.3, cost: 0.2, latency: 0.1 });
  });
});

describe("config — custom providers merge with DEFAULT_PROVIDERS", () => {
  it("should add a new provider without removing defaults", () => {
    const config = loadConfig({
      providers: {
        custom_provider: { budgetType: "free", priority: "low" },
      },
    });
    assert.ok(config.providers.custom_provider, "Custom provider should be present");
    // Default providers should still be there
    assert.ok(config.providers.openrouter);
    assert.ok(config.providers.zai);
    assert.ok(config.providers.gemini);
    assert.ok(config.providers.ollama);
  });

  it("should override a specific provider without affecting others", () => {
    const config = loadConfig({
      providers: {
        ollama: { budgetType: "subscription", priority: "high" },
      },
    });
    assert.equal(config.providers.ollama.budgetType, "subscription");
    assert.equal(config.providers.ollama.priority, "high");
    // Other providers unchanged
    assert.equal(config.providers.zai.budgetType, "subscription");
    assert.equal(config.providers.openrouter.budgetType, "free");
  });
});

describe("config — custom providerPriority", () => {
  it("should override the default providerPriority", () => {
    const customPriority = ["ollama", "gemini", "openrouter", "zai"];
    const config = loadConfig({ providerPriority: customPriority });
    assert.deepEqual(config.providerPriority, customPriority);
  });

  it("should accept a single-element priority list", () => {
    const config = loadConfig({ providerPriority: ["zai"] });
    assert.deepEqual(config.providerPriority, ["zai"]);
  });

  it("should accept an empty priority list", () => {
    const config = loadConfig({ providerPriority: [] });
    assert.deepEqual(config.providerPriority, []);
  });
});

describe("config — overrides array", () => {
  it("should pass through the overrides array", () => {
    const overrides = [
      { intent: "coding", provider: "ollama", model: "gemma4:latest", reason: "test" },
      { intent: "creative", provider: "zai", model: "glm-5.1" },
    ];
    const config = loadConfig({ overrides });
    assert.deepEqual(config.overrides, overrides);
  });

  it("should default to empty array when no overrides provided", () => {
    const config = loadConfig({});
    assert.deepEqual(config.overrides, []);
  });
});

describe("config — localVramLimitGb", () => {
  it("should default to 11", () => {
    const config = loadConfig({});
    assert.equal(config.localVramLimitGb, 11);
  });

  it("should accept custom value", () => {
    const config = loadConfig({ localVramLimitGb: 24 });
    assert.equal(config.localVramLimitGb, 24);
  });
});

describe("config — proxyPort", () => {
  it("should default to 3456", () => {
    const config = loadConfig({});
    assert.equal(config.proxyPort, 3456);
  });

  it("should accept custom value", () => {
    const config = loadConfig({ proxyPort: 7878 });
    assert.equal(config.proxyPort, 7878);
  });
});
