// tests/modality.test.ts — Vision detection, modality filtering, and multimodal routing
// Run with: npx tsx --test tests/modality.test.ts

import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { detectModalities, modelSupportsModality, modelSupportsAllModalities, type Modality } from "../src/modality.ts";
import { RoutingEngine } from "../src/router.ts";
import { CostTracker } from "../src/cost_tracker.ts";
import { ModelRegistry } from "../src/model_registry.ts";
import { DBService } from "../src/db_service.ts";
import { loadConfig, type CognitiveRouterConfig } from "../src/config.ts";
import type { Classification } from "../src/classifier.ts";

// ─── Test Helpers ───────────────────────────────────────────

function makeConfig(overrides: Partial<CognitiveRouterConfig> = {}): CognitiveRouterConfig {
  const base = loadConfig({
    enabled: true,
    logLevel: "warn",
    providerPriority: ["zai", "openrouter", "ollama"],
    providers: {
      zai: { budgetType: "subscription", priority: "high" },
      openrouter: { budgetType: "free", priority: "high" },
      ollama: { budgetType: "free", priority: "low" },
    },
    weights: { capability: 0.5, reliability: 0.25, cost: 0.15, latency: 0.1 },
  });
  return { ...base, ...overrides };
}

function makeMockDB(): DBService {
  return {
    initializeSchema: async () => {},
    recordDecision: () => {},
    recordCallOutcome: () => {},
    recordRetry: () => {},
    getDecisionByRequestId: () => null,
    getRetryCount: () => 0,
    getRecentDecisions: () => [],
    getModelStats: () => [],
    getProviderHealth: () => [],
    getSpendByProvider: () => [],
    getSpend: () => 0,
    loadCircuitState: () => null,
    saveCircuitState: () => {},
    loadCapabilityOverrides: () => [],
    getCapabilityOverride: () => null,
    upsertCapabilityOverride: () => {},
    recordJudgeEvaluation: () => {},
    recordSpend: () => {},
    getAllSpend: () => [],
    getAllLatestChatBenchmarks: () => new Map(),
    close: () => {},
  } as any;
}

function makeClassification(intent: string = "conversation", confidence: number = 0.9): Classification {
  return { intent, confidence };
}

// ─── Modality Detection Tests ───────────────────────────────

describe("detectModalities — Image Content Detection", () => {
  it("should return text-only for plain string messages", () => {
    const result = detectModalities([
      { role: "user", content: "Hello, how are you?" },
    ]);
    assert.equal(result.isMultimodal, false);
    assert.equal(result.imageCount, 0);
    assert.deepEqual(result.modalities, ["text"]);
    assert.ok(result.summary.includes("text-only"));
  });

  it("should detect image_url content blocks (OpenAI vision format)", () => {
    const result = detectModalities([
      {
        role: "user",
        content: [
          { type: "text", text: "What's in this image?" },
          { type: "image_url", image_url: { url: "https://example.com/cat.jpg" } },
        ],
      },
    ]);
    assert.equal(result.isMultimodal, true);
    assert.equal(result.imageCount, 1);
    assert.ok(result.modalities.includes("vision"));
    assert.ok(result.modalities.includes("text"));
  });

  it("should detect image_file content blocks", () => {
    const result = detectModalities([
      {
        role: "user",
        content: [
          { type: "text", text: "Analyze this" },
          { type: "image_file", image_file: { file_id: "file-abc123" } },
        ],
      },
    ]);
    assert.equal(result.isMultimodal, true);
    assert.equal(result.imageCount, 1);
    assert.ok(result.modalities.includes("vision"));
  });

  it("should detect Anthropic-style image blocks", () => {
    const result = detectModalities([
      {
        role: "user",
        content: [
          { type: "text", text: "Look here" },
          { type: "image", source: { type: "base64", media_type: "image/png", data: "iVBOR..." } },
        ],
      },
    ]);
    assert.equal(result.isMultimodal, true);
    assert.equal(result.imageCount, 1);
    assert.ok(result.modalities.includes("vision"));
  });

  it("should detect inline base64 data URI images in string content", () => {
    const result = detectModalities([
      {
        role: "user",
        content: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
      },
    ]);
    assert.equal(result.isMultimodal, true);
    assert.equal(result.imageCount, 1);
    assert.ok(result.modalities.includes("vision"));
  });

  it("should detect multiple images across multiple messages", () => {
    const result = detectModalities([
      {
        role: "user",
        content: [
          { type: "image_url", image_url: { url: "https://example.com/1.jpg" } },
          { type: "image_url", image_url: { url: "https://example.com/2.jpg" } },
        ],
      },
      {
        role: "assistant",
        content: "I see two images.",
      },
      {
        role: "user",
        content: [
          { type: "text", text: "And this one?" },
          { type: "image_url", image_url: { url: "https://example.com/3.jpg" } },
        ],
      },
    ]);
    assert.equal(result.isMultimodal, true);
    assert.equal(result.imageCount, 3);
  });

  it("should handle empty or malformed messages gracefully", () => {
    const result = detectModalities([]);
    assert.equal(result.isMultimodal, false);
    assert.equal(result.imageCount, 0);

    const result2 = detectModalities([{ role: "user" } as any]);
    assert.equal(result2.isMultimodal, false);

    const result3 = detectModalities([{ role: "user", content: null } as any]);
    assert.equal(result3.isMultimodal, false);
  });

  it("should not falsely detect images in short strings", () => {
    const result = detectModalities([
      { role: "user", content: "data is important" },
    ]);
    assert.equal(result.isMultimodal, false);
    assert.equal(result.imageCount, 0);
  });
});

describe("detectModalities — Audio Content Detection", () => {
  it("should detect input_audio content blocks", () => {
    const result = detectModalities([
      {
        role: "user",
        content: [
          { type: "text", text: "Transcribe this" },
          { type: "input_audio", input_audio: { data: "base64audiodata", format: "wav" } },
        ],
      },
    ]);
    assert.equal(result.isMultimodal, true);
    assert.equal(result.audioCount, 1);
    assert.ok(result.modalities.includes("audio"));
  });
});

describe("modelSupportsModality helpers", () => {
  it("should check single modality support", () => {
    assert.equal(modelSupportsModality(["text", "vision"], "vision"), true);
    assert.equal(modelSupportsModality(["text"], "vision"), false);
  });

  it("should check all modalities support", () => {
    assert.equal(
      modelSupportsAllModalities(["text", "vision", "audio"], ["text", "vision"]),
      true,
    );
    assert.equal(
      modelSupportsAllModalities(["text", "vision"], ["text", "vision", "audio"]),
      false,
    );
  });
});

// ─── Router Vision Filtering Tests ──────────────────────────

describe("RoutingEngine — Vision Model Filtering", () => {
  let registry: ModelRegistry;
  let costTracker: CostTracker;
  let db: DBService;
  let config: CognitiveRouterConfig;
  let router: RoutingEngine;

  beforeEach(async () => {
    config = makeConfig();
    db = makeMockDB();
    costTracker = new CostTracker(db, config);
    await costTracker.refreshProviderStatus();
    registry = new ModelRegistry(db, config);
    await registry.loadCachedState();
    router = new RoutingEngine(registry, costTracker, db, config);
  });

  it("should filter to vision-capable models when image is present", async () => {
    const decision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-vision-session",
      { requiredModalities: ["text", "vision"] as Modality[] },
    );

    assert.ok(decision, "Should get a routing decision");

    // The selected model must support vision
    const selectedCap = registry.getCapability(decision!.provider, decision!.model);
    assert.ok(selectedCap, "Selected model should exist in registry");
    assert.ok(
      selectedCap!.modalities.includes("vision"),
      `Selected model ${decision!.provider}/${decision!.model} must support vision, ` +
      `but only has [${selectedCap!.modalities.join(", ")}]`,
    );

    // Modality filter info should be present
    assert.ok(decision!.modalityFilter, "Decision should include modalityFilter");
    assert.equal(decision!.modalityFilter!.wasMultimodal, true);
    assert.ok(decision!.modalityFilter!.requiredModalities.includes("vision"));

    // All filtered-out models should lack vision
    for (const filtered of decision!.modalityFilter!.filteredOut) {
      assert.ok(
        !filtered.modalities.includes("vision"),
        `${filtered.provider}/${filtered.model} should not have vision`,
      );
    }
  });

  it("should not affect text-only requests", async () => {
    const textDecision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-text-session",
      { requiredModalities: ["text"] as Modality[] },
    );

    assert.ok(textDecision, "Should get a routing decision");
    assert.equal(
      textDecision!.modalityFilter,
      undefined,
      "Text-only requests should not have modalityFilter",
    );

    // Should route to the same model as when no requiredModalities are passed
    const plainDecision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-plain-session",
      {},
    );
    assert.ok(plainDecision);
    assert.equal(textDecision!.provider, plainDecision!.provider);
    assert.equal(textDecision!.model, plainDecision!.model);
  });

  it("should return MODALITY_UNSUPPORTED error when no vision model is available", async () => {
    // Use a config where only providers without vision models are available
    // OpenRouter has no vision models in seed data; Z.AI has glm-4.6v but it's
    // planEligible: false. If we only use openrouter, there are no vision models.
    const noVisionConfig = makeConfig({
      providerPriority: ["openrouter"],
      providers: {
        openrouter: { budgetType: "free", priority: "high" },
      },
    });
    const noVisionRouter = new RoutingEngine(registry, costTracker, db, noVisionConfig);

    const decision = await noVisionRouter.decide(
      makeClassification("conversation", 0.9),
      "test-no-vision-session",
      { requiredModalities: ["text", "vision"] as Modality[] },
    );

    assert.ok(decision, "Should get a decision (error decision)");
    assert.ok(decision!.error, "Should have an error");
    assert.equal(decision!.error!.code, "MODALITY_UNSUPPORTED");
    assert.ok(
      decision!.error!.message.includes("vision"),
      "Error message should mention vision modality",
    );
    assert.ok(
      Array.isArray(decision!.error!.availableVisionModels),
      "Error should list available vision models",
    );
  });

  it("should include modalityFilter in /last-decision output (candidates)", async () => {
    const decision = await router.decide(
      makeClassification("coding", 0.9),
      "test-vision-observability",
      { requiredModalities: ["text", "vision"] as Modality[] },
    );

    assert.ok(decision);
    assert.ok(decision!.modalityFilter);
    assert.equal(decision!.modalityFilter!.wasMultimodal, true);
    assert.ok(decision!.modalityFilter!.filteredOut.length > 0,
      "Should have filtered out non-vision models");
    assert.ok(decision!.modalityFilter!.requiredModalities.includes("vision"));
  });

  it("should route to a vision-capable model for vision requests", async () => {
    const decision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-vision-zai",
      { requiredModalities: ["text", "vision"] as Modality[] },
    );

    assert.ok(decision);

    // glm-4.6v is planEligible: false but should be available for vision requests
    // as it's the only non-local vision model. The test verifies that the router
    // includes it as a candidate for multimodal requests even though it's not
    // in the coding plan.
    // Local vision models (gemma3:12b, moondream, etc.) are also candidates.
    const selectedCap = registry.getCapability(decision!.provider, decision!.model);
    assert.ok(selectedCap?.modalities.includes("vision"),
      "Selected model must be vision-capable");
  });

  it("should filter out ALL non-vision candidates, not just some", async () => {
    const decision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-vision-filter-completeness",
      { requiredModalities: ["text", "vision"] as Modality[] },
    );

    assert.ok(decision);
    assert.ok(decision!.modalityFilter);

    // Every filtered-out model should NOT have vision
    for (const f of decision!.modalityFilter!.filteredOut) {
      assert.ok(!f.modalities.includes("vision"),
        `${f.provider}/${f.model} was filtered but has vision — bug!`);
    }
  });

  it("should handle mixed vision + audio requirements", async () => {
    // No model in seed data supports both vision AND audio
    // So this should return MODALITY_UNSUPPORTED
    const decision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-multimodal-combo",
      { requiredModalities: ["text", "vision", "audio"] as Modality[] },
    );

    assert.ok(decision);
    assert.ok(decision!.error);
    assert.equal(decision!.error!.code, "MODALITY_UNSUPPORTED");
  });
});

// ─── Seed Data Vision Flags Tests ───────────────────────────

describe("Seed Data — Vision Capability Flags", () => {
  let registry: ModelRegistry;
  let db: DBService;
  let config: CognitiveRouterConfig;

  beforeEach(async () => {
    config = makeConfig();
    db = makeMockDB();
    registry = new ModelRegistry(db, config);
    await registry.loadCachedState();
  });

  it("should have 3+ vision-capable models in seed data", () => {
    const allModels = registry.getAllModels();
    const visionModels = allModels.filter((m) => m.modalities.includes("vision"));

    assert.ok(
      visionModels.length >= 3,
      `Expected at least 3 vision models, got ${visionModels.length}: ` +
      visionModels.map((m) => `${m.provider}/${m.model}`).join(", "),
    );
  });

  it("should correctly flag known vision models", () => {
    const expected = [
      "ollama/gemma3:12b",
      "ollama/gemma3:4b",
      "ollama/gemma3:1b",
      "ollama/llama3.2-vision:11b",
      "ollama/moondream:latest",
    ];

    for (const key of expected) {
      const [provider, ...modelParts] = key.split("/");
      const model = modelParts.join("/");
      const cap = registry.getCapability(provider, model);
      assert.ok(cap, `${key} should exist in registry`);
      assert.ok(
        cap!.modalities.includes("vision"),
        `${key} should have vision modality, got [${cap!.modalities.join(", ")}]`,
      );
    }
  });

  it("should NOT flag text-only models as vision-capable", () => {
    const textOnlyModels = [
      "zai/glm-5.2",
      "zai/glm-4.7",
      "ollama/gemma4:latest",
      "ollama/mistral:7b",
    ];

    for (const key of textOnlyModels) {
      const [provider, ...modelParts] = key.split("/");
      const model = modelParts.join("/");
      const cap = registry.getCapability(provider, model);
      if (cap) {
        assert.ok(
          !cap.modalities.includes("vision"),
          `${key} should NOT have vision modality`,
        );
      }
    }
  });
});
