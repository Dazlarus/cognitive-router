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
    recordAbortEvent: () => {},
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
  it("should detect input_audio content blocks (OpenAI format)", () => {
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
    assert.ok(result.modalities.includes("text"));
    assert.ok(result.summary.includes("audio"));
  });

  it("should detect generic audio content blocks", () => {
    const result = detectModalities([
      {
        role: "user",
        content: [
          { type: "text", text: "What is this sound?" },
          { type: "audio", source: { type: "base64", media_type: "audio/wav", data: "UklGRiQAAABXQVZFZm10IBAAAAABAAEA" } },
        ],
      },
    ]);
    assert.equal(result.isMultimodal, true);
    assert.equal(result.audioCount, 1);
    assert.ok(result.modalities.includes("audio"));
  });

  it("should detect multiple audio blocks across messages", () => {
    const result = detectModalities([
      {
        role: "user",
        content: [
          { type: "input_audio", input_audio: { data: "audio1", format: "mp3" } },
          { type: "input_audio", input_audio: { data: "audio2", format: "mp3" } },
        ],
      },
      {
        role: "assistant",
        content: "I hear two audio clips.",
      },
      {
        role: "user",
        content: [
          { type: "text", text: "And this one?" },
          { type: "input_audio", input_audio: { data: "audio3", format: "wav" } },
        ],
      },
    ]);
    assert.equal(result.isMultimodal, true);
    assert.equal(result.audioCount, 3);
  });

  it("should detect mixed image + audio in the same request", () => {
    const result = detectModalities([
      {
        role: "user",
        content: [
          { type: "text", text: "Describe this image and transcribe this audio" },
          { type: "image_url", image_url: { url: "https://example.com/img.jpg" } },
          { type: "input_audio", input_audio: { data: "base64audio", format: "wav" } },
        ],
      },
    ]);
    assert.equal(result.isMultimodal, true);
    assert.equal(result.imageCount, 1);
    assert.equal(result.audioCount, 1);
    assert.ok(result.modalities.includes("vision"));
    assert.ok(result.modalities.includes("audio"));
    assert.ok(result.summary.includes("image"));
    assert.ok(result.summary.includes("audio"));
  });

  it("should not falsely detect audio in plain text messages", () => {
    const result = detectModalities([
      { role: "user", content: "Please play some audio for me" },
    ]);
    assert.equal(result.isMultimodal, false);
    assert.equal(result.audioCount, 0);
    assert.ok(!result.modalities.includes("audio"));
  });

  it("should detect audio in fallback block types", () => {
    // Test blocks that don't use standard type field but have audio data
    const result = detectModalities([
      {
        role: "user",
        content: [
          { type: "text", text: "Analyze" },
          { type: "custom_block", audio: { data: "base64" } },
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
      Array.isArray(decision!.error!.availableModalityModels),
      "Error should list available modality models",
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

  it("should route mixed vision + audio to a model supporting both", async () => {
    // glm-4.6v supports both vision AND audio, so this should succeed
    // and route to a model that has both modalities.
    const decision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-multimodal-combo",
      { requiredModalities: ["text", "vision", "audio"] as Modality[] },
    );

    assert.ok(decision, "Should get a routing decision");
    assert.ok(!decision!.error, `Should not error: ${decision!.error?.message}`);

    // Selected model must support BOTH vision and audio
    const selectedCap = registry.getCapability(decision!.provider, decision!.model);
    assert.ok(selectedCap, "Selected model should exist in registry");
    assert.ok(
      selectedCap!.modalities.includes("vision"),
      `Selected model must support vision: ${decision!.provider}/${decision!.model}`,
    );
    assert.ok(
      selectedCap!.modalities.includes("audio"),
      `Selected model must support audio: ${decision!.provider}/${decision!.model}`,
    );

    // Modality filter should reflect all required modalities
    assert.ok(decision!.modalityFilter);
    assert.ok(decision!.modalityFilter!.requiredModalities.includes("vision"));
    assert.ok(decision!.modalityFilter!.requiredModalities.includes("audio"));
    assert.ok(decision!.modalityFilter!.wasMultimodal);
  });
});

// ─── Vision Quality Scoring Tests ───────────────────────

describe("RoutingEngine — Vision Quality Scoring", () => {
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

  it("should include visionQualityScores in modalityFilter when vision is required", async () => {
    const decision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-vision-quality-info",
      { requiredModalities: ["text", "vision"] as Modality[] },
    );

    assert.ok(decision);
    assert.ok(decision!.modalityFilter);
    assert.ok(decision!.modalityFilter!.visionQualityScores,
      "modalityFilter should include visionQualityScores");
    assert.ok(decision!.modalityFilter!.visionQualityScores!.length > 0,
      "Should have at least one vision quality score");

    // Verify each score entry has the expected shape
    for (const entry of decision!.modalityFilter!.visionQualityScores!) {
      assert.ok(entry.provider, "Entry should have provider");
      assert.ok(entry.model, "Entry should have model");
      assert.ok(typeof entry.visionQuality === "number", "visionQuality should be a number");
      assert.ok(entry.visionQuality > 0 && entry.visionQuality <= 1,
        `visionQuality should be in (0,1], got ${entry.visionQuality}`);
    }
  });

  it("should prefer higher-quality vision models for image requests", async () => {
    // Route a vision request and check that the winner has a higher vision quality
    // than the average of all vision-capable models
    const decision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-vision-preference",
      { requiredModalities: ["text", "vision"] as Modality[] },
    );

    assert.ok(decision);
    const winnerCap = registry.getCapability(decision!.provider, decision!.model);
    assert.ok(winnerCap, "Winner should exist in registry");
    assert.ok(winnerCap!.visionQuality !== undefined,
      "Winner should have visionQuality");

    // The winner should have above-average vision quality among vision models
    const visionModels = registry.getAllModels().filter((m) => m.modalities.includes("vision"));
    const avgQuality = visionModels.reduce((sum, m) => sum + (m.visionQuality ?? 0.5), 0) / visionModels.length;

    assert.ok(
      winnerCap!.visionQuality! >= avgQuality,
      `Winner ${decision!.provider}/${decision!.model} visionQuality=${winnerCap!.visionQuality} ` +
      `should be >= average ${avgQuality.toFixed(3)}`,
    );
  });

  it("should score vision models differently based on quality", async () => {
    // Make two vision requests with the same intent and verify that the
    // vision quality adjustment differentiates the scores
    const decision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-vision-score-diff",
      { requiredModalities: ["text", "vision"] as Modality[] },
    );

    assert.ok(decision);
    assert.ok(decision!.candidates, "Should have candidates");

    // Find two vision-capable candidates and compare their scores
    const allCandidates = [
      { provider: decision!.provider, model: decision!.model, score: decision!.overallScore },
      ...decision!.candidates!.map((c) => ({ provider: c.provider, model: c.model, score: c.overallScore })),
    ];

    const visionCandidates = allCandidates.filter((c) => {
      const cap = registry.getCapability(c.provider, c.model);
      return cap?.modalities.includes("vision");
    });

    assert.ok(visionCandidates.length >= 2, "Should have at least 2 vision candidates");

    // Verify that candidates with different visionQuality have different scores
    // (the scoring adjustment should create differentiation)
    const glmScore = visionCandidates.find((c) => c.model === "glm-4.6v");
    const moondreamScore = visionCandidates.find((c) => c.model === "moondream:latest");

    if (glmScore && moondreamScore) {
      assert.ok(
        glmScore.score > moondreamScore.score,
        `glm-4.6v (quality=0.85) should score higher than moondream (quality=0.50): ` +
        `${glmScore.score.toFixed(4)} vs ${moondreamScore.score.toFixed(4)}`,
      );
    }
  });

  it("should not apply vision quality adjustment for text-only requests", async () => {
    const textDecision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-text-no-vision-adjust",
      { requiredModalities: ["text"] as Modality[] },
    );

    assert.ok(textDecision);
    // Text-only requests should not have vision quality info in modalityFilter
    assert.equal(textDecision!.modalityFilter, undefined);
    // Rationale should not include visionQ
    assert.ok(!textDecision!.rationale.includes("visionQ"),
      "Text-only rationale should not include visionQ");
  });

  it("should include vision quality in rationale for vision requests", async () => {
    const decision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-vision-rationale",
      { requiredModalities: ["text", "vision"] as Modality[] },
    );

    assert.ok(decision);
    // The winner's rationale should include visionQ if it's a vision model
    const winnerCap = registry.getCapability(decision!.provider, decision!.model);
    if (winnerCap?.visionQuality !== undefined) {
      assert.ok(decision!.rationale.includes("visionQ"),
        `Winner rationale should include visionQ: ${decision!.rationale}`);
    }
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

  it("should set supportsVision flag on vision-capable models", () => {
    const visionModels = registry.getAllModels().filter((m) => m.modalities.includes("vision"));
    for (const m of visionModels) {
      assert.equal(m.supportsVision, true,
        `${m.provider}/${m.model} should have supportsVision=true`);
    }

    // Text-only models should NOT have supportsVision=true
    const textOnly = registry.getAllModels().filter((m) => !m.modalities.includes("vision"));
    for (const m of textOnly) {
      assert.ok(!m.supportsVision,
        `${m.provider}/${m.model} should not have supportsVision=true`);
    }
  });

  it("should assign visionQuality scores to vision-capable models", () => {
    const visionModels = registry.getAllModels().filter((m) => m.modalities.includes("vision"));
    for (const m of visionModels) {
      assert.ok(m.visionQuality !== undefined,
        `${m.provider}/${m.model} should have visionQuality defined`);
      assert.ok(m.visionQuality! > 0 && m.visionQuality! <= 1,
        `${m.provider}/${m.model} visionQuality should be in (0, 1], got ${m.visionQuality}`);
    }
  });

  it("should NOT assign visionQuality to text-only models", () => {
    const textOnly = registry.getAllModels().filter((m) => !m.modalities.includes("vision"));
    for (const m of textOnly) {
      assert.ok(m.visionQuality === undefined,
        `${m.provider}/${m.model} should not have visionQuality`);
    }
  });

  it("should score known vision models with correct quality tiers", () => {
    const expected = {
      "zai/glm-4.6v": 0.85,
      "ollama/gemma3:12b": 0.75,
      "ollama/gemma3:4b": 0.60,
      "ollama/gemma3:1b": 0.40,
      "ollama/llama3.2-vision:11b": 0.70,
      "ollama/moondream:latest": 0.50,
    };

    for (const [key, expectedScore] of Object.entries(expected)) {
      const [provider, ...modelParts] = key.split("/");
      const model = modelParts.join("/");
      const cap = registry.getCapability(provider, model);
      assert.ok(cap, `${key} should exist`);
      assert.equal(cap!.visionQuality, expectedScore,
        `${key} should have visionQuality=${expectedScore}, got ${cap!.visionQuality}`);
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

// ─── Seed Data — Audio Capability Flags Tests ───────────────

describe("Seed Data — Audio Capability Flags", () => {
  let registry: ModelRegistry;
  let db: DBService;
  let config: CognitiveRouterConfig;

  beforeEach(async () => {
    config = makeConfig();
    db = makeMockDB();
    registry = new ModelRegistry(db, config);
    await registry.loadCachedState();
  });

  it("should have at least 1 audio-capable model in seed data", () => {
    const allModels = registry.getAllModels();
    const audioModels = allModels.filter((m) => m.modalities.includes("audio"));

    assert.ok(
      audioModels.length >= 1,
      `Expected at least 1 audio model, got ${audioModels.length}: ` +
      audioModels.map((m) => `${m.provider}/${m.model}`).join(", "),
    );
  });

  it("should correctly flag glm-4.6v as audio-capable", () => {
    const cap = registry.getCapability("zai", "glm-4.6v");
    assert.ok(cap, "glm-4.6v should exist in registry");
    assert.ok(
      cap!.modalities.includes("audio"),
      `glm-4.6v should have audio modality, got [${cap!.modalities.join(", ")}]`,
    );
    assert.equal(cap!.supportsAudio, true,
      "glm-4.6v should have supportsAudio=true");
  });

  it("should set supportsAudio flag on audio-capable models", () => {
    const audioModels = registry.getAllModels().filter((m) => m.modalities.includes("audio"));
    for (const m of audioModels) {
      assert.equal(m.supportsAudio, true,
        `${m.provider}/${m.model} should have supportsAudio=true`);
    }

    // Non-audio models should NOT have supportsAudio=true
    const nonAudio = registry.getAllModels().filter((m) => !m.modalities.includes("audio"));
    for (const m of nonAudio) {
      assert.ok(!m.supportsAudio,
        `${m.provider}/${m.model} should not have supportsAudio=true`);
    }
  });

  it("should NOT flag text-only or vision-only models as audio-capable", () => {
    const nonAudioModels = [
      "zai/glm-5.2",
      "zai/glm-4.7",
      "ollama/gemma3:12b",   // vision-only
      "ollama/moondream:latest", // vision-only
      "ollama/gemma4:latest",
    ];

    for (const key of nonAudioModels) {
      const [provider, ...modelParts] = key.split("/");
      const model = modelParts.join("/");
      const cap = registry.getCapability(provider, model);
      if (cap) {
        assert.ok(
          !cap.modalities.includes("audio"),
          `${key} should NOT have audio modality`,
        );
        assert.ok(!cap.supportsAudio,
          `${key} should NOT have supportsAudio=true`);
      }
    }
  });

  it("should have glm-4.6v with both vision and audio modalities", () => {
    const cap = registry.getCapability("zai", "glm-4.6v");
    assert.ok(cap);
    assert.ok(cap!.modalities.includes("text"));
    assert.ok(cap!.modalities.includes("vision"));
    assert.ok(cap!.modalities.includes("audio"));
    assert.equal(cap!.supportsVision, true);
    assert.equal(cap!.supportsAudio, true);
  });
});

// ─── Router Audio Filtering Tests ───────────────────────────

describe("RoutingEngine — Audio Model Filtering", () => {
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

  it("should filter to audio-capable models when audio is present", async () => {
    const decision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-audio-session",
      { requiredModalities: ["text", "audio"] as Modality[] },
    );

    assert.ok(decision, "Should get a routing decision");
    assert.ok(!decision!.error, `Should not error: ${decision!.error?.message}`);

    // Selected model must support audio
    const selectedCap = registry.getCapability(decision!.provider, decision!.model);
    assert.ok(selectedCap, "Selected model should exist in registry");
    assert.ok(
      selectedCap!.modalities.includes("audio"),
      `Selected model ${decision!.provider}/${decision!.model} must support audio, ` +
      `but only has [${selectedCap!.modalities.join(", ")}]`,
    );

    // Modality filter info should be present
    assert.ok(decision!.modalityFilter, "Decision should include modalityFilter");
    assert.equal(decision!.modalityFilter!.wasMultimodal, true);
    assert.ok(decision!.modalityFilter!.requiredModalities.includes("audio"));

    // All filtered-out models should lack audio
    for (const filtered of decision!.modalityFilter!.filteredOut) {
      assert.ok(
        !filtered.modalities.includes("audio"),
        `${filtered.provider}/${filtered.model} should not have audio`,
      );
    }
  });

  it("should return MODALITY_UNSUPPORTED when no audio model is available", async () => {
    // Use openrouter-only config — no audio models there
    const noAudioConfig = makeConfig({
      providerPriority: ["openrouter"],
      providers: {
        openrouter: { budgetType: "free", priority: "high" },
      },
    });
    const noAudioRouter = new RoutingEngine(registry, costTracker, db, noAudioConfig);

    const decision = await noAudioRouter.decide(
      makeClassification("conversation", 0.9),
      "test-no-audio-session",
      { requiredModalities: ["text", "audio"] as Modality[] },
    );

    assert.ok(decision, "Should get a decision (error decision)");
    assert.ok(decision!.error, "Should have an error");
    assert.equal(decision!.error!.code, "MODALITY_UNSUPPORTED");
    assert.ok(
      decision!.error!.message.includes("audio"),
      `Error message should mention audio: ${decision!.error!.message}`,
    );
    assert.ok(
      Array.isArray(decision!.error!.availableModalityModels),
      "Error should list available modality models",
    );
  });

  it("should return MODALITY_UNSUPPORTED with ollama-only config for audio", async () => {
    // Ollama has no audio models in seed data
    const ollamaOnlyConfig = makeConfig({
      providerPriority: ["ollama"],
      providers: {
        ollama: { budgetType: "free", priority: "low" },
      },
    });
    const ollamaRouter = new RoutingEngine(registry, costTracker, db, ollamaOnlyConfig);

    const decision = await ollamaRouter.decide(
      makeClassification("conversation", 0.9),
      "test-ollama-no-audio",
      { requiredModalities: ["text", "audio"] as Modality[] },
    );

    assert.ok(decision);
    assert.ok(decision!.error);
    assert.equal(decision!.error!.code, "MODALITY_UNSUPPORTED");
    assert.ok(decision!.error!.message.includes("audio"));
  });

  it("should not affect text-only requests (audio filter not applied)", async () => {
    const textDecision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-audio-text-only",
      { requiredModalities: ["text"] as Modality[] },
    );

    assert.ok(textDecision);
    assert.equal(
      textDecision!.modalityFilter,
      undefined,
      "Text-only requests should not have modalityFilter",
    );
  });

  it("should filter out ALL non-audio candidates, not just some", async () => {
    const decision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-audio-filter-completeness",
      { requiredModalities: ["text", "audio"] as Modality[] },
    );

    assert.ok(decision);
    assert.ok(decision!.modalityFilter);

    // Every filtered-out model should NOT have audio
    for (const f of decision!.modalityFilter!.filteredOut) {
      assert.ok(!f.modalities.includes("audio"),
        `${f.provider}/${f.model} was filtered but has audio — bug!`);
    }
  });
});

// ─── Mixed Modality (Vision + Audio) Routing Tests ─────────

describe("RoutingEngine — Mixed Modality Routing", () => {
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

  it("should route vision+audio request to a model supporting both", async () => {
    const decision = await router.decide(
      makeClassification("conversation", 0.9),
      "test-vision-audio-routing",
      { requiredModalities: ["text", "vision", "audio"] as Modality[] },
    );

    assert.ok(decision);
    assert.ok(!decision!.error, `Should not error: ${decision!.error?.message}`);

    // Selected model must support BOTH vision AND audio
    const selectedCap = registry.getCapability(decision!.provider, decision!.model);
    assert.ok(selectedCap);
    assert.ok(
      selectedCap!.modalities.includes("vision"),
      `Model must support vision: ${selectedCap!.modalities.join(", ")}`,
    );
    assert.ok(
      selectedCap!.modalities.includes("audio"),
      `Model must support audio: ${selectedCap!.modalities.join(", ")}`,
    );
  });

  it("should return MODALITY_UNSUPPORTED when no model supports all modalities", async () => {
    // Use ollama-only config: ollama has vision models but no audio models,
    // so a vision+audio request should fail.
    const ollamaOnlyConfig = makeConfig({
      providerPriority: ["ollama"],
      providers: {
        ollama: { budgetType: "free", priority: "low" },
      },
    });
    const ollamaRouter = new RoutingEngine(registry, costTracker, db, ollamaOnlyConfig);

    const decision = await ollamaRouter.decide(
      makeClassification("conversation", 0.9),
      "test-mixed-no-model",
      { requiredModalities: ["text", "vision", "audio"] as Modality[] },
    );

    assert.ok(decision);
    assert.ok(decision!.error);
    assert.equal(decision!.error!.code, "MODALITY_UNSUPPORTED");
    assert.ok(decision!.error!.message.includes("vision"));
    assert.ok(decision!.error!.message.includes("audio"));
  });

  it("should include all required modalities in error info", async () => {
    const ollamaOnlyConfig = makeConfig({
      providerPriority: ["ollama"],
      providers: {
        ollama: { budgetType: "free", priority: "low" },
      },
    });
    const ollamaRouter = new RoutingEngine(registry, costTracker, db, ollamaOnlyConfig);

    const decision = await ollamaRouter.decide(
      makeClassification("conversation", 0.9),
      "test-mixed-error-info",
      { requiredModalities: ["text", "vision", "audio"] as Modality[] },
    );

    assert.ok(decision!.error);
    assert.ok(Array.isArray(decision!.error!.requiredModalities));
    assert.ok(decision!.error!.requiredModalities!.includes("vision"));
    assert.ok(decision!.error!.requiredModalities!.includes("audio"));
  });
});
