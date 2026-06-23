// tests/classifier.test.ts — Unit tests for classifier.ts
// Run with: npx tsx --test tests/classifier.test.ts

import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { IntentClassifier, type Classification } from "../src/classifier.ts";

// ─── Tests ──────────────────────────────────────────────────

describe("IntentClassifier — classifyByKeyword", () => {
  let classifier: IntentClassifier;

  beforeEach(() => {
    classifier = new IntentClassifier({ tiebreakerThreshold: 0.7 });
  });

  it("should classify coding prompts correctly", () => {
    const result = classifier.classifyByKeyword("fix this bug in the API code");
    assert.equal(result.intent, "coding");
    assert.ok(result.confidence >= 0.4, `Confidence should be >= 0.4, got ${result.confidence}`);
  });

  it("should classify research prompts correctly", () => {
    const result = classifier.classifyByKeyword("find papers on quantum computing and compare latest developments");
    assert.equal(result.intent, "research");
    assert.ok(result.confidence >= 0.4);
  });

  it("should classify creative prompts correctly", () => {
    const result = classifier.classifyByKeyword("write a poem about autumn leaves");
    assert.equal(result.intent, "creative");
    assert.ok(result.confidence >= 0.4);
  });

  it("should classify conversation prompts correctly", () => {
    const result = classifier.classifyByKeyword("hey how are you doing today");
    assert.equal(result.intent, "conversation");
    assert.ok(result.confidence >= 0.4);
  });

  it("should classify summary prompts correctly", () => {
    const result = classifier.classifyByKeyword("summarize this article for me, give me key points");
    assert.equal(result.intent, "summary");
    assert.ok(result.confidence >= 0.4);
  });

  it("should classify retrieval prompts correctly", () => {
    const result = classifier.classifyByKeyword("what is the capital of brazil");
    assert.equal(result.intent, "retrieval");
    assert.ok(result.confidence >= 0.4);
  });

  it("should classify science prompts correctly", () => {
    const result = classifier.classifyByKeyword("derive the equation for CRISPR protein folding");
    assert.equal(result.intent, "science");
    assert.ok(result.confidence >= 0.4);
  });

  it("should classify business prompts correctly", () => {
    const result = classifier.classifyByKeyword("create a market strategy for a SaaS business plan");
    assert.equal(result.intent, "business");
    assert.ok(result.confidence >= 0.4);
  });

  it("should classify math prompts correctly", () => {
    const result = classifier.classifyByKeyword("solve the integral and calculate the eigenvalue matrix");
    assert.equal(result.intent, "math");
    assert.ok(result.confidence >= 0.4);
  });

  it("should classify analysis prompts correctly", () => {
    const result = classifier.classifyByKeyword("evaluate the trade-offs and assess the feasibility risks");
    assert.equal(result.intent, "analysis");
    assert.ok(result.confidence >= 0.4);
  });

  it("should default to 'conversation' for ambiguous prompts with no keyword matches", () => {
    const result = classifier.classifyByKeyword("the weather is mild today in the valley");
    assert.equal(result.intent, "conversation");
    assert.ok(result.confidence >= 0.4, "Even unmatched prompts should have confidence >= 0.4");
  });

  it("should always return confidence >= 0.4", () => {
    // Test a variety of prompts
    const prompts = [
      "hello world",
      "the quick brown fox",
      "a completely unrelated string of words",
      "random text without any matching keywords whatsoever",
    ];
    for (const prompt of prompts) {
      const result = classifier.classifyByKeyword(prompt);
      assert.ok(
        result.confidence >= 0.4,
        `Confidence for "${prompt}" should be >= 0.4, got ${result.confidence}`,
      );
    }
  });

  it("should handle case-insensitive matching", () => {
    const result = classifier.classifyByKeyword("FIX THIS BUG IN THE CODE");
    assert.equal(result.intent, "coding");
  });

  it("should return higher confidence for prompts with more keyword hits", () => {
    const lowHit = classifier.classifyByKeyword("fix this bug");
    const highHit = classifier.classifyByKeyword("fix this bug in the code, refactor the class and add a test for the API error");
    // Both should be coding, but the one with more keywords should have higher confidence
    assert.equal(lowHit.intent, "coding");
    assert.equal(highHit.intent, "coding");
    assert.ok(
      highHit.confidence >= lowHit.confidence,
      `High-hit confidence (${highHit.confidence}) should be >= low-hit (${lowHit.confidence})`,
    );
  });
});

describe("IntentClassifier — isInitialized()", () => {
  it("should return false before initialize() is called", () => {
    const classifier = new IntentClassifier({ tiebreakerThreshold: 0.7 });
    assert.equal(classifier.isInitialized(), false);
  });

  it("should return true after initialize() completes", async () => {
    const classifier = new IntentClassifier({ tiebreakerThreshold: 0.7 });
    // Provide a mock embed function that returns a fixed vector
    const mockEmbed = async (_text: string): Promise<number[]> => [1, 0, 0];
    await classifier.initialize(mockEmbed);
    assert.equal(classifier.isInitialized(), true);
  });
});

describe("IntentClassifier — initialize()", () => {
  it("should call the embed function for each prototype example", async () => {
    const classifier = new IntentClassifier({ tiebreakerThreshold: 0.7 });
    let callCount = 0;
    const mockEmbed = async (_text: string): Promise<number[]> => {
      callCount++;
      return [Math.random(), 0, 0];
    };

    await classifier.initialize(mockEmbed);

    // The classifier has 10 intent categories with varying numbers of examples.
    // We just need to verify embed was called multiple times — at least once
    // for each intent category.
    const INTENT_COUNT = 10;
    assert.ok(
      callCount >= INTENT_COUNT,
      `Embed function should be called at least ${INTENT_COUNT} times (once per intent), got ${callCount}`,
    );
  });

  it("should complete successfully even if some embed calls fail", async () => {
    const classifier = new IntentClassifier({ tiebreakerThreshold: 0.7 });
    let callCount = 0;
    const mockEmbed = async (_text: string): Promise<number[]> => {
      callCount++;
      // Fail every other call
      if (callCount % 2 === 0) throw new Error("Mock embed failure");
      return [1, 0, 0];
    };

    // Should not throw
    await classifier.initialize(mockEmbed);
    assert.equal(classifier.isInitialized(), true);
  });
});

describe("IntentClassifier — classify() fallback", () => {
  it("should fall back to keyword classification when embedding throws", async () => {
    const classifier = new IntentClassifier({ tiebreakerThreshold: 0.7 });

    // Initialize with a working embed function
    const initEmbed = async (_text: string): Promise<number[]> => [1, 0, 0];
    await classifier.initialize(initEmbed);

    // Now classify with a failing embed function
    const failingEmbed = async (_text: string): Promise<number[]> => {
      throw new Error("Embedding service unavailable");
    };

    const result = await classifier.classify("fix this bug in the code", failingEmbed);

    // Should fall back to keyword matching
    assert.equal(result.intent, "coding");
    assert.ok(result.confidence >= 0.4, "Fallback confidence should be >= 0.4");
  });

  it("should return a valid Classification object with intent and confidence", async () => {
    const classifier = new IntentClassifier({ tiebreakerThreshold: 0.7 });
    const mockEmbed = async (_text: string): Promise<number[]> => [Math.random(), 0, 0];
    await classifier.initialize(mockEmbed);

    const result = await classifier.classify("hello world", mockEmbed);

    assert.ok(typeof result.intent === "string", "Intent should be a string");
    assert.ok(typeof result.confidence === "number", "Confidence should be a number");
    assert.ok(Number.isFinite(result.confidence), "Confidence should be finite");
    assert.ok(result.confidence >= 0, "Confidence should be non-negative");
    assert.ok(result.confidence <= 1, "Confidence should be <= 1");
  });

  it("should auto-initialize on first classify() call if not initialized", async () => {
    const classifier = new IntentClassifier({ tiebreakerThreshold: 0.7 });
    assert.equal(classifier.isInitialized(), false);

    const mockEmbed = async (_text: string): Promise<number[]> => [1, 0, 0];
    const result = await classifier.classify("fix this bug", mockEmbed);

    // After classify, it should be initialized
    assert.equal(classifier.isInitialized(), true);
    assert.ok(typeof result.intent === "string");
  });
});
