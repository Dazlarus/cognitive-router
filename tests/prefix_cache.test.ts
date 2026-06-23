// tests/prefix_cache.test.ts — Unit tests for prefix prefix caching
// Run with: npx tsx --test tests/prefix_cache.test.ts

import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import {
  PrefixCache,
  extractSystemPrompt,
  fingerprintPrefix,
  estimateSystemTokens,
  isCacheable,
  observeZaIPrefix,
  maybeGeminiCache,
} from "../src/prefix_cache.ts";

// ─── Helpers ───

/** Generate a large system prompt of approximately N tokens. */
function makeSystemPrompt(approxTokens: number): string {
  const words = [];
  for (let i = 0; i < approxTokens; i++) {
    words.push(`word${i}`);
  }
  return words.join(" ");
}

/** Standard test prompt (~4000 tokens, well above the 2048 threshold). */
const LARGE_PROMPT = makeSystemPrompt(4000);
const SMALL_PROMPT = "You are a helpful assistant.";

// ─── Pure Function Tests ───

describe("extractSystemPrompt", () => {
  it("should extract only system messages", () => {
    const messages = [
      { role: "system", content: "You are helpful." },
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi" },
      { role: "system", content: "Be concise." },
    ];
    assert.equal(extractSystemPrompt(messages), "You are helpful.\n\nBe concise.");
  });

  it("should return empty string when no system messages", () => {
    const messages = [
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi" },
    ];
    assert.equal(extractSystemPrompt(messages), "");
  });

  it("should handle empty message array", () => {
    assert.equal(extractSystemPrompt([]), "");
  });
});

describe("fingerprintPrefix", () => {
  it("should produce a deterministic hex hash", () => {
    const fp = fingerprintPrefix(LARGE_PROMPT);
    assert.equal(fp.length, 32, "Fingerprint should be 32 chars (truncated SHA-256)");
    assert.match(fp, /^[0-9a-f]+$/, "Should be hex");
  });

  it("should produce the same hash for identical content", () => {
    assert.equal(fingerprintPrefix(LARGE_PROMPT), fingerprintPrefix(LARGE_PROMPT));
  });

  it("should produce different hashes for different content", () => {
    const promptA = "alpha".repeat(2000);
    const promptB = "beta".repeat(2000);
    assert.notEqual(fingerprintPrefix(promptA), fingerprintPrefix(promptB));
  });

  it("should normalize whitespace before hashing", () => {
    const prompt1 = "foo\n\n  bar\t\tbaz";
    const prompt2 = "foo bar baz";
    // After normalization both collapse to "foo bar baz" (within first 8192 chars)
    // But actually prompt1 has the extra whitespace stripped, so they should match
    // only if the content after normalization is the same
    assert.equal(fingerprintPrefix(prompt1), fingerprintPrefix(prompt2));
  });

  it("should only hash the first maxChars", () => {
    const prompt = "x".repeat(10000);
    const fp1 = fingerprintPrefix(prompt, 8192);
    const fp2 = fingerprintPrefix(prompt + "extra", 8192);
    assert.equal(fp1, fp2, "Should only hash first maxChars");
  });
});

describe("estimateSystemTokens", () => {
  it("should estimate ~chars/4 tokens", () => {
    assert.equal(estimateSystemTokens("abcd"), 1);
    assert.equal(estimateSystemTokens("abcdefgh"), 2);
    assert.equal(estimateSystemTokens(""), 0);
  });
});

describe("isCacheable", () => {
  it("should return true for large prompts", () => {
    assert.ok(isCacheable(LARGE_PROMPT));
  });

  it("should return false for short prompts", () => {
    assert.ok(!isCacheable(SMALL_PROMPT));
  });

  it("should return false for empty string", () => {
    assert.ok(!isCacheable(""));
  });
});

// ─── PrefixCache LRU + TTL Tests ───

describe("PrefixCache", () => {
  let cache: PrefixCache;

  beforeEach(() => {
    // Get a fresh instance by clearing the singleton
    cache = PrefixCache.instance;
    cache.clear();
  });

  afterEach(() => {
    cache.clear();
  });

  describe("observe()", () => {
    it("should return null for non-cacheable (short) prompts", () => {
      const entry = cache.observe(SMALL_PROMPT);
      assert.equal(entry, null);
    });

    it("should create a new entry on first sighting", () => {
      const entry = cache.observe(LARGE_PROMPT);
      assert.ok(entry);
      assert.equal(entry!.hitCount, 1);
    });

    it("should increment hit count on subsequent sightings", () => {
      cache.observe(LARGE_PROMPT);
      const entry = cache.observe(LARGE_PROMPT);
      assert.equal(entry!.hitCount, 2);
    });

    it("should activate caching after 2nd sighting", () => {
      cache.observe(LARGE_PROMPT);
      assert.ok(!cache.shouldCache(LARGE_PROMPT), "Should not cache after 1 sighting");
      cache.observe(LARGE_PROMPT);
      assert.ok(cache.shouldCache(LARGE_PROMPT), "Should cache after 2 sightings");
    });

    it("should track multiple distinct prefixes", () => {
      const promptA = "alpha".repeat(2000);
      const promptB = "beta".repeat(2000);
      cache.observe(promptA);
      cache.observe(promptB);
      cache.observe(promptA);
      assert.ok(cache.shouldCache(promptA));
      assert.ok(!cache.shouldCache(promptB), "B only seen once");
    });
  });

  describe("shouldCache()", () => {
    it("should return false for unknown prefixes", () => {
      assert.ok(!cache.shouldCache(LARGE_PROMPT));
    });

    it("should return false for non-cacheable prompts", () => {
      assert.ok(!cache.shouldCache(SMALL_PROMPT));
    });
  });

  describe("getStats()", () => {
    it("should track stats correctly", () => {
      let stats = cache.getStats();
      assert.equal(stats.totalPrefixes, 0);
      assert.equal(stats.totalHits, 0);

      cache.observe(LARGE_PROMPT);
      stats = cache.getStats();
      assert.equal(stats.totalPrefixes, 1);
      assert.equal(stats.totalHits, 0); // first sighting is not a "hit"
      assert.equal(stats.activeCached, 0);

      cache.observe(LARGE_PROMPT);
      stats = cache.getStats();
      assert.equal(stats.totalPrefixes, 1);
      assert.equal(stats.totalHits, 1); // second sighting is a hit
      assert.equal(stats.activeCached, 1);
      assert.equal(stats.totalCacheActivations, 1);
    });
  });

  describe("LRU eviction", () => {
    it("should enforce max entries", () => {
      // Add many distinct prefixes
      for (let i = 0; i < 40; i++) {
        cache.observe(makeSystemPrompt(3000 + i));
      }
      const stats = cache.getStats();
      assert.ok(stats.totalPrefixes <= 32, `Should have at most 32 entries, got ${stats.totalPrefixes}`);
    });
  });

  describe("clear()", () => {
    it("should reset all state", () => {
      cache.observe(LARGE_PROMPT);
      cache.observe(LARGE_PROMPT);
      cache.clear();
      const stats = cache.getStats();
      assert.equal(stats.totalPrefixes, 0);
      assert.equal(stats.totalHits, 0);
      assert.ok(!cache.shouldCache(LARGE_PROMPT));
    });
  });
});

// ─── Gemini Cache Integration Tests ───

describe("Gemini cache integration", () => {
  let cache: PrefixCache;

  beforeEach(() => {
    cache = PrefixCache.instance;
    cache.clear();
  });

  afterEach(() => {
    cache.clear();
  });

  describe("maybeGeminiCache()", () => {
    const originalFetch = globalThis.fetch;

    afterEach(() => {
      globalThis.fetch = originalFetch;
    });

    it("should not attempt cache creation on first sighting", async () => {
      let fetchCalls = 0;
      globalThis.fetch = (() => { fetchCalls++; return Promise.resolve(new Response("{}", { status: 200 })); }) as typeof fetch;

      const messages = [{ role: "system", content: LARGE_PROMPT }];
      await maybeGeminiCache("gemini-2.5-flash", messages, "fake-key");

      assert.equal(fetchCalls, 0, "Should not make API calls on first sighting");
    });

    it("should attempt cachedContent creation on second sighting", async () => {
      let cacheCreateCalls = 0;
      globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        if (url.includes("cachedContents") && init?.method === "POST") {
          cacheCreateCalls++;
          return new Response(JSON.stringify({
            name: "cachedContents/test-123",
            model: "models/gemini-2.5-flash",
          }), { status: 200, headers: { "Content-Type": "application/json" } });
        }
        return new Response("{}", { status: 200 });
      }) as typeof fetch;

      const messages = [{ role: "system", content: LARGE_PROMPT }];

      // First sighting — no cache creation
      await maybeGeminiCache("gemini-2.5-flash", messages, "fake-key");
      assert.equal(cacheCreateCalls, 0);

      // Second sighting — should create cachedContent
      await maybeGeminiCache("gemini-2.5-flash", messages, "fake-key");
      assert.equal(cacheCreateCalls, 1);
    });

    it("should reuse cachedContent on third+ sighting", async () => {
      let cacheCreateCalls = 0;
      globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        if (url.includes("cachedContents") && init?.method === "POST") {
          cacheCreateCalls++;
          return new Response(JSON.stringify({
            name: "cachedContents/test-456",
          }), { status: 200, headers: { "Content-Type": "application/json" } });
        }
        return new Response("{}", { status: 200 });
      }) as typeof fetch;

      const messages = [{ role: "system", content: LARGE_PROMPT }];

      await maybeGeminiCache("gemini-2.5-flash", messages, "fake-key"); // 1st
      await maybeGeminiCache("gemini-2.5-flash", messages, "fake-key"); // 2nd — creates cache
      await maybeGeminiCache("gemini-2.5-flash", messages, "fake-key"); // 3rd — should reuse
      await maybeGeminiCache("gemini-2.5-flash", messages, "fake-key"); // 4th — should reuse

      assert.equal(cacheCreateCalls, 1, "Should only create cache once");
    });

    it("should handle cachedContent creation failure gracefully", async () => {
      let cacheCreateCalls = 0;
      globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        if (url.includes("cachedContents") && init?.method === "POST") {
          cacheCreateCalls++;
          return new Response("Quota exceeded", { status: 429 });
        }
        return new Response("{}", { status: 200 });
      }) as typeof fetch;

      const messages = [{ role: "system", content: LARGE_PROMPT }];

      await maybeGeminiCache("gemini-2.5-flash", messages, "fake-key"); // 1st
      const result = await maybeGeminiCache("gemini-2.5-flash", messages, "fake-key"); // 2nd — fails

      assert.equal(cacheCreateCalls, 1, "Should have attempted creation");
      assert.equal(result.cachedContentName, undefined, "Should not have cache name on failure");
      // Should still work (graceful fallback)
      const stats = cache.getStats();
      assert.equal(stats.geminiCacheErrors, 1);
    });

    it("should not attempt caching for small prompts", async () => {
      const messages = [{ role: "system", content: SMALL_PROMPT }];
      const result = await maybeGeminiCache("gemini-2.5-flash", messages, "fake-key");
      assert.equal(result.cachedContentName, undefined);
      assert.equal(result.systemPrompt, SMALL_PROMPT);
    });
  });

  describe("PrefixCache.getGeminiCacheName()", () => {
    it("should return undefined when no cache exists", () => {
      assert.equal(cache.getGeminiCacheName(LARGE_PROMPT), undefined);
    });

    it("should return cache name after creation", () => {
      // Simulate a recorded cache
      cache.observe(LARGE_PROMPT);
      cache.observe(LARGE_PROMPT);
      cache.recordGeminiCache(LARGE_PROMPT, "cachedContents/test-789", "gemini-2.5-flash");

      const name = cache.getGeminiCacheName(LARGE_PROMPT);
      assert.equal(name, "cachedContents/test-789");
    });

    it("should return undefined for expired caches", async () => {
      cache.observe(LARGE_PROMPT);
      cache.observe(LARGE_PROMPT);
      cache.recordGeminiCache(LARGE_PROMPT, "cachedContents/expired", "gemini-2.5-flash");

      // Manually expire by setting expiry in the past
      const fp = fingerprintPrefix(LARGE_PROMPT);
      const entry = (cache as any).entries.get(fp) as any;
      entry.geminiCacheExpiry = Date.now() - 1000;

      assert.equal(cache.getGeminiCacheName(LARGE_PROMPT), undefined);
    });
  });
});

// ─── ZAI Prefix Observation Tests ───

describe("observeZaIPrefix()", () => {
  let cache: PrefixCache;

  beforeEach(() => {
    cache = PrefixCache.instance;
    cache.clear();
  });

  afterEach(() => {
    cache.clear();
  });

  it("should observe and track prefixes", () => {
    const messages = [{ role: "system", content: LARGE_PROMPT }];

    const result1 = observeZaIPrefix(messages);
    assert.equal(result1.systemPrompt, LARGE_PROMPT);
    assert.ok(!result1.cacheExpected, "Should not expect cache on first sighting");

    const result2 = observeZaIPrefix(messages);
    assert.ok(result2.cacheExpected, "Should expect cache on second sighting");
  });

  it("should handle non-cacheable prompts", () => {
    const messages = [{ role: "system", content: SMALL_PROMPT }];
    const result = observeZaIPrefix(messages);
    assert.equal(result.cacheExpected, false);
  });

  it("should handle messages without system prompt", () => {
    const messages = [{ role: "user", content: "Hello" }];
    const result = observeZaIPrefix(messages);
    assert.equal(result.systemPrompt, "");
    assert.equal(result.cacheExpected, false);
  });
});

// ─── Gemini Adapter Integration (request building) ───

describe("Gemini adapter prefix caching (request building)", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
    PrefixCache.instance.clear();
  });

  it("should include cachedContent field when cache is available", async () => {
    const cache = PrefixCache.instance;
    cache.clear();

    // Simulate cache being active (2 observations + successful creation)
    cache.observe(LARGE_PROMPT);
    cache.observe(LARGE_PROMPT);
    cache.recordGeminiCache(LARGE_PROMPT, "cachedContents/abc-123", "gemini-2.5-flash");

    let capturedBody: any = null;
    globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      capturedBody = JSON.parse(String(init?.body));
      return new Response(JSON.stringify({
        candidates: [{
          content: { parts: [{ text: "cached!" }] },
          finishReason: "STOP",
        }],
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }) as typeof fetch;

    // Import Gemini adapter
    const { GeminiAdapter } = await import("../src/providers.ts");

    await GeminiAdapter.chatCompletion(
      "gemini-2.5-flash",
      {
        model: "gemini-2.5-flash",
        messages: [
          { role: "system", content: LARGE_PROMPT },
          { role: "user", content: "Hello" },
        ],
      },
      "test-key",
    );

    assert.ok(capturedBody.cachedContent, "Should include cachedContent field");
    assert.equal(capturedBody.cachedContent, "cachedContents/abc-123");
    assert.equal(capturedBody.systemInstruction, undefined, "Should NOT include systemInstruction when cachedContent is used");
  });

  it("should include systemInstruction when no cache is available", async () => {
    PrefixCache.instance.clear();

    let capturedBody: any = null;
    globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      capturedBody = JSON.parse(String(init?.body));
      return new Response(JSON.stringify({
        candidates: [{
          content: { parts: [{ text: "not cached!" }] },
          finishReason: "STOP",
        }],
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }) as typeof fetch;

    const { GeminiAdapter } = await import("../src/providers.ts");

    await GeminiAdapter.chatCompletion(
      "gemini-2.5-flash",
      {
        model: "gemini-2.5-flash",
        messages: [
          { role: "system", content: LARGE_PROMPT },
          { role: "user", content: "Hello" },
        ],
      },
      "test-key",
    );

    assert.equal(capturedBody.cachedContent, undefined, "Should NOT include cachedContent");
    assert.ok(capturedBody.systemInstruction, "Should include systemInstruction");
  });
});
