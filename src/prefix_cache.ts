// src/prefix_cache.ts — Prompt prefix fingerprinting + caching coordinator
//
// OpenClaw sends the same system prompt (AGENTS.md, SOUL.md, etc.) in every
// request — 15-20K tokens of identical prefix. This module detects repeated
// prefixes and coordinates provider-specific caching mechanisms to cut latency
// and cost.
//
// Design:
// - Prefix fingerprint: SHA-256 of the first N tokens of the system message(s)
// - Frequency tracking: in-memory LRU with 1h TTL
// - After 2nd sighting of the same prefix, caching is activated
// - Gemini: manages cachedContent resources via the Gemini API
// - ZAI: GLM-4+ supports automatic prefix caching — we log and track but don't
//   need an explicit API call. We structure the request to maximize cache hits.
// - OpenRouter/Ollama: no explicit caching API; graceful no-op.

import crypto from "node:crypto";
import { logger } from "./logger.js";

// ─── Types ───

export interface PrefixCacheEntry {
  /** SHA-256 hash of the normalized system prompt prefix */
  fingerprint: string;
  /** Number of times this prefix has been seen */
  hitCount: number;
  /** First-seen timestamp (epoch ms) */
  firstSeen: number;
  /** Last-seen timestamp (epoch ms) */
  lastSeen: number;
  /** Gemini cachedContent resource name (lazy-created on 2nd hit) */
  geminiCachedContentName?: string;
  /** Whether cachedContent creation has been attempted */
  geminiCacheAttempted: boolean;
  /** Model used when creating the Gemini cache (caches are model-specific) */
  geminiCacheModel?: string;
  /** Expiry timestamp for the Gemini cache (epoch ms) */
  geminiCacheExpiry?: number;
}

export interface PrefixCacheStats {
  totalPrefixes: number;
  activeCached: number;
  totalHits: number;
  totalCacheActivations: number;
  geminiCacheCreations: number;
  geminiCacheHits: number;
  geminiCacheErrors: number;
}

// ─── Constants ───

/** Minimum prefix length (in chars) to bother caching. Below this, skip. */
const MIN_PREFIX_CHARS = 2000;

/** Approximate chars-per-token ratio for threshold checks. */
const CHARS_PER_TOKEN = 4;

/** Minimum prefix in tokens to be worth caching (Gemini requires ≥2048). */
const MIN_PREFIX_TOKENS = 2048;

/** LRU max entries — we don't need to track hundreds of distinct prefixes. */
const LRU_MAX_ENTRIES = 32;

/** TTL for prefix tracking entries (1 hour in ms). */
const ENTRY_TTL_MS = 60 * 60 * 1000;

/** TTL for Gemini cachedContent resources (in seconds for the API). */
const GEMINI_CACHE_TTL_SECONDS = 3600; // 1 hour

/** Threshold of sightings before caching is activated. */
const CACHE_ACTIVATION_THRESHOLD = 2;

// ─── Prefix Fingerprinting ───

/**
 * Extract the system prompt from a list of messages.
 * Concatenates all system messages in order.
 */
export function extractSystemPrompt(messages: Array<{ role: string; content: string }>): string {
  return messages
    .filter((m) => m.role === "system")
    .map((m) => m.content)
    .join("\n\n");
}

/**
 * Compute a deterministic fingerprint (SHA-256 hex) from a system prompt.
 * Only the first ~N tokens are hashed to catch the common prefix even if
 * trailing system messages change slightly.
 *
 * We hash the first `maxChars` of the system prompt, which corresponds to
 * approximately the first `maxChars / 4` tokens.
 */
export function fingerprintPrefix(systemPrompt: string, maxChars = 8192): string {
  // Normalize: collapse whitespace to avoid trivial differences
  const normalized = systemPrompt.slice(0, maxChars).replace(/\s+/g, " ").trim();
  return crypto.createHash("sha256").update(normalized).digest("hex").substring(0, 32);
}

/**
 * Quick estimate of token count for a system prompt.
 * Uses the ~4 chars/token heuristic.
 */
export function estimateSystemTokens(systemPrompt: string): number {
  return Math.ceil(systemPrompt.length / CHARS_PER_TOKEN);
}

/**
 * Determine if a system prompt is large enough to benefit from caching.
 */
export function isCacheable(systemPrompt: string): boolean {
  return systemPrompt.length >= MIN_PREFIX_CHARS &&
    estimateSystemTokens(systemPrompt) >= MIN_PREFIX_TOKENS;
}

// ─── Prefix Cache (LRU with TTL) ───

export class PrefixCache {
  private entries = new Map<string, PrefixCacheEntry>();
  private stats = {
    totalHits: 0,
    totalCacheActivations: 0,
    geminiCacheCreations: 0,
    geminiCacheHits: 0,
    geminiCacheErrors: 0,
  };

  /**
   * Record a sighting of a system prompt prefix.
   * Returns the current entry (updated) or null if the prefix is too short to cache.
   */
  observe(systemPrompt: string): PrefixCacheEntry | null {
    if (!isCacheable(systemPrompt)) {
      return null;
    }

    const fingerprint = fingerprintPrefix(systemPrompt);
    const now = Date.now();

    // Evict expired entries periodically
    this.evictExpired(now);

    let entry = this.entries.get(fingerprint);
    if (entry) {
      entry.hitCount++;
      entry.lastSeen = now;
      this.stats.totalHits++;

      // Move to end (most recently used) by re-inserting
      this.entries.delete(fingerprint);
      this.entries.set(fingerprint, entry);

      if (entry.hitCount === CACHE_ACTIVATION_THRESHOLD) {
        this.stats.totalCacheActivations++;
        logger.info(
          `PrefixCache: prefix ${fingerprint.substring(0, 8)}… seen ${entry.hitCount}× — activating caching ` +
          `(~${estimateSystemTokens(systemPrompt)} system tokens)`,
        );
      }

      return entry;
    }

    // New entry — enforce LRU limit
    if (this.entries.size >= LRU_MAX_ENTRIES) {
      const oldestKey = this.entries.keys().next().value;
      if (oldestKey) {
        const evicted = this.entries.get(oldestKey);
        this.entries.delete(oldestKey);
        // Clean up Gemini cache if one exists
        if (evicted?.geminiCachedContentName) {
          this.invalidateGeminiCache(evicted.geminiCachedContentName).catch(() => {});
        }
      }
    }

    entry = {
      fingerprint,
      hitCount: 1,
      firstSeen: now,
      lastSeen: now,
      geminiCacheAttempted: false,
    };
    this.entries.set(fingerprint, entry);
    logger.debug(`PrefixCache: new prefix ${fingerprint.substring(0, 8)}… (~${estimateSystemTokens(systemPrompt)} tokens)`);
    return entry;
  }

  /**
   * Check if caching should be active for a given prefix.
   * Returns true if the prefix has been seen ≥2 times.
   */
  shouldCache(systemPrompt: string): boolean {
    if (!isCacheable(systemPrompt)) return false;
    const fingerprint = fingerprintPrefix(systemPrompt);
    const entry = this.entries.get(fingerprint);
    return Boolean(entry && entry.hitCount >= CACHE_ACTIVATION_THRESHOLD);
  }

  /**
   * Get the Gemini cachedContent name for a prefix, if one exists and is valid.
   */
  getGeminiCacheName(systemPrompt: string): string | undefined {
    const fingerprint = fingerprintPrefix(systemPrompt);
    const entry = this.entries.get(fingerprint);
    if (!entry?.geminiCachedContentName) return undefined;

    // Check if the cache has expired
    if (entry.geminiCacheExpiry && Date.now() >= entry.geminiCacheExpiry) {
      logger.debug(`PrefixCache: Gemini cache ${entry.geminiCachedContentName} expired, clearing`);
      entry.geminiCachedContentName = undefined;
      entry.geminiCacheAttempted = false;
      return undefined;
    }

    return entry.geminiCachedContentName;
  }

  /**
   * Record that we've created (or found) a Gemini cachedContent resource.
   */
  recordGeminiCache(systemPrompt: string, cacheName: string, model: string): void {
    const fingerprint = fingerprintPrefix(systemPrompt);
    const entry = this.entries.get(fingerprint);
    if (!entry) return;

    entry.geminiCachedContentName = cacheName;
    entry.geminiCacheModel = model;
    entry.geminiCacheExpiry = Date.now() + (GEMINI_CACHE_TTL_SECONDS * 1000);
    this.stats.geminiCacheCreations++;
    logger.info(`PrefixCache: Gemini cachedContent created → ${cacheName} (model=${model}, ttl=${GEMINI_CACHE_TTL_SECONDS}s)`);
  }

  /**
   * Record a Gemini cache hit (cachedContent was used in a request).
   */
  recordGeminiCacheHit(): void {
    this.stats.geminiCacheHits++;
  }

  /**
   * Record a Gemini cache error.
   */
  recordGeminiCacheError(): void {
    this.stats.geminiCacheErrors++;
  }

  /**
   * Check if Gemini cache creation has already been attempted for a prefix.
   */
  geminiCacheAttempted(systemPrompt: string): boolean {
    const fingerprint = fingerprintPrefix(systemPrompt);
    const entry = this.entries.get(fingerprint);
    return Boolean(entry?.geminiCacheAttempted);
  }

  /**
   * Mark that a Gemini cache creation has been attempted.
   */
  markGeminiCacheAttempted(systemPrompt: string): void {
    const fingerprint = fingerprintPrefix(systemPrompt);
    const entry = this.entries.get(fingerprint);
    if (entry) {
      entry.geminiCacheAttempted = true;
    }
  }

  /**
   * Get the singleton instance (module-level singleton).
   */
  static get instance(): PrefixCache {
    if (!globalSingleton) {
      globalSingleton = new PrefixCache();
    }
    return globalSingleton;
  }

  /**
   * Get cache statistics for observability.
   */
  getStats(): PrefixCacheStats {
    this.evictExpired(Date.now());
    let activeCached = 0;
    for (const entry of this.entries.values()) {
      if (entry.hitCount >= CACHE_ACTIVATION_THRESHOLD) activeCached++;
    }
    return {
      totalPrefixes: this.entries.size,
      activeCached,
      ...this.stats,
    };
  }

  /**
   * Clear all entries (for testing).
   */
  clear(): void {
    // Clean up Gemini caches before clearing
    for (const entry of this.entries.values()) {
      if (entry.geminiCachedContentName) {
        this.invalidateGeminiCache(entry.geminiCachedContentName).catch(() => {});
      }
    }
    this.entries.clear();
    this.stats = {
      totalHits: 0,
      totalCacheActivations: 0,
      geminiCacheCreations: 0,
      geminiCacheHits: 0,
      geminiCacheErrors: 0,
    };
  }

  // ─── Internal ───

  private evictExpired(now: number): void {
    for (const [key, entry] of this.entries) {
      if (now - entry.lastSeen > ENTRY_TTL_MS) {
        this.entries.delete(key);
        if (entry.geminiCachedContentName) {
          this.invalidateGeminiCache(entry.geminiCachedContentName).catch(() => {});
        }
      }
    }
  }

  private async invalidateGeminiCache(cacheName: string): Promise<void> {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) return;

    const geminiBase = process.env.GEMINI_BASE_URL?.replace(/\/+$/, "") ?? "https://generativelanguage.googleapis.com/v1beta";
    try {
      const delUrl = geminiBase + "/" + cacheName + "?key=" + apiKey;
      await fetch(delUrl, { method: "DELETE" });
      logger.debug(`PrefixCache: invalidated Gemini cache ${cacheName}`);
    } catch {
      // Best-effort cleanup
    }
  }
}

// Module-level singleton
let globalSingleton: PrefixCache | null = null;

// ─── Gemini cachedContent API helpers ───

/**
 * Create a Gemini cachedContent resource from a system prompt.
 * Returns the resource name (e.g. "cachedContents/abc123") or null on failure.
 *
 * The cachedContent can be referenced in subsequent generateContent/streamGenerateContent
 * calls to avoid re-processing the system prompt.
 */
export async function createGeminiCachedContent(
  model: string,
  systemPrompt: string,
  apiKey: string,
): Promise<string | null> {
  const geminiBase = process.env.GEMINI_BASE_URL?.replace(/\/+$/, "") ?? "https://generativelanguage.googleapis.com/v1beta";

  // Build the cached content payload
  // The system instruction goes into systemInstruction, and we include
  // a minimal placeholder content so the cache has valid structure.
  const body = {
    model: `models/${model}`,
    systemInstruction: { parts: [{ text: systemPrompt }] },
    contents: [{ role: "user", parts: [{ text: "" }] }],
    ttl: `${GEMINI_CACHE_TTL_SECONDS}s`,
  };

  try {
    const createUrl = geminiBase + "/cachedContents?key=" + apiKey;
    const resp = await fetch(createUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(15_000),
    });

    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      logger.warn(`PrefixCache: Gemini cachedContent creation failed (${resp.status}): ${text.substring(0, 200)}`);
      return null;
    }

    const data = await resp.json() as any;
    return data.name ?? null;
  } catch (err) {
    logger.warn(`PrefixCache: Gemini cachedContent creation error: ${err instanceof Error ? err.message : err}`);
    return null;
  }
}

// ─── Provider-specific cache integration helpers ───

/**
 * Process a request before sending to Gemini, adding cachedContent if available.
 *
 * Called by the Gemini adapter on each request. On the first call with a new
 * system prompt, it records the prefix. On the second call, it creates a
 * cachedContent resource. On subsequent calls, it references the cached content
 * and strips the system instruction from the request body (since it's cached).
 *
 * Returns the (possibly modified) system prompt and an optional cachedContent name.
 */
export async function maybeGeminiCache(
  model: string,
  messages: Array<{ role: string; content: string }>,
  apiKey: string,
): Promise<{ systemPrompt: string; cachedContentName?: string }> {
  const cache = PrefixCache.instance;
  const systemPrompt = extractSystemPrompt(messages);

  if (!systemPrompt || !isCacheable(systemPrompt)) {
    return { systemPrompt };
  }

  // Observe the prefix (increments hit count)
  cache.observe(systemPrompt);

  // Check if we already have a cachedContent for this prefix
  const existingName = cache.getGeminiCacheName(systemPrompt);
  if (existingName) {
    cache.recordGeminiCacheHit();
    logger.debug(`PrefixCache: Gemini cache HIT → ${existingName}`);
    return { systemPrompt, cachedContentName: existingName };
  }

  // If we've seen this prefix enough times and haven't tried to create a cache yet
  if (cache.shouldCache(systemPrompt) && !cache.geminiCacheAttempted(systemPrompt)) {
    cache.markGeminiCacheAttempted(systemPrompt);

    const cacheName = await createGeminiCachedContent(model, systemPrompt, apiKey);
    if (cacheName) {
      cache.recordGeminiCache(systemPrompt, cacheName, model);
      return { systemPrompt, cachedContentName: cacheName };
    } else {
      cache.recordGeminiCacheError();
    }
  }

  return { systemPrompt };
}

/**
 * Process a request before sending to ZAI.
 *
 * ZAI (ZhipuAI/GLM) supports automatic prefix caching for identical prefixes
 * ≥1024 tokens on GLM-4+ models. There's no explicit cachedContent API —
 * caching is transparent. We observe the prefix for tracking/logging purposes.
 *
 * Returns the system prompt unchanged — ZAI handles caching internally.
 */
export function observeZaIPrefix(
  messages: Array<{ role: string; content: string }>,
): { systemPrompt: string; cacheExpected: boolean } {
  const cache = PrefixCache.instance;
  const systemPrompt = extractSystemPrompt(messages);

  if (!systemPrompt || !isCacheable(systemPrompt)) {
    return { systemPrompt, cacheExpected: false };
  }

  cache.observe(systemPrompt);
  const cacheExpected = cache.shouldCache(systemPrompt);

  if (cacheExpected) {
    logger.debug(`PrefixCache: ZAI prefix cache expected for ${fingerprintPrefix(systemPrompt).substring(0, 8)}…`);
  }

  return { systemPrompt, cacheExpected };
}
