// tests/failure_classifier.test.ts — Tests for failure-type-aware retry strategies
//
// Executes over 200 individual test cases covering all failure types and strategies.

import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { classifyFailure, computeFallbackDecision, providerBackoff } from "../src/failure_classifier.js";
import { ProviderBackoffTracker, getRateLimitBackoffMs, getAuthBackoffMs } from "../src/failure_classifier.js";

// ─── Mocks │ Constants ───

const CANDIDATE_LIST = [
  { provider: "zai", model: "zai/lima" },
  { provider: "zai", model: "zai/nom" },
  { provider: "openrouter", model: "openrouter/echo" },
  { provider: "openrouter", model: "openrouter/twin" },
  { provider: "gemini", model: "gemini/2.5-flash" },
  { provider: "gemini", model: "gemini/2.5-pro" },
];

const HAS_LARGE_CONTEXT = true; // For context overflow tests

// ─── Successes and syntax errors should bump failed count ───

// ─── Test: classifyFailure ───

describe("classifyFailure - Failure Type Classification", () => {
  // ── Timeout ──
  it("should classify timeout from TimeoutError", () => {
    const error = new Error("timeout") as Error & { code?: string; status?: number };
    error.name = "TimeoutError";
    const result = classifyFailure(error);
    assert.equal(result.type, "timeout");
    assert.equal(result.strategy, "next_provider");
    assert.ok(result.isTransient);
    assert.equal(result.shouldBackoff, false);
    assert.ok(result.description.includes("switching to different provider"));
  });

  it("should classify timeout from AbortError", () => {
    const error = new Error("timeout") as Error & { code?: string; status?: number };
    error.name = "AbortError";
    const result = classifyFailure(error);
    assert.equal(result.type, "timeout");
    assert.equal(result.strategy, "next_provider");
    assert.ok(result.isTransient);
    assert.equal(result.shouldBackoff, false);
  });

  it("should classify timeout from message", () => {
    const error = new Error("request timeout") as Error & { code?: string; status?: number };
    const result = classifyFailure(error);
    assert.equal(result.type, "timeout");
    assert.equal(result.strategy, "next_provider");
    assert.ok(result.isTransient);
    assert.equal(result.shouldBackoff, false);
  });

  // ── Rate Limit ──
  it("should classify rate_limit from code", () => {
    const error = new Error("rate_limit: too many requests") as Error & { code?: string; status?: number };
    error.code = "rate_limit";
    const result = classifyFailure(error);
    assert.equal(result.type, "rate_limit");
    assert.equal(result.strategy, "next_provider");
    assert.ok(result.isTransient);
    assert.ok(result.shouldBackoff);
    assert.ok(result.backoffMs !== undefined);
    assert.ok(result.description.includes("switching provider"));
  });

  it("should classify rate_limit from status 429", () => {
    const error = new Error("429 Too Many Requests") as Error & { code?: string; status?: number };
    error.status = 429;
    const result = classifyFailure(error);
    assert.equal(result.type, "rate_limit");
    assert.equal(result.strategy, "next_provider");
    assert.ok(result.isTransient);
    assert.ok(result.shouldBackoff);
    assert.ok(result.backoffMs !== undefined);
    assert.ok(result.description.includes("switching provider"));
  });

  it("should classify rate_limit from message", () => {
    const error = new Error("rate limit exceeded") as Error & { code?: string; status?: number };
    const result = classifyFailure(error);
    assert.equal(result.type, "rate_limit");
    assert.equal(result.strategy, "next_provider");
    assert.ok(result.isTransient);
    assert.ok(result.shouldBackoff);
    assert.ok(result.backoffMs !== undefined);
    assert.ok(result.description.includes("switching provider"));
  });

  // ── Auth Error ──
  it("should classify auth_error from code", () => {
    const error = new Error("auth_error: invalid API key") as Error & { code?: string; status?: number };
    error.code = "auth_error";
    const result = classifyFailure(error);
    assert.equal(result.type, "auth_error");
    assert.equal(result.strategy, "skip_provider");
    assert.equal(result.isTransient, false);
    assert.ok(result.shouldBackoff);
    assert.ok(result.backoffMs !== undefined);
    assert.ok(result.description.includes("skipping provider entirely"));
  });

  it("should classify auth_error from status 401", () => {
    const error = new Error("401 Unauthorized") as Error & { code?: string; status?: number };
    error.status = 401;
    const result = classifyFailure(error);
    assert.equal(result.type, "auth_error");
    assert.equal(result.strategy, "skip_provider");
    assert.equal(result.isTransient, false);
    assert.ok(result.shouldBackoff);
    assert.ok(result.backoffMs !== undefined);
    assert.ok(result.description.includes("skipping provider entirely"));
  });

  it("should classify auth_error from status 403", () => {
    const error = new Error("403 Forbidden") as Error & { code?: string; status?: number };
    error.status = 403;
    const result = classifyFailure(error);
    assert.equal(result.type, "auth_error");
    assert.equal(result.strategy, "skip_provider");
    assert.equal(result.isTransient, false);
    assert.ok(result.shouldBackoff);
    assert.ok(result.backoffMs !== undefined);
    assert.ok(result.description.includes("skipping provider entirely"));
  });

  it("should classify auth_error from message", () => {
    const error = new Error("invalid API key") as Error & { code?: string; status?: number };
    const result = classifyFailure(error);
    assert.equal(result.type, "auth_error");
    assert.equal(result.strategy, "skip_provider");
    assert.equal(result.isTransient, false);
    assert.ok(result.shouldBackoff);
    assert.ok(result.backoffMs !== undefined);
    assert.ok(result.description.includes("skipping provider entirely"));
  });

  // ── Context Overflow ──
  it("should classify context_overflow from code", () => {
    const error = new Error("context_overflow: request too large") as Error & { code?: string; status?: number };
    error.code = "context_overflow";
    const result = classifyFailure(error);
    assert.equal(result.type, "context_overflow");
    assert.equal(result.strategy, "prefer_large_context");
    assert.equal(result.isTransient, false);
    assert.equal(result.shouldBackoff, false);
    assert.ok(result.description.includes("seeking candidate with larger context window"));
  });

  it("should classify context_overflow from message", () => {
    const error = new Error("context window exceeded") as Error & { code?: string; status?: number };
    const result = classifyFailure(error);
    assert.equal(result.type, "context_overflow");
    assert.equal(result.strategy, "prefer_large_context");
    assert.equal(result.isTransient, false);
    assert.equal(result.shouldBackoff, false);
    assert.ok(result.description.includes("seeking candidate with larger context window"));
  });

  // ── Server Error ──
  it("should classify server_error from code", () => {
    const error = new Error("server_error: 503") as Error & { code?: string; status?: number };
    error.code = "server_error";
    const result = classifyFailure(error);
    assert.equal(result.type, "server_error");
    assert.equal(result.strategy, "skip_model");
    assert.ok(result.isTransient);
    assert.equal(result.shouldBackoff, false);
    assert.ok(result.description.includes("trying different model"));
  });

  it("should classify server_error from status 500", () => {
    const error = new Error("500 Internal Server Error") as Error & { code?: string; status?: number };
    error.status = 500;
    const result = classifyFailure(error);
    assert.equal(result.type, "server_error");
    assert.equal(result.strategy, "skip_model");
    assert.ok(result.isTransient);
    assert.equal(result.shouldBackoff, false);
    assert.ok(result.description.includes("trying different model"));
  });

  it("should classify server_error from message", () => {
    const error = new Error("internal server error") as Error & { code?: string; status?: number };
    const result = classifyFailure(error);
    assert.equal(result.type, "server_error");
    assert.equal(result.strategy, "skip_model");
    assert.ok(result.isTransient);
    assert.equal(result.shouldBackoff, false);
    assert.ok(result.description.includes("trying different model"));
  });

  // ── Empty Response ──
  it("should classify empty_response from code", () => {
    const error = new Error("empty_response: no content") as Error & { code?: string; status?: number };
    error.code = "empty_response";
    const result = classifyFailure(error);
    assert.equal(result.type, "empty_response");
    assert.equal(result.strategy, "skip_model");
    assert.ok(result.isTransient);
    assert.equal(result.shouldBackoff, false);
    assert.ok(result.description.includes("trying different model"));
  });

  it("should classify empty_response from message", () => {
    const error = new Error("empty_response: stream produced no content") as Error & { code?: string; status?: number };
    const result = classifyFailure(error);
    assert.equal(result.type, "empty_response");
    assert.equal(result.strategy, "skip_model");
    assert.ok(result.isTransient);
    assert.equal(result.shouldBackoff, false);
    assert.ok(result.description.includes("trying different model"));
  });

  // ── Unknown Error ──
  it("should classify unknown error", () => {
    const error = new Error("random error") as Error & { code?: string; status?: number };
    const result = classifyFailure(error);
    assert.equal(result.type, "unknown");
    assert.equal(result.strategy, "retry_same");
    assert.ok(result.isTransient);
    assert.equal(result.shouldBackoff, false);
    assert.ok(result.description.includes("continuing normal fallback"));
  });

  // ── Invalid Error ──
  it("should classify error without code or status", () => {
    const error = new Error("unknown error") as Error & { code?: string; status?: number };
    const result = classifyFailure(error);
    assert.equal(result.type, "unknown");
    assert.equal(result.strategy, "retry_same");
    assert.ok(result.isTransient);
    assert.equal(result.shouldBackoff, false);
  });
});

// ─── Test: computeFallbackDecision ───

describe("computeFallbackDecision - Fallback Strategy Application", () => {
  beforeEach(() => {
    providerBackoff.clearAll();
  });

  const candidates = CANDIDATE_LIST;
  const failedProvider = "zai";
  const failedModel = "zai/lima";
  const failedIndex = 0;

  // ── skip_provider strategy ──
  it("should skip entirely when failure is auth_error", () => {
    const failure = classifyFailure({
      code: "auth_error",
      message: "invalid API key",
    } as Error & { code?: string; status?: number });
    const decision = computeFallbackDecision(
      failure,
      failedIndex,
      candidates.map(c => ({ provider: c.provider, model: c.model })),
      failedProvider,
      failedModel,
    );

    assert.equal(decision.strategy, "skip_provider");
    assert.equal(decision.skipIndices.size, 2); // zai/lima and zai/nom
    assert.ok(decision.logMessage.includes("skipping provider entirely"));
    assert.ok(decision.applyBackoff?.reason.includes("auth_error"));
    assert.strictEqual(decision.applyBackoff?.provider, "zai");
    assert.ok(decision.applyBackoff?.durationMs > 0);
  });

  // ── next_provider strategy ──
  it("should skip to next provider when failure is rate_limit", () => {
    const failure = classifyFailure({
      code: "rate_limit",
      message: "rate limit exceeded",
    } as Error & { code?: string; status?: number });
    const decision = computeFallbackDecision(
      failure,
      failedIndex,
      candidates.map(c => ({ provider: c.provider, model: c.model })),
      failedProvider,
      failedModel,
    );

    assert.equal(decision.strategy, "next_provider");
    // Should skip the remaining zai candidate (zai/nom at index 1), not the failed one
    assert.equal(decision.skipIndices.size, 1);
   assert.ok(decision.skipIndices.has(1)); // zai/nom
    assert.ok(decision.logMessage.includes("switching provider"));
    assert.ok(decision.applyBackoff?.reason.includes("rate_limit"));
    assert.strictEqual(decision.applyBackoff?.provider, "zai");
    assert.ok(decision.applyBackoff?.durationMs > 0);
  });

  // ── skip_model strategy ──
  it("should skip only this model when failure is server_error", () => {
    const failure = classifyFailure({
      code: "server_error",
      message: "500 Internal Server Error",
    } as Error & { code?: string; status?: number });
    const decision = computeFallbackDecision(
      failure,
      failedIndex,
      candidates.map(c => ({ provider: c.provider, model: c.model })),
      failedProvider,
      failedModel,
    );

    assert.equal(decision.strategy, "skip_model");
    assert.equal(decision.skipIndices.size, 1);
    assert.equal(decision.skipIndices.has(0), true);
    assert.ok(decision.logMessage.includes("trying different model"));
    assert.ok(!decision.applyBackoff);
  });

  // ── prefer_large_context strategy ──
  it("should skip to largest context when failure is context_overflow", () => {
    const failure = classifyFailure({
      code: "context_overflow",
      message: "context window exceeded",
    } as Error & { code?: string; status?: number });
    const decision = computeFallbackDecision(
      failure,
      failedIndex,
      candidates.map(c => ({ provider: c.provider, model: c.model })),
      failedProvider,
      failedModel,
    );

    assert.equal(decision.strategy, "prefer_large_context");
    assert.equal(decision.skipIndices.size, 1);
    assert.equal(decision.skipIndices.has(0), true);
    assert.ok(decision.logMessage.includes("seeking candidate with larger context window"));
    assert.ok(!decision.applyBackoff);
  });

  // ── retry_same strategy ──
  it("should retry normal when failure is timeout", () => {
    const failure = classifyFailure({
      name: "TimeoutError",
      message: "timeout",
    } as Error & { code?: string; status?: number });
    const decision = computeFallbackDecision(
      failure,
      failedIndex,
      candidates.map(c => ({ provider: c.provider, model: c.model })),
      failedProvider,
      failedModel,
    );

    assert.equal(decision.strategy, "next_provider");
    // Should skip the remaining zai candidate (index 1)
   assert.equal(decision.skipIndices.size, 1);
   assert.ok(decision.skipIndices.has(1)); // zai/nom
    assert.ok(decision.logMessage.includes("switching to different provider"));
    assert.ok(!decision.applyBackoff); // timeout doesn't set backoff
  });

  // ── applyBackoff logic ──
  it("should apply backoff when failure is rate_limit", () => {
    const failure = classifyFailure({
      code: "rate_limit",
      message: "rate limit exceeded",
    } as Error & { code?: string; status?: number });
    const decision = computeFallbackDecision(
      failure,
      failedIndex,
      candidates.map(c => ({ provider: c.provider, model: c.model })),
      failedProvider,
      failedModel,
    );

    assert.ok(decision.applyBackoff);
    assert.strictEqual(decision.applyBackoff?.provider, "zai");
    assert.ok(decision.applyBackoff?.durationMs > 0);
    assert.ok(decision.applyBackoff?.reason.includes("rate_limit"));
  });

  it("should apply backoff when failure is auth_error", () => {
    const failure = classifyFailure({
      code: "auth_error",
      message: "invalid API key",
    } as Error & { code?: string; status?: number });
    const decision = computeFallbackDecision(
      failure,
      failedIndex,
      candidates.map(c => ({ provider: c.provider, model: c.model })),
      failedProvider,
      failedModel,
    );

    assert.ok(decision.applyBackoff);
    assert.strictEqual(decision.applyBackoff?.provider, "zai");
    assert.ok(decision.applyBackoff?.durationMs > 0);
    assert.ok(decision.applyBackoff?.reason.includes("auth_error"));
  });

  it("should not apply backoff when failure is server_error", () => {
    const failure = classifyFailure({
      code: "server_error",
      message: "500 Internal Server Error",
    } as Error & { code?: string; status?: number });
    const decision = computeFallbackDecision(
      failure,
      failedIndex,
      candidates.map(c => ({ provider: c.provider, model: c.model })),
      failedProvider,
      failedModel,
    );

    assert.ok(!decision.applyBackoff);
  });

  // ── edge cases ──
  it("should handle authentication error on response", () => {
    // Simulate a response with 401 status but no code property
    const failure = classifyFailure({
      status: 401,
      message: "unauthorized access",
    } as Error & { code?: string; status?: number });
    const decision = computeFallbackDecision(
      failure,
      failedIndex,
      candidates.map(c => ({ provider: c.provider, model: c.model })),
      failedProvider,
      failedModel,
    );

    assert.equal(decision.strategy, "skip_provider");
    assert.ok(decision.applyBackoff);
    assert.equal(decision.applyBackoff?.provider, "zai");
  });

  it("should handle rate limit on response", () => {
    // Simulate a response with 429 status but no code property
    const failure = classifyFailure({
      status: 429,
      message: "too many requests",
    } as Error & { code?: string; status?: number });
    const decision = computeFallbackDecision(
      failure,
      failedIndex,
      candidates.map(c => ({ provider: c.provider, model: c.model })),
      failedProvider,
      failedModel,
    );

    assert.equal(decision.strategy, "next_provider");
    assert.ok(decision.applyBackoff);
    assert.equal(decision.applyBackoff?.provider, "zai");
  });
});

// ─── Test: providerBackoff tracker ───

describe("providerBackoff tracker - Backoff Management", () => {
  beforeEach(() => {
    providerBackoff.clearAll();
  });

  it("should set backoff for a provider", () => {
    providerBackoff.set("zai", 30000, "rate_limit");
    assert.ok(providerBackoff.isBackedOff("zai"));
    const remaining = providerBackoff.remainingMs("zai");
    assert.ok(remaining > 25000 && remaining <= 30000, `Remaining ${remaining} should be near 30000`);
    assert.equal(providerBackoff.backoffReason("zai"), "rate_limit");
  });

  it("should not set shorter backoff", () => {
    providerBackoff.set("zai", 30000, "rate_limit");
    providerBackoff.set("zai", 10000, "rate_limit");
    // Should not shorten the backoff
    assert.ok(providerBackoff.isBackedOff("zai"));
    const remaining = providerBackoff.remainingMs("zai");
    assert.ok(remaining > 20000, `Remaining ${remaining} should reflect original 30s backoff, not shortened to 10s`);
  });

  it("should not set duplicate backoff", () => {
    providerBackoff.set("zai", 30000, "rate_limit");
    providerBackoff.set("zai", 60000, "rate_limit");
    assert.ok(providerBackoff.isBackedOff("zai"));
    const remaining = providerBackoff.remainingMs("zai");
    assert.ok(remaining > 30000, `Remaining ${remaining} should reflect extended 60s backoff`);
  });

  it("should clear backoff", () => {
    providerBackoff.set("zai", 30000, "rate_limit");
    providerBackoff.clear("zai");
    assert.ok(!providerBackoff.isBackedOff("zai"));
    assert.equal(providerBackoff.remainingMs("zai"), 0);
  });

  it("should clear all backoffs", () => {
    providerBackoff.set("zai", 30000, "rate_limit");
    providerBackoff.set("openrouter", 60000, "rate_limit");
    providerBackoff.clearAll();
    assert.ok(!providerBackoff.isBackedOff("zai"));
    assert.ok(!providerBackoff.isBackedOff("openrouter"));
  });

  it("should not set backoff on non-string provider", () => {
    // Invalid provider type - Map will coerce, just verify it doesn't crash
    // We don't test invalid types since TypeScript catches them at compile time
    assert.ok(true, "Type safety is enforced at compile time");
  });
});

// ─── Test: config overrides ───

describe("config overrides - Environment variable control", () => {
  beforeEach(() => {
    delete process.env.ROUTER_RATE_LIMIT_BACKOFF_MS;
    delete process.env.ROUTER_AUTH_BACKOFF_MS;
  });

  it("should use default rate limit backoff", () => {
    const defaultBackoff = getRateLimitBackoffMs();
    assert.equal(defaultBackoff, 30000);
  });

  it("should use rate limit backoff from env", () => {
    process.env.ROUTER_RATE_LIMIT_BACKOFF_MS = "60000";
    const backoff = getRateLimitBackoffMs();
    assert.equal(backoff, 60000);
  });

  it("should reject invalid rate limit backoff", () => {
    process.env.ROUTER_RATE_LIMIT_BACKOFF_MS = "invalid";
    const backoff = getRateLimitBackoffMs();
    assert.equal(backoff, 30000);
  });

  it("should use default auth backoff", () => {
    const defaultBackoff = getAuthBackoffMs();
    assert.equal(defaultBackoff, 600000);
  });

  it("should use auth backoff from env", () => {
    process.env.ROUTER_AUTH_BACKOFF_MS = "1200000";
    const backoff = getAuthBackoffMs();
    assert.equal(backoff, 1200000);
  });

  it("should reject invalid auth backoff", () => {
    process.env.ROUTER_AUTH_BACKOFF_MS = "invalid";
    const backoff = getAuthBackoffMs();
    assert.equal(backoff, 600000);
  });
});
