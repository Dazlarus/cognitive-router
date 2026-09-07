// src/upstream_timeout_scope.ts - per-request upstream timeout override
//
// The bench sidecar (cogrouter-bench) sends non-streaming generations that
// legitimately run minutes (max_tokens 6144 reasoning gens, CONTRACT.md §2).
// The router honors its x-router-timeout-ms header for the total request
// deadline, but every upstream provider fetch also carries its own hard
// AbortSignal.timeout(remoteTimeoutMs()) — 60s by default — which would abort
// the pinned candidate mid-generation (live: 503 at exactly 60,011ms,
// 2026-09-07 21:42).
//
// This module lets proxy-stream lift the per-attempt upstream timeout for a
// single request via AsyncLocalStorage. Adapters consult it inside
// remoteTimeoutMs()/localTimeoutMs() (and the stream variants). Zero adapter
// signature changes; the override is scoped to the requesting call only.

import { AsyncLocalStorage } from "node:async_hooks";

const upstreamTimeoutStorage = new AsyncLocalStorage<{ overrideMs: number }>();

/** 10 minutes — same cap as the x-router-timeout-ms header override. */
export const MAX_UPSTREAM_OVERRIDE_MS = 600_000;

/** Run `fn` with the per-attempt upstream timeout overridden to `ms`. */
export function withUpstreamTimeoutOverride<T>(ms: number, fn: () => Promise<T>): Promise<T> {
  return upstreamTimeoutStorage.run({ overrideMs: Math.min(ms, MAX_UPSTREAM_OVERRIDE_MS) }, fn);
}

/** Current override, or null outside an override scope. */
export function getUpstreamTimeoutOverrideMs(): number | null {
  return upstreamTimeoutStorage.getStore()?.overrideMs ?? null;
}
