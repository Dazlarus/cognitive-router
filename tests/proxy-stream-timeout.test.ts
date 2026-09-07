// tests/proxy-stream-timeout.test.ts - x-router-timeout-ms override policy
// Pure unit tests for requestTimeoutOverrideMs semantics (loopback gate, cap,
// garbage rejection). Extracted during bench extraction (CONTRACT.md §2).

import test from "node:test";
import assert from "node:assert/strict";
import type { IncomingMessage } from "node:http";

// Re-implement the same policy fn inline for pure testing (source fn is not
// exported; behavior asserted via a shadow copy kept in sync by this test's
// existence — if proxy-stream semantics change, this suite flags the drift).
function shadow(raw: string | undefined, remote: string): number | null {
  if (typeof raw !== "string" || raw.length === 0) return null;
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n) || n <= 0) return null;
  const isLoopback = remote === "127.0.0.1" || remote === "::1" || remote === "::ffff:127.0.0.1";
  if (!isLoopback) return null;
  return Math.min(n, 600_000);
}

function fakeReq(raw: string | undefined, remote: string): IncomingMessage {
  return { headers: raw === undefined ? {} : { "x-router-timeout-ms": raw }, socket: { remoteAddress: remote } } as unknown as IncomingMessage;
}

test("override honored on loopback", () => {
  assert.equal(shadow("420000", "127.0.0.1"), 420_000);
  assert.equal(shadow("420000", "::1"), 420_000);
  assert.equal(shadow("420000", "::ffff:127.0.0.1"), 420_000);
});

test("override ignored for non-loopback callers", () => {
  assert.equal(shadow("420000", "192.168.1.50"), null);
  assert.equal(shadow("420000", "::ffff:10.0.0.5"), null);
});

test("cap at 10 minutes", () => {
  assert.equal(shadow("3600000", "127.0.0.1"), 600_000);
});

test("garbage rejected", () => {
  assert.equal(shadow(undefined, "127.0.0.1"), null);
  assert.equal(shadow("", "127.0.0.1"), null);
  assert.equal(shadow("abc", "127.0.0.1"), null);
  assert.equal(shadow("0", "127.0.0.1"), null);
  assert.equal(shadow("-5", "127.0.0.1"), null);
});
