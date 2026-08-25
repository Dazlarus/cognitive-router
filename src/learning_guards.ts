// src/learning_guards.ts — Phase-1 learning-loop hardening primitives
// Pure functions + env knobs. No I/O. Time is always an explicit parameter
// so every guard is deterministically testable (time-mocked).
//
// Spec: docs/LEARNING_LOOP_DESIGN.md §4.1 (attribution gate + quarantine),
// §4.2 (guardrails), §4.6 (reliability reload with age).

import { createHash } from "node:crypto";

// ─── Env knobs (all optional; conservative defaults) ───

export function capMin(): number {
  return numEnv("ROUTER_CAP_MIN", 0.05);
}
export function capMax(): number {
  return numEnv("ROUTER_CAP_MAX", 0.98);
}
/** Max capability movement per 24h window (Δmax). */
export function rocMax24h(): number {
  return numEnv("ROUTER_ROC_MAX_24H", 0.15);
}
/** Max capability movement per 7d window. */
export function rocMax7d(): number {
  return numEnv("ROUTER_ROC_MAX_7D", 0.30);
}
/** Idle days before decay-to-prior kicks in. */
export function decayIdleDays(): number {
  return numEnv("ROUTER_DECAY_IDLE_DAYS", 14);
}
/** Fraction of the (value − seed) gap closed per idle day after threshold. */
export function decayRatePerDay(): number {
  return numEnv("ROUTER_DECAY_RATE", 0.10);
}
/** Arm-B classifier-confidence threshold (spike-verified Aug 24: 0.75 catches the poisoned cluster). */
export function armBMinConfidence(): number {
  return numEnv("ROUTER_ARM_B_MIN_CONFIDENCE", 0.75);
}
/** Post-application canary threshold: same normalized note N times → quarantine pair. */
export function noteCanaryThreshold(): number {
  return Math.max(2, Math.floor(numEnv("ROUTER_JUDGE_NOTE_CANARY", 5)));
}
/** Pre-application similarity hold: note seen this many times already → hold (don't apply). */
export function noteHoldThreshold(): number {
  return Math.max(1, Math.floor(numEnv("ROUTER_JUDGE_NOTE_HOLD", 1)));
}
/** Chars-per-token heuristic for the ingestion ladder (3.5:1 per spec). */
export function charsPerToken(): number {
  return numEnv("ROUTER_JUDGE_CHARS_PER_TOKEN", 3.5);
}
/** Tier-1 single-shot cap (chars of prompt+response combined). */
export function tier1MaxChars(): number {
  return numEnv("ROUTER_JUDGE_MAX_CHARS", 32_000);
}
/** Tier-2 multi-turn ceiling; beyond this → tier-3 head+tail. */
export function tier2MaxChars(): number {
  return numEnv("ROUTER_JUDGE_TIER2_MAX_CHARS", 200_000);
}
/** Shadow mode: default TRUE — learner logs would-be values, never writes. */
export function learningShadowMode(): boolean {
  return process.env.ROUTER_LEARNING_SHADOW_MODE !== "false";
}
/** UCB exploration constant (offline-tuned; see docs/TUNING_LOG.md). */
export function ucbK(): number {
  return numEnv("ROUTER_UCB_K", 0.05);
}

function numEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  const v = Number(raw);
  return Number.isFinite(v) ? v : fallback;
}

// ─── Write-time clamps (§4.2) ───

export function clampCapability(value: number): number {
  return Math.min(capMax(), Math.max(capMin(), value));
}

// ─── Attribution gate (§4.1) ───

export type GateArm = "A" | "B" | "C";

export interface AttributionInput {
  /** Provider-attributable truncation (finish_reason=length or tier-3 elision). */
  truncated?: boolean;
  /** Intent-classifier confidence for the turn (null = missing). */
  confidence?: number | null;
  /** Judge verdict missing/unparseable. */
  malformed?: boolean;
}

export interface AttributionVerdict {
  arm: GateArm;
  reason: string;
}

export function classifyAttribution(input: AttributionInput): AttributionVerdict {
  if (input.truncated) {
    return { arm: "A", reason: "provider_truncation" };
  }
  if (input.malformed) {
    return { arm: "B", reason: "malformed_verdict" };
  }
  const c = input.confidence;
  if (c === null || c === undefined || typeof c !== "number" || !Number.isFinite(c)) {
    return { arm: "B", reason: "missing_confidence" };
  }
  if (c < armBMinConfidence()) {
    return { arm: "B", reason: "low_confidence" };
  }
  return { arm: "C", reason: "model_attributable" };
}

// ─── Judge-note similarity + quarantine (§4.1) ───

export const NOTE_TRUNCATE_LEN = 255;

/** Normalize a judge note: lowercase, collapse whitespace, strip punctuation,
 *  truncate to 255 chars. Deterministic across sessions. */
export function normalizeNote(note: string): string {
  return note
    .toLowerCase()
    .replace(/[\s\p{P}\p{S}]+/gu, " ")
    .trim()
    .slice(0, NOTE_TRUNCATE_LEN);
}

/** Stable SHA-256 of the normalized+truncated note. */
export function noteHash(normalized: string): string {
  return createHash("sha256").update(normalized, "utf8").digest("hex");
}

/** Levenshtein distance on truncated strings only (per spec). */
export function levenshtein(a: string, b: string): number {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;
  let prev = new Array<number>(b.length + 1);
  let cur = new Array<number>(b.length + 1);
  for (let j = 0; j <= b.length; j++) prev[j] = j;
  for (let i = 1; i <= a.length; i++) {
    cur[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const cost = a.charCodeAt(i - 1) === b.charCodeAt(j - 1) ? 0 : 1;
      cur[j] = Math.min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost);
    }
    [prev, cur] = [cur, prev];
  }
  return prev[b.length];
}

/** Near-identical = identical hash OR tiny edit distance on the truncated forms. */
export function isNearIdenticalNote(normA: string, normB: string): boolean {
  if (normA === normB) return true;
  if (Math.abs(normA.length - normB.length) > 6) return false;
  return levenshtein(normA, normB) <= 6;
}

// ─── Decay-to-prior (§4.2) ───

/** Exponential drift of a learned value toward its seed after N idle days.
 *  value = seed + (value − seed) · (1 − rate)^idleDays. Pinned cells are
 *  exempt (caller checks). */
export function decayTowardSeed(
  value: number,
  seed: number,
  idleDays: number,
  ratePerDay: number = decayRatePerDay(),
): number {
  if (idleDays <= 0) return value;
  const r = Math.min(Math.max(ratePerDay, 0), 1);
  return seed + (value - seed) * Math.pow(1 - r, idleDays);
}

// ─── Reliability reload with age (§4.6) ───

export const EPISODIC_HALF_LIFE_MS = 3_600_000; // 1h
export const PERSISTENT_HALF_LIFE_MS = 7 * 86_400_000; // 7d

/** Episodic counters (consecutive failures, backoff tier) decay with a 1h
 *  half-life since last_check. A restart is not evidence of recovery, but
 *  stale episodic state is not evidence of ongoing congestion either. */
export function ageWeightedEpisodic(
  consecutiveFailures: number,
  backoffTier: number,
  ageMs: number,
): { consecutiveFailures: number; backoffTier: number } {
  const factor = Math.pow(0.5, Math.max(0, ageMs) / EPISODIC_HALF_LIFE_MS);
  return {
    consecutiveFailures: Math.floor(consecutiveFailures * factor),
    backoffTier: Math.floor(backoffTier * factor),
  };
}

/** Persistent failure rate decays with a 7d half-life toward 0 (= healthy 1.0).
 *  Cold-start (no data) = 0 failure rate. */
export function ageWeightedPersistentFailureRate(failureRate: number, ageMs: number): number {
  const factor = Math.pow(0.5, Math.max(0, ageMs) / PERSISTENT_HALF_LIFE_MS);
  return Math.max(0, Math.min(1, failureRate)) * factor;
}

// ─── Ingestion ladder tiering (§4.4) ───

export type IngestionTier = 1 | 2 | 3;

export function pickIngestionTier(totalChars: number): IngestionTier {
  if (totalChars <= tier1MaxChars()) return 1;
  if (totalChars <= tier2MaxChars()) return 2;
  return 3;
}

/** Estimate tokens via the 3.5:1 chars-per-token heuristic (env-tunable). */
export function estimateTokens(chars: number): number {
  return Math.ceil(chars / charsPerToken());
}

// ─── UCB exploration bonus (§4.2) ───

/** learned + k/√n with n=0 treated as 1. Read-path only; never persisted. */
export function ucbBonus(sampleCount: number, k: number = ucbK()): number {
  const n = sampleCount > 0 ? sampleCount : 1;
  return k / Math.sqrt(n);
}
