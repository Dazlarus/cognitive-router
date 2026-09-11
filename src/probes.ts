// src/probes.ts — Load, validate, and filter coding probes
//
// Spec: docs/EXEC_SANDBOX_SPEC.md §3 (probe format) and §5 ("probes.ts: load
// /validate probe JSON, generation filter, probesFor(subIntent, generation)")
//
// Probe files live under probes/coding/*.json, generation-tagged with the
// same PROMPT_GENERATION discipline as the ladder.

import { readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { logger } from "./logger.js";
import { PROMPT_GENERATION } from "./benchmark_ladder.js";

// ---------- types ----------

export type CodingSubIntent = "quick" | "repo" | "agent";

export interface CodingProbeHarness {
  setup: string;
  cases: Array<{ call: string; expect: string }>;
  casesHidden?: number;
  check?: string;
}

export interface CodingProbe {
  id: string;
  generation: string;
  subIntent: CodingSubIntent;
  prompt: string;
  language: "python" | "typescript";
  entry: string;
  harness: CodingProbeHarness;
}

// ---------- loader ----------

const PROBES_DIR = resolve(import.meta.dirname, "..", "probes", "coding");

/** Load every probe from probes/coding/*.json, validating required fields.
 *  Skips files that don't parse or fail validation; logs a warning for each. */
export function loadAllProbes(): CodingProbe[] {
  let files: string[];
  try {
    files = readdirSync(PROBES_DIR).filter((f) => f.endsWith(".json"));
  } catch {
    logger.warn("probes: probes/coding/ directory not found or unreadable");
    return [];
  }

  const probes: CodingProbe[] = [];

  for (const file of files.sort()) {
    const path = join(PROBES_DIR, file);
    let raw: unknown;
    try {
      raw = JSON.parse(readFileSync(path, "utf-8"));
    } catch (err) {
      logger.warn(`probes: skipping ${file} — parse error: ${err}`);
      continue;
    }

    const validated = validateProbe(raw);
    if (validated === null) {
      logger.warn(`probes: skipping ${file} — validation failed`);
      continue;
    }
    probes.push(validated);
  }

  return probes;
}

// ---------- validation ----------

export function validateProbe(raw: unknown): CodingProbe | null {
  if (typeof raw !== "object" || raw === null) return null;
  const obj = raw as Record<string, unknown>;

  const id = obj.id;
  if (typeof id !== "string" || id.length === 0) return null;

  const generation = obj.generation;
  if (typeof generation !== "string" || generation.length === 0) return null;

  const subIntent = obj.subIntent;
  if (subIntent !== "quick" && subIntent !== "repo" && subIntent !== "agent") return null;

  const prompt = obj.prompt;
  if (typeof prompt !== "string" || prompt.length === 0) return null;

  const language = obj.language;
  if (language !== "python" && language !== "typescript") return null;

  const entry = obj.entry;
  if (typeof entry !== "string" || entry.length === 0) return null;

  const harness = obj.harness;
  if (typeof harness !== "object" || harness === null) return null;
  const h = harness as Record<string, unknown>;

  const setup = h.setup;
  if (typeof setup !== "string") return null;

  const cases = h.cases;
  if (!Array.isArray(cases)) return null;

  const validatedCases: Array<{ call: string; expect: string }> = [];
  for (let i = 0; i < cases.length; i++) {
    const c = cases[i];
    if (typeof c !== "object" || c === null) return null;
    const cc = c as Record<string, unknown>;
    if (typeof cc.call !== "string" || typeof cc.expect !== "string") return null;
    validatedCases.push({ call: cc.call, expect: cc.expect });
  }
  if (validatedCases.length === 0) return null;

  const casesHidden = h.casesHidden;
  if (casesHidden !== undefined && (typeof casesHidden !== "number" || !Number.isInteger(casesHidden) || casesHidden < 0)) return null;

  const check = h.check;
  if (check !== undefined && typeof check !== "string") return null;

  return {
    id,
    generation,
    subIntent: subIntent as CodingSubIntent,
    prompt,
    language: language as "python" | "typescript",
    entry,
    harness: {
      setup,
      cases: validatedCases,
      ...(casesHidden !== undefined ? { casesHidden } : {}),
      ...(check !== undefined ? { check } : {}),
    },
  };
}

// ---------- generation filtering ----------

/**
 * Return probes matching the given subIntent and generation.
 * Filters to the current PROMPT_GENERATION by default.
 */
export function probesFor(
  subIntent: CodingSubIntent,
  generation: string = PROMPT_GENERATION,
): CodingProbe[] {
  return loadAllProbes().filter(
    (p) => p.subIntent === subIntent && p.generation === generation,
  );
}

/**
 * Return all unique (subIntent, generation) pairs seen across loaded probes.
 */
export function probeSummary(): Array<{ subIntent: CodingSubIntent; generation: string; count: number }> {
  const counts = new Map<string, { subIntent: CodingSubIntent; generation: string; count: number }>();
  for (const p of loadAllProbes()) {
    const key = `${p.subIntent}|${p.generation}`;
    const existing = counts.get(key);
    if (existing) {
      existing.count++;
    } else {
      counts.set(key, { subIntent: p.subIntent, generation: p.generation, count: 1 });
    }
  }
  return [...counts.values()];
}
