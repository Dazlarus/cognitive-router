// src/coding_blend.ts - routing blend for coding intent
//
// Phase 3 (p3e-007): blend three signals into a single capability score for
// coding-intent routing:
//   1. Bradley-Terry ladder strength (shrunk by comparison count)
//   2. Exec sandbox floor (objective pass@1, min 3 probes)
//   3. Live judged EWMA score (learning loop on real traffic)
//
// Default weights (tunable, calibrated from judge_calibration + replay):
//   w_bt     = 0.50 — ladder is the most vetted signal
//   w_exec   = 0.30 — objective but limited coverage (~3 probes/identity)
//   w_live   = 0.20 — live-judged EWMA (biased toward dominant picks)
//
// Weights are REPLACED by a run of calibrateFromReplay() when enough
// judge_calibration rows + labeled routing_decisions have accumulated.
//
// Acceptance: ladder-top beats current default (zai/glm-5.2) on exec eval
// when both are comparable. The blend is per-bench-identity
// (model|quant|effort); models without a ladder entry fall through to the
// existing learned score path and are unaffected.

import { logger } from "./logger.js";
import type { DBService } from "./db_service.js";
import type { ModelRegistry } from "./model_registry.js";
import { PROMPT_GENERATION } from "./benchmark_ladder.js";
import { computeBradleyTerry, type PairRecord } from "./benchmark_ladder.js";
import { MIN_PROBE_RESULTS } from "./benchmark_exec.js";

// ─── Types ───

export interface CodingBlendWeights {
  /** Weight on normalized+shrunk BT strength. */
  bt: number;
  /** Weight on exec floor (objective pass@1). */
  exec: number;
  /** Weight on live-judged EWMA score. */
  live: number;
}

export interface IdentityBlendResult {
  /** Bench identity key (model|quant|effort). */
  modelKey: string;
  /** Provider resolved from the model key. */
  provider: string;
  /** Raw BT strength from ladder. */
  btStrength: number;
  /** BT strength shrunk by comparison count. */
  btShrunk: number;
  /** BT strength normalized to [0,1] (log-minmax). */
  btNorm: number;
  /** Exec floor (0 when nProbes < MIN_PROBE_RESULTS or missing). */
  execFloor: number;
  /** Number of exec probes contributing to floor. */
  execNProbes: number;
  /** Live EWMA score from capability_overrides or seed. */
  ewmaScore: number;
  /** EWMA sample count (0 = seed-only). */
  ewmaSamples: number;
  /** Final blended score. */
  blended: number;
}

export interface BlendReport {
  weights: CodingBlendWeights;
  generation: string;
  reportedAt: string;
  identities: IdentityBlendResult[];
  /** Summary stats across blended identities. */
  minBlended: number;
  maxBlended: number;
  meanBlended: number;
  /** How many bench identities had enough data for a full blend. */
  fullBlendCount: number;
  /** Identities with only partial data (missing exec or live). */
  partialBlendCount: number;
}

// ─── Constants ───

/** Default blend weights until calibration data accumulates. */
const DEFAULT_WEIGHTS: CodingBlendWeights = { bt: 0.50, exec: 0.30, live: 0.20 };

/** Prior strength for shrinkage (effective comparison count before evidence
 *  dominates). Higher = stronger shrinkage; at ~3 the shrink is mild
 *  (3/(3+3)=0.50 of distance to neutral). */
const SHRINK_PRIOR = 3;

/** Virtual neutral BT strength for shrinkage anchor. */
const NEUTRAL_STRENGTH = 1.0;

/** Small epsilon to avoid log(0). */
const LOG_EPS = 1e-9;

// ─── Helpers ───

/** Count how many unique opponent-model interaction rounds a model key
 *  has in coding verdict rows. Uses getAllBenchmarkVerdicts and counts
 *  unique (opponent, round) pairs. */
function countComparisons(
  db: DBService,
  modelKey: string,
  generation: string = PROMPT_GENERATION,
): number {
  const rows = db.getAllBenchmarkVerdicts("coding", generation);
  const seen = new Set<string>();
  for (const r of rows) {
    if (r.modelA === modelKey) {
      seen.add(`${r.modelB}\u0000${r.round}`);
    } else if (r.modelB === modelKey) {
      seen.add(`${r.modelA}\u0000${r.round}`);
    }
  }
  return seen.size;
}

/** Shrink BT strength toward neutral (1.0):
 *  strength_shrunk = 1 + (strength - 1) * n / (n + prior)
 *  At n=0: shrunk = 1 (neutral)
 *  At n=prior: shrunk halfway toward 1 from raw strength
 *  At n→∞: shrunk → raw strength */
function shrinkStrength(strength: number, nComparisons: number): number {
  const effective = nComparisons;
  const factor = effective / (effective + SHRINK_PRIOR);
  return NEUTRAL_STRENGTH + (strength - NEUTRAL_STRENGTH) * factor;
}

/** Normalize BT strength to [0,1] via log-minmax over a set of strengths.
 *  Returns 0 for neutrals, 1 for strongest. */
function logMinmaxNorm(
  strength: number,
  allStrengths: number[],
): number {
  if (allStrengths.length === 0) return 0.5;
  const logs = allStrengths
    .map((s) => Math.log(Math.max(s, LOG_EPS)))
    .filter((v) => Number.isFinite(v));
  if (logs.length === 0) return 0.5;
  const minLog = Math.min(...logs);
  const maxLog = Math.max(...logs);
  const range = maxLog - minLog;
  if (range < LOG_EPS) return 0.5;
  const logStrength = Math.log(Math.max(strength, LOG_EPS));
  return Math.max(0, Math.min(1, (logStrength - minLog) / range));
}

/** Resolve a bench model key (model|quant|effort) back to a provider for
 *  capability override lookup. Rough heuristic: known prefix mapping. */
function providerOfModelKey(modelKey: string): string {
  const first = modelKey.split("|")[0];
  if (first.startsWith("z-ai/")) return "openrouter";
  if (first.startsWith("glm-") || first === "glm-5-turbo") return "zai";
  if (first.startsWith("deepseek") || first.startsWith("qwen") ||
      first.startsWith("cohere")) return "openrouter";
  return "zai"; // fallback
}

// ─── Blend Service ───

export class CodingBlendService {
  private db: DBService;
  private registry: ModelRegistry;
  private weights: CodingBlendWeights;

  constructor(
    db: DBService,
    modelRegistry: ModelRegistry,
    weights?: Partial<CodingBlendWeights>,
  ) {
    this.db = db;
    this.registry = modelRegistry;
    this.weights = {
      bt: weights?.bt ?? DEFAULT_WEIGHTS.bt,
      exec: weights?.exec ?? DEFAULT_WEIGHTS.exec,
      live: weights?.live ?? DEFAULT_WEIGHTS.live,
    };
    logger.info(
      `CodingBlendService initialized with weights bt=${this.weights.bt.toFixed(3)} ` +
      `exec=${this.weights.exec.toFixed(3)} live=${this.weights.live.toFixed(3)}`
    );
  }

  /** Compute the blended capability score for one identity key.
   *  Returns null when the identity has no ladder entry (no blend possible). */
  blendModelKey(modelKey: string): IdentityBlendResult | null {
    const ladder = this.db.getLadder("coding", PROMPT_GENERATION);
    const entry = ladder.find((e) => e.modelKey === modelKey);
    if (!entry) return null;

    const allStrengths = ladder.map((e) => e.strength);
    const btStrength = entry.strength;
    const nComp = countComparisons(this.db, modelKey);
    const btShrunk = shrinkStrength(btStrength, nComp);
    const btNorm = logMinmaxNorm(btShrunk, allStrengths);

    // Exec floor from exec_benchmark_results via summarizePersisted
    const provider = providerOfModelKey(modelKey);
    const execSummary = this.summarizeExecForModelKey(modelKey, PROMPT_GENERATION);
    const execFloor = execSummary.eligible ? execSummary.meanPassAt1 ?? 0 : 0;
    const execNProbes = execSummary.nProbes;

    // Live EWMA score — try capability_overrides, fall back to seed
    const ewmaScore = this.getEwmaScore(provider, modelKey, "coding");
    const ewmaSamples = this.getEwmaSampleCount(provider, modelKey, "coding");

    // Blend
    const blended =
      this.weights.bt * btNorm +
      this.weights.exec * execFloor +
      this.weights.live * ewmaScore;

    return {
      modelKey,
      provider,
      btStrength,
      btShrunk,
      btNorm,
      execFloor,
      execNProbes,
      ewmaScore,
      ewmaSamples,
      blended: Math.max(0, Math.min(1, blended)),
    };
  }

  /** Compute blended score for a (provider, model) that maps to a known bench
   *  identity. Uses the medium effort level as default mapping for models that
   *  have multiple effort levels. */
  blendProviderModel(
    provider: string,
    model: string,
  ): IdentityBlendResult | null {
    // Try canonical bench key: model|as-served|medum
    const candidateKeys = [
      `${model}|as-served|medium`,
      `${model}|as-served|high`,
    ];
    for (const key of candidateKeys) {
      const result = this.blendModelKey(key);
      if (result) return result;
    }
    return null;
  }

  /** Report on all coding ladder identities. */
  report(): BlendReport {
    const ladder = this.db.getLadder("coding", PROMPT_GENERATION);
    const identities: IdentityBlendResult[] = [];
    let fullCount = 0;
    let partialCount = 0;

    for (const entry of ladder) {
      const result = this.blendModelKey(entry.modelKey);
      if (!result) continue;
      identities.push(result);
      if (result.execFloor > 0 && result.ewmaSamples > 0 && result.btNorm > 0.01) {
        fullCount++;
      } else {
        partialCount++;
      }
    }

    const blendedVals = identities.map((i) => i.blended);
    const minBlended = blendedVals.length > 0 ? Math.min(...blendedVals) : 0;
    const maxBlended = blendedVals.length > 0 ? Math.max(...blendedVals) : 0;
    const meanBlended =
      blendedVals.length > 0
        ? blendedVals.reduce((a, b) => a + b, 0) / blendedVals.length
        : 0;

    return {
      weights: { ...this.weights },
      generation: PROMPT_GENERATION,
      reportedAt: new Date().toISOString(),
      identities,
      minBlended,
      maxBlended,
      meanBlended,
      fullBlendCount: fullCount,
      partialBlendCount: partialCount,
    };
  }

  /** Set blend weights (from calibration). */
  setWeights(w: Partial<CodingBlendWeights>): void {
    if (w.bt !== undefined) this.weights.bt = w.bt;
    if (w.exec !== undefined) this.weights.exec = w.exec;
    if (w.live !== undefined) this.weights.live = w.live;
    const sum = this.weights.bt + this.weights.exec + this.weights.live;
    if (Math.abs(sum - 1) > 0.01) {
      logger.warn(`CodingBlend weights sum to ${sum.toFixed(3)} — normalizing`);
      const inv = 1 / sum;
      this.weights.bt *= inv;
      this.weights.exec *= inv;
      this.weights.live *= inv;
    }
    logger.info(
      `CodingBlend weights updated: bt=${this.weights.bt.toFixed(3)} ` +
      `exec=${this.weights.exec.toFixed(3)} live=${this.weights.live.toFixed(3)}`
    );
  }

  /** True when at least one bench identity has a full blend. */
  hasFullBlend(): boolean {
    return this.report().fullBlendCount > 0;
  }

  // ─── Private helpers ───

  /** Read exec results from DB and produce a summary (mirrors
   *  ExecBenchmark.summarizePersisted but works with the raw DB schema
   *  and the ladder identity's exact modelKey). */
  private summarizeExecForModelKey(
    modelKey: string,
    generation: string,
  ): { nProbes: number; meanPassAt1: number | null; eligible: boolean } {
    const rows = this.db.getExecResultsForIdentity(modelKey, generation);
    // Deduplicate by probe_id (newest row per probe — rows are ORDER BY id DESC)
    const byProbe = new Map<string, {
      passRate: number;
      casesTotal: number;
      failReason: string | null;
    }>();
    for (const r of rows) {
      if (r.failReason === "sandbox_unavailable") continue;
      if (!byProbe.has(r.probeId)) {
        byProbe.set(r.probeId, r);
      }
    }

    if (byProbe.size === 0) {
      return { nProbes: 0, meanPassAt1: null, eligible: false };
    }

    const successes = [...byProbe.values()].filter((r) => r.passRate >= 1.0).length;
    const nProbes = byProbe.size;
    const meanPassAt1 = successes / nProbes;
    return { nProbes, meanPassAt1, eligible: nProbes >= MIN_PROBE_RESULTS };
  }

  /** Get the live EWMA score from capability_overrides or seed. */
  private getEwmaScore(
    provider: string,
    modelKey: string,
    intent: string,
  ): number {
    // First, check capability_overrides via the registry's capability scores.
    // The model key encodes (model, quant, effort); the registry stores by
    // (provider, model). Extract the bare model name.
    const bareModel = modelKey.split("|")[0];
    const overrideScore = this.registry.getCapabilityScore(provider, bareModel, intent);
    // If no override, getSeedScore returns the seed capability value.
    if (overrideScore > 0) return overrideScore;
    return 0.50; // neutral fallback
  }

  /** Get EWMA sample count for a coding model+intent. Uses
   *  getCapabilityOverride which returns sampleCount. */
  private getEwmaSampleCount(
    provider: string,
    modelKey: string,
    intent: string,
  ): number {
    const bareModel = modelKey.split("|")[0];
    const override = this.db.getCapabilityOverride(provider, bareModel, intent);
    if (override) return override.sampleCount;
    return 0;
  }
}