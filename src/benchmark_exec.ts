// src/benchmark_exec.ts — Exec-based coding benchmark orchestrator
//
// Pipeline: model-call → extractCodeBlock → sandbox run → persist to
// exec_benchmark_results. Aggregates pass@1 mean + Wilson interval per
// identity; N≥3 probe results required before an exec floor is eligible.
//
// Caching (version-hash keyed, mirrors benchmark_chat):
//   - Remotes: 48h TTL, skip when model version hash unchanged.
//   - Locals (ollama): same TTL cap, but fresh runs are gated on a
//     GPU-idle check so exec benches never contend with live traffic.
//   - sandbox_unavailable rows are never treated as cache hits (retry).
//
// Spec: docs/EXEC_SANDBOX_SPEC.md §4–§5. Probes from src/probes.ts;
// sandbox + extractCodeBlock from src/exec_sandbox.ts; identity keys from
// src/benchmark_ladder.ts.

import { logger } from "./logger.js";
import type { DBService } from "./db_service.js";
import {
  encodeKey,
  decodeKey,
  type BenchModelKey,
  type ModelCaller,
} from "./benchmark_ladder.js";
import {
  extractCodeBlock,
  type ExecSandbox,
  type FailReason,
} from "./exec_sandbox.js";
import sandboxDefault from "./exec_sandbox.js";
import { probesFor, type CodingProbe, type CodingSubIntent } from "./probes.js";
import { PROMPT_GENERATION } from "./benchmark_ladder.js";

// ─── Constants ───

/** Remote identities: re-run exec probes at most every 48h (version-pinned). */
export const REMOTE_CACHE_TTL_MS = 48 * 3_600_000;
/** Local identities: same max-age; the GPU-idle gate is the real control. */
export const LOCAL_CACHE_TTL_MS = 48 * 3_600_000;
/** Minimum distinct probe results before an exec floor may count. */
export const MIN_PROBE_RESULTS = 3;
/** Wilson z for 95%. */
const WILSON_Z = 1.96;
/** GPU utilization % below which the GPU counts as idle (nvidia-smi path). */
const DEFAULT_GPU_IDLE_PCT = 15;
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL ?? "http://localhost:11434";
const VERSION_HASH_TIMEOUT_MS = 5_000;

// ─── Types ───

/** One probe's outcome for one identity (fresh or cache-hit). */
export interface ExecProbeOutcome {
  probeId: string;
  cached: boolean;
  /** 0..1 — cases passed / cases total. */
  passRate: number;
  casesPassed: number;
  casesTotal: number;
  failReason: FailReason | null;
  /** Sandbox wall-time (persisted). 0 for cache hits. */
  durationMs: number;
  /** Model-call latency (fresh runs only). */
  latencyMs: number;
  /** Transient model-call/infra failure — NOT persisted, excluded from
   *  summaries (a dead provider must not count as a 0-score probe). */
  modelError?: boolean;
  /** p3e-005: judge also scored this response (raw 0-10) — logged as a
   *  (judge_score, pass@1) calibration pair alongside the exec outcome. */
  judgeScore?: number;
}

/** Aggregated view over an identity's exec results. */
export interface ExecIdentitySummary {
  modelKey: string;
  provider: string;
  generation: string;
  /** Distinct probes with a usable (non-sandbox_unavailable, non-modelError) result. */
  nProbes: number;
  /** Mean of per-probe pass@1 indicators (1.0 = probe passed every case). */
  meanPassAt1: number | null;
  /** Wilson 95% interval on the pass@1 proportion. */
  wilsonLow: number | null;
  wilsonHigh: number | null;
  /** Mean of case-level pass rates (graded partial credit view). */
  meanCasePassRate: number | null;
  /** true only when nProbes >= MIN_PROBE_RESULTS. */
  eligible: boolean;
  failReasonCounts: Record<string, number>;
}

/** p3e-005: judge hook result for calibration logging. */
export interface JudgeScoreResult {
  /** Raw judge score 0–10. */
  rawScore: number;
  judgeProvider?: string;
  judgeModel?: string;
}

export interface ExecBenchDeps {
  db: DBService;
  /** Sends the probe prompt to the identity; wiring applies PINNED_DECODE. */
  callModel: ModelCaller;
  /** p3e-005: optional judge on the SAME fresh response the sandbox scores —
   *  every usable pair lands in judge_calibration (spec §4). Returning null
   *  (judge down/unparseable) skips the pair; exec results are unaffected. */
  judgeScore?: (prompt: string, responseText: string) => Promise<JudgeScoreResult | null>;
  /** Sandbox; defaults to the exec_sandbox module singleton. */
  sandbox?: ExecSandbox;
  /** Model version hash (ollama digest / provider:model for remotes). */
  versionHash?: (provider: string, model: string) => Promise<string>;
  /** GPU idle check for local identities. true=idle, false=busy, null=unknown. */
  gpuIdle?: () => Promise<boolean | null>;
  /** Clock injection for tests. */
  now?: () => number;
}

export interface ExecBenchRunOptions {
  generation?: string;
  subIntent?: CodingSubIntent;
  forceRefresh?: boolean;
  /** Per-run sandbox timeout override (default: sandbox's 10s). */
  timeoutMs?: number;
  /** Test/DI override: run these probes instead of probesFor(). */
  probes?: CodingProbe[];
}

export interface ExecBenchRunReport {
  modelKey: string;
  provider: string;
  generation: string;
  modelVersionHash: string;
  /** Why nothing ran, if the run was gated (e.g. "gpu_busy"). */
  skipped: string | null;
  results: ExecProbeOutcome[];
  summary: ExecIdentitySummary;
}

// ─── Helpers ───

/** Wilson score interval for a binomial proportion. */
export function wilsonInterval(
  successes: number,
  n: number,
  z: number = WILSON_Z,
): { low: number; high: number } | null {
  if (n <= 0) return null;
  const p = successes / n;
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const center = p + z2 / (2 * n);
  const spread = z * Math.sqrt((p * (1 - p) + z2 / (4 * n)) / n);
  return {
    low: Math.max(0, (center - spread) / denom),
    high: Math.min(1, (center + spread) / denom),
  };
}

/** Default version hash: ollama digest for locals, provider/model for remotes.
 *  Mirrors ChatBenchmark.getVersionHash. */
async function defaultVersionHash(provider: string, model: string): Promise<string> {
  if (provider === "ollama") {
    try {
      const resp = await fetch(`${OLLAMA_BASE_URL}/api/show`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model }),
        signal: AbortSignal.timeout(VERSION_HASH_TIMEOUT_MS),
      });
      if (resp.ok) {
        const data = await resp.json() as any;
        const digest: string = data.digest ?? data.details?.digest ?? "";
        if (digest) return digest.replace(/^sha256:/, "").substring(0, 16);
      }
    } catch {
      // fall through to tag-based hash
    }
    return `ollama:${model}`;
  }
  return `${provider}/${model}`;
}

/** Default GPU-idle check.
 *  1) nvidia-smi: idle when every GPU's utilization is below the threshold.
 *  2) Fallback — Ollama /api/ps: no models loaded ⇒ idle; some ⇒ busy;
 *     unreachable ⇒ null (unknown → caller treats as not idle). */
async function defaultGpuIdle(): Promise<boolean | null> {
  const threshold = Number(process.env.ROUTER_EXEC_GPU_IDLE_PCT ?? DEFAULT_GPU_IDLE_PCT);
  const { execFile } = await import("node:child_process");
  const { promisify } = await import("node:util");
  const run = promisify(execFile);
  try {
    const { stdout } = await run("nvidia-smi", [
      "--query-gpu=utilization.gpu",
      "--format=csv,noheader,nounits",
    ], { timeout: 5_000 });
    const utils = stdout
      .split(/\r?\n/)
      .map((l) => parseFloat(l.trim()))
      .filter((v) => Number.isFinite(v));
    if (utils.length === 0) return null;
    return utils.every((u) => u < threshold);
  } catch {
    // nvidia-smi unavailable — try Ollama
  }
  try {
    const resp = await fetch(`${OLLAMA_BASE_URL}/api/ps`, {
      signal: AbortSignal.timeout(3_000),
    });
    if (!resp.ok) return null;
    const data = await resp.json() as any;
    const loaded: unknown[] = data?.models ?? [];
    return loaded.length === 0;
  } catch {
    return null;
  }
}

// ─── Orchestrator ───

export class ExecBenchmark {
  private readonly db: DBService;
  private readonly callModel: ModelCaller;
  private readonly judgeScore: ((prompt: string, responseText: string) => Promise<JudgeScoreResult | null>) | undefined;
  private readonly sandbox: ExecSandbox;
  private readonly versionHash: (provider: string, model: string) => Promise<string>;
  private readonly gpuIdle: () => Promise<boolean | null>;
  private readonly now: () => number;

  constructor(deps: ExecBenchDeps) {
    this.db = deps.db;
    this.callModel = deps.callModel;
    this.judgeScore = deps.judgeScore;
    this.sandbox = deps.sandbox ?? sandboxDefault;
    this.versionHash = deps.versionHash ?? defaultVersionHash;
    this.gpuIdle = deps.gpuIdle ?? defaultGpuIdle;
    this.now = deps.now ?? Date.now;
  }

  /** Benchmark one identity across the probe set for a generation. */
  async benchmarkIdentity(
    provider: string,
    key: BenchModelKey,
    opts: ExecBenchRunOptions = {},
  ): Promise<ExecBenchRunReport> {
    const generation = opts.generation ?? PROMPT_GENERATION;
    const modelKey = encodeKey(key);
    const isLocal = provider === "ollama";

    const versionHash = await this.safeVersionHash(provider, key.model);
    const probes = opts.probes ?? probesFor(opts.subIntent ?? "quick", generation);

    const results: ExecProbeOutcome[] = [];
    let skipped: string | null = null;

    if (probes.length === 0) {
      skipped = "no_probes";
      logger.warn(`exec-bench: no probes for generation ${generation} — nothing to do`);
    }

    // Local identities are gated on a GPU-idle window before ANY fresh spend,
    // but cache hits are still honored while gated (no GPU work needed).
    if (!skipped && isLocal && !opts.forceRefresh) {
      const idle = await this.gpuIdle();
      if (idle === false) {
        skipped = "gpu_busy";
        logger.info(`exec-bench: GPU busy — skipping fresh local runs for ${modelKey}`);
      } else if (idle === null) {
        skipped = "gpu_unknown";
        logger.info(`exec-bench: GPU idle state unknown — refusing fresh local runs for ${modelKey}`);
      }
    }

    for (const probe of probes) {
      if (skipped) {
        // Gated: still emit cache hits where fresh results would be skipped.
        const outcome = this.cacheLookup(modelKey, probe, versionHash, isLocal, opts);
        if (outcome) results.push(outcome);
        continue;
      }
      const outcome = await this.runOrReuse(
        provider, modelKey, probe, versionHash, isLocal, opts,
      );
      if (outcome) results.push(outcome);
    }

    const summary = this.summarize(provider, modelKey, generation, versionHash, results);
    return { modelKey, provider, generation, modelVersionHash: versionHash, skipped, results, summary };
  }

  /** Pure cache lookup (no side effects); null when no usable cache entry. */
  private cacheLookup(
    modelKey: string,
    probe: CodingProbe,
    versionHash: string,
    isLocal: boolean,
    opts: ExecBenchRunOptions,
  ): ExecProbeOutcome | null {
    if (opts.forceRefresh) return null;
    const cached = this.db.getLatestExecBenchmarkResult(modelKey, probe.id, probe.generation);
    if (
      cached &&
      cached.modelVersionHash === versionHash &&
      cached.failReason !== "sandbox_unavailable" &&
      this.ageMs(cached.timestamp) < (isLocal ? LOCAL_CACHE_TTL_MS : REMOTE_CACHE_TTL_MS)
    ) {
      return {
        probeId: probe.id,
        cached: true,
        passRate: cached.passRate,
        casesPassed: cached.casesPassed,
        casesTotal: cached.casesTotal,
        failReason: (cached.failReason as FailReason) ?? null,
        durationMs: cached.durationMs,
        latencyMs: 0,
      };
    }
    return null;
  }

  /** Cache check → (skip | model-call → extract → sandbox) → persist.
   *  Fresh runs also produce a (judge_score, pass@1) calibration pair when
   *  a judge hook is wired (p3e-005, spec §4). */
  private async runOrReuse(
    provider: string,
    modelKey: string,
    probe: CodingProbe,
    versionHash: string,
    isLocal: boolean,
    opts: ExecBenchRunOptions,
  ): Promise<ExecProbeOutcome | null> {
    if (!opts.forceRefresh) {
      const hit = this.cacheLookup(modelKey, probe, versionHash, isLocal, opts);
      if (hit) return hit;
    }

    // 1) Model call
    let responseText: string;
    let latencyMs = 0;
    try {
      const t0 = this.now();
      responseText = await this.callModel(decodeKey(modelKey), probe.prompt);
      latencyMs = this.now() - t0;
    } catch (err) {
      logger.warn(`exec-bench: model call failed for ${modelKey} × ${probe.id}: ${err}`);
      // Model call failure is transient infra — NOT persisted, excluded from
      // summaries (a down provider must not masquerade as a 0-score probe).
      return {
        probeId: probe.id, cached: false, passRate: 0, casesPassed: 0,
        casesTotal: probe.harness.cases.length, failReason: null,
        durationMs: 0, latencyMs, modelError: true,
      };
    }

    // 1b) Judge hook (p3e-005): score the same fresh response the sandbox
    // will execute. Failure → no pair, never an exec failure.
    const judged = await this.safeJudgeScore(probe.prompt, responseText);

    // 2) Extract code block (last fenced block in the target language)
    const code = extractCodeBlock(responseText, probe.language);
    if (code === null) {
      return this.persist(probe, modelKey, provider, versionHash, {
        casesPassed: 0,
        casesTotal: probe.harness.cases.length,
        passRate: 0,
        failReason: "no_code_block",
        durationMs: 0,
        backend: null,
      }, latencyMs, judged);
    }

    // 3) Sandbox run (full harness — visible + hidden cases)
    const run = await this.sandbox.run({
      language: probe.language,
      code,
      harness: probe.harness,
      ...(opts.timeoutMs ? { timeoutMs: opts.timeoutMs } : {}),
    });

    return this.persist(probe, modelKey, provider, versionHash, run, latencyMs, judged);
  }

  /** Judge hook wrapper — swallows every failure (null = no pair). */
  private async safeJudgeScore(
    prompt: string,
    responseText: string,
  ): Promise<JudgeScoreResult | null> {
    if (!this.judgeScore) return null;
    try {
      return await this.judgeScore(prompt, responseText);
    } catch (err) {
      logger.warn(`exec-bench: judge hook failed (no calibration pair): ${err}`);
      return null;
    }
  }

  /** Persist a sandbox (or extraction) outcome and return the probe outcome.
   *  p3e-005: when the judge also scored this response, the (judge_score,
   *  pass@1) pair lands in judge_calibration — unless the exec side is
   *  unusable (sandbox_unavailable), which carries no calibration signal. */
  private persist(
    probe: CodingProbe,
    modelKey: string,
    provider: string,
    versionHash: string,
    run: {
      casesPassed: number; casesTotal: number; passRate: number;
      failReason: FailReason | null; durationMs: number; backend: string | null;
    },
    latencyMs: number,
    judged?: JudgeScoreResult | null,
  ): ExecProbeOutcome {
    this.db.saveExecBenchmarkResult({
      modelKey,
      provider,
      probeId: probe.id,
      generation: probe.generation,
      passRate: run.passRate,
      casesPassed: run.casesPassed,
      casesTotal: run.casesTotal,
      failReason: run.failReason,
      durationMs: run.durationMs,
      modelVersionHash: versionHash,
      backend: run.backend,
      timestamp: new Date(this.now()).toISOString(),
    });
    if (judged && run.failReason !== "sandbox_unavailable") {
      try {
        this.db.insertJudgeCalibration({
          probeId: probe.id,
          modelKey,
          generation: probe.generation,
          judgeScore: judged.rawScore,
          passRate: run.passRate,
          casesPassed: run.casesPassed,
          casesTotal: run.casesTotal,
          judgeProvider: judged.judgeProvider ?? null,
          judgeModel: judged.judgeModel ?? null,
          execBackend: run.backend,
          timestamp: new Date(this.now()).toISOString(),
        });
      } catch (err) {
        logger.warn(`exec-bench: calibration pair not logged for ${modelKey} × ${probe.id}: ${err}`);
      }
    }
    return {
      probeId: probe.id,
      cached: false,
      passRate: run.passRate,
      casesPassed: run.casesPassed,
      casesTotal: run.casesTotal,
      failReason: run.failReason,
      durationMs: run.durationMs,
      latencyMs,
      ...(judged ? { judgeScore: judged.rawScore } : {}),
    };
  }

  /** Aggregate pass@1 mean + Wilson interval over usable probe results. */
  summarize(
    provider: string,
    modelKey: string,
    generation: string,
    versionHash: string,
    results: ExecProbeOutcome[],
  ): ExecIdentitySummary {
    // Deduplicate by probe id (latest result wins — results are already
    // newest-last from the run loop), drop sandbox_unavailable rows.
    const byProbe = new Map<string, ExecProbeOutcome>();
    for (const r of results) {
      if (r.failReason === "sandbox_unavailable") continue;
      if (r.modelError) continue;
      byProbe.set(r.probeId, r);
    }

    const failReasonCounts: Record<string, number> = {};
    let caseRateSum = 0;
    for (const r of byProbe.values()) {
      if (r.failReason) failReasonCounts[r.failReason] = (failReasonCounts[r.failReason] ?? 0) + 1;
      caseRateSum += r.passRate;
    }

    const nProbes = byProbe.size;
    if (nProbes === 0) {
      return {
        modelKey, provider, generation,
        nProbes: 0, meanPassAt1: null, wilsonLow: null, wilsonHigh: null,
        meanCasePassRate: null, eligible: false, failReasonCounts,
      };
    }

    // pass@1 per probe is a strict indicator: the single sample passed every
    // case. Wilson interval is on that binomial proportion; meanCasePassRate
    // keeps the graded partial-credit view.
    const successes = [...byProbe.values()].filter((r) => r.passRate >= 1.0).length;
    const interval = wilsonInterval(successes, nProbes);

    return {
      modelKey,
      provider,
      generation,
      nProbes,
      meanPassAt1: successes / nProbes,
      wilsonLow: interval?.low ?? null,
      wilsonHigh: interval?.high ?? null,
      meanCasePassRate: caseRateSum / nProbes,
      eligible: nProbes >= MIN_PROBE_RESULTS,
      failReasonCounts,
    };
  }

  /** Rebuild a summary from persisted rows (no model calls). */
  summarizePersisted(
    provider: string,
    modelKey: string,
    generation: string = PROMPT_GENERATION,
  ): ExecIdentitySummary {
    const rows = this.db.getExecResultsForIdentity(modelKey, generation);
    // Keep only the newest row per probe.
    const byProbe = new Map<string, typeof rows[number]>();
    for (const row of rows) {
      if (row.failReason === "sandbox_unavailable") continue;
      // rows are ORDER BY id DESC (newest first); keep the first-seen (newest) per probe
      if (!byProbe.has(row.probeId)) byProbe.set(row.probeId, row);
    }
    const outcomes: ExecProbeOutcome[] = [...byProbe.values()].map((r) => ({
      probeId: r.probeId,
      cached: true,
      passRate: r.passRate,
      casesPassed: r.casesPassed,
      casesTotal: r.casesTotal,
      failReason: (r.failReason as FailReason) ?? null,
      durationMs: r.durationMs,
      latencyMs: 0,
    }));
    return this.summarize(provider, modelKey, generation, "", outcomes);
  }

  private ageMs(timestamp: string): number {
    const t = new Date(timestamp).getTime();
    return this.now() - t;
  }

  private async safeVersionHash(provider: string, model: string): Promise<string> {
    try {
      return await this.versionHash(provider, model);
    } catch {
      return "unknown";
    }
  }
}
