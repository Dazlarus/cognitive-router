// src/judge_calibration.ts — periodic judge↔exec agreement report (p3e-005)
//
// Spec docs/EXEC_SANDBOX_SPEC.md §4: "(judge_score, pass@1) pairs" land in
// judge_calibration whenever the judge scores a response the exec sandbox
// also scored; agreement is reported periodically; persistent divergence is
// the trigger to rebuild the coding judge prompt around the rubric — data
// first. This module never calls models: it only reads judge_calibration
// rows and computes the report (write-path freeze, CONTRACT.md §6).
//
// Agreement has two faces:
//   - Threshold agreement: does the judge's "good" (raw ≥ JUDGE_PASS_RAW,
//     default 7 — the rubric's "Good — correct, clear, complete" band)
//     match an objectively passing run (pass@1 ≥ PASS_THRESHOLD, default
//     0.5 = majority of cases)? This is the binary the routing blend will
//     eventually consume.
//   - Correlation: Pearson r between raw judge score and pass@1 — catches
//     graded drift the binary hides (judge 9s passing fewer cases than 6s).
//
// The report is derived data (no new table). Cadence is generous by
// default (24h) — pairs accrue per exec-bench run (48h TTL), so an hourly
// report would mostly re-print the same numbers.

import { logger } from "./logger.js";
import type { DBService } from "./db_service.js";
import { PROMPT_GENERATION } from "./benchmark_ladder.js";

/** Default report cadence: daily (pairs accrue with exec benches, 48h TTL). */
export const DEFAULT_JUDGE_CALIB_MINUTES = 1440;
/** Spec §4 acceptance: a first agreement report at ≥20 pairs; divergence
 *  verdicts are only meaningful at this sample size. */
export const MIN_PAIRS_FOR_REPORT = 20;
/** Default raw judge score at/above which the judge calls a response "passing"
 *  (rubric band 7-8 = "Good — correct, clear, and complete"). */
export const DEFAULT_JUDGE_PASS_RAW = 7;
/** Default pass@1 fraction counting as an objective pass. */
export const DEFAULT_PASS_THRESHOLD = 0.5;
/** Agreement below this over ≥MIN_PAIRS_FOR_REPORT pairs = divergence flag. */
export const DIVERGENCE_AGREEMENT_PCT = 60;

export interface CalibrationRowInput {
  id: number;
  judgeScore: number;
  passRate: number;
  judgeProvider: string | null;
  judgeModel: string | null;
}

export interface CalibrationBucket {
  judgeScore: number;
  n: number;
  meanPassRate: number;
}

export interface JudgeCalibrationStats {
  n: number;
  /** Binary agreement % (judge-pass ↔ exec-pass), or null when n=0. */
  agreementPct: number | null;
  /** Confusion counts for the binary view. */
  judgePassExecPass: number;
  judgePassExecFail: number;
  judgeFailExecPass: number;
  judgeFailExecFail: number;
  /** Pearson r between judgeScore and passRate; null when degenerate
   *  (n<2 or zero variance on either side). */
  pearsonR: number | null;
  /** Mean pass@1 per raw judge score bucket (only non-empty buckets). */
  buckets: CalibrationBucket[];
  /** Mean (judgeScore, passRate) for the slice. */
  meanJudgeScore: number | null;
  meanPassRate: number | null;
}

export interface JudgeCalibrationReport {
  generation: string;
  reason: string;
  reportedAt: string;
  overall: JudgeCalibrationStats;
  /** Same stats split per judge identity that actually scored (fallback
   *  chain provenance — judge_provider/judge_model columns). */
  perJudge: Array<JudgeCalibrationStats & { judge: string }>;
  /** true when overall.n ≥ MIN_PAIRS_FOR_REPORT and agreementPct <
   *  DIVERGENCE_AGREEMENT_PCT — the standing trigger to rebuild the coding
   *  judge prompt (spec §4). */
  divergenceSuspected: boolean;
  /** true when a report was already in flight and this call was a no-op. */
  skipped: boolean;
}

/** Pearson correlation over paired samples; null on degenerate input. */
export function pearson(xs: number[], ys: number[]): number | null {
  const n = Math.min(xs.length, ys.length);
  if (n < 2) return null;
  let sx = 0, sy = 0;
  for (let i = 0; i < n; i++) { sx += xs[i]; sy += ys[i]; }
  const mx = sx / n, my = sy / n;
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx, dy = ys[i] - my;
    num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
  }
  if (dx2 === 0 || dy2 === 0) return null;
  return Math.round((num / Math.sqrt(dx2 * dy2)) * 1000) / 1000;
}

/** Core stats over calibration rows (pure — unit-testable without timers). */
export function computeCalibrationStats(
  rows: CalibrationRowInput[],
  opts: { judgePassRaw?: number; passThreshold?: number } = {},
): JudgeCalibrationStats {
  const judgePassRaw = opts.judgePassRaw ?? DEFAULT_JUDGE_PASS_RAW;
  const passThreshold = opts.passThreshold ?? DEFAULT_PASS_THRESHOLD;

  let jPeP = 0, jPeF = 0, jFeP = 0, jFeF = 0;
  let scoreSum = 0, rateSum = 0;
  const byBucket = new Map<number, { n: number; rateSum: number }>();
  const xs: number[] = [], ys: number[] = [];

  for (const r of rows) {
    const judgePass = r.judgeScore >= judgePassRaw;
    const execPass = r.passRate >= passThreshold;
    if (judgePass && execPass) jPeP++;
    else if (judgePass && !execPass) jPeF++;
    else if (!judgePass && execPass) jFeP++;
    else jFeF++;
    scoreSum += r.judgeScore;
    rateSum += r.passRate;
    const b = byBucket.get(r.judgeScore) ?? { n: 0, rateSum: 0 };
    b.n++; b.rateSum += r.passRate;
    byBucket.set(r.judgeScore, b);
    xs.push(r.judgeScore);
    ys.push(r.passRate);
  }

  const n = rows.length;
  const agreed = jPeP + jFeF;
  const buckets: CalibrationBucket[] = [...byBucket.entries()]
    .map(([judgeScore, b]) => ({
      judgeScore, n: b.n,
      meanPassRate: Math.round((b.rateSum / b.n) * 1000) / 1000,
    }))
    .sort((a, b) => a.judgeScore - b.judgeScore);

  return {
    n,
    agreementPct: n > 0 ? Math.round((agreed / n) * 1000) / 10 : null,
    judgePassExecPass: jPeP,
    judgePassExecFail: jPeF,
    judgeFailExecPass: jFeP,
    judgeFailExecFail: jFeF,
    pearsonR: pearson(xs, ys),
    buckets,
    meanJudgeScore: n > 0 ? Math.round((scoreSum / n) * 100) / 100 : null,
    meanPassRate: n > 0 ? Math.round((rateSum / n) * 1000) / 1000 : null,
  };
}

function buildReport(
  rows: CalibrationRowInput[],
  generation: string,
  reason: string,
  skipped: boolean,
): JudgeCalibrationReport {
  const overall = computeCalibrationStats(rows);
  const byJudge = new Map<string, CalibrationRowInput[]>();
  for (const r of rows) {
    const judge = r.judgeProvider && r.judgeModel
      ? `${r.judgeProvider}/${r.judgeModel}`
      : r.judgeModel ?? r.judgeProvider ?? "unknown";
    const list = byJudge.get(judge) ?? [];
    list.push(r);
    byJudge.set(judge, list);
  }
  const perJudge = [...byJudge.entries()]
    .map(([judge, list]) => ({ judge, ...computeCalibrationStats(list) }))
    .sort((a, b) => b.n - a.n);

  const divergenceSuspected =
    overall.n >= MIN_PAIRS_FOR_REPORT &&
    overall.agreementPct != null &&
    overall.agreementPct < DIVERGENCE_AGREEMENT_PCT;

  return {
    generation, reason,
    reportedAt: new Date().toISOString(),
    overall, perJudge, divergenceSuspected, skipped,
  };
}

export class JudgeCalibrationService {
  private readonly db: DBService;
  private readonly intervalMin: number;
  private timer: ReturnType<typeof setInterval> | undefined;
  private running = false;

  constructor(db: DBService, intervalMin?: number) {
    this.db = db;
    const raw = intervalMin ?? Number(process.env.ROUTER_JUDGE_CALIB_MINUTES ?? DEFAULT_JUDGE_CALIB_MINUTES);
    this.intervalMin = Number.isFinite(raw) && raw > 0 ? raw : DEFAULT_JUDGE_CALIB_MINUTES;
  }

  /** Compute + log the agreement report for the current generation.
   *  Safe to call concurrently — serializes on `running` (skips, doesn't
   *  queue), mirroring LadderProjectionService. */
  async report(reason: string): Promise<JudgeCalibrationReport> {
    if (this.running) {
      logger.debug(`Judge calibration report skipped (${reason}): already in progress`);
      return buildReport([], PROMPT_GENERATION, reason, true);
    }
    this.running = true;
    try {
      const rows = this.db.getJudgeCalibrationRows(PROMPT_GENERATION);
      const rep = buildReport(rows, PROMPT_GENERATION, reason, false);
      const { overall: o } = rep;
      if (o.n === 0) {
        logger.info(`Judge calibration (${reason}): no pairs yet for ${PROMPT_GENERATION}`);
        return rep;
      }
      logger.info(
        `Judge calibration (${reason}): ${o.n} pair(s), agreement ${o.agreementPct}% ` +
        `(judge-pass↔exec-pass ${o.judgePassExecPass}/${o.judgePassExecFail}, ` +
        `judge-fail↔exec-pass ${o.judgeFailExecPass}/${o.judgeFailExecFail}), ` +
        `r=${o.pearsonR ?? "n/a"}, mean judge ${o.meanJudgeScore}/10 vs mean pass@1 ${o.meanPassRate}`,
      );
      if (rep.divergenceSuspected) {
        // Spec §4: persistent divergence is the standing trigger to rebuild
        // the coding judge prompt around the rubric. Data first — flag it.
        logger.warn(
          `Judge↔exec divergence suspected: ${o.agreementPct}% agreement over ${o.n} pairs ` +
          `(< ${DIVERGENCE_AGREEMENT_PCT}% threshold) — coding judge rubric rewrite is warranted`,
        );
      }
      return rep;
    } finally {
      this.running = false;
    }
  }

  startPeriodic(): void {
    if (this.timer) return;
    this.timer = setInterval(
      () => {
        this.report("interval").catch((err) =>
          logger.error(`Periodic judge calibration report failed: ${err}`),
        );
      },
      this.intervalMin * 60_000,
    );
    this.timer.unref?.();
    logger.info(`Periodic judge calibration report armed (every ${this.intervalMin} minutes)`);
  }

  stopPeriodic(): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = undefined;
    }
  }
}
