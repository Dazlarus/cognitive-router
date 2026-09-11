// src/ladder_refresh.ts — periodic ladder projection refresh (p3e-006)
//
// The bench sidecar owns passes/judging (CONTRACT.md); verdicts reach the
// router via POST /admin/bench/sync, which hot-applies them to the ladder
// projection in-process (applySyncedVerdictsToLadder). This service is the
// periodic + on-demand backstop: it rebuilds the projection for every
// intent the projection knows, using the exact same rebuildLadder +
// replaceLadder code path as sync hot-apply and startup — the three entry
// points can never diverge.
//
// Write-path freeze (CONTRACT.md §6) holds: this module only reads
// benchmark_verdicts and rewrites the benchmark_ladder projection. It never
// calls models, never judges, never triggers bench passes.
//
// Cheapest-endpoint resolution note (task p3e-006): identity-serving cost
// order is resolved by the router's bench pin-failover arm in
// proxy-stream.ts (x-bench-failover: same-base-model equivalents appended
// cost-ordered, pin first — CONTRACT.md §2 rule 7, Daz ruling 2026-09-07
// 23:36). A router-side ModelCaller would re-open the write path §6 froze,
// so this wiring deliberately does not add one.

import { logger } from "./logger.js";
import type { DBService } from "./db_service.js";
import {
  rebuildLadder,
  PROMPT_GENERATION,
  type LadderEntry,
} from "./benchmark_ladder.js";

/** Default refresh cadence — matches the bench's pass interval default. */
export const DEFAULT_LADDER_REFRESH_MINUTES = 60;

export interface LadderIntentRefresh {
  intent: string;
  /** Identities in the rebuilt projection. */
  identities: number;
  /** Identities whose rank changed vs the previous projection (joined,
   *  left, or moved). 0 = projection already consistent. */
  changed: number;
}

export interface LadderRefreshReport {
  generation: string;
  reason: string;
  refreshedAt: string;
  durationMs: number;
  /** true when a refresh was already in flight and this call was a no-op. */
  skipped: boolean;
  intents: LadderIntentRefresh[];
}

export class LadderProjectionService {
  private readonly db: DBService;
  private readonly intervalMin: number;
  private timer: ReturnType<typeof setInterval> | undefined;
  private running = false;

  constructor(db: DBService, intervalMin?: number) {
    this.db = db;
    const raw = intervalMin ?? Number(process.env.ROUTER_LADDER_REFRESH_MINUTES ?? DEFAULT_LADDER_REFRESH_MINUTES);
    this.intervalMin = Number.isFinite(raw) && raw > 0 ? raw : DEFAULT_LADDER_REFRESH_MINUTES;
  }

  /** Rebuild the projection for every intent it currently tracks.
   *  Safe to call concurrently — serializes on `running` (skips, doesn't
   *  queue), mirroring DiscoveryService's behavior. */
  async refresh(reason: string): Promise<LadderRefreshReport> {
    const startedAt = new Date().toISOString();
    const t0 = Date.now();
    if (this.running) {
      logger.debug(`Ladder refresh skipped (${reason}): already in progress`);
      return {
        generation: PROMPT_GENERATION, reason, refreshedAt: startedAt,
        durationMs: 0, skipped: true, intents: [],
      };
    }
    this.running = true;
    try {
      const intents = this.db.getLadderIntents(PROMPT_GENERATION);
      // Cold-start backstop: a fresh projection is empty but historic
      // verdict rows may exist (restored DB, pre-hot-load installs). The
      // coding intent was the pre-extraction default — keep rebuilding it
      // so the startup path stays idempotent with the old behavior.
      if (!intents.includes("coding") && this.db.getAllBenchmarkVerdicts("coding", PROMPT_GENERATION).length > 0) {
        intents.push("coding");
      }

      const reports: LadderIntentRefresh[] = [];
      for (const intent of intents) {
        const rebuilt: LadderEntry[] = rebuildLadder(this.db, intent as any, PROMPT_GENERATION);
        if (rebuilt.length === 0) continue;
        const prev = new Map(
          this.db.getLadder(intent, PROMPT_GENERATION).map((e) => [e.modelKey, e.rank]),
        );
        let changed = 0;
        for (const e of rebuilt) {
          if (prev.get(e.modelKey) !== e.rank) changed++;
        }
        // identities that left the ladder count as changes too
        for (const key of prev.keys()) {
          if (!rebuilt.some((e) => e.modelKey === key)) changed++;
        }
        this.db.replaceLadder(intent, PROMPT_GENERATION, rebuilt);
        reports.push({ intent, identities: rebuilt.length, changed });
      }

      const report: LadderRefreshReport = {
        generation: PROMPT_GENERATION, reason, refreshedAt: startedAt,
        durationMs: Date.now() - t0, skipped: false, intents: reports,
      };
      logger.info(
        `Ladder refresh (${reason}): ${reports.length} intent(s), ` +
        `${reports.reduce((n, r) => n + r.changed, 0)} rank change(s) in ${report.durationMs}ms`,
      );
      return report;
    } finally {
      this.running = false;
    }
  }

  startPeriodic(): void {
    if (this.timer) return;
    this.timer = setInterval(
      () => {
        this.refresh("interval").catch((err) =>
          logger.error(`Periodic ladder refresh failed: ${err}`),
        );
      },
      this.intervalMin * 60_000,
    );
    this.timer.unref?.();
    logger.info(`Periodic ladder refresh armed (every ${this.intervalMin} minutes)`);
  }

  stopPeriodic(): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = undefined;
    }
  }
}
