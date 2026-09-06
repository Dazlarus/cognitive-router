// src/db_service.ts — SQLite persistence layer
// All database operations encapsulated here.

import Database from "better-sqlite3";
import { logger } from "./logger.js";
import { normalizeNote, noteHash } from "./learning_guards.js";

function noteHashOf(note: string): string {
  return noteHash(normalizeNote(note));
}

export interface DecisionRecord {
  timestamp: string;
  sessionKey: string;
  messageHash: string;
  intent: string;
  confidence: number;
  provider: string;
  model: string;
  scores: Record<string, unknown>;
  overallScore: number;
  outcome: string;
  requestId?: string;
  candidatesJson?: string | null;
  contextFilterJson?: string | null;
  modalityFilterJson?: string | null;
}

export interface CallOutcomeRecord {
  provider: string;
  model: string;
  durationMs: number;
  outcome: string;
  timestamp: string;
}

export interface RetryRecord {
  requestId: string;
  failedProvider: string;
  failedOutcome: string;
  retryProvider: string;
  timestamp: string;
}

export class DBService {
  private db: Database.Database;

  constructor(dbPath: string) {
    this.db = new Database(dbPath);
    this.db.pragma("journal_mode = WAL");
    this.db.pragma("foreign_keys = ON");
    logger.info(`Database opened: ${dbPath}`);
  }

  /** Get raw database instance for use in migrations or benchmarks */
  getDb(): Database.Database {
    return this.db;
  }

  async initializeSchema(): Promise<void> {
    try {
      this.db.exec(TABLE_SCHEMA_SQL);
    } catch (err) {
      logger.warn(`Schema initialization warning: ${err}`);
    }
    this.addColumnIfMissing("routing_decisions", "request_id", "TEXT");
    this.addColumnIfMissing("routing_decisions", "candidates_json", "TEXT");
    this.addColumnIfMissing("routing_decisions", "context_filter_json", "TEXT");
    this.addColumnIfMissing("routing_decisions", "modality_filter_json", "TEXT");
    this.db.exec(INDEX_SCHEMA_SQL);
    this.migrate_v3();
    this.migrate_v4();
    logger.info("Database schema verified.");
  }

  migrate_v3(): void {
    const version = this.db.pragma("user_version", { simple: true }) as number;
    if (version < 3) {
      logger.info(`Migrating database to user_version = 3...`);
      const migration = this.db.transaction(() => {
        this.db.prepare(`
          CREATE TABLE IF NOT EXISTS abort_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            turns_completed INTEGER,
            duration_ms INTEGER,
            session_key TEXT
          )
        `).run();

        this.db.prepare(`
          CREATE TABLE IF NOT EXISTS session_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_key TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            outcome TEXT NOT NULL,
            turns_completed INTEGER,
            duration_ms INTEGER,
            timestamp TEXT NOT NULL
          )
        `).run();

        this.db.prepare("PRAGMA user_version = 3").run();
      });
      migration();
      logger.info(`Database migrated to user_version = 3 successfully.`);
    }
  }

  rollback_v3(): void {
    logger.info(`Rolling back database user_version to 2...`);
    const rollback = this.db.transaction(() => {
      this.db.prepare(`DROP TABLE IF EXISTS abort_events`).run();
      this.db.prepare(`DROP TABLE IF EXISTS session_outcomes`).run();
      this.db.prepare("PRAGMA user_version = 2").run();
    });
    rollback();
    logger.info(`Database rolled back to user_version = 2 successfully.`);
  }

  /** Phase-1 learning-loop hardening (LEARNING_LOOP_DESIGN.md §4.2/§4.5).
   *  Additive only: new tables + new columns; never destructive. */
  migrate_v4(): void {
    const version = this.db.pragma("user_version", { simple: true }) as number;
    if (version >= 4) return;
    logger.info(`Migrating database to user_version = 4 (learning-loop hardening)...`);

    this.db.exec(`
      CREATE TABLE IF NOT EXISTS capability_overrides_archive (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        provider    TEXT NOT NULL,
        model       TEXT NOT NULL,
        intent      TEXT NOT NULL,
        score       REAL NOT NULL,
        sample_count INTEGER NOT NULL DEFAULT 1,
        last_judged TEXT NOT NULL,
        pinned      INTEGER NOT NULL DEFAULT 0,
        archived_at TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS capability_movement (
        provider         TEXT NOT NULL,
        model            TEXT NOT NULL,
        intent           TEXT NOT NULL,
        window_start_ts  TEXT NOT NULL,
        cumulative_delta REAL NOT NULL DEFAULT 0,
        last_updated     TEXT NOT NULL,
        PRIMARY KEY (provider, model, intent, window_start_ts)
      );

      CREATE TABLE IF NOT EXISTS shadow_decisions (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp        TEXT NOT NULL,
        provider         TEXT NOT NULL,
        model            TEXT NOT NULL,
        intent           TEXT NOT NULL,
        confidence       REAL,
        gate_arm         TEXT,
        rejection_reason TEXT,
        pre_clamp        REAL,
        would_be_value   REAL,
        sample_n         INTEGER
      );

      CREATE TABLE IF NOT EXISTS quarantined_pairs (
        provider    TEXT NOT NULL,
        model       TEXT NOT NULL,
        intent      TEXT NOT NULL,
        note_hash   TEXT NOT NULL,
        note_text   TEXT,
        occurrences INTEGER NOT NULL,
        quarantined_at TEXT NOT NULL,
        PRIMARY KEY (provider, model, intent, note_hash)
      );
    `);

    this.addColumnIfMissing("capability_overrides", "pinned", "INTEGER NOT NULL DEFAULT 0");
    this.addColumnIfMissing("judge_history", "no_apply", "INTEGER NOT NULL DEFAULT 0");
    this.addColumnIfMissing("judge_history", "gate_arm", "TEXT");
    this.addColumnIfMissing("judge_history", "gate_reason", "TEXT");
    this.addColumnIfMissing("judge_history", "note_hash", "TEXT");
    this.addColumnIfMissing("judge_history", "effort_level", "TEXT");

    this.db.exec(`
      CREATE INDEX IF NOT EXISTS idx_cap_archive_batch ON capability_overrides_archive(archived_at);
      CREATE INDEX IF NOT EXISTS idx_cap_movement_window ON capability_movement(provider, model, intent, window_start_ts);
      CREATE INDEX IF NOT EXISTS idx_shadow_timestamp ON shadow_decisions(timestamp);
      CREATE INDEX IF NOT EXISTS idx_shadow_pair ON shadow_decisions(provider, model, intent);
      CREATE INDEX IF NOT EXISTS idx_judge_note_hash ON judge_history(note_hash);
      CREATE INDEX IF NOT EXISTS idx_judge_no_apply ON judge_history(no_apply);
    `);

    // Backfill note hashes for existing judge history (needed by the canary).
    this.backfillJudgeNoteHashes();
    // Backfill 7d capability movement from judge_history (reverse-EMA estimate).
    this.backfillCapabilityMovement7d();

    this.db.prepare("PRAGMA user_version = 4").run();
    logger.info(`Database migrated to user_version = 4 successfully.`);
  }

  /** Reverse migration for v4 (drops only the v4-added tables). */
  rollback_v4(): void {
    logger.info(`Rolling back database user_version to 3...`);
    const rollback = this.db.transaction(() => {
      this.db.prepare(`DROP TABLE IF EXISTS quarantined_pairs`).run();
      this.db.prepare(`DROP TABLE IF EXISTS shadow_decisions`).run();
      this.db.prepare(`DROP TABLE IF EXISTS capability_movement`).run();
      this.db.prepare(`DROP TABLE IF EXISTS capability_overrides_archive`).run();
      this.db.prepare("PRAGMA user_version = 3").run();
    });
    rollback();
    logger.info(`Database rolled back to user_version = 3 successfully.`);
  }

  private backfillJudgeNoteHashes(): void {
    const rows = this.db.prepare(
      `SELECT id, judge_note FROM judge_history WHERE note_hash IS NULL AND judge_note IS NOT NULL`,
    ).all() as Array<{ id: number; judge_note: string }>;
    if (rows.length === 0) return;
    const update = this.db.prepare(`UPDATE judge_history SET note_hash = ? WHERE id = ?`);
    const tx = this.db.transaction(() => {
      for (const r of rows) {
        update.run(noteHashOf(r.judge_note), r.id);
      }
    });
    tx();
    logger.info(`Backfilled note_hash for ${rows.length} judge_history rows.`);
  }

  /** Estimate per-day capability movement over the last 7 days by un-applying
   *  the EMA (alpha=0.25) from the current stored value, day by day. Movement
   *  is attributed to the UTC day bucket in which the evals occurred. */
  private backfillCapabilityMovement7d(alpha = 0.25): void {
    const now = Date.now();
    const windowStart = new Date(now - 7 * 86_400_000).toISOString();
    const pairs = this.db.prepare(
      `SELECT DISTINCT provider, model, intent FROM judge_history WHERE timestamp >= ?`,
    ).all(windowStart) as Array<{ provider: string; model: string; intent: string }>;
    if (pairs.length === 0) return;

    const getOverride = this.db.prepare(
      `SELECT score FROM capability_overrides WHERE provider = ? AND model = ? AND intent = ?`,
    );
    const getEvals = this.db.prepare(
      `SELECT timestamp, judge_score FROM judge_history
       WHERE provider = ? AND model = ? AND intent = ? AND timestamp >= ? AND no_apply = 0
       ORDER BY timestamp ASC`,
    );
    const upsertMovement = this.db.prepare(
      `INSERT INTO capability_movement (provider, model, intent, window_start_ts, cumulative_delta, last_updated)
       VALUES (?, ?, ?, ?, ?, ?)
       ON CONFLICT(provider, model, intent, window_start_ts) DO UPDATE SET
         cumulative_delta = cumulative_delta + excluded.cumulative_delta,
         last_updated = excluded.last_updated`,
    );

    let touched = 0;
    const tx = this.db.transaction(() => {
      for (const p of pairs) {
        const cur = getOverride.get(p.provider, p.model, p.intent) as { score: number } | undefined;
        if (!cur) continue;
        const evals = getEvals.all(p.provider, p.model, p.intent, windowStart) as Array<{ timestamp: string; judge_score: number }>;
        if (evals.length === 0) continue;

        // Walk the value forward from the current score by re-deriving the
        // daily trajectory: start from value-before-window estimated by
        // un-applying every eval in the window.
        let value = cur.score;
        for (let i = evals.length - 1; i >= 0; i--) {
          value = (value - alpha * (evals[i].judge_score / 10)) / (1 - alpha);
          value = Math.min(1, Math.max(0, value));
        }
        // Now replay forward, bucketing movement per UTC day.
        let dayBucket = "";
        for (const e of evals) {
          const next = value * (1 - alpha) + (e.judge_score / 10) * alpha;
          const day = e.timestamp.slice(0, 10);
          if (day !== dayBucket) dayBucket = day;
          const delta = Math.abs(next - value);
          if (delta > 0) {
            upsertMovement.run(p.provider, p.model, p.intent, day, delta, e.timestamp);
          }
          value = next;
        }
        touched++;
      }
    });
    tx();
    logger.info(`Backfilled capability_movement for ${touched} pair(s) over the 7d window.`);
  }

  recordAbortEvent(data: {
    provider: string;
    model: string;
    turnsCompleted?: number;
    durationMs?: number;
    sessionKey?: string;
  }): void {
    const stmt = this.db.prepare(`
      INSERT INTO abort_events (timestamp, provider, model, turns_completed, duration_ms, session_key)
      VALUES (?, ?, ?, ?, ?, ?)
    `);
    stmt.run(
      new Date().toISOString(),
      data.provider,
      data.model,
      data.turnsCompleted ?? null,
      data.durationMs ?? null,
      data.sessionKey ?? null
    );
  }

  private addColumnIfMissing(tableName: string, columnName: string, definition: string): void {
    try {
      const columns = this.db.prepare(`PRAGMA table_info(${tableName})`).all() as Array<{ name: string }>;
      if (columns.some((column) => column.name === columnName)) {
        logger.debug(`Column ${columnName} already exists in ${tableName}, skipping ALTER TABLE.`);
        return;
      }

      logger.info(`Adding column ${columnName} to ${tableName}...`);
      this.db.exec(`ALTER TABLE ${tableName} ADD COLUMN ${columnName} ${definition}`);
      logger.info(`Column ${columnName} added to ${tableName}.`);
    } catch (err) {
      logger.warn(`Failed to add column ${columnName} to ${tableName}: ${err}`);
      // Don't fail schema initialization — just warn
    }
  }

  // ─── Decision Logging ───

  recordDecision(data: DecisionRecord): void {
    const stmt = this.db.prepare(`
      INSERT INTO routing_decisions
        (timestamp, session_key, message_hash, intent, confidence,
         chosen_provider, chosen_model, routing_scores, overall_score, outcome, request_id, candidates_json, context_filter_json, modality_filter_json)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    stmt.run(
      data.timestamp,
      data.sessionKey,
      data.messageHash,
      data.intent,
      data.confidence,
      data.provider,
      data.model,
      JSON.stringify(data.scores),
      data.overallScore,
      data.outcome,
      data.requestId ?? null,
      data.candidatesJson ?? null,
      data.contextFilterJson ?? null,
      data.modalityFilterJson ?? null,
    );
  }

  updateDecisionOutcome(requestId: string, outcome: string, durationMs: number): void {
    if (!requestId) return;
    this.db.prepare(
      `UPDATE routing_decisions SET outcome = ? WHERE request_id = ? AND outcome = 'PENDING'`
    ).run(outcome, requestId);
  }

  saveCircuitState(providerName: string, status: string, consecutiveFailures: number, backoffTier: number): void {
    this.db.prepare(
      `INSERT INTO provider_health (provider_name, status, rate_limit_errors, circuit_open, last_check, metadata)
       VALUES (?, ?, ?, ?, ?, ?)
       ON CONFLICT(provider_name) DO UPDATE SET
         status = excluded.status,
         rate_limit_errors = excluded.rate_limit_errors,
         circuit_open = excluded.circuit_open,
         last_check = excluded.last_check,
         metadata = excluded.metadata`
    ).run(
      providerName,
      status,
      consecutiveFailures,
      status === "circuit_open" ? 1 : 0,
      new Date().toISOString(),
      JSON.stringify({ consecutiveFailures, backoffTier }),
    );
  }

  loadCircuitState(providerName: string): { status: string; consecutiveFailures: number; backoffTier: number; lastCheck?: string | null } | null {
    const row = this.db.prepare(
      `SELECT status, metadata, last_check FROM provider_health WHERE provider_name = ?`
    ).get(providerName) as any;
    if (!row || row.status === 'HEALTHY') return null;
    try {
      const meta = JSON.parse(row.metadata || '{}');
      return {
        status: row.status.toLowerCase(),
        consecutiveFailures: meta.consecutiveFailures || 0,
        backoffTier: meta.backoffTier || 0,
        lastCheck: row.last_check ?? null,
      };
    } catch {
      return null;
    }
  }

  recordCallOutcome(data: CallOutcomeRecord): void {
    const stmt = this.db.prepare(`
      INSERT INTO call_outcomes
        (timestamp, provider, model, duration_ms, outcome)
      VALUES (?, ?, ?, ?, ?)
    `);
    stmt.run(
      data.timestamp,
      data.provider,
      data.model,
      data.durationMs,
      data.outcome,
    );

    // Update rolling stats for this model
    this.updateModelStats(data.provider, data.model, data.outcome, data.durationMs);
  }

  private updateModelStats(
    provider: string,
    model: string,
    outcome: string,
    durationMs: number,
  ): void {
    const key = `${provider}/${model}`;

    this.db.prepare(`
      INSERT INTO model_stats (provider_model, total_calls, success_count, failure_count, total_latency_ms, last_updated)
      VALUES (?, 1, ?, ?, ?, ?)
      ON CONFLICT(provider_model) DO UPDATE SET
        total_calls = total_calls + 1,
        success_count = success_count + ?,
        failure_count = failure_count + ?,
        total_latency_ms = total_latency_ms + ?,
        last_updated = ?
    `).run(
      key,
      outcome === "success" ? 1 : 0,
      outcome !== "success" ? 1 : 0,
      durationMs,
      new Date().toISOString(),
      outcome === "success" ? 1 : 0,
      outcome !== "success" ? 1 : 0,
      durationMs,
      new Date().toISOString(),
    );
  }

  // ─── Query Methods (for dashboard/CLI) ───

  getRecentDecisions(limit = 100): any[] {
    return this.db
      .prepare(
        "SELECT id, timestamp, session_key, intent, confidence, chosen_provider, chosen_model, routing_scores, overall_score, outcome, candidates_json, context_filter_json FROM routing_decisions ORDER BY timestamp DESC LIMIT ?",
      )
      .all(limit);
  }

  getModelStats(): any[] {
    return this.db
      .prepare(
        `SELECT provider_model,
                total_calls,
                success_count,
                failure_count,
                CAST(total_latency_ms AS REAL) / NULLIF(total_calls, 0) AS avg_latency_ms,
                CAST(failure_count AS REAL) / NULLIF(total_calls, 0) AS failure_rate,
                last_updated
         FROM model_stats
         ORDER BY total_calls DESC`,
      )
      .all();
  }

  getProviderHealth(): any[] {
    return this.db
      .prepare("SELECT * FROM provider_health ORDER BY provider_name")
      .all();
  }

  getSpendByProvider(): any[] {
    // Placeholder — will be populated once cost tracking is wired
    return this.db
      .prepare(
        `SELECT chosen_provider AS provider,
                COUNT(*) AS decisions,
                COUNT(CASE WHEN outcome = 'SUCCESS' THEN 1 END) AS successes,
                COUNT(CASE WHEN outcome != 'SUCCESS' THEN 1 END) AS failures
         FROM routing_decisions
         GROUP BY chosen_provider
         ORDER BY decisions DESC`,
      )
      .all();
  }

  // ─── Retry Tracking ───

  getDecisionByRequestId(requestId: string): DecisionRecord | null {
    const stmt = this.db.prepare(
      `SELECT timestamp, session_key, message_hash, intent, confidence,
              chosen_provider, chosen_model, routing_scores, overall_score, outcome, request_id
       FROM routing_decisions
       WHERE request_id = ? ORDER BY timestamp DESC LIMIT 1`,
    );
    const row = stmt.get(requestId) as Record<string, any> | null;
    if (!row) return null;

    return {
      timestamp: row.timestamp,
      sessionKey: row.session_key,
      messageHash: row.message_hash,
      intent: row.intent,
      confidence: row.confidence,
      provider: row.chosen_provider,
      model: row.chosen_model,
      scores: row.routing_scores ? JSON.parse(row.routing_scores) : {},
      overallScore: row.overall_score,
      outcome: row.outcome,
      requestId: row.request_id,
    };
  }

  recordRetry(data: RetryRecord): void {
    const stmt = this.db.prepare(`
      INSERT INTO retry_attempts
        (timestamp, request_id, failed_provider, failed_outcome, retry_provider)
      VALUES (?, ?, ?, ?, ?)
    `);
    stmt.run(
      data.timestamp,
      data.requestId,
      data.failedProvider,
      data.failedOutcome,
      data.retryProvider,
    );
  }

  getRetryCount(requestId: string): number {
    const stmt = this.db.prepare(
      `SELECT COUNT(*) AS count FROM retry_attempts WHERE request_id = ?`,
    );
    const result = stmt.get(requestId) as { count: number };
    return result.count;
  }

  // ─── Capability Overrides (LLM-as-judge feedback) ───

  /** Insert a shadow-window decision row (METADATA ONLY — never prompt/response content). */
  recordShadowDecision(data: {
    provider: string;
    model: string;
    intent: string;
    confidence: number | null;
    gateArm: string;
    rejectionReason: string | null;
    preClamp: number | null;
    wouldBeValue: number | null;
    sampleN: number | null;
  }): void {
    this.db.prepare(`
      INSERT INTO shadow_decisions
        (timestamp, provider, model, intent, confidence, gate_arm, rejection_reason, pre_clamp, would_be_value, sample_n)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      new Date().toISOString(),
      data.provider,
      data.model,
      data.intent,
      data.confidence,
      data.gateArm,
      data.rejectionReason,
      data.preClamp,
      data.wouldBeValue,
      data.sampleN,
    );
  }

  getShadowDecisionCount(): number {
    return (this.db.prepare(`SELECT COUNT(*) AS c FROM shadow_decisions`).get() as { c: number }).c;
  }

  /** Set/unset the manual pin on a capability cell. Pinned cells refuse learner updates. */
  setCapabilityPin(provider: string, model: string, intent: string, pinned: boolean): void {
    this.db.prepare(
      `UPDATE capability_overrides SET pinned = ? WHERE provider = ? AND model = ? AND intent = ?`,
    ).run(pinned ? 1 : 0, provider, model, intent);
  }

  /** Is this capability cell pinned (either pinned=1 row exists)? */
  isCapabilityPinned(provider: string, model: string, intent: string): boolean {
    const row = this.db.prepare(
      `SELECT pinned FROM capability_overrides WHERE provider = ? AND model = ? AND intent = ?`,
    ).get(provider, model, intent) as { pinned: number } | undefined;
    return (row?.pinned ?? 0) === 1;
  }

  /** Count occurrences of a normalized judge note for a pair (canary input). */
  countJudgeNoteHash(provider: string, model: string, intent: string, hash: string): number {
    const row = this.db.prepare(`
      SELECT COUNT(*) AS c FROM judge_history
      WHERE provider = ? AND model = ? AND intent = ? AND note_hash = ?
    `).get(provider, model, intent, hash) as { c: number };
    return row.c;
  }

  /** Quarantine a model/intent pair (post-application canary fired). */
  quarantinePair(provider: string, model: string, intent: string, hash: string, noteText: string, occurrences: number): void {
    this.db.prepare(`
      INSERT INTO quarantined_pairs (provider, model, intent, note_hash, note_text, occurrences, quarantined_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(provider, model, intent, note_hash) DO UPDATE SET
        occurrences = MAX(occurrences, excluded.occurrences),
        quarantined_at = excluded.quarantined_at
    `).run(provider, model, intent, hash, noteText.slice(0, 255), occurrences, new Date().toISOString());
  }

  /** Is this pair quarantined (any note hash)? */
  isPairQuarantined(provider: string, model: string, intent: string): boolean {
    const row = this.db.prepare(
      `SELECT 1 AS x FROM quarantined_pairs WHERE provider = ? AND model = ? AND intent = ? LIMIT 1`,
    ).get(provider, model, intent);
    return row !== undefined;
  }

  getQuarantinedPairs(): Array<{ provider: string; model: string; intent: string; note_hash: string; note_text: string | null; occurrences: number; quarantined_at: string }> {
    return this.db.prepare(`
      SELECT provider, model, intent, note_hash, note_text, occurrences, quarantined_at
      FROM quarantined_pairs
    `).all() as any[];
  }

  /** Sum absolute capability movement inside a time window for a pair (RoC check). */
  sumCapabilityMovement(provider: string, model: string, intent: string, sinceIso: string): number {
    const row = this.db.prepare(`
      SELECT COALESCE(SUM(cumulative_delta), 0) AS s
      FROM capability_movement
      WHERE provider = ? AND model = ? AND intent = ? AND window_start_ts >= ?
    `).get(provider, model, intent, sinceIso) as { s: number };
    return row.s;
  }

  /** Record a capability movement increment for a pair inside a window bucket. */
  recordCapabilityMovement(provider: string, model: string, intent: string, delta: number, timestamp: string): void {
    const windowStart = timestamp.slice(0, 10);
    this.db.prepare(`
      INSERT INTO capability_movement (provider, model, intent, window_start_ts, cumulative_delta, last_updated)
      VALUES (?, ?, ?, ?, ?, ?)
      ON CONFLICT(provider, model, intent, window_start_ts) DO UPDATE SET
        cumulative_delta = cumulative_delta + excluded.cumulative_delta,
        last_updated = excluded.last_updated
    `).run(provider, model, intent, windowStart, delta, timestamp);
  }

  /** Archive every capability_overrides row with a batch timestamp. Returns rows archived. */
  archiveAllCapabilityOverrides(batchTs: string): number {
    const tx = this.db.transaction(() => {
      const rows = this.db.prepare(`
        SELECT provider, model, intent, score, sample_count, last_judged, pinned
        FROM capability_overrides
      `).all() as Array<{ provider: string; model: string; intent: string; score: number; sample_count: number; last_judged: string; pinned: number }>;
      const ins = this.db.prepare(`
        INSERT INTO capability_overrides_archive
          (provider, model, intent, score, sample_count, last_judged, pinned, archived_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      `);
      for (const r of rows) {
        ins.run(r.provider, r.model, r.intent, r.score, r.sample_count, r.last_judged, r.pinned ?? 0, batchTs);
      }
      return rows.length;
    });
    return tx();
  }

  upsertCapabilityOverride(
    provider: string,
    model: string,
    intent: string,
    score: number,
    sampleCount: number,
  ): void {
    this.db.prepare(`
      INSERT INTO capability_overrides (provider, model, intent, score, sample_count, last_judged)
      VALUES (?, ?, ?, ?, ?, ?)
      ON CONFLICT(provider, model, intent) DO UPDATE SET
        score = excluded.score,
        sample_count = excluded.sample_count,
        last_judged = excluded.last_judged
    `).run(provider, model, intent, score, sampleCount, new Date().toISOString());
  }

  loadCapabilityOverrides(): Array<{ provider: string; model: string; intent: string; score: number; sampleCount: number }> {
    return this.db.prepare(`
      SELECT provider, model, intent, score, sample_count AS sampleCount
      FROM capability_overrides
    `).all() as Array<{ provider: string; model: string; intent: string; score: number; sampleCount: number }>;
  }

  /** Full override rows incl. pin state + last_judged (decay + guardrail reads). */
  loadCapabilityOverrideRows(): Array<{ provider: string; model: string; intent: string; score: number; sampleCount: number; lastJudged: string; pinned: boolean }> {
    return this.db.prepare(`
      SELECT provider, model, intent, score,
             sample_count AS sampleCount, last_judged AS lastJudged,
             pinned
      FROM capability_overrides
    `).all() as any[];
  }

  recordJudgeEvaluation(
    provider: string,
    model: string,
    intent: string,
    judgeScore: number,
    judgeNote: string,
    judgeModel: string,
  ): void {
    this.db.prepare(`
      INSERT INTO judge_history (timestamp, provider, model, intent, judge_score, judge_note, judge_model)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).run(new Date().toISOString(), provider, model, intent, judgeScore, judgeNote, judgeModel);
  }

  /** Judge evaluation with attribution-gate metadata (§4.1). */
  recordJudgeEvaluationEx(data: {
    provider: string;
    model: string;
    intent: string;
    judgeScore: number;
    judgeNote: string;
    judgeModel: string;
    noApply: boolean;
    gateArm: string | null;
    gateReason: string | null;
    noteHash?: string | null;
    timestamp?: string;
    effortLevel?: string | null;
  }): void {
    this.db.prepare(`
      INSERT INTO judge_history
        (timestamp, provider, model, intent, judge_score, judge_note, judge_model,
         no_apply, gate_arm, gate_reason, note_hash, effort_level)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      data.timestamp ?? new Date().toISOString(),
      data.provider,
      data.model,
      data.intent,
      data.judgeScore,
      data.judgeNote,
      data.judgeModel,
      data.noApply ? 1 : 0,
      data.gateArm,
      data.gateReason,
      data.noteHash ?? (data.judgeNote ? noteHashOf(data.judgeNote) : null),
      data.effortLevel ?? null,
    );
  }

  /** Effort-conditioned judge quality over a window — answers "which model
   *  wins at effort=high". Legacy rows (pre-2026-09-05) report bucket
   *  "unrecorded". Buckets with <2 samples are dropped (noise). */
  getJudgeQualityByEffort(windowDays = 7): Array<{
    provider: string; model: string; intent: string; effortLevel: string;
    samples: number; applied: number; avgScore: number; avgAppliedScore: number | null;
  }> {
    const since = new Date(Date.now() - windowDays * 86400000).toISOString();
    const rows = this.db.prepare(`
      SELECT provider, model, intent,
             COALESCE(effort_level, 'unrecorded') AS effortLevel,
             COUNT(*) AS samples,
             SUM(CASE WHEN no_apply = 0 THEN 1 ELSE 0 END) AS applied,
             AVG(judge_score) AS avgScore,
             AVG(CASE WHEN no_apply = 0 THEN judge_score END) AS avgAppliedScore
      FROM judge_history
      WHERE timestamp >= ?
      GROUP BY provider, model, intent, COALESCE(effort_level, 'unrecorded')
      HAVING COUNT(*) >= 2
      ORDER BY samples DESC
      LIMIT 200
    `).all(since) as Array<{
      provider: string; model: string; intent: string; effortLevel: string;
      samples: number; applied: number; avgScore: number; avgAppliedScore: number | null;
    }>;
    return rows;
  }

  getCapabilityOverride(provider: string, model: string, intent: string): { score: number; sampleCount: number; pinned: boolean } | null {
    const row = this.db.prepare(`
      SELECT score, sample_count AS sampleCount, pinned
      FROM capability_overrides
      WHERE provider = ? AND model = ? AND intent = ?
    `).get(provider, model, intent) as { score: number; sampleCount: number; pinned: number } | null;
    if (!row) return null;
    return { score: row.score, sampleCount: row.sampleCount, pinned: row.pinned === 1 };
  }

  /** Persist a decayed score WITHOUT touching sample_count/last_judged (decay is not a judgment). */
  updateCapabilityScoreOnly(provider: string, model: string, intent: string, score: number): void {
    this.db.prepare(
      `UPDATE capability_overrides SET score = ? WHERE provider = ? AND model = ? AND intent = ?`,
    ).run(score, provider, model, intent);
  }

  /** Atomic guarded capability apply (§4.2): BEGIN IMMEDIATE; re-check pin;
   *  sum RoC windows inside the transaction; reject-and-log over Δmax;
   *  else record movement + update override. */
  applyGuardedCapabilityUpdate(
    provider: string,
    model: string,
    intent: string,
    newScore: number,
    deltaAbs: number,
    nowIso: string,
    rocMax24h: number,
    rocMax7d: number,
  ): { applied: boolean; reason: string; sampleCount: number } {
    const run = this.db.transaction((): { applied: boolean; reason: string; sampleCount: number } => {
      const row = this.db.prepare(
        `SELECT score, sample_count, pinned FROM capability_overrides
         WHERE provider = ? AND model = ? AND intent = ?`,
      ).get(provider, model, intent) as { score: number; sample_count: number; pinned: number } | undefined;
      if (row && row.pinned === 1) {
        return { applied: false, reason: "pinned", sampleCount: row.sample_count };
      }
      const sampleCount = (row?.sample_count ?? 0);

      const since24 = new Date(new Date(nowIso).getTime() - 86_400_000).toISOString();
      const since7d = new Date(new Date(nowIso).getTime() - 7 * 86_400_000).toISOString();
      const sum24 = this.sumCapabilityMovement(provider, model, intent, since24);
      const sum7d = this.sumCapabilityMovement(provider, model, intent, since7d);
      if (sum24 + deltaAbs > rocMax24h) {
        return { applied: false, reason: "roc_24h", sampleCount };
      }
      if (sum7d + deltaAbs > rocMax7d) {
        return { applied: false, reason: "roc_7d", sampleCount };
      }

      this.recordCapabilityMovement(provider, model, intent, deltaAbs, nowIso);
      this.upsertCapabilityOverride(provider, model, intent, newScore, sampleCount + 1);
      return { applied: true, reason: "applied", sampleCount: sampleCount + 1 };
    });
    return run.immediate();
  }

  // ─── Provider Spend Tracking ───

  /**
   * Add spend for a provider within a specific period.
   * Uses UPSERT to accumulate spend within the same period.
   */
  recordSpend(provider: string, amountUsd: number, period: "daily" | "monthly"): void {
    const dateKey = period === "daily"
      ? new Date().toISOString().slice(0, 10) // YYYY-MM-DD
      : new Date().toISOString().slice(0, 7);  // YYYY-MM

    this.db.prepare(`
      INSERT INTO provider_spend (date_key, period, provider, spend_usd)
      VALUES (?, ?, ?, ?)
      ON CONFLICT(date_key, period, provider) DO UPDATE SET
        spend_usd = spend_usd + ?
    `).run(dateKey, period, provider, amountUsd, amountUsd);
  }

  /** Get spend for a specific provider + period (default: current period). */
  getSpend(provider: string, period: "daily" | "monthly", dateKey?: string): number {
    const key = dateKey ?? (period === "daily"
      ? new Date().toISOString().slice(0, 10)
      : new Date().toISOString().slice(0, 7));

    const row = this.db.prepare(`
      SELECT spend_usd FROM provider_spend
      WHERE date_key = ? AND period = ? AND provider = ?
    `).get(key, period, provider) as { spend_usd: number } | undefined;

    return row?.spend_usd ?? 0;
  }

  /** Get all provider spend for a specific period. */
  getAllSpend(period: "daily" | "monthly", dateKey?: string): Array<{ provider: string; spendUsd: number }> {
    const key = dateKey ?? (period === "daily"
      ? new Date().toISOString().slice(0, 10)
      : new Date().toISOString().slice(0, 7));

    return this.db.prepare(`
      SELECT provider, spend_usd AS spendUsd
      FROM provider_spend
      WHERE date_key = ? AND period = ?
      ORDER BY spend_usd DESC
    `).all(key, period) as Array<{ provider: string; spendUsd: number }>;
  }

  // ─── Curator Methods ───

  /** Record a curator cycle result. */
  recordCuratorRun(result: import("./curator.js").CuratorResult): void {
    this.db.prepare(`
      INSERT INTO curator_runs
        (timestamp, ollama_scanned, ollama_pulled, ollama_skipped,
         openrouter_scanned, openrouter_added, openrouter_pruned, errors)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      result.timestamp,
      result.ollamaScanned,
      JSON.stringify(result.ollamaPulled),
      JSON.stringify(result.ollamaSkipped),
      result.openrouterScanned,
      JSON.stringify(result.openrouterAdded),
      JSON.stringify(result.openrouterPruned),
      JSON.stringify(result.errors),
    );
  }

  /** Get recent curator runs for diagnostics. */
  getRecentCuratorRuns(limit = 10): any[] {
    return this.db.prepare(`
      SELECT * FROM curator_runs ORDER BY timestamp DESC LIMIT ?
    `).all(limit);
  }

  /** Record a model pull attempt (success or failure). */
  recordCuratorModelAttempt(provider: string, model: string, success: boolean): void {
    const existing = this.getCuratorModelAttempt(provider, model);
    const failures = existing ? (success ? 0 : existing.consecutiveFailures + 1) : (success ? 0 : 1);

    this.db.prepare(`
      INSERT INTO curator_model_attempts
        (provider, model, last_attempt, consecutive_failures, last_success)
      VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(provider, model) DO UPDATE SET
        last_attempt = excluded.last_attempt,
        consecutive_failures = excluded.consecutive_failures,
        last_success = CASE WHEN ? = 1 THEN excluded.last_success ELSE last_success END
    `).run(
      provider,
      model,
      new Date().toISOString(),
      failures,
      success ? new Date().toISOString() : null,
      success ? 1 : 0,
    );
  }

  /** Get the last attempt record for a model. */
  getCuratorModelAttempt(provider: string, model: string): { lastAttempt: string; consecutiveFailures: number; lastSuccess: string | null } | null {
    const row = this.db.prepare(`
      SELECT last_attempt AS lastAttempt, consecutive_failures AS consecutiveFailures, last_success AS lastSuccess
      FROM curator_model_attempts
      WHERE provider = ? AND model = ?
    `).get(provider, model) as any;
    return row ?? null;
  }

  /** Increment the prune miss counter for a model. Returns the new count. */
  incrementPruneCounter(provider: string, model: string): number {
    this.db.prepare(`
      INSERT INTO curator_prune_tracking
        (provider, model, miss_count, first_miss, last_miss)
      VALUES (?, ?, 1, ?, ?)
      ON CONFLICT(provider, model) DO UPDATE SET
        miss_count = miss_count + 1,
        last_miss = excluded.last_miss
    `).run(
      provider,
      model,
      new Date().toISOString(),
      new Date().toISOString(),
    );

    const row = this.db.prepare(`
      SELECT miss_count FROM curator_prune_tracking WHERE provider = ? AND model = ?
    `).get(provider, model) as { miss_count: number } | undefined;

    return row?.miss_count ?? 1;
  }

  /** Get the current prune miss counter for a model. */
  getPruneCounter(provider: string, model: string): number {
    const row = this.db.prepare(`
      SELECT miss_count FROM curator_prune_tracking WHERE provider = ? AND model = ?
    `).get(provider, model) as { miss_count: number } | undefined;
    return row?.miss_count ?? 0;
  }

  /** Reset the prune counter for a model (it was found alive). */
  resetPruneCounter(provider: string, model: string): void {
    this.db.prepare(`
      DELETE FROM curator_prune_tracking WHERE provider = ? AND model = ?
    `).run(provider, model);
  }

  // ─── Chat Benchmark Methods ───

  /** Save a chat benchmark result for a specific model + probe type. */
  saveChatBenchmarkResult(data: {
    modelId: string;
    provider: string;
    model: string;
    probeType: string;
    scoresJson: string;
    latencyMs: number;
    modelVersionHash: string;
  }): void {
    this.db.prepare(`
      INSERT INTO chat_benchmark_results
        (model_id, provider, model, probe_type, scores_json, latency_ms, model_version_hash, timestamp)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      data.modelId,
      data.provider,
      data.model,
      data.probeType,
      data.scoresJson,
      data.latencyMs,
      data.modelVersionHash,
      new Date().toISOString(),
    );
  }

  /** Get the most recent benchmark result for a model + probe type.
   *  Returns null if not found. */
  getLatestChatBenchmark(
    modelId: string,
    probeType: string,
  ): { scoresJson: string; latencyMs: number; modelVersionHash: string; timestamp: string } | null {
    const row = this.db.prepare(`
      SELECT scores_json AS scoresJson, latency_ms AS latencyMs,
             model_version_hash AS modelVersionHash, timestamp
      FROM chat_benchmark_results
      WHERE model_id = ? AND probe_type = ?
      ORDER BY timestamp DESC LIMIT 1
    `).get(modelId, probeType) as any;
    return row ?? null;
  }

  /** Get all latest benchmark results for a model (across all probe types).
   *  Returns a map of probeType -> result. */
  getAllLatestChatBenchmarks(
    modelId: string,
  ): Map<string, { scoresJson: string; latencyMs: number; modelVersionHash: string; timestamp: string }> {
    const rows = this.db.prepare(`
      SELECT probe_type AS probeType, scores_json AS scoresJson,
             latency_ms AS latencyMs, model_version_hash AS modelVersionHash, timestamp
      FROM chat_benchmark_results
      WHERE model_id = ?
      AND id IN (
        SELECT MAX(id) FROM chat_benchmark_results
        WHERE model_id = ?
        GROUP BY probe_type
      )
    `).all(modelId, modelId) as Array<any>;

    const result = new Map();
    for (const row of rows) {
      result.set(row.probeType, {
        scoresJson: row.scoresJson,
        latencyMs: row.latencyMs,
        modelVersionHash: row.modelVersionHash,
        timestamp: row.timestamp,
      });
    }
    return result;
  }

  /** Get the intent distribution from recent routing decisions.
   *  Returns a map of intent -> proportion (0-1). */
  getTrafficDistribution(days: number = 7): Map<string, number> {
    const since = new Date(Date.now() - days * 86_400_000).toISOString();
    const rows = this.db.prepare(`
      SELECT intent, COUNT(*) AS count
      FROM routing_decisions
      WHERE timestamp >= ?
      GROUP BY intent
    `).all(since) as Array<{ intent: string; count: number }>;

    const total = rows.reduce((sum, r) => sum + r.count, 0);
    const result = new Map<string, number>();
    for (const row of rows) {
      result.set(row.intent, total > 0 ? row.count / total : 0);
    }
    return result;
  }

  // ─── Dashboard Aggregation Queries ───

  /** Parse a time range string (e.g. "1h", "24h", "7d") into an ISO cutoff timestamp. */
  static parseTimeRange(range: string): string {
    const match = range.match(/^(\d+)([hdw])$/);
    if (!match) return new Date(Date.now() - 24 * 3_600_000).toISOString(); // default 24h
    const value = parseInt(match[1], 10);
    const unit = match[2];
    const multiplier = unit === "h" ? 3_600_000 : unit === "d" ? 86_400_000 : 7 * 86_400_000;
    return new Date(Date.now() - value * multiplier).toISOString();
  }

  /** Get provider distribution: request count, percentage, success/failure counts. */
  getDashboardProviderDistribution(sinceIso: string): Array<{
    provider: string; requests: number; percentage: number;
    successes: number; failures: number; successRate: number;
  }> {
    const rows = this.db.prepare(`
      SELECT
        chosen_provider AS provider,
        COUNT(*) AS requests,
        SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) AS successes,
        SUM(CASE WHEN outcome NOT IN ('success', 'PENDING') THEN 1 ELSE 0 END) AS failures
      FROM routing_decisions
      WHERE timestamp >= ?
      GROUP BY chosen_provider
      ORDER BY requests DESC
    `).all(sinceIso) as Array<{ provider: string; requests: number; successes: number; failures: number }>;

    const total = rows.reduce((sum, r) => sum + r.requests, 0);
    return rows.map(r => ({
      provider: r.provider,
      requests: r.requests,
      percentage: total > 0 ? Math.round((r.requests / total) * 10000) / 100 : 0,
      successes: r.successes,
      failures: r.failures,
      successRate: r.requests > 0 ? Math.round((r.successes / r.requests) * 10000) / 100 : 0,
    }));
  }

  /** Get cost breakdown by intent classification.
   *  Joins routing_decisions with provider_spend to estimate per-intent spend.
   *  When spend data isn't available, falls back to decision counts per intent. */
  getDashboardCostByIntent(sinceIso: string): Array<{
    intent: string; decisions: number; estimatedSpendUsd: number;
  }> {
    // Get decision counts per intent
    const decisionRows = this.db.prepare(`
      SELECT intent, COUNT(*) AS decisions
      FROM routing_decisions
      WHERE timestamp >= ?
      GROUP BY intent
      ORDER BY decisions DESC
    `).all(sinceIso) as Array<{ intent: string; decisions: number }>;

    // Get spend per provider for the period (from provider_spend)
    const sinceDateKey = sinceIso.slice(0, 10);
    const spendRows = this.db.prepare(`
      SELECT provider, SUM(spend_usd) AS total_spend
      FROM provider_spend
      WHERE period = 'daily' AND date_key >= ?
      GROUP BY provider
    `).all(sinceDateKey) as Array<{ provider: string; total_spend: number }>;

    // Get provider distribution per intent to allocate spend proportionally
    const providerPerIntent = this.db.prepare(`
      SELECT
        intent,
        chosen_provider AS provider,
        COUNT(*) AS cnt
      FROM routing_decisions
      WHERE timestamp >= ?
      GROUP BY intent, chosen_provider
    `).all(sinceIso) as Array<{ intent: string; provider: string; cnt: number }>;

    // Build provider→spend map
    const providerSpend = new Map<string, number>();
    for (const s of spendRows) {
      providerSpend.set(s.provider, s.total_spend);
    }

    // Build intent→provider count map
    const intentProviderCounts = new Map<string, Map<string, number>>();
    const intentTotals = new Map<string, number>();
    for (const r of providerPerIntent) {
      if (!intentProviderCounts.has(r.intent)) intentProviderCounts.set(r.intent, new Map());
      intentProviderCounts.get(r.intent)!.set(r.provider, r.cnt);
      intentTotals.set(r.intent, (intentTotals.get(r.intent) ?? 0) + r.cnt);
    }

    return decisionRows.map(r => {
      let estimatedSpend = 0;
      const providerCounts = intentProviderCounts.get(r.intent);
      const totalForIntent = intentTotals.get(r.intent) ?? 1;
      if (providerCounts) {
        for (const [provider, cnt] of providerCounts) {
          const spend = providerSpend.get(provider) ?? 0;
          estimatedSpend += spend * (cnt / totalForIntent);
        }
      }
      return {
        intent: r.intent,
        decisions: r.decisions,
        estimatedSpendUsd: Math.round(estimatedSpend * 10000) / 10000,
      };
    });
  }

  /** Get rolling average latency per provider for multiple bucket sizes.
   *  Returns nested structure: provider → bucket → { avgMs, samples }. */
  getDashboardLatencyTrends(sinceIso: string): Array<{
    provider: string;
    buckets: Array<{ bucket: string; avgMs: number | null; samples: number }>;
  }> {
    const providers = this.db.prepare(`
      SELECT DISTINCT provider FROM call_outcomes WHERE timestamp >= ?
    `).all(sinceIso) as Array<{ provider: string }>;

    const bucketDefs = [
      { name: "1h", ms: 3_600_000 },
      { name: "6h", ms: 6 * 3_600_000 },
      { name: "24h", ms: 24 * 3_600_000 },
      { name: "7d", ms: 7 * 24 * 3_600_000 },
    ];

    const now = Date.now();

    return providers.map(({ provider }) => {
      const buckets = bucketDefs.map(bd => {
        const bucketSince = new Date(now - bd.ms).toISOString();
        const row = this.db.prepare(`
          SELECT
            AVG(duration_ms) AS avg_ms,
            COUNT(*) AS samples
          FROM call_outcomes
          WHERE provider = ? AND timestamp >= ? AND outcome = 'success'
        `).get(provider, bucketSince) as { avg_ms: number | null; samples: number } | undefined;

        return {
          bucket: bd.name,
          avgMs: row?.avg_ms != null ? Math.round(row.avg_ms) : null,
          samples: row?.samples ?? 0,
        };
      });

      return { provider, buckets };
    });
  }

  /** Get model market share: which specific models are picked and how that changes over time.
   *  Returns total counts plus time-series buckets for trend visualization. */
  getDashboardModelMarketShare(sinceIso: string, rangeLabel: string): Array<{
    provider: string; model: string; requests: number; percentage: number;
    timeseries: Array<{ bucket: string; count: number }>;
  }> {
    // Determine bucket size based on range
    const bucketFmt = rangeLabel.endsWith("h") ? "%Y-%m-%dT%H:00:00" : "%Y-%m-%d";

    const rows = this.db.prepare(`
      SELECT
        chosen_provider AS provider,
        chosen_model AS model,
        COUNT(*) AS requests
      FROM routing_decisions
      WHERE timestamp >= ?
      GROUP BY chosen_provider, chosen_model
      ORDER BY requests DESC
    `).all(sinceIso) as Array<{ provider: string; model: string; requests: number }>;

    const total = rows.reduce((sum, r) => sum + r.requests, 0);

    // Get timeseries for each model
    const tsRows = this.db.prepare(`
      SELECT
        chosen_provider AS provider,
        chosen_model AS model,
        strftime('${bucketFmt}', timestamp) AS bucket,
        COUNT(*) AS count
      FROM routing_decisions
      WHERE timestamp >= ?
      GROUP BY chosen_provider, chosen_model, bucket
      ORDER BY bucket ASC
    `).all(sinceIso) as Array<{ provider: string; model: string; bucket: string; count: number }>;

    // Group timeseries by provider/model
    const tsMap = new Map<string, Array<{ bucket: string; count: number }>>();
    for (const t of tsRows) {
      const key = `${t.provider}/${t.model}`;
      if (!tsMap.has(key)) tsMap.set(key, []);
      tsMap.get(key)!.push({ bucket: t.bucket, count: t.count });
    }

    return rows.map(r => {
      const key = `${r.provider}/${r.model}`;
      return {
        provider: r.provider,
        model: r.model,
        requests: r.requests,
        percentage: total > 0 ? Math.round((r.requests / total) * 10000) / 100 : 0,
        timeseries: tsMap.get(key) ?? [],
      };
    });
  }

  /** Get success/failure rates per provider and model.
   *  Uses call_outcomes for detailed outcome breakdown. */
  getDashboardModelSuccessRates(sinceIso: string): Array<{
    provider: string; model: string; total: number;
    successes: number; failures: number; successRate: number;
    avgLatencyMs: number | null;
  }> {
    const rows = this.db.prepare(`
      SELECT
        provider,
        model,
        COUNT(*) AS total,
        SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) AS successes,
        SUM(CASE WHEN outcome != 'success' THEN 1 ELSE 0 END) AS failures,
        AVG(CASE WHEN outcome = 'success' THEN duration_ms END) AS avg_latency_ms
      FROM call_outcomes
      WHERE timestamp >= ?
      GROUP BY provider, model
      ORDER BY total DESC
    `).all(sinceIso) as Array<{
      provider: string; model: string; total: number;
      successes: number; failures: number; avg_latency_ms: number | null;
    }>;

    return rows.map(r => ({
      provider: r.provider,
      model: r.model,
      total: r.total,
      successes: r.successes,
      failures: r.failures,
      successRate: r.total > 0 ? Math.round((r.successes / r.total) * 10000) / 100 : 0,
      avgLatencyMs: r.avg_latency_ms != null ? Math.round(r.avg_latency_ms) : null,
    }));
  }

  /** Get outcome distribution: success/timeout/error/fallback percentages. */
  getDashboardOutcomeDistribution(sinceIso: string): Array<{
    outcome: string; count: number; percentage: number;
  }> {
    const rows = this.db.prepare(`
      SELECT outcome, COUNT(*) AS count
      FROM routing_decisions
      WHERE timestamp >= ?
      GROUP BY outcome
      ORDER BY count DESC
    `).all(sinceIso) as Array<{ outcome: string; count: number }>;

    const total = rows.reduce((sum, r) => sum + r.count, 0);
    return rows.map(r => ({
      outcome: r.outcome,
      count: r.count,
      percentage: total > 0 ? Math.round((r.count / total) * 10000) / 100 : 0,
    }));
  }

  /** Get total spend for the dashboard period. */
  getDashboardTotalSpend(sinceIso: string): number {
    const sinceDateKey = sinceIso.slice(0, 10);
    const row = this.db.prepare(`
      SELECT COALESCE(SUM(spend_usd), 0) AS total
      FROM provider_spend
      WHERE period = 'daily' AND date_key >= ?
    `).get(sinceDateKey) as { total: number } | undefined;
    return row?.total ?? 0;
  }

  /** Build the complete dashboard payload for a given time range. */
  getDashboardData(rangeLabel: string = "24h"): Record<string, unknown> {
    const sinceIso = DBService.parseTimeRange(rangeLabel);

    return {
      timeRange: {
        label: rangeLabel,
        since: sinceIso,
        generatedAt: new Date().toISOString(),
      },
      summary: {
        totalRequests: this.db.prepare(`
          SELECT COUNT(*) AS cnt FROM routing_decisions WHERE timestamp >= ?
        `).get(sinceIso) as { cnt: number },
        totalOutcomes: (() => {
          const row = this.db.prepare(`
            SELECT COUNT(*) AS cnt FROM call_outcomes WHERE timestamp >= ?
          `).get(sinceIso) as { cnt: number } | undefined;
          return row?.cnt ?? 0;
        })(),
        totalSpendUsd: this.getDashboardTotalSpend(sinceIso),
        activeProviders: (this.db.prepare(`
          SELECT COUNT(DISTINCT chosen_provider) AS cnt FROM routing_decisions WHERE timestamp >= ?
        `).get(sinceIso) as { cnt: number }).cnt,
        activeModels: (this.db.prepare(`
          SELECT COUNT(DISTINCT chosen_model) AS cnt FROM routing_decisions WHERE timestamp >= ?
        `).get(sinceIso) as { cnt: number }).cnt,
      },
      providerDistribution: this.getDashboardProviderDistribution(sinceIso),
      costByIntent: this.getDashboardCostByIntent(sinceIso),
      latencyTrends: this.getDashboardLatencyTrends(sinceIso),
      modelMarketShare: this.getDashboardModelMarketShare(sinceIso, rangeLabel),
      modelSuccessRates: this.getDashboardModelSuccessRates(sinceIso),
      outcomeDistribution: this.getDashboardOutcomeDistribution(sinceIso),
    };
  }

  // ─── Budget Burn-Rate Queries ───

  /** Get hourly spend for the last N hours, grouped by hour.
   *  Returns array of { hour_bucket, provider, spend_usd } sorted oldest-first. */
  getHourlySpend(hours: number = 24): Array<{ hourBucket: string; provider: string; spendUsd: number }> {
    const since = new Date(Date.now() - hours * 3_600_000).toISOString().slice(0, 13) + "00:00:00";
    const rows = this.db.prepare(`
      SELECT
        SUBSTR(date_key, 1, 13) || ':00:00' AS hour_bucket,
        provider,
        SUM(spend_usd) AS spend_usd
      FROM provider_spend
      WHERE period = 'daily'
        AND date_key >= ?
      GROUP BY hour_bucket, provider
      ORDER BY hour_bucket ASC, provider ASC
    `).all(since) as Array<{ hour_bucket: string; provider: string; spend_usd: number }>;

    return rows.map((r) => ({
      hourBucket: r.hour_bucket,
      provider: r.provider,
      spendUsd: r.spend_usd,
    }));
  }

  /** Get total spend across all providers for a specific period. */
  getTotalSpend(period: "daily" | "monthly", dateKey?: string): number {
    const key = dateKey ?? (period === "daily"
      ? new Date().toISOString().slice(0, 10)
      : new Date().toISOString().slice(0, 7));

    const row = this.db.prepare(`
      SELECT COALESCE(SUM(spend_usd), 0) AS total
      FROM provider_spend
      WHERE period = ? AND date_key = ?
    `).get(period, key) as { total: number } | undefined;

    return row?.total ?? 0;
  }

  /** Get recent request token counts for anomaly detection.
   *  Returns estimated token counts derived from recent routing decisions.
   *  Uses context_filter_json when available, otherwise falls back to
   *  spend-per-request proxy from the provider_spend table. */
  getRecentTokenCounts(limit: number = 50): number[] {
    // Try to extract estimated token counts from recent routing decisions
    // that have context_filter_json populated (contains estimatedTokens)
    const rows = this.db.prepare(`
      SELECT context_filter_json
      FROM routing_decisions
      WHERE context_filter_json IS NOT NULL
      ORDER BY timestamp DESC
      LIMIT ?
    `).all(limit) as Array<{ context_filter_json: string }>;

    if (rows.length > 0) {
      const tokens: number[] = [];
      for (const row of rows) {
        try {
          const parsed = JSON.parse(row.context_filter_json);
          if (parsed?.estimatedTokens && typeof parsed.estimatedTokens === 'number') {
            tokens.push(parsed.estimatedTokens);
          }
        } catch {
          // skip unparseable rows
        }
      }
      if (tokens.length > 0) return tokens;
    }

    // Fallback: estimate from recent call outcomes count as a proxy
    // (at least gives us *something* for anomaly detection baseline)
    const outcomeCount = this.db.prepare(`
      SELECT COUNT(*) AS cnt FROM call_outcomes
      WHERE timestamp >= datetime('now', '-1 hour')
    `).get() as { cnt: number } | undefined;

    if (outcomeCount && outcomeCount.cnt > 0) {
      // Return a synthetic baseline based on request frequency
      // This is a rough proxy; real token counts come from context_filter_json
      return Array(Math.min(outcomeCount.cnt, limit)).fill(1000);
    }

    return [];
  }

  close(): void {
    this.db.close();
    logger.info("Database connection closed.");
  }

  /** Store one raw benchmark ask (uncollapsed; 2 rows per round). */
  insertBenchmarkVerdict(data: {
    modelA: string;
    modelB: string;
    intent: string;
    promptGeneration: string;
    round: number;
    swapOrder: 0 | 1;
    verdict: "a" | "b" | "tie";
    judgeProvider?: string;
    judgeModel?: string;
  }): void {
    this.db.prepare(`
      INSERT INTO benchmark_verdicts
        (timestamp, model_a, model_b, intent, prompt_generation,
         round, swap_order, verdict, judge_provider, judge_model)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      new Date().toISOString(),
      data.modelA,
      data.modelB,
      data.intent,
      data.promptGeneration,
      data.round,
      data.swapOrder,
      data.verdict,
      data.judgeProvider ?? null,
      data.judgeModel ?? null,
    );
  }

  /** Raw ask rows for one pair (either stored direction), normalized to the
   *  (modelA=modelA arg, modelB=modelB arg) perspective: rows stored reversed
   *  come back with models swapped and the verdict flipped. */
  getBenchmarkVerdicts(
    modelA: string,
    modelB: string,
    intent: string,
    promptGeneration: string,
  ): Array<{
    modelA: string;
    modelB: string;
    round: number;
    swapOrder: 0 | 1;
    verdict: "a" | "b" | "tie";
  }> {
    const rows = this.db.prepare(`
      SELECT model_a, model_b, round, swap_order, verdict
      FROM benchmark_verdicts
      WHERE intent = ? AND prompt_generation = ?
        AND ((model_a = ? AND model_b = ?) OR (model_a = ? AND model_b = ?))
    `).all(intent, promptGeneration, modelA, modelB, modelB, modelA) as any[];
    return rows.map((r) =>
      r.model_a === modelA
        ? {
            modelA: r.model_a,
            modelB: r.model_b,
            round: r.round,
            swapOrder: r.swap_order as 0 | 1,
            verdict: r.verdict,
          }
 : {
            modelA: r.model_b,
            modelB: r.model_a,
            round: r.round,
            swapOrder: r.swap_order as 0 | 1,
            verdict: r.verdict === "a" ? "b" : r.verdict === "b" ? "a" : "tie",
          },
    );
  }

  /** All raw ask rows for an intent + generation, NOT normalized. */
  getAllBenchmarkVerdicts(
    intent: string,
    promptGeneration: string,
  ): Array<{
    modelA: string;
    modelB: string;
    round: number;
    swapOrder: 0 | 1;
    verdict: "a" | "b" | "tie";
  }> {
    const rows = this.db.prepare(`
      SELECT model_a AS modelA, model_b AS modelB, round,
             swap_order AS swapOrder, verdict
      FROM benchmark_verdicts
      WHERE intent = ? AND prompt_generation = ?
    `).all(intent, promptGeneration) as any[];
    return rows;
  }

  /** Delete all verdict rows for a pair (either stored direction).
   *  Used by comparePair's failure path: a pair that died mid-comparison
   *  must not leave partial rounds behind — a later retry would collapse
   *  them into a majority computed over fewer, order-biased rounds. */
  deleteBenchmarkVerdicts(
    modelA: string,
    modelB: string,
    intent: string,
    promptGeneration: string,
  ): void {
    this.db.prepare(`
      DELETE FROM benchmark_verdicts
      WHERE intent = ? AND prompt_generation = ?
        AND ((model_a = ? AND model_b = ?) OR (model_a = ? AND model_b = ?))
    `).run(intent, promptGeneration, modelA, modelB, modelB, modelA);
  }

  /** Distinct model keys known to the ladder for an intent + generation. */
  getLadderKeys(intent: string, promptGeneration: string): string[] {
    const rows = this.db.prepare(`
      SELECT model_key AS modelKey
      FROM benchmark_ladder
      WHERE intent = ? AND prompt_generation = ?
      ORDER BY rank ASC
    `).all(intent, promptGeneration) as any[];
    return rows.map((r) => r.modelKey);
  }

  /** Replace the stored ladder for an intent + generation (transactional). */
  replaceLadder(
    intent: string,
    promptGeneration: string,
    entries: Array<{ modelKey: string; rank: number; strength: number }>,
  ): void {
    const tx = this.db.transaction(() => {
      this.db.prepare(`
        DELETE FROM benchmark_ladder
        WHERE intent = ? AND prompt_generation = ?
      `).run(intent, promptGeneration);
      const ins = this.db.prepare(`
        INSERT INTO benchmark_ladder
          (intent, prompt_generation, model_key, rank, strength, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
      `);
      const now = new Date().toISOString();
      for (const e of entries) {
        ins.run(intent, promptGeneration, e.modelKey, e.rank, e.strength, now);
      }
    });
    tx();
  }

  /** Record a remote model with unknown pricing (the actionable gap list).
   *  Upserted at discovery time; resolved (deleted) when pricing arrives. */
  upsertPricingGap(provider: string, model: string): void {
    const now = new Date().toISOString();
    this.db.prepare(`
      INSERT INTO pricing_gaps (provider, model, first_seen, last_seen)
      VALUES (?, ?, ?, ?)
      ON CONFLICT(provider, model) DO UPDATE SET last_seen = excluded.last_seen
    `).run(provider, model, now, now);
  }

  /** Remove a gap once pricing is known. */
  resolvePricingGap(provider: string, model: string): void {
    this.db.prepare(`
      DELETE FROM pricing_gaps WHERE provider = ? AND model = ?
    `).run(provider, model);
  }

  /** All open pricing gaps (quarantined models). */
  getPricingGaps(): Array<{ provider: string; model: string; firstSeen: string; lastSeen: string }> {
    return this.db.prepare(`
      SELECT provider, model, first_seen AS firstSeen, last_seen AS lastSeen
      FROM pricing_gaps ORDER BY provider, model
    `).all() as any[];
  }

  /** Replace the cached pricing catalog (OpenClaw model catalog snapshot).
   *  Transactional full swap; generatedAt records upstream freshness. */
  replaceCatalogPricing(
    entries: Array<{ key: string; inputPerM: number; outputPerM: number; cacheReadPerM?: number }>,
    generatedAt: string,
  ): void {
    const tx = this.db.transaction(() => {
      this.db.prepare(`DELETE FROM catalog_pricing`).run();
      const ins = this.db.prepare(`
        INSERT INTO catalog_pricing (key, input_per_m, output_per_m, cache_read_per_m, generated_at)
        VALUES (?, ?, ?, ?, ?)
      `);
      for (const e of entries) {
        ins.run(e.key, e.inputPerM, e.outputPerM, e.cacheReadPerM ?? null, generatedAt);
      }
    });
    tx();
  }

  /** Cached catalog pricing as a lookup map. */
  getCatalogPricingMap(): Map<string, { inputPerM: number; outputPerM: number }> {
    const rows = this.db.prepare(`
      SELECT key, input_per_m AS inputPerM, output_per_m AS outputPerM
      FROM catalog_pricing
    `).all() as any[];
    const map = new Map<string, { inputPerM: number; outputPerM: number }>();
    for (const r of rows) map.set(r.key, { inputPerM: r.inputPerM, outputPerM: r.outputPerM });
    return map;
  }
}

// ═══════════════════════════════════════════════════════════
//  Schema
// ═══════════════════════════════════════════════════════════

const TABLE_SCHEMA_SQL = `
CREATE TABLE IF NOT EXISTS routing_decisions (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp       TEXT    NOT NULL,
  session_key     TEXT    NOT NULL,
  message_hash    TEXT,
  intent          TEXT    NOT NULL,
  confidence      REAL    NOT NULL,
  chosen_provider TEXT    NOT NULL,
  chosen_model    TEXT    NOT NULL,
  routing_scores  TEXT,
  overall_score   REAL,
  outcome         TEXT    DEFAULT 'PENDING',
  request_id      TEXT
);

CREATE TABLE IF NOT EXISTS retry_attempts (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp   TEXT    NOT NULL,
  request_id  TEXT    NOT NULL,
  failed_provider TEXT NOT NULL,
  failed_outcome TEXT NOT NULL,
  retry_provider TEXT NOT NULL,
  retry_count INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS call_outcomes (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp   TEXT    NOT NULL,
  provider    TEXT    NOT NULL,
  model       TEXT    NOT NULL,
  duration_ms INTEGER,
  outcome     TEXT
);

CREATE TABLE IF NOT EXISTS model_stats (
  provider_model   TEXT PRIMARY KEY,
  total_calls      INTEGER NOT NULL DEFAULT 0,
  success_count    INTEGER NOT NULL DEFAULT 0,
  failure_count    INTEGER NOT NULL DEFAULT 0,
  total_latency_ms INTEGER NOT NULL DEFAULT 0,
  last_updated     TEXT
);

CREATE TABLE IF NOT EXISTS provider_health (
  provider_name        TEXT PRIMARY KEY,
  status               TEXT DEFAULT 'HEALTHY',
  rate_limit_errors    INTEGER DEFAULT 0,
  circuit_open         INTEGER DEFAULT 0,
  last_check           TEXT,
  monthly_spend_usd    REAL DEFAULT 0,
  metadata             TEXT
);

CREATE TABLE IF NOT EXISTS benchmark_cache (
  model_id    TEXT PRIMARY KEY,
  benchmark   TEXT,
  score       REAL,
  context     TEXT,
  fetched_at  TEXT
);

CREATE TABLE IF NOT EXISTS capability_overrides (
  provider     TEXT NOT NULL,
  model        TEXT NOT NULL,
  intent       TEXT NOT NULL,
  score        REAL NOT NULL,
  sample_count INTEGER NOT NULL DEFAULT 1,
  last_judged  TEXT NOT NULL,
  PRIMARY KEY (provider, model, intent)
);

CREATE TABLE IF NOT EXISTS judge_history (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp    TEXT NOT NULL,
  provider     TEXT NOT NULL,
  model        TEXT NOT NULL,
  intent       TEXT NOT NULL,
  judge_score  REAL NOT NULL,
  judge_note   TEXT,
  judge_model  TEXT
);

CREATE TABLE IF NOT EXISTS provider_spend (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  date_key    TEXT NOT NULL,
  period      TEXT NOT NULL,
  provider    TEXT NOT NULL,
  spend_usd   REAL NOT NULL DEFAULT 0,
  UNIQUE(date_key, period, provider)
);

CREATE TABLE IF NOT EXISTS curator_runs (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp           TEXT NOT NULL,
  ollama_scanned      INTEGER DEFAULT 0,
  ollama_pulled       TEXT,
  ollama_skipped      TEXT,
  openrouter_scanned  INTEGER DEFAULT 0,
  openrouter_added    TEXT,
  openrouter_pruned   TEXT,
  errors              TEXT
);

CREATE TABLE IF NOT EXISTS curator_model_attempts (
  provider            TEXT NOT NULL,
  model               TEXT NOT NULL,
  last_attempt        TEXT NOT NULL,
  consecutive_failures INTEGER NOT NULL DEFAULT 0,
  last_success        TEXT,
  PRIMARY KEY (provider, model)
);

CREATE TABLE IF NOT EXISTS curator_prune_tracking (
  provider    TEXT NOT NULL,
  model       TEXT NOT NULL,
  miss_count  INTEGER NOT NULL DEFAULT 0,
  first_miss  TEXT NOT NULL,
  last_miss   TEXT NOT NULL,
  PRIMARY KEY (provider, model)
);

CREATE TABLE IF NOT EXISTS chat_benchmark_results (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id        TEXT NOT NULL,
  provider        TEXT NOT NULL,
  model           TEXT NOT NULL,
  probe_type      TEXT NOT NULL,
  scores_json     TEXT NOT NULL,
  latency_ms      INTEGER,
  model_version_hash TEXT NOT NULL,
  timestamp       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmark_verdicts (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp         TEXT NOT NULL,
  model_a           TEXT NOT NULL,
  model_b           TEXT NOT NULL,
  intent            TEXT NOT NULL,
  prompt_generation TEXT NOT NULL,
  round             INTEGER NOT NULL,
  swap_order        INTEGER NOT NULL,
  verdict           TEXT NOT NULL,
  judge_provider    TEXT,
  judge_model       TEXT
);

CREATE TABLE IF NOT EXISTS benchmark_ladder (
  intent            TEXT NOT NULL,
  prompt_generation TEXT NOT NULL,
  model_key         TEXT NOT NULL,
  rank              INTEGER NOT NULL,
  strength          REAL NOT NULL,
  updated_at        TEXT NOT NULL,
  PRIMARY KEY (intent, prompt_generation, model_key)
);

CREATE TABLE IF NOT EXISTS pricing_gaps (
  provider   TEXT NOT NULL,
  model      TEXT NOT NULL,
  first_seen TEXT NOT NULL,
  last_seen  TEXT NOT NULL,
  PRIMARY KEY (provider, model)
);

CREATE TABLE IF NOT EXISTS catalog_pricing (
  key              TEXT PRIMARY KEY,
  input_per_m      REAL NOT NULL,
  output_per_m     REAL NOT NULL,
  cache_read_per_m REAL,
  generated_at     TEXT NOT NULL
);
`;

const INDEX_SCHEMA_SQL = `
CREATE INDEX IF NOT EXISTS idx_decisions_session ON routing_decisions(session_key);
CREATE INDEX IF NOT EXISTS idx_decisions_intent ON routing_decisions(intent);
CREATE INDEX IF NOT EXISTS idx_decisions_provider ON routing_decisions(chosen_provider);
CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON routing_decisions(timestamp);
CREATE INDEX IF NOT EXISTS idx_decisions_request_id ON routing_decisions(request_id);
CREATE INDEX IF NOT EXISTS idx_retry_request_id ON retry_attempts(request_id);
CREATE INDEX IF NOT EXISTS idx_retry_timestamp ON retry_attempts(timestamp);
CREATE INDEX IF NOT EXISTS idx_outcomes_provider ON call_outcomes(provider);
CREATE INDEX IF NOT EXISTS idx_outcomes_timestamp ON call_outcomes(timestamp);
CREATE INDEX IF NOT EXISTS idx_spend_lookup ON provider_spend(period, provider, date_key);
CREATE INDEX IF NOT EXISTS idx_curator_runs_timestamp ON curator_runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_curator_attempts_provider ON curator_model_attempts(provider);
CREATE INDEX IF NOT EXISTS idx_curator_prune_provider ON curator_prune_tracking(provider);
CREATE INDEX IF NOT EXISTS idx_chat_bench_model ON chat_benchmark_results(model_id);
CREATE INDEX IF NOT EXISTS idx_chat_bench_probe ON chat_benchmark_results(probe_type);
CREATE INDEX IF NOT EXISTS idx_chat_bench_timestamp ON chat_benchmark_results(timestamp);
CREATE INDEX IF NOT EXISTS idx_bench_verdicts_pair ON benchmark_verdicts(model_a, model_b, intent, prompt_generation);
`;
