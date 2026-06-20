// src/db_service.ts — SQLite persistence layer
// All database operations encapsulated here.

import Database from "better-sqlite3";
import { logger } from "./logger.js";

export interface DecisionRecord {
  timestamp: string;
  sessionKey: string;
  messageHash: string;
  intent: string;
  confidence: number;
  provider: string;
  model: string;
  scores: Record<string, number>;
  overallScore: number;
  outcome: string;
  requestId?: string;
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

  async initializeSchema(): Promise<void> {
    try {
      this.db.exec(TABLE_SCHEMA_SQL);
    } catch (err) {
      logger.warn(`Schema initialization warning: ${err}`);
    }
    this.addColumnIfMissing("routing_decisions", "request_id", "TEXT");
    this.db.exec(INDEX_SCHEMA_SQL);
    logger.info("Database schema verified.");
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
         chosen_provider, chosen_model, routing_scores, overall_score, outcome, request_id)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    );
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
        "SELECT * FROM routing_decisions ORDER BY timestamp DESC LIMIT ?",
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

  getCapabilityOverride(provider: string, model: string, intent: string): { score: number; sampleCount: number } | null {
    const row = this.db.prepare(`
      SELECT score, sample_count AS sampleCount
      FROM capability_overrides
      WHERE provider = ? AND model = ? AND intent = ?
    `).get(provider, model, intent) as { score: number; sampleCount: number } | null;
    return row;
  }

  close(): void {
    this.db.close();
    logger.info("Database connection closed.");
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
`;
