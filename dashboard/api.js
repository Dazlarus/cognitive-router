// api.js - Cognitive Router Dashboard API
// Simple Express server to query SQLite and provide JSON data

const express = require('express');
const cors = require('cors');
const Database = require('../node_modules/better-sqlite3');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());

// SQLite database path
const DB_PATH = path.join(__dirname, '..', 'data', 'cognitive-router.db');
const db = new Database(DB_PATH, { readonly: true });

// Middleware: check database connection
app.use((req, res, next) => {
  try {
    db.prepare('SELECT 1').get();
    next();
  } catch (err) {
    res.status(500).json({ error: 'Database not accessible', details: err.message });
  }
});

// ─── API Endpoints ───

// Get provider health status (summary)
app.get('/api/providers', (req, res) => {
  const providers = db.prepare(`
    SELECT
      provider_name as name,
      status,
      rate_limit_errors,
      circuit_open as backoff,
      monthly_spend_usd,
      last_check
    FROM provider_health
  `).all();
  res.json(providers);
});

// Get recent routing decisions
app.get('/api/decisions', (req, res) => {
  const limit = parseInt(req.query.limit) || 50;
  const offset = parseInt(req.query.offset) || 0;

  const decisions = db.prepare(`
    SELECT
      timestamp,
      session_key,
      intent,
      confidence,
      chosen_provider as provider,
      chosen_model as model,
      overall_score as score,
      outcome,
      request_id
    FROM routing_decisions
    ORDER BY timestamp DESC
    LIMIT ? OFFSET ?
  `).all(limit, offset);

  res.json(decisions);
});

// Get decisions by intent
app.get('/api/decisions/intent/:intent', (req, res) => {
  const { intent } = req.params;
  const limit = parseInt(req.query.limit) || 50;

  const decisions = db.prepare(`
    SELECT
      timestamp,
      chosen_provider as provider,
      chosen_model as model,
      overall_score as score,
      outcome
    FROM routing_decisions
    WHERE intent = ?
    ORDER BY timestamp DESC
    LIMIT ?
  `).all(intent, limit);

  res.json(decisions);
});

// Get model rankings by intent (aggregated)
app.get('/api/models/rankings', (req, res) => {
  const rankings = db.prepare(`
    SELECT
      chosen_provider as provider,
      chosen_model as model,
      intent,
      COUNT(*) as calls,
      AVG(confidence) as avgConfidence,
      AVG(overall_score) as avgScore,
      SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as successes,
      COUNT(*) as total,
      ROUND(SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as successRate
    FROM routing_decisions
    WHERE timestamp > datetime('now', '-7 days')
    GROUP BY chosen_provider, chosen_model, intent
    ORDER BY intent, successRate DESC, calls DESC
  `).all();

  res.json(rankings);
});

// Get call outcomes (latency, success/failure)
app.get('/api/outcomes', (req, res) => {
  const { provider, since } = req.query;
  const limit = parseInt(req.query.limit) || 100;

  let query = `
    SELECT
      provider,
      model,
      duration_ms,
      outcome,
      timestamp
    FROM call_outcomes
    WHERE 1=1
  `;
  const params = [];

  if (provider) {
    query += ` AND provider = ?`;
    params.push(provider);
  }

  if (since) {
    query += ` AND timestamp > ?`;
    params.push(since);
  }

  query += ` ORDER BY timestamp DESC LIMIT ?`;
  params.push(limit);

  const outcomes = db.prepare(query).all(...params);
  res.json(outcomes);
});

// Get retry attempts (failure cascade analysis)
app.get('/api/retries', (req, res) => {
  const limit = parseInt(req.query.limit) || 50;

  const retries = db.prepare(`
    SELECT
      request_id,
      failed_provider,
      failed_outcome,
      retry_provider,
      timestamp
    FROM retry_attempts
    ORDER BY timestamp DESC
    LIMIT ?
  `).all(limit);

  res.json(retries);
});

// Get spend tracking (daily/monthly)
app.get('/api/spend', (req, res) => {
  const { period = 'daily' } = req.query;

  const spend = db.prepare(`
    SELECT
      date_key,
      provider,
      spend_usd
    FROM provider_spend
    WHERE period = ?
    ORDER BY date_key DESC, provider
  `).all(period);

  res.json(spend);
});

// Get benchmark results
app.get('/api/benchmarks', (req, res) => {
  const benchmarks = db.prepare(`
    SELECT
      timestamp,
      winner_model,
      winner_reason,
      notes
    FROM benchmark_results
    ORDER BY timestamp DESC
    LIMIT 10
  `).all();

  res.json(benchmarks);
});

// Get stats summary (counts, averages)
app.get('/api/stats/summary', (req, res) => {
  const stats = db.prepare(`
    SELECT
      (SELECT COUNT(*) FROM routing_decisions) as totalDecisions,
      (SELECT COUNT(*) FROM routing_decisions WHERE timestamp > datetime('now', '-1 day')) as decisionsToday,
      (SELECT COUNT(DISTINCT chosen_provider) FROM routing_decisions) as activeProviders,
      (SELECT COUNT(DISTINCT chosen_model) FROM routing_decisions) as activeModels,
      (SELECT ROUND(AVG(duration_ms), 0) FROM call_outcomes WHERE outcome = 'success' AND timestamp > datetime('now', '-1 day')) as avgLatencyMs,
      (SELECT ROUND(SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) FROM call_outcomes WHERE timestamp > datetime('now', '-1 day')) as successRateToday
  `).get();

  res.json(stats);
});

// ─── Serve Dashboard HTML ───

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// ─── Start Server ───

const PORT = process.env.PORT || 3457;
app.listen(PORT, () => {
  console.log(`Cognitive Router Dashboard API running on http://localhost:${PORT}`);
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down...');
  db.close();
  process.exit(0);
});