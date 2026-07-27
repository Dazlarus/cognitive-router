// src/benchmark_embeddings.ts - Enhanced embedding model quality benchmarking
// Captures 8+ metrics: semantic separation, dimension count, clustering quality,
// batch speed, latency, memory footprint, domain-specific probes, and overall score.

import { logger } from "./logger.js";
import { type DBService } from "./db_service.js";

// ─── Types ───

export interface EmbeddingModelInfo {
  name: string;
  provider: string;
  contextWindow: number;
  isLocal: boolean;
  dimensions: number;
}

export interface DomainScore {
  domain: string;
  separationScore: number; // 0-1, how well this domain separates from others
}

export interface EmbeddingBenchmarkResult {
  modelName: string;
  provider: string;
  dimensions: number;
  qualityScore: number;        // 0-1, inter-category semantic separation
  clusteringScore: number;     // 0-1, k-means cohesion quality
  latencyMs: number;           // average single-embed latency
  batchSpeedMs: number;        // time to process 100 texts
  memoryMb: number;            // estimated memory footprint
  vramUsageGb: number;         // estimated VRAM usage (0 if unknown)
  domainScores: DomainScore[]; // per-domain probe results
  overallScore: number;        // weighted composite
  recommendation: string;      // "NEW DEFAULT" | "GOOD ALTERNATIVE" | "SKIP"
}

export interface BenchmarkSummary {
  timestamp: string;
  results: EmbeddingBenchmarkResult[];
  winner: {
    modelName: string;
    reason: string;
  };
  notes: string[];
}

// ─── Probe Texts ───

/** Inter-category probes: 10 diverse intents for semantic separation testing */
const INTER_CATEGORY_PROBES: Record<string, string> = {
  coding: "Write efficient Python code to sort a list of dictionaries by value and handle edge cases",
  research: "Analyze the research literature on transformer attention mechanisms and summarize key findings",
  creative: "Write a short science fiction story about an AI discovering consciousness",
  conversation: "That's interesting! Tell me more about your experience with this project",
  summary: "Summarize the main arguments and supporting evidence from the provided article",
  retrieval: "Find relevant documents about machine learning model optimization techniques",
  science: "Explain the process of protein folding and its importance in biological research",
  business: "Draft a proposal for improving operational efficiency and reducing costs in Q4",
  math: "Solve the differential equation: dy/dx + 2y = e^(-x), with initial condition y(0)=1",
  analysis: "Compare and contrast the performance characteristics of three different database systems",
};

/** Domain-specific probe sets — each domain has 3 texts that should cluster together */
const DOMAIN_PROBES: Record<string, string[]> = {
  technical: [
    "The TCP handshake involves SYN, SYN-ACK, and ACK packets to establish a reliable connection.",
    "Kubernetes orchestrates containerized workloads through pods, services, and deployments.",
    "The Rust borrow checker enforces ownership rules at compile time to prevent memory safety bugs.",
  ],
  creative: [
    "The autumn leaves danced in the wind, painting the sidewalk in shades of amber and crimson.",
    "She hesitated at the door, her hand trembling as she reached for the brass handle.",
    "The old lighthouse stood defiant against the storm, its beam cutting through the darkness like a sword.",
  ],
  code: [
    "// TODO: refactor this to use async/await instead of promise chains\nfunction loadData(url) {\n  return fetch(url).then(r => r.json()).then(d => d.results);\n}",
    "/* Calculate fibonacci with memoization */\nconst fib = (n, memo = {}) => memo[n] ?? (memo[n] = n < 2 ? n : fib(n-1, memo) + fib(n-2, memo));",
    "# Deduplicates entries by key\ndef dedupe(items, key='id'):\n    seen = set()\n    return [x for x in items if not (x[key] in seen or seen.add(x[key]))]",
  ],
};

/** Batch probe texts — 100 short texts for batch speed measurement */
const BATCH_TEXTS: string[] = [
  "The quick brown fox jumps over the lazy dog.",
  "Machine learning models require large amounts of training data.",
  "Cloud computing has revolutionized how businesses manage infrastructure.",
  "Quantum entanglement allows particles to share states across vast distances.",
  "The Renaissance period saw tremendous advances in art and science.",
  ...generateBatchTexts(95),
];

function generateBatchTexts(count: number): string[] {
  const templates = [
    "Document {i}: An analysis of system performance under load conditions.",
    "Record {i}: Customer feedback regarding the new product launch initiative.",
    "Entry {i}: Technical specification for module integration and testing.",
    "Note {i}: Research findings on climate change impact assessment.",
    "Item {i}: Financial projection for quarterly revenue growth analysis.",
  ];
  return Array.from({ length: count }, (_, i) => templates[i % templates.length].replace("{i}", String(i + 5)));
}

// ─── Benchmark Class ───

export class EmbeddingBenchmark {
  private ollamaBaseUrl = "http://localhost:11434";

  constructor(private db: DBService) {}

  /** Run full benchmark on all available embedding models */
  async runFullBenchmark(): Promise<BenchmarkSummary> {
    logger.info("Starting enhanced embedding model benchmark...");

    const models = await this.listEmbeddingModels();
    logger.info(`Found ${models.length} embedding models to benchmark`);

    const results: EmbeddingBenchmarkResult[] = [];
    for (const model of models) {
      try {
        const result = await this.benchmarkModel(model);
        results.push(result);
        logger.info(
          `Benchmarked ${model.name}: quality=${result.qualityScore.toFixed(3)}, ` +
          `cluster=${result.clusteringScore.toFixed(3)}, ` +
          `dims=${result.dimensions}, batch=${result.batchSpeedMs.toFixed(0)}ms, ` +
          `overall=${result.overallScore.toFixed(3)}`,
        );
      } catch (err) {
        logger.warn(`Failed to benchmark ${model.name}: ${err instanceof Error ? err.message : err}`);
      }
    }

    results.sort((a, b) => b.overallScore - a.overallScore);
    const winner = results[0];

    const summary: BenchmarkSummary = {
      timestamp: new Date().toISOString(),
      results,
      winner: {
        modelName: winner.modelName,
        reason: this.getRecommendationReason(winner),
      },
      notes: this.generateNotes(results),
    };

    await this.persistBenchmarkSummary(summary);

    return summary;
  }

  /** List all embedding models available in Ollama */
  async listEmbeddingModels(): Promise<EmbeddingModelInfo[]> {
    try {
      const response = await fetch(`${this.ollamaBaseUrl}/api/tags`, {
        signal: AbortSignal.timeout(10000),
      });
      if (!response.ok) {
        throw new Error(`Ollama API returned ${response.status}`);
      }

      const data = (await response.json()) as any;
      const embeddingModels: EmbeddingModelInfo[] = [];

      for (const model of data.models ?? []) {
        const name = model.name as string;
        if (
          name.includes("embed") ||
          name.includes("bge") ||
          name === "embeddinggemma:latest"
        ) {
          embeddingModels.push({
            name,
            provider: "ollama",
            contextWindow: 8192,
            isLocal: true,
            dimensions: 0, // will be filled during benchmark
          });
        }
      }

      return embeddingModels;
    } catch (err) {
      logger.error(`Failed to list embedding models: ${err}`);
      throw err;
    }
  }

  /** Benchmark a single embedding model across all metrics */
  private async benchmarkModel(model: EmbeddingModelInfo): Promise<EmbeddingBenchmarkResult> {
    // ── 1. Semantic separation (inter-category probes) ──
    const latencies: number[] = [];
    const embeddings: Float32Array[] = [];
    const probeEntries = Object.entries(INTER_CATEGORY_PROBES);

    for (const [intent, text] of probeEntries) {
      const startTime = Date.now();
      try {
        const embedding = await this.generateEmbedding(model.name, text);
        const latency = Date.now() - startTime;
        latencies.push(latency);
        embeddings.push(new Float32Array(embedding));
      } catch (err) {
        logger.warn(`Failed to embed prototype "${intent}" with ${model.name}: ${err}`);
        embeddings.push(new Float32Array(768));
        latencies.push(0);
      }
    }

    const avgLatency = latencies.length > 0
      ? latencies.reduce((a, b) => a + b, 0) / latencies.length
      : 0;

    const qualityScore = this.calculateQualityScore(embeddings);
    const dimensions = embeddings[0]?.length ?? 0;
    model.dimensions = dimensions;

    // ── 2. Clustering quality (domain probes via k-means cohesion) ──
    const { clusteringScore, domainScores } = await this.evaluateDomainProbes(model.name);

    // ── 3. Batch speed (100 texts) ──
    const batchSpeedMs = await this.measureBatchSpeed(model.name);

    // ── 4. Memory footprint estimation ──
    const memoryMb = this.estimateMemoryMb(dimensions, model.name);

    // ── 5. VRAM estimate ──
    const vramUsageGb = this.estimateVramUsage(model.name);

    // ── Composite score ──
    // Weights: quality 25%, clustering 20%, latency 15%, batch 15%, memory 10%, domain avg 15%
    const latencyScore = avgLatency < 100 ? 1.0 : Math.max(0, 1 - (avgLatency - 100) / 400);
    const batchScore = batchSpeedMs < 2000 ? 1.0 : Math.max(0, 1 - (batchSpeedMs - 2000) / 8000);
    const memScore = memoryMb < 200 ? 1.0 : Math.max(0, 1 - (memoryMb - 200) / 800);
    const domainAvg = domainScores.length > 0
      ? domainScores.reduce((a, d) => a + d.separationScore, 0) / domainScores.length
      : 0;

    const overallScore =
      qualityScore * 0.25 +
      clusteringScore * 0.20 +
      latencyScore * 0.15 +
      batchScore * 0.15 +
      memScore * 0.10 +
      domainAvg * 0.15;

    let recommendation: string;
    if (overallScore > 0.80 && avgLatency < 150) {
      recommendation = "NEW DEFAULT";
    } else if (overallScore > 0.65) {
      recommendation = "GOOD ALTERNATIVE";
    } else {
      recommendation = "SKIP";
    }

    return {
      modelName: model.name,
      provider: model.provider,
      dimensions,
      qualityScore,
      clusteringScore,
      latencyMs: avgLatency,
      batchSpeedMs,
      memoryMb,
      vramUsageGb,
      domainScores,
      overallScore,
      recommendation,
    };
  }

  /** Generate an embedding for a single text */
  private async generateEmbedding(model: string, text: string): Promise<number[]> {
    const response = await fetch(`${this.ollamaBaseUrl}/api/embed`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, input: text }),
      signal: AbortSignal.timeout(90000),
    });

    if (!response.ok) {
      throw new Error(`Embedding request failed: ${response.status}`);
    }

    const data = (await response.json()) as any;
    return data.embeddings[0];
  }

  /** Generate embeddings in batch (Ollama supports batch input) */
  private async generateEmbeddingsBatch(model: string, texts: string[]): Promise<number[][]> {
    const response = await fetch(`${this.ollamaBaseUrl}/api/embed`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, input: texts }),
      signal: AbortSignal.timeout(120000),
    });

    if (!response.ok) {
      throw new Error(`Batch embedding request failed: ${response.status}`);
    }

    const data = (await response.json()) as any;
    return data.embeddings as number[][];
  }

  /** Calculate quality score from inter-category semantic separation */
  private calculateQualityScore(embeddings: Float32Array[]): number {
    const similarities: number[] = [];

    for (let i = 0; i < embeddings.length; i++) {
      for (let j = i + 1; j < embeddings.length; j++) {
        const sim = this.cosineSimilarity(embeddings[i], embeddings[j]);
        similarities.push(sim);
      }
    }

    if (similarities.length === 0) return 0;

    const avgSimilarity =
      similarities.reduce((a, b) => a + b, 0) / similarities.length;

    logger.info(
      `  → Raw avgSimilarity: ${avgSimilarity.toFixed(3)} ` +
        `(min: ${Math.min(...similarities).toFixed(3)}, max: ${Math.max(...similarities).toFixed(3)})`,
    );

    // Lower avg similarity = better separation = higher quality
    // Range: 0.2 (excellent) → 1.0, 0.6 (poor) → 0.0
    return Math.max(0, Math.min(1, 1 - (avgSimilarity - 0.2) / 0.4));
  }

  /**
   * Evaluate domain-specific probes using k-means style clustering cohesion.
   * Each domain has 3 probe texts. We compute intra-domain cosine similarity
   * (should be high) vs inter-domain similarity (should be low).
   */
  private async evaluateDomainProbes(
    modelName: string,
  ): Promise<{ clusteringScore: number; domainScores: DomainScore[] }> {
    const domainNames = Object.keys(DOMAIN_PROBES);
    const domainEmbeddings: Record<string, Float32Array[]> = {};

    // Generate embeddings for all domain probes
    for (const domain of domainNames) {
      const texts = DOMAIN_PROBES[domain];
      domainEmbeddings[domain] = [];
      for (const text of texts) {
        try {
          const emb = await this.generateEmbedding(modelName, text);
          domainEmbeddings[domain].push(new Float32Array(emb));
        } catch {
          domainEmbeddings[domain].push(new Float32Array(768));
        }
      }
    }

    // Calculate intra-domain cohesion (avg pairwise similarity within each domain)
    const domainScores: DomainScore[] = [];
    let totalIntraSim = 0;
    let totalIntraPairs = 0;

    for (const domain of domainNames) {
      const embs = domainEmbeddings[domain];
      const sims: number[] = [];

      for (let i = 0; i < embs.length; i++) {
        for (let j = i + 1; j < embs.length; j++) {
          sims.push(this.cosineSimilarity(embs[i], embs[j]));
        }
      }

      const avgIntra = sims.length > 0 ? sims.reduce((a, b) => a + b, 0) / sims.length : 0;
      totalIntraSim += sims.reduce((a, b) => a + b, 0);
      totalIntraPairs += sims.length;

      // Per-domain separation: how much higher is intra vs inter similarity
      domainScores.push({
        domain,
        separationScore: Math.max(0, Math.min(1, avgIntra)),
      });
    }

    // Calculate inter-domain separation (avg pairwise similarity across domains)
    const interSims: number[] = [];
    for (let di = 0; di < domainNames.length; di++) {
      for (let dj = di + 1; dj < domainNames.length; dj++) {
        const embsA = domainEmbeddings[domainNames[di]];
        const embsB = domainEmbeddings[domainNames[dj]];
        for (const a of embsA) {
          for (const b of embsB) {
            interSims.push(this.cosineSimilarity(a, b));
          }
        }
      }
    }

    const avgIntra = totalIntraPairs > 0 ? totalIntraSim / totalIntraPairs : 0;
    const avgInter =
      interSims.length > 0 ? interSims.reduce((a, b) => a + b, 0) / interSims.length : 0;

    // Clustering score: intra should be high, inter should be low
    // Score = (intra - inter), normalized to 0-1
    const clusteringScore = Math.max(0, Math.min(1, avgIntra - avgInter));

    logger.info(
      `  → Clustering: intra=${avgIntra.toFixed(3)}, inter=${avgInter.toFixed(3)}, ` +
        `score=${clusteringScore.toFixed(3)}`,
    );

    return { clusteringScore, domainScores };
  }

  /** Measure batch embedding speed with 100 texts */
  private async measureBatchSpeed(modelName: string): Promise<number> {
    try {
      const startTime = Date.now();
      await this.generateEmbeddingsBatch(modelName, BATCH_TEXTS);
      return Date.now() - startTime;
    } catch (err) {
      logger.warn(`Batch speed measurement failed for ${modelName}: ${err}`);
      // Fallback: estimate from single latency × 100
      const singleStart = Date.now();
      try {
        await this.generateEmbedding(modelName, "test");
        const singleMs = Date.now() - singleStart;
        return singleMs * 100;
      } catch {
        return 99999;
      }
    }
  }

  /** Estimate memory footprint based on dimensions and model size */
  private estimateMemoryMb(dimensions: number, modelName: string): number {
    // Rough heuristic: model weights + per-vector overhead
    let modelSizeMb = 0;
    if (modelName.includes("567m")) modelSizeMb = 600;
    else if (modelName.includes("1b") || modelName.includes("1.5b")) modelSizeMb = 1200;
    else if (modelName.includes("3b") || modelName.includes("4b")) modelSizeMb = 2400;
    else if (modelName.includes("8b")) modelSizeMb = 4800;
    else if (modelName.includes("nomic")) modelSizeMb = 137;
    else if (modelName.includes("bge")) modelSizeMb = 330;
    else if (modelName.includes("mini")) modelSizeMb = 120;
    else modelSizeMb = 300; // default assumption

    // Working memory for inference: ~2x dimensions × float32 × batch
    const workingMemoryMb = (dimensions * 4 * 128) / (1024 * 1024);
    return Math.round(modelSizeMb + workingMemoryMb);
  }

  /** Calculate cosine similarity between two embedding vectors */
  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) {
      const maxLength = Math.max(a.length, b.length);
      const aPadded = new Float32Array(maxLength);
      const bPadded = new Float32Array(maxLength);
      aPadded.set(a);
      bPadded.set(b);
      return this.cosineSimilarity(aPadded, bPadded);
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);

    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (normA * normB);
  }

  /** Estimate VRAM usage based on model name */
  private estimateVramUsage(modelName: string): number {
    if (modelName.includes("567m")) return 0.5;
    if (modelName.includes("1b") || modelName.includes("1.5b")) return 1.0;
    if (modelName.includes("3b") || modelName.includes("4b")) return 2.0;
    if (modelName.includes("8b")) return 4.0;
    if (modelName.includes("12b")) return 6.0;
    return 0;
  }

  private getRecommendationReason(result: EmbeddingBenchmarkResult): string {
    const reasons: string[] = [];
    if (result.qualityScore > 0.7) reasons.push("excellent semantic separation");
    else if (result.qualityScore > 0.5) reasons.push("good semantic separation");
    else reasons.push("adequate semantic separation");

    if (result.clusteringScore > 0.3) reasons.push("strong domain clustering");
    else reasons.push("weak domain clustering");

    if (result.latencyMs < 100) reasons.push("fast latency (<100ms)");
    else if (result.latencyMs < 200) reasons.push("acceptable latency");
    else reasons.push("slow latency");

    if (result.batchSpeedMs < 2000) reasons.push("excellent batch throughput");
    else if (result.batchSpeedMs < 5000) reasons.push("good batch throughput");
    else reasons.push("slow batch throughput");

    return reasons.join(", ");
  }

  private generateNotes(results: EmbeddingBenchmarkResult[]): string[] {
    const notes: string[] = [];
    const newDefault = results.find((r) => r.recommendation === "NEW DEFAULT");
    const alternatives = results.filter((r) => r.recommendation === "GOOD ALTERNATIVE");

    if (newDefault) {
      notes.push(`Recommended default: ${newDefault.modelName} (score: ${newDefault.overallScore.toFixed(3)})`);
    }
    if (alternatives.length > 0) {
      notes.push(`Good alternatives: ${alternatives.map((r) => r.modelName).join(", ")}`);
    }
    const avgLatency = results.reduce((a, b) => a + b.latencyMs, 0) / results.length;
    notes.push(`Average latency across all models: ${avgLatency.toFixed(0)}ms`);
    const avgBatch = results.reduce((a, b) => a + b.batchSpeedMs, 0) / results.length;
    notes.push(`Average batch speed (100 texts): ${avgBatch.toFixed(0)}ms`);

    // Dimension range
    const dims = results.map((r) => r.dimensions).filter((d) => d > 0);
    if (dims.length > 0) {
      notes.push(`Dimension range: ${Math.min(...dims)} - ${Math.max(...dims)}`);
    }

    return notes;
  }

  // ─── Persistence ───

  private async persistBenchmarkSummary(summary: BenchmarkSummary): Promise<void> {
    const db = this.db.getDb();

    // Insert main summary record
    const stmt = db.prepare(`
      INSERT INTO embedding_benchmark_runs
        (timestamp, winner_model, winner_reason, results_json, notes)
      VALUES (?, ?, ?, ?, ?)
    `);

    stmt.run(
      summary.timestamp,
      summary.winner.modelName,
      summary.winner.reason,
      JSON.stringify(summary.results),
      summary.notes.join("; "),
    );

    // Insert per-model per-domain scores
    const domainStmt = db.prepare(`
      INSERT INTO embedding_benchmark_scores
        (timestamp, model_name, provider, metric_type, metric_value, details_json)
      VALUES (?, ?, ?, ?, ?, ?)
    `);

    for (const result of summary.results) {
      // Overall metrics
      const metrics: Array<[string, number]> = [
        ["quality_score", result.qualityScore],
        ["clustering_score", result.clusteringScore],
        ["latency_ms", result.latencyMs],
        ["batch_speed_ms", result.batchSpeedMs],
        ["memory_mb", result.memoryMb],
        ["vram_usage_gb", result.vramUsageGb],
        ["overall_score", result.overallScore],
        ["dimensions", result.dimensions],
      ];

      for (const [metricType, metricValue] of metrics) {
        domainStmt.run(
          summary.timestamp,
          result.modelName,
          result.provider,
          metricType,
          metricValue,
          null,
        );
      }

      // Per-domain scores
      for (const ds of result.domainScores) {
        domainStmt.run(
          summary.timestamp,
          result.modelName,
          result.provider,
          `domain_${ds.domain}`,
          ds.separationScore,
          null,
        );
      }
    }

    logger.info(
      `Benchmark summary persisted to SQLite. Winner: ${summary.winner.modelName}`,
    );
  }

  /** Query the most recent benchmark results from SQLite */
  getLatestResults(limit = 5): any[] {
    const db = this.db.getDb();
    return db
      .prepare(
        `SELECT * FROM embedding_benchmark_runs ORDER BY timestamp DESC LIMIT ?`,
      )
      .all(limit);
  }

  /** Query per-model scores from the most recent benchmark */
  getLatestModelScores(): any[] {
    const db = this.db.getDb();
    const latest = db
      .prepare(
        `SELECT timestamp FROM embedding_benchmark_runs ORDER BY timestamp DESC LIMIT 1`,
      )
      .get() as { timestamp: string } | undefined;

    if (!latest) return [];

    return db
      .prepare(
        `SELECT model_name, provider, metric_type, metric_value
         FROM embedding_benchmark_scores
         WHERE timestamp = ?
         ORDER BY model_name, metric_type`,
      )
      .all(latest.timestamp);
  }
}

/** Initialize embedding benchmark tables in SQLite */
export function initializeBenchmarkTables(db: DBService): void {
  const dbInstance = db.getDb();

  // Legacy table — keep for backward compat
  dbInstance.exec(`
    CREATE TABLE IF NOT EXISTS benchmark_results (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp TEXT NOT NULL,
      winner_model TEXT NOT NULL,
      winner_reason TEXT,
      results_json TEXT,
      notes TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_benchmark_timestamp ON benchmark_results(timestamp);
  `);

  // New enhanced tables
  dbInstance.exec(`
    CREATE TABLE IF NOT EXISTS embedding_benchmark_runs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp TEXT NOT NULL,
      winner_model TEXT NOT NULL,
      winner_reason TEXT,
      results_json TEXT NOT NULL,
      notes TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_emb_runs_timestamp ON embedding_benchmark_runs(timestamp);

    CREATE TABLE IF NOT EXISTS embedding_benchmark_scores (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp TEXT NOT NULL,
      model_name TEXT NOT NULL,
      provider TEXT NOT NULL,
      metric_type TEXT NOT NULL,
      metric_value REAL NOT NULL,
      details_json TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_emb_scores_timestamp ON embedding_benchmark_scores(timestamp);
    CREATE INDEX IF NOT EXISTS idx_emb_scores_model ON embedding_benchmark_scores(model_name);
    CREATE INDEX IF NOT EXISTS idx_emb_scores_metric ON embedding_benchmark_scores(metric_type);
  `);
}
