// src/benchmark_embeddings.ts - Embedding model quality benchmarking
// Compares 8 Ollama embedding models on semantic similarity, latency, and resource usage

import { logger } from "./logger.js";
import { type DBService } from "./db_service.js";

export interface EmbeddingModel {
  name: string;
  provider: string;
  contextWindow: number;
  isLocal: boolean;
}

export interface EmbeddingBenchmarkResult {
  modelName: string;
  qualityScore: number;      // 0-1, based on semantic similarity
  latencyMs: number;        // average latency
  vramUsageGb: number;      // estimated VRAM usage (0 if unknown)
  overallScore: number;     // weighted score
  recommendation: string;   // "NEW DEFAULT" | "GOOD ALTERNATIVE" | "SKIP"
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

// Intent prototype texts for semantic similarity testing
const INTENT_PROTOTYPES = {
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

export class EmbeddingBenchmark {
  private ollamaBaseUrl = "http://localhost:11434";

  constructor(private db: DBService) {}

  /** Run full benchmark on all available embedding models */
  async runFullBenchmark(): Promise<BenchmarkSummary> {
    logger.info("Starting embedding model benchmark...");

    // Step 1: Get all available embedding models from Ollama
    const models = await this.listEmbeddingModels();
    logger.info(`Found ${models.length} embedding models to benchmark`);

    // Step 2: Benchmark each model
    const results: EmbeddingBenchmarkResult[] = [];
    for (const model of models) {
      try {
        const result = await this.benchmarkModel(model);
        results.push(result);
        logger.info(`Benchmarked ${model.name}: quality=${result.qualityScore.toFixed(3)}, latency=${result.latencyMs.toFixed(0)}ms, overall=${result.overallScore.toFixed(3)}`);
      } catch (err) {
        logger.warn(`Failed to benchmark ${model.name}: ${err instanceof Error ? err.message : err}`);
      }
    }

    // Step 3: Sort by overall score and determine winner
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

    // Step 4: Persist results to SQLite
    await this.persistBenchmarkSummary(summary);

    return summary;
  }

  /** List all embedding models available in Ollama */
  private async listEmbeddingModels(): Promise<EmbeddingModel[]> {
    try {
      const response = await fetch(`${this.ollamaBaseUrl}/api/tags`, {
        signal: AbortSignal.timeout(10000),
      });
      if (!response.ok) {
        throw new Error(`Ollama API returned ${response.status}`);
      }

      const data = await response.json() as any;
      const embeddingModels: EmbeddingModel[] = [];

      for (const model of data.models ?? []) {
        const name = model.name as string;
        // Check if it's an embedding model (name contains "embed" or is known embedding model)
        if (name.includes("embed") || name.includes("bge") || name === "embeddinggemma:latest") {
          embeddingModels.push({
            name,
            provider: "ollama",
            contextWindow: 8192, // Most embedding models have small context
            isLocal: true,
          });
        }
      }

      return embeddingModels;
    } catch (err) {
      logger.error(`Failed to list embedding models: ${err}`);
      throw err;
    }
  }

  /** Benchmark a single embedding model */
  private async benchmarkModel(model: EmbeddingModel): Promise<EmbeddingBenchmarkResult> {
    const latencies: number[] = [];
    const embeddings: Float32Array[] = [];

    // Generate embeddings for all 10 intent prototypes
    const prototypes = Object.entries(INTENT_PROTOTYPES);
    for (const [intent, text] of prototypes) {
      const startTime = Date.now();
      try {
        const embedding = await this.generateEmbedding(model.name, text);
        const latency = Date.now() - startTime;
        latencies.push(latency);
        embeddings.push(new Float32Array(embedding));
      } catch (err) {
        logger.warn(`Failed to embed prototype "${intent}" with ${model.name}: ${err}`);
        // Use zero vector as fallback
        embeddings.push(new Float32Array(768)); // Assume 768-dim embeddings
        latencies.push(0);
      }
    }

    // Calculate average latency
    const avgLatency = latencies.length > 0
      ? latencies.reduce((a, b) => a + b, 0) / latencies.length
      : 0;

    // Calculate quality score based on semantic similarity
    const qualityScore = this.calculateQualityScore(embeddings, prototypes);

    // VRAM usage - estimate based on model size (if known in name)
    const vramUsageGb = this.estimateVramUsage(model.name);

    // Calculate overall score (weighted: quality 60%, latency 40%)
    // Normalize latency: <100ms = 1.0, >500ms = 0.0, linear between
    const latencyScore = avgLatency < 100 ? 1.0 : Math.max(0, 1 - (avgLatency - 100) / 400);
    const overallScore = qualityScore * 0.6 + latencyScore * 0.4;

    // Determine recommendation
    let recommendation: string;
    if (overallScore > 0.90 && avgLatency < 100) {
      recommendation = "NEW DEFAULT";
    } else if (overallScore > 0.80) {
      recommendation = "GOOD ALTERNATIVE";
    } else {
      recommendation = "SKIP";
    }

    return {
      modelName: model.name,
      qualityScore,
      latencyMs: avgLatency,
      vramUsageGb: vramUsageGb,
      overallScore,
      recommendation,
    };
  }

  /** Generate an embedding for a single text */
  private async generateEmbedding(model: string, text: string): Promise<number[]> {
    const response = await fetch(`${this.ollamaBaseUrl}/api/embed`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        input: text,
      }),
      signal: AbortSignal.timeout(90000), // 90s timeout for first-time model loading
    });

    if (!response.ok) {
      throw new Error(`Embedding request failed: ${response.status}`);
    }

    const data = await response.json() as any;
    return data.embedding;
  }

  /** Calculate quality score based on semantic similarity between intents */
  private calculateQualityScore(embeddings: Float32Array[], prototypes: [string, string][]): number {
    // Compute cosine similarity matrix
    const similarities: number[] = [];

    for (let i = 0; i < embeddings.length; i++) {
      for (let j = i + 1; j < embeddings.length; j++) {
        const sim = this.cosineSimilarity(embeddings[i], embeddings[j]);
        similarities.push(sim);
      }
    }

    if (similarities.length === 0) return 0;

    // Good embeddings should have:
    // - Low similarity between different intents (inter-category separation)
    // - High similarity for same intents (intra-category consistency)

    // Since we only have one sample per intent, we focus on inter-category separation
    // Lower average similarity = better separation = higher quality
    const avgSimilarity = similarities.reduce((a, b) => a + b, 0) / similarities.length;

    // Convert to score: lower similarity = higher quality (inverted relationship)
    // Avg similarity typically ranges from 0.1 (excellent) to 0.8 (poor)
    // If avg similarity is 0.1, quality should be ~1.0
    // If avg similarity is 0.8, quality should be ~0.0
    const qualityScore = Math.max(0, Math.min(1, 1 - (avgSimilarity - 0.1) / 0.7));

    return qualityScore;
  }

  /** Calculate cosine similarity between two embedding vectors */
  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) {
      // Pad shorter vector with zeros
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
    // Heuristic VRAM estimates based on model size in name
    if (modelName.includes("567m")) return 0.5;
    if (modelName.includes("1b") || modelName.includes("1.5b")) return 1.0;
    if (modelName.includes("3b") || modelName.includes("4b")) return 2.0;
    if (modelName.includes("8b")) return 4.0;
    if (modelName.includes("12b")) return 6.0;
    return 0; // Unknown
  }

  /** Generate explanation for why a model won */
  private getRecommendationReason(result: EmbeddingBenchmarkResult): string {
    const reasons: string[] = [];
    if (result.qualityScore > 0.85) {
      reasons.push("excellent semantic separation");
    } else if (result.qualityScore > 0.75) {
      reasons.push("good semantic separation");
    } else {
      reasons.push("adequate semantic separation");
    }

    if (result.latencyMs < 100) {
      reasons.push("fast latency (<100ms)");
    } else if (result.latencyMs < 200) {
      reasons.push("acceptable latency");
    } else {
      reasons.push("slow latency");
    }

    return reasons.join(", ");
  }

  /** Generate notes from benchmark results */
  private generateNotes(results: EmbeddingBenchmarkResult[]): string[] {
    const notes: string[] = [];
    const newDefault = results.find(r => r.recommendation === "NEW DEFAULT");
    const alternatives = results.filter(r => r.recommendation === "GOOD ALTERNATIVE");

    if (newDefault) {
      notes.push(`Recommended default: ${newDefault.modelName} (score: ${newDefault.overallScore.toFixed(3)})`);
    }

    if (alternatives.length > 0) {
      notes.push(`Good alternatives: ${alternatives.map(r => r.modelName).join(", ")}`);
    }

    const avgLatency = results.reduce((a, b) => a + b.latencyMs, 0) / results.length;
    notes.push(`Average latency across all models: ${avgLatency.toFixed(0)}ms`);

    return notes;
  }

  /** Persist benchmark summary to SQLite */
  private async persistBenchmarkSummary(summary: BenchmarkSummary): Promise<void> {
    const stmt = this.db.getDb().prepare(`
      INSERT INTO benchmark_results (timestamp, winner_model, winner_reason, results_json, notes)
      VALUES (?, ?, ?, ?, ?)
    `);

    stmt.run(
      summary.timestamp,
      summary.winner.modelName,
      summary.winner.reason,
      JSON.stringify(summary.results),
      summary.notes.join("; "),
    );

    logger.info(`Benchmark summary persisted to SQLite. Winner: ${summary.winner.modelName}`);
  }
}

// Add table schema to DB service if not exists
export function initializeBenchmarkTables(db: DBService): void {
  const dbInstance = db.getDb();
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
}