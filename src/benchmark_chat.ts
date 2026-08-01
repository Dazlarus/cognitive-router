// src/benchmark_chat.ts — Persistent multi-intent chat model benchmarking
//
// Replaces the shallow single-probe auto-benchmark with:
//   - 3 probe types: coding, reasoning, conversation
//   - LLM-as-judge scoring (not heuristic char counting)
//   - Model version detection (Ollama digest / OpenRouter model ID)
//   - Persistent caching in SQLite (skip if <7 days old + version unchanged)
//   - Traffic-weighted composite scoring from routing_decisions

import { logger } from "./logger.js";
import type { DBService } from "./db_service.js";
import type { ModelRegistry } from "./model_registry.js";
import { getProvider } from "./providers.js";

// ─── Types ───

export type ProbeType = "coding" | "reasoning" | "conversation";

export interface ProbeResult {
  probeType: ProbeType;
  /** Judge score 0-1 */
  score: number;
  /** Raw judge response */
  judgeNote: string;
  /** Response latency in ms */
  latencyMs: number;
  /** Whether this was a cache hit */
  cached: boolean;
}

export interface ChatBenchmarkResult {
  modelId: string;
  provider: string;
  model: string;
  probeResults: ProbeResult[];
  /** Traffic-weighted composite score 0-1 */
  compositeScore: number;
  /** Version hash at time of benchmark */
  modelVersionHash: string;
  timestamp: string;
}

// ─── Constants ───

const BENCHMARK_MAX_AGE_DAYS = 7;
const PROBE_TIMEOUT_MS = 30_000;
const JUDGE_TIMEOUT_MS = 15_000;

/** Map routing intent → probe type for traffic weighting */
const INTENT_TO_PROBE: Record<string, ProbeType> = {
  coding: "coding",
  math: "reasoning",
  analysis: "reasoning",
  science: "reasoning",
  reasoning: "reasoning",
  conversation: "conversation",
  creative: "conversation",
  summary: "conversation",
  business: "conversation",
  retrieval: "conversation",
  research: "reasoning",
};

// ─── Probe Prompts ───

const PROBES: Record<ProbeType, { prompt: string; systemPrompt?: string }> = {
  coding: {
    prompt: "Write a Python function `is_balanced(s: str) -> bool` that checks if a string of brackets ((), [], {}) is balanced. Include type hints and a brief docstring. No explanation needed.",
  },
  reasoning: {
    prompt: "If all roses are flowers and some flowers fade quickly, do all roses fade quickly? Explain your reasoning step by step, then give a definitive answer.",
  },
  conversation: {
    prompt: "If you could have dinner with any historical figure, who would it be and what would you discuss? Respond naturally and conversationally.",
  },
};

/** Judge prompt for scoring probe responses */
const JUDGE_PROMPT_TEMPLATE = `You are an impartial judge evaluating an AI assistant's response to a {probe_type} probe. Rate the response quality on a scale of 0-10.

**Probe type:** {probe_type}

**User request:**
{prompt}

**Assistant response:**
{response}

**Scoring guide for {probe_type}:**
- 0-2: Completely wrong, broken, or nonsensical
- 3-4: Partially correct but with significant issues
- 5-6: Adequate — correct direction, usable but with gaps
- 7-8: Good — correct, clear, and complete
- 9-10: Excellent — exceptional quality, clarity, and insight

Respond with ONLY a JSON object, no other text:
{{"score": <integer 0-10>, "note": "<one sentence explanation>"}}`;

// ─── ChatBenchmark Class ───

export class ChatBenchmark {
  private ollamaBaseUrl = "http://localhost:11434";
  private judgeProvider: string;
  private judgeModel: string;

  constructor(
    private db: DBService,
    private registry: ModelRegistry,
  ) {
    this.judgeProvider = process.env.ROUTER_JUDGE_PROVIDER ?? "openrouter";
    this.judgeModel = process.env.ROUTER_JUDGE_MODEL ?? "qwen/qwen3-30b-a3b-instruct-2507";
  }

  /** Main entry: benchmark a single chat model with all probe types.
   *  Skips probes that have fresh cached results with matching version hash. */
  async benchmarkModel(
    provider: string,
    model: string,
    forceRefresh: boolean = false,
  ): Promise<ChatBenchmarkResult> {
    const modelId = `${provider}/${model}`;
    const versionHash = await this.getVersionHash(provider, model);
    const probeTypes: ProbeType[] = ["coding", "reasoning", "conversation"];
    const probeResults: ProbeResult[] = [];

    logger.info(`Chat benchmark starting for ${modelId} (version=${versionHash.substring(0, 12)})...`);

    for (const probeType of probeTypes) {
      // Check cache first
      if (!forceRefresh) {
        const cached = this.db.getLatestChatBenchmark(modelId, probeType);
        if (cached && cached.modelVersionHash === versionHash) {
          const age = this.ageInDays(cached.timestamp);
          if (age < BENCHMARK_MAX_AGE_DAYS) {
            const scores = JSON.parse(cached.scoresJson);
            logger.info(`  ${probeType}: cached (age=${age.toFixed(1)}d, score=${scores.score?.toFixed(2) ?? "?"})`);
            probeResults.push({
              probeType,
              score: scores.score ?? 0.5,
              judgeNote: scores.judgeNote ?? "",
              latencyMs: cached.latencyMs,
              cached: true,
            });
            continue;
          }
        }
      }

      // Run the probe
      const result = await this.runProbe(provider, model, probeType);
      probeResults.push(result);

      // Persist to DB
      this.db.saveChatBenchmarkResult({
        modelId,
        provider,
        model,
        probeType,
        scoresJson: JSON.stringify({
          score: result.score,
          judgeNote: result.judgeNote,
        }),
        latencyMs: result.latencyMs,
        modelVersionHash: versionHash,
      });

      logger.info(
        `  ${probeType}: score=${result.score.toFixed(2)} latency=${result.latencyMs}ms ` +
        `(judge: ${result.judgeNote.substring(0, 60)})`,
      );
    }

    // Compute traffic-weighted composite
    const compositeScore = this.computeTrafficWeightedScore(probeResults);

    const result: ChatBenchmarkResult = {
      modelId,
      provider,
      model,
      probeResults,
      compositeScore,
      modelVersionHash: versionHash,
      timestamp: new Date().toISOString(),
    };

    logger.info(`Chat benchmark complete for ${modelId}: composite=${compositeScore.toFixed(3)}`);
    return result;
  }

  /** Benchmark multiple models (sequential for local, respects GPU serialization). */
  async benchmarkModels(
    models: Array<{ provider: string; model: string }>,
    forceRefresh: boolean = false,
  ): Promise<ChatBenchmarkResult[]> {
    const results: ChatBenchmarkResult[] = [];
    for (const { provider, model } of models) {
      try {
        const result = await this.benchmarkModel(provider, model, forceRefresh);
        results.push(result);

        // Apply results to the model registry
        this.applyToRegistry(result);
      } catch (err) {
        logger.warn(
          `Chat benchmark failed for ${provider}/${model}: ${err instanceof Error ? err.message : err}`,
        );
      }
    }
    return results;
  }

  /** Run a single probe: send prompt to model, get response, judge it. */
  private async runProbe(
    provider: string,
    model: string,
    probeType: ProbeType,
  ): Promise<ProbeResult> {
    const probe = PROBES[probeType];
    const messages = [
      ...(probe.systemPrompt ? [{ role: "system", content: probe.systemPrompt }] : []),
      { role: "user", content: probe.prompt },
    ];

    const adapter = getProvider(provider);
    if (!adapter) {
      throw new Error(`Unknown provider: ${provider}`);
    }

    const apiKey = process.env[provider.toUpperCase() + "_API_KEY"] ?? "";
    const startTime = Date.now();

    // Get model response
    const response = await adapter.chatCompletion(
      model,
      {
        model,
        messages: messages as any,
        stream: false,
        temperature: 0.3,
        max_tokens: 500,
      },
      apiKey,
      AbortSignal.timeout(PROBE_TIMEOUT_MS),
    );

    const latencyMs = Date.now() - startTime;
    const content = response.choices?.[0]?.message?.content ?? "";

    if (!content.trim()) {
      return {
        probeType,
        score: 0,
        judgeNote: "Empty response",
        latencyMs,
        cached: false,
      };
    }

    // Judge the response
    const judgeResult = await this.judgeResponse(probeType, probe.prompt, content);

    return {
      probeType,
      score: judgeResult?.score ?? 0.5,
      judgeNote: judgeResult?.note ?? "Judge unavailable",
      latencyMs,
      cached: false,
    };
  }

  /** Use LLM-as-judge to score a probe response. */
  private async judgeResponse(
    probeType: ProbeType,
    prompt: string,
    response: string,
  ): Promise<{ score: number; note: string } | null> {
    const judgePrompt = JUDGE_PROMPT_TEMPLATE
      .replace(/{probe_type}/g, probeType)
      .replace("{prompt}", prompt.slice(0, 1000))
      .replace("{response}", response.slice(0, 2000));

    // Try judge candidates in priority order
    const candidates = this.getJudgeCandidates();

    for (const { provider: jp, model: jm } of candidates) {
      const adapter = getProvider(jp);
      if (!adapter) continue;
      const apiKey = process.env[jp.toUpperCase() + "_API_KEY"] ?? "";
      if (!apiKey && jp !== "ollama") continue;

      try {
        const result = await adapter.chatCompletion(
          jm,
          {
            model: jm,
            messages: [{ role: "user", content: judgePrompt }],
            stream: false,
            temperature: 0.1,
            max_tokens: 100,
          },
          apiKey,
          AbortSignal.timeout(JUDGE_TIMEOUT_MS),
        );

        const content = result.choices?.[0]?.message?.content ?? "";
        const parsed = this.parseJudgeResponse(content);
        if (parsed) {
          return parsed;
        }
      } catch (err) {
        logger.debug(`Judge candidate ${jp}/${jm} failed: ${err instanceof Error ? err.message : err}`);
        continue;
      }
    }

    logger.debug("Judge: all candidates exhausted, returning null");
    return null;
  }

  /** Ordered list of judge model candidates for fallback. */
  private getJudgeCandidates(): Array<{ provider: string; model: string }> {
    return [
      { provider: this.judgeProvider, model: this.judgeModel },
      // Fallback: local Ollama for judging when remote is down
      { provider: "ollama", model: "gemma4:latest" },
    ];
  }

  /** Parse judge JSON response — handles markdown fences, extra text. */
  private parseJudgeResponse(content: string): { score: number; note: string } | null {
    const cleaned = content.replace(/```(?:json)?\s*/gi, "").replace(/```/g, "").trim();

    try {
      const parsed = JSON.parse(cleaned);
      return this.validateParsed(parsed);
    } catch {
      // Fall through to regex
    }

    const jsonMatch = cleaned.match(/\{[\s\S]*?\}/);
    if (!jsonMatch) return null;

    try {
      const parsed = JSON.parse(jsonMatch[0]);
      return this.validateParsed(parsed);
    } catch {
      return null;
    }
  }

  private validateParsed(parsed: any): { score: number; note: string } | null {
    const score = Number(parsed?.score);
    if (!Number.isFinite(score) || score < 0 || score > 10) return null;
    return {
      score: Math.round(score) / 10, // Normalize to 0-1
      note: String(parsed?.note ?? parsed?.reason ?? "").slice(0, 200),
    };
  }

  /** Compute traffic-weighted composite score from probe results.
   *  Weights are derived from the actual intent distribution in routing_decisions.
   *  If no traffic data exists, uses equal weights (33% each). */
  private computeTrafficWeightedScore(probeResults: ProbeResult[]): number {
    const trafficDist = this.db.getTrafficDistribution(7); // Last 7 days

    // Map traffic intents to probe types and aggregate
    const probeWeights: Record<ProbeType, number> = {
      coding: 0,
      reasoning: 0,
      conversation: 0,
    };

    let totalTrafficWeight = 0;
    for (const [intent, proportion] of trafficDist) {
      const probeType = INTENT_TO_PROBE[intent];
      if (probeType) {
        probeWeights[probeType] += proportion;
        totalTrafficWeight += proportion;
      }
    }

    // If no traffic data, use equal weights
    if (totalTrafficWeight === 0) {
      probeWeights.coding = 1;
      probeWeights.reasoning = 1;
      probeWeights.conversation = 1;
      totalTrafficWeight = 3;
    }

    // Compute weighted average
    let weightedSum = 0;
    for (const result of probeResults) {
      const weight = probeWeights[result.probeType] / totalTrafficWeight;
      weightedSum += result.score * weight;
    }

    return weightedSum;
  }

  /** Detect model version hash.
   *  - Ollama: uses digest from /api/show (first 16 chars of sha256)
   *  - OpenRouter/ZAI/Gemini: uses model ID as version (changes when model ID changes)
   *  Falls back to "unknown" if detection fails. */
  private async getVersionHash(provider: string, model: string): Promise<string> {
    try {
      if (provider === "ollama") {
        const resp = await fetch(`${this.ollamaBaseUrl}/api/show`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model }),
          signal: AbortSignal.timeout(5000),
        });
        if (resp.ok) {
          const data = await resp.json() as any;
          // digest format: "sha256:abc123..." — use first 16 chars after prefix
          const digest = data.digest ?? data.details?.digest ?? "";
          if (digest) {
            return digest.replace(/^sha256:/, "").substring(0, 16);
          }
        }
      }
      // For remote providers, model ID itself is the version identifier
      return `${provider}/${model}`;
    } catch (err) {
      logger.debug(`Version hash detection failed for ${provider}/${model}: ${err}`);
      return "unknown";
    }
  }

  /** Apply benchmark results to the model registry, updating capability scores. */
  applyToRegistry(result: ChatBenchmarkResult): void {
    const cap = this.registry.getCapability(result.provider, result.model);
    if (!cap) return;

    // Map probe scores to capability dimensions
    for (const probe of result.probeResults) {
      switch (probe.probeType) {
        case "coding":
          // Don't overwrite if the seed score is already higher (seed may be from published benchmarks)
          cap.capabilities.coding = this.blend(cap.capabilities.coding, probe.score, cap.source);
          break;
        case "reasoning":
          cap.capabilities.reasoning = this.blend(cap.capabilities.reasoning, probe.score, cap.source);
          cap.capabilities.analysis = this.blend(cap.capabilities.analysis, probe.score * 0.98, cap.source);
          cap.capabilities.science = this.blend(cap.capabilities.science, probe.score * 0.95, cap.source);
          break;
        case "conversation":
          cap.capabilities.conversation = this.blend(cap.capabilities.conversation, probe.score, cap.source);
          cap.capabilities.creative = this.blend(cap.capabilities.creative, probe.score * 0.9, cap.source);
          cap.capabilities.summary = this.blend(cap.capabilities.summary, probe.score * 0.95, cap.source);
          break;
      }
    }

    // Mark source as auto-bench if it was inferred/benchmark, otherwise blended
    if (cap.source === "inferred" || cap.source === "auto-bench") {
      cap.source = "auto-bench";
    } else {
      cap.source = "blended";
    }

    logger.debug(
      `Applied benchmark to registry: ${result.modelId} ` +
      `coding=${cap.capabilities.coding.toFixed(2)} ` +
      `reasoning=${cap.capabilities.reasoning.toFixed(2)} ` +
      `conv=${cap.capabilities.conversation.toFixed(2)}`,
    );
  }

  /** Blend seed and observed scores.
   *  For inferred models: weight observed more heavily (they had no good baseline).
   *  For benchmark models: weight seed more heavily (published scores are reliable). */
  private blend(seed: number, observed: number, source: string): number {
    const alpha = source === "inferred" ? 0.6 : source === "auto-bench" ? 0.5 : 0.3;
    const blended = seed * (1 - alpha) + observed * alpha;
    return Math.max(0.1, Math.min(0.98, blended));
  }

  /** Check if a model needs re-benchmarking (version changed or results stale). */
  needsBenchmark(provider: string, model: string): boolean {
    const modelId = `${provider}/${model}`;
    const cached = this.db.getAllLatestChatBenchmarks(modelId);

    if (cached.size === 0) return true; // Never benchmarked

    // Check all 3 probe types exist
    const probeTypes: ProbeType[] = ["coding", "reasoning", "conversation"];
    for (const pt of probeTypes) {
      const entry = cached.get(pt);
      if (!entry) return true; // Missing probe type

      // Check age
      if (this.ageInDays(entry.timestamp) >= BENCHMARK_MAX_AGE_DAYS) return true;
    }

    // Note: we can't check version hash here without an async call.
    // The benchmarkModel method will check version hash and skip if unchanged.
    return false;
  }

  /** Age of a timestamp in days */
  private ageInDays(timestamp: string): number {
    const then = new Date(timestamp).getTime();
    return (Date.now() - then) / 86_400_000;
  }
}
