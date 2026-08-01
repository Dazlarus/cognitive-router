// src/judge.ts — LLM-as-judge quality evaluator
// Samples completed responses, asks a judge model to score them,
// and feeds normalized scores back into the capability registry.
//
// Design:
//   - Sampling: only evaluates a fraction of responses (default 10%)
//   - Async: runs AFTER the response is sent to the client (no latency impact)
//   - Resilient: judge failures are silently skipped
//   - Unbiased: skips judging if judge model == response model
//   - Persistent: scores saved to SQLite via ModelRegistry.updateCapability()

import { logger } from "./logger.js";
import { getProvider } from "./providers.js";

const JUDGE_PROMPT = `You are an impartial judge evaluating an AI assistant's response. Rate the overall quality on a scale of 0-10.

**Task category:** {intent}

**User request:**
{prompt}

**Assistant response:**
{response}

**Scoring guide:**
- 0-2: Wrong, broken, or nonsensical
- 3-4: Partially relevant but incomplete or inaccurate
- 5-6: Adequate — correct direction, usable but with gaps
- 7-8: Good — correct, clear, and complete
- 9-10: Excellent — exceptional quality and insight

Respond with ONLY a JSON object, no other text:
{{"score": <integer 0-10>, "note": "<one sentence explanation>"}}`;

export interface JudgeResult {
  /** Normalized score 0.0–1.0 */
  score: number;
  /** Raw judge score 0–10 */
  rawScore: number;
  /** Judge's brief explanation */
  note: string;
}

export class JudgeEvaluator {
  private sampleRate: number;
  private judgeProvider: string;
  private judgeModel: string;
  private minResponseLength: number;
  private maxChars: number;
  private timeoutMs: number;

  constructor() {
    this.sampleRate = parseFloat(process.env.ROUTER_JUDGE_SAMPLE_RATE ?? "0.10");
    this.judgeProvider = process.env.ROUTER_JUDGE_PROVIDER ?? "openrouter";
    this.judgeModel = process.env.ROUTER_JUDGE_MODEL ?? "qwen/qwen3-30b-a3b-instruct-2507";
    this.minResponseLength = parseInt(process.env.ROUTER_JUDGE_MIN_LENGTH ?? "50", 10);
    this.maxChars = parseInt(process.env.ROUTER_JUDGE_MAX_CHARS ?? "2000", 10);
    this.timeoutMs = parseInt(process.env.ROUTER_JUDGE_TIMEOUT_MS ?? "15000", 10);

    if (this.sampleRate > 0) {
      logger.info(
        `Judge evaluator active — sampling ${(this.sampleRate * 100).toFixed(0)}% of responses via ${this.judgeProvider}/${this.judgeModel}`,
      );
    } else {
      logger.info("Judge evaluator disabled (sample rate = 0)");
    }
  }

  /** Display string for the configured judge model (for audit logging). */
  get judgeModelId(): string {
    return `${this.judgeProvider}/${this.judgeModel}`;
  }

  /** Roll the dice — should we judge this response? */
  shouldJudge(): boolean {
    if (this.sampleRate <= 0) return false;
    return Math.random() < this.sampleRate;
  }

  /** Check if the judge model is different from the response model (avoid self-evaluation). */
  isSameModel(provider: string, model: string): boolean {
    return provider === this.judgeProvider && model === this.judgeModel;
  }

  /**
   * Evaluate a response asynchronously.
   * Returns null on any failure — callers should never await this for critical path.
   */
  async evaluate(
    prompt: string,
    response: string,
    intent: string,
  ): Promise<JudgeResult | null> {
    // Skip trivially short responses — not worth judging
    if (response.trim().length < this.minResponseLength) {
      return null;
    }

    const truncPrompt = prompt.slice(0, this.maxChars);
    const truncResponse = response.slice(0, this.maxChars);

    const filled = JUDGE_PROMPT
      .replace("{intent}", intent)
      .replace("{prompt}", truncPrompt)
      .replace("{response}", truncResponse);

    // Try judge candidates in priority order. Falls through on failure.
    const candidates = this.getJudgeCandidates();
    let lastError: Error | undefined;

    for (const { provider, model } of candidates) {
      const adapter = getProvider(provider);
      if (!adapter) continue;
      const apiKey = process.env[provider.toUpperCase() + "_API_KEY"] ?? "";
      if (!apiKey && provider !== "ollama") continue;

      try {
        const result = await adapter.chatCompletion(
          model,
          {
            model,
            messages: [{ role: "user", content: filled }],
            stream: false,
            temperature: 0.1,
            max_tokens: 100,
          },
          apiKey,
        );

        const content = result.choices?.[0]?.message?.content ?? "";
        const parsed = this.parseJudgeResponse(content);
        if (!parsed) {
          lastError = new Error(`unparseable: ${content.slice(0, 80)}`);
          continue;
        }

        logger.info(
          `Judge scored ${intent} response: ${parsed.score}/10 via ${provider}/${model} — ${parsed.note}`,
        );

        return {
          score: parsed.score / 10,
          rawScore: parsed.score,
          note: parsed.note,
        };
      } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err));
        logger.debug(`Judge candidate ${provider}/${model} failed: ${lastError.message}`);
        continue;
      }
    }

    logger.debug(`Judge: all candidates exhausted. Last error: ${lastError?.message}`);
    return null;
  }

  /** Ordered list of judge provider/model candidates for fallback. */
  private getJudgeCandidates(): Array<{ provider: string; model: string }> {
    return [
      { provider: this.judgeProvider, model: this.judgeModel },
      // Fallback: use local Ollama gemma4 for judging when remote is down
      { provider: "ollama", model: "gemma4:latest" },
    ];
  }

  /**
   * Parse the judge's JSON response.
   * Handles common formatting issues (markdown fences, extra text, etc.)
   */
  private parseJudgeResponse(content: string): { score: number; note: string } | null {
    // Strip markdown code fences if present
    const cleaned = content.replace(/```(?:json)?\s*/gi, "").replace(/```/g, "").trim();

    // Try direct JSON parse first
    try {
      const parsed = JSON.parse(cleaned);
      return this.validateParsed(parsed);
    } catch {
      // Fall through to regex extraction
    }

    // Extract first JSON object from the text
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
      score: Math.round(score),
      note: String(parsed?.note ?? parsed?.reason ?? "").slice(0, 200),
    };
  }
}
