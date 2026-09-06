// src/judge.ts — LLM-as-judge quality evaluator (Phase-1 hardened)
// Samples completed responses, asks a judge model to score them,
// and feeds normalized scores back into the capability registry.
//
// Design (LEARNING_LOOP_DESIGN.md §4.4):
//   - Sampling: only evaluates a fraction of responses (default 10%)
//   - Async: runs AFTER the response is sent to the client (no latency impact)
//   - Resilient: judge failures are silently skipped
//   - Unbiased: recusal when judge family == candidate family (Gemini-family aware)
//   - Persistent: scores saved to SQLite via ModelRegistry.applyJudgedScore()
//   - Ingestion ladder: Tier 1 single-shot ≤32k chars; Tier 2 multi-turn chunked
//     with manifest ("part N of M", verdict only after final chunk); Tier 3
//     head+tail with explicit "[N chars omitted]" markers. Truncation is always
//     MARKED, never silent. 3.5:1 chars-per-token heuristic (env-tunable).

import { logger } from "./logger.js";
import { getProvider } from "./providers.js";
import { pickIngestionTier, tier1MaxChars, tier2MaxChars, charsPerToken } from "./learning_guards.js";

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

/** Tier-2 final-chunk verdict prompt: structural assessment over the whole
 *  response (lost-in-the-middle mitigation — judge weighs the response as a
 *  whole rather than recalling middle chunks). */
const STRUCTURAL_VERDICT_PROMPT = `You have now received all {M} parts of the assistant's response. Weigh the response as a whole — its overall structure, completeness, and coherence — rather than relying on recall of any single middle part.

Rate the overall quality on a scale of 0-10.

**Task category:** {intent}
**User request (head, first 2,000 chars):**
{promptHead}

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
  /** Ingestion tier used (1|2|3). Any truncation is MARKED in the prompt. */
  tier: 1 | 2 | 3;
  /** True when the response was elided (tier 3) — feeds Arm A attribution. */
  truncated: boolean;
}

/** Chunk a string into n roughly-equal pieces. */
function chunkString(s: string, n: number): string[] {
  if (n <= 1) return [s];
  const size = Math.ceil(s.length / n);
  const out: string[] = [];
  for (let i = 0; i < s.length; i += size) out.push(s.slice(i, i + size));
  return out;
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
    // Tier-1 cap (default 32,000 chars ≈ 9k tokens at 3.5:1; env-tunable).
    this.maxChars = parseInt(process.env.ROUTER_JUDGE_MAX_CHARS ?? "32000", 10);
    this.timeoutMs = parseInt(process.env.ROUTER_JUDGE_TIMEOUT_MS ?? "15000", 10);

    if (this.sampleRate > 0) {
      logger.info(
        `Judge evaluator active — sampling ${(this.sampleRate * 100).toFixed(0)}% of responses via ${this.judgeProvider}/${this.judgeModel} ` +
        `(tier1≤${tier1MaxChars()} chars, tier2≤${tier2MaxChars()}, ${charsPerToken()}:1 chars/token)`,
      );
    } else {
      logger.info("Judge evaluator disabled (sample rate = 0)");
    }
  }

  /** Display string for the configured judge model (for audit logging). */
  get judgeModelId(): string {
    return `${this.judgeProvider}/${this.judgeModel}`;
  }

  /** Roll the dice - should we judge this response? Forced probes (exploration)
   *  always judge; the sample roll only governs normal traffic. */
  shouldJudge(force = false): boolean {
    if (force) return true;
    if (this.sampleRate <= 0) return false;
    return Math.random() < this.sampleRate;
  }

  /** Check if the judge model is different from the response model (avoid self-evaluation).
   *  Recusal is family-wide: a Gemini judge never grades Gemini candidates,
   *  an exact model match always recuses. */
  isSameModel(provider: string, model: string): boolean {
    if (provider === this.judgeProvider && model === this.judgeModel) return true;
    // Gemini-family recusal: gemini judge vs gemini provider candidates.
    if (this.judgeProvider === "gemini" && provider === "gemini") return true;
    // OpenRouter judges served under an OpenRouter candidate id of the same model.
    if (this.judgeProvider === "openrouter" && provider === "openrouter" && model === this.judgeModel) return true;
    // Cross-provider Gemini family (openrouter/google/* graded by gemini judge).
    if (this.judgeProvider === "gemini" && provider === "openrouter" && /^google\//i.test(model)) return true;
    return false;
  }

  /**
   * Evaluate a response asynchronously via the ingestion ladder.
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

    const totalChars = prompt.length + response.length;
    const tier = pickIngestionTier(totalChars);
    try {
      if (tier === 1) return await this.evaluateTier1(prompt, response, intent);
      if (tier === 2) return await this.evaluateTier2(prompt, response, intent);
      return await this.evaluateTier3(prompt, response, intent);
    } catch (err) {
      logger.debug(`Judge evaluation failed (tier ${tier}): ${err instanceof Error ? err.message : err}`);
      return null;
    }
  }

  /** Tier 1 — fits one request: single-shot verdict (default path). */
  private async evaluateTier1(prompt: string, response: string, intent: string): Promise<JudgeResult | null> {
    const filled = JUDGE_PROMPT
      .replace("{intent}", intent)
      .replace("{prompt}", prompt.slice(0, this.maxChars))
      .replace("{response}", response.slice(0, this.maxChars));
    const parsed = await this.askJudge([{ role: "user", content: filled }]);
    if (!parsed) return null;
    return {
      score: parsed.score / 10,
      rawScore: parsed.score,
      note: parsed.note,
      tier: 1,
      truncated: false,
    };
  }

  /** Tier 2 — exceeds per-request limit, fits context window: multi-turn
   *  chunked accumulation. Chunks sent as continuing turns with a manifest
   *  ("that was part N of M"); verdict requested only after the final chunk
   *  via the structural-assessment prompt. Cost O(n²) in chunks — pennies at
   *  sampled volume. */
  private async evaluateTier2(prompt: string, response: string, intent: string): Promise<JudgeResult | null> {
    const budget = tier1MaxChars();
    const promptBudget = Math.min(prompt.length, Math.floor(budget * 0.25));
    const responseBudget = budget - promptBudget;

    const promptHead = prompt.slice(0, promptBudget) +
      (prompt.length > promptBudget ? `\n[${prompt.length - promptBudget} chars omitted from request]` : "");
    const respFull = response.slice(0, tier2MaxChars());
    const elidedResp = response.length > tier2MaxChars();
    const markedResponse = elidedResp
      ? respFull + `\n[${response.length - tier2MaxChars()} chars omitted]`
      : respFull;

    // Chunk the response to fit each turn under the per-request budget.
    const perChunk = Math.max(1000, Math.floor((budget - promptHead.length - 600) ));
    const chunks = chunkString(markedResponse, Math.ceil(markedResponse.length / perChunk));
    const messages: Array<{ role: string; content: string }> = [
      {
        role: "user",
        content:
          `You are an impartial judge. Over the next ${chunks.length} messages I will send you the assistant's response in parts. ` +
          `Task category: ${intent}. Do not evaluate yet — wait for all parts and the final verdict request.\n\n` +
          `**User request (head):**\n${promptHead}`,
      },
    ];
    for (let i = 0; i < chunks.length; i++) {
      messages.push({
        role: "user",
        content: `[part ${i + 1} of ${chunks.length}]\n${chunks[i]}`,
      });
    }
    messages.push({
      role: "user",
      content: STRUCTURAL_VERDICT_PROMPT
        .replace("{M}", String(chunks.length))
        .replace("{intent}", intent)
        .replace("{promptHead}", promptHead),
    });

    const parsed = await this.askJudge(messages, 200);
    if (!parsed) return null;
    return {
      score: parsed.score / 10,
      rawScore: parsed.score,
      note: parsed.note,
      tier: 2,
      truncated: elidedResp,
    };
  }

  /** Tier 3 — exceeds the window entirely: head+tail sampling with explicit
   *  elision markers; the judge is told what it didn't see. */
  private async evaluateTier3(prompt: string, response: string, intent: string): Promise<JudgeResult | null> {
    const budget = tier1MaxChars();
    const promptHead = prompt.slice(0, 2000) + (prompt.length > 2000 ? `\n[${prompt.length - 2000} chars omitted from request]` : "");
    const half = Math.floor((budget - promptHead.length - 800) / 2);
    const head = response.slice(0, half);
    const tail = response.slice(response.length - half);
    const omitted = response.length - head.length - tail.length;
    const marked = `${head}\n[${omitted} chars omitted — middle section not shown]\n${tail}`;

    const filled = JUDGE_PROMPT
      .replace("{intent}", intent)
      .replace("{prompt}", promptHead)
      .replace("{response}", marked);
    const parsed = await this.askJudge([{ role: "user", content: filled }]);
    if (!parsed) return null;
    return {
      score: parsed.score / 10,
      rawScore: parsed.score,
      note: parsed.note,
      tier: 3,
      truncated: true,
    };
  }

  /** Try judge candidates in priority order. Falls through on failure. */
  private async askJudge(
    messages: Array<{ role: string; content: string }>,
    maxTokens = 100,
  ): Promise<{ score: number; note: string } | null> {
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
            messages: messages as any,
            stream: false,
            temperature: 0.1,
            max_tokens: maxTokens,
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
          `Judge scored response: ${parsed.score}/10 via ${provider}/${model} — ${parsed.note}`,
        );

        return parsed;
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
