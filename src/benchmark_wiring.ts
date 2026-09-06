// src/benchmark_wiring.ts - Production callers for the benchmark ladder
//
// Wires benchmark_ladder's ModelCaller/JudgeCaller interfaces to the real
// provider adapters:
//   - ModelCaller: resolves the endpoint serving an identity (caller supplies
//     the resolution strategy — v1: explicit map; discovery wiring will
//     supply cheapest-endpoint resolution), applies PINNED_DECODE, maps
//     effort onto reasoning_effort, resolves the API key, enforces a timeout.
//   - JudgeCaller: A/B comparison prompt, judge candidates with fallback
//     (mirrors judge.ts), family-wide recusal generalized to any family.
//
// Still unhooked from the proxy: discovery modes (auto/safe/manual) own the
// endpoint maps and trigger ladder insertion on new-model discovery.

import { logger } from "./logger.js";
import { getProvider } from "./providers.js";
import {
  PINNED_DECODE,
  type BenchModelKey,
  type JudgeCaller,
  type ModelCaller,
  type Verdict,
} from "./benchmark_ladder.js";

// ---------- endpoint resolution ----------

export interface BenchEndpoint {
  provider: string;
  model: string;
}

/** v1: explicit identity -> endpoint map. Discovery wiring replaces this with
 *  cheapest-endpoint resolution once the model registry tracks endpoints. */
export type EndpointResolver = (key: BenchModelKey) => BenchEndpoint | null;

/** Simple resolver over a static map (encoded key -> endpoint). */
export function mapResolver(
  map: Record<string, BenchEndpoint>,
): EndpointResolver {
  return (key) => map[`${key.model}|${key.quant}|${key.effort}`] ?? null;
}

// ---------- model caller ----------

const MODEL_CALL_TIMEOUT_MS = 120_000;

function effortToRequest(effort: string): Record<string, unknown> {
  // Minimal v1 mapping: pass reasoning_effort where providers accept it.
  // Provider-specific thinking fields (e.g. zai) get wired when discovery
  // owns endpoint metadata; the identity split itself is preserved regardless.
  switch (effort) {
    case "high":
      return { reasoning_effort: "high" };
    case "medium":
      return { reasoning_effort: "medium" };
    case "low":
      return { reasoning_effort: "low" };
    default:
      return {};
  }
}

export function makeModelCaller(resolve: EndpointResolver): ModelCaller {
  return async (key, prompt) => {
    const endpoint = resolve(key);
    if (!endpoint) {
      throw new Error(`no benchmark endpoint for ${key.model}|${key.quant}|${key.effort}`);
    }
    const adapter = getProvider(endpoint.provider);
    if (!adapter) {
      throw new Error(`no adapter for provider ${endpoint.provider}`);
    }
    const apiKey =
      process.env[endpoint.provider.toUpperCase() + "_API_KEY"] ?? "";
    if (!apiKey && endpoint.provider !== "ollama") {
      throw new Error(`no API key for provider ${endpoint.provider}`);
    }

    const result = await adapter.chatCompletion(
      endpoint.model,
      {
        model: endpoint.model,
        messages: [{ role: "user", content: prompt }],
        stream: false,
        temperature: PINNED_DECODE.temperature,
        max_tokens: PINNED_DECODE.max_tokens,
        ...effortToRequest(key.effort),
      } as any,
      apiKey,
      AbortSignal.timeout(MODEL_CALL_TIMEOUT_MS),
    );

    const content = result.choices?.[0]?.message?.content ?? "";
    if (!content.trim()) {
      throw new Error(
        `empty response from ${endpoint.provider}/${endpoint.model}`,
      );
    }
    return content;
  };
}

// ---------- family recusal ----------

/** Model family for recusal purposes. Order-insensitive, prefix-based. */
export function familyOf(provider: string, model: string): string {
  const m = model.toLowerCase();
  // OpenRouter vendor prefixes (google/gemini-..., meta-llama/..., qwen/...)
  const slash = m.indexOf("/");
  if (slash > 0) {
    const vendor = m.slice(0, slash);
    if (vendor === "google") return "gemini";
    return vendor;
  }
  // Bare model ids: match known family tokens
  const families = [
    "gemini", "gemma", "claude", "gpt", "o1", "o3", "o4",
    "glm", "deepseek", "qwen", "llama", "mistral", "mixtral",
    "kimi", "grok", "sonnet", "opus", "haiku", "nova",
  ];
  for (const f of families) {
    if (m.startsWith(f)) return f;
  }
  // Fall back to provider as the family (e.g. zai serving glm: provider
  // zai + glm model -> glm wins above; unknowns cluster by provider).
  return provider.toLowerCase();
}

/** Family-wide recusal check, generalized from judge.ts:
 *  judge family == candidate family (either side) -> recuse. */
export function judgeRecused(
  judgeProvider: string,
  judgeModel: string,
  candidate: { provider: string; model: string },
): boolean {
  const judgeFamily = familyOf(judgeProvider, judgeModel);
  return familyOf(candidate.provider, candidate.model) === judgeFamily;
}

// ---------- judge caller ----------

function getJudgeCandidates(): BenchEndpoint[] {
  // Bench judge is steerable independently of the live judge loop (Daz,
  // 2026-09-06: benchmark on ZAI until the system is proven). Falls back to
  // the live ROUTER_JUDGE_* values when bench-specific ones are unset.
  const provider =
    process.env.ROUTER_BENCH_JUDGE_PROVIDER ??
    process.env.ROUTER_JUDGE_PROVIDER ??
    "openrouter";
  const model =
    process.env.ROUTER_BENCH_JUDGE_MODEL ??
    process.env.ROUTER_JUDGE_MODEL ??
    "qwen/qwen3-30b-a3b-instruct-2507";
  const fallbackModel = process.env.ROUTER_JUDGE_FALLBACK_MODEL ?? "gemma4:latest";
  // Fallback lives on a separate endpoint family by construction (ollama).
  return [
    { provider, model },
    { provider: "ollama", model: fallbackModel },
  ];
}

const A_B_JUDGE_PROMPT = `You are an impartial judge comparing two responses (A and B) to the same prompt.

[PROMPT]
{prompt}
[END PROMPT]

[RESPONSE A]
{responseA}
[END RESPONSE A]

[RESPONSE B]
{responseB}
[END RESPONSE B]

Judge which response better fulfills the prompt: correctness, completeness,
clarity, and usefulness. A genuine tie is common and allowed — do not force a
winner when they are equally good or equally flawed.

Answer with EXACTLY one token on the first line: A, B, or TIE.`;

function parseAbVerdict(content: string): Verdict | null {
  const head = content.trim().slice(0, 200);
  const m = head.match(/\b(A|B|TIE)\b/i);
  if (!m) return null;
  const t = m[1].toUpperCase();
  return t === "A" ? "a" : t === "B" ? "b" : "tie";
}

export function makeJudgeCaller(): JudgeCaller {
  return async (prompt, responseA, responseB, keyA, keyB) => {
    const candidates = getJudgeCandidates();
    let lastError: Error | undefined;

    for (const { provider, model } of candidates) {
      // Family-wide recusal against BOTH candidates. Identity keys are
      // provider-agnostic; family derives from the model id alone.
      if (keyA && judgeRecused(provider, model, { provider: "", model: keyA.model })) continue;
      if (keyB && judgeRecused(provider, model, { provider: "", model: keyB.model })) continue;

      const adapter = getProvider(provider);
      if (!adapter) continue;
      const apiKey = process.env[provider.toUpperCase() + "_API_KEY"] ?? "";
      if (!apiKey && provider !== "ollama") continue;

      try {
        const filled = A_B_JUDGE_PROMPT
          .replace("{prompt}", prompt)
          .replace("{responseA}", responseA)
          .replace("{responseB}", responseB);

        const result = await adapter.chatCompletion(
          model,
          {
            model,
            messages: [{ role: "user", content: filled }],
            stream: false,
            temperature: 0.1,
            max_tokens: 100,
          } as any,
          apiKey,
        );

        const content = result.choices?.[0]?.message?.content ?? "";
        const verdict = parseAbVerdict(content);
        if (!verdict) {
          lastError = new Error(`unparseable verdict: ${content.slice(0, 80)}`);
          continue;
        }
        return verdict;
      } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err));
        logger.debug(
          `bench judge candidate ${provider}/${model} failed: ${lastError.message}`,
        );
      }
    }

    throw lastError ?? new Error("no judge candidates available");
  };
}
