// src/hedged_request.ts — Hedged request racing for transient failure recovery
//
// When the primary provider returns 429/503, we race two concurrent requests:
//   1. Fallback provider (fires immediately)
//   2. Primary provider retry (fires after configurable delay)
// First successful response wins; the loser is cancelled via AbortController.
//
// Guarantees:
// - At most one response is returned (the winner).
// - The losing request's AbortController is aborted.
// - If both fail, rejects with a descriptive error.

import { logger } from "./logger.js";
import type { ProviderAdapter, ChatCompletionRequest, ChatCompletionResponse } from "./providers.js";

// ─── Types ───

export type HedgeResultKind = "primary_win" | "fallback_win" | "both_fail";

export interface HedgeOutcome {
  result: HedgeResultKind;
  primaryProvider: string;
  primaryModel: string;
  fallbackProvider: string;
  fallbackModel: string;
  winnerProvider: string;
  winnerModel: string;
  winnerDurationMs: number;
  loserCancelled: boolean;
}

export interface HedgeCandidate {
  provider: string;
  model: string;
  adapter: ProviderAdapter;
  apiKey: string;
}

export interface HedgeSuccess {
  response: ChatCompletionResponse;
  outcome: HedgeOutcome;
}

// ─── Helpers ───

/**
 * Get a random delay for the hedged primary retry.
 * Configurable via HEDGE_RETRY_MIN_MS (default 5000) and HEDGE_RETRY_MAX_MS (default 10000).
 */
export function hedgeRetryDelayMs(): number {
  const min = parseInt(process.env.HEDGE_RETRY_MIN_MS ?? "5000", 10);
  const max = parseInt(process.env.HEDGE_RETRY_MAX_MS ?? "10000", 10);
  return min + Math.floor(Math.random() * Math.max(1, max - min));
}

/**
 * Race two provider requests against each other. First successful response wins;
 * the loser is cancelled via AbortController.
 *
 * The fallback fires immediately; the primary retry fires after `retryDelayMs`.
 *
 * Throws if BOTH requests fail.
 */
export async function raceHedgedRequests(
  primary: HedgeCandidate,
  fallback: HedgeCandidate,
  request: ChatCompletionRequest,
  retryDelayMs: number,
): Promise<HedgeSuccess> {
  const primaryController = new AbortController();
  const fallbackController = new AbortController();

  const hedgeStart = Date.now();

  let primaryDone = false;
  let fallbackDone = false;
  let primaryError: Error | null = null;
  let fallbackError: Error | null = null;
  let primaryDurationMs = 0;
  let fallbackDurationMs = 0;

  /**
   * Fire a provider call after an optional delay. Resolves with the response
   * or rejects on error/abort.
   */
  async function fireRequest(
    candidate: HedgeCandidate,
    controller: AbortController,
    delayMs: number,
  ): Promise<{ response: ChatCompletionResponse; durationMs: number }> {
    const callStart = Date.now();

    // Wait for delay (if any) before firing, unless already cancelled
    if (delayMs > 0) {
      await new Promise<void>((resolve) => {
        const timer = setTimeout(resolve, delayMs);
        controller.signal.addEventListener(
          "abort",
          () => {
            clearTimeout(timer);
            resolve();
          },
          { once: true },
        );
      });

      if (controller.signal.aborted) {
        throw new Error("aborted_before_start");
      }
    }

    const providerRequest = { ...request, model: candidate.model, stream: false };
    const response = await candidate.adapter.chatCompletion(
      candidate.model,
      providerRequest,
      candidate.apiKey,
      controller.signal,
    );
    return { response, durationMs: Date.now() - callStart };
  }

  const primaryPromise = fireRequest(primary, primaryController, retryDelayMs);
  const fallbackPromise = fireRequest(fallback, fallbackController, 0);

  return new Promise<HedgeSuccess>((resolve, reject) => {
    const onPrimaryWin = ({ response, durationMs }: { response: ChatCompletionResponse; durationMs: number }) => {
      primaryDone = true;
      primaryDurationMs = durationMs;
      if (!fallbackDone) {
        fallbackController.abort();
        logger.info(
          `🏁 Hedge: PRIMARY retry won (${primary.provider}/${primary.model}) in ${durationMs}ms — fallback cancelled`,
        );
      }
      resolve({
        response,
        outcome: {
          result: "primary_win",
          primaryProvider: primary.provider,
          primaryModel: primary.model,
          fallbackProvider: fallback.provider,
          fallbackModel: fallback.model,
          winnerProvider: primary.provider,
          winnerModel: primary.model,
          winnerDurationMs: durationMs,
          loserCancelled: !fallbackDone,
        },
      });
    };

    const onPrimaryFail = (err: Error) => {
      primaryDone = true;
      primaryError = err;
      primaryDurationMs = Date.now() - hedgeStart;
      if (fallbackDone && fallbackError) {
        reject(
          new Error(
            `Both hedged requests failed — primary: ${primaryError.message}; fallback: ${fallbackError.message}`,
          ),
        );
      }
    };

    const onFallbackWin = ({ response, durationMs }: { response: ChatCompletionResponse; durationMs: number }) => {
      fallbackDone = true;
      fallbackDurationMs = durationMs;
      if (!primaryDone) {
        primaryController.abort();
        logger.info(
          `🏁 Hedge: FALLBACK won (${fallback.provider}/${fallback.model}) in ${durationMs}ms — primary retry cancelled`,
        );
      }
      resolve({
        response,
        outcome: {
          result: "fallback_win",
          primaryProvider: primary.provider,
          primaryModel: primary.model,
          fallbackProvider: fallback.provider,
          fallbackModel: fallback.model,
          winnerProvider: fallback.provider,
          winnerModel: fallback.model,
          winnerDurationMs: durationMs,
          loserCancelled: !primaryDone,
        },
      });
    };

    const onFallbackFail = (err: Error) => {
      fallbackDone = true;
      fallbackError = err;
      fallbackDurationMs = Date.now() - hedgeStart;
      if (primaryDone && primaryError) {
        reject(
          new Error(
            `Both hedged requests failed — primary: ${primaryError.message}; fallback: ${fallbackError.message}`,
          ),
        );
      }
    };

    primaryPromise.then(onPrimaryWin, onPrimaryFail);
    fallbackPromise.then(onFallbackWin, onFallbackFail);
  });
}
