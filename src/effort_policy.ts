// src/effort_policy.ts - Outbound (router-chosen) effort policy
//
// Two different "effort" concepts meet at the router:
//  1. Client → router effort (OpenClaw /reasoning, spawn thinking params):
//     a routing-table weight lever only (router.ts).
//  2. Router → provider effort: the ROUTER'S OWN purchase decision. More
//     effort = more tokens = more cost, in exchange for deeper and more
//     thorough results. The router owns this trade because it is the only
//     component that sees intent, context size, and budget state together.
//
// The client's explicit lever still bounds the choice (a "high" floors it,
// an explicit "low" caps it) — it just no longer *is* the outbound value.
//
// ROUTER_EFFORT_MODE=passthrough restores pre-2026-09-05 pass-through.

import { logger } from "./logger.js";

export type EffortLevel = "low" | "medium" | "high";
export type EffortDecision = { level: EffortLevel; trace: string[] };

const LEVEL_ORDER: EffortLevel[] = ["low", "medium", "high"];

function step(level: EffortLevel, delta: number): EffortLevel {
  const i = LEVEL_ORDER.indexOf(level) + delta;
  return LEVEL_ORDER[Math.max(0, Math.min(LEVEL_ORDER.length - 1, i))];
}

// Reasoning-heavy intents earn a medium base; light intents ride low.
// The floor is "low" (never "none") — reasoning-native models (GLM-5.x)
// expect a thinking budget, and a sliver of reasoning is cheap insurance
// even for retrieval.
const HEAVY_INTENTS = new Set([
  "coding", "research", "science", "math", "analysis", "creative", "business",
]);

export function effortPolicyMode(): "auto" | "passthrough" {
  return process.env.ROUTER_EFFORT_MODE === "passthrough" ? "passthrough" : "auto";
}

export function decideOutboundEffort(input: {
  intent: string;
  /** Explicit client lever: "high" floors, "medium" pins, "low" caps. "none"/absent = router free. */
  clientHint?: string | null;
  speedMode?: string | null;
  quotaMultiplier?: number | null;
  budgetExceeded?: boolean;
}): EffortDecision {
  const trace: string[] = [];
  let level: EffortLevel = HEAVY_INTENTS.has(input.intent) ? "medium" : "low";
  trace.push(`base(${input.intent})=${level}`);

  // Budget pressure: buy less thinking when the wallet is thin.
  if ((input.quotaMultiplier ?? 1) > 1 || input.budgetExceeded) {
    const next = step(level, -1);
    if (next !== level) trace.push(`budget→${next}`);
    level = next;
  }

  // Fast lane caps effort — speed and depth fight each other.
  if (input.speedMode === "fast" && level !== "low") {
    trace.push("speed=fast→low");
    level = "low";
  }

  // Explicit client lever bounds the router's choice.
  const hint = input.clientHint && input.clientHint !== "none" ? input.clientHint : null;
  if (hint === "high" && level !== "high") {
    level = "high";
    trace.push("hint-high→high");
  } else if (hint === "medium" && level !== "medium") {
    level = "medium";
    trace.push("hint-medium→medium");
  } else if (hint === "low" && level !== "low") {
    level = "low";
    trace.push("hint-low→low");
  }

  return { level, trace };
}

logger.info(
  `Effort policy: ${effortPolicyMode()} — outbound thinking is ` +
  (effortPolicyMode() === "auto"
    ? "router-chosen (client lever clamps only)"
    : "pass-through of client params"),
);
