// src/effort_profiles.ts - Per-model CONFIG CARDS + effort-level normalization
// (Effort-matrix Phase 2, design locked 2026-09-09: effort is a first-class
// identity axis; discovery owns the config card.)
//
// A config card captures the effort/thinking surface of a model in a
// provider-independent shape so the bench roster (Phase 3) can enumerate
// CONFIGURATIONS (model x effort) without caring which door (thinking toggle
// vs effort enum vs thinking budget) each provider exposes.
//
// Precedence contract: hand-maintained SEED cards > discovery-derived cards.
// Discovery fills gaps only (mergeConfigCard implements the field-level merge).

// ─── Normalized level enum ───
// "thinking-off"/"thinking-on" are only used on the thinking_on_off axis.
export type NormalizedEffortLevel =
  | "none" | "low" | "medium" | "high" | "max" | "ultra"
  | "thinking-off" | "thinking-on";

export type EffortAxis = "thinking_on_off" | "effort_ladder";

export interface EffortProfile {
  axis: EffortAxis;
  /** Normalized level names, ascending order (weakest -> strongest). */
  levels: NormalizedEffortLevel[];
  /** Provider default when the caller sends nothing. */
  defaultLevel: NormalizedEffortLevel;
}

export interface ConfigCard {
  /** Does the provider bill (or meter) prompt-cache hits differently? */
  cacheUsage: boolean;
  /** Provider "fast mode" (e.g. flash/flex tiers). pricing = human string. */
  fastMode: { supported: boolean; pricing?: string };
  /** Null when the model has no effort surface at all. */
  effortProfile: EffortProfile | null;
  /** "low" marks best-supported guesses from ambiguous provider docs. */
  confidence?: "high" | "medium" | "low";
  /** Where the card came from: hand-maintained seed or discovery pass. */
  source?: "seed" | "discovery";
}

// ─── Per-provider raw-name normalization ───
// Maps each provider's raw effort/thinking naming to the normalized enum.
// Unknown raws clamp to undefined (caller keeps or drops them).

const ZAI_RAW: Record<string, NormalizedEffortLevel> = {
  none: "none", low: "low", medium: "medium", high: "high",
  xhigh: "max", max: "max", ultra: "max",
  disabled: "thinking-off", "thinking-off": "thinking-off",
  enabled: "thinking-on", "thinking-on": "thinking-on",
};

const OPENROUTER_RAW: Record<string, NormalizedEffortLevel> = {
  none: "none", minimal: "none", low: "low", medium: "medium",
  high: "high", max: "max", xhigh: "ultra", ultra: "ultra",
};

const OPENAI_RAW: Record<string, NormalizedEffortLevel> = {
  none: "none", minimal: "low", low: "low", medium: "medium",
  high: "high", xhigh: "ultra", max: "max",
};

const ANTHROPIC_RAW: Record<string, NormalizedEffortLevel> = {
  low: "low", medium: "medium", high: "high", max: "max",
};

const GEMINI_RAW: Record<string, NormalizedEffortLevel> = {
  none: "none", low: "low", medium: "medium", high: "high",
  off: "thinking-off", disabled: "thinking-off",
};

const PROVIDER_TABLES: Record<string, Record<string, NormalizedEffortLevel>> = {
  zai: ZAI_RAW, openrouter: OPENROUTER_RAW, openai: OPENAI_RAW,
  anthropic: ANTHROPIC_RAW, gemini: GEMINI_RAW,
};

/** Normalize a provider-raw effort name to the shared enum. Unknown -> undefined. */
export function normalizeEffortLevel(provider: string, raw: string): NormalizedEffortLevel | undefined {
  const table = PROVIDER_TABLES[provider];
  if (!table) return undefined;
  return table[String(raw).toLowerCase()];
}

// ─── Seed cards (hand-maintained; highest precedence) ───
//
// Sources (verified 2026-09-09):
// - docs.z.ai/guides/capabilities/thinking-mode + docs.z.ai/guides/llm/glm-5.3
//   (migration notice): GLM-5.3/5.3-Flash are reasoning-always-on;
//   reasoning_effort accepts low|high|max. No disabled state.
// - github.com/zai-org/GLM-5: GLM-5.2 accepts only high|max for
//   reasoning_effort (max default) and can disable thinking outright.
// - GLM-5.1 / GLM-4.7: no explicit reasoning_effort doc found; the router's
//   live code path (providers.ts buildZaiThinkingFields) uses the
//   budget_tokens ladder low/medium/high with optional off -> marked
//   confidence "low".
// - glm-5-turbo / *-flash pool aliases remap server-side to glm-5.3-flash
//   (live spike 2026-09-07, response.model) -> same card as 5.3-flash.
// - OpenRouter z-ai mirrors: OpenRouter reasoning { effort } accepts
//   minimal|low|medium|high per openrouter docs; minimal clamps to none.

const glm53Card: ConfigCard = {
  cacheUsage: false,
  fastMode: { supported: false },
  effortProfile: { axis: "effort_ladder", levels: ["low", "high", "max"], defaultLevel: "high" },
  confidence: "high",
  source: "seed",
};

const glm52Card: ConfigCard = {
  cacheUsage: false,
  fastMode: { supported: false },
  effortProfile: { axis: "effort_ladder", levels: ["thinking-off", "high", "max"], defaultLevel: "max" },
  confidence: "high",
  source: "seed",
};

const budgetLadderCard: ConfigCard = {
  cacheUsage: false,
  fastMode: { supported: false },
  effortProfile: { axis: "effort_ladder", levels: ["thinking-off", "low", "medium", "high"], defaultLevel: "medium" },
  confidence: "low",
  source: "seed",
};

const glm47Card: ConfigCard = {
  cacheUsage: false,
  fastMode: { supported: false },
  effortProfile: { axis: "thinking_on_off", levels: ["thinking-off", "low", "medium", "high"], defaultLevel: "medium" },
  confidence: "medium",
  source: "seed",
};

const flashFast = (pricing?: string): ConfigCard => ({
  cacheUsage: false,
  fastMode: { supported: true, ...(pricing ? { pricing } : {}) },
  effortProfile: { axis: "effort_ladder", levels: ["low", "high", "max"], defaultLevel: "high" },
  confidence: "high",
  source: "seed",
});

export const SEED_CONFIG_CARDS: Record<string, ConfigCard> = {
  "zai/glm-5.3": glm53Card,
  "zai/glm-5.3-flash": flashFast(),
  "zai/glm-5-turbo": flashFast(), // pool alias -> 5.3-flash
  "zai/glm-5.2": glm52Card,
  // glm-5.1 OBSERVED LIVE (probe 2026-09-09, cogrouter-bench/tmp-probe-51.log):
  // accepts thinking {disabled|enabled|budget_tokens} AND a native
  // reasoning_effort low|medium|high ladder (reasoning channel scales 3->11->75
  // chars on a tiny probe); disabled serves zero reasoning. Card raised from
  // confidence "low" (docs guess) to "medium" (direct observation).
  "zai/glm-5.1": {
    cacheUsage: false,
    fastMode: { supported: false },
    effortProfile: { axis: "effort_ladder", levels: ["thinking-off", "low", "medium", "high"], defaultLevel: "medium" },
    confidence: "medium",
    source: "seed",
  },
  "zai/glm-4.7": glm47Card,
  "zai/glm-4.6v": {
    cacheUsage: false, fastMode: { supported: false },
    effortProfile: { axis: "thinking_on_off", levels: ["thinking-off", "low", "medium", "high"], defaultLevel: "medium" },
    confidence: "low", source: "seed",
  },
  // OpenRouter mirrors of the zai lineup (reasoning.effort door).
  "openrouter/z-ai/glm-5.3": { ...glm53Card, confidence: "medium" },
  "openrouter/z-ai/glm-5.3-flash": { ...flashFast(), confidence: "medium" },
  "openrouter/z-ai/glm-5-turbo": { ...flashFast(), confidence: "medium" },
  "openrouter/z-ai/glm-5.1": { ...budgetLadderCard, confidence: "low" }, // OR mirror unprobed; zai door card observed 2026-09-09
  "openrouter/z-ai/glm-4.7": { ...glm47Card, confidence: "low" },
};

/** Canonical (vendor-suffix-free) model id -> seed card fallback. Used by
 *  discovery to give UNSEEDED remote models a heuristic card. The result is
 *  always source:"discovery", confidence:"low" (a guess until proven). */
export function heuristicCardFor(provider: string, model: string, live?: {
  cacheReadPrice?: number;
  supportsReasoningEffort?: boolean;
}): ConfigCard | undefined {
  const lower = model.toLowerCase();
  // Canonical zai family detection works for zai + openrouter "z-ai/..." slugs.
  const canon = lower.replace(/^z-ai\//, "");
  const key = `${provider}/${model}`;
  if (SEED_CONFIG_CARDS[key]) return undefined; // seeded elsewhere

  const isFlash = /flash|fast|flex|mini/.test(canon);
  const flashify = (c: ConfigCard): ConfigCard =>
    isFlash && !c.fastMode.supported ? { ...c, fastMode: { supported: true } } : c;
  if (/(glm-5\.[3-9]|glm-5-turbo)/.test(canon)) {
    return flashify({ ...glm53Card, confidence: "low", source: "discovery" });
  }
  if (/glm-5\.2/.test(canon)) {
    return flashify({ ...glm52Card, confidence: "low", source: "discovery" });
  }
  if (/(glm-5(\.\d)?|^glm-5$)/.test(canon)) {
    return flashify({ ...budgetLadderCard, confidence: "low", source: "discovery" });
  }
  if (/glm-4\.7/.test(canon)) {
    return flashify({ ...glm47Card, confidence: "low", source: "discovery" });
  }
  if (/flash|fast|flex|mini|haiku|instant/.test(canon)) {
    return {
      cacheUsage: live?.cacheReadPrice !== undefined,
      fastMode: { supported: true },
      effortProfile: live?.supportsReasoningEffort
        ? { axis: "effort_ladder", levels: ["low", "medium", "high"], defaultLevel: "medium" }
        : null,
      confidence: "low", source: "discovery",
    };
  }
  if (live?.supportsReasoningEffort) {
    return {
      cacheUsage: live.cacheReadPrice !== undefined,
      fastMode: { supported: false },
      effortProfile: { axis: "effort_ladder", levels: ["low", "medium", "high"], defaultLevel: "medium" },
      confidence: "low", source: "discovery",
    };
  }
  return undefined;
}

// ─── Merge: seed wins, discovery fills gaps ───

/** Field-level merge. Every seed-defined field beats the discovered value;
 *  discovery only fills fields the seed left undefined. A seed effortProfile
 *  wins as a unit (axis/levels/default must stay coherent). */
export function mergeConfigCard(seed: ConfigCard | undefined, discovered: ConfigCard | undefined): ConfigCard | undefined {
  if (!seed) return discovered;
  if (!discovered) return seed;
  const merged: ConfigCard = {
    cacheUsage: seed.cacheUsage ?? discovered.cacheUsage,
    fastMode: seed.fastMode?.supported !== undefined || seed.fastMode?.pricing !== undefined
      ? seed.fastMode
      : (discovered.fastMode ?? { supported: false }),
    effortProfile: seed.effortProfile ?? discovered.effortProfile,
    confidence: seed.confidence ?? discovered.confidence,
    source: "seed",
  };
  return merged;
}
