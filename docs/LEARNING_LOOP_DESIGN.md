# Learning Loop Design — Hardening a Live System

**Status:** Draft for Daz review — 2026-08-24
**Companion thread state:** `workspace/memory/cognitive-router-learning-state.md` (settled decisions, timestamps)
**Supersedes:** the July 2026 audit conclusion that the learning loop was dead code. It is live.

---

## 1. Context

The Cognitive Router routes all non-overridden agent traffic (subagents, cron, dashboards) using
`overall = capability*0.50 + reliability*0.25 + cost*0.15 + latency*0.10`, with intent classifier
confidence multiplying capability at request time. Capability scores were seeded from benchmarks
(`SEED_MODELS`) and assumed static.

**That assumption is outdated.** Commit `b40f82c` (Jul 27, "negative signal amplification") wired an
LLM-as-judge loop; `4f8c9f2` (Aug 2) extended it. The loop has been live ~4 weeks:

- `JudgeEvaluator` (`src/judge.ts`): samples ~10% of completed responses, judges async post-send,
  persists to SQLite, self-recusal when judge model == response model.
- `ModelRegistry.updateCapability()` (`src/model_registry.ts:464`): EMA, alpha=0.25, persists to
  `capability_overrides` with `sample_count` + `last_judged`.
- Call sites: `proxy-stream.ts:1090` and `:1178`.
- **31 learned capability rows exist. Learned values are live in routing today** (verified: the
  poisoned math scores appear in `/stats` intentScores).

## 2. Autopsy — what the data shows (2026-08-24)

From `judge_history` (1,103 evals, Jul 27 → present, ~25–66/day):

### Working
- Async sampling pipeline functions; judge notes are coherent and specific.
- Plausible separation on well-represented intents: summary ~0.78, doc-summary ~0.78, coding ~0.49
  (glm-5.1, n=201) — scores respond to real quality differences.
- Persistence with sample counts works (UCB's `n` is already tracked).

### Broken or missing
1. **No clamps.** glm-5.1 math capability = 0.00002 (n=37, all judged 0). glm-5.2 math = 0.09.
   A floored model is never routed, never generates evidence, never recovers.
2. **No attribution gate — and the top failure mode is systemic.** The intent classifier tags
   watchdog/system status checks as `math` (timestamps, percentages, numbers everywhere). The model
   correctly answers the actual status question; the judge, told the task was math, correctly dings
   "irrelevant — appears to be a system status log." 37 identical punishments for a classifier error
   the model had no part in. **Classification errors are being laundered into capability poison.**
3. **Judge pool adjacency.** Judge = `openrouter/qwen/qwen3-30b-a3b-instruct-2507` — which is also
   the router's OpenRouter tool *fallback* model. Violates the no-competing-traffic guardrail we
   settled on Aug 24.
4. **Absolute 0–10 single-judge scoring.** Judges are demonstrably more stable at pairwise
   comparison than absolute rating; absolute scales invite drift.
5. **2000-char truncation** of responses before judging — biases long technical/math outputs toward
   "incomplete."
6. **No decay, no exploration, no rate-of-change cap, no pins.** Scores fossilize at whatever the
   EMA converged to; new/recovering models have no path back into rotation.
7. **Bimodal residue:** 111 zero-scores and 374 eight-scores — the zeros are substantially the
   misclassification cluster, not model failures.

## 3. Settled decisions (Daz-approved Aug 24)

1. Both signal types: implicit sorter + LLM-judge. **Phase 1 = harden live loop; phase 2 = deepen
   judge.** Phase boundary = git rollback point. (Reality inverted the original implicit-first plan —
   the judge already shipped. We harden what runs.)
2. User regen/hard-correction: **logged as judge input context, never direct implicit capability
   updates** ("bad, reason unknown" is judge food, not capability food).
3. Capability learned per intent; strictly separate from reliability.
4. Reliability is temporal, dual-timescale: episodic (existing backoff/failure tiers) + persistent
   (slow long-window provider character). Restart persistence stores events with timestamps; loaded
   state ages toward long-term prior by age. State carries its birthday.
5. Update semantics: EMA (keep alpha=0.25), per-model sample counts, decay-to-prior on idle.
6. Guardrails: hard clamps [0.05, 0.98]; UCB exploration bonus k/√n; attribution gate (three-arm —
   see §4.1); rate-of-change cap per day/week; decay-to-prior after idle; manual pins; judge QC.
7. Everything persists (SQLite already in place).
8. Manual reseed of poisoned/stale rows (fold into phase 1).
9. Judge: **Gemini 3.5 Flash** primary (Daz-approved; ~$2–3/mo at volume; recusal rule = never
   grades Gemini-family candidates). DeepSeek V4 Flash documented as known-good alternate.
   Rejected: Claude (cost), local 12–14B judges (position/length bias), GLM-5.1 (family conflict
   with primary routed pool).

## 4. Phase 1 — Stabilize the live loop

### 4.1 Attribution gate (the centerpiece)
Before any judged score reaches `updateCapability()`, classify the observation into three arms:

- **Arm A — provider-attributable fault** (5xx, auth, timeout, gateway truncation): feed *reliability*
  trackers only. Never capability.
- **Arm B — low classifier confidence** (intent confidence below threshold): log to `judge_history`
  with a `no_apply` flag, apply nothing. Classification error ≠ model error.
  **Spike result (Aug 24):** pulled actual `routing_decisions` confidence for the poisoned math-intent
  days — all 788 sampled decisions sit in the 0.50–0.70 band, none above 0.70. So the misclassification
  cluster is **confidently-wrong-resistant to a single high threshold but separable**: a 0.75 confidence
  threshold would have caught ~100% of the poisoned cluster in the sample. Spike action: set the arm-B
  threshold at 0.75 initially, verify against the full history in implementation, and keep the second
  trigger below regardless — one good day of data is not a policy.
- **Arm C — model-attributable**: apply to capability through the guardrails below.

Additionally: **two quarantine triggers, pre-application and post.** Pre-application: judge-note
similarity check against recent notes for the same model/intent pair (near-identical verdict → hold,
inspect, don't apply — catches the confident-misclassifier on punishment #2 instead of #5).
Post-application: same note ≥5 times historically quarantines the pair and raises an alert.
Repeated identical verdicts mean the *task mix* is broken, not the model.

### 4.2 Guardrails (all land in phase 1)
- **Clamps** [0.05, 0.98] applied at write time in `updateCapability()`.
- **Rate-of-change cap:** a capability cell moves ≤ Δmax per 24h/7d regardless of sample volume.
- **Decay-to-prior:** no samples for N days → drift learned value toward benchmark seed; also
  catches silent provider model swaps.
- **Exploration bonus:** routing reads `learned + k/√n` (n = existing `sample_count`). **k is
  grid-tuned offline against the 1,103 historical `judge_history` evals** — replay each eval under
candidate k values, measure aggregate route flips and whether flipped routes would have scored better
  (we know the judged scores), pick k minimizing harmful flips. No live-traffic tuning.
- **Manual pins:** operator-set rows the learner refuses to overwrite (`capability_overrides` gains
  a `pinned` flag).

### 4.3 Data repair (reseed)
- **Reset to seed:** poisoned intent cells — glm-5.1 math, glm-5.2 math (both misclassification
  casualties).
- **Keep:** healthy learned rows with real n (summary, doc-summary, coding).
- **Reseed from fresh benchmarks:** the five generic-0.5 OpenRouter "inferred" placeholders.
- Archive pre-reseed `capability_overrides` to a backup table first (reversibility).

### 4.4 Judge swap + ingestion ladder
`ROUTER_JUDGE_PROVIDER=gemini`, `ROUTER_JUDGE_MODEL=gemini-3.5-flash` (env-only change; judge is
already slot-based). Self-recusal rule extended to Gemini-family candidates.

**Ingestion ladder** (replaces the 2,000-char silent truncation):

- **Tier 1 — fits one request (default):** single-shot verdict, cap raised 2,000 → 32,000 chars
  (~8k tokens). Covers effectively all sampled traffic at flash pricing.
- **Tier 2 — exceeds per-request limit, fits context window:** **multi-turn chunked accumulation.**
  Chunks sent as continuing turns with a manifest ("that was part 2 of 5"); verdict requested only
  after the final chunk — the judge never scores what it believes may be a partial answer. Cost is
  O(n²) in chunks (history re-sent per turn) — pennies at sampled volume, and it keeps the judge slot
  swappable: a smaller-per-request-cap judge degrades gracefully instead of breaking.
- **Tier 3 — exceeds the window entirely:** head+tail sampling with explicit elision markers
  (`[N chars omitted]`), judge told what it didn't see.
- Any truncation, in any tier, is **marked, never silent**.
- Long-context quality note: after tier-2 ingestion the verdict prompt asks for structural assessment
  ("weigh the response as a whole") rather than relying on recall of middle chunks (lost-in-the-middle
  attention degradation is real).

**Coding-bias check:** glm-5.1 coding (avg 5.3, n=201) was judged under head-only 2k truncation —
suspect, not proven poisoned. The shadow window doubles as the detector: if de-truncated judging
systematically scores coding above the learned value, reset-and-relearn those rows via the same
archive-table path. Verify before surgery.

### 4.5 Shadow window
Guardrailed learner runs **log-only** (`apply=false`) for a divergence window — it records the score
it *would* apply; we diff against live behavior.

**Acceptance criteria (go/no-go gates, not vibes):**
1. **Misclassification replay:** the fixed 37-turn status-log cluster is replayed through the new
   gate — required outcome: 100% no-apply (zero capability applications from misclassified turns).
2. **Healthy-row stability:** zero unexpected quarantine alerts on non-reseeded cells (summary,
   doc-summary, coding rows with real n).
3. **Bounded drift:** non-reseeded cells' would-be values move within the RoC cap envelope; any
   cell exceeding it without a genuine quality event = investigate before flip.
4. **Coding detector:** coding-intent would-be scores tracked specifically for the truncation-bias
   divergence test (§4.4).

**Open question: window length (Karl proposes 1 week; minimum 3 days of traffic given judge volume
~25–66/day).** After gates pass, flip `apply=true`. A failed gate extends the window and blocks the
flip — gates, not calendar, decide.

### 4.6 Reliability reload with age (the "birthday" rule — capability's counterpart)
Dual-timescale reliability (settled decision #4) has a phase-1 floor even though the full dual-track
builds out in phase 2: **on startup, `provider_health` loads age-weighted** — episodic state
(backoff tier, consecutive failures) resets or heavily decays on restart (a restart is not evidence
of provider recovery, but stale episodic state is not evidence of ongoing congestion either);
persistent state (long-window failure rates) loads intact but ages toward its long-term prior by
time-since-recorded. Concretely: reload records `last_updated` on read; episodic counters decay by
half-life 1h, persistent rates by half-life 7d. The full temporal-prior machinery (events with
timestamps, congestion attribution) lands in phase 2 with the reliability tracker buildout.

## 5. Phase 2 — Deepen the signal

- **Pairwise verdicts** replace absolute 0–10 for capability updates (judge compares two anonymized
  responses to the same prompt; absolute scores remain in `judge_history` for trend visibility).
- **Frozen calibration set** (~50 curated pairs with known preference) — run on judge swap and
  periodically; catches judge drift and quantifies any new judge before it touches live scores.
- **User-behavior signals:** regen/hard-correction events logged with the turn's request hash;
  surfaced to the judge as context ("user rejected this response"), never applied directly.
- **Implicit expansion:** worker/subagent task outcomes (completed/failed/tests-passed) as
  intent-labeled capability evidence — richer than HTTP success and already flowing through the
  router.
- Optional 2.5: tiered judging (cheap judge always; escalate low-confidence/ties to frontier model).
- **Phase-3 thinking, explicitly out of scope now:** executable verification for math-intent turns
  (compute the answer, compare — no LLM judgment at all). Kept out of phase 1 to prevent scope creep
  in a hardening pass; revisit once the hardened loop stabilizes.

## 6. Rejected alternatives (do not relitigate without new evidence)

| Option | Why rejected |
|---|---|
| Claude-class judge | Cost (~an order of magnitude over flash tier at equal volume); Daz benchmarks against it rather than runs it |
| Local 12–14B judge | Position/length bias poisons capability scores; calibration might catch it but we'd pay in score noise |
| GLM-5.1 as judge | GLM is the primary routed family; recusal would exclude it from most meaningful pairs |
| All-traffic learning plugin | High effort, uncertain payoff (Daz, Aug 24) — router already sees subagent/cron traffic |
| Greenfield rewrite | The live loop's failure modes are enumerable and patchable; hardening preserves 4 weeks of judge history |

## 7. Verification

- Unit: clamps, RoC cap, decay math, attribution gate arm routing (incl. quarantine canary),
  pin immutability.
- Integration: synthetic poisoned history (replay the 37× status-log pattern) → assert no capability
  application and canary alert raised.
- Post-deploy standing checks (weekly): poisoned-cell scan (`capability < 0.05`), judge-note
  repetition rate, shadow-window divergence report, calibration drift.

## 8. Rollback points

- Phase 1 = one branch/PR; shadow mode means reverting is `apply` flag + judge env vars, no data
  loss (judge_history is append-only).
- Pre-reseed archive table (§4.3) restores all learned values if the reseed proves wrong.
