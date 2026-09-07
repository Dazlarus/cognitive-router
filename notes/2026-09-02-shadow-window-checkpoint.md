# Shadow-Window Checkpoint — 2026-09-02 13:50 CT (7-day window complete)

Task: suggest-cogrouter-shadow-window-checkpoint-aug27 (Daz-directed run, 13:44 CT).
Window: 2026-08-26T17:10Z -> 2026-09-02T18:46Z (live at read). DB: data/cognitive-router.db.

## 1. Shadow decisions: 429 rows (predicted ~450)

- gate_arm: 429/429 = B (shadow-only; zero live gates armed - correct for shadow mode)
- rejection_reason: 427 low_confidence, 2 quarantine_canary
- Provider skew: 427/429 zai/glm-5.1 (shadow gate evaluates the served model; glm-5.1 is the workhorse), 1 glm-5.2, 1 openrouter/qwen3-30b
- Intent clusters: doc-summary 180, summary 72, math 44, research 13, creative 9, retrieval 2, science/conversation 1 each
- **would_be_value: NULL in all 429 rows** - the counterfactual value-delta the window was meant to capture never populated. Instrumentation gap, not accrual failure.

## 2. Routing since restart (week truth)

16,765 requests, all through the context pre-filter (context_filter_json populated):
- openrouter/deepseek-v4-flash: 9,429 (56% - the week's volume workhorse)
- zai/glm-5.1: 3,950 / glm-5.2: 2,771
- north-mini-code 426, qwen3-30b 117, gemma-4-26b 72

Today's /stats window (since 04:03 CT): 842 req, 832 tool_policy / 10 all_exhausted / 0 scored_pick. The 10 all_exhausted are consistent with pre-fix oversized-context moments (Tosha-class) before the 12:45 CT declared-window change.

## 3. Judge accrual since Aug 26: 448 rows, avg 6.70

- glm-5.1: 446 judged, avg 6.695, no_apply 427, gate_arm 0
- Movement: glm-5.1 coding +2.18, doc-summary +1.81 (improving); glm-5.2 doc-summary +1.08
- **Quarantine: zai/glm-5.1 intent=math, 7 occurrences** - passive safety caught a degrading pair on its own

## 4. VERDICT: extend window; do not arm gates; fix instrumentation first

- **No gate arming**: the evidence base is empty where it matters - would_be_value is NULL, so there is no measured counterfactual to arm anything on. Arming on low_confidence-event volume alone would be vibes-based gating.
- **ROUTER_UCB_K stays 0.02**: no signal suggesting re-tune; the Aug 25 offline tune stands.
- **Blocker to next checkpoint**: shadow gate writes must populate would_be_value (and pre_clamp) - likely the shadow-eval path computes the delta only in a later stage that never ran, or the column mapping is wrong. Investigate src/ (shadow decisions writer) before extending.
- Next checkpoint: after instrumentation fix + another 7-day accrual, or at Daz's call.

## Side observations

- The quarantine system (glm-5.1/math) proves the passive safety layer works independently of shadow gating.
- Week's traffic shape (deepseek-v4-flash 56%) vs today's (100% zai) - routing follows quota/health state; worth watching whether deepseek share returns post-weekend.
