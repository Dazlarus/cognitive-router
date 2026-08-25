# UCB k Tuning Log — 2026-08-25T05:59:40.378Z

Offline grid-tune of the exploration constant k (learned + k/√n) against the
full judge_history replay (`scripts/tune_ucb_k.ts`). No live tuning: the
selected k is a static env value (`ROUTER_UCB_K`).

- Eval count: 1132
- Method: replay EMA (alpha=0.25) per (intent, pair) in time order; at each
  event with ≥2 candidate pairs, compare argmax with vs without the bonus.
  Harmful flip = flipped-to model's observed mean score was lower than the
  baseline pick's at that moment.
- Selection rule: k=0 (no exploration) is the degenerate baseline — trivially
  zero harmful flips but it defeats the exploration guardrail, so it is
  excluded. Among k>0: minimize harmful flips; tie-break on beneficial flips.

| k | decisions | flips | harmful | beneficial | harm ratio |
|---|---|---|---|---|---|
| 0 | 923 | 0 | 0 | 0 | 0 |
| 0.02 | 923 | 36 | 1 | 35 | 0.028 |
| 0.05 | 923 | 65 | 2 | 63 | 0.031 |
| 0.08 | 923 | 144 | 8 | 136 | 0.056 |
| 0.1 | 923 | 151 | 10 | 141 | 0.066 |
| 0.15 | 923 | 182 | 12 | 170 | 0.066 |
| 0.2 | 923 | 217 | 21 | 196 | 0.097 |
| 0.3 | 923 | 267 | 38 | 229 | 0.142 |

**Selected k = 0.02** (harmful flips: 1, beneficial: 35).
Apply by setting `ROUTER_UCB_K=0.02` in the environment.
