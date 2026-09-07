# Switchyard Research — What CogRouter Can Borrow

**Date:** 2026-08-25 · **Source:** https://github.com/NVIDIA-NeMo/Switchyard (NVIDIA-NeMo, Apache-2.0, Rust, pre-alpha v0.2.0)
**Trigger:** Daz, Aug 25 — "things here CogRouter could use/learn from"

## What Switchyard Is

Rust proxy + library for LLM traffic: routes across providers, translates between
OpenAI Chat / Anthropic Messages / OpenAI Responses APIs, records Prometheus
metrics, ships composable routing algorithms. Four route types: passthrough,
random (A/B), llm_classifier, stage_router (+ escalation mode).

**Not a dependency candidate** — different stack (Rust vs TS), pre-alpha, and
CogRouter already owns the proxy slot in our stack. It's an **ideas donor**.

## Where CogRouter Is Already Ahead

- Per-request **intent classification** by content (10 categories, embeddings,
  no LLM call). Switchyard never classifies by task domain — only stage.
- **Learned** routing: capability scores + judge history + UCB bandit with
  shadow-gated read path. Switchyard's algorithms are fixed, no learning.
- Multi-factor scoring with cost/latency axes.
- Circuit breakers per-provider AND per-model with 5-tier exponential backoff.

## Borrowable Ideas (ranked)

### 1. Signal-driven stage routing — the big one
Switchyard routes by **where the agent is in its run**, not what the task is:
- WRONG→capable signals: windowed error severity, spinning (churn, no
  reads/writes), exploring (reading/planning without producing)
- PROGRESS→efficient: recent_production_intensity (writes/edits landing)
- Corroborative scoring: signed tanh-squash to [0,1]; one full signal ≈ 0.46,
  so escalation past 0.5 threshold takes ~1.5 signals of agreement;
  critical-error severity = hard override.
- Decision cascade: signal ≥ threshold → route by signal; else classifier
  (optional); else picker default tier (efficient_first vs capable_first).

This is **orthogonal to intent classification**. CogRouter answers "what kind
of task is this?"; it never asks "is this session going well?" A coding task
that's spinning needs a stronger model than the same task executing cleanly.
Our traffic through CogRouter is exactly this: OpenClaw agent turns full of
tool results, errors, edits. The signals are already in the message stream —
we're just not reading them.

**Concrete first step:** log tool-result/error/edit density per turn from
requests passing through the proxy; join against judge outcomes in
judge_history to see whether stage signals would have predicted the UCB
reward. Data first, then a stage axis in the scorer.

### 2. Context-window-aware routing
Switchyard: if the selected target exceeds its context window, remaining
targets are tried in configured order. CogRouter scores models without
checking whether the request fits — long agent sessions can silently route to
a model that will truncate or fail. Cheap fix: pre-filter candidates by
estimated token count vs declared context window, before scoring.

### 3. Escalation mode + quadrant calibration
llm_classifier mode=escalation: every turn runs weak-first, a judge reads the
answer and decides re-send to strong. More interesting is their **calibration
methodology** for routing thresholds (SWE-Bench Pro Python-75 → the 0.5):
- Pure-capable baseline (~40-75 tasks) + stratified efficient probe (~20)
- Quadrants: RESCUE (strong✓ weak✗), LOSS (strong✓ weak✗→ do-not-escalate),
  SAFE (both), HARD (neither)
- Pick threshold that rescues RESCUE without over-escalating LOSS
- Caveat: escalation inherits partial context → RESCUE is a conservative
  lower bound

Directly applicable to our **shadow-window acceptance gates** (hardening §4.5)
and to judging whether the UCB tiering should ever do weak-first escalation.

### 4. Observability: decision-source breakdown + routing headers
- Every decision carries a `decision_source` (override / signal / classifier /
  fallback / default) — published in logs AND metrics. CogRouter logs the pick
  but not a structured *why* taxonomy.
- Response header `x-model-router-selected-model` — client-visible provenance.
  OpenClaw dashboards could show which model actually served each turn.
- Prometheus `/metrics` text exposition + alerting rules + triage cheatsheet
  (symptom→cause table, incl. `model="<unknown>"` detection). We have /stats
  JSON; a scrapeable endpoint + decision-source counters is a small lift with
  real payoff for the shadow-window work.

### 5. Fixed-split random router
For honest A/B measurement. Shadow mode is paired-decision (no traffic split);
a configurable random split would let us measure real outcome differences
between two models on live traffic. Pairs with quadrant calibration (§3).

### 6. Protocol translation (OpenAI ↔ Anthropic ↔ Responses)
Bigger lift; matters only if we want non-OpenAI clients (e.g., Claude Code)
talking to CogRouter. Park unless a concrete client shows up. Their IR-first
design (decode → neutral types → route → re-encode) is the right shape if we
ever do it.

### 7. Library/server split
libsy never calls models — hands decisions back to the embedder. Elegant, and
would matter if Karl Code ever wanted router-in-process. Not now.

## Non-ideas
- Their fallback cascade ≈ our candidate retry loop, minus circuit breakers.
- Health: `{status:"ok"}` — less rich than our /health. (Our follow-up #2:
  judge fields on health — keep.)
- TOML config, NSSM etc. — no advantage over .env for us.

## Recommended sequence
1. **Stage-signal logging** (observability only, zero routing change) —
   feeds judge_history join → evidence for a stage axis.
2. **Context-window pre-filter** in candidate selection.
3. **Decision-source taxonomy + response header + Prometheus counters** —
   ride along with shadow-window work already queued.
4. **Quadrant calibration harness** against shadow-window data.
5. Escalation mode / random split — after 1-4 produce evidence.

*Authored by Karl, Aug 25 2026. Sources: repo README, stage_router + escalation
docs, zread architecture/session-affinity/metrics pages.*
