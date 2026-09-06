# Cognitive Router — Development Roadmap

> Last updated: 2026-09-05 22:25 CDT
> Location: `C:\Users\hi100\.openclaw\workspace\cognitive-router\`

## Vision

The Cognitive Router makes model pluralism (Karlism Principle 11) work in practice. It should:
1. **Discover** — actively find new models from Ollama, OpenRouter, HuggingFace (future)
2. **Classify** — determine request intent (coding, conversation, research, math...)
3. **Score** — evaluate every candidate model across capability, reliability, cost, latency
4. **Route** — pick the best fit automatically
5. **Learn** — judge samples responses, adjusts capability scores, persists circuit breaker state
6. **Maintain itself** — prune dead models, pull new ones, benchmark without human intervention

### Key Design Principles
- **No single model dependency** — the router IS the model pluralism infrastructure
- **Passive → Active discovery** — not just reading what's local, but seeking what's available
- **Embeddings are not routable** — once you pick an embedding model, you're committed for the corpus lifetime. Endpoint should proxy and expose benchmark data, not route.
- **Cost intelligence** — predictive burn rate, auto-downgrade when budget low
- **Fast failure recovery** — aggressive deprioritization, not gentle EMA decay
- **Non-blocking benchmarks** — startup is fast, benchmarks run in background

---

## Phase 1 — Complete ✅ (2026-07-27)

### Completed Improvements

| # | Improvement | Files Modified |
|---|-------------|---------------|
| 1 | **Judge fallback chain** — When OpenRouter circuit is open, judge falls back to Ollama gemma4 to keep learning loop alive | `judge.ts` |
| 2 | **Routing decision outcomes** — Added `updateDecisionOutcome()` to db_service, wired into all 4 outcome recording points (2 success, 1 failure, 1 hedge). Decisions no longer stuck at PENDING. | `db_service.ts`, `proxy-stream.ts` |
| 3 | **Circuit breaker persistence** — Save/restore circuit state to SQLite via `provider_health` table metadata column. Survives restarts. Added `POST /reset-circuit?provider=X` endpoint. | `db_service.ts`, `cost_tracker.ts`, `proxy-stream.ts` |
| 4 | **Real token counting** — Replaced `chars/4` heuristic with js-tiktoken. Blended ratio for tool-heavy JSON payloads. Added 20% context safety margin (configurable via `ROUTER_CONTEXT_SAFETY_MARGIN`). | `proxy-stream.ts`, `package.json` |
| 5 | **Auto-benchmark Ollama models** — Discovered chat models get probed with a coding prompt on startup. Non-blocking (fire-and-forget), sequential execution with 60s shared budget, 10s per-model timeout. Source tracking: `inferred` → `auto-bench`. | `model_registry.ts` |
| 6 | **Decision transparency endpoint** — `GET /last-decision?limit=N` now returns winner scores + top 5 runner-up candidates with per-dimension scores and rationale strings. Added `candidates_json` column to routing_decisions. | `router.ts`, `proxy-stream.ts`, `db_service.ts` |

### Other Phase 1 Fixes (same session)
- **Dead OpenRouter models removed**: `qwen/qwen3-coder:free` (404), `openrouter/owl-alpha` (deleted), `qwen/qwen3.6-plus:free` (deprecated), `nvidia/nemotron-3-super-120b-a12b:free` (resource exhausted)
- **GLM-5-turbo and GLM-4.7 added to Z.AI generation pool** in `model_policy.ts`
- **Judge model**: `qwen/qwen3-30b-a3b-instruct-2507` with fallback to `ollama/gemma4:latest`
- **OpenRouter liveness cron**: every 6 hours (job `408e59c9-7222-406a-adef-e8049f5025ff`)
- **Liveness script**: `scripts/openrouter-liveness.js`

### Current Architecture Notes
- **Entry point**: `node dist/server.js` (NOT proxy-stream.ts directly)
- **Port**: 3456
- **DB**: `data/cognitive-router.db` — 30K+ routing_decisions rows
- **Provider priority**: zai → openrouter → gemini → ollama
- **Z.AI generation pool**: glm-5.2, glm-5.1, glm-5-turbo, glm-4.7, glm-4.7-flash
- **OpenRouter alive models**: poolside/laguna-m.1:free, nvidia/nemotron-3-ultra-550b-a55b:free, cohere/north-mini-code:free, qwen/qwen3-30b-a3b-instruct-2507, deepseek/deepseek-v4-flash
- **Judge sampling**: 10% of responses
- **Intent categories**: coding, research, creative, conversation, summary, retrieval, science, business, math, analysis
- **Embedding benchmarking**: exists in `benchmark_embeddings.ts` (separate from chat model benchmarking)

### Files Modified in Phase 1
- `src/model_policy.ts` — GLM models added to generation pool
- `src/model_registry.ts` — auto-bench, source tracking (benchmark/inferred/auto-bench/blended)
- `src/proxy-stream.ts` — tiktoken, context safety margin, circuit reset endpoint, candidates in decisions, decision transparency endpoint
- `src/cost_tracker.ts` — circuit persistence, reset methods
- `src/judge.ts` — provider fallback chain (OpenRouter → Ollama)
- `src/db_service.ts` — updateDecisionOutcome, circuit persistence, candidates_json column
- `src/stats.ts` — dead model references cleaned up
- `src/model_registry.ts` — dead models removed, new models added
- `scripts/openrouter-liveness.js` — new file

---

## Phase 2 — Approved (10 tasks, all in TASK_QUEUE.json as `approved`)

| Task ID | Title | Priority | Effort |
|---------|-------|----------|--------|
| cogrouter-p2-001 | Active model curator — Ollama + OpenRouter scanning | 8 | Large |
| cogrouter-p2-002 | Embeddings endpoint refactor — pure proxy + enhanced benchmarking | 8 | Large |
| cogrouter-p2-003 | Benchmark system overhaul — cache, multi-probe, persistence, relevance | 7 | Medium ✅ |
| cogrouter-p2-004 | Context window enforcement — filter and reject oversized requests | 9 | Small |
| cogrouter-p2-005 | Budget-aware routing — burn rate prediction and auto-downgrade | 7 | Medium |
| cogrouter-p2-006 | Routing dashboard — aggregated stats and trends | 6 | Medium |
| cogrouter-p2-007 | Negative signal amplification — aggressive deprioritization | 7 | Small |
| cogrouter-p2-008 | Multimodal routing — vision, audio, modality-aware scoring | 5 | Medium |
| cogrouter-p2-009 | Fallback intelligence — failure-type-aware retry strategies | 6 | Small |
| cogrouter-p2-010 | Warm model awareness — check Ollama /api/ps before routing | 6 | Small |

### Detailed Scope (from TASK_QUEUE.json)

**P2-001: Active model curator**
- Scan Ollama library for GGUF models, evaluate size/compatibility
- `ollama pull` for approved models with VRAM check
- Scan OpenRouter `/api/v1/models`, auto-prune 404/deleted models (3+ consecutive failures)
- Trust signals: download count, maintainer, safety filters
- Wire into cron (every 6-12h) + manual trigger `POST /v1/curate`

**P2-002: Embeddings refactor**
- Remove routing from `/v1/embeddings` — expose all models, caller picks
- Expand benchmarks: dimension count, clustering quality (k-means cohesion), batch speed (100+ embeddings), memory, domain-specific semantic coherence
- `GET /v1/embeddings/benchmarks` endpoint for comparison data
- Persist results to SQLite

**P2-003: Benchmark overhaul** ✅
- Persist results to SQLite (`chat_benchmark_results` table), skip re-probe if <7 days old and model version unchanged
- Model version detection (Ollama digest via `/api/show`, OpenRouter/ZAI model ID)
- 3 probe types: coding, reasoning, conversation — LLM-as-judge scored (not heuristic)
- Traffic weighting: probe scores weighted by real intent distribution from `routing_decisions`
- Persistent auto-bench results applied to model registry capability scores
- Files: `benchmark_chat.ts` (new), `db_service.ts` (schema + methods), `model_registry.ts` (wiring)

**P2-004: Context window enforcement**
- Filter candidates where estimatedTokens > model.contextWindow * 0.8
- Clear error when no model can handle request size
- Log filtered models, include context window in /last-decision

**P2-005: Budget-aware routing**
- Burn rate: rolling average spend/hour from provider_spend
- Project remaining budget vs remaining period
- Auto-downgrade to free tiers when budget low
- Per-model cost efficiency metric (score_delta / cost_ratio)
- Cost anomaly detection (sudden token spikes)
- `GET /v1/budget` endpoint

**P2-006: Routing dashboard**
- `GET /v1/dashboard` with time-range param
- Provider distribution (% per provider)
- Cost per intent (aggregate spend by classification)
- Latency trends (1h, 6h, 24h, 7d buckets per provider)
- Model market share and shifts
- Success/failure rates per provider and model

**P2-007: Negative signal amplification**
- Track consecutive failures per provider+model
- 3 consecutive same-type failures → reliability to 0.1
- Pattern detection: timeout + large context → context-limited flag
- Pattern detection: rate limit → throttled flag (skip N minutes)
- Pattern detection: 500/crash → unstable flag (skip entirely)
- Patterns clear on success with flapping cooldown

**P2-008: Multimodal routing**
- Detect image_url, image_file, audio in request messages
- Add modality capability to ModelCapability (text, vision, audio flags)
- Filter candidates by required modality before scoring
- Clear error when no multimodal candidates available

**P2-009: Fallback intelligence**
- Classify failures: timeout, rate_limit, server_error, context_overflow, auth_error
- Timeout → switch provider (not model)
- Rate limit → switch provider + exponential backoff
- Server error → switch model (same provider ok)
- Context overflow → skip to candidate with larger window
- Auth error → skip provider entirely, alert

**P2-010: Warm model awareness**
- Poll Ollama `/api/ps` for resident model(s), cache 5s TTL
- Warm model: remove GPU penalty, boost latency score
- Cold model: add cold-start latency estimate (~5-10s for 7B)
- Log warm/cold status in decisions

---

## Phase 3 — Provider Coverage & Universal Discovery (Product Foundation)

> Added 2026-09-05 — this is the ticket to start building the product. The router sells model pluralism; it can only do that if it spans the actual model market. Current state: 5 providers (zai, openrouter, gemini, requesty, ollama), mostly hardcoded model lists.

### P3-001: Direct adapters for all major providers
- **Tier 1:** OpenAI, Anthropic, xAI, Mistral, DeepSeek (direct)
- **Tier 2:** Cohere, Together, Groq, Fireworks, Cerebras, Perplexity
- **Tier 3 (gated):** AWS Bedrock, Azure OpenAI — enterprise auth complexity, defer until paying users need them
- Each adapter implements the `buildZaiThinkingFields` pattern: per-model reasoning-effort dialect translated from provider docs (OpenAI `reasoning_effort`, Anthropic `thinking.budget_tokens`, etc.), streaming + buffered paths, error classification into the existing failure taxonomy, circuit breaker, per-provider budget/credential config

### P3-002: Universal model discovery — kill hardcoded lists
- Poll every provider's list-models endpoint on a cadence; ingest into model_registry with capability metadata (context window, modalities, tool support, reasoning support, pricing)
- Auto-prune dead/deprecated models (extend the openrouter-liveness pattern to all providers)
- Name-regex capability heuristics die completely — discovery metadata + measurement take over cold-start

### P3-003: Pricing intelligence, generalized
- Generalize the zai docs-based price-table approach (quota-burn weights, off-peak factors) into a per-provider pricing module with review dates
- OpenRouter `/api/v1/models` carries pricing natively — ingest it
- Wire uniformly into cost scoring and quota-burn weights

### P3-004: Scoring at market scale
- Candidate pool grows from ~dozens to 1000+ models — prefilter by capability tier/modality/context before full scoring
- Judge + EWMA latency + reliability learning loop stays provider-agnostic and measurement-based

### P3-005: Product surface
- Adding a provider = drop in a key, everything else automatic
- `/stats` + dashboard show the full market view; OpenClaw plugin config maps onto the same provider set

---

## Future Phase (Deferred)

### HuggingFace Integration
- Scan HuggingFace for new models compatible with Ollama (GGUF format)
- Evaluate by capability gap (does this fill a missing niche?)
- **Deferred** due to supply chain trust concerns — arbitrary model downloads from HuggingFace need vetting (download count, known maintainer, safety filters, license). Trust framework needed first.

### Potential Future Work
- User satisfaction signals (retry rate, context length abandoned as implicit feedback)
- Trend analysis (is a model degrading over time?)
- Embedding model comparison UI/tooling
- Multi-region provider failover
- Custom intent training (user-defined intent categories from their workload)
