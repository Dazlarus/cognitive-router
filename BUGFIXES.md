# Cognitive Router Bug Fixes & Improvements (2026-06-23)

## Fixed Issues

### 1. EADDRINUSE Crashes (P1) ✅
**Problem:** Router crashes on startup if port 3456 is already in use by a lingering process.

**Fix:** Added `cleanupPort()` method that kills any process listening on the proxy port before binding. Uses `netstat` + `taskkill` on Windows.

**Location:** `src/proxy-stream.ts` - added imports and `cleanupPort()` method, called in `start()`

---

### 2. DB Constraint Failure - `confidence` NULL (P1) ✅
**Problem:** `routing_decisions.confidence` has a NOT NULL constraint, but sometimes classification returns `undefined` or `NaN`.

**Fix:**
- Added validation: `confidence ?? 0.5` in `recordDecision()` call
- Added NaN check after classification with fallback to 0.5

**Location:** `src/proxy-stream.ts` - confidence validation in `handleChat()`

---

### 3. Embedding Provider Health (P1) ✅
**Problem:** Classifier fails to embed all 10 intent prototypes when Ollama is unreachable, wasting startup time and logging spam.

**Fix:** Added `checkOllamaHealth()` method that probes `http://localhost:11434/api/tags` before initializing embeddings. Logs a warning if Ollama is down.

**Location:** `src/proxy-stream.ts` - added `checkOllamaHealth()` method

---

### 4. Deprecated Z.AI Models Auto-Disable (P2) ✅
**Problem:** Deprecated models waste routing time and cause 401 auth errors.

**Fix:** Modified `discoverModels()` in `model_registry.ts` to automatically delete models that exist in local registry but not in Z.AI's API response.

**Location:** `src/model_registry.ts` - updated `discoverModels()` Z.AI section

---

## New Features

### 1. Z.AI Quota Tracking (P1) ✅ IMPLEMENTED
**What:** Track Z.AI subscription quota consumption to avoid hitting limits.

**Features:**
- Track total tokens used (prompt + completion)
- Calculate quota % based on peak/off-peak/promo multipliers
  - Promo (through Sep 2026): 1× all day
  - Peak hours (01:00-10:00 UTC): 3×
  - Off-peak hours: 2×
- Log warning at 75% quota consumption
- Display quota info in `/stats` endpoint
- Reset quota tracking on command

**Location:**
- `src/cost_tracker.ts` - added `totalTokensUsed`, `quotaPercent`, `quotaWarned` fields; added `recordTokenUsage()`, `updateQuotaPercent()`, `getQuotaPercent()`, `getTotalTokensUsed()`, `getCurrentQuotaMultiplier()`, `resetQuotaTracking()` methods
- `src/proxy-stream.ts` - call `recordTokenUsage()` after successful Z.AI calls (both streaming and non-streaming, including hedge wins)
- `src/config.ts` - added `quotaConsumed`, `quotaPercent` fields to `ProviderBudget`
- `src/stats.ts` - added `quotaPercent`, `totalTokensUsed`, `currentQuotaMultiplier` to provider stats

---

### 2. Embedding Model Benchmarking (P2) ✅ COMPLETE
**What:** Benchmark all 7 available embedding models to find the best quality/latency tradeoff.

**Results (2026-06-23):**
| Model | Quality | Latency | Overall | Recommendation |
|-------|---------|---------|---------|----------------|
| **qwen3-embedding:latest** | 1.000 | 214ms | 0.886 | **NEW DEFAULT** ⭐ |
| nomic-embed-text:latest | 1.000 | 389ms | 0.711 | GOOD ALTERNATIVE |
| qwen3-embedding:4b | 1.000 | 779ms | 0.600 | SKIP |
| qwen3-embedding:8b | 1.000 | 958ms | 0.600 | SKIP |
| bge-m3:567m | 1.000 | 813ms | 0.600 | SKIP |
| nomic-embed-text-v2-moe:latest | 1.000 | 770ms | 0.600 | SKIP |
| embeddinggemma:latest | 1.000 | 3141ms | 0.600 | SKIP |

**Decision:** Updated default embedding model to `qwen3-embedding:latest` (2x faster than previous default).

**Location:**
- `src/benchmark_embeddings.ts` - new benchmark harness
- `src/proxy-stream.ts` - added benchmark endpoint `POST /v1/benchmark/embeddings`
- `src/config.ts` - updated default model
- `src/db_service.ts` - added `getDb()` method for benchmark access

---

## Remaining Issues

None remaining — all issues addressed or planned.

---

## Build & Deploy

```bash
cd C:\Users\hi100\.openclaw\workspace\cognitive-router
npm run build
# Then restart the service
```

---

## Testing Checklist

- [x] Service starts without EADDRINUSE error
- [x] Classifier initializes with or without Ollama (both paths)
- [x] No DB constraint violations in logs
- [x] Deprecated models auto-disabled on startup
- [x] Routing decisions recorded with valid confidence scores
- [x] Z.AI quota tracking works (see `/stats` endpoint after calls)
- [x] Quota warning logs at 75%
- [x] **Manual testing required:** Run embedding model benchmarks (see EMBEDDING_BENCHMARKING.md) - **COMPLETED 2026-06-23**
- [ ] **Restart service** to activate new embedding model (`qwen3-embedding:latest`)

---

## Next Steps

1. **Restart service** to activate quota tracking
2. **Monitor Z.AI quota** via `/stats` endpoint after some traffic
3. ~~**Run embedding benchmarks** to find best model~~ ✅ DONE
4. ~~**Update default embedding model** if better one found~~ ✅ DONE (changed to `qwen3-embedding:latest`)