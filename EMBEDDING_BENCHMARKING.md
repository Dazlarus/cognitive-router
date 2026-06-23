# Embedding Model Benchmarking Plan

## Context
- Multiple embedding models available in Ollama (8+ models)
- Currently using `nomic-embed-text` by default
- No performance/capability data for newer models (nomic-embed-text-v2-moe, qwen3-embedding variants)
- Embeddings are critical for intent classification quality

## Models to Benchmark

### Existing Models
| Model | Context | Notes |
|-------|---------|-------|
| `nomic-embed-text:latest` | 8K | Current default |
| `nomic-embed-text-v2-moe:latest` | 8K | Newer Mixture-of-Experts version |
| `embeddinggemma:latest` | 8K | Google Gemma-based |
| `bge-m3:567m` | 8K | BGE M3 (multilingual) |

### New Models (discovered 2026-06-23)
| Model | Context | Notes |
|-------|---------|-------|
| `qwen3-embedding:4b` | 8K | Qwen 3, 4B params |
| `qwen3-embedding:8b` | 8K | Qwen 3, 8B params |
| `qwen3-embedding:latest` | 8K | Latest Qwen 3 embedding |

## Benchmark Criteria

### 1. Semantic Similarity
- Test on intent prototype embeddings (10 categories)
- Measure inter-category vs intra-category separation
- Metric: Mean Average Precision (mAP) or silhouette score

### 2. Latency
- Measure embedding generation time
- Compare single embedding vs batch embedding performance
- Target: <100ms for single embedding

### 3. Quality
- Test on real routing intent classification
- Compare classifier accuracy vs current baseline
- Metric: Intent classification accuracy %

### 4. Resource Usage
- VRAM consumption (important for RTX 4070 Ti 12GB)
- Memory footprint
- GPU utilization during embedding

## Benchmark Script

```typescript
// Embedding benchmark runner
// - Load intent prototypes from classifier
// - Generate embeddings for each model
// - Compute similarity matrix
// - Rank models by quality/latency tradeoff
// - Output recommendation
```

## Expected Output

```
Embedding Model Benchmark Results
=================================

Rank | Model                | Quality | Latency (ms) | VRAM (GB) | Score | Recommendation
-----|----------------------|---------|--------------|-----------|-------|----------------
  1  | qwen3-embedding:8b   | 0.95    | 85           | 2.3       | 0.94  | ✅ NEW DEFAULT
  2  | nomic-embed-text-v2  | 0.92    | 75           | 1.8       | 0.92  | Good alternative
  3  | qwen3-embedding:4b   | 0.88    | 65           | 1.2       | 0.90  | Fast but lower quality
  4  | nomic-embed-text     | 0.85    | 70           | 1.5       | 0.87  | Current baseline
  5  | embeddinggemma       | 0.82    | 95           | 2.0       | 0.84  | Slower
  6  | bge-m3               | 0.80    | 80           | 1.4       | 0.82  | Multilingual use
```

## Implementation Steps

1. **Create benchmark harness** (`src/benchmark_embeddings.ts`)
   - Load all available embedding models from Ollama
   - Generate embeddings for 10 intent prototypes
   - Compute similarity scores
   - Measure latency

2. **Add benchmark endpoint** (`/v1/benchmark/embeddings`)
   - Trigger on-demand benchmarking
   - Return cached results or run fresh
   - Cache results for 24h

3. **Add automatic benchmark** (cron/job)
   - Run weekly
   - Log results to SQLite
   - Alert if new best model found

4. **Update config** if better model found
   - Change default embedding model
   - Update fallback ordering

## Next Steps

Ready to implement. Want me to:
1. Create the benchmark harness first
2. Add the API endpoint for on-demand testing
3. Run initial benchmark against all 8 models

Or do you want me to integrate quota tracking first, then move to embedding benchmarking?